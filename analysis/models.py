"""All candidate models with a common fit(train_df) -> predict(df_rows) interface.
Direct models      : Impact = f(alloy feats, test_temp)
Equation-based     : (LSE, USE, T0, Cw) = g(alloy feats);  Impact = tanh_curve(test_temp; params)"""
import numpy as np, pandas as pd, xgboost as xgb, torch, torch.nn as nn
from scipy.optimize import curve_fit
from common import ALLOY_FEATS, FEATS, tanh_curve

torch.set_num_threads(4)
LOG_COLS = ['Q_time', 'T_time', 'Q_type', 'T_type']
XGB_P = dict(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=3, reg_lambda=1.0, n_jobs=12, random_state=0)
TANH_XGB_P = dict(rounds=1500, eta=0.02, depth=3)
TWO_STAGE_P = dict(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                   min_child_weight=3, reg_lambda=1.0, n_jobs=12, random_state=0)   # fixed (only 59 curves; not tuned)
import json, pathlib
_R = pathlib.Path(__file__).resolve().parents[1] / 'results'
XGB_MONO_P = dict(XGB_P)                                                   # XGB-direct with monotone constraint on test_temp
NN_P = {'NN-direct': dict(h=64, drop=0.05, lr=3e-3, wd=1e-3, epochs=1500),  # defaults (untuned); overwritten by tuned_*.json
        'tanh-NN': dict(h=64, drop=0.05, lr=3e-3, wd=1e-3, epochs=1500)}
if (_R / 'tuned_XGB-direct.json').exists(): XGB_P.update(json.load(open(_R / 'tuned_XGB-direct.json'))['params'])
if (_R / 'tuned_XGB-direct-mono.json').exists(): XGB_MONO_P.update(json.load(open(_R / 'tuned_XGB-direct-mono.json'))['params'])
if (_R / 'tuned_tanh-XGB.json').exists(): TANH_XGB_P.update(json.load(open(_R / 'tuned_tanh-XGB.json'))['params'])
for _n in NN_P:
    if (_R / f'tuned_{_n}.json').exists(): NN_P[_n].update(json.load(open(_R / f'tuned_{_n}.json'))['params'])


# ----------------------------------------------------------------------------- feature scaling
class Scaler:
    def fit(self, df, cols):
        A = df[cols].copy()
        for c in LOG_COLS:
            if c in cols: A[c] = np.log1p(A[c])
        self.cols, self.mu, self.sd = cols, A.mean().values, A.std().values + 1e-6
        return self

    def transform(self, df):
        A = df[self.cols].copy()
        for c in LOG_COLS:
            if c in self.cols: A[c] = np.log1p(A[c])
        return np.clip((A.values - self.mu) / self.sd, -3, 3).astype(np.float32)


# ----------------------------------------------------------------------------- A. XGB direct
class XGBDirect:
    name = 'XGB-direct'

    def __init__(self, log=False, **kw):
        self.log = log; self.p = {**XGB_P, **kw}

    def fit(self, df):
        self.m = xgb.XGBRegressor(**self.p); y = df.Impact.values
        self.m.fit(df[FEATS].values, np.log1p(y) if self.log else y)
        return self

    def predict(self, df):
        p = self.m.predict(df[FEATS].values)
        return np.expm1(p) if self.log else p


class XGBDirectMono(XGBDirect):
    """XGB-direct with a monotonically increasing constraint on the test temperature (xgboost monotone_constraints)."""
    name = 'XGB-direct-mono'

    def __init__(self, **kw):
        super().__init__(**{**XGB_MONO_P, **kw})

    def fit(self, df):
        self.p['monotone_constraints'] = '(' + ','.join('1' if f == 'test_temp' else '0' for f in FEATS) + ')'
        return super().fit(df)


# ----------------------------------------------------------------------------- MLP building block
def mlp(nin, nout, h=64, drop=0.05):
    return nn.Sequential(nn.Linear(nin, h), nn.SiLU(), nn.Dropout(drop),
                         nn.Linear(h, h), nn.SiLU(), nn.Dropout(drop), nn.Linear(h, nout))


def _train(net, loss_fn, epochs=1500, lr=3e-3, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    net.train()
    for ep in range(epochs):
        opt.zero_grad(); loss = loss_fn(net); loss.backward(); opt.step(); sched.step()
    net.eval()
    return net


# ----------------------------------------------------------------------------- A2. NN direct (control)
class NNDirect:
    name = 'NN-direct'

    def __init__(self, n_ens=5, **kw):
        self.n_ens = n_ens; self.hp = {**NN_P['NN-direct'], **kw}

    def fit(self, df):
        self.sc = Scaler().fit(df, FEATS)
        X = torch.tensor(self.sc.transform(df)); y = torch.tensor(df.Impact.values, dtype=torch.float32)
        yt = y / 400.; hp = self.hp
        self.nets = []
        for s in range(self.n_ens):
            torch.manual_seed(s); net = mlp(X.shape[1], 1, h=hp['h'], drop=hp['drop'])
            self.nets.append(_train(net, lambda n: ((torch.sigmoid(n(X).squeeze(1) - 1.5) - yt) ** 2).mean() * 25, hp['epochs'], hp['lr'], hp['wd'], seed=s))
        return self

    def predict_members(self, df):
        """predictions of the individual ensemble members (n_ens x n)"""
        X = torch.tensor(self.sc.transform(df))
        with torch.no_grad():
            return np.array([torch.sigmoid(n(X).squeeze(1) - 1.5).numpy() for n in self.nets]) * 400.

    def predict(self, df):
        return self.predict_members(df).mean(0)


# ----------------------------------------------------------------------------- tanh head
class TanhHead(nn.Module):
    """alloy features -> (LSE, USE, T0, Cw) with physical constraints 0<=LSE<USE, Cw>0"""

    def __init__(self, nin, h=64, drop=0.05):
        super().__init__(); self.body = mlp(nin, 4, h=h, drop=drop)

    def params(self, X):
        o = self.body(X); sg = torch.sigmoid
        LSE = 60 * sg(o[:, 0] - 2.0)                    # 0..60 J   (~7 J at init)
        USE = LSE + 400 * sg(o[:, 1] - 1.0)             # LSE..LSE+400 J (~+110 J at init)
        T0 = 500 * sg(o[:, 2] - 0.2) - 250              # -250..250 C (~-25 C at init)
        Cw = 5 + 200 * sg(o[:, 3] - 1.5)                # 5..205 C  (~40 C at init)
        return LSE, USE, T0, Cw

    def forward(self, X, T):
        LSE, USE, T0, Cw = self.params(X)
        return 0.5 * (USE + LSE) + 0.5 * (USE - LSE) * torch.tanh((T - T0) / Cw)


class TanhNN:
    name = 'tanh-NN'

    def __init__(self, n_ens=5, **kw):
        self.n_ens = n_ens; self.hp = {**NN_P['tanh-NN'], **kw}

    def fit(self, df):
        self.sc = Scaler().fit(df, ALLOY_FEATS)
        X = torch.tensor(self.sc.transform(df))
        T = torch.tensor(df.test_temp.values, dtype=torch.float32)
        y = torch.tensor(df.Impact.values, dtype=torch.float32); hp = self.hp
        self.nets = []
        for s in range(self.n_ens):
            torch.manual_seed(s); net = TanhHead(X.shape[1], h=hp['h'], drop=hp['drop'])
            self.nets.append(_train(net, lambda n: ((n(X, T) - y) ** 2).mean() / 6400., hp['epochs'], hp['lr'], hp['wd'], seed=s))
        return self

    def predict_members(self, df):
        X = torch.tensor(self.sc.transform(df)); T = torch.tensor(df.test_temp.values, dtype=torch.float32)
        with torch.no_grad():
            return np.array([n(X, T).numpy() for n in self.nets])

    def curve_params(self, df):
        X = torch.tensor(self.sc.transform(df))
        with torch.no_grad():
            P = np.array([[t.numpy() for t in n.params(X)] for n in self.nets])  # ens x 4 x n
        return pd.DataFrame(P.mean(0).T, columns=['LSE', 'USE', 'T0', 'Cw'], index=df.index)

    def predict(self, df):
        return self.predict_members(df).mean(0)


# ----------------------------------------------------------------------------- B. two-stage (curve fit -> parameter regression)
def fit_curve(T, E):
    lo, hi = E.min(), E.max()
    p0 = [max(lo, 0.5), hi, T[np.argmin(np.abs(E - (lo + hi) / 2))], 30]
    popt, _ = curve_fit(tanh_curve, T, E, p0=p0, bounds=([0, 0, -300, 3], [hi, 1.5 * hi + 5, 400, 300]), maxfev=20000)
    return popt


class TwoStage:
    name = 'two-stage'

    def __init__(self, min_pts=4):
        self.min_pts = min_pts

    def fit(self, df):
        rows = []
        for cid, g in df.groupby('cond'):
            if g.test_temp.nunique() < self.min_pts: continue
            try:
                p = fit_curve(g.test_temp.values, g.Impact.values)
            except Exception:
                continue
            rows.append([*g[ALLOY_FEATS].iloc[0].values, *p])
        A = np.array(rows); self.n_curves = len(A)
        X, P = A[:, :len(ALLOY_FEATS)], A[:, len(ALLOY_FEATS):]
        self.ms = []
        for k in range(4):
            m = xgb.XGBRegressor(**TWO_STAGE_P); m.fit(X, P[:, k]); self.ms.append(m)
        return self

    def curve_params(self, df):
        X = df[ALLOY_FEATS].values
        P = np.column_stack([m.predict(X) for m in self.ms])
        P[:, 0] = np.clip(P[:, 0], 0, None); P[:, 1] = np.maximum(P[:, 1], P[:, 0] + 1); P[:, 3] = np.clip(P[:, 3], 3, None)
        return pd.DataFrame(P, columns=['LSE', 'USE', 'T0', 'Cw'], index=df.index)

    def predict(self, df):
        P = self.curve_params(df)
        return tanh_curve(df.test_temp.values, P.LSE.values, P.USE.values, P.T0.values, P.Cw.values)


# ----------------------------------------------------------------------------- D. tanh-XGB (custom multi-output objective)
def _sp(x): return np.logaddexp(0, x)
def _sig(x): return 1 / (1 + np.exp(-x))
def _isp(y): return np.log(np.expm1(y))


class TanhXGB:
    """XGBoost with vector leaves predicting raw (r1..r4) -> LSE=10sp(r1), USE=LSE+100sp(r2), T0=50r3, Cw=5+40sp(r4);
    loss = (E(T)-y)^2, Gauss-Newton hessian."""
    name = 'tanh-XGB'

    def __init__(self, **kw):
        d = dict(rounds=1500, eta=0.02, depth=3, min_child_weight=3, subsample=0.8, colsample=0.8, reg_lambda=1.0)
        d.update(TANH_XGB_P); d.update({k: v for k, v in kw.items() if v is not None})
        self.rounds, self.eta, self.depth = d['rounds'], d['eta'], d['depth']
        self.mcw, self.subsample, self.colsample, self.reg_lambda = d['min_child_weight'], d['subsample'], d['colsample'], d['reg_lambda']

    @staticmethod
    def _unpack(R):
        LSE = 10 * _sp(R[:, 0]); USE = LSE + 100 * _sp(R[:, 1]); T0 = 50 * R[:, 2]; Cw = 5 + 40 * _sp(R[:, 3])
        return LSE, USE, T0, Cw

    def _obj(self, R, dm):
        y = dm.get_label(); T = self.T_tr
        LSE, USE, T0, Cw = self._unpack(R)
        z = (T - T0) / Cw; th = np.tanh(z); sech2 = 1 - th ** 2; B = 0.5 * (USE - LSE)
        E = 0.5 * (USE + LSE) + B * th; res = E - y
        dLSE = 0.5 * (1 - th); dUSE = 0.5 * (1 + th); dT0 = -B * sech2 / Cw; dCw = -B * sech2 * (T - T0) / Cw ** 2
        J = np.column_stack([(dLSE + dUSE) * 10 * _sig(R[:, 0]), dUSE * 100 * _sig(R[:, 1]), dT0 * 50, dCw * 40 * _sig(R[:, 3])])
        grad = 2 * res[:, None] * J; hess = 2 * J ** 2 + 1e-3
        return grad, hess

    def fit(self, df):
        X = df[ALLOY_FEATS].values; y = df.Impact.values; self.T_tr = df.test_temp.values
        base = np.tile([_isp(0.5), _isp(1.0), -0.4, _isp(1.0)], (len(y), 1))
        dm = xgb.DMatrix(X, label=y, base_margin=base); self.base = base[0]
        self.bst = xgb.train(dict(tree_method='hist', multi_strategy='multi_output_tree', num_target=4, max_depth=self.depth, eta=self.eta,
                                  subsample=self.subsample, colsample_bynode=self.colsample, min_child_weight=self.mcw, reg_lambda=self.reg_lambda, nthread=12, seed=0),
                             dm, num_boost_round=self.rounds, obj=self._obj)
        return self

    def curve_params(self, df):
        dm = xgb.DMatrix(df[ALLOY_FEATS].values, base_margin=np.tile(self.base, (len(df), 1)))
        R = self.bst.predict(dm, output_margin=True).reshape(len(df), 4)
        return pd.DataFrame(np.column_stack(self._unpack(R)), columns=['LSE', 'USE', 'T0', 'Cw'], index=df.index)

    def predict(self, df):
        P = self.curve_params(df)
        return tanh_curve(df.test_temp.values, P.LSE.values, P.USE.values, P.T0.values, P.Cw.values)


MODELS = {'XGB-direct': lambda: XGBDirect(), 'XGB-direct-mono': lambda: XGBDirectMono(), 'NN-direct': lambda: NNDirect(),
          'two-stage': lambda: TwoStage(), 'tanh-NN': lambda: TanhNN(), 'tanh-XGB': lambda: TanhXGB()}
ORDER = list(MODELS)
COL = {'XGB-direct': 'C0', 'XGB-direct-mono': 'C7', 'NN-direct': 'C9', 'two-stage': 'C2', 'tanh-NN': 'C1', 'tanh-XGB': 'C3'}
