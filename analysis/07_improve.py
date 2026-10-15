"""Accuracy-improvement experiments for the tree models (fast): derived features, monotone constraint,
per-condition sample weights, log target.  Evaluated with GroupKFold(alloy) and GroupKFold(cond), 2 repeats."""
import sys, warnings, time; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupKFold, KFold
import common, models
from common import load, metrics, ALLOY_FEATS, FEATS, DERIVED, RES

df = load()

def cv(make, scen, reps=2):
    out = []
    for r in range(reps):
        sp = KFold(5, shuffle=True, random_state=r).split(df) if scen == 'random' else \
             GroupKFold(5, shuffle=True, random_state=r).split(df, groups=df[scen].values)
        oof = np.zeros(len(df))
        for tr, te in sp:
            m = make().fit(df.iloc[tr]); oof[te] = m.predict(df.iloc[te])
        out.append(metrics(df.Impact.values, oof))
    return {k: np.mean([o[k] for o in out]) for k in ['R2', 'MAE']}

def run(label, make):
    t = time.time(); r = {s: cv(make, s) for s in ['random', 'cond', 'alloy_id']}
    print(f'{label:45s} random R2={r["random"]["R2"]:.3f} MAE={r["random"]["MAE"]:.1f} | cond R2={r["cond"]["R2"]:.3f} MAE={r["cond"]["MAE"]:.1f} | alloy R2={r["alloy_id"]["R2"]:.3f} MAE={r["alloy_id"]["MAE"]:.1f}   ({time.time()-t:.0f}s)', flush=True)
    return dict(label=label, **{f'{s}_{k}': v for s, d in r.items() for k, v in d.items()})

# feature-set switch: models read common.ALLOY_FEATS/FEATS through module globals -> patch them
def set_feats(derived):
    af = common.COMP + common.PROC + (DERIVED if derived else [])
    models.ALLOY_FEATS = af; models.FEATS = af + ['test_temp']

class XGBDirectW(models.XGBDirect):
    """XGB-direct with optional monotone constraint on test_temp, per-condition sample weights, log target"""
    def __init__(self, mono=False, weight=False, log=False, **kw):
        super().__init__(log=log, **kw); self.mono, self.weight = mono, weight
    def fit(self, df):
        F = models.FEATS
        if self.mono: self.p['monotone_constraints'] = '(' + ','.join('1' if f == 'test_temp' else '0' for f in F) + ')'
        self.m = xgb.XGBRegressor(**self.p); y = df.Impact.values
        w = None
        if self.weight:
            n = df.groupby('cond').cond.transform('size').values; w = 1 / np.sqrt(n)
        self.m.fit(df[F].values, np.log1p(y) if self.log else y, sample_weight=w); return self
    def predict(self, df):
        p = self.m.predict(df[models.FEATS].values); return np.expm1(p) if self.log else p

class TanhXGBW(models.TanhXGB):
    def __init__(self, weight=False, **kw): super().__init__(**kw); self.weight = weight
    def fit(self, df):
        if self.weight:
            n = df.groupby('cond').cond.transform('size').values; self.w = 1 / np.sqrt(n)
        else: self.w = None
        return super().fit(df)
    def _obj(self, R, dm):
        g, h = super()._obj(R, dm)
        if self.w is not None: g = g * self.w[:, None]; h = h * self.w[:, None]
        return g, h

rows = []
for derived in [False, True]:
    set_feats(derived); tag = '+derived' if derived else 'base'
    rows.append(run(f'XGB-direct {tag}', lambda: XGBDirectW()))
    rows.append(run(f'XGB-direct {tag} +mono', lambda: XGBDirectW(mono=True)))
    rows.append(run(f'XGB-direct {tag} +mono +weight', lambda: XGBDirectW(mono=True, weight=True)))
    rows.append(run(f'tanh-XGB {tag}', lambda: TanhXGBW()))
    rows.append(run(f'tanh-XGB {tag} +weight', lambda: TanhXGBW(weight=True)))
    rows.append(run(f'tanh-XGB {tag} depth3 r1500 eta0.02', lambda: TanhXGBW(rounds=1500, eta=0.02, depth=3)))
pd.DataFrame(rows).to_csv(RES / 'improve_tree_experiments.csv', index=False)
