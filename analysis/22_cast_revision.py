"""Additional analyses of the external test on the in-house cast steels.
 (a) errors of the six literature-trained models by steel and by test temperature (MAE and mean signed error);
 (b) baselines that use ONLY the cast data under the leave-one-cast-alloy-out protocol of 08_finetune.py (no literature pre-training):
     the mean of the three training steels at each test temperature, a pooled straight line, and a pooled tanh curve (Equation (1));
 (c) sensitivity of the literature-only predictions to plausible values of the elements that were not analysed in the cast steels
     (entered as zero in the main analysis);
 (d) sensitivity of the adaptation to the cast-data weight (lambda = 1, 5, 20; 5 is the value used in the paper).
usage: python 22_cast_revision.py      (run with XGBoost 3.2.0)"""
import sys, copy, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xgboost as xgb, torch
from scipy.optimize import curve_fit
import common, models
from common import load, cast_alloys, CAST_MEAS, RES, metrics, tanh_curve

af = common.COMP + common.PROC + common.DERIVED; models.ALLOY_FEATS = af; models.FEATS = af + ['test_temp']
lit = load(); cast = cast_alloys(); cd = CAST_MEAS.merge(cast, on='Alloy')
cd['cond'] = 1000 + pd.factorize(cd.Alloy)[0]; cd['alloy_id'] = cd['cond']
TEMPS = [25, 0, -20, -40]

print('training the six models on the literature database', flush=True)
fitted = {name: models.MODELS[name]().fit(lit) for name in models.ORDER}

# ------------------------------------------------------------------ (a) errors by steel and by temperature
rows = []
for name, m in fitted.items():
    p = m.predict(cd); e = p - cd.Impact.values
    rows.append(dict(model=name, group='all', key='all', n=len(e), MAE=np.abs(e).mean(), bias=e.mean()))
    for al in cast.Alloy:
        k = (cd.Alloy == al).values; rows.append(dict(model=name, group='steel', key=al, n=int(k.sum()), MAE=np.abs(e[k]).mean(), bias=e[k].mean()))
    for T in TEMPS:
        k = (cd.test_temp == T).values; rows.append(dict(model=name, group='temperature', key=str(T), n=int(k.sum()), MAE=np.abs(e[k]).mean(), bias=e[k].mean()))
ra = pd.DataFrame(rows); ra.to_csv(RES / 'cast_errors_by_steel_temperature.csv', index=False)
print('\n(a) MAE (J)\n', ra.pivot(index='model', columns='key', values='MAE').loc[models.ORDER, ['all'] + list(cast.Alloy) + [str(T) for T in TEMPS]].round(1).to_string())
print('\n(a) mean signed error, predicted - measured (J)\n', ra.pivot(index='model', columns='key', values='bias').loc[models.ORDER, ['all'] + list(cast.Alloy) + [str(T) for T in TEMPS]].round(1).to_string())

# ------------------------------------------------------------------ (b) cast-only baselines, leave-one-cast-alloy-out
rows = []
for al in cast.Alloy:
    dtr, dte = cd[cd.Alloy != al], cd[cd.Alloy == al]; T, E = dtr.test_temp.values.astype(float), dtr.Impact.values; yt = dte.Impact.values
    pm = dte.test_temp.map(dtr.groupby('test_temp').Impact.mean()).values                      # mean of the three steels at each temperature
    b, a = np.polyfit(T, E, 1); pl = a + b * dte.test_temp.values                             # pooled straight line
    popt, _ = curve_fit(tanh_curve, T, E, p0=[min(15, 0.5 * E.min()), 220, -15, 20], bounds=([0, 0, -300, 3], [E.min(), 1.5 * E.max(), 400, 300]), maxfev=20000)
    pt = tanh_curve(dte.test_temp.values.astype(float), *popt)                                 # pooled tanh curve, same bounds as Section 2.2
    for lab, p in [('mean of the other three steels at each temperature', pm), ('pooled straight line', pl), ('pooled tanh curve', pt)]:
        rows.append(dict(baseline=lab, alloy=al, **metrics(yt, p)))
    print(f'  held-out {al:6s} pooled tanh: LSE={popt[0]:.0f} USE={popt[1]:.0f} Tt={popt[2]:.1f} Cw={popt[3]:.1f}', flush=True)
rb = pd.DataFrame(rows); rb.to_csv(RES / 'cast_only_baselines_LOCO.csv', index=False)
print('\n(b) cast-only baselines (LOCO), MAE by held-out steel\n', rb.pivot(index='baseline', columns='alloy', values='MAE').round(1).to_string())
print(rb.groupby('baseline')[['R2', 'MAE', 'RMSE']].mean().round(2).to_string())

# ------------------------------------------------------------------ (c) undetermined elements
nz = {e: lit.loc[lit[e] > 0, e] for e in ['P', 'S', 'N', 'Al']}
SCN = {'as analysed (undetermined elements = 0)': {},
       'P, S = database medians': dict(P=nz['P'].median(), S=nz['S'].median()),
       'P, S, N = database medians': dict(P=nz['P'].median(), S=nz['S'].median(), N=nz['N'].median()),
       'P, S, N, Al = database medians': dict(P=nz['P'].median(), S=nz['S'].median(), N=nz['N'].median(), Al=nz['Al'].median()),
       'P, S, N = upper quartiles': dict(P=nz['P'].quantile(.75), S=nz['S'].quantile(.75), N=nz['N'].quantile(.75)),
       'P, S, N, Al = upper quartiles': dict(P=nz['P'].quantile(.75), S=nz['S'].quantile(.75), N=nz['N'].quantile(.75), Al=nz['Al'].quantile(.75)),
       'P = 0.020, S = 0.020, N = 0.010, Al = 0': dict(P=0.020, S=0.020, N=0.010)}
rows = []
for lab, v in SCN.items():
    c2 = cast.copy()
    for k, x in v.items(): c2[k] = x
    c2 = common.add_derived(c2); d2 = CAST_MEAS.merge(c2, on='Alloy')
    for name, m in fitted.items():
        p = m.predict(d2); r = metrics(d2.Impact.values, p)
        rec = dict(scenario=lab, model=name, MAE=r['MAE'], R2=r['R2'], pred25=m.predict(c2.assign(test_temp=25.)).mean(), **{k: round(float(x), 4) for k, x in v.items()})
        if hasattr(m, 'curve_params'):
            P = m.curve_params(c2.assign(test_temp=25.)); rec.update(Tt=P.T0.mean(), USE=P.USE.mean())
        rows.append(rec)
rc = pd.DataFrame(rows); rc.to_csv(RES / 'cast_missing_chemistry_sensitivity.csv', index=False)
print('\n(c) values used:', {k: {e: round(float(x), 4) for e, x in v.items()} for k, v in SCN.items()})
print('(c) cast MAE (J) by scenario\n', rc.pivot(index='scenario', columns='model', values='MAE').loc[list(SCN), models.ORDER].round(1).to_string())
print('(c) mean predicted Tt (C) / USE (J)\n', rc.pivot(index='scenario', columns='model', values='Tt').loc[list(SCN)].dropna(axis=1).round(1).to_string(),
      '\n', rc.pivot(index='scenario', columns='model', values='USE').loc[list(SCN)].dropna(axis=1).round(0).to_string())

# ------------------------------------------------------------------ (d) adaptation: sensitivity to the cast-data weight
def finetune(name, base, dtrain, lam):
    """same procedure as 08_finetune.py with the cast-data weight `lam` (5 in the paper)"""
    m = copy.deepcopy(base)
    if name == 'tanh-XGB':
        X = np.vstack([lit[models.ALLOY_FEATS].values, dtrain[models.ALLOY_FEATS].values])
        y = np.concatenate([lit.Impact.values, dtrain.Impact.values]); m.T_tr = np.concatenate([lit.test_temp.values, dtrain.test_temp.values])
        w = np.concatenate([np.ones(len(lit)), lam * np.ones(len(dtrain))]); obj0 = models.TanhXGB._obj
        def obj(R, dm):
            g, h = obj0(m, R, dm); return g * w[:, None], h * w[:, None]
        dm = xgb.DMatrix(X, label=y, base_margin=np.tile(m.base, (len(y), 1)))
        m.bst = xgb.train(dict(tree_method='hist', multi_strategy='multi_output_tree', num_target=4, max_depth=3, eta=0.02, subsample=0.8, colsample_bynode=0.8,
                               min_child_weight=3, reg_lambda=5.0, nthread=12, seed=0), dm, num_boost_round=300, obj=obj, xgb_model=base.bst)
    elif name in ('XGB-direct', 'XGB-direct-mono'):
        X = np.vstack([lit[models.FEATS].values, dtrain[models.FEATS].values]); y = np.concatenate([lit.Impact.values, dtrain.Impact.values])
        w = np.concatenate([np.ones(len(lit)), lam * np.ones(len(dtrain))])
        extra = {'monotone_constraints': base.p['monotone_constraints']} if name == 'XGB-direct-mono' else {}
        m.m = xgb.XGBRegressor(**{**models.XGB_P, 'n_estimators': 300, 'learning_rate': 0.02, 'max_depth': 3, 'reg_lambda': 5.0, **extra})
        m.m.fit(X, y, sample_weight=w, xgb_model=base.m.get_booster())
    else:
        Xl = torch.tensor(m.sc.transform(lit)); Xc = torch.tensor(m.sc.transform(dtrain))
        Tl = torch.tensor(lit.test_temp.values, dtype=torch.float32); Tc = torch.tensor(dtrain.test_temp.values, dtype=torch.float32)
        yl = torch.tensor(lit.Impact.values, dtype=torch.float32); yc = torch.tensor(dtrain.Impact.values, dtype=torch.float32)
        for net in m.nets:
            opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-3); net.train()
            for ep in range(300):
                opt.zero_grad()
                if name == 'tanh-NN': loss = ((net(Xl, Tl) - yl) ** 2).mean() + lam * ((net(Xc, Tc) - yc) ** 2).mean()
                else:
                    f = lambda X: torch.sigmoid(net(X).squeeze(1) - 1.5) * 400.
                    loss = ((f(Xl) - yl) ** 2).mean() + lam * ((f(Xc) - yc) ** 2).mean()
                (loss / 6400.).backward(); opt.step()
            net.eval()
    return m

rows = []
for name in ['XGB-direct', 'XGB-direct-mono', 'tanh-XGB', 'NN-direct', 'tanh-NN']:
    for lam in [1.0, 5.0, 20.0]:
        for al in cast.Alloy:
            dtr, dte = cd[cd.Alloy != al], cd[cd.Alloy == al]
            rows.append(dict(model=name, lam=lam, alloy=al, **metrics(dte.Impact.values, finetune(name, fitted[name], dtr, lam).predict(dte))))
        print(f'  {name:16s} lambda={lam:4.0f} LOCO MAE={np.mean([r["MAE"] for r in rows[-4:]]):.1f}', flush=True)
rd = pd.DataFrame(rows); rd.to_csv(RES / 'finetune_lambda_sensitivity.csv', index=False)
print('\n(d) LOCO fine-tuning MAE (J) by cast-data weight\n', rd.groupby(['model', 'lam']).MAE.mean().unstack().round(1).to_string())
print('done')
