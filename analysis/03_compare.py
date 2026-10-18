"""Compare all models under 3 split scenarios: random KFold, GroupKFold by heat-treatment condition, GroupKFold by alloy.
usage: python 03_compare.py [model names...]"""
import sys, time, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from common import load, RES, metrics, tanh_curve
from models import MODELS, fit_curve

df = load(); y = df.Impact.values
which = sys.argv[1:] or list(MODELS)
tag = '_'.join(which) if len(which) < 5 else 'all'
SCEN = {'random': None, 'unseen-condition': 'cond', 'unseen-alloy': 'alloy_id'}
N_REP = 3


def splits(scen, rep):
    if SCEN[scen] is None:
        return list(KFold(5, shuffle=True, random_state=rep).split(df))
    return list(GroupKFold(5, shuffle=True, random_state=rep).split(df, groups=df[SCEN[scen]].values))


Tgrid = np.linspace(-200, 300, 101)


def smoothness(model, te):
    """held-out conditions: predict on a T grid -> non-monotonicity index (NMI) and tanh-describability (R2 of tanh fit)"""
    out = []
    for cid, g in df.iloc[te].groupby('cond'):
        row = g.iloc[[0] * len(Tgrid)].copy(); row['test_temp'] = Tgrid
        p = model.predict(row); d = np.diff(p)
        nmi = -d[d < 0].sum() / max(p.max() - p.min(), 1e-6)
        try:
            pp = fit_curve(Tgrid, p)
            r2 = 1 - ((p - tanh_curve(Tgrid, *pp)) ** 2).sum() / max(((p - p.mean()) ** 2).sum(), 1e-6)
        except Exception:
            r2 = np.nan
        out.append((nmi, r2))
    return np.array(out)


rows, oof_all = [], {}
for name in which:
    for scen in SCEN:
        oofs = []
        for rep in range(N_REP):
            t0 = time.time(); oof = np.full(len(y), np.nan); sm = []
            for k, (tr, te) in enumerate(splits(scen, rep)):
                m = MODELS[name]().fit(df.iloc[tr]); oof[te] = m.predict(df.iloc[te]); sm.append(smoothness(m, te))
            sm = np.vstack(sm); oofs.append(oof)
            rows.append(dict(model=name, scenario=scen, rep=rep, **metrics(y, oof), NMI=np.nanmean(sm[:, 0]),
                             tanhR2=np.nanmedian(sm[:, 1]), sec=time.time() - t0))
            r = rows[-1]
            print(f'{name:11s} {scen:17s} rep{rep}  R2={r["R2"]:.3f} MAE={r["MAE"]:.1f} RMSE={r["RMSE"]:.1f}  '
                  f'NMI={r["NMI"]:.3f} tanhR2={r["tanhR2"]:.3f}  ({r["sec"]:.0f}s)', flush=True)
        oof_all[f'{name}|{scen}'] = np.mean(oofs, 0)
    pd.DataFrame(rows).to_csv(RES / f'cv_{tag}.csv', index=False)
    pd.DataFrame(oof_all, index=df.index).assign(cond=df.cond, Alloy=df.Alloy, test_temp=df.test_temp, Impact=y) \
        .to_csv(RES / f'oof_{tag}.csv', index=False)
r = pd.DataFrame(rows)
print(r.groupby(['scenario', 'model'])[['R2', 'MAE', 'RMSE', 'NMI', 'tanhR2']].mean().round(3).to_string())
