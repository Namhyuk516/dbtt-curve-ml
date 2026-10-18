"""Extra numbers for Section 3.4: (a) predicted energy at 25 C of all six literature-trained models for the cast steels and the
under-estimation relative to the measured 25 C mean; (b) the tanh-NN configuration selected in the previous tuning run
(2 x 128 units, dropout 0.13, lr 0.0098, wd 4.7e-4, 1000 epochs) evaluated on the current database: tuning objective and cast-steel MAE.
usage: python 18_cast_extra.py"""
import sys, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from common import load, cast_alloys, CAST_MEAS, RES, metrics
import models

df = load(); cast = cast_alloys(); cd = CAST_MEAS.merge(cast, on='Alloy'); c25 = cast.assign(test_temp=25.)
m25 = CAST_MEAS[CAST_MEAS.test_temp == 25].groupby('Alloy', sort=False).Impact.mean().loc[cast.Alloy].values
print('measured 25C mean', dict(zip(cast.Alloy, m25.round(0))))
rows = []
for name in models.ORDER:
    m = models.MODELS[name]().fit(df); p = m.predict(c25); r = metrics(cd.Impact.values, m.predict(cd))
    rows.append(dict(model=name, **{f'pred25_{a}': v for a, v in zip(cast.Alloy, p)}, under_min=(m25 - p).min(), under_max=(m25 - p).max(), MAE=r['MAE']))
    print(f'{name:16s} pred25={np.round(p,0)} under={np.round(m25-p,0)} MAE={r["MAE"]:.1f}', flush=True)
pd.DataFrame(rows).to_csv(RES / 'cast_pred25.csv', index=False)

prev = dict(h=128, drop=0.1349, lr=0.00979, wd=4.69e-4, epochs=1000)
SPL = {s: list(GroupKFold(5, shuffle=True, random_state=0).split(df, groups=df[s].values)) for s in ['cond', 'alloy_id']}
r = {}
for s, sp in SPL.items():
    oof = np.zeros(len(df))
    for tr, te in sp: oof[te] = models.TanhNN(**prev).fit(df.iloc[tr]).predict(df.iloc[te])
    r[s] = metrics(df.Impact.values, oof)['R2']
m = models.TanhNN(**prev).fit(df); rc = metrics(cd.Impact.values, m.predict(cd))
print(f'previous tanh-NN config {prev}: objective={(r["cond"]+r["alloy_id"])/2:.3f} (cond {r["cond"]:.3f}, alloy {r["alloy_id"]:.3f}) cast MAE={rc["MAE"]:.1f} R2={rc["R2"]:.2f} pred25={np.round(m.predict(c25),0)}')
pd.DataFrame([dict(model='tanh-NN (previous config)', **prev, objective=(r['cond'] + r['alloy_id']) / 2, cast_MAE=rc['MAE'], cast_R2=rc['R2'])]).to_csv(RES / 'cast_prev_config.csv', index=False)
print('done')
