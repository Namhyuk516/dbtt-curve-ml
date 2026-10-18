"""Sensitivity of the external (cast-steel) test to the network configuration: the best five Optuna trials of NN-direct and tanh-NN
(all within the fold-to-fold scatter of the tuning objective) are each trained on the full literature database and evaluated on the
four cast steels.  Shows how much the external-test result depends on a choice that cross-validation cannot resolve.
usage: python 17_cast_config_sensitivity.py"""
import sys, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from common import load, cast_alloys, CAST_MEAS, RES, metrics
import models

df = load(); cast = cast_alloys(); cd = CAST_MEAS.merge(cast, on='Alloy')
rows = []
for name, cls in [('tanh-NN', models.TanhNN), ('NN-direct', models.NNDirect)]:
    tr = pd.read_csv(RES / f'tune_trials_{name}.csv').sort_values('value', ascending=False).head(5)
    for rank, (_, t) in enumerate(tr.iterrows(), 1):
        hp = dict(h=int(t.params_h), drop=float(t.params_drop), lr=float(t.params_lr), wd=float(t.params_wd), epochs=int(t.params_epochs))
        m = cls(**hp).fit(df); p = m.predict(cd); r = metrics(cd.Impact.values, p)
        rows.append(dict(model=name, rank=rank, trial=int(t.number), objective=t.value, **hp, cast_MAE=r['MAE'], cast_R2=r['R2'],
                         pred25=np.mean(m.predict(cast.assign(test_temp=25.)))))
        print(f'{name:9s} rank{rank} trial{int(t.number):3d} obj={t.value:.3f} {hp} -> cast MAE={r["MAE"]:.1f} R2={r["R2"]:.2f}', flush=True)
res = pd.DataFrame(rows); res.to_csv(RES / 'supp_cast_config_sensitivity.csv', index=False)
print(res.groupby('model')[['objective', 'cast_MAE']].agg(['min', 'max']).round(3).to_string()); print('done')
