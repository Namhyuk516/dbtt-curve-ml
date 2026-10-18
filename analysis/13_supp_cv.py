"""Supplementary cross-validation experiments (same protocol as 03_compare.py: 5 folds x 3 repetitions, 3 split schemes).
  ablation_tree : XGB-direct / tanh-XGB / two-stage with and without the physics-derived descriptors, untuned and tuned
  ablation_nn   : NN-direct / tanh-NN with and without the physics-derived descriptors
  multi_temp    : all models (paper configuration) after removing the conditions tested at a single temperature
  fullsize      : all models after removing the sources with sub-size or unstated specimens
usage: python 13_supp_cv.py [ablation_tree|ablation_nn|multi_temp]      (run with XGBoost 3.2.0)"""
import sys, time, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
import common, models
from common import load, RES, metrics

job = sys.argv[1]
SCEN = {'random': None, 'unseen-condition': 'cond', 'unseen-alloy': 'alloy_id'}
N_REP = 3
UNTUNED_XGB = dict(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                   min_child_weight=3, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0)
UNTUNED_TANH = dict(rounds=1500, eta=0.02, depth=3, min_child_weight=3, subsample=0.8, colsample=0.8, reg_lambda=1.0)


def set_feats(derived):
    af = common.COMP + common.PROC + (common.DERIVED if derived else [])
    models.ALLOY_FEATS = af; models.FEATS = af + ['test_temp']


def cv(label, make, df, rows, **info):
    y = df.Impact.values
    for scen, grp in SCEN.items():
        for rep in range(N_REP):
            t0 = time.time(); oof = np.full(len(y), np.nan)
            sp = KFold(5, shuffle=True, random_state=rep).split(df) if grp is None else \
                 GroupKFold(5, shuffle=True, random_state=rep).split(df, groups=df[grp].values)
            for tr, te in sp:
                oof[te] = make().fit(df.iloc[tr]).predict(df.iloc[te])
            rows.append(dict(label=label, scenario=scen, rep=rep, n=len(y), **info, **metrics(y, oof)))
            r = rows[-1]; print(f'{label:32s} {scen:17s} rep{rep} R2={r["R2"]:.3f} MAE={r["MAE"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(RES / f'supp_{job}.csv', index=False)


df = load(); rows = []
if job == 'ablation_tree':
    for derived in (False, True):
        set_feats(derived); d = 'derived' if derived else 'base'
        cv(f'XGB-direct|{d}|untuned', lambda: models.XGBDirect(**UNTUNED_XGB), df, rows, model='XGB-direct', feats=d, tuned=False)
        cv(f'XGB-direct|{d}|tuned', lambda: models.XGBDirect(), df, rows, model='XGB-direct', feats=d, tuned=True)
        cv(f'XGB-direct-mono|{d}|untuned', lambda: models.XGBDirectMono(**UNTUNED_XGB), df, rows, model='XGB-direct-mono', feats=d, tuned=False)
        cv(f'XGB-direct-mono|{d}|tuned', lambda: models.XGBDirectMono(), df, rows, model='XGB-direct-mono', feats=d, tuned=True)
        cv(f'tanh-XGB|{d}|untuned', lambda: models.TanhXGB(**UNTUNED_TANH), df, rows, model='tanh-XGB', feats=d, tuned=False)
        cv(f'tanh-XGB|{d}|tuned', lambda: models.TanhXGB(), df, rows, model='tanh-XGB', feats=d, tuned=True)
        cv(f'two-stage|{d}', lambda: models.TwoStage(), df, rows, model='two-stage', feats=d, tuned=False)
elif job == 'ablation_nn':
    UNTUNED_NN = dict(h=64, drop=0.05, lr=3e-3, wd=1e-3, epochs=1500)
    for derived in (False, True):
        set_feats(derived); d = 'derived' if derived else 'base'
        cv(f'NN-direct|{d}|untuned', lambda: models.NNDirect(**UNTUNED_NN), df, rows, model='NN-direct', feats=d, tuned=False)
        cv(f'NN-direct|{d}|tuned', lambda: models.NNDirect(), df, rows, model='NN-direct', feats=d, tuned=True)
        cv(f'tanh-NN|{d}|untuned', lambda: models.TanhNN(**UNTUNED_NN), df, rows, model='tanh-NN', feats=d, tuned=False)
        cv(f'tanh-NN|{d}|tuned', lambda: models.TanhNN(), df, rows, model='tanh-NN', feats=d, tuned=True)
elif job == 'fullsize':
    set_feats(True)
    drop = common._in_ranges(df.No.values, common.SUBSIZE_ROWS + common.UNSTATED_ROWS); d2 = df[~drop].reset_index(drop=True)
    print('removed', int(drop.sum()), 'points from sub-size / unstated-specimen sources;', df.alloy_id.nunique() - d2.alloy_id.nunique(), 'steels', flush=True)
    for name in models.ORDER:
        cv(f'{name}|fullsize', models.MODELS[name], d2, rows, model=name, feats='derived', tuned=True)
elif job == 'multi_temp':
    set_feats(True); nT = df.groupby('cond').test_temp.transform('nunique'); d2 = df[nT >= 2].reset_index(drop=True)
    print('removed', len(df) - len(d2), 'single-temperature points;', df.cond.nunique() - d2.cond.nunique(), 'conditions', flush=True)
    for name in models.ORDER:
        cv(f'{name}|multi_temp', models.MODELS[name], d2, rows, model=name, feats='derived', tuned=True)
r = pd.DataFrame(rows)
print(r.groupby(['label', 'scenario'])[['R2', 'MAE', 'RMSE']].mean().round(3).to_string())
print('done')
