"""Optuna hyper-parameter tuning (derived features on).  Objective = mean R2 of GroupKFold(cond) and GroupKFold(alloy), 1 repeat.
The same budget/objective is used for every model (trees and neural networks).
usage: python 09_tune.py XGB-direct|XGB-direct-mono|tanh-XGB|NN-direct|tanh-NN n_trials"""
import sys, warnings, time; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, optuna, json
from sklearn.model_selection import GroupKFold
import common, models
from common import load, metrics, RES

name, n_trials = sys.argv[1], int(sys.argv[2])
af = common.COMP + common.PROC + common.DERIVED; models.ALLOY_FEATS = af; models.FEATS = af + ['test_temp']
df = load()
SPLITS = {s: list(GroupKFold(5, shuffle=True, random_state=0).split(df, groups=df[s].values)) for s in ['cond', 'alloy_id']}

def evaluate(make):
    r = {}
    for s, sp in SPLITS.items():
        oof = np.zeros(len(df))
        for tr, te in sp:
            m = make().fit(df.iloc[tr]); oof[te] = m.predict(df.iloc[te])
        r[s] = metrics(df.Impact.values, oof)['R2']
    return r

def xgb_space(trial):
    return dict(n_estimators=trial.suggest_int('n_estimators', 200, 2000, step=100), learning_rate=trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                max_depth=trial.suggest_int('max_depth', 2, 8), min_child_weight=trial.suggest_float('min_child_weight', 1, 30, log=True),
                subsample=trial.suggest_float('subsample', 0.5, 1.0), colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
                reg_lambda=trial.suggest_float('reg_lambda', 0.1, 30, log=True), reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                gamma=trial.suggest_float('gamma', 1e-3, 10, log=True))

def nn_space(trial):
    return dict(h=trial.suggest_categorical('h', [32, 64, 128]), drop=trial.suggest_float('drop', 0.0, 0.3),
                lr=trial.suggest_float('lr', 1e-3, 1e-2, log=True), wd=trial.suggest_float('wd', 1e-5, 1e-1, log=True),
                epochs=trial.suggest_categorical('epochs', [500, 1000, 1500, 2500]))

def objective(trial):
    if name == 'XGB-direct': p = xgb_space(trial); make = lambda: models.XGBDirect(**p)
    elif name == 'XGB-direct-mono': p = xgb_space(trial); make = lambda: models.XGBDirectMono(**p)
    elif name == 'tanh-XGB':
        p = dict(rounds=trial.suggest_int('rounds', 300, 2500, step=100), eta=trial.suggest_float('eta', 0.005, 0.1, log=True), depth=trial.suggest_int('depth', 2, 6),
                 min_child_weight=trial.suggest_float('min_child_weight', 1, 30, log=True), subsample=trial.suggest_float('subsample', 0.5, 1.0),
                 colsample=trial.suggest_float('colsample', 0.3, 1.0), reg_lambda=trial.suggest_float('reg_lambda', 0.1, 30, log=True))
        make = lambda: models.TanhXGB(**p)
    elif name == 'NN-direct': p = nn_space(trial); make = lambda: models.NNDirect(**p)
    elif name == 'tanh-NN': p = nn_space(trial); make = lambda: models.TanhNN(**p)
    else: raise ValueError(name)
    t0 = time.time(); r = evaluate(make); trial.set_user_attr('r', r)
    print(f'trial {trial.number:3d} cond={r["cond"]:.3f} alloy={r["alloy_id"]:.3f} ({time.time()-t0:.0f}s) {p}', flush=True)
    return 0.5 * (r['cond'] + r['alloy_id'])

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
best = study.best_trial
print('BEST', best.value, best.params, best.user_attrs['r'])
json.dump(dict(value=best.value, params=best.params, r=best.user_attrs['r']), open(RES / f'tuned_{name}.json', 'w'), indent=1)
study.trials_dataframe().to_csv(RES / f'tune_trials_{name}.csv', index=False)
