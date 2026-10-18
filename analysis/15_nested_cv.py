"""Nested cross-validation for the tuned tree models: the hyper-parameters are re-tuned inside every outer fold
(Optuna, n_inner trials, objective = R2 of an inner GroupKFold with the same grouping), so the outer estimate is free of the
tuning-selection bias.  Outer: 5-fold GroupKFold by condition and by alloy, one repetition.
usage: python 15_nested_cv.py XGB-direct|XGB-direct-mono|tanh-XGB [n_inner=20]      (run with XGBoost 3.2.0)"""
import sys, time, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, optuna, json
from sklearn.model_selection import GroupKFold
import common, models
from common import load, metrics, RES
optuna.logging.set_verbosity(optuna.logging.WARNING)

name = sys.argv[1]; n_inner = int(sys.argv[2]) if len(sys.argv) > 2 else 20
af = common.COMP + common.PROC + common.DERIVED; models.ALLOY_FEATS = af; models.FEATS = af + ['test_temp']
df = load(); y = df.Impact.values


def space(trial):
    if name == 'tanh-XGB':
        return dict(rounds=trial.suggest_int('rounds', 300, 2500, step=100), eta=trial.suggest_float('eta', 0.005, 0.1, log=True), depth=trial.suggest_int('depth', 2, 6),
                    min_child_weight=trial.suggest_float('min_child_weight', 1, 30, log=True), subsample=trial.suggest_float('subsample', 0.5, 1.0),
                    colsample=trial.suggest_float('colsample', 0.3, 1.0), reg_lambda=trial.suggest_float('reg_lambda', 0.1, 30, log=True))
    return dict(n_estimators=trial.suggest_int('n_estimators', 200, 2000, step=100), learning_rate=trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                max_depth=trial.suggest_int('max_depth', 2, 8), min_child_weight=trial.suggest_float('min_child_weight', 1, 30, log=True),
                subsample=trial.suggest_float('subsample', 0.5, 1.0), colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
                reg_lambda=trial.suggest_float('reg_lambda', 0.1, 30, log=True), reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                gamma=trial.suggest_float('gamma', 1e-3, 10, log=True))


def make(p):
    return {'XGB-direct': models.XGBDirect, 'XGB-direct-mono': models.XGBDirectMono, 'tanh-XGB': models.TanhXGB}[name](**p)


rows = []
for grp in ['cond', 'alloy_id']:
    oof = np.full(len(y), np.nan); t0 = time.time()
    for k, (tr, te) in enumerate(GroupKFold(5, shuffle=True, random_state=0).split(df, groups=df[grp].values)):
        dtr = df.iloc[tr]
        inner = list(GroupKFold(5, shuffle=True, random_state=1).split(dtr, groups=dtr[grp].values))

        def objective(trial):
            p = space(trial); o = np.zeros(len(dtr))
            for itr, ite in inner:
                o[ite] = make(p).fit(dtr.iloc[itr]).predict(dtr.iloc[ite])
            return metrics(dtr.Impact.values, o)['R2']
        st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=k))
        st.optimize(objective, n_trials=n_inner)
        oof[te] = make(st.best_params).fit(dtr).predict(df.iloc[te])
        print(f'{name} {grp} outer fold {k}: inner best R2={st.best_value:.3f} {st.best_params} ({time.time()-t0:.0f}s)', flush=True)
    r = metrics(y, oof); rows.append(dict(model=name, scenario={'cond': 'unseen-condition', 'alloy_id': 'unseen-alloy'}[grp], n_inner=n_inner, **r))
    print(f'NESTED {name} {grp}: R2={r["R2"]:.3f} MAE={r["MAE"]:.1f} RMSE={r["RMSE"]:.1f}', flush=True)
pd.DataFrame(rows).to_csv(RES / f'nested_cv_{name}.csv', index=False); print('done')
