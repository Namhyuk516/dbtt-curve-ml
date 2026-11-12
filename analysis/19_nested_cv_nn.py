"""Nested cross-validation for the two neural networks (companion of 15_nested_cv.py for the tree models): the hyper-parameters
are re-selected inside every outer fold (Optuna, n_inner trials, objective = R2 of an inner grouped split of the training folds with
the same grouping), so that the outer estimate is free of the tuning-selection bias.
Outer: 5-fold GroupKFold by condition and by steel, one repetition (same fold assignment as 15_nested_cv.py and repetition 0 of
03_compare.py).  To keep the computation feasible the inner loop uses a three-fold grouped split and ensembles of three networks;
the model refitted on the outer training folds is the usual ensemble of five.  Search space identical to 09_tune.py.
usage: python 19_nested_cv_nn.py NN-direct|tanh-NN [n_inner=15]"""
import sys, time, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, optuna
from sklearn.model_selection import GroupKFold
import common, models
from common import load, metrics, RES
optuna.logging.set_verbosity(optuna.logging.WARNING)

name = sys.argv[1]; n_inner = int(sys.argv[2]) if len(sys.argv) > 2 else 15
af = common.COMP + common.PROC + common.DERIVED; models.ALLOY_FEATS = af; models.FEATS = af + ['test_temp']
df = load(); y = df.Impact.values
cls = {'NN-direct': models.NNDirect, 'tanh-NN': models.TanhNN}[name]


def space(trial):
    return dict(h=trial.suggest_categorical('h', [32, 64, 128]), drop=trial.suggest_float('drop', 0.0, 0.3),
                lr=trial.suggest_float('lr', 1e-3, 1e-2, log=True), wd=trial.suggest_float('wd', 1e-5, 1e-1, log=True),
                epochs=trial.suggest_categorical('epochs', [500, 1000, 1500, 2500]))


rows, sel = [], []
for grp in ['cond', 'alloy_id']:
    oof = np.full(len(y), np.nan); t0 = time.time()
    for k, (tr, te) in enumerate(GroupKFold(5, shuffle=True, random_state=0).split(df, groups=df[grp].values)):
        dtr = df.iloc[tr]
        inner = list(GroupKFold(3, shuffle=True, random_state=1).split(dtr, groups=dtr[grp].values))

        def objective(trial):
            p = space(trial); o = np.zeros(len(dtr))
            for itr, ite in inner:
                o[ite] = cls(n_ens=3, **p).fit(dtr.iloc[itr]).predict(dtr.iloc[ite])
            return metrics(dtr.Impact.values, o)['R2']
        st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=k))
        st.optimize(objective, n_trials=n_inner)
        oof[te] = cls(**st.best_params).fit(dtr).predict(df.iloc[te])
        sel.append(dict(model=name, group=grp, fold=k, inner_R2=st.best_value, **st.best_params))
        print(f'{name} {grp} outer fold {k}: inner best R2={st.best_value:.3f} {st.best_params} ({time.time()-t0:.0f}s)', flush=True)
        pd.DataFrame(sel).to_csv(RES / f'nested_cv_{name}_selected.csv', index=False)
    r = metrics(y, oof); rows.append(dict(model=name, scenario={'cond': 'unseen-condition', 'alloy_id': 'unseen-alloy'}[grp], n_inner=n_inner, **r))
    print(f'NESTED {name} {grp}: R2={r["R2"]:.3f} MAE={r["MAE"]:.1f} RMSE={r["RMSE"]:.1f}', flush=True)
    pd.DataFrame(rows).to_csv(RES / f'nested_cv_{name}.csv', index=False)
print('done')
