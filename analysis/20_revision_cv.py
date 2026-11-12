"""Additional cross-validation experiments (same protocol as 03_compare.py: five folds, three repetitions).
  fullsize_same_test : each model is trained (a) on all training-fold measurements and (b) on the training-fold measurements from
                       full-size-specimen sources only, and both are evaluated on the SAME held-out full-size measurements, so that the
                       effect of the training-data composition is separated from the change of the evaluation population (cf. Table S7).
  publication        : grouped cross-validation with whole publications held out (leave-publication-out).
  twostage_fair      : two-stage model with fixed and with Optuna-tuned regressors, and the end-to-end models trained only on the
                       conditions eligible for the two-stage model (>= 4 test temperatures); all evaluated on the same held-out
                       measurements (all, and those of complete-curve conditions).
usage: python 20_revision_cv.py fullsize_same_test|publication|twostage_fair      (run with XGBoost 3.2.0)"""
import sys, time, json, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
import common, models
from common import load, RES, metrics, source_table

job = sys.argv[1]
af = common.COMP + common.PROC + common.DERIVED; models.ALLOY_FEATS = af; models.FEATS = af + ['test_temp']
df = load(); y = df.Impact.values
pub = np.zeros(len(df), int)
for r in source_table(df):
    for a, b in r['ranges']: pub[(df.No >= a) & (df.No <= b)] = r['No']
df['pub'] = pub
SCEN = {'random': None, 'unseen-condition': 'cond', 'unseen-alloy': 'alloy_id'}
N_REP = 3


def splits(d, grp, rep):
    if grp is None: return list(KFold(5, shuffle=True, random_state=rep).split(d))
    return list(GroupKFold(5, shuffle=True, random_state=rep).split(d, groups=d[grp].values))


rows = []
if job == 'fullsize_same_test':
    full = ~common._in_ranges(df.No.values, common.SUBSIZE_ROWS + common.UNSTATED_ROWS)
    print('full-size measurements:', int(full.sum()), 'of', len(df), flush=True)
    for name in models.ORDER:
        for scen in ['unseen-condition', 'unseen-alloy']:
            for rep in range(N_REP):
                t0 = time.time(); oa = np.full(len(y), np.nan); ob = np.full(len(y), np.nan)
                for tr, te in splits(df, SCEN[scen], rep):
                    te_f = te[full[te]]
                    if len(te_f) == 0: continue
                    oa[te_f] = models.MODELS[name]().fit(df.iloc[tr]).predict(df.iloc[te_f])
                    ob[te_f] = models.MODELS[name]().fit(df.iloc[tr[full[tr]]]).predict(df.iloc[te_f])
                for lab, o in [('all sources', oa), ('full-size sources only', ob)]:
                    rows.append(dict(model=name, scenario=scen, rep=rep, training=lab, n_test=int(full.sum()), **metrics(y[full], o[full])))
                print(f'{name:16s} {scen:17s} rep{rep} all: R2={rows[-2]["R2"]:.3f} MAE={rows[-2]["MAE"]:.1f} | full-only: R2={rows[-1]["R2"]:.3f} MAE={rows[-1]["MAE"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
            pd.DataFrame(rows).to_csv(RES / f'supp_{job}.csv', index=False)
    r = pd.DataFrame(rows); print(r.groupby(['model', 'scenario', 'training'])[['R2', 'MAE', 'RMSE']].mean().round(3).to_string())

elif job == 'publication':
    print('publications:', df['pub'].nunique(), '| publications with more than one steel:', int((df.groupby('pub').alloy_id.nunique() > 1).sum()),
          '| steels occurring in more than one publication:', int((df.groupby('alloy_id')['pub'].nunique() > 1).sum()), flush=True)
    for name in models.ORDER:
        for rep in range(N_REP):
            t0 = time.time(); oof = np.full(len(y), np.nan)
            for tr, te in splits(df, 'pub', rep):
                oof[te] = models.MODELS[name]().fit(df.iloc[tr]).predict(df.iloc[te])
            rows.append(dict(model=name, scenario='unseen-publication', rep=rep, **metrics(y, oof)))
            print(f'{name:16s} rep{rep} R2={rows[-1]["R2"]:.3f} MAE={rows[-1]["MAE"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
        pd.DataFrame(rows).to_csv(RES / f'supp_{job}.csv', index=False)
    r = pd.DataFrame(rows); print(r.groupby('model')[['R2', 'MAE', 'RMSE']].agg(['mean', 'std']).round(3).to_string())

elif job == 'twostage_fair':
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

    class TwoStageP(models.TwoStage):
        """two-stage model with its own regressor settings"""
        def __init__(self, p): super().__init__(); self.p = p
        def fit(self, d):
            old = dict(models.TWO_STAGE_P); models.TWO_STAGE_P.update(self.p)
            try: return super().fit(d)
            finally: models.TWO_STAGE_P.clear(); models.TWO_STAGE_P.update(old)

    # --- tune the two-stage regressors with the procedure of 09_tune.py (30 trials, mean R2 of the two grouped schemes, one repetition)
    SPL = {s: splits(df, s, 0) for s in ['cond', 'alloy_id']}
    def objective(trial):
        p = dict(n_estimators=trial.suggest_int('n_estimators', 50, 1000, step=50), learning_rate=trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                 max_depth=trial.suggest_int('max_depth', 2, 6), min_child_weight=trial.suggest_float('min_child_weight', 1, 10, log=True),
                 subsample=trial.suggest_float('subsample', 0.5, 1.0), colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
                 reg_lambda=trial.suggest_float('reg_lambda', 0.1, 30, log=True))
        r = []
        for s, sp in SPL.items():
            oof = np.zeros(len(y))
            for tr, te in sp: oof[te] = TwoStageP(p).fit(df.iloc[tr]).predict(df.iloc[te])
            r.append(metrics(y, oof)['R2'])
        return float(np.mean(r))
    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0)); st.optimize(objective, n_trials=30)
    print('two-stage tuned:', round(st.best_value, 3), st.best_params, flush=True)
    json.dump(dict(value=st.best_value, params=st.best_params), open(RES / 'tuned_two-stage_revision.json', 'w'), indent=1)
    tuned = st.best_params

    def eligible(d):          # conditions with >= 4 distinct test temperatures inside the training fold (= those used by the two-stage model)
        return (d.groupby('cond').test_temp.transform('nunique') >= 4).values

    complete = (df.groupby('cond').test_temp.transform('nunique') >= 4).values
    VAR = {'two-stage (fixed settings)': (lambda: models.TwoStage(), False),
           'two-stage (tuned)': (lambda: TwoStageP(tuned), False),
           'tanh-XGB, complete curves only': (lambda: models.TanhXGB(), True),
           'tanh-NN, complete curves only': (lambda: models.TanhNN(), True),
           'tanh-XGB, all measurements': (lambda: models.TanhXGB(), False),
           'tanh-NN, all measurements': (lambda: models.TanhNN(), False)}
    for lab, (make, restrict) in VAR.items():
        for scen, grp in SCEN.items():
            for rep in range(N_REP):
                t0 = time.time(); oof = np.full(len(y), np.nan); ntr = []
                for tr, te in splits(df, grp, rep):
                    dtr = df.iloc[tr]
                    if restrict: dtr = dtr[eligible(dtr)]
                    ntr.append(len(dtr)); oof[te] = make().fit(dtr).predict(df.iloc[te])
                ma, mc = metrics(y, oof), metrics(y[complete], oof[complete])
                rows.append(dict(variant=lab, scenario=scen, rep=rep, n_train=np.mean(ntr), R2=ma['R2'], MAE=ma['MAE'], RMSE=ma['RMSE'],
                                 R2_complete=mc['R2'], MAE_complete=mc['MAE']))
                print(f'{lab:32s} {scen:17s} rep{rep} all: R2={ma["R2"]:.3f} MAE={ma["MAE"]:.1f} | complete-curve points: R2={mc["R2"]:.3f} MAE={mc["MAE"]:.1f} ({time.time()-t0:.0f}s)', flush=True)
            pd.DataFrame(rows).to_csv(RES / f'supp_{job}.csv', index=False)
    r = pd.DataFrame(rows); print(r.groupby(['variant', 'scenario'])[['n_train', 'R2', 'MAE', 'R2_complete', 'MAE_complete']].mean().round(3).to_string())
print('done')
