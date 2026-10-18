"""Model A: direct XGBoost regression  Impact = f(composition, heat treatment, test temperature).
Evaluation: repeated GroupKFold by alloy (unseen alloy), plus a random KFold for reference."""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np, pandas as pd, json, xgboost as xgb
from sklearn.model_selection import KFold
from common import load, FEATS, RES, metrics, group_splits

df = load(); X = df[FEATS].values; y = df.Impact.values
PARAMS = dict(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=3, reg_lambda=1.0, n_jobs=12, random_state=0)

def fit(Xtr, ytr, log=False):
    m = xgb.XGBRegressor(**PARAMS); m.fit(Xtr, np.log1p(ytr) if log else ytr); return m
def pred(m, X, log=False):
    p = m.predict(X); return np.expm1(p) if log else p

out = []
for log in [False, True]:
    # --- grouped CV (unseen alloys) ---
    oof = np.zeros((3, len(y)))
    for r, k, tr, te in group_splits(df):
        m = fit(X[tr], y[tr], log); oof[r, te] = pred(m, X[te], log)
    for r in range(3):
        out.append(dict(model='XGB_direct'+('_log' if log else ''), split='GroupKFold(alloy)', rep=r, **metrics(y, oof[r])))
    df[f'oof_xgb{"_log" if log else ""}'] = oof.mean(0)
    # --- random CV (reference; same alloy may be in train) ---
    oof_r = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = fit(X[tr], y[tr], log); oof_r[te] = pred(m, X[te], log)
    out.append(dict(model='XGB_direct'+('_log' if log else ''), split='KFold(random)', rep=0, **metrics(y, oof_r)))

res = pd.DataFrame(out); print(res.groupby(['model','split'])[['R2','MAE','RMSE']].agg(['mean','std']).round(3))
res.to_csv(RES/'cv_xgb_direct.csv', index=False)
df[['cond','alloy_id','Alloy','test_temp','Impact','oof_xgb','oof_xgb_log']].to_csv(RES/'oof_xgb_direct.csv', index=False)
# final model on all data
m = fit(X, y, False); m.save_model(str(RES/'xgb_direct.json'))
