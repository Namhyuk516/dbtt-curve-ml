"""Sensitivity to the coding of unreported elements.  In the main analysis an element that a source does not report is entered as zero.
Here an element is treated as MISSING for all steels of a publication in which it is zero throughout (i.e., presumably not reported),
and as a true zero otherwise.  Tree models receive NaN (XGBoost handles missing values natively); the neural networks receive the mean
of the reported values plus one indicator input per element.  The physics-derived descriptors are computed as in the main analysis.
For the cast steels the eleven elements that were not analysed are missing.  Hyper-parameters as in the main analysis (not re-tuned).
Output: cross-validation (five folds, three repetitions, three schemes) and external test for both codings.
usage: python 25_missing_as_nan.py      (run with XGBoost 3.2.0)"""
import sys, time, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
import common, models
from common import load, cast_alloys, CAST_MEAS, RES, metrics, source_table, COMP

BASE_AF = common.COMP + common.PROC + common.DERIVED
df = load(); pub = np.zeros(len(df), int)
for r in source_table(df):
    for a, b in r['ranges']: pub[(df.No >= a) & (df.No <= b)] = r['No']
df['pub'] = pub
cast = cast_alloys(); ANALYSED = ['C', 'Mn', 'Si', 'Cr', 'Mo', 'Ni']
df[COMP] = df[COMP].astype(float); cast[COMP] = cast[COMP].astype(float)

# ---- missing mask: element zero for every steel of a publication -> not reported
miss = pd.DataFrame(False, index=df.index, columns=COMP)
for e in COMP:
    allzero = df.groupby('pub')[e].transform(lambda s: (s == 0).all())
    miss[e] = allzero.values
ELEM = [e for e in COMP if miss[e].any()]
print('elements with unreported values:', {e: f'{int(miss[e].sum())} measurements / {df.loc[miss[e], "pub"].nunique()} publications' for e in ELEM}, flush=True)
cmiss = pd.DataFrame({e: [e not in ANALYSED] * len(cast) for e in ELEM})

d_tree = df.copy(); c_tree = cast.copy()
for e in ELEM: d_tree.loc[miss[e], e] = np.nan; c_tree.loc[cmiss[e].values, e] = np.nan
d_nn = df.copy(); c_nn = cast.copy(); IND = []
for e in ELEM:
    mu = df.loc[~miss[e], e].mean()
    d_nn.loc[miss[e], e] = mu; c_nn.loc[cmiss[e].values, e] = mu
    d_nn['miss_' + e] = miss[e].astype(float).values; c_nn['miss_' + e] = cmiss[e].astype(float).values; IND.append('miss_' + e)
cd = {k: CAST_MEAS.merge(c, on='Alloy') for k, c in [('zero', cast), ('tree', c_tree), ('nn', c_nn)]}

SCEN = {'random': None, 'unseen-condition': 'cond', 'unseen-alloy': 'alloy_id'}
def data_for(name, coding):
    """-> (literature frame, cast frame, alloy feature list)"""
    if coding == 'zero': return df, cd['zero'], BASE_AF
    if 'NN' in name: return d_nn, cd['nn'], BASE_AF + IND
    return d_tree, cd['tree'], BASE_AF

rows, crow = [], []; y = df.Impact.values
for name in models.ORDER:
    for coding in ['zero', 'missing']:
        D, C, af = data_for(name, coding); models.ALLOY_FEATS = af; models.FEATS = af + ['test_temp']
        for scen, grp in SCEN.items():
            for rep in range(3):
                t0 = time.time(); oof = np.full(len(y), np.nan)
                sp = KFold(5, shuffle=True, random_state=rep).split(D) if grp is None else GroupKFold(5, shuffle=True, random_state=rep).split(D, groups=D[grp].values)
                for tr, te in sp: oof[te] = models.MODELS[name]().fit(D.iloc[tr]).predict(D.iloc[te])
                rows.append(dict(model=name, coding=coding, scenario=scen, rep=rep, **metrics(y, oof)))
            m_ = pd.DataFrame(rows[-3:]); print(f'{name:16s} {coding:8s} {scen:17s} R2={m_.R2.mean():.3f} MAE={m_.MAE.mean():.1f}', flush=True)
        m = models.MODELS[name]().fit(D); r = metrics(C.Impact.values, m.predict(C))
        crow.append(dict(model=name, coding=coding, cast_MAE=r['MAE'], cast_R2=r['R2'])); print(f'{name:16s} {coding:8s} cast MAE={r["MAE"]:.1f} R2={r["R2"]:.2f}', flush=True)
        pd.DataFrame(rows).to_csv(RES / 'supp_missing_coding_cv.csv', index=False); pd.DataFrame(crow).to_csv(RES / 'supp_missing_coding_cast.csv', index=False)
r = pd.DataFrame(rows)
print(r.groupby(['model', 'coding', 'scenario'])[['R2', 'MAE']].mean().unstack('scenario').round(3).to_string())
print(pd.DataFrame(crow).round(2).to_string(index=False)); print('done')
