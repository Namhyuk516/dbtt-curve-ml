"""Unseen-steel cross-validation error by steel class (from results/oof_all.csv; no re-training) -> supp_class_errors_unseen_steel.csv"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np, pandas as pd
from common import load, RES, COMP
df = load(); o = pd.read_csv(RES / 'oof_all.csv')
assert np.allclose(o.Impact.values, df.Impact.values)
def steel_class(r):
    if r.Cr >= 7: return '9–12Cr heat-resistant'
    if r.Mn >= 3: return 'Medium-Mn'
    if r.C >= 0.45: return 'High-carbon'
    return 'Plain carbon / HSLA' if r[COMP].sum() < 2.5 else 'Low-alloy'
df['cls'] = df.apply(steel_class, axis=1)
M = ['XGB-direct', 'XGB-direct-mono', 'NN-direct', 'two-stage', 'tanh-NN', 'tanh-XGB']
def rec(name, g):
    r = dict(cls=name, n=len(g), steels=g.alloy_id.nunique())
    for m in M: r[m + '_MAE'] = np.abs(o.loc[g.index, f'{m}|unseen-alloy'] - g.Impact).mean()
    return r
rows = [rec(c, df[df.cls == c]) for c in ['Low-alloy', 'Plain carbon / HSLA', '9–12Cr heat-resistant', 'Medium-Mn', 'High-carbon']]
rows.append(rec('All', df))
sub = df[df.C.between(0.10, 0.30) & (df.T_temp >= 550) & (df.Cr < 7) & (df.Mn < 3) & (df.B < 0.1)]
rows.append(rec('Low-carbon quenched-and-tempered subset', sub))
t = pd.DataFrame(rows); t.to_csv(RES / 'supp_class_errors_unseen_steel.csv', index=False, encoding='utf-8-sig'); print(t.round(1).to_string())
