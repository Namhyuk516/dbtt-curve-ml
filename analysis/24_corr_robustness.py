"""Robustness of the Spearman correlations between descriptors and fitted curve parameters (Table 5 of the article):
(a) all 61 complete curves versus the well-resolved curves only (classification of 21_param_accuracy_resolved.py);
(b) for the elements that many sources do not report (entered as zero), the correlation among the curves with a reported (non-zero) value.
usage: python 24_corr_robustness.py"""
import sys, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent)); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
from common import RES

par = pd.read_csv(RES / 'curve_params.csv'); fu = pd.read_csv(RES / 'curve_fit_uncertainty_resolved.csv').set_index('cond')
par['resolved'] = fu.resolved.reindex(par.cond).values
DESC = ['C', 'Mn', 'T_temp', 'HJ_temper', 'Ms_Andrews', 'Q_temp', 'Q_time', 'T_time', 'T_type', 'N', 'Nb', 'Al', 'Cu', 'W', 'S', 'Pcm', 'Ni', 'Mo', 'Rolled_Forged_Cast']
rows = []
for c in DESC:
    r = dict(descriptor=c, n_nonzero=int((par[c] > 0).sum()), n_nonzero_resolved=int((par.loc[par.resolved, c] > 0).sum()))
    for lab, g in [('all', par), ('resolved', par[par.resolved])]:
        for t, tn in [('T0', 'Tt'), ('USE', 'USE')]:
            rho, p = stats.spearmanr(g[c], g[t]); r[f'rho_{tn}_{lab}'] = rho; r[f'p_{tn}_{lab}'] = p
    g = par[par[c] > 0]
    if c in ('N', 'Nb', 'Al', 'S', 'Cu', 'W') and len(g) >= 8:
        for t, tn in [('T0', 'Tt'), ('USE', 'USE')]:
            rho, p = stats.spearmanr(g[c], g[t]); r[f'rho_{tn}_reported'] = rho; r[f'p_{tn}_reported'] = p
    rows.append(r)
res = pd.DataFrame(rows); res.to_csv(RES / 'supp_corr_robustness.csv', index=False)
print('curves:', len(par), '| well resolved:', int(par.resolved.sum()))
print(res.round(2).to_string(index=False)); print('done')
