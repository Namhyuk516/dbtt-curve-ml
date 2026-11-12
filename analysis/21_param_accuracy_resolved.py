"""Parameter accuracy of the equation-based models separately for well-resolved and poorly resolved reference curves
(extension of 16_param_accuracy.py; uses results/curve_fit_uncertainty.csv written by that script).
A reference curve is called well resolved when the least-squares fit of Equation (1) has a non-singular covariance matrix,
standard errors SE(Tt) <= 20 C and SE(USE) <= 20 J, no USE or Cw at a bound, and Tt inside the tested temperature range.
Per-condition predictions of Tt and USE are collected under grouped cross-validation (three repetitions) and compared with the fitted values.
usage: python 21_param_accuracy_resolved.py      (run with XGBoost 3.2.0)"""
import sys, time, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from common import load, RES
from models import MODELS

df = load(); par = pd.read_csv(RES / 'curve_params.csv').set_index('cond'); fa = pd.read_csv(RES / 'curve_fit_uncertainty.csv').set_index('cond')
fa['resolved'] = np.isfinite(fa.se_Tt) & (fa.se_Tt <= 20) & (fa.se_USE <= 20) & ~fa.USE_at_bound & ~fa.Cw_at_bound & fa.Tt_inside_range
print('well-resolved reference curves:', int(fa.resolved.sum()), 'of', len(fa), '| singular covariance:', int((~np.isfinite(fa.se_Tt)).sum()), flush=True)
for lab, g in fa.groupby('resolved'):
    print('resolved' if lab else 'poorly resolved', len(g), 'median nT %.0f' % g.nT.median(), 'median range %.0f' % g.Trange.median(),
          'SD Tt %.1f SD USE %.1f' % (par.loc[g.index].T0.std(), par.loc[g.index].USE.std()))
complete = set(par.index); rec = []
for name in ['two-stage', 'tanh-XGB', 'tanh-NN']:
    for grp, scen in [('cond', 'unseen-condition'), ('alloy_id', 'unseen-alloy')]:
        for rep in range(3):
            t0 = time.time()
            for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(df, groups=df[grp].values):
                m = MODELS[name]().fit(df.iloc[tr]); dte = df.iloc[te]
                u = dte[dte.cond.isin(complete)].drop_duplicates('cond')
                if len(u) == 0: continue
                P = m.curve_params(u.assign(test_temp=25.))
                for c, (_, p) in zip(u.cond.values, P.iterrows()):
                    rec.append(dict(model=name, scenario=scen, rep=rep, cond=c, Tt_pred=p.T0, USE_pred=p.USE, Tt_fit=par.T0[c], USE_fit=par.USE[c], resolved=bool(fa.resolved[c])))
            print(name, scen, rep, f'({time.time()-t0:.0f}s)', flush=True)
rec = pd.DataFrame(rec); rec.to_csv(RES / 'supp_param_accuracy_percond.csv', index=False)
rec['eT'] = rec.Tt_pred - rec.Tt_fit; rec['eU'] = rec.USE_pred - rec.USE_fit
out = []
for (name, scen, res, rep), g in rec.groupby(['model', 'scenario', 'resolved', 'rep']):
    out.append(dict(model=name, scenario=scen, resolved=res, rep=rep, n=len(g), MAE_Tt=g.eT.abs().mean(), bias_Tt=g.eT.mean(), MAE_USE=g.eU.abs().mean(), bias_USE=g.eU.mean(), within30=(g.eT.abs() <= 30).mean()))
out = pd.DataFrame(out); out.to_csv(RES / 'supp_param_accuracy_resolved.csv', index=False)
print(out.groupby(['model', 'scenario', 'resolved'])[['n', 'MAE_Tt', 'bias_Tt', 'MAE_USE', 'bias_USE', 'within30']].mean().round(2).to_string())
fa.reset_index().to_csv(RES / 'curve_fit_uncertainty_resolved.csv', index=False)
print('done')
