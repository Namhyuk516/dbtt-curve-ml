"""Parameter-level evaluation (reviewer M4/m4).
(a) Uncertainty of the 62 tanh fits: standard errors of T_t and USE from the least-squares covariance, and how many fits hit a bound.
(b) Accuracy of the *predicted* curve parameters of the equation-based models on the 62 complete-curve conditions under grouped CV
    (unseen condition / unseen steel, 3 repetitions): MAE of predicted vs fitted T_t and USE.
usage: python 16_param_accuracy.py      (run with XGBoost 3.2.0)"""
import sys, time, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupKFold
from common import load, RES, tanh_curve
from models import MODELS

df = load(); par = pd.read_csv(RES / 'curve_params.csv')

# ---------------------------------------------------------------- (a) fit uncertainty and bound hits
rows = []
for _, r in par.iterrows():
    g = df[df.cond == r.cond]; T, E = g.test_temp.values, g.Impact.values
    lo, hi = E.min(), E.max(); p0 = [max(lo, 0.5), hi, T[np.argmin(np.abs(E - (lo + hi) / 2))], 30]
    try:
        popt, pcov = curve_fit(tanh_curve, T, E, p0=p0, bounds=([0, 0, -300, 3], [hi, 1.5 * hi + 5, 400, 300]), maxfev=20000)
        se = np.sqrt(np.diag(pcov))
    except Exception:
        popt, se = [np.nan] * 4, [np.nan] * 4
    rows.append(dict(cond=r.cond, Alloy=r.Alloy, nT=r.nT, Trange=r.Tmax - r.Tmin, LSE=popt[0], USE=popt[1], Tt=popt[2], Cw=popt[3],
                     se_LSE=se[0], se_USE=se[1], se_Tt=se[2], se_Cw=se[3],
                     LSE_at_bound=popt[0] <= 1e-6, USE_at_bound=popt[1] >= 1.5 * hi + 5 - 1e-6, Cw_at_bound=(popt[3] <= 3 + 1e-6) or (popt[3] >= 300 - 1e-6),
                     Tt_inside_range=(r.Tmin <= popt[2] <= r.Tmax)))
fa = pd.DataFrame(rows); fa.to_csv(RES / 'curve_fit_uncertainty.csv', index=False)
print('=== (a) fit uncertainty (62 curves)')
print('median SE: Tt %.1f °C, USE %.1f J; 90th pct: Tt %.1f °C, USE %.1f J' % (fa.se_Tt.median(), fa.se_USE.median(), fa.se_Tt.quantile(.9), fa.se_USE.quantile(.9)))
print('SE_Tt > 20 °C:', int((fa.se_Tt > 20).sum()), '| SE not finite:', int((~np.isfinite(fa.se_Tt)).sum()))
print('bound hits: LSE=0:', int(fa.LSE_at_bound.sum()), ' USE upper bound:', int(fa.USE_at_bound.sum()), ' Cw at 3 or 300:', int(fa.Cw_at_bound.sum()), ' Tt outside tested range:', int((~fa.Tt_inside_range).sum()))

# ---------------------------------------------------------------- (b) predicted vs fitted parameters under grouped CV
complete = set(par.cond)
out = []
for name in ['two-stage', 'tanh-XGB', 'tanh-NN']:
    for grp, scen in [('cond', 'unseen-condition'), ('alloy_id', 'unseen-alloy')]:
        for rep in range(3):
            t0 = time.time(); preds = {}
            for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(df, groups=df[grp].values):
                m = MODELS[name]().fit(df.iloc[tr]); dte = df.iloc[te]
                u = dte[dte.cond.isin(complete)].drop_duplicates('cond')
                if len(u) == 0: continue
                P = m.curve_params(u.assign(test_temp=25.))
                for c, (_, p) in zip(u.cond.values, P.iterrows()): preds[c] = (p.T0, p.USE)
            m2 = par.set_index('cond').loc[list(preds)]
            pt = np.array([preds[c][0] for c in m2.index]); pu = np.array([preds[c][1] for c in m2.index])
            out.append(dict(model=name, scenario=scen, rep=rep, n=len(m2), MAE_Tt=np.abs(pt - m2.T0.values).mean(), bias_Tt=(pt - m2.T0.values).mean(),
                            MAE_USE=np.abs(pu - m2.USE.values).mean(), bias_USE=(pu - m2.USE.values).mean(),
                            within30=(np.abs(pt - m2.T0.values) <= 30).mean(), sec=time.time() - t0))
            r = out[-1]; print(f'{name:10s} {scen:17s} rep{rep} n={r["n"]} MAE_Tt={r["MAE_Tt"]:.1f} MAE_USE={r["MAE_USE"]:.1f} within30={r["within30"]:.2f} ({r["sec"]:.0f}s)', flush=True)
res = pd.DataFrame(out); res.to_csv(RES / 'supp_param_accuracy.csv', index=False)
print(res.groupby(['model', 'scenario'])[['MAE_Tt', 'bias_Tt', 'MAE_USE', 'bias_USE', 'within30']].mean().round(2).to_string())
print('fitted-parameter spread for reference: SD Tt %.1f °C, SD USE %.1f J' % (par.T0.std(), par.USE.std()))
print('done')
