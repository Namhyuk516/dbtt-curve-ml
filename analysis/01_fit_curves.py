"""Fit the tanh empirical DBTT equation to every (alloy, heat-treatment) condition with >=4 test temperatures.
Outputs: results/curve_params.csv and results/fig_curvefits.png"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from common import load, tanh_curve, RES, ALLOY_FEATS, alloy_label

df = load()
rows = []
for cid, g in df.groupby('cond'):
    if g.test_temp.nunique() < 4: continue
    T, E = g.test_temp.values, g.Impact.values
    lo, hi = E.min(), E.max()
    p0 = [max(lo,0.5), hi, T[np.argmin(np.abs(E-(lo+hi)/2))], 30]
    try:
        popt, _ = curve_fit(tanh_curve, T, E, p0=p0, bounds=([0, 0, -300, 3], [hi, 1.5*hi+5, 400, 300]), maxfev=20000)
        pred = tanh_curve(T, *popt); r2 = 1-((E-pred)**2).sum()/((E-E.mean())**2).sum()
    except Exception as e:
        popt, r2 = [np.nan]*4, np.nan
    rec = dict(cond=cid, Alloy=g.Alloy.iloc[0], n=len(g), nT=g.test_temp.nunique(), Tmin=T.min(), Tmax=T.max(),
               LSE=popt[0], USE=popt[1], T0=popt[2], Cw=popt[3], R2=r2)
    rec.update(g[ALLOY_FEATS].iloc[0].to_dict()); rows.append(rec)
par = pd.DataFrame(rows); par.to_csv(RES/'curve_params.csv', index=False)
print(par[['Alloy','n','nT','Tmin','Tmax','LSE','USE','T0','Cw','R2']].round(2).to_string())
print('\nmedian R2 =', par.R2.median().round(3), ' n curves =', len(par), ' R2>0.9:', (par.R2>0.9).sum())

# figure: 12 example fits
sel = par.sort_values('nT', ascending=False).head(12)
fig, axs = plt.subplots(3, 4, figsize=(14, 9)); axs = axs.ravel()
for ax, (_, r) in zip(axs, sel.iterrows()):
    g = df[df.cond == r.cond]; Tg = np.linspace(g.test_temp.min()-30, g.test_temp.max()+30, 200)
    ax.plot(g.test_temp, g.Impact, 'o', c='k', ms=4, label='data')
    ax.plot(Tg, tanh_curve(Tg, r.LSE, r.USE, r.T0, r.Cw), c='C3', label=f'tanh fit  R²={r.R2:.2f}')
    ax.axvline(r.T0, ls=':', c='gray'); ax.set_title(f'{alloy_label(df, r.Alloy)}  (T={r.T_temp:.0f}°C temper)', fontsize=9)
    ax.set_xlabel('Test T (°C)'); ax.set_ylabel('CVN (J)'); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(RES/'fig_curvefits.png', dpi=150)
