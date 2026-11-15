"""How well do eight specimens (two at each of four temperatures, -40 to 25 C) determine the transition temperature and the upper shelf of a cast steel?
Equation (1) is fitted to the eight measurements of each cast steel with the bounds of Section 2.2, and the uncertainty is estimated by a
bootstrap that respects the test design: at each temperature the two measured energies are resampled with replacement (2000 replicates).
The spread of the fitted mid-transition temperature Tt and upper-shelf energy USE shows why no such values are reported from the measurements alone.
usage: python 26_cast_tt_uncertainty.py"""
import sys, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent)); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from common import CAST_MEAS, RES, tanh_curve


def fit(T, E):
    lo, hi = E.min(), E.max()
    p0 = [min(max(lo, 0.5), lo), hi, T[np.argmin(np.abs(E - (lo + hi) / 2))], 30]
    return curve_fit(tanh_curve, T, E, p0=p0, bounds=([0, 0, -300, 3], [hi, 1.5 * hi + 5, 400, 300]), maxfev=20000)


rng = np.random.RandomState(0); rows = []
for al, g in CAST_MEAS.groupby('Alloy', sort=False):
    T = g.test_temp.values.astype(float); E = g.Impact.values
    popt, pcov = fit(T, E); se = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan] * 4
    temps = np.unique(T); B = []
    for b in range(2000):
        Eb = np.concatenate([rng.choice(E[T == t], size=2, replace=True) for t in temps]); Tb = np.repeat(temps, 2)
        try: B.append(fit(Tb, Eb)[0])
        except Exception: pass
    B = np.array(B); q = lambda k, p: np.percentile(B[:, k], p)
    hi = E.max()
    rows.append(dict(Alloy=al, LSE=popt[0], USE=popt[1], Tt=popt[2], Cw=popt[3], se_Tt=se[2], se_USE=se[1], singular=bool(not np.all(np.isfinite(pcov))),
                     Tt_lo=q(2, 2.5), Tt_hi=q(2, 97.5), USE_lo=q(1, 2.5), USE_hi=q(1, 97.5), USE_at_bound_frac=float((B[:, 1] >= 1.5 * hi + 5 - 1e-6).mean()),
                     E25_mean=E[T == 25].mean(), n_boot=len(B)))
    r = rows[-1]; print(f'{al:6s} fit: Tt={r["Tt"]:.1f} C, USE={r["USE"]:.0f} J (measured mean at 25 C: {r["E25_mean"]:.0f} J) | 95% bootstrap interval: Tt {r["Tt_lo"]:.0f} to {r["Tt_hi"]:.0f} C, USE {r["USE_lo"]:.0f} to {r["USE_hi"]:.0f} J', flush=True)
res = pd.DataFrame(rows); res['Tt_width'] = res.Tt_hi - res.Tt_lo; res['USE_width'] = res.USE_hi - res.USE_lo
res.to_csv(RES / 'cast_tt_uncertainty.csv', index=False)
print(res.round(1).to_string(index=False)); print('done')
