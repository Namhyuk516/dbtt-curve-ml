"""Validate on the user's own cast alloys (MSN, MSNN, MSCM, MSCMN): train every model on the full literature dataset,
predict full DBTT curves and compare with measured Charpy energies.
Heat treatment of the cast alloys is an ASSUMPTION set in common.cast_alloys(); a sensitivity study over
alternative heat treatments is also produced."""
import sys, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, pickle
import common
from common import load, cast_alloys, CAST_MEAS, RES, metrics
from models import MODELS, COL

df = load(); cast = cast_alloys()
which = sys.argv[1:] or list(MODELS)
Tg = np.linspace(-120, 100, 111)

fitted = {}
for name in which:
    print('training', name, flush=True); fitted[name] = MODELS[name]().fit(df)
pickle.dump({k: v for k, v in fitted.items() if 'NN' not in k}, open(RES / 'fitted_tree_models.pkl', 'wb'))


def rows_for(alloy_row, T):
    r = pd.DataFrame([alloy_row.to_dict()] * len(T)); r['test_temp'] = T; return r


# ---------- 1. curves with the assumed heat treatment ----------
fig, axs = plt.subplots(2, 2, figsize=(12, 9)); axs = axs.ravel(); recs = []
for ax, (_, a) in zip(axs, cast.iterrows()):
    meas = CAST_MEAS[CAST_MEAS.Alloy == a.Alloy]
    for name, m in fitted.items():
        p = m.predict(rows_for(a, Tg)); ax.plot(Tg, p, c=COL[name], label=name)
        if hasattr(m, 'predict_members'):                       # NN ensembles: +-1 SD band of the 5 members as an uncertainty indication
            pe = m.predict_members(rows_for(a, Tg)); ax.fill_between(Tg, pe.mean(0) - pe.std(0), pe.mean(0) + pe.std(0), color=COL[name], alpha=.15, lw=0)
        pm = m.predict(rows_for(a, meas.test_temp.values))
        recs.append(dict(Alloy=a.Alloy, model=name, **metrics(meas.Impact.values, pm)))
        if hasattr(m, 'curve_params'):
            cp = m.curve_params(rows_for(a, [25.])).iloc[0]; recs[-1].update(cp.to_dict())
    ex = meas[~meas.approx]; ap = meas[meas.approx]
    ax.plot(ex.test_temp, ex.Impact, 'ko', ms=6, label='measured')
    if len(ap): ax.plot(ap.test_temp, ap.Impact, 'o', mfc='none', mec='k', ms=6, label='measured (digitised, approx.)')
    ax.set_title(f'{a.Alloy}: {a.C:.2f}C–{a.Mn:.2f}Mn–{a.Si:.2f}Si' + (f'–{a.Cr:.2f}Cr' if a.Cr else '') + (f'–{a.Mo:.2f}Mo' if a.Mo else '') + (f'–{a.Ni:.2f}Ni' if a.Ni else '')
                 + f'\ncast, {a.Q_temp:.0f} °C/{a.Q_time/60:.0f} h WQ + {a.T_temp:.0f} °C/{a.T_time/60:.0f} h temper (band: ±1 SD of the NN ensembles)', fontsize=9)
    ax.set_xlabel('Test temperature (°C)'); ax.set_ylabel('Charpy impact energy (J)'); ax.set_ylim(0, None); ax.grid(alpha=.3)
axs[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(RES / 'fig_cast_validation.png', dpi=150)
rec = pd.DataFrame(recs); rec.to_csv(RES / 'cast_validation.csv', index=False)
print(rec.round(2).to_string())
print('\nmean over 4 alloys:\n', rec.groupby('model')[['R2', 'MAE', 'RMSE']].mean().round(2))

# ---------- 2. heat-treatment sensitivity (what if the cast alloys were treated differently?) ----------
HT = {'900WQ 4h + 600T 4h (actual)': dict(Q_temp=900, Q_time=240, Q_type=200, T_temp=600, T_time=240, T_type=1),
      '900WQ 4h + 650T 4h': dict(Q_temp=900, Q_time=240, Q_type=200, T_temp=650, T_time=240, T_type=1),
      '900WQ 4h + 550T 4h': dict(Q_temp=900, Q_time=240, Q_type=200, T_temp=550, T_time=240, T_type=1),
      '900WQ 1h + 600T 1h': dict(Q_temp=900, Q_time=60, Q_type=200, T_temp=600, T_time=60, T_type=1),
      '900 normalised 4h + 600T 4h': dict(Q_temp=900, Q_time=240, Q_type=1, T_temp=600, T_time=240, T_type=1),
      '900WQ 4h as-quenched': dict(Q_temp=900, Q_time=240, Q_type=200, T_temp=0, T_time=0, T_type=0)}
sens = []
fig, axs = plt.subplots(2, 2, figsize=(12, 9)); axs = axs.ravel()
for ax, (_, a) in zip(axs, cast.iterrows()):
    meas = CAST_MEAS[CAST_MEAS.Alloy == a.Alloy]
    for j, (lab, ht) in enumerate(HT.items()):
        b = a.copy()
        for k, v in ht.items(): b[k] = v
        b = common.add_derived(pd.DataFrame([b.to_dict()])).iloc[0]
        for name in ['tanh-XGB', 'XGB-direct', 'XGB-direct-mono']:
            if name not in fitted: continue
            m = fitted[name]; pm = m.predict(rows_for(b, meas.test_temp.values))
            sens.append(dict(Alloy=a.Alloy, HT=lab, model=name, **metrics(meas.Impact.values, pm)))
            if name == 'tanh-XGB': ax.plot(Tg, m.predict(rows_for(b, Tg)), c=f'C{j}', label=lab)
    ax.plot(meas.test_temp, meas.Impact, 'ko', ms=5, label='measured'); ax.set_title(f'{a.Alloy}  (tanh-XGB, heat-treatment sensitivity)', fontsize=9)
    ax.set_xlabel('Test temperature (°C)'); ax.set_ylabel('CVN (J)'); ax.set_ylim(0, None); ax.grid(alpha=.3)
axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(RES / 'fig_cast_HT_sensitivity.png', dpi=150)
sens = pd.DataFrame(sens); sens.to_csv(RES / 'cast_HT_sensitivity.csv', index=False)
print('\nHT sensitivity (MAE, J):\n', sens.pivot_table(index='HT', columns=['model', 'Alloy'], values='MAE').round(1).to_string())
