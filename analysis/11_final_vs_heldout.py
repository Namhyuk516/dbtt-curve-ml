"""Appendix figure: for the nine steels of Figure 5, compare curves predicted with the steel held out vs the final models."""
import sys, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent)); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import load, RES, metrics, alloy_label
from models import MODELS
df = load()
examples = ['Carbon_steel', 'G18CrMo2-6_steel', 'AISI1018', 'DP590', 'P92', 'QT_steel', 'Cluster_HSLA_steel', '3Mn-Si-Ni', 'LC_martensitic_steel']
which = ['XGB-direct', 'tanh-XGB', 'tanh-NN']; COL = {'XGB-direct': 'C0', 'tanh-XGB': 'C3', 'tanh-NN': 'C1'}
final = {m: MODELS[m]().fit(df) for m in which}
Tg = np.linspace(-200, 250, 91); rows = []
fig, axs = plt.subplots(3, 3, figsize=(15, 11)); axs = axs.ravel()
for ax, al in zip(axs, examples):
    te = df[df.Alloy == al]; aid = te.alloy_id.iloc[0]; cond = te.groupby('cond').size().idxmax(); g = te[te.cond == cond]
    row = g.iloc[[0] * len(Tg)].copy(); row['test_temp'] = Tg
    for m in which:
        held = MODELS[m]().fit(df[df.alloy_id != aid])
        ax.plot(Tg, held.predict(row), c=COL[m], ls='--', lw=1.2, label=f'{m} (alloy held out)')
        ax.plot(Tg, final[m].predict(row), c=COL[m], lw=2, label=f'{m} (final, all data)')
        rows.append(dict(Alloy=al, label=alloy_label(df, al), model=m, n_points=len(g), MAE_heldout=metrics(g.Impact, held.predict(g))['MAE'], MAE_final=metrics(g.Impact, final[m].predict(g))['MAE']))
    ax.plot(g.test_temp, g.Impact, 'ko', ms=5, label='measured')
    ax.set_title(f'{alloy_label(df, al)} (temper {g.T_temp.iloc[0]:.0f}°C)', fontsize=10); ax.set_xlabel('Test T (°C)'); ax.set_ylabel('CVN (J)'); ax.set_ylim(0, None); ax.grid(alpha=.3)
axs[0].legend(fontsize=6.5); plt.tight_layout(); plt.savefig(RES / 'fig_final_vs_heldout.png', dpi=140)
r = pd.DataFrame(rows); r.to_csv(RES / 'final_vs_heldout_examples.csv', index=False)
print(r.pivot(index='Alloy', columns='model', values=['MAE_heldout', 'MAE_final']).round(1).to_string())
print('\nin-sample (all points):')
for m in which: print(m, {k: round(v, 2) for k, v in metrics(df.Impact, final[m].predict(df)).items()})
