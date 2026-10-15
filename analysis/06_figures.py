"""Summary figures/tables from the CV runs: metrics table, parity plots, example held-out curves (smoothness), NMI bars."""
import sys, warnings, glob; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from common import load, RES, metrics, alloy_label
from models import MODELS, ORDER, COL

df = load(); y = df.Impact.values

# ---- 1. metrics table ----
cv = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / 'cv_*.csv')) if 'xgb_direct' not in f]).drop_duplicates(['model', 'scenario', 'rep'], keep='last')
tab = cv.groupby(['scenario', 'model'])[['R2', 'MAE', 'RMSE', 'NMI', 'tanhR2']].agg(['mean', 'std'])
tab.columns = [f'{a}_{b}' for a, b in tab.columns]
tab = tab.reset_index(); tab['model'] = pd.Categorical(tab.model, ORDER); tab = tab.sort_values(['scenario', 'model'])
tab.to_csv(RES / 'table_cv_summary.csv', index=False)
print(tab.round(3).to_string(index=False))

# ---- 2. parity plots (oof averaged over reps) ----
oof = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / 'oof_*.csv')) if 'xgb_direct' not in f], axis=1)
oof = oof.loc[:, ~oof.columns.duplicated(keep='last')]
scen = ['random', 'unseen-condition', 'unseen-alloy']
fig, axs = plt.subplots(3, len(ORDER), figsize=(3.4 * len(ORDER), 10))
for i, s in enumerate(scen):
    for j, m in enumerate(ORDER):
        ax = axs[i, j]; c = f'{m}|{s}'
        if c not in oof: ax.axis('off'); continue
        p = oof[c].values; mt = metrics(y, p)
        ax.scatter(y, p, s=6, alpha=.5, c=COL[m]); ax.plot([0, 370], [0, 370], 'k--', lw=.8)
        ax.set_xlim(0, 370); ax.set_ylim(0, 370); ax.set_title(f'{m} | {s}\nR²={mt["R2"]:.2f}  MAE={mt["MAE"]:.1f} J', fontsize=9)
        if i == 2: ax.set_xlabel('measured CVN (J)')
        if j == 0: ax.set_ylabel('predicted CVN (J)')
plt.tight_layout(); plt.savefig(RES / 'fig_parity_all.png', dpi=140); plt.close()

# ---- 3. example held-out curves: train with the alloy held out, predict full curve ----
examples = ['Carbon_steel', 'G18CrMo2-6_steel', 'AISI1018', 'DP590', 'P92', 'QT_steel', 'Cluster_HSLA_steel', '3Mn-Si-Ni', 'LC_martensitic_steel']
Tg = np.linspace(-200, 250, 91)
fig, axs = plt.subplots(3, 3, figsize=(15, 11)); axs = axs.ravel()
which = [m for m in ORDER if m != 'NN-direct']
for ax, al in zip(axs, examples):
    tr = df.Alloy != al; te = df[~tr]
    cond = te.groupby('cond').size().idxmax(); g = te[te.cond == cond]         # the condition with most points
    row = g.iloc[[0] * len(Tg)].copy(); row['test_temp'] = Tg
    for m in which:
        mod = MODELS[m]().fit(df[tr]); ax.plot(Tg, mod.predict(row), c=COL[m], label=m, lw=1.6)
    ax.plot(g.test_temp, g.Impact, 'ko', ms=5, label='measured (held-out alloy)')
    ax.set_title(f'{alloy_label(df, al)}  (temper {g.T_temp.iloc[0]:.0f}°C)', fontsize=10); ax.set_xlabel('Test T (°C)'); ax.set_ylabel('CVN (J)'); ax.set_ylim(0, None); ax.grid(alpha=.3)
axs[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(RES / 'fig_heldout_curves_unseen_alloy.png', dpi=140); plt.close()

# ---- 3b. same but 'unseen-condition' scenario (other tempering conditions of the alloy are in training) ----
fig, axs = plt.subplots(3, 3, figsize=(15, 11)); axs = axs.ravel()
ex2 = df.groupby('Alloy').cond.nunique(); ex2 = ex2[ex2 >= 2].index.tolist()[:9]
for ax, al in zip(axs, ex2):
    conds = df[df.Alloy == al].groupby('cond').size().sort_values().index; cond = conds[-1]
    tr = df.cond != cond; g = df[df.cond == cond]; row = g.iloc[[0] * len(Tg)].copy(); row['test_temp'] = Tg
    for m in which:
        mod = MODELS[m]().fit(df[tr]); ax.plot(Tg, mod.predict(row), c=COL[m], label=m, lw=1.6)
    ax.plot(g.test_temp, g.Impact, 'ko', ms=5, label='measured (held-out condition)')
    ax.set_title(f'{alloy_label(df, al)}  (temper {g.T_temp.iloc[0]:.0f}°C; {len(conds)-1} other conditions in training)', fontsize=9)
    ax.set_xlabel('Test T (°C)'); ax.set_ylabel('CVN (J)'); ax.set_ylim(0, None); ax.grid(alpha=.3)
axs[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(RES / 'fig_heldout_curves_unseen_condition.png', dpi=140); plt.close()

# ---- 4. bar chart: R2 by scenario + NMI ----
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, met, lab in zip(axs, ['R2', 'MAE', 'NMI'], ['R² (CVN, J)', 'MAE (J)', 'non-monotonicity index']):
    w = 0.8 / len(ORDER)
    for j, m in enumerate(ORDER):
        t = tab[tab.model == m].set_index('scenario').reindex(scen)
        ax.bar(np.arange(3) + (j - (len(ORDER) - 1) / 2) * w, t[f'{met}_mean'], w, yerr=t[f'{met}_std'], color=COL[m], label=m, capsize=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(scen); ax.set_ylabel(lab); ax.grid(axis='y', alpha=.3)
    if met == 'R2': ax.set_ylim(-0.5, 1); ax.axhline(0, c='k', lw=.6)
axs[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(RES / 'fig_cv_bars.png', dpi=150); plt.close()
print('figures done')
