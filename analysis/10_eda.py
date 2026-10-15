"""Exploratory data analysis: distributions, correlations, curve-parameter correlations, steel classes, coverage
around the cast alloys.  Outputs figures + CSV tables in results/eda_*."""
import sys, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from common import load, cast_alloys, COMP, PROC, DERIVED, ALLOY_FEATS, NICE, RES, TT

df = load(); cast = cast_alloys(); nice = lambda c: NICE.get(c, c)
U = df.drop_duplicates('cond').reset_index(drop=True)          # alloy+heat-treatment conditions
A = df.drop_duplicates('alloy_id').reset_index(drop=True)      # alloys (unique compositions)
out = open(RES / 'eda_summary.txt', 'w', encoding='utf-8')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n')

# ------------------------------------------------------------------ 1. steel classes
def steel_class(r):
    if r.Cr >= 7: return '9-12Cr (heat-resistant)'
    if r.Mn >= 3: return 'Medium-Mn'
    if r.C >= 0.45: return 'High-C (spring/medium-C)'
    tot = r[COMP].sum()
    return 'Plain C / HSLA (alloy<2.5)' if tot < 2.5 else 'Low-alloy (2.5-7)'
df['cls'] = df.apply(steel_class, axis=1); U['cls'] = U.apply(steel_class, axis=1); A['cls'] = A.apply(steel_class, axis=1)
cls_tab = df.groupby('cls').agg(points=('Impact', 'size'), conditions=('cond', 'nunique'), alloys=('alloy_id', 'nunique'),
                                Impact_med=('Impact', 'median'), Impact_max=('Impact', 'max'), Tmin=('test_temp', 'min'), Tmax=('test_temp', 'max')).round(1)
P('=== 1. steel classes ===\n', cls_tab.to_string(), '\n')
cls_tab.to_csv(RES / 'eda_steel_classes.csv')

# ------------------------------------------------------------------ 2. distributions
fig, axs = plt.subplots(3, 6, figsize=(20, 10)); axs = axs.ravel()
cols = ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Cu', 'Nb', 'Ti', 'P', 'S', 'Q_temp', 'T_temp', 'T_time', 'test_temp', 'Impact', 'HJ_temper']
for ax, c in zip(axs, cols):
    src = df if c in ('test_temp', 'Impact') else U
    ax.hist(src[c], bins=30, color='C0', alpha=.8); ax.set_title(nice(c) + ('  (per point)' if src is df else '  (per condition)'), fontsize=9)
    if c in cast:
        for v in cast[c]: ax.axvline(v, c='r', lw=1)
plt.suptitle('Feature distributions (red lines = cast alloys)', fontsize=11); plt.tight_layout(); plt.savefig(RES / 'eda_fig_distributions.png', dpi=130); plt.close()

# ------------------------------------------------------------------ 3. correlation matrices (per condition, alloy+process features; Impact at point level)
feat = [c for c in COMP if U[c].std() > 0 and (U[c] > 0).mean() > 0.05] + PROC + ['HJ_temper', 'CE_IIW', 'Ms_Andrews']
M = U[feat].copy()
sp = M.corr(method='spearman'); pe = M.corr(method='pearson')
sp.round(3).to_csv(RES / 'eda_corr_spearman_features.csv')
fig, ax = plt.subplots(figsize=(13, 11)); im = ax.imshow(sp.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(feat))); ax.set_xticklabels([nice(c) for c in feat], rotation=90, fontsize=8); ax.set_yticks(range(len(feat))); ax.set_yticklabels([nice(c) for c in feat], fontsize=8)
for i in range(len(feat)):
    for j in range(len(feat)):
        if abs(sp.values[i, j]) >= 0.4 and i != j: ax.text(j, i, f'{sp.values[i,j]:.2f}', ha='center', va='center', fontsize=6)
plt.colorbar(im, fraction=.03); ax.set_title(f'Spearman correlation between alloy/process features ({len(U)} conditions)', fontsize=10)
plt.tight_layout(); plt.savefig(RES / 'eda_fig_corr_features.png', dpi=140); plt.close()
pairs = [(feat[i], feat[j], sp.values[i, j]) for i in range(len(feat)) for j in range(i + 1, len(feat)) if abs(sp.values[i, j]) >= 0.5]
P('=== 3. strongly correlated feature pairs (|Spearman| >= 0.5, per condition) ===')
for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])): P(f'  {a:20s} {b:20s} rho={r:+.2f}')
P()

# correlation of every feature with Impact (point level) and partial: at fixed test-temperature band
P(f'=== 3b. Spearman correlation of features with Impact (all {len(df)} points) and within test-temperature bands ===')
bands = {'T<=-60': df.test_temp <= -60, '-60<T<=0': (df.test_temp > -60) & (df.test_temp <= 0), '0<T<=40': (df.test_temp > 0) & (df.test_temp <= 40), 'T>40': df.test_temp > 40}
rows = []
for c in feat + ['test_temp']:
    rec = dict(feature=c, all=stats.spearmanr(df[c], df.Impact).correlation)
    for b, m in bands.items():
        rec[b] = stats.spearmanr(df.loc[m, c], df.loc[m, 'Impact']).correlation if df.loc[m, c].std() > 0 else np.nan
    rows.append(rec)
ci = pd.DataFrame(rows).set_index('feature'); ci.round(2).to_csv(RES / 'eda_corr_with_impact.csv')
P(ci.round(2).sort_values('all').to_string(), '\n')

# ------------------------------------------------------------------ 4. curve parameters vs features (59 fitted curves)
par = pd.read_csv(RES / 'curve_params.csv'); par = par[par.R2 > 0.8]
par = par.merge(U[['cond', 'cls'] + [c for c in ['HJ_temper', 'CE_IIW', 'Ms_Andrews', 'Pcm'] if c not in par.columns]], on='cond')
par['dE'] = par.USE - par.LSE
P(f'=== 4. curve parameters (n={len(par)} fitted curves, R2>0.8) ===')
P(par[['LSE', 'USE', 'T0', 'Cw']].describe().round(1).to_string(), '\n')
P('Correlation among curve parameters (Spearman):\n', par[['LSE', 'USE', 'T0', 'Cw']].corr(method='spearman').round(2).to_string(), '\n')
rows = []
for c in feat + ['Pcm']:
    if par[c].std() == 0: continue
    for p in ['USE', 'T0', 'Cw', 'LSE']:
        r, pv = stats.spearmanr(par[c], par[p]); rows.append(dict(feature=c, param=p, rho=r, p=pv))
cp = pd.DataFrame(rows); cp.to_csv(RES / 'eda_corr_curveparams.csv', index=False)
piv = cp.pivot(index='feature', columns='param', values='rho').round(2); pv = cp.pivot(index='feature', columns='param', values='p')
P('Spearman rho of features vs curve parameters (* p<0.05, ** p<0.01):')
for f in piv.sort_values('T0').index:
    P(f'  {f:18s} ' + '  '.join(f'{p}={piv.loc[f,p]:+.2f}{"**" if pv.loc[f,p]<0.01 else "*" if pv.loc[f,p]<0.05 else "  "}' for p in ['T0', 'USE', 'Cw', 'LSE']))
P()
fig, axs = plt.subplots(2, 4, figsize=(17, 8))
for ax, c in zip(axs[0], ['C', 'HJ_temper', 'Ms_Andrews', 'Ni']):
    for k, g in par.groupby('cls'): ax.scatter(g[c], g.T0, s=25, label=k)
    ax.set_xlabel(nice(c)); ax.set_ylabel(TT + ' (°C)'); ax.grid(alpha=.3)
    r, pv_ = stats.spearmanr(par[c], par.T0); ax.set_title(f'rho={r:+.2f}, p={pv_:.3f}', fontsize=9)
for ax, c in zip(axs[1], ['C', 'HJ_temper', 'Ms_Andrews', 'Ni']):
    for k, g in par.groupby('cls'): ax.scatter(g[c], g.USE, s=25, label=k)
    ax.set_xlabel(nice(c)); ax.set_ylabel('USE (J)'); ax.grid(alpha=.3)
    r, pv_ = stats.spearmanr(par[c], par.USE); ax.set_title(f'rho={r:+.2f}, p={pv_:.3f}', fontsize=9)
axs[0, 0].legend(fontsize=7); plt.tight_layout(); plt.savefig(RES / 'eda_fig_curveparams_vs_features.png', dpi=140); plt.close()

# ------------------------------------------------------------------ 5. tempering effect within alloys (same alloy, different T_temp)
P('=== 5. within-alloy tempering effect (alloys with >=3 tempering conditions; Spearman of T_temp vs mean Impact at the common test temperature) ===')
rows = []
for al, g in df.groupby('Alloy'):
    if g.T_temp.nunique() < 3: continue
    Tc = g.test_temp.round(0).value_counts().idxmax(); h = g[g.test_temp.round(0) == Tc]
    m = h.groupby('T_temp').Impact.mean()
    if len(m) >= 3:
        r, pv_ = stats.spearmanr(m.index, m.values); rows.append(dict(Alloy=al, test_T=Tc, n_temper=len(m), rho=r, p=pv_, Impact_range=f'{m.min():.0f}-{m.max():.0f}'))
tw = pd.DataFrame(rows); P(tw.round(2).to_string(index=False), '\n'); tw.to_csv(RES / 'eda_within_alloy_tempering.csv', index=False)

# ------------------------------------------------------------------ 6. coverage: PCA of alloy space + nearest neighbours of cast alloys
Xf = A[COMP].values; sc = StandardScaler().fit(np.log1p(Xf)); Z = sc.transform(np.log1p(Xf)); pca = PCA(2).fit(Z); ZA = pca.transform(Z)
ZC = pca.transform(sc.transform(np.log1p(cast[COMP].values)))
fig, ax = plt.subplots(figsize=(9, 7))
for k, g in A.assign(z1=ZA[:, 0], z2=ZA[:, 1]).groupby('cls'): ax.scatter(g.z1, g.z2, s=30, label=k, alpha=.8)
ax.scatter(ZC[:, 0], ZC[:, 1], marker='*', s=250, c='k', label='cast alloys (this work)')
for i, n in enumerate(cast.Alloy): ax.annotate(n, ZC[i], fontsize=8, xytext=(4, 4), textcoords='offset points')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)'); ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)'); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title(f'Composition space of the {len(A)} literature alloys (PCA of log(1+wt%)) and the cast alloys', fontsize=10)
plt.tight_layout(); plt.savefig(RES / 'eda_fig_pca_alloys.png', dpi=140); plt.close()
P('=== 6. nearest literature alloys (standardised log-composition distance) to each cast alloy ===')
for i, n in enumerate(cast.Alloy):
    zc = sc.transform(np.log1p(cast[COMP].values[[i]])); d = np.sqrt(((Z - zc) ** 2).sum(1)); idx = np.argsort(d)[:4]
    P(f'  {n}: ' + '; '.join(f'{A.Alloy[j]} (C{A.C[j]:.2f} Mn{A.Mn[j]:.2f} Cr{A.Cr[j]:.2f} Ni{A.Ni[j]:.2f} Mo{A.Mo[j]:.2f}, form={A.Rolled_Forged_Cast[j]}, d={d[j]:.2f})' for j in idx))
P()
# ------------------------------------------------------------------ 7. test-temperature coverage and impact by class
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
order = cls_tab.index.tolist()
for k in order:
    g = df[df.cls == k]; axs[0].scatter(g.test_temp, g.Impact, s=10, alpha=.6, label=f'{k} (n={len(g)})')
axs[0].set_xlabel('Test T (°C)'); axs[0].set_ylabel('CVN (J)'); axs[0].legend(fontsize=7); axs[0].grid(alpha=.3); axs[0].set_title('All data points by steel class', fontsize=10)
axs[1].boxplot([df[df.cls == k].Impact for k in order], tick_labels=[k.split(' (')[0] for k in order]); axs[1].set_ylabel('CVN (J)'); axs[1].tick_params(axis='x', labelsize=7); axs[1].set_title('Impact energy by steel class', fontsize=10)
plt.tight_layout(); plt.savefig(RES / 'eda_fig_by_class.png', dpi=140); plt.close()
# points per condition
nT = df.groupby('cond').test_temp.nunique()
P('=== 7. test temperatures per condition ===', f'1 temp: {(nT==1).sum()} conditions ({df.cond.map(nT).eq(1).sum()} points), 2-3: {((nT>=2)&(nT<=3)).sum()}, >=4: {(nT>=4).sum()} conditions ({df.cond.map(nT).ge(4).sum()} points)')
P('Impact replicate scatter (same cond & same test temp, n>=2): median |diff| =', round(df.groupby(['cond', 'test_temp']).Impact.agg(lambda s: s.max() - s.min() if len(s) > 1 else np.nan).median(), 1), 'J')
out.close(); print('done')
