"""SHAP analysis.
(a) XGB-direct  : TreeExplainer on impact energy (all rows).
(b) tanh-XGB    : KernelExplainer on the curve parameters (USE, T0, LSE, Cw) as functions of alloy + heat treatment,
                  evaluated on the unique (alloy, heat-treatment) conditions."""
import sys, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, shap, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import load, FEATS, ALLOY_FEATS, NICE, RES, TT
from models import XGBDirect, TanhXGB

df = load(); nice = lambda cols: [NICE.get(c, c) for c in cols]

# ---------------- (a) XGB-direct ----------------
m = XGBDirect().fit(df)
ex = shap.TreeExplainer(m.m); sv = ex.shap_values(df[FEATS].values)
Xd = pd.DataFrame(df[FEATS].values, columns=nice(FEATS))
plt.figure(); shap.summary_plot(sv, Xd, show=False, max_display=15); plt.title('XGB-direct: SHAP for Charpy energy (J)', fontsize=10)
plt.tight_layout(); plt.savefig(RES / 'fig_shap_xgb_direct_beeswarm.png', dpi=150, bbox_inches='tight'); plt.close()
imp = pd.Series(np.abs(sv).mean(0), index=FEATS).sort_values(ascending=False); imp.to_csv(RES / 'shap_importance_xgb_direct.csv')
print('XGB-direct mean|SHAP| (J):\n', imp.round(2).head(15).to_string())
fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, f in zip(axs, ['test_temp', 'C', 'Ni', 'T_temp']):
    j = FEATS.index(f); ax.scatter(df[f], sv[:, j], s=8, c=df.test_temp, cmap='coolwarm'); ax.set_xlabel(NICE.get(f, f)); ax.set_ylabel('SHAP (J)'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(RES / 'fig_shap_xgb_direct_dependence.png', dpi=150); plt.close()

# ---------------- (b) tanh-XGB curve parameters ----------------
tm = TanhXGB().fit(df)
U = df.drop_duplicates('cond').reset_index(drop=True)   # 214 unique conditions
XU = U[ALLOY_FEATS].values
def f_params(X):
    d = pd.DataFrame(X, columns=ALLOY_FEATS); d['test_temp'] = 25.
    return tm.curve_params(d).values                     # n x 4 (LSE, USE, T0, Cw)
bg = shap.kmeans(XU, 25)
kex = shap.KernelExplainer(f_params, bg)
svp = kex.shap_values(XU, nsamples=400, silent=True)    # list of 4 arrays (n x feats)  or array n x feats x 4
if isinstance(svp, np.ndarray) and svp.ndim == 3: svp = [svp[:, :, k] for k in range(4)]
XUd = pd.DataFrame(XU, columns=nice(ALLOY_FEATS)); P = f_params(XU)
labels = ['LSE (J)', 'USE (J)', TT + ' (°C)', 'Cw (°C)']
np.save(RES / 'shap_tanh_xgb_params.npy', np.array(svp)); U[['cond', 'Alloy'] + ALLOY_FEATS].assign(LSE=P[:, 0], USE=P[:, 1], T0=P[:, 2], Cw=P[:, 3]).to_csv(RES / 'tanh_xgb_params_all_conditions.csv', index=False)
for k in [1, 2]:
    plt.figure(); shap.summary_plot(svp[k], XUd, show=False, max_display=15); plt.title(f'tanh-XGB: SHAP for {labels[k]}', fontsize=10)
    plt.tight_layout(); plt.savefig(RES / f'fig_shap_tanh_xgb_{["LSE","USE","T0","Cw"][k]}.png', dpi=150, bbox_inches='tight'); plt.close()
imp2 = pd.DataFrame({labels[k]: np.abs(svp[k]).mean(0) for k in range(4)}, index=ALLOY_FEATS)
imp2.to_csv(RES / 'shap_importance_tanh_xgb_params.csv')
print('\ntanh-XGB mean|SHAP| per curve parameter:\n', imp2.round(2).sort_values(labels[2], ascending=False).head(15).to_string())
# dependence for T0 and USE vs key elements
fig, axs = plt.subplots(2, 4, figsize=(16, 7))
for r, k in enumerate([2, 1]):
    for ax, f in zip(axs[r], ['C', 'Ni', 'Mn', 'T_temp']):
        j = ALLOY_FEATS.index(f); ax.scatter(U[f], svp[k][:, j], s=12, c='C3' if k == 2 else 'C0'); ax.set_xlabel(NICE.get(f, f)); ax.set_ylabel(f'SHAP on {labels[k]}'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(RES / 'fig_shap_tanh_xgb_dependence.png', dpi=150); plt.close()
print('done')
