"""Composite SHAP figure for the manuscript (a) XGB-direct energy, (b) tanh-XGB T0, (c) tanh-XGB USE."""
import sys, warnings; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent)); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, shap, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import load, FEATS, ALLOY_FEATS, NICE, RES, TT
from models import XGBDirect
plt.rcParams.update({'font.size': 15})
df = load(); nice = lambda cols: [NICE.get(c, c) for c in cols]
m = XGBDirect().fit(df); sv = shap.TreeExplainer(m.m).shap_values(df[FEATS].values)
svp = np.load(RES / 'shap_tanh_xgb_params.npy'); U = df.drop_duplicates('cond').reset_index(drop=True)
panels = [('(a) XGB-direct: impact energy (J)', sv, pd.DataFrame(df[FEATS].values, columns=nice(FEATS))),
          ('(b) tanh-XGB: ' + TT + ' (°C)', svp[2], pd.DataFrame(U[ALLOY_FEATS].values, columns=nice(ALLOY_FEATS))),
          ('(c) tanh-XGB: USE (J)', svp[1], pd.DataFrame(U[ALLOY_FEATS].values, columns=nice(ALLOY_FEATS)))]
fig = plt.figure(figsize=(21, 7.5))
for k, (title, s, X) in enumerate(panels):
    ax = fig.add_subplot(1, 3, k + 1); plt.sca(ax)
    shap.summary_plot(s, X, show=False, max_display=12, plot_size=None, color_bar=(k == 2))
    ax.set_title(title, fontsize=16, loc='left'); ax.set_xlabel('SHAP value', fontsize=14); ax.tick_params(labelsize=13)
plt.tight_layout(); plt.savefig(RES / 'fig_shap_composite.png', dpi=150); print('ok')
