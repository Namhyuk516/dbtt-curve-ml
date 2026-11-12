"""Re-draws the manuscript figures with larger lettering.  The figure scripts are executed unchanged; matplotlib is patched so that
every figure is drawn on a smaller canvas (same pixel width), which enlarges all text, symbols and lines relative to the panel, and a
minimum font size is enforced so that the smallest lettering is 7 pt at the printed width of 13.9 cm (text column).  Titles that become wider
than their panel are wrapped.  The numerical output of the scripts is unchanged.
usage: python 23_figures_large_text.py [script names...]      (run with XGBoost 3.2.0)"""
import sys, runpy, types, warnings, pathlib
warnings.filterwarnings('ignore')
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.figure as mfig, matplotlib.text as mtext
from matplotlib.colors import LinearSegmentedColormap

PRINT_W = 5.46            # printed figure width (in): width of the text column of the journal template
MIN_PT = 7.0              # smallest lettering at the printed width (pt)
PX_W = 2600               # pixel width of the saved figures
SCALE = {'10_eda.py': 0.40, '01_fit_curves.py': 0.55, '06_figures.py': 0.40, '12_shap_composite.py': 0.48, '04_cast.py': 0.50,
         '08_finetune.py': 0.60, '11_final_vs_heldout.py': 0.40}
ARGS = {'08_finetune.py': ['1']}
K = [1.0]

_figure, _savefig = plt.figure, mfig.Figure.savefig   # plt.subplots calls plt.figure
def figure(*a, **kw):
    if 'figsize' in kw: kw['figsize'] = tuple(K[0] * np.array(kw['figsize'], float))
    return _figure(*a, **kw)
def savefig(self, fname, *a, **kw):
    w = self.get_figwidth(); fmin = MIN_PT * w / PRINT_W
    for t in self.findobj(mtext.Text):
        if t.get_text() and t.get_fontsize() < fmin: t.set_fontsize(fmin)
    for ax in self.axes:                                   # tick labels are created lazily
        ax.tick_params(labelsize=max(fmin, min(l.get_fontsize() for l in ax.get_xticklabels() + ax.get_yticklabels()) if ax.get_xticklabels() else fmin))
    self.canvas.draw(); r = self.canvas.get_renderer()
    for ax in self.axes:                                   # wrap titles wider than the panel (at most two extra line breaks)
        for t in (ax.title, ax._left_title):
            for sep, rep in (('  (', '\n('), (' (', '\n('), (': ', ':\n'), (' | ', '\n'), (', ', ',\n'), (' [', '\n[')):
                s_ = t.get_text()
                if not s_ or max(t.get_window_extent(r).width, 0) <= 1.3 * ax.get_window_extent(r).width: break
                if sep in s_.split('\n')[0] or sep in s_.split('\n')[-1]:
                    k = s_.rfind(sep) if sep == ' [' else s_.find(sep)
                    t.set_text(s_[:k] + rep + s_[k + len(sep):]); self.canvas.draw()
    rect = getattr(self, '_tl_rect', None)
    if self.legends:                                       # figure legend below the panels: reserve exactly its height
        for t in self.legends[0].get_texts(): t.set_fontsize(max(fmin, t.get_fontsize()))
        self.canvas.draw(); rect = (0, self.legends[0].get_window_extent(self.canvas.get_renderer()).height / (self.get_figheight() * self.dpi) + 0.012, 1, 1)
    try: self.tight_layout(rect=rect)
    except Exception: pass
    kw['dpi'] = PX_W / w
    return _savefig(self, fname, *a, **kw)
plt.figure, mfig.Figure.savefig = figure, savefig


# ---- minimal stand-in for the two shap functions used by 12_shap_composite.py (the shap package needs numba, which may be unavailable);
#      TreeExplainer values are the exact TreeSHAP values computed by XGBoost itself (pred_contribs), identical to shap.TreeExplainer.
class _TreeExplainer:
    def __init__(self, model): self.m = model
    def shap_values(self, X):
        import xgboost as xgb
        return self.m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]

_RB = LinearSegmentedColormap.from_list('red_blue', ['#008bfb', '#8f37bb', '#ff0052'])
def _summary_plot(sv, X, show=False, max_display=12, plot_size=None, color_bar=True):
    ax = plt.gca(); names = list(X.columns); V = X.values; rng = np.random.RandomState(0)
    order = np.argsort(np.abs(sv).sum(0))[::-1][:max_display][::-1]
    for row, j in enumerate(order):
        s = sv[:, j]; v = V[:, j].astype(float); idx = rng.permutation(len(s)); s, v = s[idx], v[idx]
        nb = 100; q = np.round(nb * (s - s.min()) / (s.max() - s.min() + 1e-8)); layer = 0; last = -1; ys = np.zeros(len(s))
        for i in np.argsort(q + rng.randn(len(s)) * 1e-6):
            if q[i] != last: layer = 0
            ys[i] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1); layer += 1; last = q[i]
        ys *= 0.9 * (0.4 / max(np.max(ys + 1), 1e-9)) if np.max(np.abs(ys)) > 0 else 0
        lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
        if lo == hi: lo, hi = np.nanmin(v), np.nanmax(v)
        c = np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo else np.full(len(v), 0.5)
        ax.scatter(s, row + ys, c=c, cmap=_RB, vmin=0, vmax=1, s=9, alpha=1, linewidths=0, zorder=3, rasterized=len(s) > 500)
    ax.axvline(0, color='#999999', lw=0.6, zorder=1)
    for row in range(len(order)): ax.axhline(row, color='#cccccc', lw=0.5, dashes=(1, 5), zorder=0)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[j] for j in order]); ax.set_ylim(-1, len(order))
    for sp in ('top', 'right', 'left'): ax.spines[sp].set_visible(False)
    ax.tick_params(axis='y', length=0)
    if color_bar:
        import matplotlib.cm as cm
        cb = plt.colorbar(cm.ScalarMappable(cmap=_RB), ax=ax, ticks=[0, 1], aspect=60); cb.set_ticklabels(['Low', 'High']); cb.set_label('Feature value', labelpad=-12); cb.outline.set_visible(False)
try:
    import shap                                             # use the real package when it can be imported
except Exception:
    sys.modules['shap'] = types.SimpleNamespace(TreeExplainer=_TreeExplainer, summary_plot=_summary_plot)

try:
    from sklearn.decomposition import PCA                   # only used for a diagnostic figure of 10_eda.py; numpy stand-in if unavailable
except Exception:
    class _PCA:
        def __init__(self, n): self.n = n
        def fit(self, Z):
            self.mu = Z.mean(0); u, s, vt = np.linalg.svd(Z - self.mu, full_matrices=False)
            self.components_ = vt[:self.n]; self.explained_variance_ratio_ = (s ** 2 / (s ** 2).sum())[:self.n]; return self
        def transform(self, Z): return (Z - self.mu) @ self.components_.T
    sys.modules['sklearn.decomposition'] = types.SimpleNamespace(PCA=_PCA)

for name in (sys.argv[1:] or list(SCALE)):
    K[0] = SCALE[name]; print('=====', name, 'canvas scale', K[0], flush=True)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault); matplotlib.use('Agg')        # a script may change rcParams
    argv = sys.argv; sys.argv = [str(HERE / name)] + ARGS.get(name, [])
    try: runpy.run_path(str(HERE / name), run_name='__main__')
    finally: sys.argv = argv; plt.close('all')
print('figures done')
