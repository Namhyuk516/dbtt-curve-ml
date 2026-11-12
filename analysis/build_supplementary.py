# -*- coding: utf-8 -*-
"""Build the Supplementary Materials (Tables S1-S13) from the result files in ../results.
Tables and table parts that are new in the revised version are highlighted (HL = True)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import docx
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parent; RES = ROOT.parent / 'results'
sys.path.insert(0, str(ROOT.parent / 'analysis'))
import importlib.util
spec = importlib.util.spec_from_file_location('content', ROOT / 'content.py'); C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
import re
TITLE = re.sub('<[^>]+>', '', C.TITLE)

d = docx.Document()
sec = d.sections[0]; sec.orientation = WD_ORIENT.LANDSCAPE; sec.page_width, sec.page_height = Cm(29.7), Cm(21.0)
for m in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'): setattr(sec, m, Cm(1.8))
st = d.styles['Normal']; st.font.name = 'Palatino Linotype'; st.font.size = Pt(10)


HL = 'clean' not in sys.argv          # highlight what is new in the revised version (python build_supplementary.py clean -> no highlighting)
from docx.enum.text import WD_COLOR_INDEX


def para(text, bold=False, size=10, align=None, after=6, new=False):
    p = d.add_paragraph(); p.paragraph_format.space_after = Pt(after)
    if align: p.alignment = align
    for k, seg in enumerate(re.split(r'\*\*', text)):
        for j, part in enumerate(re.split(r'<sup>|</sup>', seg)):
            if part:
                r = p.add_run(part); r.bold = bold or k % 2 == 1; r.font.size = Pt(size); r.font.superscript = j % 2 == 1
                if new and HL: r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return p


def rule(cell, edge, sz=8):
    tcPr = cell._tc.get_or_add_tcPr(); b = tcPr.find(qn('w:tcBorders'))
    if b is None: b = OxmlElement('w:tcBorders'); tcPr.append(b)
    e = OxmlElement(f'w:{edge}'); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), str(sz)); e.set(qn('w:space'), '0'); e.set(qn('w:color'), 'auto'); b.append(e)


def table(header, rows, widths, size=8, align_left=(0,), new=False, new_rows=()):
    t = d.add_table(rows=1 + len(rows), cols=len(header)); t.autofit = False
    for i, row in enumerate([header] + rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.width = Cm(widths[j]); p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in align_left else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(v)); r.font.size = Pt(size); r.bold = i == 0
            if HL and (new or (i - 1) in new_rows): r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            if i == 0: rule(c, 'top'); rule(c, 'bottom', 4)
            if i == len(rows): rule(c, 'bottom')
    d.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


f3 = lambda v: '–' if pd.isna(v) else f'{v:.2f}'.replace('-', '−')
f1 = lambda v: '–' if pd.isna(v) else f'{v:.1f}'.replace('-', '−')
SC = ['random', 'unseen-condition', 'unseen-alloy']

para('Supplementary Materials', bold=True, size=14, after=4)
para(TITLE, bold=True, size=11, after=8)
para(C.AUTHORS.replace('<sup>1,</sup>*', '<sup>1,</sup>*'), size=10, after=4)
for a_ in C.AFFILIATIONS: para(a_, size=8, after=2)
para('', after=6)

# ------------------------------------------------------------------ Table S1: data sources
from common import load
df = load(); src = pd.read_csv(RES / 'TableS1_sources.csv', encoding='utf-8-sig')
src['n'] = [int(df.No.between(a, b).sum()) for a, b in zip(src.row_start, src.row_end)]
g = src.groupby('doi', dropna=False, sort=False) if False else None
rows, seen = [], {}
for _, r in src.iterrows():
    key = r.doi if isinstance(r.doi, str) else f'{r.authors}{r.year}'
    if key in seen: rows[seen[key]][3] += r.n; continue
    vol = '' if pd.isna(r.volume) else f', {r.volume}'; pg = '' if pd.isna(r.pages) else f', {r.pages}'
    ref = f'{r.authors} {r.title}. {r.journal} {int(r.year)}{vol}{pg}.'
    seen[key] = len(rows); rows.append([0, ref, r.doi if isinstance(r.doi, str) else '–', r.n])
rows = [r for r in rows if r[3] > 0]
for i, r in enumerate(rows, 1): r[0] = i
assert sum(r[3] for r in rows) == len(df), (sum(r[3] for r in rows), len(df))
# specimen geometry per source (TableS1_specimens.csv, extracted from the methods sections)
spec_path = RES / 'TableS1_specimens.csv'
if spec_path.exists():
    sp = pd.read_csv(spec_path, encoding='utf-8-sig').set_index('row_start')
    def spec_for(ref):   # reference text -> specimen size and notch, matched through the source's authors + year
        for _, s in src.iterrows():
            if ref.startswith(str(s.authors)) and (str(int(s.year)) in ref):
                q = sp.loc[s.row_start]; raw = str(q.specimen_size); notch = str(q.notch)
                if raw.startswith('not stated'): size = 'not stated'
                elif 'implied' in raw or 'stated as' in raw: size = 'standard 10 × 10 × 55 mm (dimensions not printed)'
                elif raw.startswith('10 × 10 × 55'): size = '10 × 10 × 55 mm'
                else: size = raw.split(' (')[0].replace('half-size', 'sub-size')
                return size + ('; V-notch' if notch == 'V' else '; notch not stated')
        return '–'
    for r in rows: r.insert(3, spec_for(r[1]))
    para(f'**Table S1.** Sources of the literature database ({len(rows)} publications, {len(df)} Charpy measurements). "Specimen" gives the Charpy specimen size '
         'and notch as stated in the source; "standard … (dimensions not printed)" denotes sources that describe the specimens only as standard Charpy V-notch '
         'specimens, and "not stated" sources that give no information. The last column gives the number of measurements taken from each publication.')
    table(['No.', 'Reference', 'DOI', 'Specimen', 'Measurements'], rows, [0.9, 12.9, 4.6, 5.2, 2.5], size=7.5, align_left=(1, 2, 3))
else:
    para(f'**Table S1.** Sources of the literature database ({len(rows)} publications, {len(df)} Charpy measurements). '
         'The last column gives the number of measurements taken from each publication.')
    table(['No.', 'Reference', 'DOI', 'Measurements'], rows, [1.0, 16.6, 5.6, 2.9], size=7.5, align_left=(1, 2))
d.add_page_break()

MODELS6 = ['XGB-direct', 'XGB-direct-mono', 'tanh-XGB', 'two-stage', 'NN-direct', 'tanh-NN']
disp = lambda m: m.replace('two-stage', 'Two-stage')
cvs = pd.read_csv(RES / 'table_cv_summary.csv').set_index(['model', 'scenario'])
HDR = ['Random R²', 'Random MAE (J)', 'Unseen cond. R²', 'Unseen cond. MAE (J)', 'Unseen steel R²', 'Unseen steel MAE (J)']

# ------------------------------------------------------------------ Table S2: descriptor / tuning ablation
ab = pd.concat([pd.read_csv(RES / 'supp_ablation_tree.csv'), pd.read_csv(RES / 'supp_ablation_nn.csv')])
m = ab.groupby(['model', 'feats', 'tuned', 'scenario'])[['R2', 'MAE']].mean().unstack('scenario')
rows = []
for model in MODELS6:
    for feats in ['base', 'derived']:
        for tuned in ([False] if model == 'two-stage' else [False, True]):
            r = [disp(model), 'primary only (24)' if feats == 'base' else 'primary + physics-derived (32)', 'tuned' if tuned else 'default']
            for s in SC: r += [f3(m.loc[(model, feats, tuned), ('R2', s)]), f1(m.loc[(model, feats, tuned), ('MAE', s)])]
            rows.append(r)
para('**Table S2.** Effect of the physics-derived descriptors and of hyper-parameter tuning on the cross-validation results '
     '(five folds, mean of three repetitions; same protocol as Table 6 of the article). "Default" denotes the untuned settings of Section 2.4; '
     '"tuned" the Optuna-selected settings (same objective and budget for every model). The two-stage model was not tuned. '
     'The rows "primary + physics-derived, tuned" correspond to Table 6.')
table(['Model', 'Descriptors', 'Hyper-parameters'] + HDR, rows, [2.6, 5.0, 2.8, 2.2, 2.6, 2.6, 3.0, 2.4, 2.9])
d.add_page_break()

# ------------------------------------------------------------------ Table S3: errors by steel class (unseen steel)
ce = pd.read_csv(RES / 'supp_class_errors_unseen_steel.csv', encoding='utf-8-sig')
rows = [[r.cls, int(r.n), int(r.steels)] + [f1(r[f'{k}_MAE']) for k in MODELS6] for _, r in ce.iterrows()]
para('**Table S3.** Mean absolute error (J) under unseen-steel cross-validation by steel class (classes of Table 2 of the article), '
     'computed from the out-of-fold predictions averaged over the three repetitions (the values for all data therefore differ slightly from Table 6). The last row is the subset of the database that '
     'resembles the in-house cast steels (0.10–0.30 wt.% C, Cr < 7, Mn < 3, tempered at ≥ 550 °C).')
table(['Steel class', 'Measurements', 'Steels'] + [disp(k) for k in MODELS6], rows, [6.4, 2.4, 1.6, 2.4, 2.9, 2.4, 2.4, 2.4, 2.4])

# ------------------------------------------------------------------ Table S4: without the single-temperature conditions
mt = pd.read_csv(RES / 'supp_multi_temp.csv'); mm = mt.groupby(['model', 'scenario'])[['R2', 'MAE']].mean()
rows = []
for model in MODELS6:
    for lab, get in [(f'all conditions ({len(df)})', lambda s, k: cvs.loc[(model, s), k + '_mean']), (f'conditions with ≥ 2 test temperatures ({int(mt.n.iloc[0])})', lambda s, k: mm.loc[(model, s), k])]:
        r = [disp(model), lab]
        for s in SC: r += [f3(get(s, 'R2')), f1(get(s, 'MAE'))]
        rows.append(r)
nT = df.groupby('cond').test_temp.nunique(); n1 = int((nT == 1).sum())
para(f'**Table S4.** Cross-validation results after removing the {n1} conditions that were tested at a single temperature ({int(df.cond.map(nT).eq(1).sum())} measurements), '
     'with the same protocol and hyper-parameters as Table 6.')
table(['Model', 'Data'] + HDR, rows, [2.6, 5.6, 2.3, 2.7, 2.7, 3.1, 2.5, 3.0])
d.add_page_break()

# ------------------------------------------------------------------ Table S5: nested cross-validation (all tuned models)
cv0 = pd.read_csv(RES / 'cv_all.csv'); cv0 = cv0[cv0.rep == 0].set_index(['model', 'scenario'])
nc = pd.concat([pd.read_csv(f) for f in RES.glob('nested_cv_*.csv') if 'selected' not in f.name]).set_index(['model', 'scenario'])
rows = []
for model in ['XGB-direct', 'XGB-direct-mono', 'tanh-XGB', 'NN-direct', 'tanh-NN']:
    r = [model]
    for s_ in ['unseen-condition', 'unseen-alloy']:
        r += [f3(cv0.loc[(model, s_), 'R2']), f1(cv0.loc[(model, s_), 'MAE']), f3(nc.loc[(model, s_), 'R2']), f1(nc.loc[(model, s_), 'MAE']), f3(nc.loc[(model, s_), 'R2'] - cv0.loc[(model, s_), 'R2'])]
    rows.append(r)
para('**Table S5.** Nested cross-validation of the five tuned models. In the nested procedure the hyper-parameters are re-selected inside every outer fold, so that no '
     'information from the held-out fold enters the selection (tree models: Optuna, 20 trials on an inner five-fold grouped split of the training folds; neural networks: '
     '15 trials on an inner three-fold grouped split with ensembles of three networks, the selected configuration being refitted as an ensemble of five). Outer: five-fold '
     'grouped split, one repetition. "Global" denotes the hyper-parameters selected once on the whole database (Section 2.4), evaluated on the same fold assignment as the '
     'nested procedure (repetition 1 of Table 6; the three-repetition means of Table 6 are lower, e.g., 0.29–0.32 for the tree models under unseen-steel splitting). '
     'ΔR² = nested − global; negative values indicate an optimistic bias of the globally tuned estimate.', new=True)
table(['Model', 'Unseen cond. R² (global)', 'Unseen cond. MAE (global)', 'Unseen cond. R² (nested)', 'Unseen cond. MAE (nested)', 'Unseen cond. ΔR²',
       'Unseen steel R² (global)', 'Unseen steel MAE (global)', 'Unseen steel R² (nested)', 'Unseen steel MAE (nested)', 'Unseen steel ΔR²'], rows,
      [2.6, 2.3, 2.3, 2.3, 2.3, 2.0, 2.3, 2.3, 2.3, 2.3, 2.0], new=True)

# ------------------------------------------------------------------ Table S6: parameter-level accuracy of the equation-based models
pr = pd.read_csv(RES / 'supp_param_accuracy_percond.csv'); fu = pd.read_csv(RES / 'curve_fit_uncertainty_resolved.csv')
pr['eT'] = pr.Tt_pred - pr.Tt_fit; pr['eU'] = pr.USE_pred - pr.USE_fit
def _acc(g):
    q = g.groupby('rep').apply(lambda x: pd.Series(dict(MAE_Tt=x.eT.abs().mean(), bias_Tt=x.eT.mean(), MAE_USE=x.eU.abs().mean(), bias_USE=x.eU.mean(), within30=(x.eT.abs() <= 30).mean()))).mean()
    return [f1(q.MAE_Tt), f1(q.bias_Tt), f1(q.MAE_USE), f1(q.bias_USE), f'{100 * q.within30:.0f}%']
nres = int(fu.resolved.sum()); rows = []; newr = []
for model in ['two-stage', 'tanh-XGB', 'tanh-NN']:
    for s_ in ['unseen-condition', 'unseen-alloy']:
        g = pr[(pr.model == model) & (pr.scenario == s_)]; sl = s_.replace('unseen-alloy', 'unseen steel').replace('unseen-condition', 'unseen condition')
        rows.append([disp(model), sl, f'all ({len(fu)})'] + _acc(g))
        newr += [len(rows), len(rows) + 1]
        rows.append([disp(model), sl, f'well resolved ({nres})'] + _acc(g[g.resolved]))
        rows.append([disp(model), sl, f'poorly resolved ({len(fu) - nres})'] + _acc(g[~g.resolved]))
sd = lambda m, c: fu.loc[m, c].std()
para(f'**Table S6.** Accuracy of the curve parameters predicted by the equation-based models for the {len(fu)} conditions with complete curves, '
     'compared with the parameters fitted directly to the measurements (Equation (1)); grouped cross-validation, mean of three repetitions. '
     f'For reference, the fitted parameters have standard deviations of {fu.Tt.std():.0f} °C (Tt) and {fu.USE.std():.0f} J (USE) across the '
     f'{len(fu)} curves, and their median standard errors from the least-squares fit are {fu.se_Tt.median():.1f} °C and {fu.se_USE.median():.1f} J (fits with a singular covariance matrix counted as infinite). '
     '"Within 30 °C" is the fraction of conditions whose predicted Tt lies within ±30 °C of the fitted value.')
para(f'All {len(fu)} fitted curves are used as reference values in the rows "all", irrespective of the quality of the fit. A reference curve is classed as well resolved when the '
     'least-squares fit has a non-singular covariance matrix, standard errors of at most 20 °C in Tt and 20 J in USE, no USE or Cw at a bound, and Tt inside the tested '
     f'temperature range; {nres} curves meet these criteria (median of {fu[fu.resolved].nT.median():.0f} test temperatures over {fu[fu.resolved].Trange.median():.0f} °C; standard deviation of the fitted values '
     f'{sd(fu.resolved, "Tt"):.0f} °C and {sd(fu.resolved, "USE"):.0f} J) and {len(fu) - nres} do not (median of {fu[~fu.resolved].nT.median():.0f} test temperatures over {fu[~fu.resolved].Trange.median():.0f} °C; '
     f'{sd(~fu.resolved, "Tt"):.0f} °C and {sd(~fu.resolved, "USE"):.0f} J). For the poorly resolved curves the reference values themselves are uncertain, so the errors in those rows are indicative only.', new=True)
table(['Model', 'Scheme', 'Reference curves', 'MAE Tt (°C)', 'Bias Tt (°C)', 'MAE USE (J)', 'Bias USE (J)', 'Within 30 °C'], rows, [2.6, 3.2, 3.8, 2.4, 2.4, 2.4, 2.4, 2.4], new_rows=newr)
d.add_page_break()
# ------------------------------------------------------------------ Table S7: full-size-specimen sources only
fs_path = RES / 'supp_fullsize.csv'
if fs_path.exists():
    fs = pd.read_csv(fs_path); fm = fs.groupby(['model', 'scenario'])[['R2', 'MAE']].mean()
    rows = []
    for model in MODELS6:
        for lab, get in [(f'all sources ({len(df)})', lambda s, k: cvs.loc[(model, s), k + '_mean']), (f'full-size sources only ({int(fs.n.iloc[0])})', lambda s, k: fm.loc[(model, s), k])]:
            r = [disp(model), lab]
            for s in SC: r += [f3(get(s, 'R2')), f1(get(s, 'MAE'))]
            rows.append(r)
    para(f'**Table S7.** Sensitivity of the cross-validation results to the specimen geometry of the sources: results with the full database and after removing the '
         f'{len(df) - int(fs.n.iloc[0])} measurements from sources that used sub-size specimens (5 × 10, 7 × 10 or 2.5 × 10 mm) or did not state the specimen size '
         '(same protocol and hyper-parameters as Table 6).')
    table(['Model', 'Data'] + HDR, rows, [2.6, 5.6, 2.3, 2.7, 2.7, 3.1, 2.5, 3.0])
    d.add_page_break()

# ------------------------------------------------------------------ Table S8: sensitivity of the external test to the network configuration
cs_path = RES / 'supp_cast_config_sensitivity.csv'
if cs_path.exists():
    cs = pd.read_csv(cs_path)
    rows = [[r['model'], int(r['rank']), f"{int(r['h'])} / {r['drop']:.2f} / {r['lr']:.4f} / {r['wd']:.1e} / {int(r['epochs'])}", f3(r['objective']), f1(r['cast_MAE']), f3(r['cast_R2'])] for _, r in cs.iterrows()]
    pc = RES / 'cast_prev_config.csv'
    if pc.exists():
        q = pd.read_csv(pc).iloc[0]; rows.append(['tanh-NN', '–', f"{int(q['h'])} / {q['drop']:.2f} / {q['lr']:.4f} / {q['wd']:.1e} / {int(q['epochs'])}", f3(q['objective']), f1(q['cast_MAE']), f3(q['cast_R2'])])
    para('**Table S8.** Sensitivity of the external test to the network configuration. The five best Optuna trials of each network (ranked by the tuning objective, '
         'the mean R² of the unseen-condition and unseen-steel schemes on one fold assignment) were trained on the full literature database and evaluated on the 32 '
         'measurements of the four cast steels. The last row is a wider tanh-NN configuration selected by an earlier tuning run on a slightly different version of the '
         'database, re-evaluated on the present database.')
    table(['Model', 'Rank', 'Width / dropout / learning rate / weight decay / epochs', 'Objective', 'Cast MAE (J)', 'Cast R²'], rows, [2.4, 1.2, 8.0, 2.2, 2.6, 2.2], align_left=(2,))
    d.add_page_break()

# ------------------------------------------------------------------ Table S9: leave-publication-out cross-validation
pb = pd.read_csv(RES / 'supp_publication.csv'); pg_ = pb.groupby('model')[['R2', 'MAE', 'RMSE']].agg(['mean', 'std'])
rows = [[disp(m_), f3(cvs.loc[(m_, 'unseen-alloy'), 'R2_mean']), f1(cvs.loc[(m_, 'unseen-alloy'), 'MAE_mean']), f3(pg_.loc[m_, ('R2', 'mean')]), f3(pg_.loc[m_, ('R2', 'std')]),
         f1(pg_.loc[m_, ('MAE', 'mean')]), f1(pg_.loc[m_, ('RMSE', 'mean')])] for m_ in MODELS6]
para('**Table S9.** Cross-validation with whole publications held out (five folds grouped by source publication, mean of three repetitions; same hyper-parameters as '
     'Table 6), compared with the unseen-steel scheme of Table 6. No steel of the database occurs in more than one publication, but 10 of the 40 publications contribute '
     'two to six steels, so that under steel grouping other steels of the same publication (same laboratory, specimen geometry and testing practice) can be present in the '
     'training folds; holding out whole publications removes this possible source of leakage.', new=True)
table(['Model', 'Unseen steel R² (Table 6)', 'Unseen steel MAE (J) (Table 6)', 'Unseen publication R²', 'SD of R² over repetitions', 'Unseen publication MAE (J)', 'Unseen publication RMSE (J)'],
      rows, [2.8, 3.2, 3.6, 3.2, 3.4, 3.4, 3.4], new=True)

# ------------------------------------------------------------------ Table S10: two-stage versus end-to-end learning on the same training conditions
tf = pd.read_csv(RES / 'supp_twostage_fair.csv'); tm = tf.groupby(['variant', 'scenario'])[['n_train', 'R2', 'MAE', 'R2_complete', 'MAE_complete']].mean()
import json as _json
tp = _json.load(open(RES / 'tuned_two-stage_revision.json'))['params']
LAB = [('two-stage (fixed settings)', 'Two-stage, fixed settings (Table 6)', 'complete curves'), ('two-stage (tuned)', 'Two-stage, tuned regressors', 'complete curves'),
       ('tanh-XGB, complete curves only', 'tanh-XGB, end-to-end', 'complete curves'), ('tanh-NN, complete curves only', 'tanh-NN, end-to-end', 'complete curves'),
       ('tanh-XGB, all measurements', 'tanh-XGB, end-to-end (Table 6)', 'all measurements'), ('tanh-NN, all measurements', 'tanh-NN, end-to-end (Table 6)', 'all measurements')]
rows = []
for key, lab, data in LAB:
    r = [lab, data]
    for s_ in SC: r += [f3(tm.loc[(key, s_), 'R2']), f3(tm.loc[(key, s_), 'R2_complete']), f1(tm.loc[(key, s_), 'MAE_complete'])]
    rows.append(r)
para('**Table S10.** Two-stage and end-to-end learning with the same training data. "Complete curves" denotes training on the conditions of the training folds that were '
     'tested at four or more temperatures, i.e., the conditions that the two-stage procedure can use; "all measurements" the full training folds. All variants are evaluated on '
     'the same held-out measurements: R² (all) refers to all held-out measurements as in Table 6, R² (cc) and MAE (cc) to the held-out measurements that belong to complete-curve '
     f'conditions ({int((df.groupby("cond").test_temp.transform("nunique") >= 4).sum())} measurements). The tuned two-stage regressors were selected with the Optuna procedure of Section 2.4 (30 trials; '
     f'{int(tp["n_estimators"])} trees, learning rate {tp["learning_rate"]:.4f}, depth {int(tp["max_depth"])}, minimum child weight {tp["min_child_weight"]:.1f}, λ = {tp["reg_lambda"]:.2f}, '
     f'column/row subsampling {tp["colsample_bytree"]:.2f}/{tp["subsample"]:.2f}); the end-to-end models use the settings of Section 2.4, which were tuned on all measurements. '
     'Five folds, mean of three repetitions.', new=True)
table(['Model', 'Training data', 'Random R² (all)', 'Random R² (cc)', 'Random MAE (cc)', 'Unseen cond. R² (all)', 'Unseen cond. R² (cc)', 'Unseen cond. MAE (cc)',
       'Unseen steel R² (all)', 'Unseen steel R² (cc)', 'Unseen steel MAE (cc)'], rows, [5.2, 2.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], new=True)
d.add_page_break()

# ------------------------------------------------------------------ Table S11: undetermined elements of the cast steels
mc = pd.read_csv(RES / 'cast_missing_chemistry_sensitivity.csv'); order_ = list(dict.fromkeys(mc.scenario))
pv = mc.pivot(index='scenario', columns='model', values='MAE'); tt_ = mc.pivot(index='scenario', columns='model', values='Tt'); us_ = mc.pivot(index='scenario', columns='model', values='USE')
def _val(sc_):
    q = mc[mc.scenario == sc_].iloc[0]
    return ' / '.join('0' if pd.isna(q[e]) else f'{q[e]:.3f}'.rstrip('0').rstrip('.') for e in ['P', 'S', 'N', 'Al'])
LBL = {'as analysed (undetermined elements = 0)': 'As analysed (main analysis)', 'P, S = database medians': 'P, S at database medians', 'P, S, N = database medians': 'P, S, N at database medians',
       'P, S, N, Al = database medians': 'P, S, N, Al at database medians', 'P, S, N = upper quartiles': 'P, S, N at upper quartiles', 'P, S, N, Al = upper quartiles': 'P, S, N, Al at upper quartiles',
       'P = 0.020, S = 0.020, N = 0.010, Al = 0': 'High P, S and N'}
rows = [[LBL[sc_], _val(sc_)] + [f1(pv.loc[sc_, m_]) for m_ in MODELS6] + [f'{tt_.loc[sc_, "tanh-NN"]:.0f} / {us_.loc[sc_, "tanh-NN"]:.0f}'.replace('-', '−')] for sc_ in order_]
para('**Table S11.** Sensitivity of the external test to the elements that were not analysed in the cast steels (P, S, N and Al; entered as zero in the main analysis). '
     'The literature-trained models of Table 8 were evaluated on the 32 cast-steel measurements after assigning the same assumed concentrations (wt.%) to all four steels: '
     'the medians or upper quartiles of the non-zero values in the literature database, or a high-impurity case. The physics-derived descriptors were recomputed. '
     'The last column gives the mean transition temperature and upper-shelf energy predicted by tanh-NN for the four steels.', new=True)
table(['Assumed concentrations', 'P / S / N / Al (wt.%)'] + [disp(m_) + ' MAE (J)' for m_ in MODELS6] + ['tanh-NN Tt (°C) / USE (J)'], rows,
      [5.6, 4.2, 2.1, 2.5, 2.1, 2.1, 2.1, 2.1, 3.0], new=True)

# ------------------------------------------------------------------ Table S12: cast-only baselines and cast-data weight
cb = pd.read_csv(RES / 'cast_only_baselines_LOCO.csv'); ls = pd.read_csv(RES / 'finetune_lambda_sensitivity.csv'); ft = pd.read_csv(RES / 'finetune_LOCO_derived.csv')
AL = ['MSN', 'MSNN', 'MSCM', 'MSCMN']; rows = []
BL = {'mean of the other three steels at each temperature': 'Cast data only: mean of the three training steels at each test temperature', 'pooled straight line': 'Cast data only: straight line fitted to the 24 measurements',
      'pooled tanh curve': 'Cast data only: Equation (1) fitted to the 24 measurements'}
for k_, lab in BL.items():
    g = cb[cb.baseline == k_].set_index('alloy'); rows.append([lab] + [f1(g.MAE[a_]) for a_ in AL] + [f1(g.MAE.mean()), f3(g.R2.mean())])
for m_ in ['tanh-NN', 'NN-direct', 'tanh-XGB', 'XGB-direct', 'XGB-direct-mono']:
    for meth, lab in [('calib', 'linear re-calibration'), ('ft', 'fine-tuning, λ = 5')]:
        g = ft[(ft.model == m_) & (ft.method == meth)].set_index('alloy'); rows.append([f'{m_}: {lab} (Table 8)'] + [f1(g.MAE[a_]) for a_ in AL] + [f1(g.MAE.mean()), f3(g.R2.mean())])
para('**Table S12.** Leave-one-cast-steel-out errors (MAE, J) on the held-out cast steel: baselines that use only the 24 measurements of the three training cast steels, without any '
     'literature data, compared with the adapted literature-trained models of Table 8. The pooled tanh curve was fitted with the bounds of Section 2.2.', new=True)
table(['Method'] + AL + ['Mean MAE (J)', 'Mean R²'], rows, [10.5, 2.0, 2.0, 2.0, 2.0, 2.6, 2.2], new=True)
lm = ls.groupby(['model', 'lam']).MAE.mean().unstack()
rows = [[m_] + [f1(lm.loc[m_, l_]) for l_ in [1.0, 5.0, 20.0]] for m_ in ['tanh-NN', 'NN-direct', 'tanh-XGB', 'XGB-direct', 'XGB-direct-mono']]
para('Sensitivity of the fine-tuned models to the weight λ of the cast data (mean MAE over the four held-out steels, J). λ = 5 is the value used in Table 8; it was fixed before the '
     'leave-one-out evaluation and not optimised on the held-out steels. This is a separate run, so that the tree-model values at λ = 5 differ from Table 8 by up to 0.7 J because of the '
     'random row and column subsampling of the boosting rounds.', new=True)
table(['Model', 'λ = 1', 'λ = 5', 'λ = 20'], rows, [4.0, 2.4, 2.4, 2.4], new=True)

d.add_page_break()
# ------------------------------------------------------------------ Table S13: robustness of the rank correlations
cr = pd.read_csv(RES / 'supp_corr_robustness.csv')
NM = {'T_temp': 'Tempering temperature', 'HJ_temper': 'PHJ', 'Ms_Andrews': 'Ms (Andrews)', 'Q_temp': 'Austenitising temperature', 'Q_time': 'Austenitising time', 'T_time': 'Tempering time',
      'T_type': 'Tempering cooling rate', 'Rolled_Forged_Cast': 'Product form (cast = 0)'}
def _rho(r, k):
    v, p_ = r.get('rho_' + k), r.get('p_' + k)
    return '–' if pd.isna(v) else f'{v:+.2f}'.replace('-', '−') + (' **' if p_ < 0.01 else ' *' if p_ < 0.05 else '')
rows = [[NM.get(r.descriptor, r.descriptor), f'{int(r.n_nonzero)} / {int(r.n_nonzero_resolved)}', _rho(r, 'Tt_all'), _rho(r, 'Tt_resolved'), _rho(r, 'Tt_reported'),
         _rho(r, 'USE_all'), _rho(r, 'USE_resolved'), _rho(r, 'USE_reported')] for _, r in cr.iterrows()]
para(f'**Table S13.** Robustness of the Spearman rank correlations between descriptors and fitted curve parameters (Table 5 of the article): all {len(fu)} complete curves, the {nres} well-resolved curves only '
     '(classification of Table S6), and, for elements that many sources do not report and that are then entered as zero, the curves with a reported (non-zero) value only. '
     'The second column gives the number of curves with a non-zero value of the descriptor among all and among the well-resolved curves. ** p < 0.01, * p < 0.05.', new=True)
table(['Descriptor', 'Non-zero (all / well resolved)', 'Tt, all curves', 'Tt, well resolved', 'Tt, reported values only', 'USE, all curves', 'USE, well resolved', 'USE, reported values only'],
      rows, [5.0, 3.6, 2.6, 2.6, 3.0, 2.6, 2.6, 3.0], new=True)

out = ROOT / ('Supplementary_Materials_ver5_revised.docx' if HL else 'Supplementary_Materials_ver5_clean.docx'); d.save(out); print('written', out)
