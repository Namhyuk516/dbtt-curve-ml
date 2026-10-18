# -*- coding: utf-8 -*-
"""Build Supplementary_Materials.docx (Tables S1-S4) from the result files in ../results."""
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


def para(text, bold=False, size=10, align=None, after=6):
    p = d.add_paragraph(); p.paragraph_format.space_after = Pt(after)
    if align: p.alignment = align
    for k, seg in enumerate(re.split(r'\*\*', text)):
        if seg:
            r = p.add_run(seg); r.bold = bold or k % 2 == 1; r.font.size = Pt(size)
    return p


def rule(cell, edge, sz=8):
    tcPr = cell._tc.get_or_add_tcPr(); b = tcPr.find(qn('w:tcBorders'))
    if b is None: b = OxmlElement('w:tcBorders'); tcPr.append(b)
    e = OxmlElement(f'w:{edge}'); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), str(sz)); e.set(qn('w:space'), '0'); e.set(qn('w:color'), 'auto'); b.append(e)


def table(header, rows, widths, size=8, align_left=(0,)):
    t = d.add_table(rows=1 + len(rows), cols=len(header)); t.autofit = False
    for i, row in enumerate([header] + rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.width = Cm(widths[j]); p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in align_left else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(v)); r.font.size = Pt(size); r.bold = i == 0
            if i == 0: rule(c, 'top'); rule(c, 'bottom', 4)
            if i == len(rows): rule(c, 'bottom')
    d.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


f3 = lambda v: '–' if pd.isna(v) else f'{v:.2f}'.replace('-', '−')
f1 = lambda v: '–' if pd.isna(v) else f'{v:.1f}'.replace('-', '−')
SC = ['random', 'unseen-condition', 'unseen-alloy']

para('Supplementary Materials', bold=True, size=14, after=4)
para(TITLE, bold=True, size=11, after=12)

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
    table(['No.', 'Reference', 'DOI', 'Specimen', 'Measurements'], rows, [0.9, 13.4, 4.6, 5.2, 2.0], size=7.5, align_left=(1, 2, 3))
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

# ------------------------------------------------------------------ Table S5: nested cross-validation (tree models)
nc = pd.concat([pd.read_csv(f) for f in RES.glob('nested_cv_*.csv')]).set_index(['model', 'scenario'])
rows = []
for model in ['XGB-direct', 'XGB-direct-mono', 'tanh-XGB']:
    r = [model]
    for s in ['unseen-condition', 'unseen-alloy']:
        r += [f3(cvs.loc[(model, s), 'R2_mean']), f1(cvs.loc[(model, s), 'MAE_mean']), f3(nc.loc[(model, s), 'R2']), f1(nc.loc[(model, s), 'MAE'])]
    rows.append(r)
para(f'**Table S5.** Nested cross-validation of the tuned tree models. In the nested procedure the hyper-parameters are re-selected inside every outer fold '
     f'(Optuna, {int(nc.n_inner.iloc[0])} trials on an inner five-fold grouped split of the training folds), so that no information from the held-out fold enters the '
     'selection; outer: five-fold grouped split, one repetition. "Table 6" denotes the values obtained with the globally tuned hyper-parameters.')
table(['Model', 'Unseen cond. R² (Table 6)', 'Unseen cond. MAE (Table 6)', 'Unseen cond. R² (nested)', 'Unseen cond. MAE (nested)',
       'Unseen steel R² (Table 6)', 'Unseen steel MAE (Table 6)', 'Unseen steel R² (nested)', 'Unseen steel MAE (nested)'], rows, [2.8, 2.8, 2.9, 2.8, 2.9, 2.8, 2.9, 2.8, 2.9])

# ------------------------------------------------------------------ Table S6: parameter-level accuracy of the equation-based models
pa_path = RES / 'supp_param_accuracy.csv'
if pa_path.exists():
    pa = pd.read_csv(pa_path); pm = pa.groupby(['model', 'scenario'])[['MAE_Tt', 'bias_Tt', 'MAE_USE', 'bias_USE', 'within30', 'n']].mean()
    fu = pd.read_csv(RES / 'curve_fit_uncertainty.csv')
    rows = []
    for model in ['two-stage', 'tanh-XGB', 'tanh-NN']:
        for s in ['unseen-condition', 'unseen-alloy']:
            q = pm.loc[(model, s)]
            rows.append([disp(model), s.replace('unseen-alloy', 'unseen steel').replace('unseen-condition', 'unseen condition'), f1(q.MAE_Tt), f1(q.bias_Tt), f1(q.MAE_USE), f1(q.bias_USE), f'{100 * q.within30:.0f}%'])
    para(f'**Table S6.** Accuracy of the curve parameters predicted by the equation-based models for the {int(pa.n.max())} conditions with complete curves, '
         'compared with the parameters fitted directly to the measurements (Equation (1)); grouped cross-validation, mean of three repetitions. '
         f'For reference, the fitted parameters have standard deviations of {fu.Tt.std():.0f} °C (<i>T</i><sub>t</sub>) and {fu.USE.std():.0f} J (USE) across the '
         f'{len(fu)} curves, and their median standard errors from the least-squares fit are {fu.se_Tt.median():.1f} °C and {fu.se_USE.median():.1f} J. '
         '"Within 30 °C" is the fraction of conditions whose predicted <i>T</i><sub>t</sub> lies within ±30 °C of the fitted value.'.replace('<i>T</i><sub>t</sub>', 'Tt'))
    table(['Model', 'Scheme', 'MAE Tt (°C)', 'Bias Tt (°C)', 'MAE USE (J)', 'Bias USE (J)', 'Within 30 °C'], rows, [2.8, 3.4, 2.6, 2.6, 2.6, 2.6, 2.6])
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

out = ROOT / 'Supplementary_Materials.docx'; d.save(out); print('written', out)
