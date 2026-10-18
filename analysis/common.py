"""Common data handling for the low-alloy-steel DBTT modelling study."""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Impact_data_all_no_byukjin_dedup.csv"
RES = ROOT / "results"; RES.mkdir(exist_ok=True)

COMP = ['C','Mn','Si','Cr','Ni','Mo','Co','V','Cu','B','N','P','S','Al','Nb','Ti','W']
PROC = ['Rolled_Forged_Cast','Q_temp','Q_time','Q_type','T_temp','T_time','T_type']
DERIVED = ['CE_IIW','Pcm','HJ_temper','Ms_Andrews','Ac3_Andrews','logQrate','logTrate','sum_alloy']   # physics-informed, see add_derived()
ALLOY_FEATS = COMP + PROC + DERIVED       # features that describe an alloy + heat treatment
FEATS = ALLOY_FEATS + ['test_temp']      # + test temperature (for direct models)
NICE = {'Rolled_Forged_Cast':'Form (R/F/C)','Q_temp':'Aust. T (°C)','Q_time':'Aust. time (min)',
        'Q_type':'Aust. cooling rate','CE_IIW':'CE (IIW)','Pcm':'Pcm','HJ_temper':'Hollomon-Jaffe P','Ms_Andrews':'Ms (Andrews)','Ac3_Andrews':'Ac3 (Andrews)','logQrate':'log aust. cooling rate','logTrate':'log temper cooling rate','sum_alloy':'Total alloy (wt%)','T_temp':'Temper T (°C)','T_time':'Temper time (min)',
        'T_type':'Temper cooling rate','test_temp':'Test T (°C)'}

def add_derived(df):
    """Physics-informed derived features (all computed from existing columns; safe for unseen alloys)."""
    d = df
    df['CE_IIW'] = d.C + d.Mn/6 + (d.Cr + d.Mo + d.V)/5 + (d.Ni + d.Cu)/15
    df['Pcm'] = d.C + d.Si/30 + (d.Mn + d.Cu + d.Cr)/20 + d.Ni/60 + d.Mo/15 + d.V/10 + 5*d.B
    # Hollomon-Jaffe tempering parameter (K, hours); 0 when not tempered
    df['HJ_temper'] = np.where(d.T_temp > 0, (d.T_temp + 273.15)*(20 + np.log10(np.maximum(d.T_time, 1)/60.))/1000., 0.0)
    df['Ms_Andrews'] = 539 - 423*d.C - 30.4*d.Mn - 12.1*d.Cr - 17.7*d.Ni - 7.5*d.Mo
    df['Ac3_Andrews'] = 910 - 203*np.sqrt(d.C) - 15.2*d.Ni + 44.7*d.Si + 104*d.V + 31.5*d.Mo + 13.1*d.W
    df['logQrate'] = np.log10(np.maximum(d.Q_type, 0.01)); df['logTrate'] = np.log10(np.maximum(d.T_type, 0.01))
    df['sum_alloy'] = d[COMP].sum(axis=1)
    return df

# Data file: 604 V-notch Charpy measurements of steels.
# Sources with sub-size specimens (5x10, 7x10, 2.5x10 mm) or without specimen information are kept and used for the sensitivity analysis of
# 13_supp_cv.py fullsize (row ranges = column No of the data file).
SUBSIZE_ROWS = [(47, 49), (297, 301), (354, 362), (422, 445), (324, 332), (478, 534)]
UNSTATED_ROWS = [(80, 96), (221, 226), (243, 246)]


def _in_ranges(no, ranges):
    return np.any([(no >= a) & (no <= b) for a, b in ranges], axis=0)


def load(derived=True):
    df = pd.read_csv(DATA, encoding='cp949')
    # Kelvin -> Celsius for readability (0 = no tempering stays a flag: set to -273 -> use 0 sentinel instead)
    df['test_temp'] = df['test_temp'] - 273.15
    df['Q_temp'] = df['Q_temp'] - 273.15
    df['T_temp'] = np.where(df['T_temp'] > 0, df['T_temp'] - 273.15, 0.0)   # 0 => as-quenched / no temper
    # condition id = alloy + composition + heat treatment  (one DBTT curve)
    # an alloy = a unique composition (the same steel may appear under several names, e.g. SCM420-T2xx/T3xx,
    # CA_/COQ_AISI4340); a condition = composition + heat treatment = one DBTT curve
    df['alloy_id'] = pd.factorize(df[COMP].astype(str).agg('|'.join, axis=1))[0]
    df['cond'] = pd.factorize(df[COMP + PROC].astype(str).agg('|'.join, axis=1))[0]
    if derived: add_derived(df)
    return df

def cast_alloys():
    """User's own cast alloys (My_cast_data.png). Confirmed heat treatment: Q&T, 900 C x 4 h water quench + 600 C x 4 h temper.
    ASTM E23 standard V-notch specimens."""
    base = dict(Co=0,V=0,Cu=0,B=0,N=0,P=0,S=0,Al=0,Nb=0,Ti=0,W=0,
                Rolled_Forged_Cast=0.0, Q_temp=900.0, Q_time=240, Q_type=200, T_temp=600.0, T_time=240, T_type=1)
    rows = {
      'MSN'  : dict(C=0.24,Mn=1.17,Si=0.35,Cr=0.00,Mo=0.00,Ni=0.36),
      'MSNN' : dict(C=0.18,Mn=1.19,Si=0.32,Cr=0.00,Mo=0.00,Ni=0.72),
      'MSCM' : dict(C=0.19,Mn=1.30,Si=0.42,Cr=0.29,Mo=0.12,Ni=0.00),
      'MSCMN': dict(C=0.23,Mn=1.45,Si=0.51,Cr=0.35,Mo=0.11,Ni=0.39),
    }
    out = []
    for k,v in rows.items():
        d = dict(base); d.update(v); d['Alloy']=k; out.append(d)
    return add_derived(pd.DataFrame(out))

# measured Charpy energies (J) of the cast alloys: exact values from Own_cast_data.xlsx (all four alloys).
# (until 2026-10-02 MSNN / MSCMN were digitised from the plot; replaced by the tabulated values)
CAST_MEAS = pd.DataFrame([
  ('MSN',  25, 212.00, False), ('MSN',  25, 209.30, False), ('MSN',  0, 135.30, False), ('MSN',  0, 142.50, False),
  ('MSN', -20,  97.97, False), ('MSN', -20, 100.10, False), ('MSN',-40,  35.97, False), ('MSN',-40,  21.92, False),
  ('MSCM', 25, 227.10, False), ('MSCM', 25, 219.80, False), ('MSCM', 0, 192.90, False), ('MSCM', 0, 163.60, False),
  ('MSCM',-20,  43.59, False), ('MSCM',-20,  48.80, False), ('MSCM',-40, 24.43, False), ('MSCM',-40, 29.30, False),
  ('MSNN', 25, 182.60, False), ('MSNN', 25, 192.70, False), ('MSNN', 0, 134.40, False), ('MSNN', 0, 156.40, False),
  ('MSNN',-20,  90.86, False), ('MSNN',-20, 141.60, False), ('MSNN',-40, 50.23, False), ('MSNN',-40, 19.41, False),
  ('MSCMN',25, 176.80, False), ('MSCMN',25, 212.00, False), ('MSCMN',0, 124.10, False), ('MSCMN',0, 173.70, False),
  ('MSCMN',-20, 28.33, False), ('MSCMN',-20, 37.55, False), ('MSCMN',-40,42.49, False), ('MSCMN',-40, 19.28, False),
], columns=['Alloy','test_temp','Impact','approx'])

TT = r'$T_\mathrm{t}$'     # symbol of the transition temperature in figures (was T0; renamed to avoid clash with the Master-Curve T0)


def source_table(df):
    """Table S1 numbering: TableS1_sources.csv rows -> (No., n points in df); duplicate publications merged, empty ones dropped."""
    src = pd.read_csv(RES / 'TableS1_sources.csv', encoding='utf-8-sig')
    src['n'] = [int(df.No.between(a, b).sum()) for a, b in zip(src.row_start, src.row_end)]
    rows, seen = [], {}
    for _, r in src.iterrows():
        key = r.doi if isinstance(r.doi, str) else f'{r.authors}{r.year}'
        if key in seen: rows[seen[key]]['n'] += r.n; rows[seen[key]]['ranges'].append((r.row_start, r.row_end)); continue
        seen[key] = len(rows); rows.append(dict(r.to_dict(), ranges=[(r.row_start, r.row_end)]))
    rows = [r for r in rows if r['n'] > 0]
    for i, r in enumerate(rows, 1): r['No'] = i
    return rows


def alloy_label(df, alloy):
    """Display name for a steel in figures/tables: internal code -> readable name + Table S1 source number."""
    name = alloy.replace('_steel', ' steel').replace('_', ' ').replace('3Mn-Si-Ni', '3Mn–Si–Ni')
    no = df.loc[df.Alloy == alloy, 'No']
    refs = sorted({r['No'] for r in source_table(df) if any(a <= n <= b for n in no for a, b in r['ranges'])})
    return f'{name} [S{",".join(map(str, refs))}]'


def tanh_curve(T, LSE, USE, T0, C):
    """Charpy energy vs temperature: E = A + B tanh((T-T0)/C), A=(USE+LSE)/2, B=(USE-LSE)/2"""
    A = 0.5*(USE+LSE); B = 0.5*(USE-LSE)
    return A + B*np.tanh((T - T0)/C)

def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ss = ((y-y.mean())**2).sum(); sr = ((y-p)**2).sum()
    return dict(R2=1-sr/ss, MAE=np.abs(y-p).mean(), RMSE=np.sqrt(((y-p)**2).mean()))

def group_splits(df, n_splits=5, n_repeats=3, group='alloy_id'):
    """Repeated GroupKFold by alloy: a held-out alloy never appears in training."""
    from sklearn.model_selection import GroupKFold
    g = df[group].values
    for r in range(n_repeats):
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=r)
        for k,(tr,te) in enumerate(gkf.split(df, groups=g)):
            yield r, k, tr, te
