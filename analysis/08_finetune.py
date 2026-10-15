"""Fine-tuning / domain adaptation to the user's own cast steels.
Protocol: leave-one-cast-alloy-out (LOCO). A model pre-trained on the literature data (634 pts) is fine-tuned with the
Charpy data of 3 cast alloys (24 pts) and evaluated on the 4th.  Methods:
  - none        : pre-trained model as is (baseline)
  - calib       : linear re-calibration of the pre-trained prediction  (y = a + b * pred) fitted on the 3 alloys
  - ft          : fine-tuning
        tanh-NN / NN-direct : continue training all weights (low lr) on cast data + literature replay
        tanh-XGB / XGB-direct: continued boosting (xgb_model=pretrained) on literature + up-weighted cast data
usage: python 08_finetune.py [derived:0/1] [models...]"""
import sys, warnings, copy; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, xgboost as xgb, torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import common, models
from common import load, cast_alloys, CAST_MEAS, metrics, RES, tanh_curve

derived = bool(int(sys.argv[1])) if len(sys.argv) > 1 else True
which = sys.argv[2:] or ['XGB-direct', 'XGB-direct-mono', 'tanh-XGB', 'NN-direct', 'tanh-NN']
af = common.COMP + common.PROC + (common.DERIVED if derived else [])
models.ALLOY_FEATS = af; models.FEATS = af + ['test_temp']
LAMBDA = 5.0      # weight of a cast data point relative to a literature point during fine-tuning

lit = load(); cast = cast_alloys()
cd = CAST_MEAS.merge(cast, on='Alloy')                      # 32 rows with features
cd['cond'] = 1000 + pd.factorize(cd.Alloy)[0]; cd['alloy_id'] = cd['cond']
COL = {'none': 'gray', 'calib': 'C2', 'ft': 'C3'}
if 'XGB-direct-mono' in which and 'tuned_XGB-direct-mono.json' not in [p.name for p in RES.glob('tuned_*.json')]: print('warning: XGB-direct-mono not tuned')


def finetune(name, base, dtrain):
    """return a fine-tuned copy of pre-trained model `base` using cast rows `dtrain`"""
    m = copy.deepcopy(base)
    if name in ('tanh-XGB',):
        X = np.vstack([lit[models.ALLOY_FEATS].values, dtrain[models.ALLOY_FEATS].values])
        y = np.concatenate([lit.Impact.values, dtrain.Impact.values]); T = np.concatenate([lit.test_temp.values, dtrain.test_temp.values])
        w = np.concatenate([np.ones(len(lit)), LAMBDA * np.ones(len(dtrain))])
        m.T_tr = T; m.w = w
        obj0 = models.TanhXGB._obj
        def obj(R, dm):
            g, h = obj0(m, R, dm); return g * w[:, None], h * w[:, None]
        dm = xgb.DMatrix(X, label=y, base_margin=np.tile(m.base, (len(y), 1)))
        m.bst = xgb.train(dict(tree_method='hist', multi_strategy='multi_output_tree', num_target=4, max_depth=3, eta=0.02,
                               subsample=0.8, colsample_bynode=0.8, min_child_weight=3, reg_lambda=5.0, nthread=12, seed=0),
                          dm, num_boost_round=300, obj=obj, xgb_model=base.bst)
    elif name in ('XGB-direct', 'XGB-direct-mono'):
        X = np.vstack([lit[models.FEATS].values, dtrain[models.FEATS].values]); y = np.concatenate([lit.Impact.values, dtrain.Impact.values])
        w = np.concatenate([np.ones(len(lit)), LAMBDA * np.ones(len(dtrain))])
        extra = {'monotone_constraints': base.p['monotone_constraints']} if name == 'XGB-direct-mono' else {}
        m.m = xgb.XGBRegressor(**{**models.XGB_P, 'n_estimators': 300, 'learning_rate': 0.02, 'max_depth': 3, 'reg_lambda': 5.0, **extra})
        m.m.fit(X, y, sample_weight=w, xgb_model=base.m.get_booster())
    elif name in ('tanh-NN', 'NN-direct'):
        Xl = torch.tensor(m.sc.transform(lit)); Xc = torch.tensor(m.sc.transform(dtrain))
        Tl = torch.tensor(lit.test_temp.values, dtype=torch.float32); Tc = torch.tensor(dtrain.test_temp.values, dtype=torch.float32)
        yl = torch.tensor(lit.Impact.values, dtype=torch.float32); yc = torch.tensor(dtrain.Impact.values, dtype=torch.float32)
        for net in m.nets:
            opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-3); net.train()
            for ep in range(300):
                opt.zero_grad()
                if name == 'tanh-NN':
                    loss = ((net(Xl, Tl) - yl) ** 2).mean() + LAMBDA * ((net(Xc, Tc) - yc) ** 2).mean()
                else:
                    f = lambda X: torch.sigmoid(net(X).squeeze(1) - 1.5) * 400.
                    loss = ((f(Xl) - yl) ** 2).mean() + LAMBDA * ((f(Xc) - yc) ** 2).mean()
                (loss / 6400.).backward(); opt.step()
            net.eval()
    return m


Tg = np.linspace(-100, 60, 81)
rows, curves = [], {}
for name in which:
    print('pre-training', name, flush=True); base = models.MODELS[name]().fit(lit)
    for al in cast.Alloy:
        dtr, dte = cd[cd.Alloy != al], cd[cd.Alloy == al]
        p0 = base.predict(dte)
        # linear recalibration on the 3 training alloys
        ptr = base.predict(dtr); b, a = np.polyfit(ptr, dtr.Impact.values, 1); p1 = a + b * p0
        ft = finetune(name, base, dtr); p2 = ft.predict(dte)
        for meth, p in [('none', p0), ('calib', p1), ('ft', p2)]:
            rows.append(dict(model=name, alloy=al, method=meth, **metrics(dte.Impact.values, p)))
        row = pd.DataFrame([cast[cast.Alloy == al].iloc[0].to_dict()] * len(Tg)); row['test_temp'] = Tg
        curves[(name, al)] = dict(none=base.predict(row), calib=a + b * base.predict(row), ft=ft.predict(row))
        print(f'  {name:11s} held-out {al:6s}  MAE none={rows[-3]["MAE"]:.1f}  calib={rows[-2]["MAE"]:.1f}  ft={rows[-1]["MAE"]:.1f}', flush=True)
    # final: fine-tune on all four alloys (for the paper's final curves) and store the parameters
    ft_all = finetune(name, base, cd)
    if hasattr(ft_all, 'curve_params'):
        P = ft_all.curve_params(cast.assign(test_temp=25.)); P.insert(0, 'Alloy', cast.Alloy.values); P.to_csv(RES / f'finetuned_params_{name}.csv', index=False)
        print(P.round(1).to_string(index=False))

res = pd.DataFrame(rows); res.to_csv(RES / f'finetune_LOCO{"_derived" if derived else ""}.csv', index=False)
print('\nLOCO mean over 4 cast alloys:')
print(res.groupby(['model', 'method'])[['R2', 'MAE', 'RMSE']].mean().round(2).to_string())

# figure: per model, 4 alloys, before/after
for name in which:
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, al in zip(axs, cast.Alloy):
        meas = CAST_MEAS[CAST_MEAS.Alloy == al]; c = curves[(name, al)]
        for meth, lab in [('none', 'pre-trained (literature only)'), ('calib', 'linear re-calibration'), ('ft', 'fine-tuned on other 3 cast alloys')]:
            ax.plot(Tg, c[meth], c=COL[meth], label=lab, lw=1.8)
        ax.plot(meas.test_temp, meas.Impact, 'ko', ms=5, label='measured (held out)')
        ax.set_title(f'{name}: {al} (LOCO)', fontsize=10); ax.set_xlabel('Test T (°C)'); ax.set_ylabel('CVN (J)'); ax.set_ylim(0, 300); ax.grid(alpha=.3)
    axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(RES / f'fig_finetune_{name}.png', dpi=140); plt.close()
print('done')
