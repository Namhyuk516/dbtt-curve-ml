# Learning the Ductile-to-Brittle Transition Curve of Steels: data and code

Companion repository of the manuscript *Learning the Ductile-to-Brittle Transition Curve of Steels: Direct Machine Learning versus
Empirical-Equation-Constrained Models with Validation on In-House Cast Steels* (submitted to *Metals*, MDPI).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23384571.svg)](https://doi.org/10.5281/zenodo.23384571)

## Citation
Seo, N.; Park, J.-W.; Yang, J.; Lee, S.-J.; Kim, M.-S. dbtt-curve-ml: data and code for 'Learning the Ductile-to-Brittle Transition Curve of Steels' (v1.1.0). Zenodo, 2026. https://doi.org/10.5281/zenodo.23384571
(see CITATION.cff; the version used in the manuscript is 10.5281/zenodo.23460289).

## Contents
- `data/charpy_literature_database.csv` – the 604 literature Charpy V-notch measurements used in the article (cp949 encoding; 58 steels, 169 alloy/heat-treatment conditions, 40 publications). Column `No` is the record number used in `data/TableS1_sources.csv` (row ranges per source publication). `data/TableS1_specimens.csv` lists the Charpy specimen size, notch, test standard and orientation stated in each source. Test and heat-treatment temperatures are stored in K.
- `data/in_house_cast_steels.xlsx` – composition and Charpy energies of the four in-house cast steels (also hard-coded in `analysis/common.py`).
- `analysis/` – all scripts (see "Reproduction"). `models.py` contains the six models, including the tanh-constrained XGBoost objective.
- `results/tables/` – cross-validation results (`table_cv_summary.csv`, `oof_all.csv`), curve fits, SHAP importances, cast-steel validation,
  fine-tuning, supplementary analyses (`supp_*.csv`, `nested_cv_*.csv`) and the tuned hyper-parameters (`tuned_*.json`).
- `results/figures/` – all figures of the article and the supplementary material.

## Requirements
Python 3.12 with numpy, pandas, scipy, scikit-learn, **xgboost 3.2.0** (the multi-output custom objective of tanh-XGB and the exact numbers
of the article depend on this version), torch (CPU), optuna, shap, matplotlib.

## Reproduction
```
python analysis/01_fit_curves.py
python analysis/09_tune.py XGB-direct 60      # also XGB-direct-mono 60, tanh-XGB 30, NN-direct 30, tanh-NN 30
python analysis/03_compare.py                 # Table 6 (about 1.5 h)
python analysis/06_figures.py; python analysis/11_final_vs_heldout.py; python analysis/14_class_errors.py
python analysis/04_cast.py; python analysis/05_shap.py; python analysis/12_shap_composite.py; python analysis/08_finetune.py 1
python analysis/13_supp_cv.py ablation_tree; python analysis/13_supp_cv.py ablation_nn; python analysis/13_supp_cv.py multi_temp
python analysis/15_nested_cv.py XGB-direct 20   # also XGB-direct-mono, tanh-XGB
python analysis/10_eda.py
```
`analysis/run_pipeline.ps1` runs everything after tuning in four parallel chains (Windows PowerShell; edit the Python path at the top).

## Licence
Code: MIT (see `LICENSE`). Data: CC BY 4.0 (see `data/LICENSE-DATA.md`).




