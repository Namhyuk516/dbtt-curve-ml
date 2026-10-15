# -*- coding: utf-8 -*-
"""Manuscript content (MDPI Metals).  Inline markup: <i>, <b>, <sub>, <sup>."""
from pathlib import Path
R = Path(__file__).resolve().parent.parent / 'results'
OUT_NAME = 'Manuscript_DBTT_ML_Metals.docx'

ARTICLE_TYPE = 'Article'
TITLE = ('Learning the Ductile-to-Brittle Transition Curve of Steels: Direct Machine Learning versus '
         'Empirical-Equation-Constrained Models with Validation on In-House Cast Steels')
AUTHORS = 'Namhyuk Seo <sup>1,</sup>*'
AFFILIATIONS = ['<sup>1</sup> [Affiliation, Department, University/Institute, City, Country]; seonh9405@gmail.com',
                '* Correspondence: seonh9405@gmail.com']
ABSTRACT = (
    'The Charpy impact energy of body-centred-cubic steels varies sigmoidally with temperature, and the resulting ductile-to-brittle transition (DBT) curve is '
    'commonly described by a hyperbolic-tangent (tanh) equation. This study compares two ways of learning this curve from heterogeneous literature data: direct '
    'regression of the impact energy, and an empirical-equation-constrained approach in which the learner predicts the four tanh parameters and is trained end-to-end '
    'through the equation. A database of 616 Charpy measurements from 43 publications (61 steels, 174 alloy/heat-treatment conditions) was compiled and supplemented '
    'with physics-derived descriptors, and six tree and neural-network models were tuned with the same budget. Under random cross-validation the models reach R<sup>2</sup> = 0.76–0.91, but when whole steels are held out '
    'R<sup>2</sup> falls to 0.30 or less for all of them. The equation-constrained models return smooth sigmoids with interpretable parameters without loss of accuracy '
    'relative to their direct counterparts. In an external test on four in-house quenched-and-tempered cast steels, every literature-trained model under-estimated the '
    'room-temperature energy (mean absolute error 39–67 J); fine-tuning with three of the cast steels reduced the error on the fourth to 19–20 J for the neural '
    'networks. SHAP analysis attributed both the transition temperature and the upper-shelf energy mainly to carbon, tempering and the Andrews M<sub>s</sub> temperature.')
KEYWORDS = ('ductile-to-brittle transition; Charpy impact energy; steel; cast steel; machine learning; XGBoost; physics-informed neural network; '
            'transfer learning; SHAP')

F = lambda n: str(R / n)
BODY = [
# ----------------------------------------------------------------------------- 1 Introduction
('h1', '1. Introduction'),
('p', 'Ferritic and martensitic steels lose their capacity for plastic deformation at low temperature, and the fracture mode changes from ductile micro-void '
      'coalescence to brittle cleavage over a narrow temperature interval [@morris2008,pineau2016]. This ductile-to-brittle transition (DBT) is routinely characterised by Charpy V-notch '
      'impact testing [@astm2018], and the temperature dependence of the absorbed energy, the DBT curve, is the basis of toughness specifications for pressure vessels, '
      'pipelines, wind-turbine components and heavy cast parts. Because a full curve requires specimens at 6–10 temperatures, and because every change in composition '
      'or heat treatment shifts the curve, there is a long-standing interest in predicting the DBT curve from the chemistry and processing route of a steel.'),
('p', 'Two families of predictive tools exist. The first is the empirical description of the curve itself. The hyperbolic-tangent (tanh) equation proposed by Oldfield [@oldfield1975] '
      'and its Boltzmann-sigmoid equivalent describe the impact energy with four parameters: the lower-shelf energy (LSE), the upper-shelf energy (USE), the temperature of the '
      'mid-transition (<i>T</i><sub>t</sub>) and the width of the transition region; the transition temperature is denoted <i>T</i><sub>t</sub> here to avoid confusion with the reference temperature <i>T</i><sub>0</sub> of the master curve. The master-curve concept of Wallin [@wallin2002] plays the same role for fracture toughness, and '
      'physically based correlations relate transition-temperature shifts to composition and irradiation dose in reactor pressure vessel steels [@eason2013]. Ahmedabadi [@ahmedabadi2024] recently '
      'compared such sigmoidal models, including a saturation-modified logistic function, with supervised machine-learning regressors on normalised impact energy and '
      'concluded that the mathematical models capture the trends of individual data sets more faithfully than generic learners. The second family is data-driven '
      'modelling. Neural networks have been used in ferrous metallurgy since the 1990s, for example to model the impact toughness of C–Mn weld metals [@bhadeshia1999,bhadeshia1995] and the '
      'Charpy transition temperature of irradiated ferritic–martensitic steels [@cottrell2007,windsor2008] and the embrittlement of reactor pressure vessel steels [@mathew2018,ferreno2022], and the materials-informatics paradigm has since provided general-purpose descriptor sets and tree-based '
      'learners [@agrawal2016,ward2016]. Applications to steels include property prediction from industrial big data [@guo2019], physical-metallurgy-guided design of ultrahigh-strength stainless '
      'steels [@shen2019], modelling of medium-carbon steels [@reddy2009] and microalloyed pipeline steels [@datta2013,pattanayak2015].'),
('p', 'Both families have limitations when the goal is a complete DBT curve for a new steel. Fitting the four-parameter tanh equation requires a full curve for every '
      'condition; the many literature reports that give the impact energy at one or two temperatures can only be used if the analyst fixes some of the parameters '
      'from physical assumptions, as in the reduced-parameter algorithms of Switzner et al. [@switzner2023]. Generic regressors that take the test temperature as one more input have '
      'no knowledge of the sigmoidal shape; with a few hundred training points they produce piecewise-constant or oscillating curves, and their upper shelf is not defined. '
      'Moreover, the performance reported for such models is almost always obtained by random cross-validation, in which measurements of the same alloy appear in both the '
      'training and the test sets. It is well documented in materials informatics that random splits overestimate the ability of a model to extrapolate to new compositions '
      '[@zhang2018,meredig2018,xiong2020], and this concern is particularly relevant for small, heterogeneous Charpy databases.'),
('p', 'Physics-informed machine learning offers a middle route: instead of learning the target directly, the learner predicts the parameters of a known constitutive '
      'equation and is trained through that equation [@raissi2019,karniadakis2021]. For the DBT curve this means predicting (LSE, USE, <i>T</i><sub>t</sub>, width) from composition and heat treatment '
      'and computing the impact energy at the requested temperature with the tanh equation. Every training point, including single-temperature measurements, then '
      'contributes to the four parameters, and the prediction is a smooth sigmoid by construction. Instead of the analyst fixing parameters, the learner supplies them from '
      'the composition and heat treatment. The neural-network studies of the Charpy transition of irradiated steels [@cottrell2007,windsor2008] predicted a single '
      'scalar, the transition temperature or its shift; to the author\'s knowledge the whole curve has not been learned end-to-end in this way, nor has such a '
      'formulation been compared systematically with direct regression on a literature-scale Charpy database or tested on independently produced steels.'),
('p', 'The objectives of this work are therefore (i) to compile a literature database of Charpy impact energies of steels with full composition and '
      'heat-treatment descriptors; (ii) to compare direct machine-learning regression (gradient-boosted trees with and without a monotonicity constraint on the test temperature, and a neural network) with three empirical-equation-based '
      'formulations, namely the classical two-stage curve-fit-then-regress procedure and two end-to-end tanh-constrained learners (a neural network and a gradient-boosted '
      'tree model with a custom objective); (iii) to evaluate the models under random, unseen-heat-treatment and unseen-alloy cross-validation, and with a curve-shape '
      'metric; (iv) to validate the models on four in-house cast Fe–C–Mn–(Cr, Mo, Ni) steels that are not part of the database, and to examine whether a small amount '
      'of in-house data can be used for fine-tuning; and (v) to interpret the models with SHAP [@lundberg2017,lundberg2020] in terms of the physical curve parameters.'),

# ----------------------------------------------------------------------------- 2 Materials and Methods
('h1', '2. Materials and Methods'),
('h2', '2.1. Literature Database'),
('p', 'Charpy impact data were collected from 43 publications on wrought, forged and cast steels tested with V-notch specimens. Each record contains the chemical '
      'composition (17 elements), the product form (rolled, forged or cast), the austenitising or normalising temperature, holding time and cooling rate, the tempering '
      'temperature, holding time and cooling rate, the test temperature and the absorbed energy. Cooling rates are coded as water quenching (200), oil quenching (50), '
      'air cooling (1) and furnace cooling (0.02); the codes are rounded order-of-magnitude estimates of the cooling rate in °C/s and enter the models both directly and as '
      'logarithms. The absence of tempering is coded as a tempering temperature of zero. Specimen size, notch orientation and sampling position are not reported '
      'consistently in the sources and were not recorded; most sources state full-size 10 × 10 × 55 mm specimens, but sub-size specimens and different notch orientations '
      'cannot be excluded (Section 4.4). One source reports Fe–Cr–B wear-resistant cast alloys with 0.5–4.75 wt.% boron; because the fracture behaviour of these '
      'boride-containing alloys differs from that of steels, the 18 corresponding measurements were excluded. The final database contains 616 measurements of 61 steels '
      'in 174 distinct alloy/heat-treatment conditions. Table 1 summarises the ranges of the descriptors, and Table 2 groups the steels into classes. The source '
      'publications are listed in Table S1 (Supplementary Materials).'),
('table', 'Table 1. Ranges (minimum–maximum) and medians of the descriptors in the literature database (616 measurements). Compositions in wt.%.',
    ['Descriptor', 'Range', 'Median', 'Descriptor', 'Range', 'Median'],
    [['C', '0.005–1.01', '0.17', 'B', '0–0.008', '0'],
     ['Mn', '0.01–6.50', '0.84', 'N', '0–0.040', '0'],
     ['Si', '0–1.60', '0.26', 'P', '0–0.065', '0.005'],
     ['Cr', '0–11.56', '0.60', 'S', '0–0.074', '0.004'],
     ['Ni', '0–6.00', '0.09', 'Al', '0–0.30', '0'],
     ['Mo', '0–1.48', '0.03', 'W', '0–2.00', '0'],
     ['V', '0–0.25', '0', 'Co', '0–1.47', '0'],
     ['Cu', '0–2.10', '0', 'Austenitising T (°C)', '800–1200', '900'],
     ['Nb', '0–0.073', '0', 'Austenitising time (min)', '5–480', '40'],
     ['Ti', '0–0.178', '0', 'Tempering T (°C)', 'none, 200–860', '600'],
     ['Test temperature (°C)', '−200 to 288', '−25', 'Tempering time (min)', '0–10,080', '60'],
     ['Impact energy (J)', '0.9–361', '53.6', 'Product form', 'rolled 424 / forged 127 / cast 65', '–']],
    None, [1500, 1300, 800, 1900, 1500, 857], 16),
('table', 'Table 2. Steel classes in the database (statistics refer to all measurements of the class).',
    ['Class (criterion)', 'Measurements', 'Conditions', 'Steels', 'Median CVN (J)', 'Test T range (°C)'],
    [['Low-alloy (2.5–7 wt.% alloy)', '271', '94', '28', '71.5', '−197 to 288'],
     ['Plain carbon / HSLA (<2.5 wt.%)', '173', '42', '15', '49.2', '−197 to 200'],
     ['9–12Cr heat-resistant (Cr ≥ 7)', '77', '14', '10', '78.8', '−197 to 199'],
     ['Medium-Mn (Mn ≥ 3)', '52', '11', '5', '56.3', '−200 to 25'],
     ['High-carbon (C ≥ 0.45)', '43', '13', '3', '10.8', '−40 to 80']],
    None, [2600, 1050, 1000, 800, 1100, 1307], 16),
('p', 'A condition is defined as a unique combination of steel and heat treatment, i.e., one DBT curve. Of the 174 conditions, 95 were tested at a single temperature '
      '(97 measurements), 17 at two or three temperatures and 62 at four or more temperatures (472 measurements). Replicate specimens tested at the same temperature '
      'differ by 3 J (median), so the scatter within a condition is small compared with the differences between conditions.'),

('h2', '2.2. Empirical Description of the DBT Curve'),
('p', 'The impact energy <i>E</i> at test temperature <i>T</i> was described by the tanh equation [@oldfield1975] written in terms of the shelf energies:'),
('eq', '<i>E</i>(<i>T</i>) = (USE + LSE)/2 + [(USE − LSE)/2]·tanh[(<i>T</i> − <i>T</i><sub>t</sub>)/<i>C</i><sub>w</sub>],', 1),
('pn', 'where LSE and USE are the lower- and upper-shelf energies, <i>T</i><sub>t</sub> is the temperature at which <i>E</i> = (USE + LSE)/2, taken here as the DBT '
       'temperature (DBTT), and <i>C</i><sub>w</sub> is a width parameter; the transition region extends over roughly ±2<i>C</i><sub>w</sub> around <i>T</i><sub>t</sub>. '
       'Equation (1) was fitted by bounded non-linear least squares to every condition with at least four test temperatures (62 curves). The bounds were '
       '0 ≤ LSE ≤ min(<i>E</i>), 0 ≤ USE ≤ 1.5 max(<i>E</i>), −300 ≤ <i>T</i><sub>t</sub> ≤ 400 °C and 3 ≤ <i>C</i><sub>w</sub> ≤ 300 °C.'),

('h2', '2.3. Descriptors'),
('p', 'The 24 primary descriptors are the 17 elemental concentrations and the 7 processing variables of Table 1 (product form, austenitising temperature, time and '
      'cooling-rate code, tempering temperature, time and cooling-rate code). Eight physics-derived descriptors were added because tree models and small neural networks '
      'cannot easily construct such combinations from a few hundred samples: the IIW carbon equivalent CE = C + Mn/6 + (Cr + Mo + V)/5 + (Ni + Cu)/15, the Ito–Bessyo '
      'parameter P<sub>cm</sub>, the total alloy content, the logarithms of the two cooling-rate codes, the Hollomon–Jaffe tempering parameter [@hollomon1945]'),
('eq', '<i>P</i><sub>HJ</sub> = <i>T</i><sub>temp</sub> (K) · [20 + log<sub>10</sub> <i>t</i><sub>temp</sub> (h)] / 1000,', 2),
('pn', 'set to zero for untempered conditions, and the Andrews estimates [@andrews1965] of the martensite-start and Ac<sub>3</sub> temperatures, e.g.'),
('eq', '<i>M</i><sub>s</sub> (°C) = 539 − 423 C − 30.4 Mn − 12.1 Cr − 17.7 Ni − 7.5 Mo.', 3),
('pn', 'The Andrews formulae were derived for steels with less than about 0.6 wt.% C and moderate alloy contents and are applied here outside that range for the high-carbon '
       'and medium-Mn steels; they are used as descriptors that rank the steels by hardenability and transformation temperature, not as accurate predictions of '
       '<i>M</i><sub>s</sub>. For the direct models the test temperature is a 33rd input; for the equation-based models it enters only through Equation (1).'),

('h2', '2.4. Models'),
('p', 'Six models were built with a common interface (Table 3). Three are conventional regressors of the impact energy and three are based on Equation (1).'),
('table', 'Table 3. Models compared in this study. Hyper-parameters of all models except the two-stage regressors were tuned with Optuna [@akiba2019] (Section 2.4); all settings are listed in the public code repository.',
    ['Model', 'Type', 'Output', 'Description'],
    [['XGB-direct', 'direct', '<i>E</i>', 'Gradient-boosted regression trees [@chen2016], 33 inputs including the test temperature'],
     ['XGB-direct-mono', 'direct', '<i>E</i>', 'As XGB-direct, with a monotonically increasing constraint on the test temperature (xgboost monotone_constraints)'],
     ['NN-direct', 'direct', '<i>E</i>', 'Multilayer perceptron (two hidden SiLU layers; width and dropout tuned), ensemble of 5, output bounded to 0–400 J by a sigmoid'],
     ['Two-stage', 'equation', 'LSE, USE, <i>T</i><sub>t</sub>, <i>C</i><sub>w</sub>', 'Equation (1) fitted to each of the 62 curves with ≥ 4 temperatures; four independent XGBoost regressors map descriptors to the fitted parameters (the procedure of [@ahmedabadi2024])'],
     ['tanh-NN', 'equation, end-to-end', 'LSE, USE, <i>T</i><sub>t</sub>, <i>C</i><sub>w</sub>', 'Multilayer perceptron with a four-output head; parameters bounded by sigmoids; trained by minimising the squared error of Equation (1) at the measured temperatures; ensemble of 5'],
     ['tanh-XGB', 'equation, end-to-end', 'LSE, USE, <i>T</i><sub>t</sub>, <i>C</i><sub>w</sub>', 'XGBoost with vector-valued leaves (multi-output trees) and a custom objective: gradient and Gauss–Newton Hessian of the squared error of Equation (1) with respect to the four raw outputs']],
    None, [1150, 1350, 1500, 3857], 16),
('p', 'In the end-to-end models the four raw network or tree outputs <i>r</i><sub>1</sub>…<i>r</i><sub>4</sub> are mapped to physically admissible parameters, '
      '0 ≤ LSE < USE and <i>C</i><sub>w</sub> > 0, through bounded monotonic transformations (sigmoid or softplus), and the loss is'),
('eq', '<i>L</i> = (1/<i>N</i>) Σ<sub><i>i</i></sub> [<i>E</i>(<i>T</i><sub><i>i</i></sub>; <i>g</i>(<b>x</b><sub><i>i</i></sub>)) − <i>E</i><sub><i>i</i></sub>]<sup>2</sup>,', 4),
('pn', 'where <i>g</i>(<b>x</b>) denotes the mapping from the descriptors of a condition to the four curve parameters and the sum runs over all measurements, including '
       'those of conditions tested at a single temperature. A single measurement cannot identify four curve parameters on its own; such conditions constrain the '
       'curve only through the parameters that the model shares between steels of similar composition and heat treatment, which is the intended mechanism of the '
       'end-to-end formulation (the effect of removing these conditions is examined in Table S4). For tanh-XGB the per-sample gradient and Hessian required by the boosting algorithm were derived analytically '
       'as <i>g<sub>k</sub></i> = 2<i>r</i> ∂<i>E</i>/∂<i>r<sub>k</sub></i> and <i>h<sub>k</sub></i> = 2(∂<i>E</i>/∂<i>r<sub>k</sub></i>)<sup>2</sup> + ε, with <i>r</i> the '
       'residual, which corresponds to a Gauss–Newton approximation and keeps the Hessian positive. The neural networks were implemented in PyTorch [@paszke2019] (AdamW with '
       'cosine learning-rate decay, full-batch training, inputs standardised and clipped to ±3 standard deviations, ensembles of five networks with different seeds); '
       'the tree models in XGBoost 3.2 and scikit-learn [@pedregosa2011]. To avoid an unequal comparison, the hyper-parameters of every model except the two-stage '
       'regressors were selected with the same Optuna procedure, maximising the mean R<sup>2</sup> of the two grouped cross-validation schemes described below '
       '(single repetition): 60 trials for XGB-direct and XGB-direct-mono (number of trees, learning rate, depth, minimum child weight, subsampling, L1/L2 '
       'regularisation and γ), 30 trials for tanh-XGB (the same quantities), NN-direct and tanh-NN (hidden width 32–128, dropout 0–0.3, learning rate, weight decay and '
       'number of epochs). The selected settings were: XGB-direct, 1600 trees, learning rate 0.011, depth 5, minimum child weight 16, λ = 0.47, column/row subsampling '
       '0.34/0.52; XGB-direct-mono, 300 trees, learning rate 0.024, depth 6, λ = 19; tanh-XGB, 900 rounds, learning rate 0.0079, depth 3, minimum child weight 11, '
       'λ = 13, column/row subsampling 0.56/0.50; NN-direct, 2 × 64 units, dropout 0.02, learning rate 0.0097, weight decay 5 × 10<sup>−4</sup>, 2500 epochs; tanh-NN, '
       '2 × 128 units, dropout 0.13, learning rate 0.0098, weight decay 5 × 10<sup>−4</sup>, 1000 epochs. The two-stage regressors were kept at fixed settings '
       '(300 trees, depth 3) because only 62 curves are available for their training.'),

('h2', '2.5. Evaluation'),
('p', 'Three five-fold cross-validation schemes were applied, each repeated three times with different fold assignments: (i) random splitting of the 616 measurements, '
      'which tests interpolation along known curves; (ii) grouping by condition, so that a heat-treatment condition is predicted from other conditions, possibly of the '
      'same steel; and (iii) grouping by steel, so that all conditions of a steel are held out together and the model must extrapolate to an unseen composition [@zhang2018,meredig2018,xiong2020]. '
      'The coefficient of determination R<sup>2</sup>, the mean absolute error (MAE) and the root-mean-square error (RMSE) of the impact energy were computed on the '
      'pooled out-of-fold predictions. The curve shape was quantified on every held-out condition by predicting the energy on a 5 °C grid from −200 to 300 °C and '
      'computing the non-monotonicity index'),
('eq', 'NMI = Σ<sub><i>j</i></sub> max(0, −Δ<i>E<sub>j</sub></i>) / (<i>E</i><sub>max</sub> − <i>E</i><sub>min</sub>),', 5),
('pn', 'i.e., the total decrease of the predicted energy with increasing temperature relative to the range of the curve (NMI = 0 for a monotonically increasing curve), '
       'and the R<sup>2</sup> of a tanh fit to the predicted grid. The tuning objective used a single repetition of schemes (ii) and (iii), which overlaps with the folds '
       'used for reporting. To quantify the resulting selection bias, a nested cross-validation was performed for the three tree models, in which the '
       'hyper-parameters are re-selected with Optuna (20 trials) on an inner five-fold grouped split inside every outer fold (Table S5); the neural networks were '
       'excluded from the nested procedure because of their training time.'),

('h2', '2.6. In-House Cast Steels and Fine-Tuning'),
('p', 'Four Fe–C–Mn–Si steels with additions of Ni, Cr and Mo (Table 4) were prepared from electrolytic iron and the alloying additions and melted in a conventional '
      'induction furnace in ambient atmosphere while argon gas was blown onto the melt. The melts were poured into a sand step mould 160 mm high and 200 mm wide with steps of '
      'nominally 10, 20 and 40 mm thickness. The ingate was located at the 40 mm step, so that the mould filled in the order 10, 20 and 40 mm, and the 40 mm step was open to '
      'the air. Charpy specimens were machined from the thinnest step, whose as-cast thickness was 12 mm. The specimens were austenitised at 900 °C for 4 h in a box furnace, water quenched and tempered at '
      '600 °C for 4 h in the same furnace. Standard 10 × 10 × 55 mm V-notch specimens according to ASTM E23 [@astm2018] were tested in duplicate at 25, 0, '
      '−20 and −40 °C. None of these steels, nor any measurement from the same laboratory, is contained in the literature database, so they constitute a true external '
      'test in which the product form (cast), the long austenitising and tempering times and the laboratory are all new to the models. Only the six elements of Table 4 '
      'were analysed; the remaining elements of the descriptor set (in particular P, S, N and Al) were not determined and were entered as zero, and the deoxidation '
      'practice and pouring temperature were not recorded. No microstructural, hardness or fractographic characterisation was performed. These gaps are discussed '
      'in Section 4.4.'),
('table', 'Table 4. Composition (wt.%, balance Fe) and measured Charpy impact energy of the in-house cast steels after 900 °C/4 h water quenching and 600 °C/4 h tempering (mean of two specimens ± half their difference, J).',
    ['Steel', 'C', 'Mn', 'Si', 'Cr', 'Mo', 'Ni', '25 °C', '0 °C', '−20 °C', '−40 °C'],
    [['MSN', '0.24', '1.17', '0.35', '–', '–', '0.36', '211 ± 1', '139 ± 4', '99 ± 1', '29 ± 7'],
     ['MSNN', '0.18', '1.19', '0.32', '–', '–', '0.72', '188 ± 5', '145 ± 11', '116 ± 25', '35 ± 15'],
     ['MSCM', '0.19', '1.30', '0.42', '0.29', '0.12', '–', '223 ± 4', '178 ± 15', '46 ± 3', '27 ± 2'],
     ['MSCMN', '0.23', '1.45', '0.51', '0.35', '0.11', '0.39', '194 ± 18', '149 ± 25', '33 ± 5', '31 ± 12']],
    None, [800, 500, 500, 500, 500, 500, 500, 1000, 1000, 1000, 1057], 16),
('p', 'Two uses of the cast data were examined. First, every model trained on the complete literature database predicted the four DBT curves without any information '
      'from the cast steels. Second, a leave-one-cast-alloy-out protocol was used to test fine-tuning [@jha2019,yamada2019]: the literature-trained model was adapted with the '
      'measurements of three cast steels (24 points) and evaluated on the fourth. For the neural networks all weights were re-trained for 300 epochs at a learning rate '
      'of 5 × 10<sup>−4</sup> on the loss <i>L</i><sub>lit</sub> + λ<i>L</i><sub>cast</sub> with λ = 5, where the literature term acts as a replay buffer against '
      'catastrophic forgetting; for the tree models 300 additional boosting rounds were added to the trained booster on the union of the literature data and the '
      'five-fold up-weighted cast data. A linear re-calibration of the literature prediction, <i>E</i> = <i>a</i> + <i>b</i>·<i>E</i><sub>pred</sub> fitted on the three '
      'training steels, served as the simplest possible adaptation baseline.'),

('h2', '2.7. Interpretation'),
('p', 'Shapley additive explanations (SHAP) [@lundberg2017,lundberg2020] were computed for the two tree models trained on the full database. For XGB-direct the exact TreeExplainer [@lundberg2020] '
      'gives the contribution of every descriptor, including the test temperature, to the predicted impact energy of each measurement. For tanh-XGB the interest lies '
      'in the curve parameters rather than in the energy, so the model-agnostic KernelExplainer was applied to the mapping from descriptors to <i>T</i><sub>t</sub> and USE '
      'on the 174 unique conditions with a 25-centroid k-means background and 400 samples per explanation. In addition, Spearman rank correlations between the descriptors and the fitted curve '
      'parameters of the 62 complete curves were computed as a model-free reference.'),

('h2', '2.8. Software and Use of AI Assistance'),
('p', 'The analysis was performed in Python with XGBoost 3.2 [@chen2016], PyTorch [@paszke2019], scikit-learn [@pedregosa2011], Optuna [@akiba2019] and the SHAP '
      'library [@lundberg2020]; the complete code and data are available (Data Availability Statement). The generative AI assistant Claude (Anthropic) was used during '
      'this work to assist with writing and debugging the analysis code and with drafting and editing the manuscript text. All analyses were executed by the author, '
      'and all numerical values, figures and interpretations were checked by the author, who takes full responsibility for the content.'),

# ----------------------------------------------------------------------------- 3 Results
('h1', '3. Results'),
('h2', '3.1. Database Characteristics'),
('p', 'Figure 1 shows all measurements against the test temperature by steel class. The database spans the whole transition from lower shelves of a few joules to '
      'upper shelves above 300 J, but it is strongly heterogeneous: the high-carbon spring steels never exceed 41 J, whereas low-alloy and HSLA steels tempered above '
      '600 °C reach 200–360 J. The impact energy at a given temperature therefore depends far more on the steel than on the temperature, which is why the overall '
      'Spearman correlation between test temperature and energy is only +0.34 and why the composition–energy correlations are weak when computed on all data '
      '(|ρ| ≤ 0.29). Restricting the correlation to the upper-shelf band (0–40 °C) reveals the expected ordering: carbon (ρ = −0.58), the tempering temperature '
      '(+0.57), nitrogen (+0.46), the Andrews M<sub>s</sub> temperature (+0.45) and the austenitising temperature (+0.41).'),
('fig', F('eda_fig_by_class.png'), '<b>Figure 1.</b> Literature database: (<b>a</b>) Charpy impact energy of all 616 measurements versus test temperature, coloured by steel class; (<b>b</b>) distribution of the impact energy per class.'),
('p', 'Among the descriptors themselves, the only strong collinearities are those expected from metallurgical practice: tempering temperature and Hollomon–Jaffe '
      'parameter (ρ = +0.99), P and S (+0.71, steel cleanliness), Ni and Cu (+0.67, Cu-precipitation-strengthened steels), carbon and M<sub>s</sub> (−0.63), carbon '
      'equivalent and M<sub>s</sub> (−0.62) and Mn and Cr (−0.58, medium-Mn versus 9–12Cr steels). Ni, Cu, Cr and W occur almost exclusively in particular steel '
      'classes, so their individual effects cannot be separated from the class effect by any model trained on this database. Cast products have somewhat higher '
      'carbon contents than wrought products (ρ = −0.32 with cast coded as 0).'),
('p', 'Equation (1) describes the 62 complete curves well (median R<sup>2</sup> = 0.993, minimum 0.94); examples are shown in Figure 2. The fitted parameters range from '
      '0 to 52 J for LSE, from 5 to 398 J for USE, from −165 to +80 °C for <i>T</i><sub>t</sub> and from 3 to 300 °C for <i>C</i><sub>w</sub>. USE and <i>T</i><sub>t</sub> are '
      'negatively correlated (ρ = −0.44): steels with a high upper shelf also tend to have a low transition temperature. Figure 3 and Table 5 relate the fitted '
      'parameters to the descriptors. Carbon, M<sub>s</sub> and nitrogen are significantly correlated with both <i>T</i><sub>t</sub> and USE (C: +0.30 and −0.55; '
      'M<sub>s</sub>: −0.29 and +0.46; N: −0.36 and +0.37), carbon having the strongest effect on USE. Tempering, expressed either as the tempering temperature or as '
      '<i>P</i><sub>HJ</sub>, is the strongest correlate of USE (+0.60) but is not significantly related to <i>T</i><sub>t</sub> (−0.20). The grain-refining micro-alloying '
      'additions N, Nb and Al lower <i>T</i><sub>t</sub> (ρ = −0.36, −0.36, −0.31), and so does a faster cooling rate after tempering (−0.47). Nickel, whose beneficial '
      'effect on the transition temperature of ferritic steels is well established, is not significantly correlated with <i>T</i><sub>t</sub> in this database because the '
      'Ni-bearing curves are few and belong to the 9–12Cr and Cu-bearing classes. The four complete curves of cast products show a higher USE (ρ = −0.29; median 197 J '
      'versus 100 J for rolled products) and a narrower transition (+0.28) than wrought products, but this comparison rests on four curves only.'),
('fig', F('fig_curvefits.png'), '<b>Figure 2.</b> Examples of Equation (1) fitted to individual conditions of the database (twelve conditions with the largest number of test temperatures; the bracketed numbers refer to the sources in Table S1). Dotted lines mark the fitted <i>T</i><sub>t</sub>.'),
('fig', F('eda_fig_curveparams_vs_features.png'), '<b>Figure 3.</b> Fitted transition temperature <i>T</i><sub>t</sub> (top row) and upper-shelf energy USE (bottom row) of the 62 complete curves versus carbon content, Hollomon–Jaffe tempering parameter, Andrews M<sub>s</sub> temperature and nickel content. Spearman rank correlation coefficients and p-values are given in the panel titles.'),
('table', 'Table 5. Spearman rank correlation ρ between descriptors and the curve parameters fitted to the 62 complete curves (only descriptors with at least one significant correlation are listed; ** p < 0.01, * p < 0.05).',
    ['Descriptor', '<i>T</i><sub>t</sub>', 'USE', '<i>C</i><sub>w</sub>', 'LSE'],
    [['C', '+0.30 *', '−0.55 **', '+0.12', '−0.02'],
     ['Mn', '−0.19', '−0.10', '+0.27 *', '−0.06'],
     ['Tempering temperature', '−0.20', '+0.60 **', '−0.22', '+0.10'],
     ['<i>P</i><sub>HJ</sub>', '−0.16', '+0.60 **', '−0.22', '+0.11'],
     ['M<sub>s</sub> (Andrews)', '−0.29 *', '+0.46 **', '−0.14', '+0.11'],
     ['Austenitising temperature', '−0.14', '+0.39 **', '−0.12', '+0.13'],
     ['Austenitising time', '+0.24', '+0.40 **', '−0.17', '+0.32 *'],
     ['Tempering time', '+0.36 **', '+0.16', '−0.01', '+0.11'],
     ['Tempering cooling rate', '−0.47 **', '+0.08', '+0.24', '−0.17'],
     ['N', '−0.36 **', '+0.37 **', '+0.07', '−0.09'],
     ['Nb', '−0.36 **', '+0.14', '+0.13', '−0.23'],
     ['Al', '−0.31 *', '+0.23', '+0.29 *', '−0.18'],
     ['Cu', '+0.31 *', '−0.22', '−0.09', '+0.16'],
     ['W', '−0.13', '+0.33 **', '−0.19', '−0.15'],
     ['S', '0.00', '−0.26 *', '+0.05', '+0.04'],
     ['P<sub>cm</sub>', '+0.25', '−0.31 *', '+0.06', '−0.12'],
     ['Ni', '+0.21', '−0.07', '−0.03', '+0.31 *'],
     ['Mo', '+0.21', '+0.09', '−0.08', '+0.30 *'],
     ['Product form (cast = 0)', '−0.10', '−0.29 *', '+0.28 *', '−0.02']],
    None, [2857, 1250, 1250, 1250, 1250], 16),

('h2', '3.2. Cross-Validation'),
('p', 'Table 6 and Figure 4 summarise the cross-validation results. Under random splitting the five models that use all measurements reach R<sup>2</sup> = 0.76–0.91 '
      '(MAE 13–27 J); the neural networks are the most accurate interpolators, and the two-stage procedure is clearly inferior (R<sup>2</sup> = 0.58) because it discards '
      'the 112 conditions with fewer than four test temperatures and learns from 62 curves only. When whole heat-treatment conditions are held out, the tree models '
      'retain R<sup>2</sup> = 0.57–0.63 and the neural networks 0.51–0.52 (MAE 33–39 J), while the two-stage model falls to 0.30. When whole steels are held out, every '
      'model loses most of its predictive power: R<sup>2</sup> = 0.30 for XGB-direct and tanh-XGB, 0.28 for XGB-direct-mono, 0.16 for tanh-NN and 0.08 for NN-direct, '
      'with MAE = 47–55 J for all models including the two-stage procedure. The neural networks are also much less stable in this scheme (standard deviation over the '
      'three fold assignments 0.18–0.24 in R<sup>2</sup>, against 0.04–0.06 for the trees), which means that their tuned settings, selected on one fold assignment, '
      'transfer poorly to others. The nested cross-validation of the tree models (Table S5) gives unseen-steel R<sup>2</sup> = 0.34, 0.33 and 0.17 for XGB-direct, '
      'XGB-direct-mono and tanh-XGB; the selection bias of the single-level tuning is therefore negligible for the direct trees but amounts to about 0.13 for tanh-XGB, '
      'and the honest range for extrapolation to unseen steels is R<sup>2</sup> ≈ 0.2–0.3. The residual error under alloy grouping, 47–55 J, should be compared with '
      'the standard deviation of the impact energy in the database, 80 J.'),
('table', 'Table 6. Cross-validation results (mean of three repetitions; the standard deviation over repetitions is ≤ 0.06 in R<sup>2</sup> and ≤ 3 J in MAE for the tree models and the two-stage model, and up to 0.24 in R<sup>2</sup> and 8 J in MAE for the neural networks under unseen-steel splitting). NMI is the non-monotonicity index of Equation (5) averaged over the held-out conditions; tanh-R<sup>2</sup> is the median R<sup>2</sup> of a tanh fit to the predicted curves.',
    ['Scheme', 'Model', 'R<sup>2</sup>', 'MAE (J)', 'RMSE (J)', 'NMI', 'tanh-R<sup>2</sup>'],
    [['Random', 'XGB-direct', '0.796', '24.9', '36.0', '0.09', '0.985'],
     ['', 'XGB-direct-mono', '0.763', '27.2', '38.8', '0', '0.994'],
     ['', 'NN-direct', '0.907', '12.9', '24.2', '0.06', '0.996'],
     ['', 'Two-stage', '0.584', '28.2', '51.4', '0', '1'],
     ['', 'tanh-NN', '0.877', '15.1', '27.9', '0', '1'],
     ['', 'tanh-XGB', '0.839', '20.4', '31.9', '0', '1'],
     ['Unseen condition', 'XGB-direct', '0.598', '36.7', '50.5', '0.09', '0.985'],
     ['', 'XGB-direct-mono', '0.572', '38.9', '52.1', '0', '0.994'],
     ['', 'NN-direct', '0.523', '34.2', '55.0', '0.05', '0.995'],
     ['', 'Two-stage', '0.300', '44.6', '66.6', '0', '1'],
     ['', 'tanh-NN', '0.512', '34.7', '55.6', '0', '1'],
     ['', 'tanh-XGB', '0.633', '33.4', '48.2', '0', '1'],
     ['Unseen steel', 'XGB-direct', '0.299', '50.8', '66.6', '0.09', '0.990'],
     ['', 'XGB-direct-mono', '0.282', '51.8', '67.5', '0', '0.995'],
     ['', 'NN-direct', '0.076', '51.4', '76.1', '0.04', '0.995'],
     ['', 'Two-stage', '0.070', '55.2', '76.7', '0', '1'],
     ['', 'tanh-NN', '0.158', '50.3', '72.8', '0', '1'],
     ['', 'tanh-XGB', '0.299', '47.4', '66.7', '0', '1']],
    None, [1700, 1500, 900, 900, 950, 850, 1057], 16),
('fig', F('fig_cv_bars.png'), '<b>Figure 4.</b> Cross-validation metrics of the six models under random, unseen-condition and unseen-steel splitting: R<sup>2</sup> (left), MAE (middle) and non-monotonicity index (right). Error bars are the standard deviation over three repetitions. The NMI of the equation-based models and of XGB-direct-mono is zero by construction.'),
('p', 'The curve-shape metrics separate the model families irrespective of accuracy. The unconstrained direct XGBoost model predicts curves with NMI = 0.09, i.e., the '
      'cumulative decrease of energy with increasing temperature amounts to about one tenth of the curve height. The monotonicity constraint removes this defect '
      'completely (NMI = 0) at a cost of 0.02–0.03 in R<sup>2</sup>, but the curves of XGB-direct-mono remain staircases whose upper shelf is defined only by the highest '
      'test temperature in the training data (Figure 5). The direct neural network is nearly monotonic (NMI = 0.04–0.06) because it is a smooth function of the '
      'temperature input, but its curves are not sigmoids and have no defined upper shelf either. The three equation-based models return exact tanh curves. Figure 5 '
      'illustrates these differences on nine steels held out under alloy grouping. The examples also show the origin of the residual error: for the carbon, Q&T, P92 and '
      '3Mn–Si–Ni steels the transition temperature is predicted within about 30 °C, whereas the upper shelf is frequently under- or over-estimated by 50–100 J, and for '
      'two steels with unusually low upper shelves (AISI 1018 and the low-carbon martensitic steel) all models predict the typical shelf of their composition class. '
      'When the same nine steels are included in the training data, the final models reproduce them with MAE = 4–48 J (XGB-direct), 1–9 J (tanh-NN) and 4–33 J '
      '(tanh-XGB; Appendix A, Figure A1), so the errors in Figure 5 arise from the extrapolation to unseen compositions and not from a lack of model capacity.'),
('fig', F('fig_heldout_curves_unseen_alloy.png'), '<b>Figure 5.</b> Predicted DBT curves for nine steels that were completely excluded from training (unseen-steel scheme), compared with the measurements (symbols). Blue: XGB-direct; grey: XGB-direct-mono; green: two-stage; orange: tanh-NN; red: tanh-XGB. The bracketed numbers refer to the sources in Table S1.'),

('h2', '3.3. Interpretation with SHAP'),
('p', 'Figure 6a shows the SHAP summary of XGB-direct. The test temperature is the most influential input (mean |SHAP| = 30 J), followed by the tempering temperature '
      '(12 J), the Hollomon–Jaffe parameter (9 J), M<sub>s</sub> (8 J), P<sub>cm</sub> (6.5 J) and carbon (6 J); all other descriptors contribute less than 5 J on average. '
      'Because the test temperature dominates, the influence of composition on the shape of the curve cannot be read from this plot. The parameter-wise explanations of '
      'tanh-XGB (Figure 6b,c) resolve this. For the transition temperature, carbon dominates (mean |SHAP| = 8 °C; the dependence is monotonic and nearly linear, from '
      'about −15 °C at 0.05 wt.% C to +10 °C at 0.5 wt.% C), followed by M<sub>s</sub> (5 °C), <i>P</i><sub>HJ</sub> (5 °C), the tempering temperature (4.6 °C), the '
      'product form (3.4 °C), nitrogen and manganese (2 °C). For the upper-shelf energy the order is carbon (17 J), <i>P</i><sub>HJ</sub> (12 J), tempering temperature '
      '(10 J), M<sub>s</sub> (9 J), copper (4 J) and nitrogen (4 J). The dependence of USE on the tempering temperature is a step: tempering above ~600 °C adds 20–45 J to '
      'the upper shelf relative to lower tempering temperatures. The contribution of the product form should be read with caution, because most cast conditions in the '
      'database are high-carbon or untempered steels tested at a single temperature, so that the form is confounded with these attributes. With that reservation, the '
      'trends are consistent with the model-free correlations of Table 5 and with metallurgical experience, and they show that the end-to-end model has learned a '
      'physically sensible mapping from composition and heat treatment to the curve parameters.'),
('fig', F('fig_shap_composite.png'), '<b>Figure 6.</b> SHAP summary plots: (<b>a</b>) XGB-direct, contribution of each input to the predicted impact energy of the 616 measurements; (<b>b</b>) tanh-XGB, contribution of each descriptor to the predicted transition temperature <i>T</i><sub>t</sub>; (<b>c</b>) tanh-XGB, contribution to the predicted upper-shelf energy. Each point is one measurement (a) or one alloy/heat-treatment condition (b, c); the colour gives the descriptor value.'),

('h2', '3.4. Prediction of the In-House Cast Steels'),
('p', 'Figure 7 compares the DBT curves predicted from the literature database alone with the measurements of the four cast steels, and Table 7 lists the errors. In '
      'composition the cast steels lie in the densely populated centre of the database, between the plain-carbon/HSLA and low-alloy classes; their nearest neighbours in '
      'standardised composition space are wrought quenched-and-tempered steels with 0.17–0.21 wt.% C and 1.0–1.4 wt.% Mn and the cast G18CrMo2-6 steel. The combination '
      'of a cast product form with quenching and tempering is, however, represented in the database by that single steel. No model predicts the cast steels well from '
      'the literature alone: the mean absolute errors range from 39 J (two-stage) and 46 J (tanh-XGB) to 54–58 J (direct trees), 59 J (tanh-NN) and 67 J (NN-direct), '
      'and every model under-estimates the measured energy at 25 °C (188–223 J) of all four steels, by 46–64 J (two-stage), 55–93 J (tanh-NN, tanh-XGB) and 67–119 J '
      '(direct models). The equation-based models place the transition between −5 and +8 °C, whereas the measurements indicate a transition around −20 to −10 °C, and '
      'predict upper shelves of 145–167 J (tanh-XGB), 156–164 J (tanh-NN) and 183–226 J (two-stage). The ±1 standard-deviation bands of the neural-network ensembles '
      '(Figure 7) are narrow (typically ±10–20 J) and do not contain the measurements, i.e., the ensembles are confidently wrong. The ranking of the models in this test is '
      'almost the reverse of the cross-validation ranking: the two-stage model, the least accurate model in cross-validation, is the best here because its upper shelf is '
      'learned only from complete curves of well-characterised steels, most of which have high shelves. The sensitivity of the tanh-XGB prediction to the assumed heat '
      'treatment is considerable: with the as-quenched condition the error rises to 67–91 J, and tempering at 650 °C gives slightly lower errors than the actual 600 °C, '
      'so the correct specification of the heat treatment is as important as the composition.'),
('fig', F('fig_cast_validation.png'), '<b>Figure 7.</b> DBT curves of the four in-house cast steels predicted by the six models trained on the literature database only (no information from the cast steels), compared with the measured Charpy energies (symbols). The shaded bands are ±1 standard deviation of the five ensemble members of NN-direct (cyan) and tanh-NN (orange).'),
('table', 'Table 7. Errors of the literature-trained models on the cast steels (mean over the four steels, eight measurements each) and results of the leave-one-cast-alloy-out experiment (linear re-calibration and fine-tuning with the other three steels).',
    ['Model', 'Literature only MAE (J)', 'Literature only R<sup>2</sup>', 'Linear re-calibration MAE (J)', 'Fine-tuned MAE (J)', 'Fine-tuned R<sup>2</sup>'],
    [['Two-stage', '39.4', '0.57', '–', '–', '–'],
     ['tanh-XGB', '46.1', '0.35', '29.6', '26.3', '0.74'],
     ['XGB-direct', '54.4', '0.28', '22.6', '31.6', '0.74'],
     ['XGB-direct-mono', '58.0', '0.10', '41.8', '32.7', '0.71'],
     ['tanh-NN', '59.0', '0.05', '24.8', '20.1', '0.87'],
     ['NN-direct', '67.0', '−0.25', '18.8', '19.0', '0.90']],
    None, [1500, 1300, 1150, 1400, 1250, 1257], 16),
('p', 'Adapting the models with the data of three cast steels improves the prediction of the fourth for all of them (Table 7 and Figure 8). The fine-tuned neural '
      'networks reach MAE = 19–20 J and R<sup>2</sup> = 0.87–0.90 on the held-out steel (15–24 J for the individual steels), the tree models 26–33 J. This is close to the '
      'scatter between duplicate specimens of the cast steels (mean absolute difference between the two specimens 19 J, up to 51 J at −20 °C), so the adapted models '
      'cannot be distinguished further with these data. A simple linear re-calibration of the literature prediction is almost as effective for NN-direct (18.8 J) and '
      'XGB-direct (22.6 J), which indicates that the adaptation consists mainly of lifting the level of the curve. The curve parameters of the tanh-NN fine-tuned on all '
      'four steels are <i>T</i><sub>t</sub> = −15, −19, −10 and −7 °C and USE = 207, 199, 225 and 210 J for MSN, MSNN, MSCM and MSCMN, respectively; the differences '
      'between the four steels are of the order of the specimen scatter and cannot be attributed to the individual alloying additions.'),
('fig', F('fig_finetune_tanh-NN.png'), '<b>Figure 8.</b> Leave-one-cast-alloy-out adaptation of tanh-NN. For each panel the model was pre-trained on the literature database (grey), re-calibrated linearly (green) or fine-tuned (red) with the measurements of the other three cast steels, and evaluated on the steel shown (symbols).'),

# ----------------------------------------------------------------------------- 4 Discussion
('h1', '4. Discussion'),
('h2', '4.1. Direct Regression versus Equation-Constrained Learning'),
('p', 'In terms of point-wise accuracy, embedding the tanh equation in the learner costs nothing. With the same tuning budget, tanh-XGB is as accurate as or more '
      'accurate than XGB-direct under all three schemes (R<sup>2</sup> 0.84 versus 0.80, 0.63 versus 0.60 and 0.30 versus 0.30), and tanh-NN is within 0.03 of NN-direct '
      'under random and unseen-condition splitting and better under unseen-steel splitting. The constraint pays off in three further respects. First, the predicted '
      'curves are sigmoids with a defined upper shelf, lower shelf and transition temperature, so that the quantities engineers specify (a transition temperature, a '
      '27 J or 41 J temperature, a USE) can be read off directly. A monotonicity constraint on the test temperature removes the non-monotonicity of the direct tree '
      'model but not its staircase shape, and it leaves the upper shelf undefined beyond the range of the training temperatures (Figure 5). Second, the parameters can '
      'be interpreted separately, which turned SHAP from a description of the test-temperature effect into a ranking of composition and heat-treatment effects on '
      '<i>T</i><sub>t</sub> and USE (Figure 6). Third, the end-to-end formulation uses the 144 measurements from conditions with one to three test temperatures, which '
      'the two-stage procedure cannot exploit; this is why the two-stage model, although it also produces perfect sigmoids, is the least accurate model under random '
      'and unseen-condition splitting. The benefit of these additional measurements is, however, confined to interpolation: when the single-temperature conditions are '
      'removed from the database altogether, the unseen-steel accuracy of the tree models does not deteriorate (Table S4). The end-to-end tanh-constrained learner thus '
      'combines the shape guarantee of the mathematical models discussed by Ahmedabadi [@ahmedabadi2024] and Switzner et al. [@switzner2023] with the data efficiency of a '
      'generic learner.'),
('p', 'The choice of the learner inside the constraint matters less than the evaluation scheme. Once the networks are tuned with the same budget as the trees, all '
      'five learners that use the full database are within 0.12 of each other under unseen-condition splitting, and their differences under unseen-steel splitting '
      '(0.08–0.30) are of the order of the repeat-to-repeat scatter of the networks. The trees are the more stable extrapolators (standard deviation 0.04–0.06 against '
      '0.18–0.24), presumably because they partition the descriptor space into the steel classes of Table 2, whereas the networks interpolate smoothly between classes '
      'and are more sensitive to which steels fall into the training folds. The per-class errors (Table S3) show no model that is uniformly superior: the networks are '
      'much better on the high-carbon spring steels (6–11 J against 25–44 J), the trees slightly better on the plain-carbon/HSLA class, and for the low-carbon '
      'quenched-and-tempered steels that resemble the cast steels all models have errors of 52–59 J.'),
('h2', '4.2. Limits of Extrapolation and the Value of Physics-Derived Descriptors'),
('p', 'The drop from R<sup>2</sup> ≈ 0.8–0.9 (random) to ≤ 0.30 (unseen steel; 0.17–0.34 in the nested procedure) quantifies how much of the apparent accuracy of '
      'Charpy-energy models comes from memorising known curves, in line with the extrapolation studies of Meredig et al. [@meredig2018] and Xiong et al. [@xiong2020]. '
      'The reasons are visible in the database analysis. The upper shelf spans almost two orders of magnitude (fitted USE 5–398 J) and is determined by inclusion '
      'content, microstructure and specimen geometry, none of which is fully captured by the composition and the few processing variables that publications report; '
      'several elements are confounded with steel classes; and 95 of the 174 conditions are single-temperature measurements that constrain the shelf levels only weakly. '
      'Adding descriptors that encode textbook metallurgy, the carbon equivalents, the Hollomon–Jaffe parameter and the Andrews M<sub>s</sub>, raised the unseen-steel '
      'R<sup>2</sup> of the tuned tree models from 0.19–0.24 to 0.28–0.30 (Table S2), and the Hollomon–Jaffe parameter and M<sub>s</sub> appear among the top four SHAP features of both tree '
      'models; the networks did not benefit from them. The gain is modest and of the order of the fold-to-fold scatter, so the descriptors are best regarded as a '
      'cheap and consistent, rather than a decisive, improvement, in line with the physical-metallurgy-guided strategy advocated for steels by Shen et al. [@shen2019] '
      'and Datta and Chattopadhyay [@datta2013]. A substantial improvement can only be expected from information that the present descriptors do not contain, such as '
      'specimen size, inclusion content or grain size, which are the most probable causes of the remaining shelf-level errors.'),
('h2', '4.3. External Test on the Cast Steels and Adaptation with In-House Data'),
('p', 'The external test is the most instructive and the most sobering result of this work. All six literature-trained models under-estimated the room-temperature '
      'energy of the four cast steels by 46–119 J, and the model ranking bore little relation to the cross-validation ranking. Three causes can be identified. The '
      'combination of a cast product form with quenching and tempering is represented in the database by a single steel, so that the models associate casting with '
      'the low energies of the high-carbon and untempered cast conditions that dominate the cast subset. The elements that were not analysed in the cast steels (P, S, '
      'N, Al) were entered as zero, which places them at the clean end of the database and is a source of error of unknown sign. Finally, the measured transition of '
      'the cast steels is unusually sharp (from about 30 J at −40 °C to about 200 J at 25 °C), and the ensemble bands show that none of the models considered such a '
      'curve plausible for this composition. The result confirms that a model selected by grouped cross-validation on literature data does not automatically give a '
      'useful prediction for a specific new product, and that an external test on independently produced material is necessary before such a model is used.'),
('p', 'Adaptation with 24 measurements from three related cast steels reduced the error on the fourth to the level of the specimen-to-specimen scatter (19–20 J '
      'for the networks). This result should not be over-interpreted. The four steels differ only slightly in composition (0.18–0.24 wt.% C, 1.2–1.5 wt.% Mn, small '
      'Ni, Cr and Mo additions) and share one heat treatment, so that the leave-one-out experiment is an interpolation among near-identical steels, and a linear '
      're-calibration of the literature prediction achieves almost the same error as fine-tuning. What the experiment demonstrates is that the literature-trained model '
      'provides a curve shape that a handful of in-house measurements can shift to the right level, in the spirit of transfer learning in materials informatics '
      '[@jha2019,yamada2019]; whether the adapted model extrapolates to a genuinely different in-house steel remains to be shown.'),
('h2', '4.4. Limitations'),
('p', 'The literature database is small and unbalanced, the heat-treatment descriptors are coarse (cooling rates are coded, prior microstructure is not recorded), '
      'specimen size and notch orientation are not available for every source, and 95 of the 174 conditions were tested at a single temperature. The hyper-parameters '
      'were selected on folds that overlap with the reporting folds; the nested cross-validation shows that this inflates the unseen-steel R<sup>2</sup> of tanh-XGB by '
      'about 0.13 but not that of the direct trees, and the fold-to-fold scatter of the tuned networks indicates that their settings are partly specific to the fold '
      'assignment used for tuning. The in-house test has several weaknesses that could not be remedied within this study: the highest test temperature of 25 °C does not '
      'establish the upper shelf, so that the comparison of predicted and measured shelf levels is indicative only; only two specimens were tested per temperature, '
      'with differences of up to 51 J between them; no microstructural, hardness or fractographic characterisation was performed; and P, S, N and Al were not '
      'analysed. The cast specimens were machined from a 12 mm step, so that only about 1 mm of the as-cast surface was removed on each side, and casting defects in '
      'the thin section may contribute to the scatter. Finally, Equation (1) assumes a symmetric transition; asymmetric functions such as the generalised logistic '
      'model used by Ahmedabadi [@ahmedabadi2024] could be substituted in the same end-to-end framework without changing the methodology.'),

# ----------------------------------------------------------------------------- 5 Conclusions
('h1', '5. Conclusions'),
('p', 'A literature database of 616 Charpy impact measurements on 61 steels was used to compare direct machine-learning regression of the impact energy with '
      'learners constrained by the empirical tanh equation of the DBT curve, with all models tuned under the same protocol, and the models were tested on four '
      'independently produced cast steels. The main conclusions are as follows.'),
('bullets', '', [
    'Embedding the tanh equation in the learner does not reduce the accuracy: the end-to-end tanh-constrained tree model is as accurate as or more accurate than '
    'direct XGBoost regression under random, unseen-condition and unseen-steel cross-validation, and the tanh-constrained network is within 0.03 in R<sup>2</sup> of '
    'the direct network. The two-stage procedure that fits the equation first and regresses the parameters afterwards is markedly less accurate under random and '
    'unseen-condition splitting (R<sup>2</sup> = 0.58 and 0.30) because it cannot use conditions tested at fewer than four temperatures.',
    'The equation-constrained models return smooth sigmoidal curves with a defined upper shelf, lower shelf and transition temperature. A monotonicity constraint '
    'removes the non-monotonicity of direct tree regression but not its staircase shape or its undefined upper shelf.',
    'Extrapolation to steels absent from the training data is the limiting case for all models (R<sup>2</sup> ≤ 0.30; 0.17–0.34 in nested cross-validation; MAE '
    '47–55 J against a standard deviation of 80 J). With equal tuning, trees and neural networks are of similar accuracy, the trees being the more stable '
    'extrapolators. Physics-derived descriptors improve the unseen-steel R<sup>2</sup> of the tree models by about 0.05–0.1.',
    'SHAP analysis of the equation-constrained tree model attributes the transition temperature mainly to carbon, M<sub>s</sub> and tempering, and the upper-shelf '
    'energy to carbon and the tempering parameter, in agreement with model-free rank correlations and with metallurgical experience.',
    'None of the literature-trained models predicted the DBT curves of four in-house Q&T cast Fe–C–Mn–(Cr, Mo, Ni) steels adequately (MAE 39–67 J; the room-temperature '
    'energy was under-estimated by 46–119 J), and the ranking of the models differed from the cross-validation ranking. Adapting the models with three of the cast '
    'steels reduced the error on the fourth to 19–20 J, the level of the specimen scatter, but mainly by shifting the level of the curve. An external test on '
    'independently produced material, followed by adaptation with a few in-house measurements, is therefore a necessary step before a literature-trained DBT-curve '
    'model is applied to a new product.']),

# ----------------------------------------------------------------------------- back matter
('back', '<b>Supplementary Materials:</b> The following supporting information is provided: Table S1: sources of the literature database; Table S2: effect of the physics-derived descriptors and of hyper-parameter tuning on the cross-validation results; Table S3: unseen-steel cross-validation errors by steel class; Table S4: cross-validation results without the single-temperature conditions; Table S5: nested cross-validation of the tree models.'),
('back', '<b>Author Contributions:</b> Conceptualization, methodology, software, validation, investigation, data curation, writing—original draft preparation, writing—review and editing, visualization, N.S. The author has read and agreed to the published version of the manuscript.'),
('back', '<b>Funding:</b> This research received no external funding.'),
('back', '<b>Data Availability Statement:</b> The compiled literature database (with the source list of Table S1), the in-house Charpy data, the analysis code and all model outputs are openly available in the GitHub repository [URL to be inserted], archived at Zenodo (DOI: [to be inserted]).'),
('back', '<b>Acknowledgments:</b> The generative AI assistant Claude (Anthropic) was used to assist with the development of the analysis code and with drafting and editing the text of this manuscript, as described in Section 2.8. All model results, numerical values, figures and interpretations were checked by the author, who takes full responsibility for the content.'),
('back', '<b>Conflicts of Interest:</b> The author declares no conflicts of interest.'),
('h1', 'Appendix A'),
('p', 'Figure A1 compares, for the nine steels of Figure 5, the curves predicted when the steel is excluded from training (dashed lines, as in Figure 5) with the '
      'curves of the final models trained on the complete database (solid lines). The in-sample fit of the final models over all 616 measurements is R<sup>2</sup> = 0.90 '
      '(MAE 16.9 J) for XGB-direct, 0.98 (6.7 J) for tanh-NN and 0.94 (13.7 J) for tanh-XGB. The mean absolute errors on the nine steels are listed in Table A1.'),
('fig', F('fig_final_vs_heldout.png'), '<b>Figure A1.</b> DBT curves of the nine steels of Figure 5 predicted with the steel excluded from training (dashed) and by the final models trained on all data (solid), compared with the measurements (symbols). Blue: XGB-direct; red: tanh-XGB; orange: tanh-NN.'),
('table', 'Table A1. Mean absolute error (J) on the nine steels of Figure 5 when the steel is held out of training and when it is included (final models); bracketed numbers refer to the sources in Table S1.',
    ['Steel', 'XGB-direct held out', 'XGB-direct final', 'tanh-NN held out', 'tanh-NN final', 'tanh-XGB held out', 'tanh-XGB final'],
    [['Carbon steel [S6]', '56.3', '20.8', '54.3', '5.2', '64.4', '14.9'],
     ['G18CrMo2-6 steel [S35]', '43.3', '10.6', '78.3', '8.7', '40.0', '15.2'],
     ['AISI 1018 [S11]', '52.8', '7.6', '56.2', '1.2', '33.9', '4.3'],
     ['DP590 [S11]', '48.6', '11.6', '85.8', '4.6', '23.3', '7.8'],
     ['P92 [S12]', '16.5', '8.8', '22.1', '3.3', '58.3', '7.0'],
     ['QT steel [S42]', '37.4', '8.0', '29.8', '9.3', '71.9', '8.1'],
     ['Cluster HSLA steel [S16]', '82.3', '48.1', '86.3', '5.9', '48.0', '33.4'],
     ['3Mn–Si–Ni [S11]', '21.2', '3.6', '18.2', '2.9', '19.3', '5.2'],
     ['LC martensitic steel [S28]', '77.4', '8.2', '34.5', '5.0', '65.6', '9.8']],
    None, [2157, 950, 950, 950, 950, 950, 950], 16),
]

REFERENCES = [   # (key, text); numbered automatically by first appearance of [@key] in the manuscript
 ('morris2008', 'Morris, J.W., Jr. Stronger, tougher steels. <i>Science</i> <b>2008</b>, <i>320</i>, 1022–1023.'),
 ('pineau2016', 'Pineau, A.; Benzerga, A.A.; Pardoen, T. Failure of metals I: Brittle and ductile fracture. <i>Acta Mater.</i> <b>2016</b>, <i>107</i>, 424–483.'),
 ('astm2018', 'ASTM E23-18; Standard Test Methods for Notched Bar Impact Testing of Metallic Materials. ASTM International: West Conshohocken, PA, USA, 2018.'),
 ('oldfield1975', 'Oldfield, W. Curve-fitting impact test data: A statistical procedure. <i>ASTM Stand. News</i> <b>1975</b>, <i>3</i>, 24–29.'),
 ('wallin2002', 'Wallin, K. Master curve analysis of the “Euro” fracture toughness dataset. <i>Eng. Fract. Mech.</i> <b>2002</b>, <i>69</i>, 451–481.'),
 ('eason2013', 'Eason, E.D.; Odette, G.R.; Nanstad, R.K.; Yamamoto, T. A physically-based correlation of irradiation-induced transition temperature shifts for RPV steels. <i>J. Nucl. Mater.</i> <b>2013</b>, <i>433</i>, 240–254.'),
 ('ahmedabadi2024', 'Ahmedabadi, P.M. Comparison of mathematical and supervised machine-learning models for ductile-to-brittle transition in bcc alloys. <i>Mater. Res. Express</i> <b>2024</b>, <i>11</i>, 076506.'),
 ('bhadeshia1999', 'Bhadeshia, H.K.D.H. Neural networks in materials science. <i>ISIJ Int.</i> <b>1999</b>, <i>39</i>, 966–979.'),
 ('bhadeshia1995', 'Bhadeshia, H.K.D.H.; MacKay, D.J.C.; Svensson, L.-E. Impact toughness of C–Mn steel arc welds—Bayesian neural network analysis. <i>Mater. Sci. Technol.</i> <b>1995</b>, <i>11</i>, 1046–1051.'),
 ('cottrell2007', 'Cottrell, G.A.; Kemp, R.; Bhadeshia, H.K.D.H.; Odette, G.R.; Yamamoto, T. Neural network analysis of Charpy transition temperature of irradiated low-activation martensitic steels. <i>J. Nucl. Mater.</i> <b>2007</b>, <i>367–370</i>, 603–609.'),
 ('windsor2008', 'Windsor, C.G.; Cottrell, G.A.; Kemp, R. Prediction of the Charpy transition temperature in highly irradiated ferritic steels. <i>Model. Simul. Mater. Sci. Eng.</i> <b>2008</b>, <i>16</i>, 075008.'),
 ('ferreno2022', 'Ferreño, D.; Serrano, M.; Kirk, M.; Sainz-Aja, J.A. Prediction of the transition-temperature shift using machine learning algorithms and the Plotter database. <i>Metals</i> <b>2022</b>, <i>12</i>, 186.'),
 ('switzner2023', 'Switzner, N.T.; Anderson, J.; Ali Ahmed, L.; Rosenfeld, M.; Veloo, P. Algorithms to estimate the ductile to brittle transition temperature, upper shelf energy, and their uncertainties for steel using Charpy V-notch shear area and absorbed energy data. <i>Metals</i> <b>2023</b>, <i>13</i>, 877.'),
 ('mathew2018', 'Mathew, J.; Parfitt, D.; Wilford, K.; Riddle, N.; Alamaniotis, M.; Chroneos, A.; Fitzpatrick, M.E. Reactor pressure vessel embrittlement: Insights from neural network modelling. <i>J. Nucl. Mater.</i> <b>2018</b>, <i>502</i>, 311–322.'),
 ('agrawal2016', 'Agrawal, A.; Choudhary, A. Perspective: Materials informatics and big data: Realization of the “fourth paradigm” of science in materials science. <i>APL Mater.</i> <b>2016</b>, <i>4</i>, 053208.'),
 ('ward2016', 'Ward, L.; Agrawal, A.; Choudhary, A.; Wolverton, C. A general-purpose machine learning framework for predicting properties of inorganic materials. <i>npj Comput. Mater.</i> <b>2016</b>, <i>2</i>, 16028.'),
 ('guo2019', 'Guo, S.; Yu, J.; Liu, X.; Wang, C.; Jiang, Q. A predicting model for properties of steel using the industrial big data based on machine learning. <i>Comput. Mater. Sci.</i> <b>2019</b>, <i>160</i>, 95–104.'),
 ('shen2019', 'Shen, C.; Wang, C.; Wei, X.; Li, Y.; van der Zwaag, S.; Xu, W. Physical metallurgy-guided machine learning and artificial intelligent design of ultrahigh-strength stainless steel. <i>Acta Mater.</i> <b>2019</b>, <i>179</i>, 201–214.'),
 ('reddy2009', 'Reddy, N.S.; Krishnaiah, J.; Hong, S.-G.; Lee, J.S. Modeling medium carbon steels by using artificial neural networks. <i>Mater. Sci. Eng. A</i> <b>2009</b>, <i>508</i>, 93–105.'),
 ('datta2013', 'Datta, S.; Chattopadhyay, P.P. Soft computing techniques in advancement of structural metals. <i>Int. Mater. Rev.</i> <b>2013</b>, <i>58</i>, 475–504.'),
 ('pattanayak2015', 'Pattanayak, S.; Dey, S.; Chatterjee, S.; Chowdhury, S.G.; Datta, S. Computational intelligence based designing of microalloyed pipeline steel. <i>Comput. Mater. Sci.</i> <b>2015</b>, <i>104</i>, 60–68.'),
 ('zhang2018', 'Zhang, Y.; Ling, C. A strategy to apply machine learning to small datasets in materials science. <i>npj Comput. Mater.</i> <b>2018</b>, <i>4</i>, 25.'),
 ('meredig2018', 'Meredig, B.; Antono, E.; Church, C.; Hutchinson, M.; Ling, J.; Paradiso, S.; Blaiszik, B.; Foster, I.; Gibbons, B.; Hattrick-Simpers, J.; et al. Can machine learning identify the next high-temperature superconductor? Examining extrapolation performance for materials discovery. <i>Mol. Syst. Des. Eng.</i> <b>2018</b>, <i>3</i>, 819–825.'),
 ('xiong2020', 'Xiong, Z.; Cui, Y.; Liu, Z.; Zhao, Y.; Hu, M.; Hu, J. Evaluating explorative prediction power of machine learning algorithms for materials discovery using k-fold forward cross-validation. <i>Comput. Mater. Sci.</i> <b>2020</b>, <i>171</i>, 109203.'),
 ('raissi2019', 'Raissi, M.; Perdikaris, P.; Karniadakis, G.E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. <i>J. Comput. Phys.</i> <b>2019</b>, <i>378</i>, 686–707.'),
 ('karniadakis2021', 'Karniadakis, G.E.; Kevrekidis, I.G.; Lu, L.; Perdikaris, P.; Wang, S.; Yang, L. Physics-informed machine learning. <i>Nat. Rev. Phys.</i> <b>2021</b>, <i>3</i>, 422–440.'),
 ('lundberg2017', 'Lundberg, S.M.; Lee, S.-I. A unified approach to interpreting model predictions. In <i>Advances in Neural Information Processing Systems 30</i>; Curran Associates: Red Hook, NY, USA, 2017; pp. 4765–4774.'),
 ('lundberg2020', 'Lundberg, S.M.; Erion, G.; Chen, H.; DeGrave, A.; Prutkin, J.M.; Nair, B.; Katz, R.; Himmelfarb, J.; Bansal, N.; Lee, S.-I. From local explanations to global understanding with explainable AI for trees. <i>Nat. Mach. Intell.</i> <b>2020</b>, <i>2</i>, 56–67.'),
 ('hollomon1945', 'Hollomon, J.H.; Jaffe, L.D. Time-temperature relations in tempering steel. <i>Trans. AIME</i> <b>1945</b>, <i>162</i>, 223–249.'),
 ('andrews1965', 'Andrews, K.W. Empirical formulae for the calculation of some transformation temperatures. <i>J. Iron Steel Inst.</i> <b>1965</b>, <i>203</i>, 721–727.'),
 ('chen2016', 'Chen, T.; Guestrin, C. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, 13–17 August 2016; pp. 785–794.'),
 ('akiba2019', 'Akiba, T.; Sano, S.; Yanase, T.; Ohta, T.; Koyama, M. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Anchorage, AK, USA, 4–8 August 2019; pp. 2623–2631.'),
 ('paszke2019', 'Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.; et al. PyTorch: An imperative style, high-performance deep learning library. In <i>Advances in Neural Information Processing Systems 32</i>; Curran Associates: Red Hook, NY, USA, 2019; pp. 8024–8035.'),
 ('pedregosa2011', 'Pedregosa, F.; Varoquaux, G.; Gramfort, A.; Michel, V.; Thirion, B.; Grisel, O.; Blondel, M.; Prettenhofer, P.; Weiss, R.; Dubourg, V.; et al. Scikit-learn: Machine learning in Python. <i>J. Mach. Learn. Res.</i> <b>2011</b>, <i>12</i>, 2825–2830.'),
 ('jha2019', 'Jha, D.; Choudhary, K.; Tavazza, F.; Liao, W.-K.; Choudhary, A.; Campbell, C.; Agrawal, A. Enhancing materials property prediction by leveraging computational and experimental data using deep transfer learning. <i>Nat. Commun.</i> <b>2019</b>, <i>10</i>, 5316.'),
 ('yamada2019', 'Yamada, H.; Liu, C.; Wu, S.; Koyama, Y.; Ju, S.; Shiomi, J.; Morikawa, J.; Yoshida, R. Predicting materials properties with little data using shotgun transfer learning. <i>ACS Cent. Sci.</i> <b>2019</b>, <i>5</i>, 1717–1730.'),
]
DISCLAIMER = ('<b>Disclaimer/Publisher’s Note:</b> The statements, opinions and data contained in all publications are solely those of the individual author(s) and '
              'contributor(s) and not of MDPI and/or the editor(s). MDPI and/or the editor(s) disclaim responsibility for any injury to people or property resulting from '
              'any ideas, methods, instructions or products referred to in the content.')
