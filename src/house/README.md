# House Stage 1: production configurations

`house/stage1.py` holds the two fixed Stage 1 specifications. The specification is fixed, but the coefficients are refit on each training window:

| name | prediction |
|---|---|
| `house_model_a` | Model A |
| `house_ensemble_ab_85_15` | 0.85 × Model A + 0.15 × Model B |

- **Model A (polling):** `partisan_lean + gen_margin_raw + inc_signed + partisan_lean:inc_signed`, with the `inc_signed` coefficient fixed at **+2.5**.
- **Model B (fundamentals):** `partisan_lean + fundraising_advantage + pres_party_signed:is_midterm + pres_party_signed:cpi_region_log_yoy`.
- **Fitting:** both models use equal-year WLS. Each valid training race gets weight ∝ 1 / (valid training races in its year).

The module docstring records the methodological choices: the sign conventions, which rows are used, the training windows, and why 2024 is used. The specification comes from `notebooks/house/01_reg_ensemble_compare.ipynb` and `01_reg_final_2024_holdout.ipynb`.

On 2026-11-02, Model A dropped `gb_momentum_30d_raw` for parsimony. Momentum is national and nearly collinear with `gen_margin_raw` (r = 0.92 across elections). Its coefficient swung by 2–3 points when a single training election was left out, and it bought only about 0.1 MAE, mostly in 2018. The evidence is in `01_reg_model_a_no_momentum.ipynb` and `01_reg_generic_ballot_smoothing.ipynb`. Those notebooks, and the two above, were run with momentum still in Model A, so their numbers are for the earlier spec.

Rolling out-of-sample results for the current spec, pooled over 2018–2024:

| subset | MAE | winners called |
|---|---|---|
| ex-ante competitive | 4.68 | 86.1% |
| ex-post competitive | 5.87 | 74.8% |
| all districts | 5.17 | 96.1% |

## Rolling out-of-sample predictions (for Stage 2)

Each eval year is fit on 2006 through the election before it (2018: 2006–2016, …, 2024: 2006–2022). An eval year's own results never enter its fit.

```python
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("../../src").resolve()))
sys.path.insert(0, str(Path("../..").resolve()))  # repository root, for experiments.house
from house import stage1

CONFIG = stage1.HOUSE_ENSEMBLE_AB_85_15  # or stage1.HOUSE_MODEL_A
rolling = stage1.rolling_predictions(CONFIG)  # {2018: Stage1Result, 2020: ..., 2022: ..., 2024: ...}
oos = pd.concat([r.predictions for r in rolling.values()], ignore_index=True)
rolling[2024].metrics          # the blend scored like a single model
rolling[2024].coefficients     # coef per term and component; inc_signed is the fixed 2.5
rolling[2024].components["A_polling"].model.summary()
```

`predictions` has one row per predicted district, with these columns:

- `year`, `state`, `district`
- `train_start`, `train_end`
- `prediction`, `actual`, and `residual` (actual − prediction)
- `prediction_<component>` for each component model

Districts with imputed (uncontested) targets aren't predicted in years that have targets.

## The 2026 forecast

```python
forecast = stage1.forecast(CONFIG)  # fit on 2006-2024, predict 2026; actual and residual are NA
forecast.predictions
```

Model A needs 2026's generic-ballot margin (`gen_margin_raw`) in `data/processed/house/modeling.csv`. Until they are there, the forecast raises: every 2026 row is excluded for a missing predictor.

## Tests

`tests/test_stage1_configs.py` covers:

- the predictor definitions
- signed incumbency
- the fixed +2.5 coefficient
- equal-year weighting, checked against a hand-computed WLS
- chronological windows, including a leakage check
- exact 85/15 blending
