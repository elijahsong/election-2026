# Senate Stage 1: production configurations

`senate/stage1.py` holds two fixed Stage 1 specifications. The specification is fixed, but the coefficients are refit
on each training window. The interface mirrors `house/stage1.py`, so a future Senate Stage 2 / Monte Carlo can read
it the same way. The code is separate from the House's.

| name | prediction |
|---|---|
| `senate_s0` | `adj_margin_2p_norm ~ partisan_lean + gen_margin_raw` (no incumbency) |
| `senate_s1` | the same, plus `2.5 × inc_signed`, imposed rather than estimated |

- **Fitting:** equal-year WLS. Each training race gets weight ∝ 1 / (training races in its year).
- **Not included:** fundraising, race-level polls and expert ratings. Polls (`data/processed/senate/polls.csv`) and
  ratings (`expert-ratings.csv`) come in after Stage 1.
- **Evidence:** the specifications come from `notebooks/senate/stage1/01_reg_senate_fundamentals.ipynb`, the
  2012–2022 backtest. The module docstring records the rows, windows and sign conventions.

Rolling out-of-sample results (each year fit on 2006 through the election before it):

| | S0 | S1 |
|---|---|---|
| MAE, all, 2012–2022 (selection) | 7.95 | 7.24 |
| MAE, all, 2024 (not used in selection) | 4.83 | 4.49 |
| MAE, ex-ante competitive, 2024 (n = 11) | 3.99 | 3.60 |
| MAE, all, pooled 2012–2024 (n = 233) | 7.51 | 6.85 |
| MAE, ex-ante competitive, pooled (n = 72) | 7.01 | 6.11 |
| MAE, ex-post competitive, pooled (n = 53) | 6.09 | 5.32 |
| winners called, all, pooled | 88.8% | 90.6% |
| bias (actual − predicted), all, pooled | −0.28 | −0.50 |

## Rolling out-of-sample predictions (for Stage 2)

```python
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("../../src").resolve()))
from senate import stage1

CONFIG = stage1.SENATE_S1  # or stage1.SENATE_S0
rolling = stage1.rolling_predictions(CONFIG)  # {2012: Stage1Result, ..., 2024: ...}
oos = pd.concat([r.predictions for r in rolling.values()], ignore_index=True)
rolling[2024].metrics        # n, mae, rmse, bias, accuracy_pct for all / competitive_ex_ante / competitive_ex_post
rolling[2024].scored         # predictions with an actual, plus the two competitive flags
rolling[2024].coefficients   # inc_signed is the fixed 2.5 in S1
rolling[2024].model.summary()
```

Each race is keyed `year, state, class`, so a regular election and a special in one state-year are separate rows.
`predictions` has one row per predicted race:

- `year`, `state`, `class`, `special`
- `train_start`, `train_end`
- `prediction`, `actual`, and `residual` (actual − prediction)

In years with results, only observed D-vs-R races are predicted. Imputed targets and AK 2022
(`senate.regression.EXCLUDED_RACES`) are left out.

Differences from the House interface:
- the key has `class` instead of `district`;
- a config is one formula, not a blend of components, so a result has `model` instead of `components`;
- `ROLLING_EVAL_YEARS` starts in 2012, because the Senate has about 33 races a year.

## The 2026 forecast

```python
forecast = stage1.forecast(CONFIG)  # fit on 2006-2024, predict all 35 races; actual and residual are NA
```

2026's `gen_margin_raw` is the hand-transcribed election-eve generic ballot (D+8.7). NE has no Democratic nominee
(Osborn is an independent), and nor do ID and SD. Their predictions are D-vs-R margins from the model, not contests
on the ballot. `polls.csv` and `expert-ratings.csv` flag these races (`quality_flags`).

## Tests

`tests/test_senate_stage1_configs.py` covers:
- the fixed +2.5 in S1, and none in S0;
- chronological windows;
- the excluded race;
- the 2026 forecast's rows.

`tests/test_senate_stage1.py` covers the backtest code in `senate/regression/`.
