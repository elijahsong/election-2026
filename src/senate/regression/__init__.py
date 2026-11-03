"""Senate Stage 1: fundamentals-only regression of the Democratic two-party margin (separate from the House code).

  data      load_data, inventory, KEY, TARGET, EVAL_RACE_TYPES, EXCLUDED_RACES
  backtest  backtest (OLS or equal-year / recency WLS, fixed terms; expanding window), baseline, drop_one_cycle
  metrics   compare, residuals_by, subsets (competitive definitions)
"""

from .backtest import Backtest, backtest, baseline, drop_one_cycle, fit, fixed_offset, training_rows, training_weights
from .data import EVAL_RACE_TYPES, KEY, EXCLUDED_RACES, TARGET, inventory, load_data
from .metrics import EX_ANTE_THRESHOLD, EX_POST_THRESHOLD, compare, residuals_by, subsets
