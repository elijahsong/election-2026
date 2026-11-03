"""Senate Monte Carlo around the Stage 2 blend (Senate-only).

  errors       estimate_error_sds (S-M0, S-M1, S-M2), simulate_margins, load_regions
  calibration  blend_out_of_sample, backtest (2018-2024 on earlier years only), polling_group_diagnostics
  chamber      2026: forecast_points (independent-race heuristic), simulate_2026, count_seats, control, plots
  track        resolve_candidates, track_table: the congress-track Senate rows

S-M0 is the baseline; S-M1 (polled / unpolled race scales) and S-M2 (regional error) are experiments.
"""

from . import calibration, chamber, errors, track
from .errors import MODELS, estimate_error_sds, simulate_margins
