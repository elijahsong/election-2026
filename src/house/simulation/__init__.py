"""House Stage 2 simulation infrastructure shared between experiments: `from house import simulation`."""

from .calibration import (
    Calibration, YearCalibration, backtest, brier, call_party, central_interval, coverage, group_seats, out_of_sample,
    reliability, scored, seat_percentile, seat_quantile, seat_summary, win_probability,
)
from .hierarchical import (
    FACTORS, LEVELS, NATIONAL_PRIOR_ELECTIONS, NATIONAL_PRIOR_SD, Draws, estimate_error_sds, simulate_margins, t_scale,
)
from .factors import (
    COLLEGE_REFERENCE_SD, DEMOGRAPHIC_PRIOR_ELECTIONS, REGIONAL_PRIOR_ELECTIONS, REGIONS_PATH, allocate_budget,
    attach_factors, college_terciles, demographic_loadings, estimate_factor_sds, factor_moments, load_regions,
)
from .races import (
    DEM, GOP, OTHER, THIRD_PARTY_UPSET_PROB, TREATMENTS, RaceDraws, attach_predictions, forecast_table,
    historical_races, lineup_targets, races_2026, simulate_races,
)
from .resolvers import JungleSameParty, RankedChoice, Resolver, resolve_candidate
from .top_two import TopTwoSameParty, load_top_two, predict_top_two
from .inputs import Stage1Inputs, load_candidates_2026, load_targets_raw, stage1_forecast, stage1_inputs
from .checks import (
    check_backtest, check_engine, check_engine_draws, check_forecast, check_lineup_run, check_same_draws,
    check_variance_budget,
)
from .report import FORMATS, TRACK_TEMPLATE_PATH, race_id, seat_table, stack, styled, track_table
from .visualization import (
    plot_backtest_seat_distributions, plot_reliability, plot_seat_distribution, plot_seat_errors,
)

__all__ = [
    "Calibration", "YearCalibration", "backtest", "brier", "call_party", "central_interval", "coverage", "group_seats",
    "out_of_sample", "reliability", "scored", "seat_percentile", "seat_quantile", "seat_summary", "win_probability",
    "FACTORS", "LEVELS", "NATIONAL_PRIOR_ELECTIONS", "NATIONAL_PRIOR_SD", "Draws", "estimate_error_sds",
    "simulate_margins", "t_scale",
    "COLLEGE_REFERENCE_SD", "DEMOGRAPHIC_PRIOR_ELECTIONS", "REGIONAL_PRIOR_ELECTIONS", "REGIONS_PATH",
    "allocate_budget", "attach_factors", "college_terciles", "demographic_loadings", "estimate_factor_sds",
    "factor_moments", "load_regions",
    "DEM", "GOP", "OTHER", "THIRD_PARTY_UPSET_PROB", "TREATMENTS", "RaceDraws", "attach_predictions",
    "forecast_table", "historical_races", "lineup_targets", "races_2026", "simulate_races",
    "JungleSameParty", "RankedChoice", "Resolver", "resolve_candidate",
    "TopTwoSameParty", "load_top_two", "predict_top_two",
    "plot_seat_distribution", "plot_backtest_seat_distributions", "plot_seat_errors", "plot_reliability",
    "Stage1Inputs", "load_candidates_2026", "load_targets_raw", "stage1_forecast", "stage1_inputs",
    "check_backtest", "check_engine", "check_engine_draws", "check_forecast", "check_lineup_run", "check_same_draws",
    "check_variance_budget",
    "FORMATS", "TRACK_TEMPLATE_PATH", "race_id", "seat_table", "stack", "styled", "track_table",
]
