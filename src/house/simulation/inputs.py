"""Stage 2 preprocessing: the inputs every Stage 2 notebook or experiment builds the same way.

    load_targets_raw()         data/processed/house/targets_raw.csv: the pre-reclassification race types
    load_candidates_2026()     data/processed/house/candidates_2026.csv (data-processing/house/candidates_2026.py)
    stage1_inputs(config, ...) one Stage 1 configuration's rolling out-of-sample run, on races.lineup_targets, as
                               Stage1Inputs: the scored residuals (oos) that fit Stage 2, and every prediction,
                               restored lineups included, that the simulation uses

Nothing here makes a modeling choice: which configurations, years, fallback, seed and error model to use stay in the
calling notebook. The known optional-column warnings from loading and fitting are silenced, as the notebooks did.
"""

import warnings
from dataclasses import dataclass

import pandas as pd

from house import stage1
from house.regression.data import ROOT

from .calibration import out_of_sample
from .races import lineup_targets

TARGETS_RAW_PATH = ROOT / "data/processed/house/targets_raw.csv"
CANDIDATES_2026_PATH = ROOT / "data/processed/house/candidates_2026.csv"


def load_targets_raw(path=TARGETS_RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_candidates_2026(path=CANDIDATES_2026_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


@dataclass
class Stage1Inputs:
    """One configuration's Stage 1 rolling run, ready for Stage 2."""
    config: stage1.Stage1Config
    rolling: dict              # {year: Stage1Result}, fit on lineup_targets(targets)
    oos: pd.DataFrame          # scored residuals only (out_of_sample): fits the Stage 2 error SDs
    predictions: pd.DataFrame  # every prediction, unscored lineup rows too (scored_only=False): simulated


def stage1_inputs(config: stage1.Stage1Config, *, predictors: pd.DataFrame, targets: pd.DataFrame,
                  eval_years=stage1.ROLLING_EVAL_YEARS) -> Stage1Inputs:
    """stage1.rolling_predictions(config, eval_years) on lineup_targets(targets), so the races restored to their
    pre-election D-vs-R lineups are predicted too (never scored or trained on; checks.check_lineup_run confirms every
    other prediction and residual is unchanged)."""
    with warnings.catch_warnings(action="ignore"):
        rolling = stage1.rolling_predictions(config, eval_years, predictors=predictors,
                                             targets=lineup_targets(targets))
    return Stage1Inputs(config=config, rolling=rolling, oos=out_of_sample(rolling),
                        predictions=out_of_sample(rolling, scored_only=False))


def stage1_forecast(config: stage1.Stage1Config, *, predictors: pd.DataFrame, targets: pd.DataFrame,
                    year: int = stage1.FORECAST_YEAR) -> stage1.Stage1Result:
    """stage1.forecast, with the known loading and fitting warnings silenced."""
    with warnings.catch_warnings(action="ignore"):
        return stage1.forecast(config, year, predictors=predictors, targets=targets)
