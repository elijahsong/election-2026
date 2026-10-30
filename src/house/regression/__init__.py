"""House Stage 1 regression. Everything a notebook needs is importable from here: `import house.regression as hr`."""

from . import metrics
from .backtest import attach_outcomes, experiment_data, fit_model, predict, run_experiment, score
from .config import FEATURE_SETS, HOLDOUT_YEARS, MODEL_CONFIGS, check_holdout, describe_config, resolve_features
from .data import PREDICTORS_PATH, TARGETS_PATH, join_target, load_predictors, load_targets, select_years, split_years
from .models import MODELS, make_model
from .schema import (
    COLUMNS, KEY, SUPP_KEY, TARGETS_SCHEMA, TARGETS_SUPP_SCHEMA, TRANCHES, SchemaError, ValidationReport,
    validate_schema,
)

__all__ = [
    "metrics",
    "attach_outcomes", "experiment_data", "fit_model", "predict", "run_experiment", "score",
    "FEATURE_SETS", "HOLDOUT_YEARS", "MODEL_CONFIGS", "check_holdout", "describe_config", "resolve_features",
    "PREDICTORS_PATH", "TARGETS_PATH", "join_target", "load_predictors", "load_targets", "select_years", "split_years",
    "MODELS", "make_model",
    "COLUMNS", "KEY", "SUPP_KEY", "TARGETS_SCHEMA", "TARGETS_SUPP_SCHEMA", "TRANCHES", "SchemaError",
    "ValidationReport", "validate_schema",
]
