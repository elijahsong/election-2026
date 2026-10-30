"""Fit on the training years, predict the evaluation years, and score only when actual outcomes exist.

fit_model, predict, attach_outcomes, and score are independent. run_experiment takes the predictors and targets as
separate, unfiltered tables (data.load_predictors, data.load_targets), selects the training and evaluation rows with
experiment_data (which joins and checks them and enforces config.HOLDOUT_YEARS), then chains the first three for one
config. An expanding-window backtest is a loop over
run_experiment with a different train_years / eval_years per window.
"""

import pandas as pd

from . import metrics
from .config import check_holdout, resolve_features
from .data import join_target, select_years, split_years
from .models import make_model
from .schema import KEY, validate_schema


def fit_model(train: pd.DataFrame, features: list[str], target: str, model_type: str, model_params: dict):
    """A model fitted on every row of `train`; the key, features, and target must be valid on all of them."""
    validate_schema(train, required=[*features, target]).raise_if_errors()
    return make_model(model_type, **model_params).fit(train[features], train[target])


def predict(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """KEY + prediction for every row of df. The target isn't needed."""
    validate_schema(df, required=features).raise_if_errors()
    out = df[KEY].copy()
    out["prediction"] = model.predict(df[features])
    return out.reset_index(drop=True)


def attach_outcomes(predictions: pd.DataFrame, df: pd.DataFrame, target: str) -> pd.DataFrame:
    """predictions plus `actual` (df's target) and `residual` = actual - prediction; rows without a target keep NA."""
    actual = df[KEY + [target]].rename(columns={target: "actual"})
    out = predictions.merge(actual, on=KEY, how="left", validate="one_to_one")
    out["residual"] = out["actual"] - out["prediction"]
    return out


def score(scored: pd.DataFrame) -> pd.DataFrame:
    """n, MAE, RMSE, and bias per year and over all rows with an actual; raises if none has one (e.g. 2026).

    Rows without an actual (no observed target, or a modeled imputation blanked by data.join_target) aren't scored.
    """
    scored = scored[scored["actual"].notna()]
    if scored.empty:
        raise ValueError("No rows with an actual outcome to score")

    def row(year, group: pd.DataFrame) -> dict:
        a, p = group["actual"], group["prediction"]
        return {"year": year, "n": len(group), "mae": metrics.mae(a, p), "rmse": metrics.rmse(a, p),
                "bias": metrics.bias(a, p)}

    return pd.DataFrame([row(year, g) for year, g in scored.groupby("year")] + [row("all", scored)])


TRAINING_INPUTS = ["non_imputed", "all"]


def experiment_data(predictors: pd.DataFrame, targets: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training rows, evaluation rows) for the config: KEY, every predictor, and the config's target.

    The target is joined onto the predictors by key and checked (data.join_target) before the years are selected.
    Training keeps only rows with a target. The config's "training_inputs" picks which: "non_imputed" (the default)
    leaves out modeled imputations, which join_target blanks; "all" trains on them too. Evaluation rows are every row of
    the eval_years, with modeled imputations always blank, so score() scores observed targets only.
    """
    check_holdout(config)
    target, training_inputs = config["target"], config.get("training_inputs", "non_imputed")
    if training_inputs not in TRAINING_INPUTS:
        raise ValueError(f"Unknown training_inputs {training_inputs!r}; expected one of {TRAINING_INPUTS}")
    df = join_target(predictors, targets, target)
    train, evaluation = split_years(df, config["train_years"], config["eval_years"])
    if training_inputs == "all":
        train = select_years(join_target(predictors, targets, target, keep_modeled=True), config["train_years"])
    return train[train[target].notna()], evaluation


def run_experiment(predictors: pd.DataFrame, targets: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, object]:
    """Fit on the config's train_years and predict its eval_years; returns (predictions, fitted model).

    predictors and targets are the whole tables, every year; experiment_data picks the training and evaluation rows.
    The model is fitted on the training rows with a target; rows without one (e.g. uncontested races, whose imputed
    targets join_target blanks unless "training_inputs" is "all") are left out. predictions holds KEY + prediction,
    actual, and residual for every evaluation row (NA for rows without an observed target, e.g. 2026). Scoring is
    separate: score(predictions) scores the rows with an actual outcome.
    """
    features, target = resolve_features(config), config["target"]
    train, evaluation = experiment_data(predictors, targets, config)
    model = fit_model(train, features, target, config["model_type"], config["model_params"])
    predictions = predict(model, evaluation, features)
    return attach_outcomes(predictions, evaluation, target), model
