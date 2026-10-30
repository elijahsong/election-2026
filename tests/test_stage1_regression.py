import math

import pandas as pd
import pytest

from house.regression import backtest, metrics
from house.regression.data import join_target, split_years
from house.regression.schema import (
    MODELED_IMPUTATION_METHODS,
    TARGETS_SCHEMA,
    SchemaError,
    validate_schema,
)

TARGET = "adj_margin_2p_norm"
FEATURES = ["gen_margin_raw", "partisan_lean", "inc_signed"]
OLS = {"model_type": "ols", "model_params": {"add_constant": True}, "features": FEATURES, "target": TARGET}


def synthetic_frame(years=(2006, 2008, 2010, 2012)) -> pd.DataFrame:
    """SYNTHETIC rows, not election data: four districts per year, target exactly 1 + 2*gen + 0.5*lean - 3*inc."""
    df = pd.DataFrame([
        {"year": y, "state": s, "district": d, "gen_margin_raw": float(i), "partisan_lean": 10.0 * d - 5 * i,
         "inc_signed": (-1) ** (d + i)}
        for i, y in enumerate(years) for s, d in [("AK", 0), ("NC", 1), ("NC", 2), ("NC", 3)]
    ])
    df[TARGET] = 1 + 2 * df["gen_margin_raw"] + 0.5 * df["partisan_lean"] - 3 * df["inc_signed"]
    return df


def tables(years=(2006, 2008, 2010, 2012)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """synthetic_frame split into (predictors, targets), as load_predictors and load_targets return them."""
    df = synthetic_frame(years).assign(imputation_method=None)
    return df.drop(columns=[TARGET, "imputation_method"]), df[["year", "state", "district", TARGET, "imputation_method"]]


def issues(report_frame: pd.DataFrame) -> set:
    return set(zip(report_frame["column"], report_frame["issue"]))


def test_validation_levels():
    df = synthetic_frame().assign(notes="unregistered columns are ignored")
    df.loc[1, "partisan_lean"] = float("nan")  # (2006, NC, 1)

    report = validate_schema(df.drop(columns="inc_signed"))
    assert report.ok
    assert {("partisan_lean", "null"), ("inc_signed", "missing_column")} <= issues(report.warnings)
    assert "notes" not in set(report.issues["column"])
    assert TARGET not in set(validate_schema(df.drop(columns=TARGET)).issues["column"])  # a missing target isn't flagged
    assert report.issues.loc[report.issues["issue"] == "null", "keys"].item() == [(2006, "NC", 1)]

    blocked = validate_schema(df, required=["partisan_lean"])
    assert issues(blocked.errors) == {("partisan_lean", "null")}
    with pytest.raises(SchemaError) as err:
        blocked.raise_if_errors()
    assert err.value.report is blocked

    flags = df[["year", "state", "district"]].assign(quality_flags=None)  # nullable: null means no flags
    assert "quality_flags" not in set(validate_schema(flags, schema=TARGETS_SCHEMA).issues["column"])
    assert issues(validate_schema(flags, required=["quality_flags"], schema=TARGETS_SCHEMA).errors) == \
        {("quality_flags", "null")}  # unless required

    bad_keys = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    bad_keys.loc[2, "state"] = "nc"
    assert issues(validate_schema(bad_keys).errors) == {("state", "invalid_value"),
                                                       ("year, state, district", "duplicate_key")}


def test_split_uses_exactly_the_listed_years():
    train, evaluation = split_years(synthetic_frame(), [2006, 2010], [2008, 2012])
    assert sorted(train["year"].unique()) == [2006, 2010]
    assert sorted(evaluation["year"].unique()) == [2008, 2012]
    with pytest.raises(ValueError, match="both"):
        split_years(synthetic_frame(), [2006, 2010], [2010])
    with pytest.raises(ValueError, match="not in the data"):
        split_years(synthetic_frame(), [2006, 2014], [2012])


def test_ols_fit_and_predict_without_targets():
    train, evaluation = split_years(synthetic_frame(), [2006, 2010], [2012])
    model = backtest.fit_model(train, FEATURES, TARGET, "ols", {"add_constant": True})
    assert model.results.params.round(8).tolist() == [1, 2, 0.5, -3]

    predictions = backtest.predict(model, evaluation.drop(columns=TARGET), FEATURES)
    assert list(predictions.columns) == ["year", "state", "district", "prediction"]
    assert predictions["prediction"].round(8).tolist() == evaluation[TARGET].tolist()


def test_join_target_requires_matching_keys():
    predictors, targets = tables()
    joined = join_target(predictors, targets[targets["year"] < 2012], TARGET)  # 2012: predictors only, like 2026
    assert len(joined) == len(predictors) and list(joined.columns) == [*predictors.columns, TARGET]
    assert joined.loc[joined["year"] == 2012, TARGET].isna().all() and joined[TARGET].notna().sum() == 12

    with pytest.raises(ValueError, match="only one table"):
        join_target(predictors.iloc[1:], targets, TARGET)  # (2006, AK, 0) has a target but no predictors
    with pytest.raises(ValueError, match="only one table"):
        join_target(predictors, targets.iloc[1:], TARGET)
    with pytest.raises(ValueError, match="not in the predictors"):
        join_target(predictors[predictors["year"] < 2012], targets, TARGET)
    with pytest.raises(ValueError, match="also a predictor"):
        join_target(synthetic_frame(), targets, TARGET)
    with pytest.raises(ValueError, match="not in the targets"):
        join_target(predictors, targets, "margin_2p_norm")
    with pytest.raises(ValueError, match="imputation_method"):
        join_target(predictors, targets.drop(columns="imputation_method"), TARGET)


def test_experiment_scores_only_when_targets_exist():
    predictors, targets = tables()
    targets.loc[targets["year"] == 2008, TARGET] = float("nan")  # a year that is neither trained on nor scored
    config = {**OLS, "train_years": [2006, 2010], "eval_years": [2012]}

    predictions, _ = backtest.run_experiment(predictors, targets, config)
    scores = backtest.score(predictions)
    assert scores["n"].tolist() == [4, 4]
    assert scores[["mae", "rmse", "bias"]].abs().max().max() < 1e-8

    predictions, _ = backtest.run_experiment(predictors, targets[targets["year"] < 2012], config)  # e.g. 2026
    assert len(predictions) == 4 and predictions["actual"].isna().all()
    with pytest.raises(ValueError, match="No rows with an actual"):
        backtest.score(predictions)


def test_modeled_imputations_are_neither_trained_on_nor_scored():
    predictors, targets = tables()
    modeled = (targets["state"] == "NC") & (targets["district"] == 3)
    targets.loc[modeled, TARGET] = 1000.0  # far off the true line: training on it would bend the fit
    # NC-3's four years alternate between the two modeled methods
    targets.loc[modeled, "imputation_method"] = [*MODELED_IMPUTATION_METHODS, *MODELED_IMPUTATION_METHODS][:4]
    targets.loc[targets["district"] == 2, "imputation_method"] = "some_observed_method"  # kept
    config = {**OLS, "train_years": [2006, 2008, 2010], "eval_years": [2012]}

    assert join_target(predictors, targets, TARGET).loc[modeled.to_numpy(), TARGET].isna().all()
    predictions, model = backtest.run_experiment(predictors, targets, config)
    assert model.results.nobs == 9 and model.results.params.round(8).tolist() == [1, 2, 0.5, -3]
    assert predictions.loc[predictions["district"] == 3, "actual"].isna().all()
    assert backtest.score(predictions)["n"].tolist() == [3, 3]


def test_training_inputs_all_trains_on_modeled_imputations_but_never_scores_them():
    predictors, targets = tables()
    modeled = (targets["state"] == "NC") & (targets["district"] == 3)
    targets.loc[modeled, "imputation_method"] = MODELED_IMPUTATION_METHODS[0]  # values stay on the true line
    config = {**OLS, "train_years": [2006, 2008, 2010], "eval_years": [2012]}

    _, model = backtest.run_experiment(predictors, targets, config)
    assert model.results.nobs == 9
    predictions, model = backtest.run_experiment(predictors, targets, {**config, "training_inputs": "all"})
    assert model.results.nobs == 12
    assert predictions.loc[predictions["district"] == 3, "actual"].isna().all()
    with pytest.raises(ValueError, match="training_inputs"):
        backtest.run_experiment(predictors, targets, {**config, "training_inputs": "imputed"})


def test_holdout_years_need_explicit_opt_in():
    config = {**OLS, "train_years": [2020, 2022], "eval_years": [2024]}  # config.HOLDOUT_YEARS is [2024]
    with pytest.raises(ValueError, match="use_holdout"):
        backtest.run_experiment(*tables((2020, 2022, 2024)), config)
    predictions, _ = backtest.run_experiment(*tables((2020, 2022, 2024)), {**config, "use_holdout": True})
    assert len(predictions) == 4


def test_metrics():
    actual, predicted = [0, 2, 4], [1, 1, 7]  # errors +1, -1, +3
    assert metrics.mae(actual, predicted) == pytest.approx(5 / 3)
    assert metrics.rmse(actual, predicted) == pytest.approx(math.sqrt(11 / 3))
    assert metrics.bias(actual, predicted) == pytest.approx(1)
