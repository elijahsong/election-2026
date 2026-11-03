import numpy as np
import pandas as pd
import pytest
from test_stage1_regression import TARGET, tables

from house.experiments import evaluation, tracking
from house.experiments import run as run_mod
from house.experiments.run import run_ols, run_ols_fixed, run_wls
from house.regression.config import COMPETITIVE_SUBSETS, EVAL_RACE_TYPES
from house.regression.schema import MODELED_IMPUTATION_METHODS

FORMULA = f"{TARGET} ~ gen_margin_raw + partisan_lean + inc_signed"
YEARS = {"train_years": [2006, 2008, 2010], "eval_years": [2012], "predict_years": [2014]}


def synthetic_tables():
    """SYNTHETIC: targets on the true line for 2006-2012, NC-3 imputed 4 points above it, and 2014 predictors only."""
    predictors, targets = tables((2006, 2008, 2010, 2012, 2014))
    targets = targets[targets["year"] < 2014].copy()
    modeled = (targets["state"] == "NC") & (targets["district"] == 3)
    targets.loc[modeled, TARGET] += 4
    targets.loc[modeled, "imputation_method"] = MODELED_IMPUTATION_METHODS[0]
    predictors["college"] = 30.0 + predictors["district"] ** 2 + np.arange(len(predictors)) % 5  # not collinear
    predictors["gb_margin_2p_norm"] = predictors["gen_margin_raw"]  # the default ex-ante competitive definition uses it
    targets["is_imputed"] = modeled
    targets["race_type"] = np.where(modeled, "dem_uncontested", "dr")
    targets["margin_2p_norm"] = targets[TARGET]
    targets["winner_party"] = np.where(targets[TARGET] > 0, "D", "R")
    return predictors, targets


def test_run_ols_excludes_imputed_rows_from_training_and_evaluation_and_saves(tmp_path):
    predictors, targets = synthetic_tables()
    result = run_ols(FORMULA, **YEARS, name="classic_ols", predictors=predictors, targets=targets, save=True,
                     results_dir=tmp_path)

    assert result.counts["n_train"] == 9
    assert result.counts["excluded"] == {"train:imputed_target": 3, "eval:imputed_target": 1}
    assert result.model.params.round(8).tolist() == [1, 2, 0.5, -3]
    # NC-3 2012 (imputed 4 points off the true line) is neither predicted nor scored
    assert result.metrics.index.get_level_values("subset").unique().tolist() == \
        ["valid", "competitive_ex_ante", "competitive_ex_post"]
    assert result.metrics.loc[("valid", 2012), ["n", "mae"]].tolist() == [3, pytest.approx(0, abs=1e-8)]
    assert result.coverage.loc[2012, ["n_districts", "imputed_target", "n_valid"]].tolist() == [4, 1, 3]
    assert result.counts["n_eval"] == 3
    predicted = result.predictions[result.predictions["role"] == "predict"]
    assert len(predicted) == 4 and predicted["actual"].isna().all() and predicted["prediction"].notna().all()

    loaded = tracking.load_run(result.run_id, tmp_path)
    assert loaded.config == result.config and loaded.counts == result.counts
    assert loaded.metrics.equals(result.metrics) and loaded.coverage.equals(result.coverage)
    assert loaded.predictions.equals(result.predictions)
    assert len(tracking.load_runs(tmp_path)) == 1


def test_missing_predictors_are_excluded_with_reasons_and_ablations_can_share_rows():
    predictors, targets = synthetic_tables()
    predictors.loc[(predictors["year"] == 2006) & (predictors["district"] == 1), "college"] = np.nan
    predictors.loc[(predictors["year"] == 2012) & (predictors["district"] == 2), "college"] = np.inf
    data = {"predictors": predictors, "targets": targets}

    with pytest.warns(UserWarning, match="missing:college"):
        full = run_ols(FORMULA + " + college", **YEARS, **data)
    assert full.counts["excluded"] == {"train:imputed_target": 3, "train:missing:college": 1,
                                       "eval:imputed_target": 1, "eval:missing:college": 1}
    assert (full.counts["n_train"], full.counts["n_eval"]) == (8, 2)

    plain = run_ols(FORMULA, **YEARS, **data)
    with pytest.warns(UserWarning):
        shared = run_ols(FORMULA, **YEARS, **data, require_complete=["college"])
    compared = tracking.compare([full, plain, shared])
    assert compared["same_train_rows"].tolist() == [True, False, True]
    assert compared["same_eval_rows"].tolist() == [True, False, True]
    formulas = [FORMULA + " + college", f"{TARGET} ~ partisan_lean * hispanic + C(state)"]
    assert evaluation.shared_columns(formulas, predictors.columns) == \
        ["gen_margin_raw", "partisan_lean", "inc_signed", "college", "state"]  # hispanic isn't a column here


def test_rolling_best_of_averages_windows_with_equal_weight():
    predictors, targets = synthetic_tables()
    data = {"predictors": predictors, "targets": targets, "require_complete": ["college"]}
    formulas = {"base": FORMULA, "no_inc": f"{TARGET} ~ gen_margin_raw + partisan_lean"}
    windows = {year: {"main": [run_ols(formula, train_years=train, eval_years=[year], **data, name=name)
                               for name, formula in formulas.items()]}
               for year, train in {2010: [2006, 2008], 2012: [2006, 2008, 2010]}.items()}
    metrics, coefficients = tracking.rolling_best_of(windows, rank_by="mae", top=1)
    maes = [blocks["main"][1].metrics.loc[("valid", "all"), "mae"] for blocks in windows.values()]
    no_inc = metrics.loc[("main", "no_inc")]
    assert no_inc["mae"] == pytest.approx(sum(maes) / 2) and no_inc[["mae_2010", "mae_2012"]].tolist() == maes
    assert no_inc["Δmae"] == pytest.approx(no_inc["mae"] - metrics.loc[("main", "base"), "mae"])
    assert list(coefficients.index) == list(metrics.index)


def test_training_weights_and_wls():
    train = pd.DataFrame({"year": [2006, 2006, 2006, 2008, 2010]})
    equal = run_mod.training_weights(train, equal_year=True)
    assert equal.groupby(train["year"]).sum().round(8).nunique() == 1 and equal.mean() == pytest.approx(1)
    recency = run_mod.training_weights(train, half_life=4)
    assert recency.iloc[0] / recency.iloc[4] == pytest.approx(0.5)  # 2006 is one half-life (4 years) before 2010
    midterm = run_mod.training_weights(train, midterm_weight=1.25)
    assert midterm.iloc[0] / midterm.iloc[3] == pytest.approx(1.25)  # 2006 midterm vs 2008 presidential
    assert run_mod.training_weights(train).eq(1).all()

    predictors, targets = synthetic_tables()
    data = {"predictors": predictors, "targets": targets}
    ols, wls = run_ols(FORMULA, **YEARS, **data), run_wls(FORMULA, **YEARS, **data)
    assert wls.model.params.round(8).equals(ols.model.params.round(8))  # no weighting: WLS is OLS
    weighted = run_wls(FORMULA, **YEARS, **data, equal_year=True, half_life=6, midterm_weight=1.25)
    assert weighted.config["weights"] == {"equal_year": True, "half_life": 6, "midterm_weight": 1.25}
    assert weighted.counts["train_rows_hash"] == ols.counts["train_rows_hash"]  # same rows, only reweighted


def test_fixed_coefficients_are_imposed_and_the_rest_re_estimated():
    predictors, targets = synthetic_tables()
    data = {"predictors": predictors, "targets": targets}
    ols = run_ols(FORMULA, **YEARS, **data)
    # the synthetic target's true inc_signed coefficient is -3: imposing it changes nothing
    true = run_ols_fixed(FORMULA, **YEARS, **data, fixed={"inc_signed": -3})
    assert true.model.params.round(8).tolist() == [1, 2, 0.5]  # Intercept, gen_margin_raw, partisan_lean
    assert true.coefficients.loc["inc_signed", "coef"] == -3 and pd.isna(true.coefficients.loc["inc_signed", "std_err"])
    assert np.allclose(true.predictions["prediction"], ols.predictions["prediction"])
    # fixing it at 0 is dropping the term, on the same rows
    zero = run_ols_fixed(FORMULA, **YEARS, **data, fixed={"inc_signed": 0})
    dropped = run_ols(f"{TARGET} ~ gen_margin_raw + partisan_lean", **YEARS, **data, require_complete=["inc_signed"])
    assert np.allclose(zero.predictions["prediction"], dropped.predictions["prediction"])
    assert zero.counts["eval_rows_hash"] == ols.counts["eval_rows_hash"]
    with pytest.raises(ValueError, match="columns in the formula"):
        run_ols_fixed(FORMULA, **YEARS, **data, fixed={"college": 1})
    # WLS with a fixed coefficient: unweighted it is run_ols_fixed; weighted it fits the same adjusted formula
    assert np.allclose(run_wls(FORMULA, **YEARS, **data, fixed={"inc_signed": 0}).predictions["prediction"],
                       zero.predictions["prediction"])
    both = run_wls(FORMULA, **YEARS, **data, half_life=6, fixed={"inc_signed": 5})
    assert both.coefficients.loc["inc_signed", "coef"] == 5 and both.config["weights"]["half_life"] == 6


def test_blends_score_like_a_model_and_weights_never_peek():
    from house.experiments import ensemble
    predictors, targets = synthetic_tables()
    data = {"predictors": predictors, "targets": targets, "require_complete": ["college"]}
    a = run_ols(FORMULA, **YEARS, **data)  # exact on the synthetic line
    b = run_ols(f"{TARGET} ~ gen_margin_raw + partisan_lean", **YEARS, **data)  # misses inc_signed
    scores = ensemble.blend_scores(a, b, [0.0, 0.5, 1.0], targets, predictors, TARGET, COMPETITIVE_SUBSETS,
                                   EVAL_RACE_TYPES)
    assert scores.loc[(0.0, "valid"), "mae"] == pytest.approx(a.metrics.loc[("valid", "all"), "mae"], abs=1e-9)
    assert scores.loc[(1.0, "valid"), "mae"] == pytest.approx(b.metrics.loc[("valid", "all"), "mae"])
    assert scores.loc[(0.5, "valid"), "mae"] == pytest.approx(scores.loc[(1.0, "valid"), "mae"] / 2)
    # the weight for 2014 comes from 2010 and 2012 only, even though 2014 favors b
    windows = {2010: scores, 2012: scores, 2014: scores.assign(mae=-scores["mae"])}
    assert ensemble.choose_weight(windows, 2014, "valid", "mae") == (0.0, [2010, 2012])
    with pytest.raises(ValueError, match="No window before"):
        ensemble.choose_weight(windows, 2010, "valid", "mae")
    corr = ensemble.error_correlation([(b, b)], ["valid"])
    assert corr.loc["valid", "corr"] == pytest.approx(1) and corr.loc["valid", "n"] == 3


def test_formulas_with_interactions_categoricals_and_transforms():
    predictors, targets = synthetic_tables()
    data = {"predictors": predictors, "targets": targets}
    result = run_ols(f"{TARGET} ~ gen_margin_raw + partisan_lean * inc_signed + C(state) + np.log(college)",
                     **YEARS, **data)
    assert {"partisan_lean:inc_signed", "C(state)[T.NC]", "np.log(college)"} <= set(result.coefficients.index)
    with pytest.raises(ValueError, match="left-hand side"):
        run_ols(f"np.exp({TARGET}) ~ gen_margin_raw", **YEARS, **data)


def test_guards_and_prediction_only_runs():
    predictors, targets = tables((2020, 2022, 2024))
    with pytest.raises(ValueError, match="use_holdout"):  # 2024 in training, not just evaluation
        run_ols(FORMULA, train_years=[2022, 2024], predict_years=[2020], predictors=predictors, targets=targets)
    with pytest.raises(ValueError, match="both train and eval"):
        run_ols(FORMULA, train_years=[2020, 2022], eval_years=[2022], predictors=predictors, targets=targets)

    predicted = run_ols(FORMULA, train_years=[2020, 2022], predict_years=[2024], use_holdout=True,
                        predictors=predictors, targets=targets)
    assert predicted.metrics is None and predicted.counts["n_predict"] == 4

