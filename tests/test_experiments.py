import numpy as np
import pandas as pd
import pytest
from test_stage1_regression import TARGET, tables

from experiments.house import evaluation, tracking
from experiments.house import run as run_mod
from experiments.house.run import parse_args, run_ols, run_ols_fixed, run_wls
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

    assert result.run_id == "classic_ols-t06-thru-10-v12-p14"
    again = run_ols(FORMULA, **YEARS, name="classic_ols", predictors=predictors, targets=targets, save=True,
                    results_dir=tmp_path)
    assert again.run_id == "classic_ols-t06-thru-10-v12-p14-2"
    loaded = tracking.load_run(result.run_id, tmp_path)
    assert loaded.config == result.config and loaded.counts == result.counts
    assert loaded.metrics.equals(result.metrics) and loaded.coverage.equals(result.coverage)
    assert loaded.predictions.equals(result.predictions)
    assert len(tracking.load_runs(tmp_path)) == 2
    assert tracking.load_runs(tmp_path / "nothing_saved").empty  # no runs.csv yet: an empty history, not an error


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
    assert compared["change"].tolist() == ["(base)", "- college", "- college"]
    assert compared.index.tolist() == ["ols", "ols #2", "ols #3"]
    assert len(tracking.compare(([full, plain]), full=True).columns) > len(compared.columns)
    by_subset = tracking.compare([full, plain], subsets=["valid", "competitive_ex_post"])
    assert {"accuracy_pct_valid", "mae_competitive_ex_post"} <= set(by_subset.columns)
    with pytest.raises(ValueError, match="weren't scored"):
        tracking.compare([full, plain], subsets=["competitive_typo"])
    by_district = tracking.compare_predictions([full, plain])
    assert len(by_district) == 2 and list(by_district.columns)[-2:] == ["ols", "ols #2"]
    formulas = [FORMULA + " + college", f"{TARGET} ~ partisan_lean * hispanic + C(state)"]
    assert evaluation.shared_columns(formulas, predictors.columns) == \
        ["gen_margin_raw", "partisan_lean", "inc_signed", "college", "state"]  # hispanic isn't a column here


def test_best_of_blocks_keeps_each_base_and_its_best_variants():
    predictors, targets = synthetic_tables()
    data = {"predictors": predictors, "targets": targets, "require_complete": ["college"]}
    runs = [run_ols(formula, **YEARS, **data, name=name) for name, formula in
            [("base", FORMULA), ("no_inc", f"{TARGET} ~ gen_margin_raw + partisan_lean"),
             ("no_lean", f"{TARGET} ~ gen_margin_raw + inc_signed"), ("plus_college", FORMULA + " + college")]]
    metrics, coefficients = tracking.best_of_blocks({"main": runs}, rank_by="mae", top=2)
    shown = metrics.index.get_level_values("run").tolist()
    assert shown[0] == "base" and len(shown) == 3
    assert metrics["mae"].iloc[1] <= metrics["mae"].iloc[2]  # best first
    assert metrics["Δmae"].iloc[1] == pytest.approx(metrics["mae"].iloc[1] - metrics["mae"].iloc[0])
    assert pd.isna(metrics["Δmae"].iloc[0])  # no delta for the base itself
    assert metrics["Δ|bias|"].iloc[1] == pytest.approx(abs(metrics["bias"].iloc[1]) - abs(metrics["bias"].iloc[0]))
    assert "Δbias" not in metrics and metrics["n_train"].iloc[0] == runs[0].counts["n_train"]
    tracking.style_best_of(metrics).to_html()  # renders
    side_by_side = pd.concat({"valid": metrics[["accuracy_pct", "Δaccuracy_pct"]]}, axis=1)  # (subset, metric)
    side_by_side[("valid", "Δaccuracy_pct")] = 1.0
    assert "+1.00 ✓" in tracking.style_best_of(side_by_side).to_html()  # styled by the metric level
    assert list(coefficients.index) == list(metrics.index)
    # a second block whose base differs only in a fixed coefficient still shows its base
    fixed = [run_ols_fixed(FORMULA, **YEARS, **data, fixed={"inc_signed": 1.0}, name="base_fixed")]
    two, _ = tracking.best_of_blocks({"main": runs, "fixed": fixed}, top=0)
    assert ("fixed", "base_fixed") in two.index


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
    assert tracking.describe_change(weighted.config, ols.config) == "wls: equal_year, half_life 6, midterm x1.25"


def test_fixed_coefficients_are_imposed_and_the_rest_re_estimated():
    predictors, targets = synthetic_tables()
    data = {"predictors": predictors, "targets": targets}
    ols = run_ols(FORMULA, **YEARS, **data)
    # the synthetic target's true inc_signed coefficient is -3: imposing it changes nothing
    true = run_ols_fixed(FORMULA, **YEARS, **data, fixed={"inc_signed": -3})
    assert true.model.params.round(8).tolist() == [1, 2, 0.5]  # Intercept, gen_margin_raw, partisan_lean
    assert true.coefficients.loc["inc_signed", "coef"] == -3 and pd.isna(true.coefficients.loc["inc_signed", "std_err"])
    assert np.allclose(true.predictions["prediction"], ols.predictions["prediction"])
    assert true.config["fit_formula"] == f"I({TARGET} - -3.0 * inc_signed) ~ gen_margin_raw + partisan_lean + " \
        "inc_signed - inc_signed"
    # fixing it at 0 is dropping the term, on the same rows
    zero = run_ols_fixed(FORMULA, **YEARS, **data, fixed={"inc_signed": 0})
    dropped = run_ols(f"{TARGET} ~ gen_margin_raw + partisan_lean", **YEARS, **data, require_complete=["inc_signed"])
    assert np.allclose(zero.predictions["prediction"], dropped.predictions["prediction"])
    assert zero.counts["eval_rows_hash"] == ols.counts["eval_rows_hash"]
    assert tracking.describe_change(zero.config, ols.config) == "fixed inc_signed = 0"
    with pytest.raises(ValueError, match="columns in the formula"):
        run_ols_fixed(FORMULA, **YEARS, **data, fixed={"college": 1})
    unrestricted = run_ols_fixed(FORMULA, **YEARS, **data, fixed={})  # nothing fixed: run_ols
    assert np.allclose(unrestricted.predictions["prediction"], ols.predictions["prediction"])
    # WLS with a fixed coefficient: unweighted it is run_ols_fixed; weighted it fits the same adjusted formula
    assert np.allclose(run_wls(FORMULA, **YEARS, **data, fixed={"inc_signed": 0}).predictions["prediction"],
                       zero.predictions["prediction"])
    both = run_wls(FORMULA, **YEARS, **data, half_life=6, fixed={"inc_signed": 5})
    assert both.config["fit_formula"].startswith(f"I({TARGET} - 5.0 * inc_signed)")
    assert both.coefficients.loc["inc_signed", "coef"] == 5 and both.config["weights"]["half_life"] == 6


def test_blends_score_like_a_model_and_weights_never_peek():
    from experiments.house import ensemble
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


def test_describe_change_diffs_terms_and_years():
    base = {"target": TARGET, "formula": FORMULA, "train_years": [2006, 2008, 2010], "eval_years": [2012],
            "predict_years": []}
    assert tracking.describe_change(base, base) == "(same)"
    variant = {**base, "formula": f"{TARGET} ~ gen_margin_raw + partisan_lean * college - 1",
               "train_years": [2008, 2012, 2016, 2020]}
    assert tracking.describe_change(variant, base) == \
        "- Intercept - inc_signed + college + partisan_lean:college t08,12,16,20"


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
    assert tracking.run_name(predicted.config) == "ols-t20,22-p24"


def test_run_names_and_cli():
    def name(train, eval_years=(), predict_years=()):
        return tracking.run_name({"name": "classic_ols", "estimator": "ols", "train_years": list(train),
                                  "eval_years": list(eval_years), "predict_years": list(predict_years)})

    assert name(range(2006, 2022, 2), [2022]) == "classic_ols-t06-thru-20-v22"
    assert name([2008, 2012, 2016, 2020], [2022]) == "classic_ols-t08,12,16,20-v22"
    assert name(range(2006, 2026, 2), predict_years=[2026]) == "classic_ols-t06-thru-24-p26"

    argv = ["--formula", FORMULA, "--train-years", "2006-2012,2016", "--eval-years", "2022", "--name", "x",
            "--require-complete", "college"]
    assert parse_args(argv) == {
        "formula": FORMULA, "train_years": [2006, 2008, 2010, 2012, 2016], "eval_years": [2022], "predict_years": [],
        "name": "x", "use_holdout": False, "require_complete": ["college"], "save": True,
        "competitive": {"competitive_ex_ante": ("partisan_lean + gb_margin_2p_norm", 10),
                        "competitive_ex_post": ("margin_2p_norm", 10)},
    }
    assert parse_args([*argv, "--competitive-threshold", "7.5"])["competitive"]["competitive_ex_post"] == \
        ("margin_2p_norm", 7.5)


def test_excluded_by_year_lists_districts_with_a_total():
    excluded = pd.DataFrame({"year": [2006, 2006, 2006, 2008], "state": ["NC", "AK", "AZ", "NC"],
                             "district": [3, 0, 12, 3], "role": "train",
                             "reason": ["imputed_target", "imputed_target", "imputed_target", "missing:college"]})
    table = evaluation.excluded_by_year(excluded, max_districts=2)
    assert table.index.tolist() == [2006, 2008, "total"]
    assert table["n"].tolist() == [3, 1, 4]
    assert table.loc[2006, "districts"] == "AK-AL, AZ-12, ... (+1)"
    assert table.loc[2008, ["reason", "districts"]].tolist() == ["missing:college", "NC-03"]
