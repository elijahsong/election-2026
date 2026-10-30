import numpy as np
import pandas as pd
import pytest
from test_stage1_regression import TARGET, tables

from experiments.house import evaluation, tracking
from experiments.house.run import parse_args, run_ols
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
    return predictors, targets


def test_run_ols_excludes_imputed_training_scores_three_subsets_and_saves(tmp_path):
    predictors, targets = synthetic_tables()
    result = run_ols(FORMULA, **YEARS, name="classic_ols", predictors=predictors, targets=targets, save=True,
                     results_dir=tmp_path)

    assert result.counts["n_train"] == 9 and result.counts["excluded"] == {"train:imputed_target": 3}
    assert result.model.params.round(8).tolist() == [1, 2, 0.5, -3]
    m = result.metrics
    assert (m.loc["non_imputed", "n"], m.loc["non_imputed", "mae"]) == (3, pytest.approx(0, abs=1e-8))
    # NC-3 is predicted on the true line, 4 below its imputed actual: errors 0, 0, 0, -4
    assert (m.loc["imputed", "n"], m.loc["imputed", "bias"]) == (1, pytest.approx(-4))
    assert m.loc["all"].tolist() == pytest.approx([4, 1, 2, -1])
    predicted = result.predictions[result.predictions["role"] == "predict"]
    assert len(predicted) == 4 and predicted["actual"].isna().all() and predicted["prediction"].notna().all()

    assert result.run_id == "classic_ols-t06-thru-10-v12-p14"
    again = run_ols(FORMULA, **YEARS, name="classic_ols", predictors=predictors, targets=targets, save=True,
                    results_dir=tmp_path)
    assert again.run_id == "classic_ols-t06-thru-10-v12-p14-2"
    loaded = tracking.load_run(result.run_id, tmp_path)
    assert loaded.config == result.config and loaded.counts == result.counts
    assert loaded.metrics.equals(result.metrics)
    assert loaded.predictions.equals(result.predictions)
    assert len(tracking.load_runs(tmp_path)) == 2


def test_missing_predictors_are_excluded_with_reasons_and_ablations_can_share_rows():
    predictors, targets = synthetic_tables()
    predictors.loc[(predictors["year"] == 2006) & (predictors["district"] == 1), "college"] = np.nan
    predictors.loc[(predictors["year"] == 2012) & (predictors["district"] == 2), "college"] = np.inf
    data = {"predictors": predictors, "targets": targets}

    with pytest.warns(UserWarning, match="missing:college"):
        full = run_ols(FORMULA + " + college", **YEARS, **data)
    assert full.counts["excluded"] == {"train:imputed_target": 3, "train:missing:college": 1,
                                       "eval:missing:college": 1}
    assert (full.counts["n_train"], full.counts["n_eval"]) == (8, 3)

    plain = run_ols(FORMULA, **YEARS, **data)
    with pytest.warns(UserWarning):
        shared = run_ols(FORMULA, **YEARS, **data, require_complete=["college"])
    compared = tracking.compare([full, plain, shared])
    assert compared["same_train_rows"].tolist() == [True, False, True]
    assert compared["same_eval_rows"].tolist() == [True, False, True]
    assert compared["change"].tolist() == ["(base)", "- college", "- college"]
    assert compared.index.tolist() == ["ols", "ols #2", "ols #3"]
    assert len(tracking.compare(([full, plain]), full=True).columns) > len(compared.columns)
    by_district = tracking.compare_predictions([full, plain])
    assert len(by_district) == 3 and list(by_district.columns)[-2:] == ["ols", "ols #2"]
    formulas = [FORMULA + " + college", f"{TARGET} ~ partisan_lean * hispanic + C(state)"]
    assert evaluation.shared_columns(formulas, predictors.columns) == \
        ["gen_margin_raw", "partisan_lean", "inc_signed", "college", "state"]  # hispanic isn't a column here


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
    }


def test_excluded_by_year_lists_districts_with_a_total():
    excluded = pd.DataFrame({"year": [2006, 2006, 2006, 2008], "state": ["NC", "AK", "AZ", "NC"],
                             "district": [3, 0, 12, 3], "role": "train",
                             "reason": ["imputed_target", "imputed_target", "imputed_target", "missing:college"]})
    table = evaluation.excluded_by_year(excluded, max_districts=2)
    assert table.index.tolist() == [2006, 2008, "total"]
    assert table["n"].tolist() == [3, 1, 4]
    assert table.loc[2006, "districts"] == "AK-AL, AZ-12, ... (+1)"
    assert table.loc[2008, ["reason", "districts"]].tolist() == ["missing:college", "NC-03"]
