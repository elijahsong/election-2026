import copy

import pytest
from test_stage1_regression import TARGET, tables

from experiments.house import tracking
from experiments.house.configs import MODEL_CONFIGS
from experiments.house.run import make_config, parse_args, run_experiment
from house.regression.schema import MODELED_IMPUTATION_METHODS


def synthetic_tables():
    """SYNTHETIC: targets on the true line for 2006-2012, NC-3 imputed 4 points above it, and 2014 predictors only."""
    predictors, targets = tables((2006, 2008, 2010, 2012, 2014))
    targets = targets[targets["year"] < 2014].copy()
    modeled = (targets["state"] == "NC") & (targets["district"] == 3)
    targets.loc[modeled, TARGET] += 4
    targets.loc[modeled, "imputation_method"] = MODELED_IMPUTATION_METHODS[0]
    return predictors, targets


def test_run_scores_both_target_sets_and_saves_without_overwriting(tmp_path):
    predictors, targets = synthetic_tables()
    years = {"train_years": [2006, 2008, 2010], "eval_years": [2012], "predict_years": [2014]}
    result = run_experiment("classic_ols", **years, predictors=predictors, targets=targets, save=True,
                            results_dir=tmp_path)

    assert result.run_id == "classic_ols-t06-thru-10-v12-p14"
    assert result.summary["n_train"] == 9  # modeled imputations left out of training
    assert result.coefficients["coef"].round(8).tolist() == [1, 2, 0.5, -3]
    s = result.summary
    assert (s["n_non_imputed"], s["mae_non_imputed"]) == (3, pytest.approx(0, abs=1e-8))
    # NC-3 is predicted on the true line, 4 below its imputed actual: errors 0, 0, 0, -4
    assert (s["n_all"], s["mae_all"], s["rmse_all"], s["bias_all"]) == pytest.approx((4, 1, 2, -1))
    predicted = result.predictions[result.predictions["role"] == "predict"]
    assert len(predicted) == 4 and predicted["actual"].isna().all() and predicted["prediction"].notna().all()

    again = run_experiment("classic_ols", **years, predictors=predictors, targets=targets, save=True,
                           results_dir=tmp_path)
    assert again.run_id == "classic_ols-t06-thru-10-v12-p14-2"
    loaded = tracking.load_run(result.run_id, tmp_path)
    assert loaded.config == result.config and loaded.summary == result.summary
    assert len(tracking.load_runs(tmp_path)) == 2

    ablation = run_experiment("classic_ols", **years, drop_features=["inc_signed"], name="no_inc",
                              predictors=predictors, targets=targets)  # not saved
    train_all = run_experiment("classic_ols", **years, training_inputs="all", predictors=predictors, targets=targets)
    compared = tracking.compare([result.run_id, ablation, train_all], tmp_path)
    assert compared["same_eval_rows"].all()
    assert compared["same_train_rows"].tolist() == [True, True, False]
    assert len(tracking.compare_predictions([result.run_id, ablation], tmp_path)) == 4


def test_feature_changes_resolve_in_order_without_touching_the_spec():
    before = copy.deepcopy(MODEL_CONFIGS)
    config = make_config("classic_ols", [2006], [2008], add_features=["college", "gen_margin_raw"],
                         drop_features=["inc_signed"])
    assert config["features"] == ["gen_margin_raw", "partisan_lean", "college"]
    assert make_config("classic_ols", [2006], [2008], features=["college"], add_features=["hispanic"])["features"] \
        == ["college", "hispanic"]
    assert MODEL_CONFIGS == before
    with pytest.raises(ValueError, match="drop_features"):
        make_config("classic_ols", [2006], [2008], drop_features=["incumbent"])


def test_run_names():
    def name(train, eval_years=(), predict_years=(), suffix=None):
        return tracking.run_name(make_config("classic_ols", train, eval_years, predict_years, name=suffix))

    assert name(range(2006, 2022, 2), [2022]) == "classic_ols-t06-thru-20-v22"
    assert name([2008, 2012, 2016, 2020], [2022]) == "classic_ols-t08,12,16,20-v22"
    assert name(range(2006, 2026, 2), predict_years=[2026], suffix="prod") == "classic_ols-t06-thru-24-p26-prod"


def test_cli_matches_python_api_and_holdout_stays_protected():
    argv = ["--model", "classic_ols", "--train-years", "2008,2010,2012", "--eval-years", "2022",
            "--add-features", "college", "--inputs", "all", "--name", "extra"]
    assert parse_args(argv) == {
        "model": "classic_ols", "train_years": [2008, 2010, 2012], "eval_years": [2022], "predict_years": [],
        "features": None, "add_features": ["college"], "drop_features": [], "training_inputs": "all",
        "use_holdout": False, "name": "extra", "save": True,
    }
    predictors, targets = tables((2020, 2022, 2024))
    with pytest.raises(ValueError, match="use_holdout"):  # 2024 in training, not just evaluation
        run_experiment("classic_ols", train_years=[2022, 2024], predict_years=[2020], predictors=predictors,
                       targets=targets)
