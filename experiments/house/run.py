"""One experiment runner for notebooks and the command line.

Notebook (from notebooks/house, put the repository root on sys.path first):

    from experiments.house.run import run_experiment
    result = run_experiment(model="classic_ols", train_years=list(range(2006, 2022, 2)), eval_years=[2022],
                            drop_features=["inc_signed"], name="no_inc")       # save=False by default
    result.metrics; result.model.results.summary(); result.predictions.head()

CLI (from the repository root; saves by default, --no-save to skip):

    python -m experiments.house.run --model classic_ols --train-years 2006,2008,2010,2012,2014,2016,2018,2020 \\
        --eval-years 2022 --drop-features inc_signed --name no_inc

Both build the same config and call house.regression's run_experiment, which fits on train_years and predicts the
eval_years and predict_years together (and enforces HOLDOUT_YEARS: 2024 needs use_holdout / --use-holdout, in training
too). Only eval_years are scored, on two sets of targets, whatever the training inputs: "all" (observed and
modeled-imputed targets) and "non_imputed" (observed only, the rows hr.score scores). predict_years (e.g. 2026) need no
targets.
"""

import argparse
import hashlib
import subprocess
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

import house.regression as hr
from house.regression.backtest import TRAINING_INPUTS
from house.regression.schema import KEY, MODELED_IMPUTATION_METHODS

from . import ROOT, tracking
from .configs import MODEL_CONFIGS

METRIC_COLUMNS = ["targets", "year", "n", "mae", "rmse", "bias"]


@dataclass
class ExperimentResult:
    config: dict  # fully resolved: the model spec, exact features, years, training inputs (config.json when saved)
    model: object  # the fitted house.regression model, e.g. model.results.summary()
    predictions: pd.DataFrame  # year, state, district, role, prediction, actual, imputed, residual
    metrics: pd.DataFrame  # n, mae, rmse, bias by targets ("all" / "non_imputed") and year (and "all" years)
    coefficients: pd.DataFrame  # one row per term
    summary: dict  # the runs.csv row
    run_id: str | None = None  # set when saved


def resolve_features(spec: dict, features=None, add_features=(), drop_features=()) -> list[str]:
    """The predictors: `features` if given, else the spec's; then add_features appended and drop_features removed.
    Order is kept and duplicates dropped. Dropping a predictor that isn't there is an error (likely a typo)."""
    names = list(dict.fromkeys([*(features if features is not None else hr.resolve_features(spec)), *add_features]))
    unknown = [f for f in drop_features if f not in names]
    if unknown:
        raise ValueError(f"drop_features not among the predictors {names}: {unknown}")
    names = [f for f in names if f not in drop_features]
    if not names:
        raise ValueError("No predictors left")
    return names


def make_config(model: str, train_years, eval_years=(), predict_years=(), features=None, add_features=(),
                drop_features=(), training_inputs: str = "non_imputed", use_holdout: bool = False,
                name: str | None = None) -> dict:
    """The fully resolved config: a copy of MODEL_CONFIGS[model] with an explicit feature list and every run setting.
    It is a valid house.regression config ("features", no "feature_set")."""
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model!r}; expected one of {sorted(MODEL_CONFIGS)}")
    if not eval_years and not predict_years:
        raise ValueError("Give eval_years, predict_years, or both")
    spec = MODEL_CONFIGS[model]
    return {
        "model": model,
        **{k: v for k, v in spec.items() if k not in ("feature_set", "features")},
        "features": resolve_features(spec, features, add_features, drop_features),
        "add_features": list(add_features),
        "drop_features": list(drop_features),
        "train_years": [int(y) for y in train_years],
        "eval_years": [int(y) for y in eval_years],
        "predict_years": [int(y) for y in predict_years],
        "training_inputs": training_inputs,
        "use_holdout": use_holdout,
        "name": name,
    }


def outcomes(predictions: pd.DataFrame, targets: pd.DataFrame, config: dict) -> pd.DataFrame:
    """KEY + role, prediction, actual (the target, modeled imputations included), imputed, and residual (actual -
    prediction). predict_years rows are "predict" and keep no actual."""
    target = config["target"]
    actual = targets[KEY].assign(actual=targets[target],
                                 imputed=targets["imputation_method"].isin(MODELED_IMPUTATION_METHODS))
    out = predictions[KEY].assign(role="eval", prediction=predictions["prediction"])
    out = out.merge(actual, on=KEY, how="left", validate="one_to_one")
    out.loc[out["year"].isin(config["predict_years"]), ["role", "actual"]] = ["predict", None]
    out["imputed"] = out["imputed"].fillna(False).astype(bool)  # rows without targets (2026)
    out["residual"] = out["actual"] - out["prediction"]
    return out


def score(predictions: pd.DataFrame) -> pd.DataFrame:
    """hr.score of the "eval" rows with an actual, for "all" targets and for "non_imputed" ones."""
    evaluated = predictions[(predictions["role"] == "eval") & predictions["actual"].notna()]
    subsets = {"all": evaluated, "non_imputed": evaluated[~evaluated["imputed"]]}
    frames = [hr.score(rows).assign(targets=name) for name, rows in subsets.items() if not rows.empty]
    return pd.concat(frames, ignore_index=True)[METRIC_COLUMNS] if frames else pd.DataFrame(columns=METRIC_COLUMNS)


def _overall(metrics: pd.DataFrame) -> dict:
    """n, mae, rmse, bias over all evaluation years, as {"mae_all": ..., "mae_non_imputed": ...}; NaN when unscored."""
    out = {}
    for targets in ["all", "non_imputed"]:
        row = metrics[(metrics["targets"] == targets) & (metrics["year"] == "all")]
        for m in ["n", "mae", "rmse", "bias"]:
            out[f"{m}_{targets}"] = row[m].item() if len(row) else (0 if m == "n" else float("nan"))
    return out


def _hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()[:12]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def run_experiment(model: str, *, train_years, eval_years=(), predict_years=(), features=None, add_features=(),
                   drop_features=(), training_inputs: str = "non_imputed", use_holdout: bool = False,
                   name: str | None = None, predictors: pd.DataFrame | None = None,
                   targets: pd.DataFrame | None = None, save: bool = False,
                   results_dir=tracking.RESULTS_DIR) -> ExperimentResult:
    """Fit MODEL_CONFIGS[model] on train_years, predict eval_years and predict_years, and score the eval_years.

    predictors and targets default to hr.load_predictors() and hr.load_targets(); pass them to skip reloading.
    save=True writes the run to results/house (tracking.save_run); a result can also be saved later.
    """
    config = make_config(model, train_years, eval_years, predict_years, features, add_features, drop_features,
                         training_inputs, use_holdout, name)
    predictors = hr.load_predictors() if predictors is None else predictors
    targets = hr.load_targets() if targets is None else targets
    hr_config = {**config, "eval_years": config["eval_years"] + config["predict_years"]}
    train, _ = hr.experiment_data(predictors, targets, hr_config)
    raw, fitted = hr.run_experiment(predictors, targets, hr_config)
    predictions = outcomes(raw, targets, config)
    metrics = score(predictions)
    scored = predictions[(predictions["role"] == "eval") & predictions["actual"].notna()]
    summary = {
        "run_id": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "name": name,
        "model_type": config["model_type"],
        "target": config["target"],
        "features": ",".join(config["features"]),
        **{k: ",".join(map(str, config[k])) for k in ["train_years", "eval_years", "predict_years"]},
        "training_inputs": training_inputs,
        "use_holdout": use_holdout,
        "n_train": len(train),
        "n_predict": int((predictions["role"] == "predict").sum()),
        **_overall(metrics),
        **{f"fit_{k}": v for k, v in fitted.fit_stats().items()},
        "train_rows_hash": _hash(train[KEY].sort_values(KEY)),
        "eval_rows_hash": _hash(scored[[*KEY, "imputed"]].sort_values(KEY)),
        "data_hash": _hash(predictors) + "-" + _hash(targets),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    result = ExperimentResult(config, fitted, predictions, metrics, fitted.coefficients(), summary)
    if save:
        tracking.save_run(result, results_dir)
    return result


def _list(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _years(text: str) -> list[int]:
    return [int(y) for y in _list(text)]


def parse_args(argv=None) -> dict:
    """run_experiment's keyword arguments from the command line."""
    parser = argparse.ArgumentParser(prog="python -m experiments.house.run", description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--train-years", required=True, type=_years, help="comma-separated, e.g. 2006,2008,2010")
    parser.add_argument("--eval-years", type=_years, default=[], help="scored against actual targets")
    parser.add_argument("--predict-years", type=_years, default=[], help="predicted only; no targets needed")
    parser.add_argument("--features", type=_list, help="replace the model's predictors (comma-separated)")
    parser.add_argument("--add-features", type=_list, default=[])
    parser.add_argument("--drop-features", type=_list, default=[])
    parser.add_argument("--inputs", dest="training_inputs", choices=TRAINING_INPUTS, default="non_imputed",
                        help="training targets: non_imputed (observed only) or all (with modeled imputations)")
    parser.add_argument("--use-holdout", action="store_true", help="allow HOLDOUT_YEARS (2024)")
    parser.add_argument("--name", help="suffix for the run folder, e.g. no_inc")
    parser.add_argument("--no-save", dest="save", action="store_false")
    return vars(parser.parse_args(argv))


def main(argv=None) -> ExperimentResult:
    result = run_experiment(**parse_args(argv))
    print(hr.describe_config(result.config), end="\n\n")
    print(result.metrics.round(3).to_string(index=False) if len(result.metrics) else "No eval_years scored", end="\n\n")
    print(result.coefficients.round(3).to_string(), end="\n\n")
    print(f"Saved results/house/runs/{result.run_id}" if result.run_id else "Not saved (--no-save)")
    return result


if __name__ == "__main__":
    main()
