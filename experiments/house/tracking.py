"""Saved experiments: one folder per run in results/house/runs/, plus one summary row per run in results/house/runs.csv.

A run's folder is named {model}-t{train years}-v{eval years}[-p{predict years}][-{name}], e.g.
classic_ols-t06-thru-20-v22 or classic_ols-t08,12,16,20-v22-no_inc; a "-2", "-3", ... suffix keeps names unique.
The folder name is the run_id. It holds:

    config.json        the fully resolved config (exact features, full years); rerun from this, not the folder name
    summary.json       the runs.csv row: sample sizes, metrics, fit statistics, row hashes, data hash, git commit
    metrics.csv        n, MAE, RMSE, bias by evaluation targets ("all" / "non_imputed") and year
    coefficients.csv   one row per model term
    predictions.csv    one row per district: year, state, district, role ("eval" / "predict"), prediction, actual,
                       imputed, residual

Two runs used the same training (evaluation) rows exactly when their train_rows_hash (eval_rows_hash) match.
"""

import json
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from house.regression.schema import KEY

from . import ROOT

RESULTS_DIR = ROOT / "results/house"
METRICS = [f"{m}_{t}" for t in ["all", "non_imputed"] for m in ["n", "mae", "rmse", "bias"]]


def abbreviate_years(years) -> str:
    """Two-digit years: "06-thru-20" for an unbroken biennial run of 3+ years, else comma-separated ("08,12,16")."""
    years = sorted(years)
    if len(years) >= 3 and all(b - a == 2 for a, b in pairwise(years)):
        return f"{years[0] % 100:02d}-thru-{years[-1] % 100:02d}"
    return ",".join(f"{y % 100:02d}" for y in years)


def run_name(config: dict) -> str:
    """The readable folder name for a config, before any collision suffix."""
    parts = [config["model"], "t" + abbreviate_years(config["train_years"])]
    parts += ["v" + abbreviate_years(config["eval_years"])] if config["eval_years"] else []
    parts += ["p" + abbreviate_years(config["predict_years"])] if config["predict_years"] else []
    parts += [config["name"]] if config.get("name") else []
    return "-".join(parts)


def save_run(result, results_dir: Path = RESULTS_DIR) -> str:
    """Write a run_experiment result to its own new folder and append it to runs.csv; returns (and sets) its run_id."""
    runs_dir = Path(results_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    base = run_id = run_name(result.config)
    i = 1
    while (runs_dir / run_id).exists():
        i += 1
        run_id = f"{base}-{i}"
    folder = runs_dir / run_id
    folder.mkdir()  # never exist_ok: an existing run is never overwritten
    result.run_id = result.summary["run_id"] = run_id
    (folder / "config.json").write_text(json.dumps(result.config, indent=2))
    (folder / "summary.json").write_text(json.dumps(result.summary, indent=2))
    result.metrics.to_csv(folder / "metrics.csv", index=False)
    result.coefficients.to_csv(folder / "coefficients.csv", index_label="term")
    result.predictions.to_csv(folder / "predictions.csv", index=False)

    path = Path(results_dir) / "runs.csv"
    runs = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([runs, pd.DataFrame([result.summary])], ignore_index=True).to_csv(path, index=False)
    return run_id


def load_runs(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Every saved run, oldest first; e.g. load_runs().sort_values("mae_non_imputed")."""
    return pd.read_csv(Path(results_dir) / "runs.csv")


def load_run(run_id: str, results_dir: Path = RESULTS_DIR) -> SimpleNamespace:
    """A saved run with the same attributes as a run_experiment result, except the fitted model."""
    folder = Path(results_dir) / "runs" / run_id
    return SimpleNamespace(
        run_id=run_id,
        config=json.loads((folder / "config.json").read_text()),
        summary=json.loads((folder / "summary.json").read_text()),
        metrics=pd.read_csv(folder / "metrics.csv"),
        coefficients=pd.read_csv(folder / "coefficients.csv", index_col="term"),
        predictions=pd.read_csv(folder / "predictions.csv", dtype={"state": str}),
    )


def _runs(runs: list, results_dir: Path) -> dict:
    """{label: run} for run_ids (loaded) and/or results. A saved run is labeled by its run_id, an unsaved one by its
    would-be folder name plus " (unsaved)"; repeated labels get " #2", " #3", ..."""
    out = {}
    for run in runs:
        run = load_run(run, results_dir) if isinstance(run, str) else run
        label = label_base = run.run_id or f"{run_name(run.config)} (unsaved)"
        i = 1
        while label in out:
            i += 1
            label = f"{label_base} #{i}"
        out[label] = run
    return out


def compare(runs: list, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Metrics side by side for run_ids and/or results, saved or not, with whether each run used the first run's
    training and evaluation rows (so a feature change isn't confused with a change in sample)."""
    summaries = pd.DataFrame({label: run.summary for label, run in _runs(runs, results_dir).items()}).T
    out = summaries[["features", "training_inputs", "n_train", *METRICS, "fit_r2"]].copy()
    out["same_train_rows"] = summaries["train_rows_hash"] == summaries["train_rows_hash"].iloc[0]
    out["same_eval_rows"] = summaries["eval_rows_hash"] == summaries["eval_rows_hash"].iloc[0]
    return out


def compare_predictions(runs: list, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Evaluation rows every run predicted (inner join on year, state, district): actual and imputed from the first
    run, then one prediction column per run."""
    out = None
    for label, run in _runs(runs, results_dir).items():
        rows = run.predictions[run.predictions["role"] == "eval"]
        if out is None:
            out = rows[[*KEY, "actual", "imputed"]]
        out = out.merge(rows[[*KEY, "prediction"]].rename(columns={"prediction": label}), on=KEY)
    return out
