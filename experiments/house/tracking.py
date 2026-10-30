"""Saved experiments: one folder per run in results/house/runs/, plus one summary row per run in results/house/runs.csv.

A run's folder is named {name}-t{train years}[-v{eval years}][-p{predict years}], e.g. classic_ols-t06-thru-20-v22,
classic_ols-t08,12,16,20-v22, or classic_ols-t06-thru-24-p26 (name defaults to the estimator, e.g. "ols"); a "-2",
"-3", ... suffix keeps names unique, so a run is never overwritten. The folder name is the run_id. It holds:

    config.json          what was run: estimator, formula, target, full years, holdout use, timestamp, git commit,
                         data hash; rerun from this, not the folder name
    metrics.json         metrics by subset (non_imputed / imputed / all; null for a prediction-only run), counts
                         (sample sizes, exclusions by reason, row hashes), and fit diagnostics
    coefficients.csv     one row per model term
    predictions.parquet  one row per district: year, state, district, role (train / eval / predict), prediction,
                         actual, is_imputed, residual
    excluded.csv         every left-out row: year, state, district, role, reason

Two runs used the same training (scored evaluation) rows exactly when their train_rows_hash (eval_rows_hash) match.
Nothing here depends on the estimator: a result needs config, metrics, counts, fit, coefficients, predictions, excluded.
"""

import json
import re
import subprocess
import uuid
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from patsy import ModelDesc

from house.regression.config import (
    HOLDOUT_YEARS,
    MODEL_COLORS,
    TRANCHE_COLORS,
    YEAR_COLORS,
)
from house.regression.schema import COLUMNS, KEY

from . import ROOT
from .evaluation import METRICS as METRIC_NAMES
from .evaluation import SUBSETS, rows_hash

RESULTS_DIR = ROOT / "results/house"
METRICS = [f"{m}_{s}" for s in SUBSETS for m in METRIC_NAMES]
YEAR_KEYS = ["train_years", "eval_years", "predict_years"]
# ANSI codes for describe, printed bold: house.regression.config's colors, plus one for predict_years
DESCRIBE_COLORS = {**YEAR_COLORS, "predict_years": "94"}  # bright blue


def abbreviate_years(years) -> str:
    """Two-digit years: "06-thru-20" for an unbroken biennial run of 3+ years, else comma-separated ("08,12,16")."""
    years = sorted(years)
    if len(years) >= 3 and all(b - a == 2 for a, b in pairwise(years)):
        return f"{years[0] % 100:02d}-thru-{years[-1] % 100:02d}"
    return ",".join(f"{y % 100:02d}" for y in years)


def run_name(config: dict) -> str:
    """The readable folder name for a config, before any collision suffix."""
    parts = [config.get("name") or config["estimator"], "t" + abbreviate_years(config["train_years"])]
    parts += ["v" + abbreviate_years(config["eval_years"])] if config["eval_years"] else []
    parts += ["p" + abbreviate_years(config["predict_years"])] if config["predict_years"] else []
    return "-".join(parts)


def describe(result, color: bool = True) -> str:
    """A result (saved or not) for printing, in four sections separated by rules:
    - the model: estimator, formula, target, and name;
    - the data: years, the formula's variables grouped by tranche, require_complete, and holdout status;
    - the sample: rows per role and exclusions by reason;
    - provenance: run_id, time, git commit, data hash, and statsmodels version.
    Colored like hr.describe_config unless color=False."""
    config, counts = result.config, result.counts

    def paint(text: str, code: str) -> str:
        return f"\033[1;{code}m{text}\033[0m" if color else text

    tranches = {}
    for name in config.get("variables", []):
        column = COLUMNS.get(name)
        tranches.setdefault((column.tranche or column.role) if column else "unregistered", []).append(name)  # key: C(state)
    excluded = {}
    for key, n in counts["excluded"].items():
        role, reason = key.split(":", 1)
        excluded.setdefault(role, []).append(f"{reason} {n}")

    def rows(role: str, text: str) -> str:
        return text + (f" (excluded: {', '.join(excluded[role])})" if role in excluded else "")

    estimator = config["estimator"]
    model = {"estimator": paint(estimator, MODEL_COLORS.get(estimator, "39")), "formula": config.get("formula", "-"),
             "target": config["target"], "name": config["name"] or "-"}
    data = {k: paint(", ".join(map(str, config[k])), DESCRIBE_COLORS[k]) if config[k] else "-" for k in YEAR_KEYS}
    data |= {
        "variables": "\n".join(f"{paint(t, TRANCHE_COLORS.get(t, '39'))}: [{', '.join(names)}]"
                               for t, names in tranches.items()) or "-",
        "require_complete": ", ".join(config["require_complete"]) or "-",
        "use_holdout": f"{config['use_holdout']} (HOLDOUT_YEARS: {HOLDOUT_YEARS})",
    }
    sample = {"train": rows("train", f"{counts['n_train']} rows")}
    if config["eval_years"]:
        sample["eval"] = rows("eval", f"{counts['n_eval']} rows, {counts['n_eval_scored']} scored")
    if config["predict_years"]:
        sample["predict"] = rows("predict", f"{counts['n_predict']} rows")
    origin = {
        "run_id": result.run_id or "(unsaved)",
        "timestamp": config["timestamp"],
        "git_commit": f"{config['git_commit']}{' (dirty)' if config['git_dirty'] else ''}",
        "data_hash": config["data_hash"],
        "statsmodels": config.get("statsmodels", "-"),
    }
    sections = [model, data, sample, origin]
    width = max(len(name) for section in sections for name in section)
    visible = [re.sub(r"\033\[[\d;]*m", "", line) for section in sections for value in section.values()
               for line in value.split("\n")]
    rule = "=" * min(width + 2 + max(map(len, visible)), 80)

    def lines(values: dict) -> list[str]:
        return [f"{name:<{width}}  " + value.replace("\n", "\n" + " " * (width + 2)) for name, value in values.items()]

    return f"\n{rule}\n".join("\n".join(lines(section)) for section in sections)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def provenance(predictors: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """When, from which code, and on which data a run was made."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "uid": uuid.uuid4().hex[:8],
        "git_commit": _git("rev-parse", "--short", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain")),
        "data_hash": rows_hash(predictors) + "-" + rows_hash(targets),
    }


def _metric_values(metrics: pd.DataFrame | None) -> dict:
    """{"mae_non_imputed": ..., ...}; n=0 and NaN when unscored."""
    out = {}
    for subset in SUBSETS:
        for m in METRIC_NAMES:
            value = metrics.loc[subset, m] if metrics is not None else (0 if m == "n" else float("nan"))
            out[f"{m}_{subset}"] = value.item() if hasattr(value, "item") else value
    return out


def summary_row(result) -> dict:
    """The runs.csv row for a result (saved or not)."""
    config, counts = result.config, result.counts
    excluded = {f"excluded_{role}": sum(n for key, n in counts["excluded"].items() if key.startswith(role + ":"))
                for role in ["train", "eval", "predict"]}
    return {
        "run_id": result.run_id,
        **{k: config.get(k) for k in ["timestamp", "uid", "name", "estimator", "formula", "target"]},
        **{k: ",".join(map(str, config[k])) for k in YEAR_KEYS},
        "use_holdout": config["use_holdout"],
        "require_complete": ",".join(config.get("require_complete", [])),
        **{k: counts[k] for k in ["n_train", "n_eval", "n_eval_scored", "n_eval_no_target", "n_predict"]},
        **excluded,
        **_metric_values(result.metrics),
        **{f"fit_{k}": v for k, v in result.fit.items()},
        **{k: counts[k] for k in ["train_rows_hash", "eval_rows_hash"]},
        **{k: config.get(k) for k in ["data_hash", "git_commit", "git_dirty"]},
    }


def _metrics_json(metrics: pd.DataFrame | None) -> dict | None:
    if metrics is None:
        return None
    return {subset: {k: None if pd.isna(v) else v for k, v in row.items()}
            for subset, row in metrics.to_dict("index").items()}


def save_run(result, results_dir: Path = RESULTS_DIR) -> str:
    """Write a result to its own new folder and append it to runs.csv; returns (and sets) its run_id."""
    runs_dir = Path(results_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    base = run_id = run_name(result.config)
    i = 1
    while (runs_dir / run_id).exists():
        i += 1
        run_id = f"{base}-{i}"
    folder = runs_dir / run_id
    folder.mkdir()  # never exist_ok: an existing run is never overwritten
    result.run_id = result.config["run_id"] = run_id
    (folder / "config.json").write_text(json.dumps(result.config, indent=2))
    metrics = {"metrics": _metrics_json(result.metrics), "counts": result.counts, "fit": result.fit}
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
    result.coefficients.to_csv(folder / "coefficients.csv", index_label="term")
    result.predictions.to_parquet(folder / "predictions.parquet", index=False)
    result.excluded.to_csv(folder / "excluded.csv", index=False)

    path = Path(results_dir) / "runs.csv"
    runs = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([runs, pd.DataFrame([summary_row(result)])], ignore_index=True).to_csv(path, index=False)
    return run_id


def load_runs(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Every saved run, oldest first; e.g. load_runs().sort_values("mae_non_imputed")."""
    return pd.read_csv(Path(results_dir) / "runs.csv")


def load_run(run_id: str, results_dir: Path = RESULTS_DIR) -> SimpleNamespace:
    """A saved run with the same attributes as a run_ols result, except the fitted model."""
    folder = Path(results_dir) / "runs" / run_id
    saved = json.loads((folder / "metrics.json").read_text())
    metrics = None
    if saved["metrics"] is not None:
        metrics = pd.DataFrame.from_dict(saved["metrics"], orient="index").astype({"n": int, "mae": float,
                                                                                   "rmse": float, "bias": float})
        metrics.index.name = "subset"
    return SimpleNamespace(
        run_id=run_id,
        config=json.loads((folder / "config.json").read_text()),
        metrics=metrics,
        counts=saved["counts"],
        fit=saved["fit"],
        coefficients=pd.read_csv(folder / "coefficients.csv", index_col="term"),
        predictions=pd.read_parquet(folder / "predictions.parquet"),
        excluded=pd.read_csv(folder / "excluded.csv", dtype={"state": str}),
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


def _names(runs) -> list[str]:
    """Each run's name (or its estimator, unnamed), with repeats as "#2", "#3", ..."""
    names = []
    for run in runs:
        name = label = run.config.get("name") or run.config["estimator"]
        i = 1
        while label in names:
            i += 1
            label = f"{name} #{i}"
        names.append(label)
    return names


def _terms(formula: str) -> list[str]:
    """The right-hand side's patsy terms, e.g. ["Intercept", "partisan_lean", "partisan_lean:inc_signed"]."""
    return [term.name() for term in ModelDesc.from_formula(formula).rhs_termlist]


def describe_change(config: dict, base: dict) -> str:
    """How a run differs from `base`: its target, formula terms removed (-) and added (+), and years; "(same)" if
    none. a*b counts as its terms a, b, and a:b."""
    parts = [f"target {config['target']}"] if config["target"] != base["target"] else []
    if config.get("formula") and base.get("formula"):
        terms, base_terms = _terms(config["formula"]), _terms(base["formula"])
        parts += [f"- {t}" for t in base_terms if t not in terms] + [f"+ {t}" for t in terms if t not in base_terms]
    for key, prefix in zip(YEAR_KEYS, "tvp"):
        if sorted(config[key]) != sorted(base[key]):
            parts.append(prefix + abbreviate_years(config[key]) if config[key] else f"no {key}")
    return " ".join(parts) or "(same)"


COMPARE_COLUMNS = ["change", "n_train", "n_non_imputed", "mae_non_imputed", "rmse_non_imputed", "bias_non_imputed",
                   "n_imputed", "mae_imputed", "fit_r2_adj", "same_train_rows", "same_eval_rows"]


def compare(runs: list, results_dir: Path = RESULTS_DIR, full: bool = False) -> pd.DataFrame:
    """One row per run (run_ids and/or results, saved or not), against the first run, labeled by run name:
    - change: what differs from the first run (describe_change);
    - n_train, then the non_imputed metrics (primary) and the imputed n and MAE, and adjusted R2;
    - same_train_rows / same_eval_rows: whether the run used the first run's training and scored evaluation rows, so a
      formula change isn't confused with a change in sample.
    full=True gives every runs.csv column instead, labeled like compare_predictions. highlight() formats either one
    for a notebook."""
    labeled = _runs(runs, results_dir)
    summaries = pd.DataFrame([summary_row(run) for run in labeled.values()], index=list(labeled))
    if full:
        return summaries
    first, base = summaries.iloc[0], next(iter(labeled.values())).config
    out = summaries[COMPARE_COLUMNS[1:-2]].copy()
    out.insert(0, "change", [describe_change(run.config, base) for run in labeled.values()])
    out.iloc[0, 0] = "(base)"
    out["same_train_rows"] = summaries["train_rows_hash"] == first["train_rows_hash"]
    out["same_eval_rows"] = summaries["eval_rows_hash"] == first["eval_rows_hash"]
    out.index = pd.Index(_names(labeled.values()), name="run")
    return out


def _header(column: str) -> tuple[str, str]:
    """A two-level header: mae_non_imputed -> (non_imputed, mae), fit_r2_adj -> (fit, r2_adj),
    same_train_rows -> (same rows, train), anything else -> ("", column)."""
    for subset in SUBSETS:
        if column.endswith("_" + subset) and column.removesuffix("_" + subset) in METRIC_NAMES:
            return subset, column.removesuffix("_" + subset)
    if column.startswith("fit_"):
        return "fit", column.removeprefix("fit_")
    if column.startswith("same_") and column.endswith("_rows"):
        return "same rows", column.removeprefix("same_").removesuffix("_rows")
    return "", column


def highlight(compared: pd.DataFrame):
    """compare()'s table for a notebook (a pandas Styler): headers grouped by subset, metrics to 2 decimals, the best
    non_imputed MAE, RMSE, and |bias| in bold, and same rows as ✓ (a different sample as a red ✗)."""
    shown = compared.set_axis(pd.MultiIndex.from_tuples(map(_header, compared.columns)), axis=1)
    formats = {}
    for column in shown.columns:
        metric = column[1]
        if column[0] == "same rows":
            formats[column] = lambda v: "✓" if v else "✗"
        elif metric in ("mae", "rmse"):
            formats[column] = "{:.2f}"
        elif metric == "bias":
            formats[column] = "{:+.2f}"
        elif pd.api.types.is_float_dtype(shown[column]):
            formats[column] = "{:.3f}"
    styler = shown.style.format(formats, na_rep="-")  # one call: each format() call resets columns it doesn't list

    def bold_best(col: pd.Series) -> list[str]:
        values = col.abs() if col.name[1] == "bias" else col
        return ["font-weight: bold" if v == values.min() else "" for v in values]

    best = [c for c in [("non_imputed", m) for m in ("mae", "rmse", "bias")] if c in shown.columns]
    styler = styler.apply(bold_best, subset=best)
    same = [c for c in shown.columns if c[0] == "same rows"]
    styler = styler.map(lambda v: "" if v else "color: #d03b3b; font-weight: bold", subset=same)
    if ("", "change") in shown.columns:
        styler = styler.set_properties(subset=[("", "change")], **{"text-align": "left", "white-space": "pre"})
    return styler


def compare_predictions(runs: list, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """District by district: one row per evaluation district every run predicted (an inner join on year, state,
    district), with its actual target value (`actual`, e.g. the observed adj_margin_2p_norm) and is_imputed, then
    each run's prediction for that district in a column named after the run (as in compare)."""
    labeled = _runs(runs, results_dir)
    out = None
    for name, run in zip(_names(labeled.values()), labeled.values()):
        rows = run.predictions[run.predictions["role"] == "eval"]
        if out is None:
            out = rows[[*KEY, "actual", "is_imputed"]]
        out = out.merge(rows[[*KEY, "prediction"]].rename(columns={"prediction": name}), on=KEY)
    return out
