"""Saved experiments: one folder per run in results/house/runs/, plus one summary row per run in results/house/runs.csv.

A run's folder is named {name}-t{train years}[-v{eval years}][-p{predict years}], e.g. classic_ols-t06-thru-20-v22,
classic_ols-t08,12,16,20-v22, or classic_ols-t06-thru-24-p26 (name defaults to the estimator, e.g. "ols"); a "-2",
"-3", ... suffix keeps names unique, so a run is never overwritten. The folder name is the run_id. It holds:

    config.json          what was run: estimator, formula, target, full years, holdout use, timestamp, git commit,
                         data hash; rerun from this, not the folder name
    metrics.json         metrics by subset (valid, then the competitive subsets) and eval year, plus pooled ("all"),
                         and coverage (eval districts by exclusion reason; both null for a prediction-only run),
                         counts (sample sizes, exclusions by reason, row hashes), and fit diagnostics
    coefficients.csv     one row per model term
    predictions.parquet  one row per district: year, state, district, role (train / eval / predict), prediction,
                         actual, residual (actual - prediction)
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
from .evaluation import SUMMARY_METRICS, VALID, rows_hash

RESULTS_DIR = ROOT / "results/house"
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


def competitive_text(config: dict) -> str:
    """A config's competitive subsets, e.g. "competitive_ex_ante: abs(partisan_lean) <= 10"; "-" if none."""
    return "; ".join(f"{name}: abs({column}) <= {threshold:g}"
                     for name, (column, threshold) in config.get("competitive", {}).items()) or "-"


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
    if "weights" in config:
        model["weights"] = weights_text(config)
    if config.get("fixed"):
        model["fixed"] = ", ".join(f"{column} = {c:g}" for column, c in config["fixed"].items())
        model["fit_formula"] = config["fit_formula"]
    data = {k: paint(", ".join(map(str, config[k])), DESCRIBE_COLORS[k]) if config[k] else "-" for k in YEAR_KEYS}
    data |= {
        "variables": "\n".join(f"{paint(t, TRANCHE_COLORS.get(t, '39'))}: [{', '.join(names)}]"
                               for t, names in tranches.items()) or "-",
        "require_complete": ", ".join(config["require_complete"]) or "-",
        "competitive": competitive_text(config).replace("; ", "\n"),
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
    """{"mae_valid": ..., "mae_competitive_ex_ante": ...}: SUMMARY_METRICS of each subset, pooled over the eval years
    ("all"); only valid, with n=0 and NaN, when unscored."""
    if metrics is None:
        return {f"{m}_{VALID}": 0 if m == "n" else float("nan") for m in SUMMARY_METRICS}
    out = {}
    for subset in metrics.index.get_level_values("subset").unique():
        for m in SUMMARY_METRICS:
            value = metrics.loc[(subset, "all"), m]
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
        "competitive": competitive_text(config),  # mae_competitive_* of runs with different thresholds don't compare
        **{k: counts[k] for k in ["n_train", "n_eval", "n_eval_scored", "n_eval_no_target", "n_predict"]},
        **excluded,
        **_metric_values(result.metrics),
        **{f"fit_{k}": v for k, v in result.fit.items()},
        **{k: counts[k] for k in ["train_rows_hash", "eval_rows_hash"]},
        **{k: config.get(k) for k in ["data_hash", "git_commit", "git_dirty"]},
    }


def _records(table: pd.DataFrame | None) -> list[dict] | None:
    """A metrics or coverage table as JSON records (index columns first), NaN as null."""
    if table is None:
        return None
    return [{k: None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
            for row in table.reset_index().to_dict("records")]


def _from_records(records: list[dict] | None, index: list[str]) -> pd.DataFrame | None:
    if records is None:
        return None
    return pd.DataFrame.from_records(records).set_index(index).apply(pd.to_numeric)  # null back to NaN


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
    metrics = {"metrics": _records(result.metrics), "coverage": _records(result.coverage), "counts": result.counts,
               "fit": result.fit}
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
    result.coefficients.to_csv(folder / "coefficients.csv", index_label="term")
    result.predictions.to_parquet(folder / "predictions.parquet", index=False)
    result.excluded.to_csv(folder / "excluded.csv", index=False)

    path = Path(results_dir) / "runs.csv"
    runs = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([runs, pd.DataFrame([summary_row(result)])], ignore_index=True).to_csv(path, index=False)
    return run_id


def load_runs(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Every saved run, oldest first; e.g. load_runs().sort_values("mae_valid"). Empty (no rows, no columns) until a
    run is saved, when results_dir/runs.csv doesn't exist yet."""
    path = Path(results_dir) / "runs.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_run(run_id: str, results_dir: Path = RESULTS_DIR) -> SimpleNamespace:
    """A saved run with the same attributes as a run_ols result, except the fitted model and `scored` (rebuild it with
    evaluation.evaluate_predictions from the saved predictions and the targets)."""
    folder = Path(results_dir) / "runs" / run_id
    saved = json.loads((folder / "metrics.json").read_text())
    return SimpleNamespace(
        run_id=run_id,
        config=json.loads((folder / "config.json").read_text()),
        metrics=_from_records(saved["metrics"], ["subset", "year"]),
        coverage=_from_records(saved["coverage"], ["year"]),
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


def weights_text(config: dict) -> str:
    """A run's estimator and WLS weighting, e.g. "wls: equal_year, half_life 8, midterm x1.25"; "ols" for OLS."""
    weights = config.get("weights") or {}
    parts = (["equal_year"] if weights.get("equal_year") else []) + \
        ([f"half_life {weights['half_life']:g}"] if weights.get("half_life") is not None else []) + \
        ([f"midterm x{weights['midterm_weight']:g}"] if weights.get("midterm_weight", 1) != 1 else [])
    return config.get("estimator", "ols") + (f": {', '.join(parts)}" if "weights" in config else "")


def describe_change(config: dict, base: dict) -> str:
    """How a run differs from `base`: its target, formula terms removed (-) and added (+), and years; "(same)" if
    none. a*b counts as its terms a, b, and a:b."""
    parts = [f"target {config['target']}"] if config["target"] != base["target"] else []
    if (config.get("estimator"), config.get("weights")) != (base.get("estimator"), base.get("weights")):
        parts.append(weights_text(config))
    if config.get("fixed") != base.get("fixed"):
        parts.append("fixed " + (", ".join(f"{column} = {c:g}" for column, c in config["fixed"].items())
                                 if config.get("fixed") else "none"))
    if config.get("formula") and base.get("formula"):
        terms, base_terms = _terms(config["formula"]), _terms(base["formula"])
        parts += [f"- {t}" for t in base_terms if t not in terms] + [f"+ {t}" for t in terms if t not in base_terms]
    for key, prefix in zip(YEAR_KEYS, "tvp"):
        if sorted(config[key]) != sorted(base[key]):
            parts.append(prefix + abbreviate_years(config[key]) if config[key] else f"no {key}")
    return " ".join(parts) or "(same)"


COMPARE_METRICS = ["n", "mae", "rmse", "bias", "accuracy_pct"]  # shown per subset by compare


def compare(runs: list, results_dir: Path = RESULTS_DIR, full: bool = False, subsets=(VALID,)) -> pd.DataFrame:
    """One row per run (run_ids and/or results, saved or not), against the first run, labeled by run name:
    - change: what differs from the first run (describe_change);
    - n_train, then COMPARE_METRICS (pooled over eval years) for each of `subsets` ("valid" and/or the runs'
      competitive subset names), and adjusted R2;
    - same_train_rows / same_eval_rows: whether the run used the first run's training and scored evaluation rows, so a
      formula change isn't confused with a change in sample.
    full=True gives every runs.csv column instead, labeled like compare_predictions. highlight() formats either one
    for a notebook."""
    labeled = _runs(runs, results_dir)
    summaries = pd.DataFrame([summary_row(run) for run in labeled.values()], index=list(labeled))
    if full:
        return summaries
    unknown = [s for s in subsets if f"n_{s}" not in summaries.columns]
    if unknown:
        raise ValueError(f"Subsets {unknown} weren't scored by these runs (their competitive subsets: "
                         f"{[c.removeprefix('n_') for c in summaries.columns if c.startswith('n_competitive')]})")
    first, base = summaries.iloc[0], next(iter(labeled.values())).config
    out = summaries[["n_train", *[f"{m}_{s}" for s in subsets for m in COMPARE_METRICS], "fit_r2_adj"]].copy()
    out.insert(0, "change", [describe_change(run.config, base) for run in labeled.values()])
    out.iloc[0, 0] = "(base)"
    out["same_train_rows"] = summaries["train_rows_hash"] == first["train_rows_hash"]
    out["same_eval_rows"] = summaries["eval_rows_hash"] == first["eval_rows_hash"]
    out.index = pd.Index(_names(labeled.values()), name="run")
    return out


def _header(column: str, subsets: list[str]) -> tuple[str, str]:
    """A two-level header: mae_valid -> (valid, mae) for each of `subsets`, fit_r2_adj -> (fit, r2_adj),
    same_train_rows -> (same rows, train), anything else -> ("", column)."""
    for subset in subsets:
        for metric in SUMMARY_METRICS:
            if column == f"{metric}_{subset}":
                return subset, metric
    if column.startswith("fit_"):
        return "fit", column.removeprefix("fit_")
    if column.startswith("same_") and column.endswith("_rows"):
        return "same rows", column.removeprefix("same_").removesuffix("_rows")
    return "", column


def highlight(compared: pd.DataFrame):
    """compare()'s table for a notebook (a pandas Styler): headers grouped by subset, metrics to 2 decimals, the best
    MAE, RMSE, |bias|, and accuracy of each subset in bold, and same rows as ✓ (a different sample as a red ✗)."""
    subsets = [c.removeprefix("mae_") for c in compared.columns if c.startswith("mae_")]
    shown = compared.set_axis(pd.MultiIndex.from_tuples([_header(c, subsets) for c in compared.columns]), axis=1)
    formats = {}
    for column in shown.columns:
        metric = column[1]
        if column[0] == "same rows":
            formats[column] = lambda v: "✓" if v else "✗"
        elif metric in ("mae", "rmse", "missed_pct", "accuracy_pct", "precision_D_pct", "precision_R_pct"):
            formats[column] = "{:.1f}" if metric.endswith("_pct") else "{:.2f}"
        elif metric == "bias":
            formats[column] = "{:+.2f}"
        elif pd.api.types.is_float_dtype(shown[column]):
            formats[column] = "{:.3f}"
    styler = shown.style.format(formats, na_rep="-")  # one call: each format() call resets columns it doesn't list

    def bold_best(col: pd.Series) -> list[str]:  # lowest MAE / RMSE / |bias|, highest accuracy
        values = {"bias": col.abs(), "accuracy_pct": -col}.get(col.name[1], col)
        return ["font-weight: bold" if v == values.min() else "" for v in values]

    best = [c for c in shown.columns if c[0] in subsets and c[1] in ("mae", "rmse", "bias", "accuracy_pct")]
    styler = styler.apply(bold_best, subset=best)
    same = [c for c in shown.columns if c[0] == "same rows"]
    styler = styler.map(lambda v: "" if v else "color: #d03b3b; font-weight: bold", subset=same)
    if ("", "change") in shown.columns:
        styler = styler.set_properties(subset=[("", "change")], **{"text-align": "left", "white-space": "pre"})
    return styler


BEST_OF_METRICS = ["n", "mae", "rmse", "bias", "accuracy_pct", "precision_D_pct", "precision_R_pct", "missed_races",
                   "missed_pct"]
HIGHER_IS_BETTER = ["accuracy_pct", "precision_D_pct", "precision_R_pct"]  # rank_by: highest first; bias by |bias|
GOOD, BAD = "#0ca30c", "#d03b3b"  # improvement / regression text, always with a ✓ / ✗ marker


def _rows(run) -> tuple[str, str]:
    return run.counts["train_rows_hash"], run.counts["eval_rows_hash"]


def _same_model(a, b) -> bool:
    """Whether two runs are the same model on the same rows: estimator, formula, fixed coefficients, weights, rows."""
    def spec(run) -> tuple:
        return tuple(run.config.get(k) for k in ["estimator", "formula", "fixed", "weights"]) + _rows(run)
    return spec(a) == spec(b)


def _check_same_rows(block: str, runs: list) -> None:
    """Raise unless every run of a block trains and scores exactly its base's (first run's) rows."""
    for run in runs[1:]:
        if _rows(run) != _rows(runs[0]):
            raise ValueError(f"{block}: {run.config['name']} doesn't use its base's rows; give the block's runs the "
                             "same require_complete")


def best_of_blocks(blocks: dict, subset: str = VALID, rank_by: str = "mae", top: int | None = 2,
                   tiebreak: str = "mae") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(metrics, coefficients) for ablation blocks ({block name: list of results, its base first}) on one subset,
    pooled over the eval years, indexed by (block, run):

    - each block's `top` variants (None: all) by `rank_by` (lowest MAE / RMSE / |bias| / missed, highest accuracy or
      precision; ties by `tiebreak`, lowest first), after the first block's base. Another block's base is shown too when its model (estimator, formula, fixed coefficients, weights) or rows differ
      from the first's (e.g. the FEC block, without the districts FEC has no data for), so every delta has its base;
    - metrics: n_train, BEST_OF_METRICS, then Δ<metric> = variant - its own block's base, which scores exactly the
      same rows (raises otherwise); blank for base rows. Signed bias gets Δ|bias| = |variant bias| - |base bias|
      instead, so a negative value always means less bias. style_best_of marks each Δ as an improvement or regression;
    - coefficients: each shown run's coefficients (columns: terms, blank where a run doesn't use the term)."""
    first_base = next(iter(blocks.values()))[0]
    rows, coefficients = {}, {}
    for block, runs in blocks.items():
        _check_same_rows(block, runs)
        base, variants = runs[0], runs[1:]

        def values(run) -> pd.Series:
            return run.metrics.loc[(subset, "all"), BEST_OF_METRICS]

        def deltas(run, base) -> pd.Series:
            if run is base:
                return pd.Series(dtype=float)
            out = (values(run) - values(base)).drop(["n", "bias"])
            out["|bias|"] = abs(values(run)["bias"]) - abs(values(base)["bias"])
            return out.add_prefix("Δ")

        ranked = sorted(variants, key=lambda run: (
            -values(run)[rank_by] if rank_by in HIGHER_IS_BETTER else abs(values(run)[rank_by]),
            values(run)[tiebreak]))
        same_as_first = _same_model(base, first_base)
        shown = ([base] if base is first_base or not same_as_first else []) + ranked[:top]
        for run in shown:
            rows[(block, run.config["name"])] = pd.concat([pd.Series({"n_train": run.counts["n_train"]}), values(run),
                                                           deltas(run, base)])
            coefficients[(block, run.config["name"])] = run.coefficients["coef"]
    metrics = pd.DataFrame.from_dict(rows, orient="index")
    metrics.index.names = ["block", "run"]
    coefficients = pd.DataFrame.from_dict(coefficients, orient="index")
    coefficients.index.names = ["block", "run"]
    return metrics, coefficients


def rolling_best_of(windows: dict, subset: str = VALID, rank_by: str = "mae", top: int | None = 2,
                    tiebreak: str = "mae") -> tuple[pd.DataFrame, pd.DataFrame]:
    """best_of_blocks for a rolling backtest: windows = {window label, e.g. its eval year: ablation blocks as
    best_of_blocks takes them}, with the same blocks and runs (by name) in every window. Each run is scored on `subset`
    in each window, then averaged over the windows with equal weight. Indexed by (block, run):

    - metrics: the means of n_train and BEST_OF_METRICS; |bias|, the mean of each window's |bias| (so opposite-signed
      biases in different years don't cancel); <rank_by>_<window> for each window; then Δ<metric> = variant - its
      block's base, of the means (Δ|bias| of the mean |bias|). In every window each variant scores exactly its base's
      rows (raises otherwise). Runs are ranked and shown as in best_of_blocks, by the mean (rank_by="bias" ranks by
      |bias|; ties by the mean `tiebreak`); another block's base is shown when its model or rows differ from the first block's in any window;
    - coefficients: each shown run's coefficients averaged over the windows."""
    values, coefs = {}, {}
    for window, blocks in windows.items():
        for block, runs in blocks.items():
            _check_same_rows(block, runs)
            for run in runs:
                v = run.metrics.loc[(subset, "all"), BEST_OF_METRICS]
                values[(block, run.config["name"], window)] = pd.concat(
                    [pd.Series({"n_train": run.counts["n_train"]}), v, pd.Series({"|bias|": abs(v["bias"])})])
                coefs[(block, run.config["name"], window)] = run.coefficients["coef"]
    by_window = pd.DataFrame.from_dict(values, orient="index").rename_axis(["block", "run", "window"])
    mean = by_window.groupby(level=["block", "run"], sort=False).mean()
    windowed = by_window[rank_by].unstack("window")[list(windows)].add_prefix(f"{rank_by}_")
    coefficients = (pd.DataFrame.from_dict(coefs, orient="index").rename_axis(["block", "run", "window"])
                    .groupby(level=["block", "run"], sort=False).mean())

    first_windows = next(iter(windows.values()))
    first_block = next(iter(first_windows))
    key = "|bias|" if rank_by == "bias" else rank_by
    rows = {}
    for block, runs in first_windows.items():
        base, names = runs[0].config["name"], [run.config["name"] for run in runs[1:]]
        ranked = sorted(names, key=lambda name: (-mean.loc[(block, name), key] if key in HIGHER_IS_BETTER
                                                 else mean.loc[(block, name), key],
                                                 mean.loc[(block, name), tiebreak]))
        same_as_first = all(_same_model(blocks[block][0], blocks[first_block][0]) for blocks in windows.values())
        for name in ([base] if block == first_block or not same_as_first else []) + ranked[:top]:
            row = pd.concat([mean.loc[(block, name)], windowed.loc[(block, name)]])
            if name != base:
                delta = (mean.loc[(block, name)] - mean.loc[(block, base)]).drop(["n_train", "n", "bias"])
                row = pd.concat([row, delta.add_prefix("Δ")])
            rows[(block, name)] = row
    metrics = pd.DataFrame.from_dict(rows, orient="index").rename_axis(["block", "run"])
    order = ["n_train", *BEST_OF_METRICS[:4], "|bias|", *BEST_OF_METRICS[4:], *windowed.columns]
    metrics = metrics[order + [c for c in metrics.columns if c.startswith("Δ")]]
    return metrics, coefficients.loc[metrics.index]


def style_best_of(metrics: pd.DataFrame):
    """best_of_blocks' metrics for a notebook (a pandas Styler): signed bias and deltas with their sign, each Δ marked
    ✓ (green) when it improves on the block's base and ✗ (red) when it's worse, by the metric's direction (lower
    MAE, RMSE, |bias|, and misses; higher accuracy and precision); a zero Δ is unmarked. Two-level columns, e.g.
    (subset, metric) side by side, are styled by their last level."""
    def metric_of(column) -> str:
        return column[-1] if isinstance(column, tuple) else column

    def delta_format(higher_is_better: bool, digits: int):
        def fmt(v) -> str:
            if pd.isna(v):
                return "-"
            better = v > 0 if higher_is_better else v < 0
            return f"{v:+.{digits}f}" + ("" if v == 0 else " ✓" if better else " ✗")
        return fmt

    plain = {"n_train": "{:.0f}", "n": "{:.0f}", "missed_races": "{:.0f}", "bias": "{:+.2f}"}
    formats = {column: plain[metric_of(column)] for column in metrics.columns if metric_of(column) in plain}
    deltas = [column for column in metrics.columns if metric_of(column).startswith("Δ")]
    for column in deltas:
        metric = metric_of(column).removeprefix("Δ")
        formats[column] = delta_format(metric in HIGHER_IS_BETTER, 0 if metric == "missed_races" else 2)

    def color(col: pd.Series) -> list[str]:
        higher = metric_of(col.name).removeprefix("Δ") in HIGHER_IS_BETTER
        return ["" if pd.isna(v) or v == 0 else f"color: {GOOD if (v > 0) == higher else BAD}" for v in col]

    return metrics.style.format(formats, precision=2, na_rep="-").apply(color, subset=deltas)


def compare_predictions(runs: list, results_dir: Path = RESULTS_DIR, values: str = "prediction") -> pd.DataFrame:
    """District by district: one row per evaluation district every run predicted (an inner join on year, state,
    district), with its actual target value (`actual`, e.g. the observed adj_margin_2p_norm), then
    each run's prediction for that district in a column named after the run (as in compare).
    values="residual" gives each run's residual (actual - prediction; positive: Democrats outperformed the run's
    forecast) instead; NaN where there is no actual."""
    labeled = _runs(runs, results_dir)
    out = None
    for name, run in zip(_names(labeled.values()), labeled.values()):
        rows = run.predictions[run.predictions["role"] == "eval"]
        if out is None:
            out = rows[[*KEY, "actual"]]
        out = out.merge(rows[[*KEY, values]].rename(columns={values: name}), on=KEY)
    return out
