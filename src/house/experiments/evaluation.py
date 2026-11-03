"""Estimator-agnostic experiment steps: prepare the training / evaluation rows, check and assemble predictions, and
score them. run.run_ols uses them around smf.ols; a future estimator (run_wls, run_ridge, ...) can reuse them as is.

Rows are never dropped silently. prepare_data gives every left-out row of the requested years a reason, first match
wins:

    imputed_target   training and evaluation: a modeled imputation (MODELED_IMPUTATION_METHODS), never ground truth.
                     Scoring against it would reward agreeing with the partisan lean the imputation was built from.
                     Jungle and first-round RCV targets have no imputation_method, so they are non-imputed and kept
    missing_target   training only: no finite target
    missing:<var>    any role: <var>, a predictor the formula uses (or a require_complete column), is null or nonfinite

Evaluation is separate from training eligibility and works on any model's predictions (KEY + prediction): it never
refits. evaluate_predictions re-derives, from the targets table, which eval-year districts have a valid target, first
match wins:

    missing_target   no finite target (e.g. a year without targets, 2026)
    imputed_target   is_imputed is not False: a modeled imputation, never ground truth
    race_type:<t>    an observed target of a race type not in EVAL_RACE_TYPES (config.py)
    no_prediction    a valid target the run didn't predict (e.g. excluded upstream for a missing predictor)

and scores the rest, "valid", plus each competitive subset (config.COMPETITIVE_SUBSETS: abs(column) <= threshold, a
subset of valid), per eval year and pooled ("all"):

    n, mae, rmse, bias   residual = actual - predicted; bias is the mean residual, positive when Democrats
                         outperformed the forecast
    winner counts        by margin sign (metrics.winner_counts): n_classified excludes rows where either margin is
                         exactly 0 (n_actual_tie, n_predicted_tie); missed_races, missed_pct = 100 * missed /
                         n_classified, accuracy_pct, precision_D_pct / precision_R_pct (of the races predicted
                         for a party, the % it won), and the 2x2 confusion counts actual_<D|R>_pred_<D|R>
    n_winner_conflict    rows whose actual margin sign disagrees with targets.winner_party (e.g. an RCV race won
                         after the first round); classification still uses the margin sign

An empty subset has n=0, zero counts, and NaN metrics and percentages.

Each scored row's outcome is "correct", "dem_miss" (predicted R, the D won), "gop_miss" (predicted D, the R won), or
"tie" (either margin exactly 0: not classified). outcome_report and state_summary read it.

The residual is actual - predicted everywhere (predictions, scored, metrics, reports, and statsmodels' model.resid):
positive means Democrats outperformed the forecast, negative that Republicans did.
"""

import hashlib
import numbers
import warnings

import numpy as np
import pandas as pd
from patsy import ModelDesc
from patsy.eval import ast_names

import house.regression as hr
from house.regression import metrics
from house.regression.config import COMPETITIVE_SUBSETS, EVAL_RACE_TYPES
from house.regression.schema import KEY, MODELED_IMPUTATION_METHODS

VALID = "valid"  # the subset of every valid target; competitive subsets are named in config.COMPETITIVE_SUBSETS
SUMMARY_METRICS = ["n", "mae", "rmse", "bias", "missed_races", "missed_pct", "accuracy_pct", "precision_D_pct",
                   "precision_R_pct"]  # pooled per subset in runs.csv
MAX_ABS_MARGIN = 100  # margins are D-R points in [-100, 100]; predictions beyond are flagged, never clipped


def formula_variables(formula: str, columns) -> tuple[str, list[str]]:
    """(target, predictor columns) of a patsy formula: the left-hand side, which must be one bare column name (it is
    what predictions are scored against), and the names on the right-hand side that are among `columns`."""
    desc = ModelDesc.from_formula(formula)
    lhs = [factor.code for term in desc.lhs_termlist for factor in term.factors]
    if len(lhs) != 1 or not lhs[0].isidentifier():
        raise ValueError(f"The left-hand side must be one target column, e.g. adj_margin_2p_norm; got {lhs}")
    names = [name for term in desc.rhs_termlist for factor in term.factors for name in ast_names(factor.code)]
    return lhs[0], [name for name in dict.fromkeys(names) if name in set(columns)]


def shared_columns(formulas, columns) -> list[str]:
    """Every predictor column any of `formulas` uses, in first-use order: pass it as require_complete to each run so
    they all drop the same rows (a missing value in any of these columns excludes the row from every run)."""
    return list(dict.fromkeys(v for formula in formulas for v in formula_variables(formula, columns)[1]))


def _valid(values: pd.Series) -> pd.Series:
    """True where a value is present and, for a numeric column, finite."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.Series(np.isfinite(values.astype(float)), index=values.index)
    return values.notna()


def prepare_data(predictors: pd.DataFrame, targets: pd.DataFrame, target: str, variables, train_years,
                 eval_years=(), predict_years=(), *,
                 use_holdout: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(training rows, evaluation + prediction rows, excluded rows) for the listed years.

    The target is joined onto the predictors by key (hr.join_target checks the keys match and are unique), and rows
    with a modeled imputation are excluded from training and evaluation. The year lists must be disjoint, and HOLDOUT_YEARS (2024) need use_holdout=True in any of them.
    Every row gets a "role" (train / eval / predict); predict rows get a blank target, and eval rows without a target
    are kept (predicted, not scored). `excluded` lists each left-out row's KEY, role, and reason (module docstring).
    """
    years = {"train": [int(y) for y in train_years], "eval": [int(y) for y in eval_years],
             "predict": [int(y) for y in predict_years]}
    if not years["eval"] and not years["predict"]:
        raise ValueError("Give eval_years, predict_years, or both")
    for a, b in [("train", "eval"), ("train", "predict"), ("eval", "predict")]:
        overlap = sorted(set(years[a]) & set(years[b]))
        if overlap:
            raise ValueError(f"Years in both {a} and {b}: {overlap}")
    hr.check_holdout({"train_years": years["train"], "eval_years": years["eval"] + years["predict"],
                      "use_holdout": use_holdout})
    df = hr.join_target(predictors, targets, target, keep_modeled=True)
    imputed = targets[KEY].assign(is_imputed=targets["imputation_method"].isin(MODELED_IMPUTATION_METHODS))
    df = df.merge(imputed, on=KEY, how="left", validate="one_to_one")
    df["is_imputed"] = df["is_imputed"].eq(True)  # years without targets (2026) are not imputed
    data = pd.concat([hr.select_years(df, ys).assign(role=role)
                      for role, ys in years.items() if ys or role == "train"], ignore_index=True)
    data.loc[data["role"] == "predict", target] = np.nan

    reason = pd.Series(None, index=data.index, dtype=object)

    def exclude(mask: pd.Series, why: str) -> None:
        reason.loc[mask & reason.isna()] = why

    is_train = data["role"] == "train"
    exclude((data["role"] != "predict") & data["is_imputed"], "imputed_target")
    exclude(is_train & ~_valid(data[target]), "missing_target")
    for v in dict.fromkeys(variables):
        exclude(~_valid(data[v]), f"missing:{v}")

    excluded = data.loc[reason.notna(), [*KEY, "role"]].assign(reason=reason.dropna()).reset_index(drop=True)
    by_reason = excluded.groupby(["role", "reason"]).size()
    missing = by_reason[by_reason.index.get_level_values("reason").str.startswith("missing:")]
    if len(missing):
        warnings.warn(f"Rows excluded for missing or nonfinite predictors:\n{missing.to_string()}", stacklevel=3)

    kept = data[reason.isna()]
    train, eval_data = kept[kept["role"] == "train"], kept[kept["role"] != "train"]
    for role, rows in [("train", train), ("eval", eval_data[eval_data["role"] == "eval"]),
                       ("predict", eval_data[eval_data["role"] == "predict"])]:
        if (years[role] or role == "train") and rows.empty:
            raise ValueError(f"Every {role} row is excluded:\n{by_reason.get(role, pd.Series()).to_string()}")
    return train, eval_data, excluded


def _keys(rows: pd.DataFrame, n: int = 5) -> str:
    keys = list(rows[KEY].itertuples(index=False, name=None))
    return ", ".join(map(str, keys[:n])) + (f", ... (+{len(keys) - n})" if len(keys) > n else "")


def assemble_predictions(train: pd.DataFrame, fitted: pd.Series, eval_data: pd.DataFrame, predicted: pd.Series,
                         target: str) -> pd.DataFrame:
    """One row per training, evaluation, and prediction row: KEY, role, prediction (fitted values for training rows),
    actual, and residual = actual - prediction (NA without an actual).

    `fitted` and `predicted` must be indexed like `train` and `eval_data`. Raises on misaligned, missing, or nonfinite
    predictions; warns, without clipping, about eval / predict predictions outside +-MAX_ABS_MARGIN (in-sample fitted
    values aren't flagged).
    """
    frames = []
    for rows, values in [(train, fitted), (eval_data, predicted)]:
        values = pd.Series(values)
        if not values.index.equals(rows.index):
            raise ValueError(f"Predictions don't line up with the {rows['role'].iloc[0]} rows they were made for")
        frames.append(rows[[*KEY, "role"]].assign(prediction=values.astype(float), actual=rows[target]))
    out = pd.concat(frames, ignore_index=True)
    bad = ~np.isfinite(out["prediction"])
    if bad.any():
        raise ValueError(f"{int(bad.sum())} missing or nonfinite predictions: {_keys(out[bad])}")
    out["residual"] = out["actual"] - out["prediction"]
    wild = (out["role"] != "train") & (out["prediction"].abs() > MAX_ABS_MARGIN)
    if wild.any():
        warnings.warn(f"{int(wild.sum())} predictions outside +-{MAX_ABS_MARGIN} (not clipped): {_keys(out[wild])}",
                      stacklevel=3)
    return out


def eligible_targets(predictions: pd.DataFrame, targets: pd.DataFrame, target: str,
                     race_types=EVAL_RACE_TYPES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(valid rows, reasons) for the years of `predictions` (KEY + prediction, one model's eval rows).

    valid: KEY, race_type, winner_party, actual (targets[target]), and prediction, for every district with a finite,
    non-imputed target of an eligible race type that was predicted. reasons: every district of those years (in the
    targets or the predictions) with its exclusion reason, None if valid (module docstring)."""
    needed = [target, "is_imputed", "race_type", "winner_party"]
    missing = [c for c in needed if c not in targets.columns]
    if missing:
        raise ValueError(f"Targets lack the columns evaluation needs: {missing}")
    years = sorted(predictions["year"].unique())
    rows = targets.loc[targets["year"].isin(years), [*KEY, *needed]].rename(columns={target: "actual"}).merge(
        predictions[[*KEY, "prediction"]], on=KEY, how="outer", validate="one_to_one")

    reason = pd.Series(None, index=rows.index, dtype=object)

    def exclude(mask: pd.Series, why) -> None:  # why: a reason, or a Series of them
        mask = mask & reason.isna()
        reason.loc[mask] = why[mask] if isinstance(why, pd.Series) else why

    exclude(~_valid(rows["actual"]), "missing_target")
    exclude(~rows["is_imputed"].eq(False), "imputed_target")
    exclude(~rows["race_type"].isin(list(race_types)), "race_type:" + rows["race_type"].astype(str))
    exclude(~_valid(rows["prediction"]), "no_prediction")
    valid = rows[reason.isna()].drop(columns="is_imputed").sort_values(KEY).reset_index(drop=True)
    return valid, rows[KEY].assign(reason=reason)


def _competitive_values(valid: pd.DataFrame, column: str, targets: pd.DataFrame,
                        predictors: pd.DataFrame | None) -> np.ndarray:
    """`column` for each valid row: a column name, or arithmetic of them (e.g. "partisan_lean + gb_margin_2p_norm").
    Each name is taken from the targets if they have it, else from the predictors; a null in any gives a null."""
    values = valid[KEY]
    for name in dict.fromkeys(ast_names(column)):
        table = next((t for t in [targets, predictors] if t is not None and name in t.columns), None)
        if table is None:
            raise ValueError(f"Competitiveness column {name!r} (in {column!r}) is in neither the targets nor the "
                             "predictors")
        values = values.merge(table[[*KEY, name]], on=KEY, how="left")
    return pd.Series(values.eval(column)).to_numpy(dtype=float)


def score_table(valid: pd.DataFrame, competitive: dict, targets: pd.DataFrame,
                predictors: pd.DataFrame | None) -> pd.DataFrame:
    """One row per valid target: KEY, actual_margin, predicted_margin, residual (actual - predicted), abs_residual,
    actual_winner, predicted_winner, correct_winner (NA when either is a tie), outcome, race_type, winner_party,
    winner_conflict, and for each competitive subset its column's value (named as written, e.g.
    "partisan_lean + gb_margin_2p_norm") and a boolean column named after the subset."""
    out = valid[KEY].copy()
    out["actual_margin"] = valid["actual"].astype(float)
    out["predicted_margin"] = valid["prediction"].astype(float)
    out["residual"] = out["actual_margin"] - out["predicted_margin"]
    out["abs_residual"] = out["residual"].abs()
    out["actual_winner"] = metrics.winner(out["actual_margin"])
    out["predicted_winner"] = metrics.winner(out["predicted_margin"])
    classified = (out["actual_winner"] != "tie") & (out["predicted_winner"] != "tie")
    out["correct_winner"] = (out["actual_winner"] == out["predicted_winner"]).astype("boolean").where(classified)
    out["outcome"] = np.select([~classified, out["actual_winner"] == out["predicted_winner"],
                                out["actual_winner"] == "D"], ["tie", "correct", "dem_miss"], "gop_miss")
    out["race_type"] = valid["race_type"]
    out["winner_party"] = valid["winner_party"]
    out["winner_conflict"] = valid["winner_party"].notna() & (valid["winner_party"] != out["actual_winner"])
    for name, (column, threshold) in competitive.items():
        if not isinstance(threshold, numbers.Real) or isinstance(threshold, bool) or not 0 <= threshold < np.inf:
            raise ValueError(f"Competitive threshold for {name!r} must be a non-negative number of points, e.g. 5 or "
                             f"7.5; got {threshold!r}")
        values = _competitive_values(valid, column, targets, predictors)
        if column not in out.columns:
            out[column] = values
        out[name] = np.abs(values) <= threshold  # NaN compares False: undetermined rows are not competitive
    return out


def _subset_metrics(rows: pd.DataFrame) -> dict:
    a, p = rows["actual_margin"], rows["predicted_margin"]
    if rows.empty:
        accuracy = {"mae": np.nan, "rmse": np.nan, "bias": np.nan}
    else:
        accuracy = {"mae": metrics.mae(a, p), "rmse": metrics.rmse(a, p), "bias": metrics.bias(a, p)}
    return {"n": len(rows), **accuracy, **metrics.winner_counts(a, p),
            "n_winner_conflict": int(rows["winner_conflict"].sum())}


def evaluate_predictions(predictions: pd.DataFrame, targets: pd.DataFrame, predictors: pd.DataFrame | None, target: str,
                         competitive: dict = COMPETITIVE_SUBSETS,
                         race_types=EVAL_RACE_TYPES) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(metrics, coverage, scored) for one model's eval predictions (KEY + prediction; any estimator).

    metrics: indexed by (subset, year), subsets "valid" then each `competitive` name, each eval year then "all" (pooled);
    columns n, mae, rmse, bias, the winner counts, and n_winner_conflict (module docstring). coverage: one row per eval
    year and "all": n_districts, the count per exclusion reason, n_valid, and per competitive subset its n and the
    valid rows whose column is null (<name>_undetermined). scored: one row per valid target (score_table).
    `competitive` maps a subset name to (column, threshold), where column is a column name or arithmetic of them;
    predictors are needed only for a column they hold."""
    valid, reasons = eligible_targets(predictions, targets, target, race_types)
    scored = score_table(valid, competitive, targets, predictors)
    years = sorted(predictions["year"].unique())

    subsets = {VALID: scored, **{name: scored[scored[name]] for name in competitive}}
    rows = {}
    for name, subset in subsets.items():
        for year in years:
            rows[(name, year)] = _subset_metrics(subset[subset["year"] == year])
        rows[(name, "all")] = _subset_metrics(subset)
    metrics_table = pd.DataFrame.from_dict(rows, orient="index")
    metrics_table.index.names = ["subset", "year"]

    def coverage_row(year_reasons: pd.DataFrame, year_scored: pd.DataFrame) -> dict:
        out = {"n_districts": len(year_reasons)}
        out |= {r: int((year_reasons["reason"] == r).sum()) for r in reason_names}
        out["n_valid"] = len(year_scored)
        for name, (column, _) in competitive.items():
            out[name] = int(year_scored[name].sum())
            out[f"{name}_undetermined"] = int(year_scored[column].isna().sum())
        return out

    found = reasons["reason"].dropna().unique()
    reason_names = ["missing_target", "imputed_target", *sorted(r for r in found if r.startswith("race_type:")),
                    "no_prediction"]
    coverage = pd.DataFrame.from_dict(
        {**{year: coverage_row(reasons[reasons["year"] == year], scored[scored["year"] == year]) for year in years},
         "all": coverage_row(reasons, scored)}, orient="index")
    coverage.index.name = "year"
    return metrics_table, coverage, scored


def _subset_rows(scored: pd.DataFrame, subset: str | None) -> pd.DataFrame:
    """scored, or its rows in a competitive subset (a boolean column of scored)."""
    if subset is None or subset == VALID:
        return scored
    if subset not in scored.columns:
        raise ValueError(f"No subset {subset!r} in scored (its competitive subsets are boolean columns)")
    return scored[scored[subset]]


def worst_misses(scored: pd.DataFrame, n: int, subset: str | None = None) -> pd.DataFrame:
    """The n rows of `subset` (None: every valid row) with the largest abs_residual; ties in KEY order."""
    rows = _subset_rows(scored, subset).sort_values(KEY)
    return rows.sort_values("abs_residual", ascending=False, kind="stable").head(n).reset_index(drop=True)


def competitive_examples(scored: pd.DataFrame, subset: str, n: int, seed: int, districts=None) -> pd.DataFrame:
    """Example rows of `subset`, in KEY order: a random sample of n (all of them if fewer), reproducible by seed, or
    exactly `districts`, a list of (year, state, district) keys, each of which must be in the subset."""
    rows = _subset_rows(scored, subset)
    if districts is None:
        picked = rows.sample(n=min(n, len(rows)), random_state=seed)
    else:
        wanted = pd.DataFrame(list(districts), columns=KEY)
        picked = wanted.merge(rows, on=KEY, how="left", indicator=True)
        absent = picked[picked["_merge"] == "left_only"]
        if not absent.empty:
            raise ValueError(f"Not in the {subset} subset: {_keys(absent)}")
        picked = picked.drop(columns="_merge")
    return picked.sort_values(KEY).reset_index(drop=True)


def confusion_matrix(scored: pd.DataFrame, subset: str | None = None, year=None, percent: bool = False) -> pd.DataFrame:
    """Counts of actual (rows) by predicted (columns) winner, D and R, for `subset` and one year (None: every eval
    year); percent=True gives each cell as a percentage of all four (the classified rows), so the off-diagonal cells
    add up to missed_pct. Rows where either margin is a tie are left out; metrics counts them (n_actual_tie,
    n_predicted_tie)."""
    rows = _subset_rows(scored, subset)
    rows = rows if year is None else rows[rows["year"] == year]
    rows = rows[(rows["actual_winner"] != "tie") & (rows["predicted_winner"] != "tie")]
    table = pd.crosstab(rows["actual_winner"], rows["predicted_winner"])
    table = table.reindex(index=["D", "R"], columns=["D", "R"], fill_value=0)
    table.index.name, table.columns.name = "actual", "predicted"
    return 100 * table / table.to_numpy().sum() if percent else table  # NaN when nothing is classified


OUTCOME_HEADINGS = {
    "correct": "PREDICTED CORRECTLY",
    "dem_miss": "DEM MISS: predicted R but voted D",
    "gop_miss": "GOP MISS: predicted D but voted R",
    "tie": "NOT CLASSIFIED: a margin of exactly 0",  # listed only when there is one
}


def outcome_report(scored: pd.DataFrame, subset: str | None = None) -> str:
    """The rows of `subset` (None: every valid row) as text, by eval year, then by outcome (OUTCOME_HEADINGS); within
    each, states in alphabetical order and each state's districts closest actual margin first, one line each:

        * CA-45 (R): predicted margin: -3.10, actual: -4.79, residual: -1.69

    with the actual winner in parentheses and residual = actual - predicted."""
    rows = _subset_rows(scored, subset)
    rows = rows.assign(closeness=rows["actual_margin"].abs()).sort_values(["year", "state", "closeness", "district"])
    lines = [f"{subset or VALID}: {len(rows)} districts; residual = actual - predicted"]
    for year, year_rows in rows.groupby("year"):
        lines.append(f"\n===== {year} =====")
        for outcome, heading in OUTCOME_HEADINGS.items():
            group = year_rows[year_rows["outcome"] == outcome]
            if outcome == "tie" and group.empty:
                continue
            lines.append(f"\n{heading} ({len(group)})")
            lines += [f"* {district_label(r.state, r.district)} ({r.actual_winner}): predicted margin: "
                      f"{r.predicted_margin:+.2f}, actual: {r.actual_margin:+.2f}, "
                      f"residual: {r.residual:+.2f}" for r in group.itertuples()]
            if group.empty:
                lines.append("* (none)")
    return "\n".join(lines)


def state_summary(scored: pd.DataFrame, subsets) -> pd.DataFrame:
    """One row per state, pooled over the eval years (a district counts once per year), worst |mean_residual| first:
    n_districts, mean_residual (actual - predicted: positive means Democrats outperformed the forecast),
    mean_abs_residual, n_missed (wrong winner; dem_miss + gop_miss) and pct_missed (of the classified rows, ties left
    out), then for each competitive subset in `subsets` its n, missed, and pct_missed."""
    rows = scored.assign(missed=scored["outcome"].isin(["dem_miss", "gop_miss"]),
                         classified=scored["outcome"] != "tie")
    by_state = rows.groupby("state")

    def pct(missed: pd.Series, classified: pd.Series) -> pd.Series:
        return 100 * missed / classified.where(classified > 0)  # NaN when none is classified

    out = pd.DataFrame({
        "n_districts": by_state.size(),
        "mean_residual": by_state["residual"].mean(),
        "mean_abs_residual": by_state["abs_residual"].mean(),
        "n_missed": by_state["missed"].sum(),
        "n_dem_miss": by_state["outcome"].apply(lambda o: (o == "dem_miss").sum()),
        "n_gop_miss": by_state["outcome"].apply(lambda o: (o == "gop_miss").sum()),
    })
    out["pct_missed"] = pct(out["n_missed"], by_state["classified"].sum())
    for subset in subsets:
        in_subset = rows[subset]
        out[f"n_{subset}"] = in_subset.groupby(rows["state"]).sum()
        out[f"missed_{subset}"] = (in_subset & rows["missed"]).groupby(rows["state"]).sum()
        out[f"pct_missed_{subset}"] = pct(out[f"missed_{subset}"],
                                          (in_subset & rows["classified"]).groupby(rows["state"]).sum())
    return out.sort_values("mean_residual", key=abs, ascending=False, kind="stable")


def district_label(state: str, district: int) -> str:
    """AZ-03, or AK-AL for an at-large seat (district 0)."""
    return f"{state}-{'AL' if district == 0 else f'{district:02d}'}"


def excluded_by_year(excluded: pd.DataFrame, max_districts: int | None = 5) -> pd.DataFrame:
    """The excluded rows (a result's `excluded`) by year: role, reason, the number of districts, and which ones (the
    first max_districts, then "(+n)"; None lists all), with a total row at the bottom."""
    def districts(rows: pd.DataFrame) -> str:
        labels = [district_label(s, d) for s, d in sorted(zip(rows["state"], rows["district"]))]
        if max_districts is None or len(labels) <= max_districts:
            return ", ".join(labels)
        return ", ".join(labels[:max_districts]) + f", ... (+{len(labels) - max_districts})"

    rows = [{"year": year, "role": role, "reason": reason, "n": len(group), "districts": districts(group)}
            for (year, role, reason), group in excluded.groupby(["year", "role", "reason"])]
    rows.append({"year": "total", "role": "", "reason": "", "n": len(excluded), "districts": ""})
    return pd.DataFrame(rows).set_index("year")


def rows_hash(df: pd.DataFrame) -> str:
    """A short hash of a frame's values; equal for two runs exactly when they used the same rows."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()[:12]


def counts(predictions: pd.DataFrame, excluded: pd.DataFrame) -> dict:
    """Sample sizes by role, exclusions by "role:reason", and hashes of the training and scored evaluation rows (so
    ablations can confirm they compare identical rows)."""
    role = predictions["role"]
    train = predictions[role == "train"]
    evaluated = predictions[role == "eval"]
    scored = evaluated[evaluated["actual"].notna()]
    return {
        "n_train": len(train),
        "n_eval": len(evaluated),
        "n_eval_scored": len(scored),
        "n_eval_no_target": len(evaluated) - len(scored),
        "n_predict": int((role == "predict").sum()),
        "excluded": {f"{r}:{why}": int(n) for (r, why), n in excluded.groupby(["role", "reason"]).size().items()},
        "train_rows_hash": rows_hash(train[KEY].sort_values(KEY)),
        "eval_rows_hash": rows_hash(scored[KEY].sort_values(KEY)),
    }
