"""Estimator-agnostic experiment steps: prepare the training / evaluation rows, check and assemble predictions, and
score them. run.run_ols uses them around smf.ols; a future estimator (run_wls, run_ridge, ...) can reuse them as is.

Rows are never dropped silently. prepare_data gives every left-out row of the requested years a reason, first match
wins:

    imputed_target   training only: a modeled imputation (MODELED_IMPUTATION_METHODS), never ground truth. Jungle
                     and first-round RCV targets have no imputation_method, so they are non-imputed and kept
    missing_target   training only: no finite target
    missing:<var>    any role: <var>, a predictor the formula uses (or a require_complete column), is null or nonfinite

Evaluation rows of imputed targets are kept and scored separately: evaluate_predictions reports "non_imputed" (the
primary metric), "imputed" (agreement with synthetic targets), and "all", from the same predictions.
"""

import hashlib
import warnings

import numpy as np
import pandas as pd
from patsy import ModelDesc
from patsy.eval import ast_names

import house.regression as hr
from house.regression import metrics
from house.regression.schema import KEY, MODELED_IMPUTATION_METHODS

SUBSETS = ["non_imputed", "imputed", "all"]
METRICS = ["n", "mae", "rmse", "bias"]
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

    The target is joined onto the predictors by key (hr.join_target checks the keys match and are unique), with an
    is_imputed flag. The year lists must be disjoint, and HOLDOUT_YEARS (2024) need use_holdout=True in any of them.
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
    exclude(is_train & data["is_imputed"], "imputed_target")
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
    actual, is_imputed, and residual = actual - prediction (NA without an actual).

    `fitted` and `predicted` must be indexed like `train` and `eval_data`. Raises on misaligned, missing, or nonfinite
    predictions; warns, without clipping, about eval / predict predictions outside +-MAX_ABS_MARGIN (in-sample fitted
    values aren't flagged).
    """
    frames = []
    for rows, values in [(train, fitted), (eval_data, predicted)]:
        values = pd.Series(values)
        if not values.index.equals(rows.index):
            raise ValueError(f"Predictions don't line up with the {rows['role'].iloc[0]} rows they were made for")
        frames.append(rows[[*KEY, "role"]].assign(prediction=values.astype(float), actual=rows[target],
                                                   is_imputed=rows["is_imputed"]))
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


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame | None:
    """n, mae, rmse, and bias (prediction - actual) of the eval rows with an actual, for SUBSETS; None when there are no
    eval rows (a prediction-only run). An empty subset has n=0 and NaN metrics."""
    evaluated = predictions[predictions["role"] == "eval"]
    if evaluated.empty:
        return None
    scored = evaluated[evaluated["actual"].notna()]
    subsets = {"non_imputed": scored[~scored["is_imputed"]], "imputed": scored[scored["is_imputed"]], "all": scored}

    def row(rows: pd.DataFrame) -> dict:
        if rows.empty:
            return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan}
        a, p = rows["actual"], rows["prediction"]
        return {"n": len(rows), "mae": metrics.mae(a, p), "rmse": metrics.rmse(a, p), "bias": metrics.bias(a, p)}

    out = pd.DataFrame.from_dict({name: row(rows) for name, rows in subsets.items()}, orient="index")
    out.index.name = "subset"
    return out


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
        "eval_rows_hash": rows_hash(scored[[*KEY, "is_imputed"]].sort_values(KEY)),
    }
