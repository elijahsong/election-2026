"""Load the predictors (modeling.csv) and targets (targets.csv) separately, join the config's target onto the
predictors, and select the exact training / evaluation years the user lists.

Both tables are loaded whole, before any year or predictor is chosen. join_target checks that they line up before
anything reaches a model: every year of the targets must be in the predictors and, in those years, both tables must
have exactly the same keys. Predictor years without targets (2026) keep a blank target, and so do targets imputed from
other races (schema.MODELED_IMPUTATION_METHODS), which are never ground truth.
"""

from pathlib import Path

import pandas as pd

from .schema import COLUMNS, KEY, MODELED_IMPUTATION_METHODS, TARGET_COLUMNS, TARGETS_SCHEMA, Column, validate_schema

ROOT = Path(__file__).resolve().parents[3]
PREDICTORS_PATH = "data/processed/house/modeling.csv"
TARGETS_PATH = "data/processed/house/targets.csv"


def _read(path: str | Path, schema: dict[str, Column]) -> pd.DataFrame:
    """Read a .csv or .parquet table; warn once about optional-column issues and raise on key errors.

    A relative path is taken from the repository root, so the same path works from any notebook directory.
    """
    path = ROOT / path
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={c.name: str for c in schema.values() if c.kind == "str"})
    report = validate_schema(df, schema=schema)
    report.warn(stacklevel=3)  # attribute the warning to the load_* function's caller
    report.raise_if_errors()
    return df


def load_predictors(path: str | Path = PREDICTORS_PATH) -> pd.DataFrame:
    """The modeling table: the key and every predictor, all years. It must not hold targets."""
    df = _read(path, COLUMNS)
    targets = [c for c in df.columns if c in TARGET_COLUMNS and c not in KEY]
    if targets:
        raise ValueError(f"Target columns in the predictors {path}: {targets}; targets come from {TARGETS_PATH}")
    return df


def load_targets(path: str | Path = TARGETS_PATH) -> pd.DataFrame:
    """The targets table: the key, every target, and its provenance, all years."""
    return _read(path, TARGETS_SCHEMA)


def join_target(predictors: pd.DataFrame, targets: pd.DataFrame, target: str,
                keep_modeled: bool = False) -> pd.DataFrame:
    """predictors plus the `target` column of targets, by key; raises unless the two tables line up.

    Both keys must be valid and unique, `target` must be in targets and not in predictors, every targets year must be in
    predictors, and in those years both must have exactly the same keys. Predictor rows of other years get a blank
    target, and so do rows whose imputation_method is in MODELED_IMPUTATION_METHODS, so they are neither trained on nor
    scored, unless keep_modeled=True (for a config with "training_inputs": "all").
    """
    validate_schema(predictors).raise_if_errors()
    validate_schema(targets, schema=TARGETS_SCHEMA).raise_if_errors()
    if target not in targets.columns:
        raise ValueError(f"Target {target!r} not in the targets")
    if target in predictors.columns:
        raise ValueError(f"Target {target!r} is also a predictor column; targets come only from the targets table")
    if "imputation_method" not in targets.columns:
        raise ValueError("No imputation_method column in the targets; it marks the imputed targets to leave out")
    missing_years = sorted(set(targets["year"]) - set(predictors["year"]))
    if missing_years:
        raise ValueError(f"Target years not in the predictors: {missing_years}")
    shared = predictors[predictors["year"].isin(targets["year"])]
    keys = shared[KEY].merge(targets[KEY], how="outer", indicator=True)
    unmatched = keys[keys["_merge"] != "both"]
    if not unmatched.empty:
        sides = unmatched["_merge"].map({"left_only": "predictors only", "right_only": "targets only"})
        listed = [(*key, side) for key, side in zip(unmatched[KEY].itertuples(index=False, name=None), sides)]
        raise ValueError(f"{len(unmatched)} keys in only one table, in years both cover: {listed[:5]}"
                         + (f", ... (+{len(listed) - 5})" if len(listed) > 5 else ""))
    values = targets[[*KEY, target]].copy()
    if not keep_modeled:
        values.loc[targets["imputation_method"].isin(MODELED_IMPUTATION_METHODS), target] = None
    return predictors.merge(values, on=KEY, how="left", validate="one_to_one")


def select_years(df: pd.DataFrame, years) -> pd.DataFrame:
    """The rows of exactly `years`; every listed year must be in df."""
    years = list(years)
    if not years:
        raise ValueError("No years given")
    if len(set(years)) != len(years):
        raise ValueError(f"Years listed more than once: {sorted(y for y in set(years) if years.count(y) > 1)}")
    missing = sorted(set(years) - set(df["year"]))
    if missing:
        raise ValueError(f"Years not in the data: {missing}")
    return df[df["year"].isin(years)]


def split_years(df: pd.DataFrame, train_years, eval_years) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training rows, evaluation rows) for the listed years, which must not overlap."""
    overlap = sorted(set(train_years) & set(eval_years))
    if overlap:
        raise ValueError(f"Years in both training and evaluation: {overlap}")
    return select_years(df, train_years), select_years(df, eval_years)
