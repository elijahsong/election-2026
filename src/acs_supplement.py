"""Add national- and state-relative college, income, and unemployment columns to acs.csv, written to a new file."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import acs

REFERENCE_FIELDS = ["college", "median_income", "unemployment"]
SUPPLEMENT_COLUMNS = [
    "rel_college", "log_rel_income", "rel_state_college",
    "log_rel_state_income", "rel_unemployment", "rel_state_unemployment",
]
NATIONAL_GEO = {"for": "us:1"}
STATE_GEO = {"for": "state:*"}
# Election years with no ACS vintage of their own -> the election year whose vintage their rows are on. 2026's rows
# (notebooks/03c) are 2024 1-year values: copied for unchanged districts, or rebuilt and adjusted to the 2024 1-year
# (acs_vintage "2024_1yr_rebuilt") for redrawn ones, so they are compared with 2024 1-year national and state values.
REFERENCE_YEAR = {2026: 2024}
REBUILT_SUFFIX = "_rebuilt"


def reference_year(year: int) -> int:
    """The election year whose ACS vintage supplies a year's national and state comparison values."""
    return REFERENCE_YEAR.get(year, year)


def allowed_vintages(year: int) -> set[str]:
    """acs_vintage labels a year's rows may carry: its vintage, plus its rebuilt variant for borrowed-vintage years."""
    label = acs.acs_vintage_label(reference_year(year))
    return {label, label + REBUILT_SUFFIX} if year in REFERENCE_YEAR else {label}


def read_acs(path: str | Path) -> pd.DataFrame:
    """Load a master acs.csv and check it is the acs.py schema, one vintage per year, ready for comparison."""
    df = pd.read_csv(path, dtype={"state": str, "acs_vintage": str})
    if list(df.columns) != acs.COLUMNS:
        raise ValueError(f"{path} columns {list(df.columns)} do not match the acs.py schema")
    if df.empty:
        raise ValueError(f"{path} has no rows")
    if df.duplicated(["year", "state", "district"]).any():
        raise ValueError(f"{path} has duplicate (year, state, district) keys")
    if df[REFERENCE_FIELDS].isna().any().any() or (df["median_income"] <= 0).any():
        raise ValueError(f"{path} has missing or non-positive college, income, or unemployment values")
    for year, vintages in df.groupby("year")["acs_vintage"]:
        # national and state references are fetched from this same vintage, so it must be the mapped one
        allowed = allowed_vintages(int(year))
        if not set(vintages) <= allowed:
            raise ValueError(f"{path} {year} rows use {sorted(set(vintages))}, expected {' or '.join(sorted(allowed))}")
    return df


def fetch_reference(year: int, key: str) -> tuple[pd.Series, pd.DataFrame]:
    """National values and per-state values (indexed by USPS code) from the same ACS vintage as the districts."""
    variables = acs.field_variables(year, REFERENCE_FIELDS)
    specs = {field: acs.ACS_FIELDS[year][field] for field in REFERENCE_FIELDS}
    acs.verify_variables(year, variables)

    def fields(raw: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
        values = acs.parse_estimates(raw, variables, ids)
        # rounded like the district rows, so an at-large district (the same area as its state) compares as exactly 0
        return acs.round_fields(acs.compute_fields(values, specs, ids))

    national = acs.fetch_acs(year, key, NATIONAL_GEO, variables)
    if len(national) != 1:
        raise ValueError(f"Expected one national row for {year}, got {len(national)}")
    states = acs.fetch_acs(year, key, STATE_GEO, variables)
    states = states[states["state"].isin(acs.STATE_FIPS)].reset_index(drop=True)  # drops DC and Puerto Rico
    states["state"] = states["state"].map(acs.STATE_FIPS)
    if len(states) != len(acs.STATE_FIPS):
        raise ValueError(f"Expected {len(acs.STATE_FIPS)} state rows for {year}, got {len(states)}")

    national_values = fields(national, pd.Series(["US"])).iloc[0]
    state_values = fields(states, states["state"]).set_axis(states["state"])
    if national_values["median_income"] <= 0 or (state_values["median_income"] <= 0).any():
        raise ValueError(f"Non-positive reference median income in {year}")
    return national_values, state_values


def _log_ratio(x: pd.Series, ref: pd.Series | float) -> pd.Series:
    """log(x / ref) via log1p of the exact integer gap, accurate near 0 where most districts sit."""
    return np.log1p((x - ref) / ref)


def add_relative_columns(df: pd.DataFrame, national: pd.Series, states: pd.DataFrame) -> pd.DataFrame:
    """Append the six relative columns to one year's district rows."""
    state = states.reindex(df["state"]).set_axis(df.index)
    if state.isna().any().any():
        raise ValueError(f"No state reference for {sorted(df.loc[state.isna().any(axis=1), 'state'].unique())}")
    out = df.copy()
    out["rel_college"] = df["college"] - national["college"]
    out["log_rel_income"] = _log_ratio(df["median_income"], national["median_income"])
    out["rel_state_college"] = df["college"] - state["college"]
    out["log_rel_state_income"] = _log_ratio(df["median_income"], state["median_income"])
    out["rel_unemployment"] = df["unemployment"] - national["unemployment"]
    out["rel_state_unemployment"] = df["unemployment"] - state["unemployment"]
    # percentage-point gaps keep the inputs' 2 decimals; log ratios keep 4 (0.01% of income); + 0.0 turns -0.0 into 0.0
    out[SUPPLEMENT_COLUMNS] = out[SUPPLEMENT_COLUMNS].round({c: 4 if c.startswith("log_") else 2 for c in SUPPLEMENT_COLUMNS}) + 0.0
    return out


def build_supplement(input_path: str | Path = "data/interim/acs.csv",
                     output_path: str | Path = "data/interim/acs_supplement.csv") -> pd.DataFrame:
    """Rebuild the supplement for every year in acs.csv; acs.csv itself is left untouched."""
    if Path(output_path).resolve() == Path(input_path).resolve():
        raise ValueError("Write the supplement to a separate file; acs.py only accepts its own schema in acs.csv")
    df = read_acs(input_path)
    key = acs.census_key()
    references = {}  # reference year -> (national, states), so a borrowed vintage is fetched once
    parts = []
    for year, rows in tqdm(df.groupby("year"), desc="ACS supplement", unit="year"):
        ref = reference_year(int(year))
        if ref not in references:
            references[ref] = fetch_reference(ref, key)
        parts.append(add_relative_columns(rows, *references[ref]))

    out = pd.concat(parts).sort_values(["year", "state", "district"], ignore_index=True)
    if not np.isfinite(out[SUPPLEMENT_COLUMNS].to_numpy()).all():
        raise ValueError("Non-finite relative values")
    out = out[acs.COLUMNS + SUPPLEMENT_COLUMNS]
    acs.write_csv_atomic(out, output_path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/interim/acs.csv")
    parser.add_argument("--output", default="data/interim/acs_supplement.csv")
    args = parser.parse_args()
    out = build_supplement(args.input, args.output)
    tqdm.write(f"{len(out)} rows across {out['year'].nunique()} years -> {args.output}")


if __name__ == "__main__":
    main()
