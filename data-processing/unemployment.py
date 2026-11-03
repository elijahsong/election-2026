"""State and national unemployment-rate features, shared by the House and Senate pipelines.

```bash
python data-processing/unemployment.py   # -> data/processed/unemployment_monthly.csv, unemployment_state.csv
```

unemployment_monthly.csv  month, area: the monthly unemployment rate (%, seasonally adjusted) from FIRST_MONTH on,
                          for the 50 states, DC and the nation (area US). A month BLS didn't publish stays blank
unemployment_state.csv    year, state: unemp_data_month and the features below for each election year

house_features.py and senate_features.py add these to prices_f.csv (attach_states), next to the CPI and gasoline
features.

Timing: the same rule as cpi_gas.py (cpi_gas.data_month), so every year uses September, and the 12-month change is
September over September. BLS publishes the national September rate in early October and the state rates about
three weeks after the month ends, both before Election Day. (October's national rate is sometimes out before Election
Day, e.g. November 1, 2024, but the state rates never are, so September is used for both.)

Features, for state s and data month M (rates in %, changes and gaps in percentage points, natural logs as
log_rel_income):
  unemp_state_rate      u_s(M)
  unemp_us_rate         u_US(M)
  unemp_state_minus_us  u_s(M) - u_US(M)
  log_unemp_state       log(u_s(M))
  log_rel_unemp         log(u_s(M) / u_US(M))
  unemp_change_12m      u_s(M) - u_s(M - 12)
  unemp_change_3m       u_s(M) - u_s(M - 3)
  unemp_change_12m_minus_us  unemp_change_12m - (u_US(M) - u_US(M - 12)): the state's 12-month change against the
                        nation's

Missing data is never filled: a feature that needs a blank month (October 2025, which BLS didn't publish because of the
2025 shutdown) is blank.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "house"))
import acs  # write_csv_atomic, STATE_FIPS
import cpi_gas  # read_fred_csv, data_month, log_ratio, election_years, FIRST_MONTH

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/unemployment_rate"
STATE_PATH = RAW_DIR / "monthly.csv"  # FRED data list 337: <state>UR for the 50 states and DC
NATIONAL_PATH = RAW_DIR / "UNRATE.csv"
MONTHLY_PATH = ROOT / "data/processed/unemployment_monthly.csv"
STATE_FEATURES_PATH = ROOT / "data/processed/unemployment_state.csv"

STATES = sorted(set(acs.STATE_FIPS.values()) | {"DC"})
FEATURES = ["unemp_state_rate", "unemp_us_rate", "unemp_state_minus_us", "log_unemp_state", "log_rel_unemp",
            "unemp_change_12m", "unemp_change_3m", "unemp_change_12m_minus_us"]
# the rates are published to one decimal, so the rates, gaps and changes keep 1 and the logs 4
DECIMALS = {c: 4 if c.startswith("log_") else 1 for c in FEATURES}


def monthly_rates(states: pd.DataFrame, national: pd.Series) -> pd.DataFrame:
    """The long monthly table (month, area, unemployment_rate) from FIRST_MONTH to the last month with a value, every
    month present; areas are the USPS codes and US."""
    wide = pd.concat([states.rename(columns=lambda c: c.removesuffix("UR")), national.rename("US")], axis=1)
    last = wide.dropna(how="all").index.max()
    wide = wide.reindex(pd.period_range(cpi_gas.FIRST_MONTH, last, freq="M"))[[*STATES, "US"]]
    if (wide <= 0).any().any() or (wide >= 100).any().any():
        raise ValueError("Unemployment rates outside (0, 100)")
    long = wide.stack(future_stack=True).rename("unemployment_rate")
    long.index.names = ["month", "area"]
    return long.reset_index()


def state_features(monthly: pd.DataFrame, election_dates: dict[int, str]) -> pd.DataFrame:
    """One row per election year and state: the features of the module docstring, from each year's data month
    (September) and the months before it. NaN in any rate a feature needs leaves that feature NaN."""
    rates = monthly.pivot(index="month", columns="area", values="unemployment_rate")

    def at(month: pd.Period) -> pd.Series:
        return rates.reindex([month]).iloc[0]

    rows = []
    for year, election_date in sorted(election_dates.items()):
        m = cpi_gas.data_month(election_date)
        now = at(m)
        state, us = now[STATES], now["US"]
        f = pd.DataFrame(index=STATES)
        f["unemp_state_rate"] = state
        f["unemp_us_rate"] = us
        f["unemp_state_minus_us"] = state - us
        f["log_unemp_state"] = np.log(state)
        f["log_rel_unemp"] = cpi_gas.log_ratio(state, us)
        f["unemp_change_12m"] = state - at(m - 12)[STATES]
        f["unemp_change_3m"] = state - at(m - 3)[STATES]
        f["unemp_change_12m_minus_us"] = f["unemp_change_12m"] - (us - at(m - 12)["US"])
        f = f.round(DECIMALS) + 0.0  # + 0.0 turns -0.0 into 0.0
        rows.append(f.rename_axis("state").reset_index().assign(year=year, unemp_data_month=str(m)))
    out = pd.concat(rows, ignore_index=True)
    return out[["year", "state", "unemp_data_month", *FEATURES]]


def attach_states(df: pd.DataFrame, state_f: pd.DataFrame) -> pd.DataFrame:
    """df (with year and state) plus FEATURES from its state's row. Raises for a (year, state) state_f lacks; blank
    features stay blank."""
    missing = df[["year", "state"]].merge(state_f[["year", "state"]], how="left", indicator=True)
    missing = missing.loc[missing["_merge"] == "left_only", ["year", "state"]].drop_duplicates()
    if len(missing):
        raise ValueError(f"No unemployment features for {len(missing)} year-states, e.g. "
                         f"{missing.head(3).values.tolist()}")
    return df.merge(state_f[["year", "state", *FEATURES]], on=["year", "state"], how="left", validate="many_to_one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=MONTHLY_PATH.parent, type=Path)
    args = parser.parse_args()

    states = cpi_gas.read_fred_csv(STATE_PATH, [f"{s}UR" for s in STATES])
    national = cpi_gas.read_fred_csv(NATIONAL_PATH, ["UNRATE"])["UNRATE"]
    monthly = monthly_rates(states, national)
    features = state_features(monthly, cpi_gas.election_years())

    print(f"Monthly rates {monthly['month'].min()} to {monthly['month'].max()}, {monthly['area'].nunique()} areas")
    by_month = monthly[monthly["unemployment_rate"].isna()].groupby("month")["area"].nunique()
    print("Blank months: " + (", ".join(f"{m} ({n} areas)" for m, n in by_month.items()) or "none"))
    print(f"Data month per election: {sorted(features['unemp_data_month'].unique())}")
    for column in FEATURES:
        years = sorted(int(y) for y in features.loc[features[column].isna(), "year"].unique())
        if years:
            print(f"{column} blank in {years}")
    monthly_out = monthly.assign(month=monthly["month"].astype(str))
    acs.write_csv_atomic(monthly_out, args.output_dir / MONTHLY_PATH.name)
    acs.write_csv_atomic(features, args.output_dir / STATE_FEATURES_PATH.name)
    print(f"-> {args.output_dir / MONTHLY_PATH.name} ({len(monthly_out)} rows), "
          f"{args.output_dir / STATE_FEATURES_PATH.name} ({len(features)} rows)")


if __name__ == "__main__":
    main()
