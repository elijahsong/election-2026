"""CPI-U inflation and regular-gasoline price features by Census region, shared by the House and Senate pipelines.

```bash
python data-processing/cpi_gas.py   # -> data/processed/cpi_gas_monthly.csv and data/processed/cpi_gas_region.csv
```

cpi_gas_monthly.csv  month, region: the monthly BLS levels from FIRST_MONTH (January 2004) on, for the nation (us)
                     and the four Census regions. cpi is CPI-U all items, not seasonally adjusted (1982-84 = 100); gas is the
                     average price of regular unleaded gasoline ($/gallon). A month BLS didn't publish stays blank
cpi_gas_region.csv   year, census_region: the features below for each election year, from the price_data_month levels

house_features.py and senate_features.py give every district / race its state's region row (attach_regions).

Timing: BLS publishes month M's CPI and average prices in mid-month M + 1 (September 2024 on October 10, 2024;
October 2024 on November 13, 2024). A month is treated as public from the first day of the month PUBLICATION_LAG_MONTHS
after it, so a November election uses September data, the latest month out before Election Day in every year. Every year
must use September (data_month raises otherwise), so the 12- and 24-month changes are always September over
September and the 6-month changes September over March.
Neither NSA CPI-U nor the average prices are revised after release, so today's files are what was public then.

Features, for region r and data month M (percent changes are 100 * (x / x_before - 1), so 3.5 means 3.5%):
  cpi_region_yoy           CPI_r(M) vs CPI_r(M - 12)
  cpi_us_yoy               CPI_us(M) vs CPI_us(M - 12)
  cpi_region_minus_us_yoy  cpi_region_yoy - cpi_us_yoy (percentage points)
  cpi_region_24m           CPI_r(M) vs CPI_r(M - 24)
  cpi_region_6m            CPI_r(M) vs CPI_r(M - 6)
  gas_region_price_3m      mean of gas_r over M - 2 .. M ($/gallon)
  gas_region_yoy           gas_r(M) vs gas_r(M - 12)
  gas_region_minus_us      gas_r(M) - gas_us(M) ($/gallon)
  gas_region_real_price    gas_r(M) * CPI_r(REAL_PRICE_REFERENCE_MONTH) / CPI_r(M): constant reference-month dollars
  gas_region_6m_change     gas_r(M) vs gas_r(M - 6)

Log features (natural logs, as log_rel_income; log changes are 100 * log(x / x_before), roughly a percent). Real
prices here are in September 2005 dollars: each CPI series is rescaled to DEFLATOR_BASE_MONTH = 100 and
real_r = gas_r / (CPI_r / 100), regional gas by its regional CPI and national gas by the national CPI:
  cpi_region_log_yoy       100 * log(CPI_r(M) / CPI_r(M - 12))
  cpi_region_log_diff_us   cpi_region_log_yoy - 100 * log(CPI_us(M) / CPI_us(M - 12)) (percentage points)
  log_real_gas_price       log(real_r(M))
  log_rel_gas_price        log(gas_r(M) / gas_us(M))
  log_rel_real_gas         log(real_r(M) / real_us(M)): the nominal price gap, adjusted for the difference in general
                           inflation accumulated since September 2005; not a cost-of-living comparison across regions,
                           since regional CPI levels aren't comparable price levels
  log_gas_yoy              100 * log(gas_r(M) / gas_r(M - 12))
  log_real_gas_yoy         100 * log(real_r(M) / real_r(M - 12))

Relative trend (the region's 12-month change against the nation's, as cpi_region_minus_us_yoy is for CPI):
  gas_region_minus_us_yoy  gas_region_yoy - (gas_us(M) vs gas_us(M - 12)) (percentage points)

Missing data is never filled: a feature that needs a blank month (October 2025's CPI, which BLS didn't publish
because of the 2025 shutdown) or a month before FIRST_MONTH is blank, and so is a 3-month average missing any month.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "house"))
import acs  # write_csv_atomic, STATE_FIPS

ROOT = Path(__file__).resolve().parents[1]
CPI_DIR = ROOT / "data/raw/fred-cpi"
# The national series as BLS's Data Finder exports it (one row per year, one column per month); the regional CPI and
# all gasoline series are FRED downloads (observation_date plus one column per series)
NATIONAL_CPI_PATH = CPI_DIR / "SeriesReport-20261102172633_254299.csv"
GAS_PATH = ROOT / "data/raw/fred_gas/fred-gas-price.csv"
ELECTION_DATES_PATH = ROOT / "data/reference/election_dates.json"
MONTHLY_PATH = ROOT / "data/processed/cpi_gas_monthly.csv"
REGION_PATH = ROOT / "data/processed/cpi_gas_region.csv"

CPI_SERIES = {"us": "CUUR0000SA0", "northeast": "CUUR0100SA0", "midwest": "CUUR0200SA0", "south": "CUUR0300SA0",
              "west": "CUUR0400SA0"}
GAS_SERIES = {"us": "APU000074714", "northeast": "APU010074714", "midwest": "APU020074714",
              "south": "APU030074714", "west": "APU040074714"}
REGIONS = ["northeast", "midwest", "south", "west"]
# The regional CPI files start in January 2004, so 2006 has its 24-month change (from September 2004); national CPI
# and gasoline start in January 2005, and their 2004 months are blank (no feature needs them)
FIRST_MONTH = pd.Period("2004-01", "M")
FIRST_YEAR, LAST_YEAR = 2006, 2026  # election years of the House and Senate modeling data
PUBLICATION_LAG_MONTHS = 2
# September 2005, the first September all ten series have (every reference month is a September, as every data
# month is): each CPI series is rescaled to 100 here to deflate gas for the log real-price features, so the regional
# and national deflators share one reference date
DEFLATOR_BASE_MONTH = pd.Period("2005-09", "M")
# September 2026, the 2026 election's data month: gas_region_real_price is in September 2026 dollars of each region,
# so 2026's real price equals its nominal price
REAL_PRICE_REFERENCE_MONTH = pd.Period("2026-09", "M")

# Census Bureau regions; BLS's regional CPI and average-price areas are the same four (West includes AK and HI)
CENSUS_REGION_STATES = {
    "northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "midwest": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "south": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"],
    "west": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
}
STATE_REGION = {state: region for region, states in CENSUS_REGION_STATES.items() for state in states}

CPI_FEATURES = ["cpi_region_yoy", "cpi_us_yoy", "cpi_region_minus_us_yoy", "cpi_region_24m", "cpi_region_6m",
                "cpi_region_log_yoy", "cpi_region_log_diff_us"]
GAS_FEATURES = ["gas_region_price_3m", "gas_region_yoy", "gas_region_minus_us", "gas_region_real_price",
                "gas_region_6m_change", "log_real_gas_price", "log_rel_gas_price", "log_rel_real_gas", "log_gas_yoy",
                "log_real_gas_yoy", "gas_region_minus_us_yoy"]
FEATURES = CPI_FEATURES + GAS_FEATURES
DECIMALS = 4
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def read_fred_csv(path: Path, series: list[str]) -> pd.DataFrame:
    """FRED's CSV download (observation_date, then one column per series) as a month-indexed frame of the series.
    A blank value (a month BLS didn't publish) stays NaN."""
    df = pd.read_csv(path)
    missing = [s for s in series if s not in df.columns]
    if missing:
        raise ValueError(f"{path.name} lacks {missing}")
    return df.set_index(pd.PeriodIndex(df["observation_date"], freq="M"))[series].astype(float)


def read_bls_series_report(path: Path, series_id: str) -> pd.Series:
    """BLS Data Finder's SeriesReport CSV (a header block naming the series, then Year, Jan .. Dec, HALF1, HALF2) as
    a month-indexed series; the half-year averages are dropped and a blank month stays NaN."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    named = [line.split(",")[1] for line in lines if line.startswith("Series Id:")]
    if named != [series_id]:
        raise ValueError(f"{path.name} is series {named}, expected {series_id}")
    header = next(i for i, line in enumerate(lines) if line.startswith("Year,"))
    wide = pd.read_csv(path, skiprows=header, encoding="utf-8-sig").set_index("Year")[MONTH_NAMES]
    long = wide.stack(future_stack=True)
    months = [pd.Period(year=year, month=MONTH_NAMES.index(name) + 1, freq="M") for year, name in long.index]
    return pd.Series(long.to_numpy(dtype=float), index=pd.PeriodIndex(months), name=series_id)


def monthly_levels(national_cpi: pd.Series, regional_cpi: pd.DataFrame, gas: pd.DataFrame) -> pd.DataFrame:
    """The long monthly table (month, region, cpi, gas) from FIRST_MONTH to the last month any series has a value
    for, with every month present: a month a series lacks, or left blank, is NaN."""
    cpi = pd.concat([national_cpi.rename(CPI_SERIES["us"]), regional_cpi], axis=1)
    cpi = cpi.rename(columns={v: k for k, v in CPI_SERIES.items()})[list(CPI_SERIES)]
    gas = gas.rename(columns={v: k for k, v in GAS_SERIES.items()})[list(GAS_SERIES)]
    last = max(cpi.dropna(how="all").index.max(), gas.dropna(how="all").index.max())
    months = pd.period_range(FIRST_MONTH, last, freq="M")
    cpi, gas = cpi.reindex(months), gas.reindex(months)
    if (cpi <= 0).any().any() or (gas <= 0).any().any():
        raise ValueError("Non-positive CPI or gasoline levels")
    long = pd.concat({"cpi": cpi.stack(future_stack=True), "gas": gas.stack(future_stack=True)}, axis=1)
    long.index.names = ["month", "region"]
    return long.reset_index()


def read_monthly(path: Path = MONTHLY_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.assign(month=pd.PeriodIndex(df["month"], freq="M"))


def data_month(election_date: str) -> pd.Period:
    """The latest month whose CPI and average prices are public before election_date: month M counts as public from
    the first day of month M + PUBLICATION_LAG_MONTHS."""
    date = pd.Timestamp(election_date)
    if date.day == 1:
        raise ValueError(f"{election_date}: data_month assumes the election falls after the first of the month")
    month = date.to_period("M") - PUBLICATION_LAG_MONTHS
    if month.month != 9:
        raise ValueError(f"{election_date}: data month {month}; every year must use September, so that the 12- and "
                         "24-month changes compare September with September")
    return month


def pct_change(now: pd.Series, before: pd.Series) -> pd.Series:
    return 100 * (now / before - 1)


def log_ratio(x: pd.Series, ref: pd.Series | float) -> pd.Series:
    """log(x / ref) as log1p((x - ref) / ref), the form acs_supplement.py uses for log_rel_income."""
    return np.log1p((x - ref) / ref)


def region_features(monthly: pd.DataFrame, election_dates: dict[int, str]) -> pd.DataFrame:
    """One row per election year and Census region: the features of the module docstring, from the levels of each
    year's data_month and the months before it. NaN in any level a feature needs leaves that feature NaN."""
    cpi = monthly.pivot(index="month", columns="region", values="cpi")
    gas = monthly.pivot(index="month", columns="region", values="gas")
    deflator = cpi / cpi.reindex([DEFLATOR_BASE_MONTH]).iloc[0] * 100  # each series, DEFLATOR_BASE_MONTH = 100
    real_gas = gas / (deflator / 100)  # September 2005 dollars: regional gas by regional CPI, national by national

    def at(table: pd.DataFrame, month: pd.Period) -> pd.Series:
        """The regions' levels in one month; NaN for a month outside the table."""
        return table.reindex([month]).iloc[0]

    rows = []
    for year, election_date in sorted(election_dates.items()):
        m = data_month(election_date)
        cpi_now, gas_now = at(cpi, m), at(gas, m)
        cpi_yoy = pct_change(cpi_now, at(cpi, m - 12))
        f = pd.DataFrame(index=REGIONS)
        f["cpi_region_yoy"] = cpi_yoy
        f["cpi_us_yoy"] = cpi_yoy["us"]
        f["cpi_region_minus_us_yoy"] = cpi_yoy - cpi_yoy["us"]
        f["cpi_region_24m"] = pct_change(cpi_now, at(cpi, m - 24))
        f["cpi_region_6m"] = pct_change(cpi_now, at(cpi, m - 6))
        f["gas_region_price_3m"] = gas.reindex([m - 2, m - 1, m]).mean(skipna=False)
        gas_yoy = pct_change(gas_now, at(gas, m - 12))
        f["gas_region_yoy"] = gas_yoy
        f["gas_region_minus_us"] = gas_now - gas_now["us"]
        f["gas_region_real_price"] = gas_now * at(cpi, REAL_PRICE_REFERENCE_MONTH) / cpi_now
        f["gas_region_6m_change"] = pct_change(gas_now, at(gas, m - 6))
        cpi_log_yoy = 100 * log_ratio(cpi_now, at(cpi, m - 12))
        f["cpi_region_log_yoy"] = cpi_log_yoy
        f["cpi_region_log_diff_us"] = cpi_log_yoy - cpi_log_yoy["us"]
        real_now = at(real_gas, m)
        f["log_real_gas_price"] = np.log(real_now)
        f["log_rel_gas_price"] = log_ratio(gas_now, gas_now["us"])
        f["log_rel_real_gas"] = log_ratio(real_now, real_now["us"])
        f["log_gas_yoy"] = 100 * log_ratio(gas_now, at(gas, m - 12))
        f["log_real_gas_yoy"] = 100 * log_ratio(real_now, at(real_gas, m - 12))
        f["gas_region_minus_us_yoy"] = gas_yoy - gas_yoy["us"]
        rows.append(f.round(DECIMALS).rename_axis("census_region").reset_index()
                    .assign(year=year, election_date=election_date, price_data_month=str(m)))
    out = pd.concat(rows, ignore_index=True)
    return out[["year", "census_region", "election_date", "price_data_month", *FEATURES]]


def attach_regions(keys: pd.DataFrame, region_f: pd.DataFrame) -> pd.DataFrame:
    """keys (year, state and the rest of a race key) plus census_region, price_data_month and FEATURES from the
    state's region row. Raises for a state with no region or a year region_f lacks; blank features stay blank."""
    unmapped = sorted(set(keys["state"]) - set(STATE_REGION))
    if unmapped:
        raise ValueError(f"States with no Census region: {unmapped}")
    missing_years = sorted(set(keys["year"]) - set(region_f["year"]))
    if missing_years:
        raise ValueError(f"No CPI / gasoline features for {missing_years}")
    out = keys.assign(census_region=keys["state"].map(STATE_REGION))
    return out.merge(region_f[["year", "census_region", "price_data_month", *FEATURES]],
                     on=["year", "census_region"], how="left", validate="many_to_one")


def election_years(path: Path = ELECTION_DATES_PATH) -> dict[int, str]:
    dates = json.loads(path.read_text())
    return {int(y): d["election_day"] for y, d in dates.items() if FIRST_YEAR <= int(y) <= LAST_YEAR}


def month_ranges(months: list[pd.Period]) -> str:
    """Sorted months as consecutive ranges, e.g. "2004-01 to 2004-12, 2025-10"."""
    runs = []
    for month in months:
        if runs and month == runs[-1][1] + 1:
            runs[-1][1] = month
        else:
            runs.append([month, month])
    return ", ".join(str(a) if a == b else f"{a} to {b}" for a, b in runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=MONTHLY_PATH.parent, type=Path)
    args = parser.parse_args()

    regional = [s for r, s in CPI_SERIES.items() if r != "us"]
    regional_cpi = pd.concat([read_fred_csv(CPI_DIR / f"{s}.csv", [s]) for s in regional], axis=1)
    monthly = monthly_levels(read_bls_series_report(NATIONAL_CPI_PATH, CPI_SERIES["us"]), regional_cpi,
                             read_fred_csv(GAS_PATH, list(GAS_SERIES.values())))
    features = region_features(monthly, election_years())

    print(f"Monthly levels {monthly['month'].min()} to {monthly['month'].max()}; blank months:")
    for column in ["cpi", "gas"]:
        for region in CPI_SERIES:
            blank = monthly.loc[(monthly["region"] == region) & monthly[column].isna(), "month"]
            if len(blank):
                print(f"  {column} {region}: {month_ranges(list(blank))}")
    print("Data month per election: " + ", ".join(f"{y} {m}" for y, m in
                                                  features.groupby("year")["price_data_month"].first().items()))
    for column in FEATURES:
        years = sorted(int(y) for y in features.loc[features[column].isna(), "year"].unique())
        if years:
            print(f"{column} blank in {years}")
    monthly_out = monthly.assign(month=monthly["month"].astype(str))
    acs.write_csv_atomic(monthly_out, args.output_dir / MONTHLY_PATH.name)
    acs.write_csv_atomic(features, args.output_dir / REGION_PATH.name)
    print(f"-> {args.output_dir / MONTHLY_PATH.name} ({len(monthly_out)} rows), "
          f"{args.output_dir / REGION_PATH.name} ({len(features)} rows)")


if __name__ == "__main__":
    main()
