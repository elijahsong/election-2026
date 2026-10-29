"""Build congressional-district ACS demographics (one row per House election year x district) into acs.csv."""

import argparse
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from tqdm import tqdm

API_ROOT = "https://api.census.gov/data"

# House election year -> (ACS year, product). 2006 uses the 2007 ACS because the 2006 ACS is on 109th-Congress
# boundaries (pre-2006 Texas/Georgia redraws); 2020 uses the 5-year because the 2020 1-year was never released.
YEAR_TO_ACS_VINTAGE = {
    2006: (2007, "acs1"),
    2008: (2008, "acs1"),
    2010: (2010, "acs1"),
    2012: (2012, "acs1"),
    2014: (2014, "acs1"),
    2016: (2016, "acs1"),
    2018: (2018, "acs1"),
    2020: (2020, "acs5"),
    2022: (2022, "acs1"),
    2024: (2024, "acs1"),
    2026: None,
}

# Congress whose districts each vintage is tabulated on, checked against every row's NAME. Each has the same map as
# that election year: the 110th and 111th share the post-2006 maps, so the 2008 ACS's 110th districts match the 2008
# election, and the 2010 election ran on the 111th's; 2020 ran on the 116th's, except North Carolina below.
BOUNDARY_CONGRESS = {
    2006: 110, 2008: 110, 2010: 111, 2012: 113, 2014: 114,
    2016: 115, 2018: 116, 2020: 116, 2022: 118, 2024: 119,
}

# States whose election-year map the ACS never tabulated: their published rows (on the wrong map) are dropped and
# rebuilt from block groups by nc_2020.py. NC's court-ordered 2020 map (117th Congress) has no ACS tables; the
# 2016-2020 5-year uses the 116th.
# https://www2.census.gov/programs-surveys/acs/summary_file/2020/data/5_year_seq_by_state/A_Note_re_2016-2020_ACS_5yr_Congressional_Districts.pdf
REBUILT_BOUNDARIES = {2020: "NC"}

HOUSE_SEATS = 435

STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA",
    "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
    "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK",
    "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT",
    "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY",
}

# Census annotation values that stand in for an estimate that is missing, suppressed, or not computable
CENSUS_SENTINELS = {-111111111, -222222222, -333333333, -555555555, -666666666, -888888888, -999999999}

COLUMNS = [
    "year", "state", "district", "acs_vintage", "white", "black", "hispanic", "college", "median_income",
    "median_age", "moved_within_past_yr", "foreign_born", "unemployment", "poverty", "avg_household_size",
]
PCT_FIELDS = ["white", "black", "hispanic", "college", "moved_within_past_yr", "foreign_born", "unemployment", "poverty"]


class Ratio(NamedTuple):
    """scale * sum(num) / sum(den), or scale * (1 - sum(num) / sum(den)) when complement is set."""

    num: tuple[str, ...]
    den: tuple[str, ...]
    complement: bool = False
    scale: float = 100


def _cells(table: str, *numbers: int) -> tuple[str, ...]:
    return tuple(f"{table}_{n:03d}E" for n in numbers)


# 1-year vintages read the collapsed C tables. For unreliable areas Census withholds a whole detailed B table but
# still publishes its C version (e.g. B03002 is null for AR-1, LA-3, ME-1, MS-1, OH-14 in 2007). C cells are sums of
# B cells, so the estimates are identical. The 5-year product isn't filtered and has no C tables, so 2020 reads B.
COLLEGE_CELLS = {  # bachelor's and higher, male then female
    "B15002": (15, 16, 17, 18, 32, 33, 34, 35),  # bachelor's, master's, professional, doctorate
    "C15002": (8, 9, 16, 17),  # bachelor's, graduate or professional
}
FOREIGN_BORN_CELL = {"B05002": 13, "C05002": 8}

# C23001 (sex by age by employment status) is the labor-force table before B23025 starts in 2012. Ages 16-64 come in
# blocks of 7 cells (in LF, armed forces, civilian, employed, unemployed, ...) and 65+ in blocks of 5 with no
# armed-forces split, so their "in labor force" cell is already civilian. Female cells sit 46 rows below male.
C23001_AGES_UNDER_65 = ["16 to 19 years", "20 to 24 years", "25 to 44 years", "45 to 54 years", "55 to 64 years"]
C23001_AGES_65_PLUS = ["65 to 74 years", "75 years and over"]
C23001_CIVILIAN = [6 + 7 * i for i in range(5)]
C23001_UNEMPLOYED = [8 + 7 * i for i in range(5)]
C23001_LF_65_PLUS = [39 + 5 * i for i in range(2)]
C23001_UNEMPLOYED_65_PLUS = [41 + 5 * i for i in range(2)]
C23001_FEMALE_OFFSET = 46

UNEMPLOYMENT_C23001 = Ratio(
    num=_cells("C23001", *(n + off for off in (0, C23001_FEMALE_OFFSET)
                           for n in C23001_UNEMPLOYED + C23001_UNEMPLOYED_65_PLUS)),
    den=_cells("C23001", *(n + off for off in (0, C23001_FEMALE_OFFSET)
                           for n in C23001_CIVILIAN + C23001_LF_65_PLUS)),
)
UNEMPLOYMENT_B23025 = Ratio(num=("B23025_005E",), den=("B23025_003E",))


def vintage_fields(acs_year: int, product: str) -> dict[str, Ratio | str]:
    """Field formulas for one ACS vintage."""
    t = "C" if product == "acs1" else "B"
    race, college, birth, poverty = f"{t}03002", f"{t}15002", f"{t}05002", f"{t}17001"
    return {
        "white": Ratio(_cells(race, 3), _cells(race, 1)),
        "black": Ratio(_cells(race, 4), _cells(race, 1)),
        "hispanic": Ratio(_cells(race, 12), _cells(race, 1)),
        # B15002/C15002 in every year, since B15003 is missing from the 2007 ACS
        "college": Ratio(_cells(college, *COLLEGE_CELLS[college]), _cells(college, 1)),
        "median_income": "B19013_001E",
        "median_age": "B01002_001E",
        "moved_within_past_yr": Ratio(("B07003_004E",), ("B07003_001E",), complement=True),
        "foreign_born": Ratio(_cells(birth, FOREIGN_BORN_CELL[birth]), _cells(birth, 1)),
        "unemployment": UNEMPLOYMENT_C23001 if acs_year < 2012 else UNEMPLOYMENT_B23025,
        "poverty": Ratio(_cells(poverty, 2), _cells(poverty, 1)),
        "avg_household_size": "B25010_001E",
    }


ACS_FIELDS = {year: vintage_fields(*vintage) for year, vintage in YEAR_TO_ACS_VINTAGE.items() if vintage is not None}

# Expected label endings, matched on letters and digits only because punctuation drifts between vintages
# ("Total:", "--", "foreign-born"). Concepts are checked separately because some labels are just "Total".
EXPECTED_LABELS = {
    **{f"{t}03002_{n:03d}E": label for t in "BC" for n, label in (
        (1, "Total"),
        (3, "Total!!Not Hispanic or Latino!!White alone"),
        (4, "Total!!Not Hispanic or Latino!!Black or African American alone"),
        (12, "Total!!Hispanic or Latino"))},
    "B15002_001E": "Total",
    **{f"B15002_{n + off:03d}E": f"Total!!{sex}!!{degree}"
       for off, sex in ((0, "Male"), (17, "Female"))
       for n, degree in ((15, "Bachelor's degree"), (16, "Master's degree"),
                         (17, "Professional school degree"), (18, "Doctorate degree"))},
    "C15002_001E": "Total",
    **{f"C15002_{n + off:03d}E": f"Total!!{sex}!!{degree}"
       for off, sex in ((0, "Male"), (8, "Female"))
       for n, degree in ((8, "Bachelor's degree"), (9, "Graduate or professional degree"))},
    "B19013_001E": "Median household income in the past 12 months (in {acs_year} inflation-adjusted dollars)",
    "B01002_001E": "Median age!!Total",
    "B07003_001E": "Total",
    "B07003_004E": "Total!!Same house 1 year ago",
    "B05002_001E": "Total",
    "B05002_013E": "Total!!Foreign born",
    "C05002_001E": "Total",
    "C05002_008E": "Total!!Foreign born",
    "B23025_003E": "Total!!In labor force!!Civilian labor force",
    "B23025_005E": "Total!!In labor force!!Civilian labor force!!Unemployed",
    **{f"{t}17001_{n:03d}E": label for t in "BC" for n, label in (
        (1, "Total"), (2, "Total!!Income in the past 12 months below poverty level"))},
    "B25010_001E": "Total",
}
for off, sex in ((0, "Male"), (C23001_FEMALE_OFFSET, "Female")):
    for i, age in enumerate(C23001_AGES_UNDER_65):
        EXPECTED_LABELS[f"C23001_{C23001_CIVILIAN[i] + off:03d}E"] = f"Total!!{sex}!!{age}!!In labor force!!Civilian"
        EXPECTED_LABELS[f"C23001_{C23001_UNEMPLOYED[i] + off:03d}E"] = f"Total!!{sex}!!{age}!!In labor force!!Civilian!!Unemployed"
    for i, age in enumerate(C23001_AGES_65_PLUS):
        EXPECTED_LABELS[f"C23001_{C23001_LF_65_PLUS[i] + off:03d}E"] = f"Total!!{sex}!!{age}!!In labor force"
        EXPECTED_LABELS[f"C23001_{C23001_UNEMPLOYED_65_PLUS[i] + off:03d}E"] = f"Total!!{sex}!!{age}!!In labor force!!Unemployed"

TABLE_CONCEPTS = {
    **{f"{t}03002": "Hispanic or Latino origin by race" for t in "BC"},
    **{f"{t}15002": "Sex by educational attainment for the population 25 years and over" for t in "BC"},
    "B19013": "Median household income in the past 12 months",
    "B01002": "Median age by sex",
    "B07003": "Geographical mobility in the past year by sex",
    **{f"{t}05002": "Place of birth by" for t in "BC"},
    "C23001": "Sex by age by employment status for the population 16 years and over",
    "B23025": "Employment status for the population 16 years and over",
    **{f"{t}17001": "Poverty status in the past 12 months by sex by age" for t in "BC"},
    "B25010": "Average household size of occupied housing units by tenure",
}

ROUNDING = {**{f: 2 for f in PCT_FIELDS}, "avg_household_size": 2, "median_age": 1, "median_income": 0}

DISTRICT_GEO = {"for": "congressional district:*", "in": "state:*"}
MAX_VARS_PER_REQUEST = 48  # the API caps a request at 50 variables, including NAME
RETRY_STATUSES = {429, 500, 502, 503, 504}


def acs_vintage_label(year: int) -> str:
    """Label the ACS product used for an election year, e.g. 2018_1yr or 2016-2020_5yr."""
    acs_year, product = resolve_vintage(year)
    return f"{acs_year - 4}-{acs_year}_5yr" if product == "acs5" else f"{acs_year}_1yr"


def resolve_vintage(year: int) -> tuple[int, str]:
    """Return (ACS year, product) for an election year, or raise if no matching ACS release exists."""
    if year not in YEAR_TO_ACS_VINTAGE:
        raise ValueError(f"Unsupported election year {year}; supported: {sorted(YEAR_TO_ACS_VINTAGE)}")
    if YEAR_TO_ACS_VINTAGE[year] is None:
        raise ValueError(f"No ACS vintage yet for the {year} election; add it to YEAR_TO_ACS_VINTAGE once released")
    return YEAR_TO_ACS_VINTAGE[year]


def spec_variables(specs: dict[str, Ratio | str]) -> list[str]:
    """Census variable IDs behind field formulas, in a stable order."""
    seen = {}
    for spec in specs.values():
        for var in (spec,) if isinstance(spec, str) else spec.num + spec.den:
            seen[var] = None
    return list(seen)


def field_variables(year: int, fields: list[str] | None = None) -> list[str]:
    """Census variable IDs behind an election year's fields (all fields by default), in a stable order."""
    specs = ACS_FIELDS[year]
    return spec_variables({field: specs[field] for field in fields or specs})


def _get_json(url: str, retries: int = 4, timeout: float = 60, backoff: float = 2.0):
    """GET a Census API URL and parse JSON, retrying transient failures. The key is redacted from errors."""
    shown = re.sub(r"key=[^&]+", "key=***", url)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUSES or attempt == retries:
                detail = e.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(f"Census API HTTP {e.code} for {shown}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries:
                raise RuntimeError(f"Census API request failed for {shown}: {e}") from None
        time.sleep(backoff * 2**attempt)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # an invalid or missing key returns an HTML page with status 200
        raise RuntimeError(f"Census API returned non-JSON for {shown} (check CENSUS_API): {body[:200]!r}") from None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def census_key() -> str:
    key = os.environ.get("CENSUS_API")
    if not key:
        raise RuntimeError("Set the CENSUS_API environment variable to a Census API key")
    return key


def verify_variables(year: int, variables: list[str] | None = None, labels: dict[str, str] = EXPECTED_LABELS,
                     concepts: dict[str, str] = TABLE_CONCEPTS) -> None:
    """Check each variable (all the year's fields by default) has the expected label and concept in that vintage."""
    acs_year, product = resolve_vintage(year)
    variables = variables or field_variables(year)
    problems = []
    tables = sorted({v.split("_")[0] for v in variables})
    # disable=None hides the bar when output is not a terminal (logs, tests)
    for table in tqdm(tables, desc=f"{year} metadata", unit="table", leave=False, disable=None):
        meta = _get_json(f"{API_ROOT}/{acs_year}/acs/{product}/groups/{table}.json")["variables"]
        for var in (v for v in variables if v.startswith(table + "_")):
            if var not in meta:
                problems.append(f"{var}: missing")
                continue
            label = _norm(meta[var]["label"])
            expected = _norm(labels[var].format(acs_year=acs_year))
            if not label.endswith(expected):
                problems.append(f"{var}: label {meta[var]['label']!r} does not end with {labels[var]!r}")
            concept = meta[var].get("concept")  # absent in the 2010 metadata
            if concept and _norm(concepts[table]) not in _norm(concept):
                problems.append(f"{var}: concept {concept!r} is not {concepts[table]!r}")
    if problems:
        raise ValueError(f"ACS {acs_year} {product} variable metadata mismatch:\n  " + "\n  ".join(problems))


def fetch_acs(year: int, key: str, geo: dict[str, str] = DISTRICT_GEO, variables: list[str] | None = None) -> pd.DataFrame:
    """Fetch an election year's vintage estimates (all its fields by default) for one geography as raw strings."""
    acs_year, product = resolve_vintage(year)
    variables = variables or field_variables(year)
    frames = []
    chunks = range(0, len(variables), MAX_VARS_PER_REQUEST)
    for i in tqdm(chunks, desc=f"{year} {geo['for']}", unit="request", leave=False, disable=None):
        params = {"get": ",".join(["NAME", *variables[i : i + MAX_VARS_PER_REQUEST]]), **geo, "key": key}
        rows = _get_json(f"{API_ROOT}/{acs_year}/acs/{product}?{urllib.parse.urlencode(params)}")
        chunk = pd.DataFrame(rows[1:], columns=rows[0])
        geo_cols = [c for c in chunk.columns if c != "NAME" and c not in variables]
        if chunk.duplicated(geo_cols).any():
            raise ValueError(f"Duplicate {geo['for']} rows in ACS {acs_year} {product} response")
        frames.append(chunk if not frames else chunk.drop(columns="NAME"))
    raw = frames[0]
    for chunk in frames[1:]:
        raw = raw.merge(chunk, on=geo_cols, how="outer", validate="one_to_one")
    return raw


def parse_estimates(raw: pd.DataFrame, variables: list[str], ids: pd.Series) -> pd.DataFrame:
    """Parse estimate strings to floats, raising on any missing value or Census sentinel (ids label rows in errors)."""
    values = raw[variables].apply(pd.to_numeric, errors="coerce")
    # every estimate used here is a count, median, or average, so any negative value is an annotation code
    bad = values.isna() | values.isin(CENSUS_SENTINELS) | (values < 0)
    if bad.any().any():
        rows, cols = bad.to_numpy().nonzero()
        shown = raw[variables].astype(object).where(raw[variables].notna(), "null")
        where = [f"{ids.iat[r]}: {variables[c]}={shown.iat[r, c]}" for r, c in zip(rows, cols)]
        raise ValueError(f"{len(where)} missing or annotated ACS estimates (null = withheld by Census):\n  "
                         + "\n  ".join(where[:20]))
    return values


def compute_fields(values: pd.DataFrame, specs: dict[str, Ratio | str], ids: pd.Series) -> pd.DataFrame:
    """Evaluate field formulas on parsed estimates, unrounded."""
    out = pd.DataFrame(index=values.index)
    for field, spec in specs.items():
        if isinstance(spec, str):
            out[field] = values[spec]
            continue
        num, den = values[list(spec.num)].sum(axis=1), values[list(spec.den)].sum(axis=1)
        if (den <= 0).any():
            raise ValueError(f"{field}: zero denominator for {ids[den <= 0].tolist()}")
        share = num / den
        out[field] = spec.scale * (1 - share if spec.complement else share)
    return out


def round_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Round estimates to the published precision; median_income becomes an integer."""
    out = df.round({f: d for f, d in ROUNDING.items() if f in df})
    if "median_income" in out:
        out["median_income"] = out["median_income"].astype(int)
    return out


def transform(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """Turn a raw API response into schema rows for the 50 states on the election year's boundaries."""
    df = raw[raw["state"].isin(STATE_FIPS)]  # drops DC and Puerto Rico
    df = df[df["congressional district"] != "ZZ"]  # ZZ = area not assigned to any district
    cd = df["congressional district"]
    if not cd.str.fullmatch(r"\d{2}").all():
        raise ValueError(f"Unexpected district codes: {sorted(cd[~cd.str.fullmatch(r'\d{2}')].unique())}")

    congress = df["NAME"].str.extract(r"\((\d+)(?:st|nd|rd|th) Congress\)", expand=False)
    expected = BOUNDARY_CONGRESS[year]
    if (congress != str(expected)).any():
        found = sorted(congress.fillna("none").unique())
        raise ValueError(f"{year} expects {expected}th-Congress districts but the ACS NAMEs show {found}")

    df = df.assign(state=df["state"].map(STATE_FIPS), district=cd.astype(int))  # at-large "00" -> 0
    df = df[df["state"] != REBUILT_BOUNDARIES.get(year)].reset_index(drop=True)
    ids = df["state"] + "-" + df["district"].astype(str)
    values = parse_estimates(df, field_variables(year), ids)

    keys = pd.DataFrame({"year": year, "state": df["state"], "district": df["district"],
                         "acs_vintage": acs_vintage_label(year)})
    out = pd.concat([keys, round_fields(compute_fields(values, ACS_FIELDS[year], ids))], axis=1)
    return out[COLUMNS].sort_values(["state", "district"], ignore_index=True)


def normalize_counts(district_counts_by_year: dict, year: int) -> dict[str, int]:
    """Pick and validate one year's per-state district counts from a JSON-style {year: {state: n}} mapping."""
    by_year = {int(y): counts for y, counts in district_counts_by_year.items()}
    if year not in by_year:
        raise ValueError(f"No district counts supplied for {year}")
    counts = {str(s): n for s, n in by_year[year].items()}
    if set(counts) != set(STATE_FIPS.values()):
        missing, extra = set(STATE_FIPS.values()) - set(counts), set(counts) - set(STATE_FIPS.values())
        raise ValueError(f"{year} district counts must cover exactly the 50 states; missing {sorted(missing)}, extra {sorted(extra)}")
    if not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in counts.values()):
        raise ValueError(f"{year} district counts must be positive integers")
    if sum(counts.values()) != HOUSE_SEATS:
        raise ValueError(f"{year} district counts total {sum(counts.values())}, not {HOUSE_SEATS}")
    return counts


def validate(df: pd.DataFrame, year: int, counts: dict[str, int]) -> None:
    """Check one year's rows against the district counts, the primary key, missing values, and plausible ranges."""
    if list(df.columns) != COLUMNS:
        raise ValueError(f"Columns {list(df.columns)} do not match the schema")
    if df.duplicated(["year", "state", "district"]).any():
        raise ValueError(f"Duplicate (year, state, district) keys in {year}")

    found = df.groupby("state")["district"].apply(set).to_dict()
    problems = []
    for state, n in counts.items():
        expected = {0} if n == 1 else set(range(1, n + 1))
        if found.get(state, set()) != expected:
            problems.append(f"{state}: expected {sorted(expected)}, got {sorted(found.get(state, set()))}")
    if problems:
        raise ValueError(f"{year} districts do not match the supplied counts:\n  " + "\n  ".join(problems))
    if len(df) != HOUSE_SEATS:
        raise ValueError(f"{year} has {len(df)} districts, expected {HOUSE_SEATS}")

    if df.isna().any().any():
        raise ValueError(f"Missing values in {year}: {df.columns[df.isna().any()].tolist()}")
    ranges = {**{f: (0, 100) for f in PCT_FIELDS}, "median_income": (1, 300_000),
              "median_age": (15, 70), "avg_household_size": (1, 6)}
    for field, (lo, hi) in ranges.items():
        bad = df[(df[field] < lo) | (df[field] > hi)]
        if len(bad):
            raise ValueError(f"{year} {field} outside [{lo}, {hi}]: {bad[['state', 'district', field]].to_dict('records')[:10]}")
    if (df["white"] + df["black"] + df["hispanic"] > 100.01).any():
        raise ValueError(f"{year} white + black + hispanic exceeds 100%")


def update_csv(df_year: pd.DataFrame, path: str | Path, year: int) -> pd.DataFrame:
    """Replace one election year's rows in the master CSV, keep other years, sort, and write atomically."""
    path = Path(path)
    if path.exists():
        existing = pd.read_csv(path, dtype={"state": str, "acs_vintage": str})
        if list(existing.columns) != COLUMNS:
            raise ValueError(f"{path} columns {list(existing.columns)} do not match the schema")
        existing = existing[existing["year"] != year]
        combined = pd.concat([existing, df_year], ignore_index=True) if len(existing) else df_year
    else:
        combined = df_year
    combined = combined.sort_values(["year", "state", "district"], ignore_index=True)
    if combined.duplicated(["year", "state", "district"]).any():
        raise ValueError(f"Duplicate (year, state, district) keys after merging into {path}")
    write_csv_atomic(combined, path)
    return combined


def write_csv_atomic(df: pd.DataFrame, path: str | Path) -> None:
    """Write a CSV via a temp file in the same directory, so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            df.to_csv(f, index=False)
        # mkstemp creates owner-only files; give the CSV the permissions a normal write would (0666 minus umask)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def build_acs_year(year: int, district_counts_by_year: dict, output_path: str | Path = "acs.csv") -> pd.DataFrame:
    """Fetch, validate, and upsert one House election year's district demographics into output_path."""
    resolve_vintage(year)
    counts = normalize_counts(district_counts_by_year, year)
    key = census_key()
    verify_variables(year)
    df = transform(fetch_acs(year, key), year)
    if year in REBUILT_BOUNDARIES:
        import nc_2020  # imported here because nc_2020 builds on this module

        df = pd.concat([df, nc_2020.rebuild(key)]).sort_values(["state", "district"], ignore_index=True)
    validate(df, year, counts)
    update_csv(df, output_path, year)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("years", nargs="*", type=int, help="House election years to build")
    parser.add_argument("--all", action="store_true", help="build every year with an ACS vintage")
    parser.add_argument("--counts", default="data/reference/district_counts.json", help="JSON {year: {state: n}}")
    parser.add_argument("--output", default="data/interim/acs.csv")
    args = parser.parse_args()

    years = [y for y, v in YEAR_TO_ACS_VINTAGE.items() if v is not None] if args.all else args.years
    if not years:
        parser.error("give election years or --all")
    with open(args.counts) as f:
        district_counts_by_year = json.load(f)
    bar = tqdm(years, desc="ACS", unit="year")
    for year in bar:
        bar.set_postfix_str(str(year))
        df = build_acs_year(year, district_counts_by_year, args.output)
        tqdm.write(f"{year} ({df['acs_vintage'].iat[0]}): {len(df)} districts -> {args.output}")


if __name__ == "__main__":
    main()
