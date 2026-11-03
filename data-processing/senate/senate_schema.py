"""Column order, dtypes and keys of every file in data/processed/senate/, and one check for all of them.

The dtypes are what pandas gives when the file is read back with read_output (county_fips and county_id as text):
int64 never null, Int64 a nullable integer, float64 (NaN allowed where the column is nullable), str text, bool.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "house"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acs  # write_csv_atomic, STATE_FIPS
from cpi_gas import FEATURES as _CPI_GAS_FEATURES
from unemployment import FEATURES as _UNEMPLOYMENT_FEATURES

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data/processed/senate"

KEY = ["year", "state", "class"]
SUPP_KEY = [*KEY, "contest"]
COUNTY_KEY = [*KEY, "county_id", "county_fips"]
AT_LARGE = {"AK": "02000", "HI": "15000"}  # synthetic state-aggregate codes, not county FIPS

_VOTES = ["dem_votes", "gop_votes", "3p_votes", "total_votes"]
_PCTS = ["dem_pct", "gop_pct", "3p_pct"]
_COUNTS = ["dem_candidate_count", "gop_candidate_count", "3p_candidate_count"]

TARGETS = {
    "year": "int64", "state": "str", "class": "int64",
    **dict.fromkeys(_VOTES, "float64"), **dict.fromkeys(_PCTS, "float64"),
    "margin_votes": "float64", "margin_raw": "float64", "margin_2p_norm": "float64",
    "race_type": "str", "winner_party": "str", **dict.fromkeys(_COUNTS, "int64"),
    "is_rcv": "bool", "is_special_election": "bool", "adj_margin_2p_norm": "float64",
    "dem_votes_imputed": "float64", "gop_votes_imputed": "float64", "is_imputed": "bool", "imputation_method": "str",
    "needs_imputation": "bool", "is_nonstandard_race": "bool",
    "source_totalvotes": "int64", "nonvote_votes": "int64", "is_unofficial": "bool", "quality_flags": "str",
    "source": "str",
}
TARGETS_SUPP = {
    "year": "int64", "state": "str", "class": "int64", "contest": "str", "stage": "str", "election_date": "str",
    **dict.fromkeys(_VOTES, "Int64"), **dict.fromkeys(_PCTS, "float64"),
    "margin_votes": "Int64", "margin_raw": "float64", "margin_2p_norm": "float64",
    "race_type": "str", "winner_party": "str", **dict.fromkeys(_COUNTS, "int64"),
    "is_rcv": "bool", "is_special_election": "bool", "source_totalvotes": "Int64", "nonvote_votes": "Int64",
    "is_unofficial": "bool", "quality_flags": "str", "source": "str",
}
TARGETS_CANDIDATES = {
    "year": "int64", "state": "str", "class": "int64", "contest": "str", "stage": "str", "candidate": "str",
    "parties": "str", "party_group": "str", "is_candidate": "bool", "is_writein": "bool", "votes": "int64",
    "source": "str",
}
BASIC_F = {"year": "int64", "state": "str", "class": "int64", "gen_margin_raw": "float64",
           "partisan_lean": "float64", "inc_signed": "int64"}
# the price features may be blank where a month they need is missing (none is, 2006-2026)
_PRICE_FEATURES = [*_CPI_GAS_FEATURES, *_UNEMPLOYMENT_FEATURES]
PRICES_F = {"year": "int64", "state": "str", "class": "int64", "census_region": "str", "price_data_month": "str",
            **dict.fromkeys(_PRICE_FEATURES, "float64")}
# senate_acs.py's fields (white .. housing_burden) and national comparisons; never blank
_ACS_VALUES = ["white", "black", "hispanic", "college", "median_income", "median_age", "moved_within_past_yr",
               "foreign_born", "unemployment", "poverty", "avg_household_size", "housing_burden", "rel_college",
               "log_rel_income", "rel_unemployment", "rel_housing_burden"]
ACS_F = {"year": "int64", "state": "str", "class": "int64", "acs_vintage": "str",
         **dict.fromkeys(_ACS_VALUES, "float64")}
FEC_F = {"year": "int64", "state": "str", "class": "int64", "fundraising_advantage": "float64",
         "spending_advantage": "float64", "small_dollar_share_diff": "float64", "dem_fec_missing": "bool",
         "gop_fec_missing": "bool"}
_PVI = {"prev_pres_year": "int64", "anteprev_pres_year": "int64", "prev_rel_dem_2p": "float64",
        "anteprev_rel_dem_2p": "float64", "pvi": "float64", "partisan_lean": "float64"}
STATE_PVI = {"year": "int64", "state": "str", **_PVI}
COUNTY_PVI = {"year": "int64", "state": "str", "county_id": "str", "county_fips": "str", "county_name": "str",
              "is_at_large": "bool", **_PVI}
COUNTY_RESULTS = {
    "year": "int64", "state": "str", "class": "int64", "county_id": "str", "county_fips": "str", "county_name": "str",
    "is_at_large": "bool", **dict.fromkeys(_VOTES, "Int64"), **dict.fromkeys(_PCTS, "float64"),
    "margin_votes": "Int64", "margin_raw": "float64", "margin_2p_norm": "float64", "race_type": "str",
    "is_special_election": "bool", "is_rcv": "bool", "source_totalvotes": "Int64", "nonvote_votes": "Int64",
    "is_unofficial": "bool", "quality_flags": "str", "source": "str", "source_version": "str",
}
COUNTY_RESULTS_SUPP = {**COUNTY_RESULTS, "contest": "str", "stage": "str", "election_date": "str"}
NAT_POLITICAL_F = {
    "year": "int64", "gb_cutoff_date": "str", "gb_date": "str",
    **dict.fromkeys(["gb_margin_raw", "gb_margin_2p_norm", "gb_dem_pct", "gb_gop_pct", "gb_und_pct", "gb_dem_hi",
                     "gb_dem_lo", "gb_gop_hi", "gb_gop_lo", "gb_margin_30d_mean_raw", "gb_margin_30d_ewma_raw",
                     "gb_momentum_30d_raw", "gb_margin_30d_mean_2p_norm", "gb_margin_30d_ewma_2p_norm",
                     "gb_momentum_30d_2p_norm"], "float64"),
}
# 2026's generic-ballot row is one election-eve value transcribed by hand from FiftyPlusOne (01-generic-ballot.ipynb):
# no hi/lo bands and, without 30 days of history, no trend fields. Every FiveThirtyEight year has all of them.
GB_BANDS_AND_TRENDS = ["gb_dem_hi", "gb_dem_lo", "gb_gop_hi", "gb_gop_lo", "gb_margin_30d_mean_raw",
                       "gb_margin_30d_ewma_raw", "gb_momentum_30d_raw", "gb_margin_30d_mean_2p_norm",
                       "gb_margin_30d_ewma_2p_norm", "gb_momentum_30d_2p_norm"]
# Race identity shared by polls.csv and expert-ratings.csv (senate_polls.race_table): special is the
# senate_incumbency.csv flag and election_id is "<year>_<state>_<class>"
RACE_ID = {"year": "int64", "state": "str", "class": "int64", "special": "bool", "election_id": "str"}
# senate_polls.py: polls_clean.csv has one row per poll question (source_line is its line in the raw file)
POLLS_CLEAN = {
    **RACE_ID, "stage": "str", "rcp_table": "int64", "pollster": "str", "pollster_raw": "str",
    "pollster_partisan": "str", "start_date": "str", "end_date": "str", "sample_size": "Int64", "sample_type": "str",
    "dem_candidate": "str", "gop_candidate": "str", "ind_candidate": "str", "matchup": "str",
    "dem_pct": "float64", "gop_pct": "float64", "other_pct": "float64", "margin_raw": "float64",
    "poll_margin_dem": "float64", "is_duplicate": "bool", "duplicate_of": "Int64", "quality_flags": "str",
    "source_line": "int64",
}
# polls.csv: one row per race (senate_incumbency.csv, 2006-2026), the mean of its eligible polls
POLLS = {
    **RACE_ID, "election_date": "str", "cutoff_date": "str", "window_start": "str", "poll_margin_dem": "float64",
    "n_polls": "Int64", "n_pollsters": "Int64", "latest_poll_date": "str", "poll_available": "bool",
    "poll_source": "str", "quality_flags": "str",
}
# senate_ratings.py: expert-ratings.csv, one row per race 2016-2026
_PROVIDERS = ["sabato", "cook", "ie"]
EXPERT_RATINGS = {
    **RACE_ID, **{f"{p}_rating": "str" for p in _PROVIDERS}, **{f"{p}_prob_dem": "float64" for p in _PROVIDERS},
    "expert_consensus_dem": "float64", "n_experts": "int64", "coverage_complete": "bool", "as_of_date": "str",
    "quality_flags": "str", "source": "str",
}
PRES_STATE = {"year": "int64", "state": "str", **dict.fromkeys(_VOTES, "Int64"),
              "dem_2p_norm": "float64", "gop_2p_norm": "float64", "margin_2p_norm": "float64"}
PRES_COUNTY = {"year": "int64", "state": "str", "county_id": "str", "county_fips": "str", "county_name": "str",
               **dict.fromkeys(_VOTES, "Int64"),
               "dem_2p_norm": "float64", "gop_2p_norm": "float64", "margin_2p_norm": "float64"}



def _without(dtypes: dict[str, str], key: list[str]) -> dict[str, str]:
    return {c: t for c, t in dtypes.items() if c not in key}


# basic_f, acs_f, fec_f and prices_f joined on KEY, then nat_political_f on year (as the House modeling.csv)
MODELING = {**BASIC_F, **_without(ACS_F, KEY), **_without(FEC_F, KEY), **_without(PRICES_F, KEY),
            **_without(NAT_POLITICAL_F, ["year"])}

# file -> (columns and dtypes, key, columns allowed to be null)
OUTPUTS = {
    "targets.csv": (TARGETS, KEY, None),
    "targets_raw.csv": (TARGETS, KEY, None),
    "targets_supp.csv": (TARGETS_SUPP, SUPP_KEY, None),
    "targets_candidates.csv": (TARGETS_CANDIDATES, [*SUPP_KEY, "candidate"], {"parties"}),
    "basic_f.csv": (BASIC_F, KEY, {"gen_margin_raw"}),
    "prices_f.csv": (PRICES_F, KEY, set(_PRICE_FEATURES)),
    "acs_f.csv": (ACS_F, KEY, set()),
    "fec_f.csv": (FEC_F, KEY, {"small_dollar_share_diff"}),
    # a year without a generic-ballot row would leave gen_margin_raw and every gb_ column blank
    "modeling.csv": (MODELING, KEY, {"gen_margin_raw", "small_dollar_share_diff", *_PRICE_FEATURES,
                                     *(c for c in NAT_POLITICAL_F if c != "year")}),
    "state_pvi.csv": (STATE_PVI, ["year", "state"], set()),
    "county_pvi.csv": (COUNTY_PVI, ["year", "state", "county_id", "county_fips"], set()),
    "county_results.csv": (COUNTY_RESULTS, COUNTY_KEY, None),
    "county_results_supp.csv": (COUNTY_RESULTS_SUPP, [*COUNTY_KEY, "contest"], None),
    "nat_political_f.csv": (NAT_POLITICAL_F, ["year"], set(GB_BANDS_AND_TRENDS)),
    "polls_clean.csv": (POLLS_CLEAN, ["source_line"], {"pollster_partisan", "sample_size", "sample_type",
                                                       "dem_candidate", "gop_candidate", "ind_candidate", "dem_pct",
                                                       "gop_pct", "other_pct", "margin_raw", "poll_margin_dem",
                                                       "duplicate_of", "quality_flags"}),
    # n_polls / n_pollsters are blank for a published average (2026 FiftyPlusOne), whose poll count isn't known
    "polls.csv": (POLLS, KEY, {"poll_margin_dem", "n_polls", "n_pollsters", "latest_poll_date", "poll_source",
                               "quality_flags"}),
    "expert-ratings.csv": (EXPERT_RATINGS, KEY, {*(c for c in EXPERT_RATINGS if c.startswith(tuple(_PROVIDERS))),
                                                 "expert_consensus_dem", "as_of_date", "quality_flags"}),
    "pres_state_votes.csv": (PRES_STATE, ["year", "state"], set()),
    "pres_county_votes.csv": (PRES_COUNTY, ["year", "state", "county_id", "county_fips"], set()),
}
TEXT_COLUMNS = {"county_id": str, "county_fips": str}


def read_output(name: str, directory: Path = OUTPUT_DIR) -> pd.DataFrame:
    """A processed file with its documented dtypes: nullable integers as Int64 and FIPS codes as text."""
    dtypes, _, _ = OUTPUTS[name]
    read_as = {c: "Int64" for c, t in dtypes.items() if t == "Int64"} | \
              {c: str for c in TEXT_COLUMNS if c in dtypes}
    return pd.read_csv(directory / name, dtype=read_as)


def check_output(df: pd.DataFrame, name: str) -> list[str]:
    """Blocking problems of a processed frame: column order, a missing or duplicate key, a null outside the nullable
    columns (None means "any column may be null but the key"), and the FIPS rules (five digits, the state's prefix,
    and the synthetic AT_LARGE codes only on AK/HI At-Large rows)."""
    dtypes, key, nullable = OUTPUTS[name]
    problems = []
    if list(df.columns) != list(dtypes):
        problems.append(f"{name}: columns {list(df.columns)} != {list(dtypes)}")
        return problems
    if df[key].isna().any().any():
        problems.append(f"{name}: null key values in {df[key].columns[df[key].isna().any()].tolist()}")
    dup = df[df.duplicated(key, keep=False)]
    if len(dup):
        problems.append(f"{name}: {len(dup)} rows share a key, e.g. {dup[key].head(3).values.tolist()}")
    if nullable is not None:
        nulls = [c for c in df.columns if c not in nullable and df[c].isna().any()]
        if nulls:
            problems.append(f"{name}: unexpected nulls in {nulls}")
    if "county_fips" in df.columns:
        fips = df["county_fips"].astype(str)
        state_fips = {usps: code for code, usps in acs.STATE_FIPS.items()}
        bad = ~fips.str.fullmatch(r"\d{5}") | (fips.str[:2] != df["state"].map(state_fips))
        if bad.any():
            problems.append(f"{name}: {int(bad.sum())} county_fips not five digits with the state's prefix, e.g. "
                            f"{df.loc[bad, ['state', 'county_fips']].head(3).values.tolist()}")
        synthetic = fips.isin(AT_LARGE.values())
        at_large = df["county_id"] == "At-Large"
        if (synthetic != at_large).any() or (df.loc[at_large, "state"].map(AT_LARGE) != fips[at_large]).any():
            problems.append(f"{name}: synthetic At-Large codes must be exactly the AK/HI At-Large rows")
        if "is_at_large" in df.columns and (df["is_at_large"] != at_large).any():
            problems.append(f"{name}: is_at_large must mark exactly the At-Large rows")
    return problems


def check_dtypes(name: str, directory: Path = OUTPUT_DIR) -> list[str]:
    """Columns whose dtype on re-reading the written file isn't the documented one (an all-null nullable text
    column reads as float64 and is accepted)."""
    dtypes, _, _ = OUTPUTS[name]
    df = read_output(name, directory)
    problems = []
    for column, expected in dtypes.items():
        got = str(df[column].dtype)
        got = "str" if got in ("object", "string", "str") else got
        # a vote count with no blanks reads as int64; is_imputed with blanks (no target) reads as text
        if got != expected and not (expected == "str" and df[column].isna().all()) \
                and not (expected == "float64" and got == "int64") \
                and not (column == "is_imputed" and got == "str" and df[column].isna().any()):
            problems.append(f"{name}: {column} reads as {got}, documented {expected}")
    return problems


def write_outputs(frames: dict[str, pd.DataFrame], directory: Path = OUTPUT_DIR) -> None:
    """Check each frame (check_output), write it atomically, then check its dtypes as read back; raises on any
    problem, before writing when the frames themselves are wrong."""
    problems = [p for name, df in frames.items() for p in check_output(df, name)]
    if problems:
        raise ValueError("Output problems:\n" + "\n".join(problems))
    for name, df in frames.items():
        acs.write_csv_atomic(df, directory / name)
        print(f"-> {directory / name} ({len(df)} rows)")
    problems = [p for name in frames for p in check_dtypes(name, directory)]
    if problems:
        raise ValueError("Dtype problems after writing:\n" + "\n".join(problems))
