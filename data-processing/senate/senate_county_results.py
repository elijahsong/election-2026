"""County-level U.S. Senate results, 2006-2024, into data/processed/senate/. Run after senate_targets.py.

  county_results.csv       year, state, class, county_id, county_fips: the Election Day (November) round of every
                           race whose county results exist
  county_results_supp.csv  the same, plus contest, stage and election_date, for later rounds (runoffs)

Sources:
  48 contiguous states: Heideman v1.0.0 (data/raw/heideman/v1.0.0), regular elections only. Each county's candidate
    rows are summed into dem/gop/3p_votes; total_votes is their sum (Heideman's votes exclude blank and over/under
    votes, so source_totalvotes and nonvote_votes are blank). county_id is the county FIPS.
  Louisiana: Louisiana SOS official parish results (data/raw/la_sos_parish) for the rounds Heideman lacks, 2014's
    November round and 2016's runoff. ParishValue n is the n-th parish alphabetically, FIPS 22000 + 2n - 1 (checked
    against Heideman's 2016 and 2020 November parishes). Each candidate's party is the "(DEM)" suffix of its name.
  Hand-entered county tables (data/raw/senate_manual, MANUAL_ROUNDS): one row per county, the first column the
    county's name (matched to Heideman's counties of the state), with dem_votes, gop_votes, the listed third-party
    columns (3p_votes is their sum) and, optionally, total_votes, which must equal the candidates' sum (total_votes
    is that sum). A race may be split into several files (one per candidate, WV 2010), joined on county. Other
    columns (shares, margins, undervotes) are not read. The November rounds of GA 2008 and GA 2022 and the MS 2008,
    DE 2010, NY 2010, WV 2010, SC 2014, OK 2014, AZ 2020, OK 2022 and NE 2024 specials. Where a file's third-party
    counts are incomplete (DE 2010 special), only D and R are read: 3p_votes, total_votes and the shares of the
    total are blank, the margins are D and R's, and the rows are flagged 3p_votes_unknown.
    MANUAL_VALUE_FIXES corrects documented typos in a file (NE 2024 special) without editing it.
  Official files parsed directly (PARSED_ROUNDS): the MN 2018 special from the Minnesota SOS county results file
    (data/raw/mn-2018-senate-special), and the November rounds of the MS 2018 and GA 2020 specials from precinct
    results summed by county (data/raw/ms-2018-senate-special-r1, data/raw/ga_2020_senate_special_r1).
  Alaska and Hawaii: one synthetic At-Large county-equivalent per contest (county_id and county_name "At-Large",
    county_fips 02000 / 15000: state-aggregate codes, not county FIPS) holding the actual statewide votes of
    targets_raw.csv / targets_supp.csv, flagged at_large_statewide_totals. No county distribution is invented.

Corrections to Heideman: HEIDEMAN_COUNTY_SWAPS lists counties whose votes Heideman has swapped (OK 2014: Beaver and
  Beckham); their rows trade county FIPS and name. Heideman leaves out Kansas City, MO, which Missouri tabulates apart
  from its counties; its votes (KANSAS_CITY_SENATE, from the Kansas City Election Board's official results and, for
  2024, OpenElections) are added to Jackson County, the only county the city's election board serves. The corrected
  rows aren't flagged; both corrections are documented here and in the README. HEIDEMAN_VALUE_FIXES corrects typos in
  Heideman's counts (VT 2012: three counties), and HEIDEMAN_MISSING_COUNTIES adds counties it leaves out (AR 2022
  Phillips, ID 2022 Power) from official results. Six races whose Heideman rows don't add up to the statewide result
  (NH 2016, NY 2016, SC 2016, DE 2018, NJ 2018, SC 2022) are replaced by OpenElections results summed by county
  (OPENELECTIONS_ROUNDS, data/raw/openelections_senate).

Rounds: Heideman's rows are each race's final round, so for GA 2008, GA 2022 and LA 2014 (HEIDEMAN_RUNOFFS) they are
  the runoff (Heideman labels only GA 2022's stage runoff); they go to county_results_supp.csv, and the November
  round is in county_results.csv for LA 2014 (LA SOS) and GA 2008 and GA 2022 (senate_manual).

Reconciliation: each contest's county sums are compared with its statewide result, less the votes that belong to no
  county (NON_COUNTY_VOTES: RI's federal precincts, Maine's UOCAVA ballots). Where D or R differ by more than
  RECONCILE_TOLERANCE, every county row of the contest is flagged county_sum_differs_from_statewide and the contest is
  reported; nothing is forced to match.

Runoffs without county results: the November-round rows of a race whose runoff (targets_supp.csv) has no county
  rows are flagged runoff_county_results_missing (GA 2020, both seats; MS 2018 special).

Parties: a Heideman candidate is D, R or 3P as the statewide results (senate_targets.py) count that candidate,
  matched by surname within the contest; Heideman's own party_group is used only when no statewide candidate
  matches. So King and Sanders are D (senate_targets.CAUCUS_DEMOCRATS) and Osborn, whom Heideman codes DEM, is 3P,
  as in targets.csv (flag party_reassigned where Heideman's party differs). Heideman's 2006-2014 rows name only the two major-party nominees; when a
  statewide D or R candidate with at least MIN_HIDDEN_SHARE of the vote has no Heideman row, that candidate is
  inside Heideman's "All other candidates" row, so the county's votes for that party are unknown: dem_votes or
  gop_votes, the margins and the shares are blank and the row is flagged party_votes_in_other (DE 2006's O'Donnell).

race_type, is_special_election, is_rcv and is_unofficial are the statewide contest's. Gaps (no county rows):
  the WY 2008 special (left out by choice), the GA 2020 and MS 2018 runoffs, and the counties Heideman's gaps file
  lists. Nothing is imputed or apportioned from statewide totals.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from senate_schema import AT_LARGE, COUNTY_RESULTS, COUNTY_RESULTS_SUPP, KEY, OUTPUT_DIR, read_output, write_outputs
from house_targets import name_key
import senate_targets

ROOT = Path(__file__).resolve().parents[2]
HEIDEMAN_DIR = ROOT / "data/raw/heideman/v1.0.0"
LA_PARISH_DIR = ROOT / "data/raw/la_sos_parish"
LA_SOS_DIR = ROOT / "data/raw/la_sos"
HEIDEMAN_VERSION = "Heideman v1.0.0 (doi:10.5281/zenodo.22976553)"
FIRST_YEAR, LAST_YEAR = 2006, 2024
HEIDEMAN_RUNOFFS = {(2008, "GA"), (2022, "GA"), (2014, "LA")}
# (year, state) -> two counties whose Senate votes Heideman has swapped; their rows trade county_fips and county_name.
# OK 2014: Heideman's Beaver has Beckham's votes and vice versa (Beckham's vote is 3-4x Beaver's in every other year,
# and in Heideman's 2014 House rows and the official 2014 special-election counts)
HEIDEMAN_COUNTY_SWAPS = {(2014, "OK"): ("40007", "40009")}
# Kansas City, MO, which Missouri tabulates apart from its counties and Heideman leaves out. Its election board
# serves only the part of the city inside Jackson County, so these votes are added to Jackson County. Each named
# candidate's count, from the files in data/raw/kansas_city_mo (see its SOURCE.md); the board's lumped write-ins
# aren't included.
JACKSON_COUNTY_MO = "29095"
KANSAS_CITY_SENATE = {
    2006: [("Claire McCaskill", "dem", 73556), ("Jim Talent", "gop", 21512), ("Frank Gilmour", "3p", 1363),
           ("Lydia Lewis", "3p", 548)],
    2010: [("Robin Carnahan", "dem", 62448), ("Roy Blunt", "gop", 21866), ("Jonathan Dine", "3p", 1433),
           ("Jerry Beck", "3p", 773)],
    2012: [("Claire McCaskill", "dem", 110098), ("Todd Akin", "gop", 20660), ("Jonathan Dine", "3p", 4814)],
    2016: [("Jason Kander", "dem", 100102), ("Roy Blunt", "gop", 24048), ("Jonathan Dine", "3p", 2274),
           ("Fred Ryman", "3p", 788), ("Johnathan McFarland", "3p", 1487)],
    2018: [("Claire McCaskill", "dem", 89944), ("Josh Hawley", "gop", 20739), ("Japheth Campbell", "3p", 929),
           ("Jo Crain", "3p", 586), ("Craig O'Dear", "3p", 987)],
    2022: [("Trudy Busch Valentine", "dem", 68143), ("Eric Schmitt", "gop", 16353), ("Jonathan Dine", "3p", 1134),
           ("Paul Venable", "3p", 373)],
    2024: [("Lucas Kunce", "dem", 95675), ("Josh Hawley", "gop", 24524), ("W. C. Young", "3p", 1749),
           ("Jared Young", "3p", 810), ("Nathan Kline", "3p", 1182)],
}
KANSAS_CITY_SOURCE = "plus Kansas City (Kansas City Election Board; 2024 OpenElections; data/raw/kansas_city_mo)"
# (year, state) -> {(county_fips, candidate): (Heideman's votes, corrected votes)}: typos in Heideman's counts. VT 2012:
# OpenElections' town results (data/raw/openelections_senate) have the corrected counts, which make the statewide
# result exact; each Heideman value must match before it is replaced
HEIDEMAN_VALUE_FIXES = {
    (2012, "VT"): {
        ("50005", "John Macgovern"): (4158, 3858),  # Caledonia
        ("50007", "Bernie Sanders"): (55615, 55115),  # Chittenden
        ("50023", "Bernie Sanders"): (21502, 20797),  # Washington
    },
}
# (year, state, class) -> counties Heideman leaves out: county_fips, name, D, R and third-party votes, source. Each is
# exactly the statewide result minus Heideman's other counties
HEIDEMAN_MISSING_COUNTIES = {
    (2022, "AR", 3): [("05107", "Phillips", 2281, 1929, 66,
                       "Arkansas SOS official results, 2022-11-08 (data/raw/arkansas_sos_2022)")],
    (2022, "ID", 3): [("16077", "Power", 420, 1428, 124,
                       "Idaho SOS official county canvass, 2022-11-08 (data/raw/idaho_sos_2022)")],
}
# (year, state, class) -> D, R and third-party votes in the statewide result that belong to no county: RI's "Federal
# Precinct" ballots (federal offices only, counted statewide) and Maine's UOCAVA ballots (overseas and military
# voters). From OpenElections (data/raw/openelections_senate), ME 2024 from the Maine SOS (data/raw/maine_sos_2024).
# The reconciliation compares the county sums with the statewide result minus these votes.
NON_COUNTY_VOTES = {
    (2012, "ME", 1): (1519, 587, 702 + 43 + 38 + 22),  # King (counted D), Summers, Dill + the others
    (2018, "ME", 1): (1475, 249, 698),  # King (counted D), Brakey, Ringelstein
    (2020, "ME", 2): (4068, 1089, 491),
    (2020, "RI", 2): (1147, 81, 0),
    (2024, "RI", 1): (1356, 112, 5),
    (2024, "ME", 1): (3080, 796, 2411 + 147),  # King (counted D), Kouzounas, Costello + Cherry
}
# (year, state, contest) -> LA SOS parish file date; Heideman has the other Louisiana rounds
LA_PARISH_ROUNDS = {(2014, "LA", "first_round"): "20141104", (2016, "LA", "runoff"): "20161210"}
LA_CHECK_ROUNDS = {(2016, "LA", "first_round"): "20161108", (2020, "LA", "first_round"): "20201103",
                   (2014, "LA", "runoff"): "20141206"}  # compared with Heideman, not used
LA_PARTY_BUCKETS = {"DEM": "dem", "REP": "gop"}
MANUAL_DIR = ROOT / "data/raw/senate_manual"
# (year, state, class, contest) -> (files in MANUAL_DIR, joined on county; their third-party vote columns, or None
# when the third-party counts are incomplete); see data/raw/senate_manual/SOURCE.md
MANUAL_ROUNDS = {
    (2008, "GA", 2, "first_round"): (["ga2008-senate-r1.csv"], ["lib_votes", "garcia_votes", "salomone_votes"]),
    (2008, "MS", 1, "first_round"): (["ms2008-senate.csv"], []),
    (2010, "DE", 2, "first_round"): (["de2010-senate.csv"], None),  # 3p_votes lacks a candidate in Kent, New Castle
    (2010, "NY", 1, "first_round"): (["ny2010-senate-special.csv"], ["3p_votes"]),
    (2010, "WV", 1, "first_round"): (["wv-2010-senate-dem.csv", "wv-2010-senate-gop.csv",
                                      "wv-2010-senate-3p-mountain.csv", "wv-2010-senate-3p-constitution.csv"],
                                     ["3p_mtn_votes", "3p_constitution_votes"]),
    (2014, "SC", 3, "first_round"): (["sc-2014-senate-special-revised.csv"], ["3p_votes", "write_in_votes"]),
    (2014, "OK", 3, "first_round"): (["ok-2014-senate-special.csv"], ["3p_votes"]),
    (2020, "AZ", 3, "first_round"): (["az-2020-senate-special.csv"], ["3p_votes"]),
    (2022, "GA", 3, "first_round"): (["ga-2022-senate-r1.csv"], ["3p_votes"]),
    (2022, "OK", 2, "first_round"): (["ok-2022-senate-special.csv"], ["3p_votes"]),
    (2024, "NE", 2, "first_round"): (["ne-2024-senate-special.csv"], []),
}
# (year, state, class, contest) -> {(county, column): (value in the file, corrected value)}: typos in a hand-entered
# file, corrected here so the raw file stays as saved; each file value must match before it is replaced
MANUAL_VALUE_FIXES = {
    # NE 2024 special: a count that disagrees with its own share and with the statewide total, and three
    # total_votes that aren't the county's D + R (the shares agree with D + R)
    (2024, "NE", 2, "first_round"): {
        ("Dakota", "gop_votes"): (5206, 4068),  # 67.2% R of D + R; brings statewide R to 585,103 exactly
        ("Dakota", "total_votes"): (7187, 6049),  # the file's total is D + R with the typo
        ("Cass", "dem_votes"): (4506, 4606),  # 29.9% D of D + R; brings statewide D within 1 vote
        ("Cass", "total_votes"): (15314, 15414),
        ("Saunders", "total_votes"): (4321, 13321),
        ("Banner", "total_votes"): (372, 382),
        ("Loup", "total_votes"): (432, 428),
    },
}
UNKNOWN_3P_FLAG = "3p_votes_unknown"
# County or precinct files parsed directly (not hand-entered); see each folder's SOURCE.md
MN_2018_DIR = ROOT / "data/raw/mn-2018-senate-special"
MN_FIELDS = ["state", "county_id", "precinct", "office_id", "office", "district", "candidate_order", "candidate",
             "suffix", "incumbent", "party", "precincts_reporting", "precincts", "votes", "pct", "office_votes"]
MN_PARTY_BUCKETS = {"DFL": "dem", "R": "gop"}  # the others (LMN, UA, write-ins) are 3p
OPENELECTIONS_DIR = ROOT / "data/raw/openelections_senate"
MS_2018_PRECINCTS = ROOT / "data/raw/ms-2018-senate-special-r1/20181106__ms__general__precinct.txt"
# nonpartisan ballot: the parties senate_targets.PARTY_OVERRIDES gives the statewide candidates
MS_2018_SPECIAL_PARTIES = {"CINDY HYDE-SMITH": "gop", "CHRIS MCDANIEL": "gop", "MIKE ESPY": "dem",
                           "TOBEY BERNARD BARTEE": "dem"}
GA_2020_PRECINCTS = ROOT / "data/raw/ga_2020_senate_special_r1/20201103__ga__general.txt"
GA_VOTE_COLUMNS = ["election_day_votes", "advanced_votes", "absentee_by_mail_votes", "provisional_votes"]
GA_PARTY_BUCKETS = {"Democrat": "dem", "Republican": "gop"}  # Libertarian, Green and independents are 3p
MIN_HIDDEN_SHARE = 0.005
RECONCILE_TOLERANCE = (100, 0.001)  # votes, share of the statewide total
SUM_DIFFERS_FLAG = "county_sum_differs_from_statewide"
RUNOFF_MISSING_FLAG = "runoff_county_results_missing"
VOTE_COLUMNS = ["dem_votes", "gop_votes", "3p_votes", "total_votes"]


def surname(names: pd.Series) -> pd.Series:
    """The last name in a candidate name, upper case, suffixes dropped; "LAST, FIRST" names give LAST."""
    names = names.fillna("").astype(str)
    last_first = names.str.contains(",") & ~names.str.contains(r",\s*(?:JR|SR|II|III|IV)\.?$", case=False)
    keyed = name_key(names.where(~last_first, names.str.split(",").str[0]))
    keyed = keyed.str.replace(r"\s+(JR|SR|II|III|IV)$", "", regex=True)
    return keyed.str.split().str[-1].fillna("")


def statewide_candidates() -> pd.DataFrame:
    """Every 2006-2024 statewide contest's named candidates: KEY, contest, candidate, surname, bucket, votes."""
    races = pd.read_csv(senate_targets.INCUMBENCY_PATH)
    races = races[races["year"].between(FIRST_YEAR, LAST_YEAR)]
    df = senate_targets.assign_class(senate_targets.read_source(), races)
    cands = senate_targets.candidates(df)
    cands = cands[cands["is_candidate"] & (cands["bucket"] != "nonvote")].copy()
    cands["surname"] = surname(cands["candidate"])
    return cands


def read_heideman(directory: Path = HEIDEMAN_DIR) -> pd.DataFrame:
    h = pd.read_parquet(directory / "us_county_results_long.parquet")
    h = h[(h["office"] == "senate") & h["year"].between(FIRST_YEAR, LAST_YEAR)].copy()
    h["year"] = h["year"].astype(int)
    h = h.rename(columns={"state_po": "state_code"}).drop(columns="state").rename(columns={"state_code": "state"})
    runoff = pd.Series(list(zip(h["year"], h["state"])), index=h.index).isin(HEIDEMAN_RUNOFFS)
    # Heideman labels GA 2022's rows runoff, and GA 2008's and LA 2014's general, though they are runoffs too
    if (runoff < (h["stage"] == "runoff")).any():
        raise ValueError("Heideman runoff rows outside HEIDEMAN_RUNOFFS")
    h["contest"] = np.where(runoff, "runoff", "first_round")
    for (year, state), (a, b) in HEIDEMAN_COUNTY_SWAPS.items():
        rows = (h["year"] == year) & (h["state"] == state)
        names = h[rows].drop_duplicates("county_fips").set_index("county_fips")["county_name"]
        swap = {a: b, b: a}
        h.loc[rows, "county_fips"] = h.loc[rows, "county_fips"].replace(swap)
        h.loc[rows, "county_name"] = h.loc[rows, "county_fips"].map(names)
    for (year, state), fixes in HEIDEMAN_VALUE_FIXES.items():
        for (fips, candidate), (was, corrected) in fixes.items():
            row = (h["year"] == year) & (h["state"] == state) & (h["county_fips"] == fips) & (h["candidate"] == candidate)
            if row.sum() != 1 or h.loc[row, "votes"].item() != was:
                raise ValueError(f"{year} {state} {fips} {candidate} isn't {was}, so HEIDEMAN_VALUE_FIXES doesn't apply")
            h.loc[row, "votes"] = corrected
    return h


def heideman_counties(h: pd.DataFrame, contests: pd.DataFrame, cands: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """County rows from Heideman, one per (year, state, contest, county): votes by statewide bucket, flags, source."""
    regular = contests[~contests["is_special_election"]][[*KEY, "contest"]]
    h = h.merge(regular, on=["year", "state", "contest"], how="left", validate="many_to_one")
    if h["class"].isna().any():
        raise ValueError(f"Heideman contests without a statewide contest: "
                         f"{h.loc[h['class'].isna(), ['year', 'state', 'contest']].drop_duplicates().values.tolist()}")
    h["class"] = h["class"].astype(int)
    h["surname"] = surname(h["candidate"])
    aggregate = h["candidate"].str.startswith("All other") | h["candidate"].str.startswith("Other candidates") \
        | h["candidate"].isin(["None", "Writein", "Write-ins", "Scattering"])
    contest = [*KEY, "contest"]
    match = cands[[*contest, "surname", "bucket"]].drop_duplicates([*contest, "surname"], keep=False)
    h = h.merge(match, on=[*contest, "surname"], how="left")
    # a name Heideman spells differently ("Chamblissr", "Bill Cassi") matches the one statewide candidate, not
    # matched by surname, whose votes are within max(100, 1%) of the Heideman candidate's statewide sum
    totals = h[~aggregate].groupby([*contest, "candidate"], as_index=False).agg(
        votes=("votes", "sum"), surname=("surname", "first"), bucket=("bucket", "first"))
    free = cands[~cands.set_index([*contest, "surname"]).index.isin(
        pd.MultiIndex.from_frame(h.loc[h["bucket"].notna(), [*contest, "surname"]]))]
    by_votes = totals[totals["bucket"].isna()].merge(free[[*contest, "votes", "bucket"]], on=contest,
                                                       suffixes=("", "_state"))
    by_votes = by_votes[(by_votes["votes"] - by_votes["votes_state"]).abs()
                        <= np.maximum(100, 0.01 * by_votes["votes_state"])]
    by_votes = by_votes.drop_duplicates([*contest, "candidate"], keep=False)
    vote_match = h.merge(by_votes[[*contest, "candidate", "bucket_state"]], on=[*contest, "candidate"],
                         how="left")["bucket_state"].to_numpy()
    h["bucket"] = h["bucket"].fillna(pd.Series(vote_match, index=h.index))
    h["matched"] = h["bucket"].notna() & ~aggregate
    h.loc[aggregate, "bucket"] = "3p"
    fallback = h["party_group"].map({"DEM": "dem", "REP": "gop", "OTHER": "3p"})
    unmatched = h[h["bucket"].isna()].groupby(["year", "state", "candidate", "party_group"])["votes"].sum()
    unmatched = unmatched[unmatched > 0].reset_index().values.tolist()
    reassigned = h["bucket"].notna() & ~aggregate & (h["bucket"] != fallback)
    h["bucket"] = h["bucket"].fillna(fallback)
    h["reassigned"] = reassigned

    # statewide D / R candidates whose votes Heideman puts in its "All other candidates" row: no Heideman
    # candidate matched them, by surname or by votes
    matched_names = h.loc[h["matched"] & h["surname"].notna(), [*contest, "surname"]].drop_duplicates()
    matched_votes = by_votes[[*contest, "votes_state", "bucket_state"]].rename(
        columns={"votes_state": "votes", "bucket_state": "bucket"})
    major = cands[cands["bucket"].isin(["dem", "gop"]) & cands["contest"].isin(["first_round", "runoff"])]
    total = cands.groupby([*KEY, "contest"])["votes"].transform("sum")
    major = major[(major["votes"] >= MIN_HIDDEN_SHARE * total.loc[major.index])]
    hidden = major.merge(matched_names, how="left", indicator=True).query("_merge == 'left_only'") \
        .drop(columns="_merge")
    hidden = hidden.merge(matched_votes, on=[*contest, "votes", "bucket"], how="left", indicator=True) \
        .query("_merge == 'left_only'").drop(columns="_merge")
    hidden = hidden.merge(h[[*KEY, "contest"]].drop_duplicates(), on=[*KEY, "contest"])  # contests Heideman has
    hidden_party = hidden.groupby([*KEY, "contest"])["bucket"].agg(set)

    key = [*KEY, "contest", "county_fips"]
    votes = h.pivot_table(index=key, columns="bucket", values="votes", aggfunc="sum", fill_value=0) \
        .reindex(columns=["dem", "gop", "3p"], fill_value=0)
    out = h.groupby(key).agg(
        county_name=("county_name", "first"), source=("source", lambda s: "; ".join(sorted(s.unique()))),
        heideman_flags=("quality_flag", lambda f: ";".join(sorted({x for v in f for x in str(v).split(";") if x}))),
        reassigned=("reassigned", "any"))
    out = out.join(votes.rename(columns={"dem": "dem_votes", "gop": "gop_votes", "3p": "3p_votes"})).reset_index()
    out["total_votes"] = out["dem_votes"] + out["gop_votes"] + out["3p_votes"]
    hide = out.set_index([*KEY, "contest"]).index.map(lambda k: hidden_party.get(k, set()))
    out["hidden"] = list(hide)
    for column, bucket in [("dem_votes", "dem"), ("gop_votes", "gop")]:
        out[column] = out[column].astype("Int64").mask(out["hidden"].map(lambda s, b=bucket: b in s))
    out["3p_votes"] = out["3p_votes"].astype("Int64").mask(out["hidden"].map(bool))
    flags = [out["heideman_flags"].replace("", None),
             pd.Series(np.where(out["reassigned"], "party_reassigned", None), index=out.index),
             pd.Series(np.where(out["hidden"].map(bool), "party_votes_in_other", None), index=out.index)]
    out["quality_flags"] = pd.concat(flags, axis=1).apply(lambda r: ";".join(r.dropna()) or None, axis=1)
    out["source_version"] = HEIDEMAN_VERSION
    return out, unmatched


def missing_counties() -> pd.DataFrame:
    """HEIDEMAN_MISSING_COUNTIES as county rows."""
    rows = [(year, state, class_, "first_round", fips, name, dem, gop, third, dem + gop + third, source)
            for (year, state, class_), counties in HEIDEMAN_MISSING_COUNTIES.items()
            for fips, name, dem, gop, third, source in counties]
    out = pd.DataFrame(rows, columns=[*KEY, "contest", "county_fips", "county_name", *VOTE_COLUMNS, "source"])
    for c in VOTE_COLUMNS:
        out[c] = out[c].astype("Int64")
    return out.assign(quality_flags=None, source_version=None)


def add_kansas_city(heideman: pd.DataFrame) -> pd.DataFrame:
    """Heideman's county rows with KANSAS_CITY_SENATE's votes added to Jackson County, MO."""
    out = heideman.copy()
    for year, candidates in KANSAS_CITY_SENATE.items():
        row = (out["year"] == year) & (out["state"] == "MO") & (out["county_fips"] == JACKSON_COUNTY_MO)
        if row.sum() != 1:
            raise ValueError(f"MO {year}: expected one Heideman Jackson County row, found {row.sum()}")
        for bucket in ["dem", "gop", "3p"]:
            out.loc[row, f"{bucket}_votes"] += sum(votes for _, b, votes in candidates if b == bucket)
        out.loc[row, "total_votes"] += sum(votes for _, _, votes in candidates)
        out.loc[row, "source"] = out.loc[row, "source"] + "; " + KANSAS_CITY_SOURCE
    return out


def read_la_parishes(date: str) -> pd.DataFrame:
    """Louisiana SOS parish results of the U.S. Senate race on `date`: county_fips, bucket, votes per candidate."""
    races = json.loads((LA_SOS_DIR / f"{date}_RacesCandidates_Multiparish.json").read_text())["Races"]["Race"]
    [race] = [r for r in races if r["SpecificTitle"] == "U. S. Senator"]
    party = {c["ID"]: c["Desc"].rsplit("(", 1)[1].rstrip(")") for c in race["Choice"]}
    [path] = LA_PARISH_DIR.glob(f"{date}_VotesRaceByParish_{race['ID']}.json")
    rows = []
    for parish in json.loads(path.read_text())["Parishes"]["Parish"]:
        fips = f"22{2 * int(parish['ParishValue']) - 1:03d}"
        rows += [(fips, LA_PARTY_BUCKETS.get(party[c["ID"]], "3p"), int(c["VoteTotal"])) for c in parish["Choice"]]
    return pd.DataFrame(rows, columns=["county_fips", "bucket", "votes"])


def la_counties(names: pd.Series) -> pd.DataFrame:
    """LA_PARISH_ROUNDS county rows, with parish names from Heideman (`names`: county_fips -> name)."""
    frames = []
    for (year, state, contest), date in LA_PARISH_ROUNDS.items():
        votes = read_la_parishes(date).pivot_table(index="county_fips", columns="bucket", values="votes",
                                                   aggfunc="sum", fill_value=0) \
            .reindex(columns=["dem", "gop", "3p"], fill_value=0)
        votes = votes.rename(columns={"dem": "dem_votes", "gop": "gop_votes", "3p": "3p_votes"}).reset_index()
        frames.append(votes.assign(year=year, state=state, contest=contest,
                                   source=f"Louisiana SOS official parish results ({date[:4]}-{date[4:6]}-{date[6:]})",
                                   quality_flags=None, source_version=None))
    out = pd.concat(frames, ignore_index=True)
    out["county_name"] = out["county_fips"].map(names)
    for c in ["dem_votes", "gop_votes", "3p_votes"]:
        out[c] = out[c].astype("Int64")
    out["total_votes"] = out["dem_votes"] + out["gop_votes"] + out["3p_votes"]
    return out


def check_la(h: pd.DataFrame) -> list[str]:
    """LA_CHECK_ROUNDS: LA SOS parish D/R/total votes against Heideman's (they should agree parish by parish)."""
    lines = []
    for (year, state, contest), date in LA_CHECK_ROUNDS.items():
        sos = read_la_parishes(date).pivot_table(index="county_fips", columns="bucket", values="votes", aggfunc="sum",
                                                 fill_value=0).reindex(columns=["dem", "gop", "3p"], fill_value=0)
        rows = h[(h["year"] == year) & (h["state"] == state) & (h["contest"] == contest)]
        heid = rows.set_index("county_fips")[["dem_votes", "gop_votes", "3p_votes"]].astype(float)
        heid.columns = ["dem", "gop", "3p"]
        diff = (sos - heid).abs()
        lines.append(f"LA SOS vs Heideman {year} {contest}: {len(sos)} vs {len(heid)} parishes, "
                     f"max |D| {diff['dem'].max():.0f}, max |R| {diff['gop'].max():.0f}, "
                     f"max |3P| {diff['3p'].max():.0f}")
    return lines


def read_manual(names: list[str], columns: list[str]) -> pd.DataFrame:
    """The hand-entered files `names` joined on county (each file's first column), keeping `columns`. A statewide
    "Total" row is dropped. A column in several files (total_votes) must agree across them, and every county must be
    in every file."""
    merged = None
    for name in names:
        df = pd.read_csv(MANUAL_DIR / name, thousands=",")
        df = df.rename(columns={df.columns[0]: "county"})
        df = df[df["county"].str.strip().str.upper() != "TOTAL"]
        df = df[["county", *[c for c in columns if c in df]]]
        if merged is None:
            merged = df
            continue
        merged = merged.merge(df, on="county", how="outer", suffixes=("", "_again"), validate="one_to_one")
        for again in [c for c in merged if c.endswith("_again")]:
            if not merged[again].equals(merged[again.removesuffix("_again")]):
                raise ValueError(f"{names}: {again.removesuffix('_again')} differs between the files")
            merged = merged.drop(columns=again)
    if merged.isna().any().any():
        raise ValueError(f"{names}: counties missing from some of the files")
    return merged


def county_key(names: pd.Series) -> pd.Series:
    """Upper-case letters only, " County" dropped: "Beaufort County", "BEAUFORT" and "Beaufort" match."""
    return names.str.upper().str.removesuffix(" COUNTY").str.replace(r"[^A-Z]", "", regex=True)


def with_fips(df: pd.DataFrame, h: pd.DataFrame, state: str, label: str) -> pd.DataFrame:
    """`df` (a `county` name column) with the county_fips and county_name of Heideman's counties of `state`."""
    counties = h[h["state"] == state].drop_duplicates("county_fips")
    df = df.merge(counties[["county_fips", "county_name"]].assign(key=county_key(counties["county_name"])),
                  left_on=county_key(df["county"]), right_on="key", how="left", validate="one_to_one")
    if df["county_fips"].isna().any():
        raise ValueError(f"{label}: counties without a Heideman FIPS: "
                         f"{df.loc[df['county_fips'].isna(), 'county'].tolist()}")
    return df


def bucket_votes(df: pd.DataFrame) -> pd.DataFrame:
    """Per county: dem_votes, gop_votes, 3p_votes and their sum, total_votes, from rows of county, bucket, votes."""
    out = df.pivot_table(index="county", columns="bucket", values="votes", aggfunc="sum", fill_value=0) \
        .reindex(columns=["dem", "gop", "3p"], fill_value=0)
    out = out.rename(columns={"dem": "dem_votes", "gop": "gop_votes", "3p": "3p_votes"}).reset_index()
    out["total_votes"] = out["dem_votes"] + out["gop_votes"] + out["3p_votes"]
    return out


def read_mn_2018_special() -> pd.DataFrame:
    """Minnesota SOS county results, office 0103 (U.S. Senator Special Election), by MN_PARTY_BUCKETS. The fields are
    listed in overall_Index.txt and the county IDs named in county_index.txt; each county's office total must equal
    its candidates' sum."""
    df = pd.read_csv(MN_2018_DIR / "mn_2018_county_results_all.txt", sep=";", header=None, names=MN_FIELDS, dtype=str)
    df = df[df["office_id"] == "0103"].copy()
    index = pd.read_csv(MN_2018_DIR / "county_index.txt", sep=";", header=None,
                        names=["county_id", "county", "precincts"], dtype=str)
    df["county"] = df["county_id"].map(index.set_index("county_id")["county"])
    df["bucket"] = df["party"].map(MN_PARTY_BUCKETS).fillna("3p")
    df["votes"] = df["votes"].astype(int)
    out = bucket_votes(df)
    office_total = df.groupby("county")["office_votes"].first().astype(int)
    if (out.set_index("county")["total_votes"] != office_total).any():
        raise ValueError("MN 2018 special: a county's office total isn't its candidates' sum")
    return out


def read_ms_2018_special() -> pd.DataFrame:
    """Mississippi precinct results of the U.S. Senate special (office U.S. Senate, district Unexpired Term), summed
    by county; parties from MS_2018_SPECIAL_PARTIES (the ballot was nonpartisan)."""
    df = pd.read_csv(MS_2018_PRECINCTS, dtype=str)
    df = df[(df["office"] == "U.S. Senate") & (df["district"] == "Unexpired Term")].copy()
    df["bucket"] = df["candidate"].str.upper().map(MS_2018_SPECIAL_PARTIES)
    if df["bucket"].isna().any():
        raise ValueError(f"MS 2018 special: candidates without a party: "
                         f"{df.loc[df['bucket'].isna(), 'candidate'].unique()}")
    df["votes"] = df["votes"].astype(int)
    return bucket_votes(df)


def read_ga_2020_special() -> pd.DataFrame:
    """Georgia precinct results of office "U.S. Senate (Special)" (the November 3 jungle round), the four vote-mode
    columns summed, then precincts summed by county; party buckets from GA_PARTY_BUCKETS, as the statewide target's."""
    df = pd.read_csv(GA_2020_PRECINCTS, dtype=str)
    df = df[df["office"] == "U.S. Senate (Special)"].copy()
    df["votes"] = df[GA_VOTE_COLUMNS].astype(int).sum(axis=1)
    df["bucket"] = df["party"].map(GA_PARTY_BUCKETS).fillna("3p")
    return bucket_votes(df)


def read_openelections(name: str, office: str, buckets: dict[str, str]) -> pd.DataFrame:
    """An OpenElections results file (county, town or precinct rows) summed by county. Rows for `office` only; total
    rows (a county, town or precinct named "Total", "TOTALS", "Statewide Total", ...), and blank, void and total "candidates", are dropped. Each
    candidate's ballot lines are summed: `buckets` maps candidate names to dem / gop, everyone else is 3p."""
    df = pd.read_csv(OPENELECTIONS_DIR / name, dtype=str)
    df = df[df["office"] == office].copy()
    places = [c for c in ["county", "town", "precinct", "election_district"] if c in df]
    total_row = df[places].apply(lambda c: c.str.upper().str.contains(r"\bTOTALS?\b").fillna(False)).any(axis=1)
    df = df[~total_row & ~df["candidate"].isin(["Blank", "Void", "Total"])].copy()
    df = df[df["county"] != "UOCAVA"].copy()  # Maine's overseas and military ballots: in no county (NON_COUNTY_VOTES)
    if name == "20161108__nh__general__town.csv":
        # the Belknap rows include a state summary: a heading row ("Belknap County", candidates' names as votes),
        # one row per county named after it (Belknap has no town named after a county), and a blank note row
        summary = (df["county"] == "Belknap") & df["town"].isin(
            [*df["county"].unique(), "Belknap County", "correction received from clerk"])
        df = df[~summary].copy()
    df["votes"] = df["votes"].str.replace(",", "").fillna("0").astype(int)  # NH leaves a zero count blank
    df["bucket"] = df["candidate"].map(buckets).fillna("3p")
    return bucket_votes(df)


# (year, state, class, contest) -> OpenElections file in OPENELECTIONS_DIR, office, the D and R candidates' names
# (each name as it appears in the file). These replace Heideman's rows for the race, which are incomplete: ME 2012
# (no votes for Dill, the Democrat, who is third party here since King is counted as the Democrat), NH 2016
# (towns missing from every county), NY 2016 (part of Long's vote under "Wendy Ross", in 3P), SC 2016 (only the
# DEM line of fusion candidate Dixon, and only one line of Bledsoe), DE 2018 (New Castle has 11 votes), NJ 2018
# (Middlesex short), SC 2022 (most counties slightly over).
OPENELECTIONS_ROUNDS = {
    (2012, "ME", 1, "first_round"): ("20121106__me__general__town.csv", "U.S. Senate",
                                     {"Angus King": "dem", "Charles Summers": "gop"}),
    (2016, "NH", 3, "first_round"): ("20161108__nh__general__town.csv", "USS",
                                     {"Maggie Hassan": "dem", "Hassan": "dem", "Kelly Ayotte": "gop", "Ayotte": "gop"}),
    (2016, "NY", 3, "first_round"): ("20161108__ny__general__county.csv", "U.S. Senate",
                                     {"Charles E. Schumer": "dem", "Wendy Long": "gop"}),
    (2016, "SC", 3, "first_round"): ("20161108__sc__general__precinct.csv", "U.S. Senate",
                                     {"Thomas Dixon": "dem", "Tim Scott": "gop"}),
    (2018, "DE", 1, "first_round"): ("20181106__de__general__precinct.csv", "U.S. Senate",
                                     {"Carper T": "dem", "Arlett R": "gop"}),
    (2018, "NJ", 1, "first_round"): ("20181106__nj__general__county.csv", "U.S. Senate",
                                     {"ROBERT MENENDEZ (w)": "dem", "BOB HUGIN": "gop"}),
    (2022, "SC", 3, "first_round"): ("20221108__sc__general__precinct.csv", "U.S. Senate",
                                     {"DEM Krystle Matthews": "dem", "Krystle Matthews": "dem",
                                      "REP Tim Scott": "gop", "Tim Scott": "gop"}),
}


# (year, state, class, contest) -> (reader of an official file: county, dem/gop/3p/total_votes; source)
PARSED_ROUNDS = {
    (2018, "MN", 2, "first_round"): (read_mn_2018_special, "Minnesota SOS county results, 2018-11-06 "
                                                           "(data/raw/mn-2018-senate-special)"),
    (2018, "MS", 2, "first_round"): (read_ms_2018_special, "Mississippi precinct results summed by county, 2018-11-06 "
                                                           "(data/raw/ms-2018-senate-special-r1)"),
    (2020, "GA", 3, "first_round"): (read_ga_2020_special, "Georgia precinct results summed by county, 2020-11-03 "
                                                           "(OpenElections, data/raw/ga_2020_senate_special_r1)"),
}


def manual_counties(h: pd.DataFrame) -> pd.DataFrame:
    """MANUAL_ROUNDS, PARSED_ROUNDS and OPENELECTIONS_ROUNDS county rows; county names are matched to Heideman's
    counties of the state."""
    frames = []
    for (year, state, class_, contest), (names, third_party) in MANUAL_ROUNDS.items():
        name = ", ".join(names)
        df = read_manual(names, ["dem_votes", "gop_votes", *(third_party or []), "total_votes"]).assign(file=name)
        for (county, column), (was, corrected) in MANUAL_VALUE_FIXES.get((year, state, class_, contest), {}).items():
            row = county_key(df["county"]) == county_key(pd.Series([county])).item()
            if row.sum() != 1 or df.loc[row, column].item() != was:
                raise ValueError(f"{name}: {county} {column} isn't {was}, so MANUAL_VALUE_FIXES doesn't apply")
            df.loc[row, column] = corrected
        if third_party is None:
            df["3p_votes"] = df["total_votes"] = pd.NA
        else:
            df["3p_votes"] = df[third_party].sum(axis=1)
            candidates = df["dem_votes"] + df["gop_votes"] + df["3p_votes"]
            if "total_votes" in df and (candidates != df["total_votes"]).any():
                raise ValueError(f"{name}: total_votes isn't the sum of the candidates' votes")
            df["total_votes"] = candidates
        df["source"] = "Hand-entered county results (data/raw/senate_manual: " + df["file"] + ")"
        df["quality_flags"] = UNKNOWN_3P_FLAG if third_party is None else None
        frames.append(with_fips(df, h, state, name).assign(year=year, state=state, contest=contest,
                                                           **{"class": class_}))
    for (year, state, class_, contest), (read, source) in PARSED_ROUNDS.items():
        df = read().assign(source=source, quality_flags=None)
        frames.append(with_fips(df, h, state, source).assign(year=year, state=state, contest=contest,
                                                             **{"class": class_}))
    for (year, state, class_, contest), (name, office, buckets) in OPENELECTIONS_ROUNDS.items():
        source = f"OpenElections results summed by county (data/raw/openelections_senate/{name})"
        df = read_openelections(name, office, buckets).assign(source=source, quality_flags=None)
        frames.append(with_fips(df, h, state, source).assign(year=year, state=state, contest=contest,
                                                             **{"class": class_}))
    out = pd.concat(frames, ignore_index=True)
    for c in VOTE_COLUMNS:
        out[c] = out[c].astype("Int64")
    return out[["year", "state", "class", "contest", "county_fips", "county_name", *VOTE_COLUMNS, "source",
                "quality_flags"]].assign(source_version=None)


def at_large(contests: pd.DataFrame) -> pd.DataFrame:
    """One At-Large row per AK / HI contest, with the statewide votes and source."""
    rows = contests[contests["state"].isin(AT_LARGE)].copy()
    rows["county_fips"] = rows["state"].map(AT_LARGE)
    rows["county_name"] = "At-Large"
    rows["quality_flags"] = "at_large_statewide_totals"
    rows["source_version"] = None
    return rows[[*KEY, "contest", "county_fips", "county_name", *VOTE_COLUMNS, "source_totalvotes", "nonvote_votes",
                 "source", "quality_flags", "source_version"]]


def finish(df: pd.DataFrame, contests: pd.DataFrame) -> pd.DataFrame:
    """Shares, margins and the statewide contest's descriptors, in COUNTY_RESULTS_SUPP's columns."""
    df = df.merge(contests[[*KEY, "contest", "stage", "election_date", "race_type", "is_special_election", "is_rcv",
                            "is_unofficial"]], on=[*KEY, "contest"], how="left", validate="many_to_one")
    df["county_id"] = np.where(df["state"].isin(AT_LARGE), "At-Large", df["county_fips"])
    df["is_at_large"] = df["county_id"] == "At-Large"
    for c in ["source_totalvotes", "nonvote_votes"]:
        df[c] = df[c].astype("Int64") if c in df else pd.Series(pd.NA, index=df.index, dtype="Int64")
    dem, gop = df["dem_votes"].astype("Float64"), df["gop_votes"].astype("Float64")
    total = df["total_votes"].astype("Float64").where(df["total_votes"] > 0)
    df["dem_pct"], df["gop_pct"] = 100 * dem / total, 100 * gop / total
    df["3p_pct"] = 100 * df["3p_votes"].astype("Float64") / total
    df["margin_votes"] = (df["dem_votes"] - df["gop_votes"]).astype("Int64")
    df["margin_raw"] = 100 * (dem - gop) / total
    df["margin_2p_norm"] = 100 * (dem - gop) / (dem + gop).where(dem + gop > 0)
    for c in ["dem_pct", "gop_pct", "3p_pct", "margin_raw", "margin_2p_norm"]:
        df[c] = df[c].astype("float64")
    return df[list(COUNTY_RESULTS_SUPP)].sort_values([*KEY, "contest", "county_fips"], ignore_index=True)


def reconcile(counties: pd.DataFrame, contests: pd.DataFrame) -> pd.DataFrame:
    """Per contest with county rows: county sums minus the statewide result less its NON_COUNTY_VOTES, and whether D
    or R differ beyond RECONCILE_TOLERANCE (blank D / R sums are compared as missing)."""
    sums = counties.groupby([*KEY, "contest"]).agg(
        counties=("county_fips", "size"), **{c: (c, lambda v: v.sum() if v.notna().all() else np.nan)
                                              for c in VOTE_COLUMNS})
    state = contests.set_index([*KEY, "contest"])[VOTE_COLUMNS].astype(float)
    for (year, st, class_), (dem, gop, third) in NON_COUNTY_VOTES.items():
        state.loc[(year, st, class_, "first_round")] -= [dem, gop, third, dem + gop + third]
    diff = (sums[VOTE_COLUMNS].astype(float) - state.reindex(sums.index)).add_suffix("_diff")
    out = sums.join(diff).reset_index()
    tolerance = np.maximum(RECONCILE_TOLERANCE[0], RECONCILE_TOLERANCE[1] * state.reindex(sums.index)[
        "total_votes"].to_numpy())
    out["differs"] = (out[["dem_votes_diff", "gop_votes_diff"]].abs().max(axis=1) > tolerance).to_numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    raw = read_output("targets_raw.csv", args.output_dir).assign(contest="first_round", stage="general")
    raw["election_date"] = raw["year"].astype(str).map(
        lambda y: json.loads(senate_targets.ELECTION_DATES_PATH.read_text())[y]["election_day"])
    supp = read_output("targets_supp.csv", args.output_dir)
    contests = pd.concat([raw[supp.columns.intersection(raw.columns)],
                          supp[supp["contest"] != "first_round"]], ignore_index=True)
    for c in VOTE_COLUMNS + ["source_totalvotes", "nonvote_votes"]:
        contests[c] = contests[c].astype("Int64")

    cands = statewide_candidates()
    h = read_heideman()
    heideman, unmatched = heideman_counties(h, contests, cands)
    heideman = add_kansas_city(heideman)
    replaced = pd.MultiIndex.from_frame(heideman[[*KEY, "contest"]]).isin(list(OPENELECTIONS_ROUNDS))
    if replaced.any() != bool(OPENELECTIONS_ROUNDS) or \
            heideman.loc[replaced, [*KEY, "contest"]].drop_duplicates().shape[0] != len(OPENELECTIONS_ROUNDS):
        raise ValueError("an OPENELECTIONS_ROUNDS race has no Heideman rows to replace")
    heideman = heideman[~replaced]
    names = heideman.drop_duplicates("county_fips").set_index("county_fips")["county_name"]
    la = la_counties(names)
    regular = contests.loc[~contests["is_special_election"], [*KEY, "contest"]]
    la = la.merge(regular, on=["year", "state", "contest"], validate="many_to_one")
    manual = manual_counties(h)
    counties = pd.concat([heideman, missing_counties(), la, manual, at_large(contests)], ignore_index=True)
    out = finish(counties, contests)

    check = reconcile(out, contests)
    differs = pd.MultiIndex.from_frame(out[[*KEY, "contest"]]).isin(
        pd.MultiIndex.from_frame(check.loc[check["differs"], [*KEY, "contest"]]))
    out["quality_flags"] = out["quality_flags"].where(~differs, out["quality_flags"].fillna("").map(
        lambda f: f"{f};{SUM_DIFFERS_FLAG}" if f else SUM_DIFFERS_FLAG))
    runoffs = pd.MultiIndex.from_frame(contests.loc[contests["contest"] == "runoff", KEY])
    covered_runoffs = pd.MultiIndex.from_frame(out.loc[out["contest"] == "runoff", KEY])
    missing_runoffs = runoffs.difference(covered_runoffs)
    missing = (out["contest"] == "first_round").to_numpy() & pd.MultiIndex.from_frame(out[KEY]).isin(missing_runoffs)
    out["quality_flags"] = out["quality_flags"].where(~missing, out["quality_flags"].fillna("").map(
        lambda f: f"{f};{RUNOFF_MISSING_FLAG}" if f else RUNOFF_MISSING_FLAG))
    main_rows = out[out["contest"] == "first_round"][list(COUNTY_RESULTS)].reset_index(drop=True)
    supp_rows = out[out["contest"] != "first_round"].reset_index(drop=True)

    covered = main_rows[KEY].drop_duplicates()
    gaps = raw[KEY].merge(covered, how="left", indicator=True).query("_merge == 'left_only'")[KEY]
    gaps = gaps.merge(raw[[*KEY, "is_special_election"]], on=KEY)
    print(f"county_results: {len(main_rows)} rows, {len(covered)} of {len(raw)} races; per year "
          f"{main_rows.drop_duplicates(KEY).groupby('year').size().to_dict()}")
    print(f"county_results_supp: {len(supp_rows)} rows, contests "
          f"{supp_rows[[*KEY, 'contest']].drop_duplicates().values.tolist()}")
    print(f"Races without county results ({len(gaps)}): "
          + ", ".join(f"{y} {s} {c}{' special' if sp else ''}" for y, s, c, sp in gaps.itertuples(index=False)))
    print(f"Runoffs without county results (their November rows flagged {RUNOFF_MISSING_FLAG}): "
          f"{missing_runoffs.tolist()}")
    print(f"Heideman candidates with no statewide match (party_group used): {unmatched}")
    print("Rows flagged party_reassigned: "
          f"{int(out['quality_flags'].fillna('').str.contains('party_reassigned').sum())}; party_votes_in_other: "
          + str(out.loc[out["quality_flags"].fillna("").str.contains("party_votes_in_other"), [*KEY, "contest"]]
                .drop_duplicates().values.tolist()))
    for line in check_la(heideman):
        print(line)
    at_large_ok = check[check["state"].isin(AT_LARGE)][[f"{c}_diff" for c in VOTE_COLUMNS]].abs().sum().sum() == 0
    print(f"AK/HI At-Large rows equal the statewide totals: {at_large_ok}")
    print(f"Reconciliation with statewide results: {len(check)} contests, {int(check['differs'].sum())} differ "
          f"beyond max(100 votes, 0.1%) on D or R:")
    print(check[check["differs"]].to_string(index=False))
    if not at_large_ok:
        raise ValueError("AK/HI At-Large rows don't equal the statewide totals")
    write_outputs({"county_results.csv": main_rows, "county_results_supp.csv": supp_rows}, args.output_dir)


if __name__ == "__main__":
    main()
