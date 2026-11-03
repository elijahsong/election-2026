"""Build statewide U.S. Senate general-election results, 2006-2024, in data/processed/senate, following the House
targets (data-processing/house/house_targets.py and its two imputation scripts).

  targets_raw.csv    one row per year, state, class (every race in data/interim/senate_incumbency.csv), the Election
                     Day (November) round exactly as observed
  targets.csv        the same rows and columns, with the House target rules applied: jungle and RCV targets set from
                     their observed first-round party totals, the TREATED_UNCONTESTED races treated as uncontested, and
                     every uncontested race's target imputed (house_impute_targets_uncontested.impute_uncontested)
  targets_supp.csv   one row per year, state, class, contest for every key with more than one contest (a runoff, or a
                     same-day special for the seat's last weeks), every jungle or RCV first round, and the actual result
                     of every race whose targets.csv target is imputed but had an opponent; with stage and election_date
  targets_candidates.csv  each candidate (and aggregate write-in / non-vote row) of every targets_supp.csv contest

Source: MEDSL U.S. Senate statewide 1976-2024 (data/raw/medsl_senate; all 50 states, specials and runoffs), with the
  official corrections below. Heideman's county files aren't used here: their 2006-2014 Senate rows name only the two
  major-party nominees, and three hold a runoff in place of the November round (see senate_county_results.py, which
  reconciles them with these totals).

Contests: a key (year, state, class) has its November round (contest first_round, stage general), and possibly a
  runoff (contest runoff, stage runoff; the original cycle year even when held the next January) or a same-day special
  for the seat's last weeks (contest special, stage general; CA 2022 and 2024, which repeat the full-term race). The
  November round is the one the main tables use; winner_party is the runoff's winner where there is one, and
  election_day_leader_lost flags a November leader who lost it. Class comes from senate_incumbency.csv, matched on
  (year, state, special), and every one of its 2006-2024 races must have exactly one November round.

Official corrections (each entry must apply; flag official_results_correction where counts change):
  ROUND_FIXES     MEDSL's GA 2008 November round is labelled "pre" and its runoff "gen"; its MS 2018 special has
                  only the November 27 runoff, labelled "gen"
  SPECIAL_FIXES   MEDSL's one IL 2010 contest is marked special but holds the full-term race (Kirk 1,778,698, as in
                  FEC's full-term results); the same-day special for the seat's last weeks isn't in MEDSL
  LA_OFFICIAL     LA 2014, 2016 and 2020 come from Louisiana's official results (data/raw/la_sos): MEDSL gives
                  2014 and 2016's top candidates their December runoff totals (plus 1,000,000 in 2014). The runoffs
                  are added from the same files
  FEC_CONTESTS    taken from FEC's "Federal Elections" (official results, data/raw/fec_federal_elections): CT 2022
                  (missing from MEDSL), the MS 2018 special's November round, and nine contests whose MEDSL counts
                  are partial, lack candidates, or aren't the certified count (each entry gives the reason). FEC
                  lists no blank or void ballots, so these contests' nonvote_votes is 0
  PARTY_OVERRIDES party labels for nonpartisan ballots and party-less source rows: MS 2008 and 2018 specials, AK 2022,
                  WY 2020. Where MEDSL's party_detailed is blank but party_simplified is DEMOCRAT or REPUBLICAN (MS and
                  NV 2024), party_simplified is used

Votes, race_type, targets and flags follow house_targets.py exactly (DEM_LABELS / GOP_LABELS plus three fused or
  short labels, NONVOTE_NAMES, AGGREGATE_NAMES plus Nevada's "None of these candidates", race_types, DR_TYPES):
  dem/gop/3p_votes are the named candidates' line-summed votes, total_votes their sum (non-votes left out, in
  nonvote_votes), and margins are Democratic-positive, 0-100. Unlike the House files, the two independents who
  caucus with Democrats, King (ME 2012, 2018, 2024) and Sanders (VT 2006, 2012, 2018, 2024), are counted as the
  Democrat, and those races' Democratic nominees (ME's Dill, Ringelstein, Costello) as third party (CAUCUS_DEMOCRATS,
  flag independent_counted_as_dem). The other independents (Orman, McMullin, Osborn) are third party.
  Jungle (dr_jungle): Louisiana in LA_JUNGLE_YEARS, and the nonpartisan specials JUNGLE_SPECIALS. RCV (dr_rcv):
  ME from 2018 and AK from 2022 (RCV_FROM). MEDSL's AK 2022 counts are the RCV report's round 1 (Murkowski 114,118),
  which counts a ballot for its first continuing ranking; FEC's first-choice summary differs by about 600 votes per
  candidate (flag rcv_round1_tabulation).

FEC check (report only, flag differs_from_fec): every 2006-2022 contest's D, R and total votes are compared with FEC's
  official results; a difference above max(100 votes, 0.1% of the total) is flagged. FEC has no 2024 results yet.
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from senate_schema import KEY, OUTPUT_DIR, SUPP_KEY, TARGETS, TARGETS_CANDIDATES, TARGETS_SUPP, read_output, \
    write_outputs
from house_targets import (AGGREGATE_NAMES, DEM_LABELS, DR_TYPES, GOP_LABELS, IMPUTE_RACE_TYPES, LA_JUNGLE_YEARS,
                           NONVOTE_NAMES, RCV_FROM, STANDARD_RACE_TYPES, name_key, race_types)
from house_impute_targets_jungle_rcv import impute_jungle_rcv
from house_impute_targets_uncontested import impute_uncontested, read_shave

ROOT = Path(__file__).resolve().parents[2]
MEDSL_PATH = ROOT / "data/raw/medsl_senate/1976-2024-senate-state.csv"
INCUMBENCY_PATH = ROOT / "data/interim/senate_incumbency.csv"
LA_SOS_DIR = ROOT / "data/raw/la_sos"
FEC_DIR = ROOT / "data/raw/fec_federal_elections"
ELECTION_DATES_PATH = ROOT / "data/reference/election_dates.json"
FIRST_YEAR, LAST_YEAR = 2006, 2024

SENATE_DEM_LABELS = DEM_LABELS | {"DEMOCRATIC/WORKING FAMILIES"}  # NY 2024's fused line
SENATE_GOP_LABELS = GOP_LABELS | {"REPUBLICAN/CONSERVATIVE", "GOP"}  # NY 2024; WA 2024's top-two label
SENATE_AGGREGATE_NAMES = AGGREGATE_NAMES | {"NONE OF THESE CANDIDATES", "NONE OF THE ABOVE", "OTHER CANDIDATES"}
MEDSL_SOURCE = "MEDSL U.S. Senate statewide 1976-2024 (doi:10.7910/DVN/PEJ5QU, version {})"
FEC_SOURCE = "FEC Federal Elections {} (official results)"
LA_SOURCE = "Louisiana SOS official results ({})"

# (year, state, special) -> MEDSL stage -> contest
ROUND_FIXES = {
    (2008, "GA", False): {"pre": "first_round", "gen": "runoff"},
    (2018, "MS", True): {"gen": "runoff"},  # the November round comes from FEC_CONTESTS
}
SPECIAL_FIXES = {(2010, "IL"): False}
# same-day specials for the seat's last weeks; contest special under the full-term race's key
UNEXPIRED_TERM_SPECIALS = {(2022, "CA"), (2024, "CA")}
# (year, state) -> (November, runoff) dates of Louisiana's official results files; None: no runoff
LA_OFFICIAL = {(2014, "LA"): ("20141104", "20141206"), (2016, "LA"): ("20161108", "20161210"),
               (2020, "LA"): ("20201103", None)}
LA_PARTIES = {"DEM": "DEMOCRAT", "REP": "REPUBLICAN", "LBT": "LIBERTARIAN", "IND": "INDEPENDENT", "GRN": "GREEN"}
# (year, state, special, contest) -> why the contest is taken from FEC's official results
FEC_CONTESTS = {
    (2022, "CT", False, "first_round"): "missing from MEDSL",
    (2018, "MS", True, "first_round"): "MEDSL has only the runoff",
    (2006, "WV", False, "first_round"): "MEDSL's counts are partial (Byrd 159,154 of 296,276)",
    (2010, "UT", False, "first_round"): "MEDSL's counts are partial (Lee 360,403 of 390,179)",
    (2022, "MO", False, "first_round"): "MEDSL lacks the Libertarian and Constitution candidates (49,429 votes)",
    (2006, "DE", False, "first_round"): "MEDSL lacks O'Donnell's write-in votes (11,127)",
    # MEDSL's counts are pre-certification: every candidate a little below (or, ME, above) the certified count
    (2006, "ME", False, "first_round"): "MEDSL isn't the certified count (Snowe 405,596, certified 402,598)",
    (2010, "CO", False, "first_round"): "MEDSL isn't the certified count (Bennet 851,590, certified 854,685)",
    (2012, "NY", False, "first_round"): "MEDSL isn't the certified count (Gillibrand 4,808,878, certified 4,822,330)",
    (2018, "PA", False, "first_round"): "MEDSL isn't the certified count (Casey 2,777,680, certified 2,792,437)",
    (2018, "WV", False, "first_round"): "MEDSL isn't the certified count (Manchin 288,808, certified 290,510)",
}
FEC_PARTIES = {"D": "DEMOCRAT", "DEM": "DEMOCRAT", "R": "REPUBLICAN", "REP": "REPUBLICAN", "WF": "WORKING FAMILIES",
               "LIB": "LIBERTARIAN", "LBT": "LIBERTARIAN", "CON": "CONSTITUTION", "MTP": "MOUNTAIN", "REF": "REFORM",
               "N": "NONPARTISAN"}
# (year, state, special, candidate as the source spells it) -> party
PARTY_OVERRIDES = {
    (2008, "MS", True, "RONNIE MUSGROVE"): "DEMOCRAT",  # nonpartisan special ballot
    (2018, "MS", True, "CINDY HYDE-SMITH"): "REPUBLICAN",  # nonpartisan special ballot (FEC party N)
    (2018, "MS", True, "MIKE ESPY"): "DEMOCRAT",
    (2018, "MS", True, "CHRIS MCDANIEL"): "REPUBLICAN",
    (2018, "MS", True, "TOBEY BERNARD BARTEE"): "DEMOCRAT",
    (2022, "AK", False, "MURKOWSKI, LISA"): "REPUBLICAN",  # MEDSL gives the top-four candidates no party
    (2022, "AK", False, "TSHIBAKA, KELLY C."): "REPUBLICAN",
    (2022, "AK", False, "KELLEY, BUZZ A."): "REPUBLICAN",
    (2022, "AK", False, "CHESBRO, PATRICIA R."): "DEMOCRAT",
    (2020, "WY", False, "CYNTHIA M. LUMMIS"): "REPUBLICAN",  # MEDSL gives WY 2020's candidates no party
    (2020, "WY", False, "MERAV BEN DAVID"): "DEMOCRAT",
}
# (year, state, class) -> independent who caucuses with Democrats, as the source spells the name: counted as the
# Democrat (bucket dem), and the race's Democratic nominee, if any, as third party
CAUCUS_DEMOCRATS = {
    (2012, "ME", 1): "ANGUS KING",  # Dill (D) third party
    (2018, "ME", 1): "ANGUS S. KING, JR.",  # Ringelstein (D) third party
    (2024, "ME", 1): "ANGUS S. KING JR.",  # Costello (D) third party
    (2006, "VT", 1): "BERNARD SANDERS",  # no Democratic nominee in Sanders's races
    (2012, "VT", 1): "BERNARD SANDERS",
    (2018, "VT", 1): "BERNIE SANDERS",
    (2024, "VT", 1): "BERNIE SANDERS",
}
# nonpartisan all-candidate specials with a runoff provision: (year, state)
JUNGLE_SPECIALS = {(2008, "MS"), (2018, "MS"), (2020, "GA")}
# runoff dates: (year, state, special) -> date
RUNOFF_DATES = {(2008, "GA", False): "2008-12-02", (2014, "LA", False): "2014-12-06",
                (2016, "LA", False): "2016-12-10", (2018, "MS", True): "2018-11-27",
                (2020, "GA", False): "2021-01-05", (2020, "GA", True): "2021-01-05",
                (2022, "GA", False): "2022-12-06"}
RCV_ROUND1_TABULATION = [(2022, "AK", False)]
# Races with a D and an R whose D vs R margin isn't a contested result, treated as uncontested in targets.csv (as in
# house_targets.TREATED_UNCONTESTED): a major party finished third, behind an independent who won. The race_type
# becomes <party>_vs_3p for the major party that finished second, and the target is imputed
TREATED_UNCONTESTED = {
    (2006, "CT", 1),  # Lieberman (CfL) 49.7, Lamont (D) 39.7, Schlesinger (R) 9.6
    (2010, "AK", 3),  # Murkowski (write-in) 39.5, Miller (R) 35.5, McAdams (D) 23.5
    (2010, "FL", 3),  # Rubio (R) 48.9, Crist (I) 29.7, Meek (D) 20.2
    (2016, "AK", 3),  # Murkowski (R) 44.4, Miller (L) 29.2, Stock (I) 13.2, Metcalfe (D) 11.6
}
TREATED_UNCONTESTED_FLAG = "major_party_third_treated_uncontested"
FEC_TOLERANCE = (100, 0.001)  # votes, share of total_votes


def contest_source(df: pd.DataFrame) -> pd.Series:
    return df["version"].astype(str).map(MEDSL_SOURCE.format)


def read_medsl(path: Path = MEDSL_PATH) -> pd.DataFrame:
    """MEDSL rows of 2006-2024 (and GA's January 2021 runoffs, as 2020), in the common row format: year, state,
    special, contest, candidate, party, writein, candidatevotes, totalvotes, unofficial, source."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df = df[df["year"].between(FIRST_YEAR, LAST_YEAR) | (df["year"] == 2021)].copy()
    df = df.drop(columns="state").rename(columns={"state_po": "state", "party_detailed": "party"})
    df["stage"] = df["stage"].str.lower()
    df["contest"] = df["stage"].map({"gen": "first_round", "runoff": "runoff", "gen runoff": "runoff"})
    df.loc[df["year"] == 2021, ["year", "contest"]] = [2020, "runoff"]
    for (year, state, special), stages in ROUND_FIXES.items():
        rows = (df["year"] == year) & (df["state"] == state) & (df["special"] == special)
        if set(df.loc[rows, "stage"]) != set(stages):
            raise ValueError(f"ROUND_FIXES {(year, state, special)}: stages {set(df.loc[rows, 'stage'])}")
        df.loc[rows, "contest"] = df.loc[rows, "stage"].map(stages)
    for (year, state), special in SPECIAL_FIXES.items():
        rows = (df["year"] == year) & (df["state"] == state)
        if rows.sum() == 0 or (df.loc[rows, "special"] == special).any():
            raise ValueError(f"SPECIAL_FIXES {(year, state)} doesn't apply")
        df.loc[rows, "special"] = special
    if df["contest"].isna().any():
        raise ValueError(f"Unknown MEDSL stages: {set(df.loc[df['contest'].isna(), 'stage'])}")
    # MEDSL's 2024 MS and NV rows give the major parties only in party_simplified
    major = df["party_simplified"].isin(["DEMOCRAT", "REPUBLICAN"])
    df["party"] = df["party"].fillna(df["party_simplified"].where(major))
    # a party-less row with no name is an aggregate write-in / other row
    df["candidate"] = df["candidate"].fillna(pd.Series(np.where(df["writein"], "WRITEIN", "OTHER"), index=df.index))
    df["source"] = contest_source(df)
    return df[["year", "state", "special", "contest", "candidate", "party", "writein", "candidatevotes",
               "totalvotes", "unofficial", "source"]]


def read_la_official(date: str, directory: Path = LA_SOS_DIR) -> pd.DataFrame:
    """Every U.S. Senate candidate in Louisiana's official results for `date` (yyyymmdd): candidate (upper case, as
    "First Last"), party (from the "(DEM)" suffix), votes."""
    races = json.loads((directory / f"{date}_RacesCandidates_Multiparish.json").read_text())["Races"]["Race"]
    votes = {race["ID"]: race for race in
             json.loads((directory / f"{date}_Votes_Multiparish.json").read_text())["Races"]["Race"]}
    [race] = [r for r in races if r["SpecificTitle"] == "U. S. Senator"]
    names = {choice["ID"]: choice["Desc"] for choice in race["Choice"]}
    rows = []
    for choice in votes[race["ID"]]["Choice"]:
        name, party = re.fullmatch(r"(.*) \((\w+)\)", names[choice["ID"]]).groups()
        rows.append((name.upper(), LA_PARTIES.get(party, party), int(choice["VoteTotal"])))
    return pd.DataFrame(rows, columns=["candidate", "party", "votes"])


def la_official_rows() -> pd.DataFrame:
    """LA_OFFICIAL's November rounds and runoffs in the common row format."""
    frames = []
    for (year, state), dates in LA_OFFICIAL.items():
        for contest, date in zip(["first_round", "runoff"], dates):
            if date is None:
                continue
            r = read_la_official(date)
            frames.append(pd.DataFrame({
                "year": year, "state": state, "special": False, "contest": contest, "candidate": r["candidate"],
                "party": r["party"], "writein": False, "candidatevotes": r["votes"].astype(float),
                "totalvotes": float(r["votes"].sum()), "unofficial": False,
                "source": LA_SOURCE.format(f"{date[:4]}-{date[4:6]}-{date[6:]}")}))
    return pd.concat(frames, ignore_index=True)


def fec_file(year: int, directory: Path = FEC_DIR) -> Path:
    [path] = directory.glob(f"federalelections{year}.xls*")
    return path


def read_fec(year: int, directory: Path = FEC_DIR) -> pd.DataFrame:
    """FEC's U.S. Senate rows for `year`: state, unexpired (a special for an unexpired term), candidate ("Last,
    First"), party (FEC's code), general and runoff votes. Total and combined-party rows are dropped."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        book = pd.ExcelFile(fec_file(year, directory))
        sheet = next(s for s in book.sheet_names if "senate" in s.lower() and ("result" in s.lower()
                                                                               or "by state" in s.lower())
                     or "Resuts" in s)  # 2012's House & Senate sheet is misspelled
        df = pd.read_excel(book, sheet, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    district = df.iloc[:, 3].astype(str).str.upper()
    name = next(c for c in df.columns if c.upper() in ("CANDIDATE NAME", "CANDIDATE NAME (LAST, FIRST)",
                                                       "LAST NAME, FIRST", "LAST NAME,  FIRST"))
    general = next(c for c in df.columns if c.upper().startswith("GENERAL") and "%" not in c)
    runoff = next((c for c in df.columns if c.upper().startswith("GE RUNOFF") and "%" not in c), None)
    out = pd.DataFrame({
        "state": df["STATE ABBREVIATION"].str.strip(), "unexpired": district.str.contains("UN"),
        "candidate": df[name].str.strip(),
        "party": df["PARTY"].fillna("").str.replace("\xa0", "").str.replace("*", "").str.strip().str.upper(),
        "general": pd.to_numeric(df[general], errors="coerce"),
        "runoff": pd.to_numeric(df[runoff], errors="coerce") if runoff else np.nan})
    keep = district.str.startswith("S") & out["candidate"].notna() & ~out["party"].str.contains("COMBINED") \
        & ~out["candidate"].str.contains("Total", case=False, na=False)
    return out[keep].reset_index(drop=True)


def fec_rows(year: int, state: str, special: bool) -> pd.DataFrame:
    """One FEC contest's November round in the common row format; "Scattered" is an aggregate row."""
    fec = read_fec(year)
    fec = fec[(fec["state"] == state) & (fec["unexpired"] == special) & fec["general"].notna()]
    if fec.empty:
        raise ValueError(f"FEC has no {year} {state} special={special} contest")
    # "Last, First" or "Last, First, Jr." -> "FIRST LAST" / "FIRST LAST JR."
    parts = fec["candidate"].str.split(",")
    candidate = parts.map(lambda p: " ".join([*(x.strip() for x in p[1:2]), p[0].strip(),
                                               *(x.strip() for x in p[2:])])).str.upper()
    writein = fec["party"].str.split(r"[/()]", regex=True).map(lambda c: "W" in c)  # e.g. W, REP/W
    candidate = candidate.where(candidate != "SCATTERED", "SCATTERING")
    return pd.DataFrame({
        "year": year, "state": state, "special": special, "contest": "first_round", "candidate": candidate,
        "party": fec["party"].str.split("/").str[0].map(FEC_PARTIES), "writein": writein,
        "candidatevotes": fec["general"].astype(float), "totalvotes": float(fec["general"].sum()),
        "unofficial": False, "source": FEC_SOURCE.format(year)}).reset_index(drop=True)


def read_source() -> pd.DataFrame:
    """Every 2006-2024 contest's rows: MEDSL with ROUND_FIXES and SPECIAL_FIXES, LA_OFFICIAL and FEC_CONTESTS in place
    of MEDSL's rows for those contests, and PARTY_OVERRIDES; official_correction marks the replaced contests."""
    df = read_medsl()
    df["official_correction"] = False
    contest = ["year", "state", "special", "contest"]
    replaced = []
    la = la_official_rows()
    replaced.append(la)
    for year, state, special, name in FEC_CONTESTS:
        replaced.append(fec_rows(year, state, special))
    replaced = pd.concat(replaced, ignore_index=True)
    keys = pd.MultiIndex.from_frame(replaced[contest].drop_duplicates())
    in_medsl = pd.MultiIndex.from_frame(df[contest]).isin(keys)
    medsl_counts = df[in_medsl].groupby(contest)["candidatevotes"].apply(lambda v: sorted(v.tolist()))
    official_counts = replaced.groupby(contest)["candidatevotes"].apply(lambda v: sorted(v.tolist()))
    differs = official_counts.index.map(lambda k: medsl_counts.get(k) != official_counts[k])
    replaced = replaced.merge(pd.DataFrame({"official_correction": differs}, index=official_counts.index),
                              left_on=contest, right_index=True)
    df = pd.concat([df[~in_medsl], replaced], ignore_index=True)
    for (year, state, special, candidate), party in PARTY_OVERRIDES.items():
        rows = (df["year"] == year) & (df["state"] == state) & (df["special"] == special) \
            & (df["candidate"] == candidate)
        if not rows.any():
            raise ValueError(f"PARTY_OVERRIDES {(year, state, special, candidate)} matches no row")
        df.loc[rows, "party"] = party
        df.loc[rows, "party_override"] = True
    df["party_override"] = df["party_override"].fillna(False).astype(bool)
    return df


def assign_class(df: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Rows plus class and the key's contest label: a same-day unexpired-term special (UNEXPIRED_TERM_SPECIALS) is
    contest special under its full-term race's key. Raises unless every race has exactly one November round and
    every contest has a race."""
    df = df.copy()
    partial = pd.Series(list(zip(df["year"], df["state"])), index=df.index).isin(UNEXPIRED_TERM_SPECIALS) \
        & df["special"]
    df.loc[partial, "contest"] = "special"
    df["race_special"] = (df["special"] & ~partial).astype(int)
    df = df.merge(races[["year", "state", "special", "class"]].rename(columns={"special": "race_special"}),
                  on=["year", "state", "race_special"], how="left", validate="many_to_one")
    if df["class"].isna().any():
        raise ValueError("Contests without a senate_incumbency.csv race: "
                         f"{df.loc[df['class'].isna(), ['year', 'state', 'special']].drop_duplicates().values.tolist()}")
    df["class"] = df["class"].astype(int)
    first = df.loc[df["contest"] == "first_round", KEY].drop_duplicates()
    missing = races[KEY].merge(first, how="left", indicator=True).query("_merge == 'left_only'")
    if len(missing):
        raise ValueError(f"Races without a November round: {missing[KEY].values.tolist()}")
    return df


def candidates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per contest and candidate name, party lines summed: bucket (dem, gop, cross, 3p, nonvote),
    is_candidate, writein (every line) and votes. House house_targets.candidates, keyed by SUPP_KEY, plus
    CAUCUS_DEMOCRATS."""
    party = df["party"]
    lines = df.assign(is_dem=party.isin(SENATE_DEM_LABELS), is_gop=party.isin(SENATE_GOP_LABELS),
                      is_nonvote=party.isna() & df["candidate"].isin(NONVOTE_NAMES),
                      is_candidate=~(party.isna() & df["candidate"].isin(NONVOTE_NAMES | SENATE_AGGREGATE_NAMES)))
    out = lines.groupby([*SUPP_KEY, "candidate", "is_candidate"], as_index=False).agg(
        is_dem=("is_dem", "any"), is_gop=("is_gop", "any"), is_nonvote=("is_nonvote", "any"),
        writein=("writein", "all"), votes=("candidatevotes", "sum"))
    out["bucket"] = np.select([out["is_nonvote"], out["is_dem"] & out["is_gop"], out["is_dem"], out["is_gop"]],
                              ["nonvote", "cross", "dem", "gop"], "3p")
    caucus = pd.Series([CAUCUS_DEMOCRATS.get(k) for k in zip(out["year"], out["state"], out["class"])], index=out.index)
    is_caucus = out["candidate"] == caucus
    if is_caucus.sum() != len(CAUCUS_DEMOCRATS):
        raise ValueError("A CAUCUS_DEMOCRATS candidate matches no row")
    out.loc[caucus.notna() & (out["bucket"] == "dem"), "bucket"] = "3p"
    out.loc[is_caucus, "bucket"] = "dem"
    return out[[*SUPP_KEY, "candidate", "bucket", "is_candidate", "writein", "votes"]]


def build_contests(df: pd.DataFrame, election_dates: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per contest with every targets column, the candidate rows), from assign_class's rows."""
    cands = candidates(df)
    named = cands[cands["is_candidate"] & (cands["bucket"] != "nonvote")]
    votes = cands.pivot_table(index=SUPP_KEY, columns="bucket", values="votes", aggfunc="sum", fill_value=0)
    votes = votes.reindex(columns=["dem", "gop", "3p", "cross", "nonvote"], fill_value=0)
    n = named.pivot_table(index=SUPP_KEY, columns="bucket", values="votes", aggfunc="size", fill_value=0)
    n = n.reindex(index=votes.index, columns=["dem", "gop", "3p", "cross"], fill_value=0)
    out = df.groupby(SUPP_KEY).agg(
        special=("special", "first"), source_totalvotes=("totalvotes", "max"),
        totalvotes_values=("totalvotes", "nunique"), is_unofficial=("unofficial", "any"), source=("source", "first"),
        party_override=("party_override", "any"), official_correction=("official_correction", "any"))

    reported = cands[cands["bucket"] != "nonvote"].groupby(SUPP_KEY)["votes"].max() > 1
    total = votes[["dem", "gop", "3p", "cross"]].sum(axis=1)
    dem, gop = votes["dem"].where(reported & (n["cross"] == 0)), votes["gop"].where(reported & (n["cross"] == 0))
    valid_total = total.where(reported & (total > 0))
    out["dem_votes"], out["gop_votes"] = dem, gop
    out["3p_votes"], out["total_votes"] = votes["3p"].where(reported), total.where(reported)
    out["dem_pct"], out["gop_pct"] = 100 * dem / valid_total, 100 * gop / valid_total
    out["3p_pct"] = 100 * votes["3p"] / valid_total
    out["margin_votes"] = dem - gop
    out["margin_raw"] = 100 * (dem - gop) / valid_total
    out["margin_2p_norm"] = 100 * (dem - gop) / (dem + gop).where(dem + gop > 0)
    year, state = out.index.get_level_values("year"), out.index.get_level_values("state")
    contest, keys = out.index.get_level_values("contest"), out.index.droplevel("contest")
    out["is_rcv"] = year >= state.map(RCV_FROM).fillna(9999)
    out["is_special_election"] = out["special"].astype(bool)
    jungle_special = pd.Series(list(zip(year, state)), index=out.index).isin(JUNGLE_SPECIALS) & out["special"]
    jungle = (((state == "LA") & year.isin(LA_JUNGLE_YEARS)) | jungle_special) & (contest != "runoff")
    writein_only = named[named["bucket"].isin(["dem", "gop"])].groupby([*SUPP_KEY, "bucket"])["writein"].all() \
        .groupby(SUPP_KEY).any().reindex(out.index, fill_value=False)
    out["race_type"] = race_types(n, writein_only, out["is_rcv"], jungle)
    out["dem_candidate_count"], out["gop_candidate_count"], out["3p_candidate_count"] = n["dem"], n["gop"], n["3p"]
    has_target = out["race_type"].isin(DR_TYPES) & out["margin_2p_norm"].notna()
    out["adj_margin_2p_norm"] = out["margin_2p_norm"].where(has_target)
    out["is_imputed"] = pd.Series(False, index=out.index, dtype=object).where(has_target)
    out["needs_imputation"] = ~has_target
    out["is_nonstandard_race"] = ~out["race_type"].isin(STANDARD_RACE_TYPES)
    out["stage"] = np.where(contest == "runoff", "runoff", "general")
    out["election_date"] = [RUNOFF_DATES[(y, s, sp)] if c == "runoff" else election_dates[str(y)]["election_day"]
                            for y, s, sp, c in zip(year, state, out["special"], contest)]

    # winner: the runoff's winner where there is one; otherwise the top named candidate, unless tied or a jungle
    # round's leader without a majority (then only a same-party top two settles it)
    ranked = named.sort_values([*SUPP_KEY, "votes"], ascending=[True] * len(SUPP_KEY) + [False]).groupby(SUPP_KEY)
    top, second = (ranked.nth(i).set_index(SUPP_KEY).reindex(out.index) for i in (0, 1))
    has_runoff = (contest == "first_round") & keys.isin(keys[contest == "runoff"])
    tied = reported & (top["votes"] == second["votes"]) & ~has_runoff
    no_majority = jungle & ~has_runoff & reported & second["votes"].notna() & (top["votes"] <= total / 2)
    same_party_top_two = top["bucket"] == second["bucket"]
    party = {"dem": "D", "gop": "R", "3p": "3P"}
    winner = top["bucket"].map(party).where(~tied & (~no_majority | same_party_top_two))
    runoff_winner = winner[contest == "runoff"].droplevel("contest").reindex(keys).to_numpy()
    out["winner_party"] = winner.where(~has_runoff, runoff_winner)
    runoff_top = top.loc[contest == "runoff", "candidate"].droplevel("contest").reindex(keys).to_numpy()
    leader_lost = has_runoff & (name_key(top["candidate"].fillna("")).to_numpy() != name_key(
        pd.Series(runoff_top, index=out.index).fillna("")).to_numpy())

    other_contests = pd.Series(contest + "_in_source", index=keys)[contest != "first_round"].groupby(KEY).agg(";".join)
    rcv_tabulation = pd.Series(list(zip(year, state, out["special"])), index=out.index).isin(RCV_ROUND1_TABULATION)
    caucus_dem = pd.Series(list(zip(year, state, out.index.get_level_values("class"))), index=out.index) \
        .isin(CAUCUS_DEMOCRATS)
    flags = {
        "other_contests": other_contests.reindex(keys).where(contest == "first_round").to_numpy(),
        "votes_not_reported": ~reported,
        "source_totalvotes_differs": reported & (out["source_totalvotes"] != total)
                                     & (out["source_totalvotes"] != total + votes["nonvote"]),
        "source_totalvotes_inconsistent": out["totalvotes_values"] > 1,
        "party_override": out["party_override"],
        "winner_party_unresolved": out["winner_party"].isna() & ~tied & ~no_majority & ~has_runoff,
        "winner_tied": tied,
        "no_majority_runoff_not_in_source": no_majority,
        "election_day_leader_lost": leader_lost,
        "rcv_round1_tabulation": rcv_tabulation & (contest == "first_round"),
        "official_results_correction": out["official_correction"],
        "independent_counted_as_dem": caucus_dem,
    }
    parts = pd.DataFrame({name: mask if name == "other_contests" else np.where(mask, name, None)
                          for name, mask in flags.items()}, index=out.index)
    out["quality_flags"] = parts.apply(lambda row: ";".join(row.dropna()) or None, axis=1)
    out["nonvote_votes"] = votes["nonvote"].astype("int64")
    out = out.reset_index()
    for c in ["dem_votes", "gop_votes", "3p_votes", "total_votes", "margin_votes", "source_totalvotes"]:
        out[c] = out[c].astype("Int64")
    return out, cands


def add_flag(flags: pd.Series, flag: str, mask) -> pd.Series:
    return flags.where(~np.asarray(mask), flags.fillna("").map(lambda f: f"{f};{flag}" if f else flag))


def fec_check(contests: pd.DataFrame) -> pd.DataFrame:
    """Per 2006-2022 general and runoff contest: our D, R and total votes against FEC's, and whether they differ
    beyond FEC_TOLERANCE. A candidate is D (R) if any of their FEC party codes is D, DEM, DFL or DNL (R, REP)."""
    rows = []
    for year in range(FIRST_YEAR, 2023, 2):
        fec = read_fec(year)
        codes = fec["party"].str.split(r"[/()\s]+", regex=True)
        for column, stage in [("general", "general"), ("runoff", "runoff")]:
            # a candidate's party is read only from the lines with votes in this round (a primary-only line, such
            # as a write-in on the other party's primary ballot, has none)
            lines = fec.assign(d=codes.map(lambda c: bool({"D", "DEM", "DFL", "DNL"} & set(c))),
                               r=codes.map(lambda c: bool({"R", "REP"} & set(c))))[fec[column] > 0]
            by_candidate = lines.groupby(["state", "unexpired", "candidate"]).agg(
                d=("d", "any"), r=("r", "any"), votes=(column, "sum"))
            # CAUCUS_DEMOCRATS: the independent is D and the race's Democratic nominee isn't (FEC: "Last, First")
            caucus = {state: re.sub(r",?\s+JR\.?$", "", name).split()[-1]
                      for (y, state, _), name in CAUCUS_DEMOCRATS.items() if y == year}
            states = by_candidate.index.get_level_values("state")
            fec_surname = by_candidate.index.get_level_values("candidate").str.split(",").str[0].str.upper()
            in_caucus_race = states.isin(list(caucus)) & ~by_candidate.index.get_level_values("unexpired")
            by_candidate["d"] = np.where(in_caucus_race, fec_surname == states.map(caucus), by_candidate["d"])
            sums = by_candidate.assign(dv=by_candidate["votes"].where(by_candidate["d"], 0),
                                       rv=by_candidate["votes"].where(by_candidate["r"] & ~by_candidate["d"], 0)) \
                .groupby(["state", "unexpired"]).agg(fec_dem=("dv", "sum"), fec_gop=("rv", "sum"),
                                                     fec_total=("votes", "sum"))
            sums = sums[sums["fec_total"] > 0].reset_index().assign(year=year, stage=stage)
            rows.append(sums)
    fec = pd.concat(rows, ignore_index=True).rename(columns={"unexpired": "is_special_election"})
    # FEC doesn't always mark a special as unexpired-term (AZ 2020): a state-year with one contest in each source
    # matches whatever the flag
    ours = contests[contests["year"] <= 2022]
    single = ours.groupby(["year", "state", "stage"])["contest"].transform("size").eq(1)
    fec_single = fec.groupby(["year", "state", "stage"])["fec_total"].transform("size").eq(1)
    fec.loc[fec_single, "is_special_election"] = fec.loc[fec_single, ["year", "state", "stage"]].merge(
        ours.loc[single, ["year", "state", "stage", "is_special_election"]], how="left")["is_special_election"] \
        .fillna(fec.loc[fec_single, "is_special_election"]).to_numpy()
    out = ours[[*SUPP_KEY, "stage", "is_special_election", "dem_votes", "gop_votes", "total_votes"]].merge(
        fec, on=["year", "state", "is_special_election", "stage"], how="left")
    tolerance = np.maximum(FEC_TOLERANCE[0], FEC_TOLERANCE[1] * out["total_votes"].astype(float))
    out["d_diff"] = out["dem_votes"].astype(float) - out["fec_dem"]
    out["r_diff"] = out["gop_votes"].astype(float) - out["fec_gop"]
    out["total_diff"] = out["total_votes"].astype(float) - out["fec_total"]
    out["differs"] = (out[["d_diff", "r_diff"]].abs().max(axis=1) > tolerance) | out["fec_total"].isna()
    return out


def treated_uncontested_candidates(raw: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    """D-R first rounds where a third-party candidate outpolled a major party's best candidate (for review)."""
    named = cands[cands["is_candidate"] & (cands["bucket"] != "nonvote") & (cands["contest"] == "first_round")]
    best = named.groupby([*KEY, "bucket"])["votes"].max().unstack()
    dr = raw.loc[raw["race_type"].isin(DR_TYPES), KEY]
    best = best.reindex(pd.MultiIndex.from_frame(dr))
    hit = best["3p"] > best[["dem", "gop"]].min(axis=1)
    return best[hit].reset_index()


def treat_uncontested(targets: pd.DataFrame) -> pd.DataFrame:
    """TREATED_UNCONTESTED races become <party>_vs_3p (the major party that finished second) with blank targets."""
    out = targets.copy()
    for key in TREATED_UNCONTESTED:
        row = (out["year"] == key[0]) & (out["state"] == key[1]) & (out["class"] == key[2])
        if row.sum() != 1 or out.loc[row, "race_type"].item() not in DR_TYPES:
            raise ValueError(f"TREATED_UNCONTESTED {key} isn't one D-R race")
        side = "dem" if (out.loc[row, "dem_votes"] > out.loc[row, "gop_votes"]).item() else "gop"
        out.loc[row, "race_type"] = f"{side}_vs_3p"
        out.loc[row, ["adj_margin_2p_norm", "is_imputed"]] = None
        out.loc[row, ["needs_imputation", "is_nonstandard_race"]] = True
        out.loc[row, "quality_flags"] = add_flag(out.loc[row, "quality_flags"], TREATED_UNCONTESTED_FLAG, [True])
    return out


def build(source: pd.DataFrame, races: pd.DataFrame, lean: pd.DataFrame, shave: pd.DataFrame,
          election_dates: dict) -> dict[str, pd.DataFrame]:
    df = assign_class(source, races)
    contests, cands = build_contests(df, election_dates)
    check = fec_check(contests)
    differs = pd.MultiIndex.from_frame(contests[SUPP_KEY]).isin(
        pd.MultiIndex.from_frame(check.loc[check["differs"] & check["fec_total"].notna(), SUPP_KEY]))
    contests["quality_flags"] = add_flag(contests["quality_flags"], "differs_from_fec", differs)

    first = contests[contests["contest"] == "first_round"].sort_values(KEY, ignore_index=True)
    raw = first.reindex(columns=list(TARGETS))
    for c in ["dem_votes_imputed", "gop_votes_imputed"]:
        raw[c] = raw[c].astype("Int64")
    raw["imputation_method"] = raw["imputation_method"].astype(object)

    # targets.csv: jungle and RCV targets observed from their party totals, TREATED_UNCONTESTED, then the
    # uncontested imputation with the state's partisan lean
    targets = impute_jungle_rcv(treat_uncontested(raw))
    targets = impute_uncontested(targets, lean, shave, key=KEY)[list(TARGETS)]

    treated = targets["quality_flags"].fillna("").str.contains(TREATED_UNCONTESTED_FLAG)
    imputed_with_opponent = targets.loc[targets["race_type"].isin(["dem_vs_3p", "gop_vs_3p"]) | treated, KEY]
    supp_keys = pd.MultiIndex.from_frame(pd.concat(
        [first.loc[first["race_type"].isin(IMPUTE_RACE_TYPES), KEY], imputed_with_opponent]))
    keys = pd.MultiIndex.from_frame(contests[KEY])
    in_supp = contests.duplicated(KEY, keep=False) | keys.isin(supp_keys)
    supp = contests[in_supp].sort_values(SUPP_KEY, ignore_index=True)[list(TARGETS_SUPP)]

    supp_index = pd.MultiIndex.from_frame(supp[SUPP_KEY])
    lines = df[pd.MultiIndex.from_frame(df[SUPP_KEY]).isin(supp_index)].groupby([*SUPP_KEY, "candidate"]).agg(
        parties=("party", lambda p: "; ".join(sorted(p.dropna().unique())) or None), source=("source", "first"))
    candidate_rows = cands[pd.MultiIndex.from_frame(cands[SUPP_KEY]).isin(supp_index)] \
        .join(lines, on=[*SUPP_KEY, "candidate"]).rename(columns={"writein": "is_writein"})
    candidate_rows["party_group"] = candidate_rows["bucket"].map(
        {"dem": "D", "gop": "R", "3p": "3P", "cross": "D+R", "nonvote": "nonvote"})
    candidate_rows["stage"] = np.where(candidate_rows["contest"] == "runoff", "runoff", "general")
    candidate_rows["votes"] = candidate_rows["votes"].astype("int64")
    candidate_rows = candidate_rows.sort_values([*SUPP_KEY, "votes"], ascending=[True] * len(SUPP_KEY) + [False])
    return {"targets_raw.csv": raw, "targets.csv": targets, "targets_supp.csv": supp,
            "targets_candidates.csv": candidate_rows.reset_index(drop=True)[list(TARGETS_CANDIDATES)],
            "_fec_check": check, "_treated_candidates": treated_uncontested_candidates(raw, cands)}


def check_targets(out: dict[str, pd.DataFrame], races: pd.DataFrame) -> list[str]:
    """Blocking problems: targets keys aren't the 2006-2024 races; the House rules weren't applied as documented;
    imputation changed a vote column."""
    raw, targets = out["targets_raw.csv"], out["targets.csv"]
    problems = []
    expected = races.loc[races["year"].between(FIRST_YEAR, LAST_YEAR), KEY].sort_values(KEY, ignore_index=True)
    for name, df in [("targets_raw.csv", raw), ("targets.csv", targets)]:
        if not df[KEY].sort_values(KEY, ignore_index=True).equals(expected):
            problems.append(f"{name}: keys aren't senate_incumbency.csv's 2006-2024 races")
    if targets["race_type"].eq("cross_endorsed").any():
        problems.append("A cross-endorsed race: house_targets.py's cross-endorsed rule isn't implemented here")
    merged = targets.merge(raw, on=KEY, suffixes=("", "_raw"))
    for c in ["dem_votes", "gop_votes", "3p_votes", "total_votes"]:
        if not merged[c].astype("Float64").equals(merged[f"{c}_raw"].astype("Float64")):
            problems.append(f"targets.csv {c} differs from targets_raw.csv")
    observed = targets["is_imputed"].eq(False)
    if not np.allclose(targets.loc[observed, "adj_margin_2p_norm"], targets.loc[observed, "margin_2p_norm"]):
        problems.append("An observed target isn't its margin_2p_norm")
    applied = targets["quality_flags"].fillna("").str.contains(TREATED_UNCONTESTED_FLAG).sum()
    if applied != len(TREATED_UNCONTESTED):
        problems.append(f"TREATED_UNCONTESTED: {applied} of {len(TREATED_UNCONTESTED)} applied")
    return problems


def report(out: dict[str, pd.DataFrame]) -> str:
    raw, targets, supp, check = out["targets_raw.csv"], out["targets.csv"], out["targets_supp.csv"], out["_fec_check"]
    flags = raw["quality_flags"].dropna().str.split(";").explode().str.replace(r"\(.*", "", regex=True)
    imputed = targets["imputation_method"].value_counts()
    differ = check[check["differs"]]
    lines = [
        f"targets_raw.csv: {len(raw)} races, {raw['year'].min()}-{raw['year'].max()}; "
        f"regular {int((~raw['is_special_election']).sum())}, special {int(raw['is_special_election'].sum())}",
        "races per year: " + ", ".join(f"{y} {n}" for y, n in raw.groupby("year").size().items()),
        "race_type (raw): " + ", ".join(f"{k} {v}" for k, v in raw["race_type"].value_counts().items()),
        "race_type (targets.csv): " + ", ".join(f"{k} {v}" for k, v in targets["race_type"].value_counts().items()),
        "quality_flags (raw): " + ", ".join(f"{k} {v}" for k, v in flags.value_counts().items()),
        f"targets.csv: target set on {targets['adj_margin_2p_norm'].notna().sum()}, observed "
        f"{int(targets['is_imputed'].eq(False).sum())}; imputed: "
        + ", ".join(f"{k} {v}" for k, v in imputed.items()),
        f"is_rcv {int(raw['is_rcv'].sum())}, winner_party blank {int(raw['winner_party'].isna().sum())}, "
        "winner_party: " + ", ".join(f"{k} {v}" for k, v in raw["winner_party"].value_counts().items()),
        f"targets_supp.csv: {len(supp)} contests of {len(supp.drop_duplicates(KEY))} keys ("
        + ", ".join(f"{k} {v}" for k, v in supp["contest"].value_counts().items()) + ")",
        f"FEC check (2006-2022): {len(check)} contests, {int(check['fec_total'].isna().sum())} not found in FEC, "
        f"{int((check['differs'] & check['fec_total'].notna()).sum())} differ beyond max(100 votes, 0.1%):",
        differ[[*SUPP_KEY, "dem_votes", "fec_dem", "d_diff", "gop_votes", "fec_gop", "r_diff", "total_diff"]]
        .to_string(index=False),
        "D-R races where a third-party candidate outpolled a major party (TREATED_UNCONTESTED candidates):",
        out["_treated_candidates"].to_string(index=False),
    ]
    return "\n".join(lines)


def read_lean(output_dir: Path, races: pd.DataFrame) -> pd.DataFrame:
    """KEY + lean, the state's D two-party share (0-1) in a neutral year: 0.5 + pvi / 100 (state_pvi.csv)."""
    pvi = read_output("state_pvi.csv", output_dir)
    lean = races[KEY].merge(pvi[["year", "state", "pvi"]], on=["year", "state"], how="left", validate="many_to_one")
    return lean[KEY].assign(lean=0.5 + lean["pvi"] / 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    races = pd.read_csv(INCUMBENCY_PATH)
    races = races[races["year"].between(FIRST_YEAR, LAST_YEAR)]
    out = build(read_source(), races, read_lean(args.output_dir, races), read_shave(),
                json.loads(ELECTION_DATES_PATH.read_text()))
    print(report(out))
    problems = check_targets(out, races)
    if problems:
        raise ValueError("Target problems:\n" + "\n".join(problems))
    write_outputs({k: v for k, v in out.items() if not k.startswith("_")}, args.output_dir)


if __name__ == "__main__":
    main()
