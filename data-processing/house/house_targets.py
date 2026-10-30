"""Build observed U.S. House general-election results from MEDSL's candidate-level returns
(data/reference/1976-2024-house.tab), in data/processed/house. Nothing is imputed.

  targets_raw.csv    one row per year, state, district, 1976-2024, exactly as observed
  targets.csv        the Stage 1 targets: targets_raw.csv's rows from FIRST_YEAR (2006, the first year of modeling.csv)
                     on, except that jungle and RCV targets are left to imputation (IMPUTE_RACE_TYPES) and a
                     cross-endorsed candidate counts for their larger line (flag cross_endorsed_assigned)
  targets_supp.csv   a companion to targets.csv, from FIRST_YEAR on: one row per year, state, district, contest for
                     every key whose source has more than one contest or whose race is jungle or RCV. contest is
                     first_round (the November round, the one the targets use), runoff (a December runoff), or special
                     (a same-day special for the unexpired term). It isn't joined into either targets file; it is kept
                     for future imputation of adj_margin_2p_norm, post-processing and model evaluation.
  targets_candidates.csv  one row per year, state, district, contest, candidate of every targets_supp.csv contest:
                     each candidate's line-summed votes (plus aggregate write-in and non-vote rows), so a jungle or RCV
                     race's per-candidate breakdown is kept. It joins to targets_supp.csv on year, state, district,
                     contest, and its votes add up to that file's party totals and candidate counts.

Votes:
  Each named candidate's party lines are summed (fusion tickets, per the codebook). A candidate is D if one of their
  lines is a DEM_LABELS label, R if one is a GOP_LABELS label, and third party otherwise; one on both D and R lines
  (NY 1976-1990) is cross-endorsed and counted as neither, so that row's D/R votes and margins stay blank.
  PARTY_OVERRIDES recodes two Republican nominees the source lists without a party (TX-22 2006, VA-9 2002; flag
  party_override). Party-less WRITEIN / SCATTERING / OTHER rows count toward 3p_votes but aren't candidates. Blank, void, over/under
  and exhausted-ballot rows (NONVOTE_NAMES) aren't votes: their sum is kept in nonvote_votes, outside total_votes.
  total_votes = dem + gop + 3p votes; the source's own totalvotes is kept as source_totalvotes.
  Contests whose every candidate has at most 1 vote (FL and others list unopposed candidates as 1, 0 or -1) have
  no reported votes: their vote, percentage and margin fields are blank (flag votes_not_reported).

Contest per key: one per year, state, district, always the November round. Thirteen keys have a second contest: a
  December runoff (TX 1996 8/9/25, LA 2002 5, TX 2006 23, and the seven LA_OFFICIAL races; flag runoff_in_source) or
  a same-day special for the unexpired term (TX 2006 22, where the regular election is used; flag special_in_source).
  This matches the other jungle races, whose runoffs the source lacks. is_special_election marks a November round
  that is a special.
  targets_supp.csv has every contest of those keys from FIRST_YEAR on, with the same columns computed per contest;
  a runoff is never dr_jungle. In an RCV race the November round is the first round, treated like a jungle round.

race_type, from named candidates per party (RACE_TYPES in schema.py; README.md has the full table). Races with a D
  and an R are, in this order of precedence:
  dr_writein              one major party has only write-in candidates (effectively uncontested on the ballot)
  dr_rcv                  ranked-choice voting (is_rcv)
  dr_jungle               an all-candidate November round: LA in LA_JUNGLE_YEARS, and the TX specials of 1996, 2006
  dr / dr_multi           an ordinary race; dr_multi when either party has more than one candidate (party write-ins)
Other races:
  dem_vs_3p / gop_vs_3p   one major-party candidate against named third-party or write-in candidates only
  dem_/gop_uncontested    a single named candidate (aggregate write-in votes may exist)
  dem_vs_dem / gop_vs_gop two or more of one major party and none of the other
  cross_endorsed, no_major_party

Targets: adj_margin_2p_norm = margin_2p_norm for DR_TYPES rows with reported votes,
  and blank otherwise; is_imputed is False where the target is set and blank where it isn't. needs_imputation marks
  the rows without an observed target, and is_nonstandard_race every race_type but dr and dr_multi.
  dem/gop_votes_imputed and imputation_method are placeholders, blank for now.
  In targets.csv only: dr_jungle and dr_rcv rows have blank targets and needs_imputation True (their combined D vs R
  vote is the target, to be filled by imputation from targets_raw.csv / targets_supp.csv; the vote columns are kept),
  and a cross-endorsed candidate who was the only major-party candidate counts for their larger line, which sets
  race_type, the candidate count and winner_party (none from 2006 on; D/R votes stay blank).

winner_party (D, R, 3P): the runoff's top candidate where the source has the runoff, else the contest's own.
  Blank on a tie or a cross-endorsed winner, and in a jungle round whose leader has at most 50% and whose runoff the
  source lacks, unless the top two share a party (flag no_majority_runoff_not_in_source).

is_rcv: ME from 2018 and AK from 2022 (RCV_FROM). The first-round result is wanted, and the source has it except in
  ME-2 2018, 2022 and 2024 (RCV_FIRST_ROUND_NOT_IN_SOURCE; flag rcv_first_round_not_in_source): 2022 and 2024 carry a
  later round, and 2018 matches neither round. Rows keep what the source gives.

Official corrections: every 2006-2024 Louisiana race was checked against Louisiana's official results
  (data/reference/la_sos). In six (LA-2 2006, LA-5/6 2014, LA-3/4 2016, LA-5 2020) MEDSL gives the top two candidates'
  December runoff totals in place of their November counts; LA_OFFICIAL replaces them with the official November counts
  (flag official_results_correction; `source` names both sources). It also adds the official runoffs of those six and
  LA-3 2012 as runoff contests. Every other race matches the official November results.

DC (a non-voting delegate) is excluded, as in the feature tables. Validation: the key must be unique and valid, every
year must have 435 districts and match data/reference/district_counts.json where it has that year; everything
else is reported, not blocking.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import acs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from house.regression.schema import (
    CANDIDATE_KEY, KEY, SUPP_KEY, TARGET_CANDIDATE_COLUMNS, TARGET_COLUMNS, TARGET_PROVENANCE_COLUMNS,
    TARGET_SUPP_COLUMNS, TARGETS_CANDIDATES_SCHEMA, TARGETS_SCHEMA, TARGETS_SUPP_SCHEMA, validate_schema,
)

SOURCE_PATH = ROOT / "data/reference/1976-2024-house.tab"
DISTRICT_COUNTS_PATH = ROOT / "data/reference/district_counts.json"
LA_OFFICIAL_DIR = ROOT / "data/reference/la_sos"  # Louisiana SOS official results; see its README.md
OUTPUT_DIR = ROOT / "data/processed/house"
FIRST_YEAR = 2006

DEM_LABELS = {"DEMOCRAT", "DEMOCRATIC", "DEMOCRATIC-FARMER-LABOR", "DEMOCRATIC-FARM-LABOR",
              "DEMOCRATIC-NONPARTISAN LEAGUE", "DEMOCRATIC-NPL", "WRITE-IN (DEMOCRATIC)"}
# INDEPENDENT-REPUBLICAN is the Minnesota GOP's 1992 ballot name; VT 2024's nominee is listed "REPUBLICAN, LIBERTARIAN"
GOP_LABELS = {"REPUBLICAN", "INDEPENDENT-REPUBLICAN", "WRITE-IN (REPUBLICAN)", "REPUBLICAN, LIBERTARIAN"}
# party-less rows that aren't votes for anyone; the combined BLANK .../SCATTERING labels can't be split, so they count
# as non-votes too
NONVOTE_NAMES = {"BLANK", "BLANKS", "BLANK VOTE", "BLANK VOTES", "BLANK VOTE/SCATTERING",
                 "BLANK VOTE/VOID VOTE/SCATTERING", "VOID", "VOID VOTE", "OVER VOTE", "OVER VOTES", "OVERVOTES",
                 "UNDER VOTES", "UNDERVOTES", "EXHAUSTED BALLOT", "EXHAUSTED BALLOTS"}
# party-less rows of votes for unnamed others (and four bare party-line names): 3p votes, not candidates
AGGREGATE_NAMES = {"WRITEIN", "OTHER", "OTHERS", "ALL OTHERS", "OTHER WRITE-INS", "SCATTERING", "SCATTER",
                   "MISCELLANEOUS", "WORKING FAMILIES", "INDEPENDENT", "MODERATE"}
RCV_FROM = {"ME": 2018, "AK": 2022}
# Louisiana years whose November election is an all-candidate (jungle) round with a December runoff. 1978-1996 it was
# the general after an earlier open primary; 1976, 2008 and 2010 had closed primaries and plurality generals
LA_JUNGLE_YEARS = {*range(1998, 2007, 2), *range(2012, 2025, 2)}
# MEDSL marks TX-23 2006's December runoff (Rodriguez vs Bonilla) as the regular election, and the November round as
# the special; these keys' regular contest is treated as the runoff
RUNOFF_FIXES = [(2006, "TX", 23)]
# Louisiana House races (2006-2024) with a December runoff, and their (November, runoff) dates in Louisiana's official
# results (LA_OFFICIAL_DIR). Each runoff is added as a runoff contest. Every MEDSL November candidate must match one
# official candidate, and where the counts differ they are replaced by the official ones (flag
# official_results_correction): MEDSL gives the top two candidates' runoff totals in place of their November counts
# (in LA-5 2020, the runoff totals plus 100,000). LA-3 2012's November counts already match; every other 2006-2024
# Louisiana race matches the official November results and had no runoff
LA_OFFICIAL = {
    (2006, "LA", 2): ("20061107", "20061209"),
    (2012, "LA", 3): ("20121106", "20121208"),
    (2014, "LA", 5): ("20141104", "20141206"),
    (2014, "LA", 6): ("20141104", "20141206"),
    (2016, "LA", 3): ("20161108", "20161210"),
    (2016, "LA", 4): ("20161108", "20161210"),
    (2020, "LA", 5): ("20201103", "20201205"),
}
# official name (as name_key gives it) -> MEDSL's, where the two spell a candidate differently
LA_NAME_ALIASES = {"JEFF LANDRY": "JEFFREY M LANDRY"}
# Republican nominees MEDSL lists as party-less write-ins; each must match exactly one party-less write-in row
PARTY_OVERRIDES = {
    (2006, "TX", 22, "SHELLEY SEKULA GIBBS"): "REPUBLICAN",  # write-in nominee after DeLay withdrew
    (2002, "VA", 9, "J KATZEN"): "REPUBLICAN",  # Jay Katzen, against Boucher
}
DR_TYPES = ["dr", "dr_multi", "dr_jungle", "dr_rcv", "dr_writein"]
# dr_multi's extra candidates are almost always a party write-in with a handful of votes next to the nominee; every
# other race_type is flagged is_nonstandard_race
STANDARD_RACE_TYPES = ["dr", "dr_multi"]
# Jungle and RCV races: the combined D vs R first-round vote is the right target, but it is left to the imputation step
# (from targets_raw.csv / targets_supp.csv), so targets.csv blanks their targets; targets_raw.csv keeps them
IMPUTE_RACE_TYPES = ["dr_jungle", "dr_rcv"]
# RCV races whose source rows aren't first-round counts (flag rcv_first_round_not_in_source): ME-2 2022 and 2024 carry
# a later round (with exhausted ballots), and ME-2 2018 matches neither round. Every other RCV row was decided in round
# 1 (ME) or is the first round (AK)
RCV_FIRST_ROUND_NOT_IN_SOURCE = [(2018, "ME", 2), (2022, "ME", 2), (2024, "ME", 2)]
CONTEST = [*KEY, "stage", "special", "runoff"]  # what identifies a contest in the source
SEATS = 435


def at(df: pd.DataFrame, year: int, state: str, district: int) -> pd.Series:
    return (df["year"] == year) & (df["state"] == state) & (df["district"] == district)


def medsl_source(version: pd.Series) -> pd.Series:
    return "MEDSL U.S. House 1976-2024 (version " + version.astype(str) + ")"


def name_key(names: pd.Series) -> pd.Series:
    """Candidate names comparable across sources: upper case, letters and spaces only, a "(PARTY)" suffix dropped."""
    return names.str.replace(r"\s*\([A-Z]+\)$", "", regex=True).str.upper() \
        .str.replace(r"[^A-Z ]", "", regex=True).str.split().str.join(" ")


def read_la_official(date: str, directory: Path = LA_OFFICIAL_DIR) -> pd.DataFrame:
    """Every U.S. House candidate in Louisiana's official results for `date` (yyyymmdd): district, name, votes."""
    races = json.loads((directory / f"{date}_RacesCandidates_Multiparish.json").read_text())["Races"]["Race"]
    votes = {race["ID"]: race for race in
             json.loads((directory / f"{date}_Votes_Multiparish.json").read_text())["Races"]["Race"]}
    rows = []
    for race in races:
        match = re.fullmatch(r"U\. S\. Representative -- (\d+)\w\w Congressional District", race["SpecificTitle"])
        if match:
            names = {choice["ID"]: choice["Desc"] for choice in race["Choice"]}
            rows += [(int(match.group(1)), names[c["ID"]], int(c["VoteTotal"])) for c in votes[race["ID"]]["Choice"]]
    return pd.DataFrame(rows, columns=["district", "name", "votes"])


def apply_la_official(df: pd.DataFrame, directory: Path = LA_OFFICIAL_DIR) -> pd.DataFrame:
    """df with LA_OFFICIAL's November counts replaced by the official ones where they differ, and their runoffs
    appended."""
    runoffs = []
    for (year, state, district), (november, december) in LA_OFFICIAL.items():
        rows = at(df, year, state, district) & (df["runoff"] != "TRUE")
        names = name_key(df.loc[rows, "candidate"])
        official = {}
        for date in (november, december):
            results = read_la_official(date, directory).query("district == @district")
            official[date] = results.set_index(name_key(results["name"]).replace(LA_NAME_ALIASES))["votes"]
        unmatched = set(names) ^ set(official[november].index) | set(official[december].index) - set(names)
        if unmatched:
            raise ValueError(f"LA_OFFICIAL {(year, state, district)}: names in only one source: {sorted(unmatched)}")
        november_votes = names.map(official[november]).to_numpy()
        if (df.loc[rows, "candidatevotes"].to_numpy() != november_votes).any():
            df.loc[rows, "candidatevotes"] = november_votes
            df.loc[rows, "totalvotes"] = official[november].sum()
            df.loc[rows, "source"] += \
                f"; Louisiana SOS official results ({november[:4]}-{november[4:6]}-{november[6:]})"
        runoff = df[rows][names.isin(official[december].index).to_numpy()].assign(
            candidatevotes=names[names.isin(official[december].index)].map(official[december]).to_numpy(),
            totalvotes=official[december].sum(), runoff="TRUE",
            source=f"Louisiana SOS official results ({december[:4]}-{december[4:6]}-{december[6:]})")
        runoffs.append(runoff)
    return pd.concat([df, *runoffs], ignore_index=True)


def read_source(path: Path = SOURCE_PATH) -> pd.DataFrame:
    """The MEDSL rows, with state_po as `state`, party "NA" as null, runoff (TRUE / FALSE / NA) as text, `source`,
    and RUNOFF_FIXES, PARTY_OVERRIDES and LA_OFFICIAL applied; raises unless every fix and
    RCV_FIRST_ROUND_NOT_IN_SOURCE entry matches."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"runoff": str, "party": str})
    df["source"] = medsl_source(df["version"])
    df["runoff"] = df["runoff"].str.upper()
    df["party"] = df["party"].replace("NA", None)
    df = df.drop(columns="state").rename(columns={"state_po": "state"})
    for key in RUNOFF_FIXES:
        rows = at(df, *key) & ~df["special"]
        if not rows.any():
            raise ValueError(f"RUNOFF_FIXES {key} matches no regular contest")
        df.loc[rows, "runoff"] = "TRUE"
    for (*key, candidate), party in PARTY_OVERRIDES.items():
        rows = at(df, *key) & (df["candidate"] == candidate) & df["party"].isna() & df["writein"]
        if rows.sum() != 1:
            raise ValueError(f"PARTY_OVERRIDES {(*key, candidate)} matches {rows.sum()} party-less write-in rows, "
                             "not 1")
        df.loc[rows, "party"] = party
    for year, state, district in RCV_FIRST_ROUND_NOT_IN_SOURCE:
        if year < RCV_FROM.get(state, 9999) or not at(df, year, state, district).any():
            raise ValueError(f"RCV_FIRST_ROUND_NOT_IN_SOURCE {(year, state, district)} is not an RCV race in the "
                             "source")
    return apply_la_official(df)


def label_contests(df: pd.DataFrame) -> pd.DataFrame:
    """Rows plus contest: first_round for each key's November round, runoff for a December runoff, and special for a
    same-day special for the unexpired term held alongside the regular election."""
    contests = df[CONTEST].drop_duplicates()
    contests["is_runoff"] = contests["runoff"] == "TRUE"
    contests = contests.sort_values([*KEY, "is_runoff", "special"])
    contests["contest"] = np.select([~contests.duplicated(KEY), contests["is_runoff"]], ["first_round", "runoff"],
                                    "special")
    if contests.duplicated(SUPP_KEY).any():
        raise ValueError(f"Keys with two contests of one kind:\n{contests[contests.duplicated(SUPP_KEY, keep=False)]}")
    return df.merge(contests[[*CONTEST, "contest"]], on=CONTEST, validate="many_to_one")


def candidates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per key, contest and candidate name, party lines summed: bucket, is_candidate, writein (every line),
    votes.

    bucket is dem, gop, cross (both), or 3p; party-less NONVOTE_NAMES rows are bucket nonvote, and they and the
    AGGREGATE_NAMES rows aren't candidates.
    """
    party = df["party"]
    lines = df.assign(is_dem=party.isin(DEM_LABELS), is_gop=party.isin(GOP_LABELS),
                      is_nonvote=party.isna() & df["candidate"].isin(NONVOTE_NAMES),
                      is_candidate=~(party.isna() & df["candidate"].isin(NONVOTE_NAMES | AGGREGATE_NAMES)))
    out = lines.groupby([*SUPP_KEY, "candidate", "is_candidate"], as_index=False).agg(
        is_dem=("is_dem", "any"), is_gop=("is_gop", "any"), is_nonvote=("is_nonvote", "any"),
        writein=("writein", "all"), votes=("candidatevotes", "sum"))
    out["bucket"] = np.select([out["is_nonvote"], out["is_dem"] & out["is_gop"], out["is_dem"], out["is_gop"]],
                              ["nonvote", "cross", "dem", "gop"], "3p")
    return out[[*SUPP_KEY, "candidate", "bucket", "is_candidate", "writein", "votes"]]


def race_types(n: pd.DataFrame, writein_only: pd.Series, rcv: pd.Series, jungle: pd.Series) -> pd.Series:
    """RACE_TYPES from named-candidate counts per bucket (columns dem, gop, 3p, cross) and the race's kind; a D-R race
    is dr_writein, dr_rcv, dr_jungle, then dr / dr_multi, in that order of precedence."""
    d, r = n["dem"], n["gop"]
    dr = (d > 0) & (r > 0) & (n["cross"] == 0)
    one = np.where(d > 0, "dem", "gop")
    return pd.Series(np.select(
        [n["cross"] > 0, dr & writein_only, dr & rcv, dr & jungle, dr & (d + r == 2), dr, d > 1, r > 1,
         (d + r > 0) & (n["3p"] > 0), d + r > 0],
        ["cross_endorsed", "dr_writein", "dr_rcv", "dr_jungle", "dr", "dr_multi", "dem_vs_dem", "gop_vs_gop",
         np.char.add(one, "_vs_3p"), np.char.add(one, "_uncontested")],
        "no_major_party"), index=n.index)


def build_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(targets_raw.csv, targets.csv, targets_supp.csv, targets_candidates.csv) from read_source's rows, DC excluded.
    The first two have TARGET_COLUMNS, then TARGET_PROVENANCE_COLUMNS, by key; targets_supp.csv has
    TARGET_SUPP_COLUMNS, and targets_candidates.csv TARGET_CANDIDATE_COLUMNS."""
    df = label_contests(df[df["state"] != "DC"])
    if "source" not in df.columns:
        df = df.assign(source=medsl_source(df["version"]))
    cands = candidates(df)
    named = cands[cands["is_candidate"] & (cands["bucket"] != "nonvote")]
    votes = cands.pivot_table(index=SUPP_KEY, columns="bucket", values="votes", aggfunc="sum", fill_value=0)
    votes = votes.reindex(columns=["dem", "gop", "3p", "cross", "nonvote"], fill_value=0)
    n = named.pivot_table(index=SUPP_KEY, columns="bucket", values="votes", aggfunc="size", fill_value=0)
    n = n.reindex(index=votes.index, columns=["dem", "gop", "3p", "cross"], fill_value=0)
    overridden = pd.MultiIndex.from_frame(df[[*KEY, "candidate"]]).isin(list(PARTY_OVERRIDES)) & df["writein"]
    out = df.assign(party_override=overridden).groupby(SUPP_KEY).agg(
        special=("special", "first"), source_totalvotes=("totalvotes", "max"),
        totalvotes_values=("totalvotes", "nunique"), is_unofficial=("unofficial", "any"), source=("source", "first"),
        party_override=("party_override", "any"))

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
    state, year = out.index.get_level_values("state"), out.index.get_level_values("year")
    contest, keys = out.index.get_level_values("contest"), out.index.droplevel("contest")
    out["is_rcv"] = year >= state.map(RCV_FROM).fillna(9999)
    out["is_special_election"] = out["special"].astype(bool)
    # every special in the source is a Texas all-candidate election (1996 and 2006); a runoff never is one
    jungle = (((state == "LA") & year.isin(LA_JUNGLE_YEARS)) | out["special"]) & (contest != "runoff")
    writein_only = named[named["bucket"].isin(["dem", "gop"])].groupby([*SUPP_KEY, "bucket"])["writein"].all() \
        .groupby(SUPP_KEY).any().reindex(out.index, fill_value=False)
    out["race_type"] = race_types(n, writein_only, out["is_rcv"], jungle)
    out["dem_candidate_count"], out["gop_candidate_count"], out["3p_candidate_count"] = n["dem"], n["gop"], n["3p"]
    has_target = out["race_type"].isin(DR_TYPES) & out["margin_2p_norm"].notna()
    out["adj_margin_2p_norm"] = out["margin_2p_norm"].where(has_target)
    out["is_imputed"] = pd.Series(False, index=out.index, dtype=object).where(has_target)
    out["needs_imputation"] = ~has_target
    out["is_nonstandard_race"] = ~out["race_type"].isin(STANDARD_RACE_TYPES)

    # winner: the runoff's winner where the source has the runoff; otherwise the top named candidate, unless tied,
    # cross-endorsed, or a jungle round's leader without a majority (then only a same-party top two settles it)
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

    # on a first round: the key's other contests in the source
    other_contests = pd.Series(contest + "_in_source", index=keys)[contest != "first_round"].groupby(KEY).agg(";".join)
    rcv_not_first = pd.MultiIndex.from_tuples(RCV_FIRST_ROUND_NOT_IN_SOURCE, names=KEY)
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
        "rcv_first_round_not_in_source": keys.isin(rcv_not_first) & (contest == "first_round"),
        "official_results_correction": out["source"].str.contains("Louisiana SOS").to_numpy()
                                       & (contest == "first_round"),
    }
    parts = pd.DataFrame({name: mask if name == "other_contests" else np.where(mask, name, None)
                          for name, mask in flags.items()}, index=out.index)
    out["quality_flags"] = parts.apply(lambda row: ";".join(row.dropna()) or None, axis=1)
    out["nonvote_votes"] = votes["nonvote"]

    # a cross-endorsed candidate's larger major-party line (D or R; blank on a tie or with two such candidates)
    lines = df.assign(d=df["candidatevotes"].where(df["party"].isin(DEM_LABELS), 0),
                      r=df["candidatevotes"].where(df["party"].isin(GOP_LABELS), 0))
    lines = lines.groupby([*SUPP_KEY, "candidate"])[["d", "r"]].sum()
    cross = cands[cands["bucket"] == "cross"].join(lines, on=[*SUPP_KEY, "candidate"]) \
        .drop_duplicates(SUPP_KEY, keep=False).set_index(SUPP_KEY)
    out["cross_line"] = pd.Series(np.select([cross["d"] > cross["r"], cross["r"] > cross["d"]], ["D", "R"], ""),
                                  index=cross.index).replace("", None)

    out = out.reset_index()
    for c in ["dem_votes", "gop_votes", "3p_votes", "total_votes", "margin_votes"]:
        out[c] = out[c].astype("Int64")
    first = out[out["contest"] == "first_round"]
    raw = first.reset_index(drop=True).reindex(columns=TARGET_COLUMNS + TARGET_PROVENANCE_COLUMNS)
    for c in ["dem_votes_imputed", "gop_votes_imputed"]:
        raw[c] = raw[c].astype("Int64")
    raw["imputation_method"] = raw["imputation_method"].astype(object)

    # targets.csv: targets_raw.csv from FIRST_YEAR, with the jungle and RCV targets left to imputation and each
    # cross-endorsed candidate counted for their larger line
    recent = (raw["year"] >= FIRST_YEAR).to_numpy()
    targets, cross_line = raw[recent].copy(), first["cross_line"].to_numpy()[recent]
    impute = targets["race_type"].isin(IMPUTE_RACE_TYPES)
    targets.loc[impute, ["adj_margin_2p_norm", "is_imputed"]] = None
    targets.loc[impute, "needs_imputation"] = True
    assign = (targets["race_type"] == "cross_endorsed") & pd.notna(cross_line) \
        & (targets["dem_candidate_count"] + targets["gop_candidate_count"] == 0)
    side = pd.Series(cross_line, index=targets.index).map({"D": "dem", "R": "gop"})
    targets.loc[assign, "race_type"] = side + np.where(targets["3p_candidate_count"] > 0, "_vs_3p", "_uncontested")
    for prefix in ["dem", "gop"]:
        targets.loc[assign & (side == prefix), f"{prefix}_candidate_count"] += 1
    unresolved = assign & targets["quality_flags"].fillna("").str.contains("winner_party_unresolved")
    targets.loc[unresolved, "winner_party"] = pd.Series(cross_line, index=targets.index)
    targets.loc[assign, "quality_flags"] = targets.loc[assign, "quality_flags"].fillna("").str.split(";").map(
        lambda f: ";".join([x for x in f if x and x != "winner_party_unresolved"] + ["cross_endorsed_assigned"]))
    targets = targets.reset_index(drop=True)

    # targets_supp.csv, from FIRST_YEAR: every contest of the keys with more than one, or a jungle or RCV first round
    impute_keys = pd.MultiIndex.from_frame(first.loc[first["race_type"].isin(IMPUTE_RACE_TYPES), KEY])
    in_supp = out.duplicated(KEY, keep=False) | pd.MultiIndex.from_frame(out[KEY]).isin(impute_keys)
    supp = out[in_supp & (out["year"] >= FIRST_YEAR)].reset_index(drop=True)[TARGET_SUPP_COLUMNS]

    # targets_candidates.csv: the line-summed candidates, and aggregate and non-vote rows, of targets_supp.csv's
    # contests
    supp_keys = pd.MultiIndex.from_frame(supp[SUPP_KEY])
    lines = df[pd.MultiIndex.from_frame(df[SUPP_KEY]).isin(supp_keys)].groupby([*SUPP_KEY, "candidate"]).agg(
        parties=("party", lambda p: "; ".join(sorted(p.dropna().unique())) or None), source=("source", "first"))
    candidate_rows = cands[pd.MultiIndex.from_frame(cands[SUPP_KEY]).isin(supp_keys)] \
        .join(lines, on=[*SUPP_KEY, "candidate"]).rename(columns={"writein": "is_writein"})
    candidate_rows["party_group"] = candidate_rows["bucket"].map(
        {"dem": "D", "gop": "R", "3p": "3P", "cross": "D+R", "nonvote": "nonvote"})
    candidate_rows = candidate_rows.sort_values([*SUPP_KEY, "votes"], ascending=[True] * len(SUPP_KEY) + [False])
    return raw, targets, supp, candidate_rows.reset_index(drop=True)[TARGET_CANDIDATE_COLUMNS]


def check_coverage(out: pd.DataFrame, district_counts: dict) -> list[str]:
    """Blocking problems: a year without 435 districts, or a year whose per-state counts differ from the reference."""
    problems = []
    for year, rows in out.groupby("year"):
        if len(rows) != SEATS:
            problems.append(f"{year}: {len(rows)} districts, expected {SEATS}")
        if str(year) in district_counts:
            got = rows.groupby("state").size().to_dict()
            bad = {s: (got.get(s, 0), n) for s, n in district_counts[str(year)].items() if got.get(s, 0) != n}
            if bad:
                problems.append(f"{year}: district counts (got, expected) differ: {bad}")
    return problems


def report(out: pd.DataFrame, targets: pd.DataFrame, supp: pd.DataFrame, source: pd.DataFrame) -> str:
    target = out["adj_margin_2p_norm"]
    flags = out["quality_flags"].dropna().str.split(";").explode().str.replace(r"\(.*", "", regex=True)
    lines = [
        (f"{len(out)} rows, {out['year'].min()}-{out['year'].max()} ({out['year'].nunique()} elections); "
         f"{(source['state'] == 'DC').sum()} DC source rows excluded"),
        f"adj_margin_2p_norm set on {target.notna().sum()} rows, blank on {target.isna().sum()}",
        "race_type: " + ", ".join(f"{k} {v}" for k, v in out["race_type"].value_counts().items()),
        "quality_flags: " + ", ".join(f"{k} {v}" for k, v in flags.value_counts().items()),
        (f"needs_imputation {int(out['needs_imputation'].sum())} "
         f"({int(out.loc[out['year'] >= FIRST_YEAR, 'needs_imputation'].sum())} from {FIRST_YEAR}), "
         f"is_nonstandard_race {int(out['is_nonstandard_race'].sum())} "
         f"({int(out.loc[out['year'] >= FIRST_YEAR, 'is_nonstandard_race'].sum())} from {FIRST_YEAR})"),
        (f"is_rcv {int(out['is_rcv'].sum())}, is_special_election {int(out['is_special_election'].sum())}, "
         f"winner_party blank {out['winner_party'].isna().sum()}"),
        "Blank target by year: " + ", ".join(f"{y} {n}" for y, n in target.isna().groupby(out["year"]).sum().items()),
        (f"targets.csv: {len(targets)} rows from {FIRST_YEAR}; target set on "
         f"{targets['adj_margin_2p_norm'].notna().sum()}, "
         f"needs_imputation {int(targets['needs_imputation'].sum())} "
         f"(of which {int(targets['race_type'].isin(IMPUTE_RACE_TYPES).sum())} jungle / RCV), cross-endorsed assigned "
         f"{int(targets['quality_flags'].fillna('').str.contains('cross_endorsed_assigned').sum())}"),
        (f"targets_supp.csv (from {FIRST_YEAR}): {len(supp)} contests of {len(supp.drop_duplicates(KEY))} keys ("
         + ", ".join(f"{k} {v}" for k, v in supp["contest"].value_counts().items()) + ")"),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=SOURCE_PATH, type=Path)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    source = read_source(args.source)
    raw, targets, supp, candidates_out = build_targets(source)
    validations = {"targets_raw.csv": validate_schema(raw, schema=TARGETS_SCHEMA),
                   "targets.csv": validate_schema(targets, schema=TARGETS_SCHEMA),
                   "targets_supp.csv": validate_schema(supp, schema=TARGETS_SUPP_SCHEMA, key=SUPP_KEY),
                   "targets_candidates.csv": validate_schema(candidates_out, schema=TARGETS_CANDIDATES_SCHEMA,
                                                             key=CANDIDATE_KEY)}
    problems = check_coverage(raw, json.loads(DISTRICT_COUNTS_PATH.read_text()))
    print(report(raw, targets, supp, source))
    print(f"targets_candidates.csv: {len(candidates_out)} rows for {len(supp)} contests")
    for name, validation in validations.items():
        print(f"{name} schema check (nulls in placeholder and blank-target fields are expected):\n"
              + validation.summary())
    for validation in validations.values():
        validation.raise_if_errors()
    if problems:
        raise ValueError("Coverage problems:\n" + "\n".join(problems))
    for name, frame in [("targets_raw.csv", raw), ("targets.csv", targets), ("targets_supp.csv", supp),
                        ("targets_candidates.csv", candidates_out)]:
        acs.write_csv_atomic(frame, args.output_dir / name)
        print(f"-> {args.output_dir / name}")


if __name__ == "__main__":
    main()
