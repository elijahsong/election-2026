"""Senate general-election polls: the individual polls (polls_clean.csv) and one polling average per race
(polls.csv), in data/processed/senate/. Run after incumbency.py and fec_finance.py.

Source: data/raw/All-US-Senate-polls-2006-2024/2006_to_2024_senate_polls.csv (Jack Whitcomb's scrape of
RealClearPolling; see that folder's SOURCE.md). It has no race id, only year and state, so in the ten state-years with
two Senate races its polls can't be told apart. But the scrape kept RealClearPolling's page order: each run of
consecutive rows with the same year and state is one RCP table (rcp_table below). Each table in a two-race
state-year is assigned to its race by hand in TABLE_CLASSES, from Wikipedia's polling tables.

polls_clean.csv: one row per poll question
  year, state, class, special, election_id  the race (senate_incumbency.csv); year is the election cycle (GA's
                      January 2021 runoff polls are 2020); election_id is "<year>_<state>_<class>"
  stage               general, or runoff for polls that ended after Election Day (LA 2014, MS 2018 special,
                      GA 2020 class 2)
  rcp_table           1, 2, ... the race's RCP tables in file order. A later table is usually RCP's "with
                      third-party candidate" version of the same race, and repeats the first table's polls
  pollster            the source label without RCP's * markers, with HTML entities decoded and POLLSTER_ALIASES
                      applied (spelling variants of one firm only; sponsor/firm labels stay distinct)
  pollster_raw        the source label; pollster_partisan D or R where the label carries "(D)" or "(R)"
  start_date, end_date  fieldwork, ISO dates; DATE_FIXES corrects seven typos. The source has no release date
  sample_size, sample_type  sample_type upper-cased (LV, RV, A; one AV)
  dem_candidate, gop_candidate, ind_candidate  the race's nominees (senate_fec.csv: dem_name, gop_name, ind_name).
                      The source names no candidates; its polls are between the nominees (RCP's general-election
                      tables). ind_candidate is the independent who is the Republican's main opponent (Orman,
                      McMullin, Osborn)
  matchup             which shares the poll reports: dem-gop, dem-gop-other, gop-other, dem-other, dem or gop
  dem_pct, gop_pct, other_pct  the source's dem, rep, other, except where an independent counts as the Democrat
                      (INDEPENDENTS_AS_DEM, as in targets.csv): the independent is dem_pct and the Democratic nominee
                      other_pct. other_pct is blank unless RCP listed a third candidate
  margin_raw          dem_pct - gop_pct
  poll_margin_dem     the Democratic-positive two-party margin, 100 * (dem_pct - gop_pct) / (dem_pct + gop_pct);
                      blank where either share is blank, and for AK 2022 (alaska_four_candidates)
  is_duplicate, duplicate_of  the poll repeats an earlier row (duplicate_of is its source_line): the same pollster and
                      field dates in another RCP table of the race, or the same dates, sample size and D and R shares
                      under another pollster label (one poll published twice)
  quality_flags       independent_counted_as_dem, date_fixed, sample_fields_shifted, alaska_four_candidates
  source_line         the row's line number in the raw file (header = line 1)

polls.csv: one row per race in senate_incumbency.csv, 2006-2026 (aggregate_polls)
  election_date, cutoff_date  Election Day and the forecast cutoff, the day before (data/reference/
                      election_dates.json, as the generic ballot's gb_cutoff_date), or --as-of if earlier
  window_start        Election Day - WINDOW_DAYS. Eligible polls: general stage, not is_duplicate, with a
                      poll_margin_dem, and an end_date from window_start to the day before Election Day that is
                      available by the cutoff (end_date + PUBLICATION_LAG_DAYS <= cutoff_date)
  poll_margin_dem     the simple mean of the eligible polls' poll_margin_dem; blank without eligible polls (never 0)
  n_polls, n_pollsters, latest_poll_date  the eligible polls, their distinct pollster labels, the latest end_date
  poll_available      n_polls > 0, or a published average (2026)
  poll_source         RCP polls (2006-2024), or the FiftyPlusOne average for 2026 (add_published_averages:
                      data/reference/polls_2026_fiftyplusone.csv, transcribed by hand on 2026-11-02; n_polls,
                      n_pollsters and latest_poll_date are blank there). A different averaging method from the
                      historical simple mean
  quality_flags       the race's nominee exceptions (race_table): independent_vs_gop, no_dem_nominee,
                      no_gop_nominee, independent_counted_as_dem
"""

import argparse
import html
import json
import sys
from pathlib import Path

import pandas as pd

from senate_schema import KEY, OUTPUT_DIR, POLLS, POLLS_CLEAN, RACE_ID, write_outputs
from senate_targets import CAUCUS_DEMOCRATS

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/All-US-Senate-polls-2006-2024/2006_to_2024_senate_polls.csv"
INCUMBENCY_PATH = ROOT / "data/interim/senate_incumbency.csv"
NOMINEES_PATH = ROOT / "data/processed/senate_fec.csv"
ELECTION_DATES_PATH = ROOT / "data/reference/election_dates.json"
# 2026: FiftyPlusOne's published Senate polling averages, transcribed by hand on election eve (the RCP source ends in
# 2024). A race not listed had no FiftyPlusOne average.
AVERAGES_2026_PATH = ROOT / "data/reference/polls_2026_fiftyplusone.csv"
RCP_SOURCE = "RCP polls, simple mean of the window"
FIFTYPLUSONE_SOURCE = "FiftyPlusOne average (data/reference/polls_2026_fiftyplusone.csv, manual entry)"

# The polling average: polls whose fieldwork ended in the WINDOW_DAYS days before Election Day (Election Day itself
# excluded). The source has no release dates, so a poll counts as public PUBLICATION_LAG_DAYS after it ended.
WINDOW_DAYS = 30
PUBLICATION_LAG_DAYS = 0

STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new-hampshire": "NH", "new-jersey": "NJ",
    "new-mexico": "NM", "new-york": "NY", "north-carolina": "NC", "north-dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode-island": "RI", "south-carolina": "SC",
    "south-dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA",
    "washington": "WA", "west-virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}

# The race of each RCP table in a state-year with two races, keyed by (year, state, the table's first source_line).
# Matched against the races' Wikipedia polling tables (pollster, dates, R and D shares), checked 2026-11-02.
TABLE_CLASSES = {
    # Schumer-Townsend (regular). Siena 10/27-30 (64-32) and Rasmussen 10/19 (59-31) match exactly; every poll has
    # Schumer-like D 54-67, while Gillibrand polled 48-57. Gillibrand-DioGuardi isn't in the source.
    (2010, "NY", 1408): 3,
    (2014, "OK", 2128): 2,  # Inhofe-Silverstein (regular): 4 of 4 checked polls match exactly
    (2014, "OK", 2152): 3,  # Lankford-Johnson (special): 3 of 4 exact, the 4th 1 point off
    # Graham-Hutto, Ravenel in other (regular): 3 of 4 match; Scott-Dickerson (special) isn't in the source
    (2014, "SC", 2803): 2,
    (2018, "MN", 3303): 1,  # Klobuchar-Newberger (regular): polls D+24 on average, result D+24
    (2018, "MN", 3598): 2,  # Smith-Housley (special): polls D+9, result D+11
    (2018, "MS", 3589): 1,  # Wicker-Baria (regular): NBC/Marist 10/13-18 60-32 matches exactly
    # Hyde-Smith-Espy (special): head-to-head questions, 3 of 3 exact; the 11/19-21 poll is the runoff
    (2018, "MS", 3652): 2,
    (2020, "GA", 3796): 2,  # Perdue-Ossoff (regular), November: 3 of 4 checked match exactly
    (2020, "GA", 4015): 2,  # Perdue-Ossoff runoff: 12 of 12 match (only 2 also match Loeffler-Warnock)
    (2022, "OK", 4137): 3,  # Lankford-Madison Horn (regular): 5 of 5 exact; Mullin-Kendra Horn isn't in the source
    (2024, "NE", 4897): 1,  # Fischer-Osborn (regular, Osborn in other); Ricketts-Love isn't in the source
}

# Tables left out, keyed like TABLE_CLASSES.
DROPPED_TABLES = {
    # rep is blank and dem 4-16%: a minor candidate's shares from a multi-candidate table, in the dem column
    (2006, "MD", 346): "minor candidate in the dem column",
    # rep 33-35, dem (Costello) 7-9, other 1-3: King, counted as the Democrat, isn't in the table
    (2024, "ME", 4905): "King missing",
}

# Independents counted as the Democrat, as in targets.csv: the source has them in other and the Democratic nominee
# (if any) in dem, so the two are swapped. Gross (AK 2020) was the Democratic nominee, an independent on the ballot.
INDEPENDENTS_AS_DEM = {(year, state) for year, state, _ in CAUCUS_DEMOCRATS} | {(2020, "AK")}

# Field dates where start is after end: source_line -> (source start, source end, corrected start, corrected end),
# from the races' Wikipedia polling tables (same pollster, shares and sample).
DATE_FIXES = {
    472: ("2007/10/05", "2007/01/07", "2007/10/05", "2007/10/07"),  # NM 2008 SurveyUSA
    2840: ("2016/08/19", "2016/08/14", "2016/08/19", "2016/08/21"),  # UT 2016 PPP
    3369: ("2018/03/22", "2017/03/29", "2018/03/22", "2018/03/29"),  # TN 2018 MTSU
    3792: ("2020/07/18", "2020/06/24", "2020/07/18", "2020/07/24"),  # ME 2020 Colby College
    3867: ("2020/07/18", "2019/07/20", "2020/07/18", "2020/07/20"),  # MI 2020 FOX News
    4230: ("2022/10/04", "2022/10/03", "2022/10/02", "2022/10/04"),  # IA 2022 Emerson
    4400: ("2024/10/30", "2024/10/02", "2024/10/30", "2024/11/02"),  # AZ 2024 The Hill/Emerson
}
# NY 2010 New York Times: the sample type is in sample_size and sample_type is blank
SHIFTED_SAMPLE_LINE = 1415

# Spelling variants of one pollster's label (after removing * and decoding HTML entities). Sponsor/firm labels
# such as "NBC News/Marist" and "Marist" are left distinct.
POLLSTER_ALIASES = {
    "Rasmussen": "Rasmussen Reports", "Gravis Marketing": "Gravis", "Suffolk University": "Suffolk",
    "University of New Hampshire": "UNH", "Insider Advantage": "InsiderAdvantage", "Gonzales Res.": "Gonzales Research",
    "Benenson Strat (D)": "Benenson Strategy (D)", "Benenson Strat. (D)": "Benenson Strategy (D)",
    "Magellan (R)": "Magellan Strategies (R)", "Marquette University": "Marquette", "Monmouth University": "Monmouth",
    "Loras College": "Loras", "Elon University": "Elon", "Pan Atlantic SMS": "Pan Atlantic",
    "Middle Tn. State U.": "Middle Tn. State Univ.", "Mercyhurst University": "Mercyhurst",
    "Rutgers-Eagleton Poll": "Rutgers-Eagleton", "Remington Research Group (R)": "Remington Research (R)",
    "Ivan Moore Res.": "Ivan Moore Research", "Associated Press-GfK": "Associated Press/GfK",
    "Quinnipiac/WSJ/WP.com": "Quinnipiac/WSJ/WP", "FL Chamber of Commerce": "FL Chamber", "American Res. Group": "ARG",
    "Univ. of Cinci": "University of Cincinnati", "CNN/ORC": "CNN/Opinion Research",
}

# AK 2022 (RCV): Murkowski and Tshibaka are both Republicans, so rep is one of them and the D-R margin isn't the
# race's margin; the margins are left blank
ALASKA_FOUR_CANDIDATES = {(2022, "AK")}


def add_flag(df: pd.DataFrame, mask: pd.Series, flag: str) -> None:
    df.loc[mask, "quality_flags"] = df.loc[mask, "quality_flags"].map(lambda f: f"{f};{flag}" if f else flag)


def fix_rows(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["quality_flags"] = ""
    for line, (start, end, new_start, new_end) in DATE_FIXES.items():
        if (df.at[line, "start_date"], df.at[line, "end_date"]) != (start, end):
            raise ValueError(f"DATE_FIXES line {line}: source dates are not {start}-{end}")
        df.loc[line, ["start_date", "end_date"]] = [new_start, new_end]
    add_flag(df, df.index.isin(DATE_FIXES), "date_fixed")
    if df.at[SHIFTED_SAMPLE_LINE, "sample_size"] != "LV" or pd.notna(df.at[SHIFTED_SAMPLE_LINE, "sample_type"]):
        raise ValueError(f"line {SHIFTED_SAMPLE_LINE}: sample fields are no longer shifted")
    df.loc[SHIFTED_SAMPLE_LINE, ["sample_size", "sample_type"]] = [None, "LV"]
    add_flag(df, df.index == SHIFTED_SAMPLE_LINE, "sample_fields_shifted")
    return df


def assign_races(df: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Each RCP table's class: the state-year's only race, or TABLE_CLASSES where it has two. Rows of
    DROPPED_TABLES are left out."""
    df["state"] = df["state"].map(STATES)
    new_table = (df["state"] != df["state"].shift()) | (df["year"] != df["year"].shift())
    df["table_line"] = df.index.to_series().where(new_table).ffill().astype(int)
    tables = list(df.drop_duplicates("table_line")[["year", "state", "table_line"]].itertuples(index=False,
                                                                                               name=None))
    if not set(TABLE_CLASSES) | set(DROPPED_TABLES) <= set(tables):
        raise ValueError(f"TABLE_CLASSES or DROPPED_TABLES entries match no table: "
                         f"{sorted(set(TABLE_CLASSES) | set(DROPPED_TABLES) - set(tables))}")

    classes = races.groupby(["year", "state"])["class"].agg(list)
    table_class = {}
    for year, state, line in tables:
        if (year, state, line) in DROPPED_TABLES:
            continue
        if (year, state) not in classes:
            raise ValueError(f"Polls for {year} {state}, which has no race in senate_incumbency.csv")
        if len(classes[(year, state)]) == 1:
            table_class[line] = classes[(year, state)][0]
        elif TABLE_CLASSES.get((year, state, line)) in classes[(year, state)]:
            table_class[line] = TABLE_CLASSES[(year, state, line)]
        else:
            raise ValueError(f"Table at line {line} ({year} {state}, two races) has no class in TABLE_CLASSES")
    df = df[df["table_line"].isin(table_class)].copy()
    df["class"] = df["table_line"].map(table_class)
    return df


def race_table(races: pd.DataFrame, nominees: pd.DataFrame, election_dates: dict) -> pd.DataFrame:
    """Every race in senate_incumbency.csv with its ids, Election Day, the forecast cutoff (the day before), its
    nominees and the nominee exceptions, which polls.csv and expert-ratings.csv share:
      independent_vs_gop         an independent is the Republican's main opponent and there's no Democrat (Orman,
                                 McMullin, Osborn): a non-Republican win isn't a Democratic win, and the polls have
                                 no D share
      no_dem_nominee             no Democrat, and senate_fec.csv names no independent opponent: AR 2020
                                 (Libertarian), ID and SD 2026 (independents Achilles, Bengs). As with
                                 independent_vs_gop, a non-Republican win isn't a Democratic win
      no_gop_nominee             no Republican (AR 2008, CA 2016 and 2018)
      independent_counted_as_dem King or Sanders, counted as the Democrat as in targets.csv
    The nominees are known before the election; nothing here comes from the result."""
    out = races[["year", "state", "class", "special"]].copy()
    out["special"] = out["special"].astype(bool)
    out["election_id"] = out["year"].astype(str) + "_" + out["state"] + "_" + out["class"].astype(str)
    out["election_date"] = out["year"].map(lambda y: election_dates[str(y)]["election_day"])
    out["cutoff_date"] = out["year"].map(lambda y: election_dates[str(y)]["day_before"])
    names = nominees[[*KEY, "dem_name", "gop_name", "ind_name", "dem_is_independent"]].rename(
        columns={"dem_name": "dem_candidate", "gop_name": "gop_candidate", "ind_name": "ind_candidate"})
    out = out.merge(names, on=KEY, how="left", validate="one_to_one", indicator=True)
    if (out["_merge"] != "both").any():
        raise ValueError(f"Races missing from senate_fec.csv: {out.loc[out['_merge'] != 'both', KEY].values.tolist()}")
    flags = pd.DataFrame({
        "independent_vs_gop": out["ind_candidate"].notna() & out["dem_candidate"].isna(),
        "no_dem_nominee": out["dem_candidate"].isna() & out["ind_candidate"].isna(),
        "no_gop_nominee": out["gop_candidate"].isna(),
        "independent_counted_as_dem": out["dem_is_independent"].astype(bool),
    })
    out["quality_flags"] = flags.apply(lambda row: ";".join(row.index[row]) or None, axis=1)
    return out.drop(columns=["_merge", "dem_is_independent"])


def standardize_pollster(label: pd.Series) -> pd.Series:
    clean = label.map(html.unescape).str.replace("*", "", regex=False).str.split().str.join(" ")
    return clean.replace(POLLSTER_ALIASES)


def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """is_duplicate and duplicate_of (the first row's source_line): the same poll (race, stage, pollster and field
    dates) in a later RCP table, or the same race, stage, dates, sample size and D and R shares under another
    pollster label. The first row in rcp_table, then file, order is kept."""
    df = df.sort_values([*KEY, "stage", "rcp_table", "source_line"])
    same_poll = [*KEY, "stage", "pollster", "start_date", "end_date"]
    same_numbers = [*KEY, "stage", "start_date", "end_date", "sample_size", "dem_pct", "gop_pct"]
    first_by_poll = df.groupby(same_poll, dropna=False)["source_line"].transform("first")
    first_by_numbers = df.groupby(same_numbers, dropna=True)["source_line"].transform("first")
    duplicate_of = first_by_poll.where(first_by_poll != df["source_line"])
    by_numbers = first_by_numbers.where(first_by_numbers != df["source_line"])
    df["duplicate_of"] = duplicate_of.fillna(by_numbers).astype("Int64")
    df["is_duplicate"] = df["duplicate_of"].notna()
    return df


def build_clean(raw: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """polls_clean.csv from the raw file (line-number index) and race_table's rows."""
    df = assign_races(fix_rows(raw), races)
    df["source_line"] = df.index
    df["start_date"] = pd.to_datetime(df["start_date"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d")
    df["end_date"] = pd.to_datetime(df["end_date"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d")
    df = df.merge(races, on=KEY, how="left", validate="many_to_one", suffixes=("", "_race")).set_index(df.index)
    df["stage"] = (df["end_date"] > df["election_date"]).map({True: "runoff", False: "general"})
    df["sample_size"] = pd.to_numeric(df["sample_size"]).astype("Int64")
    df["sample_type"] = df["sample_type"].str.upper()
    df["pollster_raw"] = df["pollster"]
    df["pollster"] = standardize_pollster(df["pollster_raw"])
    df["pollster_partisan"] = df["pollster_raw"].str.extract(r"\((D|R)\)", expand=False)

    df = df.rename(columns={"dem": "dem_pct", "rep": "gop_pct", "other": "other_pct"})
    swap = pd.Series(list(zip(df["year"], df["state"])), index=df.index).isin(INDEPENDENTS_AS_DEM)
    df.loc[swap, ["dem_pct", "other_pct"]] = df.loc[swap, ["other_pct", "dem_pct"]].to_numpy()
    add_flag(df, swap, "independent_counted_as_dem")
    shares = df[["dem_pct", "gop_pct", "other_pct"]].notna()
    df["matchup"] = shares.apply(lambda row: "-".join(n for n, has in zip(["dem", "gop", "other"], row) if has), axis=1)

    df["margin_raw"] = df["dem_pct"] - df["gop_pct"]
    df["poll_margin_dem"] = 100 * df["margin_raw"] / (df["dem_pct"] + df["gop_pct"])
    alaska = pd.Series(list(zip(df["year"], df["state"])), index=df.index).isin(ALASKA_FOUR_CANDIDATES)
    df.loc[alaska, "poll_margin_dem"] = None
    add_flag(df, alaska, "alaska_four_candidates")

    df["rcp_table"] = df.groupby([*KEY, "stage"])["table_line"].rank(method="dense").astype(int)
    df = mark_duplicates(df)
    df["quality_flags"] = df["quality_flags"].replace("", None)
    return df.sort_values([*KEY, "stage", "source_line"])[list(POLLS_CLEAN)].reset_index(drop=True)


def aggregate_polls(clean: pd.DataFrame, races: pd.DataFrame, as_of: str | None = None,
                    window_days: int = WINDOW_DAYS, publication_lag_days: int = PUBLICATION_LAG_DAYS) -> pd.DataFrame:
    """polls.csv: each race's simple mean poll_margin_dem over its eligible polls (see the module docstring). Races
    without eligible polls keep a blank margin, n_polls 0 and poll_available False. as_of (ISO date) moves the cutoff
    earlier, for a cycle in progress; a poll is never used before it ended plus publication_lag_days."""
    out = races[[*RACE_ID, "election_date", "cutoff_date", "quality_flags"]].copy()
    if as_of is not None:
        out["cutoff_date"] = out["cutoff_date"].where(out["cutoff_date"] <= as_of, as_of)
    election_day = pd.to_datetime(out["election_date"])
    out["window_start"] = (election_day - pd.Timedelta(days=window_days)).dt.strftime("%Y-%m-%d")

    polls = clean.merge(out[[*KEY, "election_date", "cutoff_date", "window_start"]], on=KEY, validate="many_to_one")
    available = (pd.to_datetime(polls["end_date"]) + pd.Timedelta(days=publication_lag_days)).dt.strftime("%Y-%m-%d")
    eligible = polls[(polls["stage"] == "general") & ~polls["is_duplicate"] & polls["poll_margin_dem"].notna()
                     & (polls["end_date"] >= polls["window_start"]) & (polls["end_date"] < polls["election_date"])
                     & (available <= polls["cutoff_date"])]
    means = eligible.groupby(KEY).agg(poll_margin_dem=("poll_margin_dem", "mean"), n_polls=("poll_margin_dem", "size"),
                                      n_pollsters=("pollster", "nunique"), latest_poll_date=("end_date", "max"))
    out = out.merge(means.reset_index(), on=KEY, how="left", validate="one_to_one")
    out[["n_polls", "n_pollsters"]] = out[["n_polls", "n_pollsters"]].fillna(0).astype("Int64")
    out["poll_available"] = out["n_polls"] > 0
    out["poll_source"] = out["poll_available"].map({True: RCP_SOURCE, False: None})
    return out.sort_values(KEY, ignore_index=True)[list(POLLS)]


def add_published_averages(polls: pd.DataFrame, averages: pd.DataFrame, year: int) -> pd.DataFrame:
    """polls.csv with `year`'s races set from published averages (state, class, dem_pct, gop_pct, as_of_date):
    poll_margin_dem = 100 * (dem - gop) / (dem + gop), poll_available True, n_polls and n_pollsters blank (unknown),
    latest_poll_date blank. An average dated after the race's cutoff_date isn't used. The year must have no RCP
    polls, so the two sources never mix within a year."""
    averages = averages.assign(year=year)
    out = polls.set_index(KEY)
    rows = averages.set_index(KEY)
    unknown = rows.index.difference(out.index)
    if len(unknown) or rows.index.duplicated().any():
        raise ValueError(f"Averages for unknown or repeated races: {list(unknown)}")
    if out.loc[out.index.get_level_values("year") == year, "poll_available"].any():
        raise ValueError(f"{year} already has RCP polls; don't mix sources within a year")
    rows = rows[rows["as_of_date"] <= out.loc[rows.index, "cutoff_date"]]
    margin = 100 * (rows["dem_pct"] - rows["gop_pct"]) / (rows["dem_pct"] + rows["gop_pct"])
    out.loc[rows.index, "poll_margin_dem"] = margin
    out.loc[rows.index, ["n_polls", "n_pollsters"]] = pd.NA
    out.loc[rows.index, "poll_available"] = True
    out.loc[rows.index, "poll_source"] = FIFTYPLUSONE_SOURCE
    return out.reset_index()[list(POLLS)]


def report(clean: pd.DataFrame, polls: pd.DataFrame, raw: pd.DataFrame) -> None:
    print(f"{len(raw)} source rows -> {len(clean)} polls ({len(raw) - len(clean)} in dropped tables: "
          f"{sorted(DROPPED_TABLES.values())})")
    print("stage: " + str(clean["stage"].value_counts().to_dict()))
    print(f"duplicates: {int(clean['is_duplicate'].sum())}; blank poll_margin_dem: "
          f"{int(clean['poll_margin_dem'].isna().sum())}, in "
          + ", ".join(f"{y} {s}" for y, s in clean.loc[clean["poll_margin_dem"].isna(), ["year", "state"]]
                      .drop_duplicates().itertuples(index=False)))
    print("flags: " + str(clean["quality_flags"].str.split(";").explode().value_counts().to_dict()))
    print(f"polls.csv: races with a {WINDOW_DAYS}-day average, by year:")
    by_year = polls.groupby("year").agg(races=("poll_available", "size"), with_polls=("poll_available", "sum"),
                                        mean_n_polls=("n_polls", lambda n: round(float(n[n > 0].mean()), 1)
                                                      if (n > 0).any() else None))
    print(by_year.to_string())
    missing = polls[~polls["poll_available"] & (polls["year"] < 2026)]
    print(f"without eligible polls, 2006-2024 ({len(missing)}): "
          + ", ".join(missing["election_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    parser.add_argument("--as-of", help="ISO date: use only polls public by then (for a cycle in progress)")
    args = parser.parse_args()

    raw = pd.read_csv(RAW_PATH, dtype={"sample_size": str})
    raw.index = raw.index + 2  # file line numbers
    races = race_table(pd.read_csv(INCUMBENCY_PATH), pd.read_csv(NOMINEES_PATH),
                       json.loads(ELECTION_DATES_PATH.read_text()))
    clean = build_clean(raw, races.drop(columns=["quality_flags"]))
    polls = add_published_averages(aggregate_polls(clean, races, as_of=args.as_of), pd.read_csv(AVERAGES_2026_PATH),
                                   year=2026)
    report(clean, polls, raw)
    write_outputs({"polls_clean.csv": clean, "polls.csv": polls}, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
