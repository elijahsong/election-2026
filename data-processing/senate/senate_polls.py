"""Senate general-election polls, one row per poll question, keyed to the race (year, state, class). Run after
incumbency.py.

Source: data/raw/All-US-Senate-polls-2006-2024/2006_to_2024_senate_polls.csv (Jack Whitcomb's scrape of
RealClearPolling; see that folder's SOURCE.md). It has no race id, only year and state, so in the ten state-years with
two Senate races its polls can't be told apart. But the scrape kept RealClearPolling's page order: each run of
consecutive rows with the same year and state is one RCP table (rcp_table below). Each table in a two-race
state-year is assigned to its race by hand in TABLE_CLASSES, from Wikipedia's polling tables.

polls.csv (data/processed/senate/)
  year, state, class  the race (senate_incumbency.csv). year is the election cycle (GA's January 2021 runoff polls
                      are 2020)
  stage               general, or runoff for polls that ended after Election Day (LA 2014, MS 2018 special,
                      GA 2020 class 2)
  rcp_table           1, 2, ... the race's RCP tables in file order. A later table is usually RCP's "with
                      third-party candidate" version of the same race, and repeats some of the first table's polls
  pollster            as in the source (RCP's trailing * markers kept)
  start_date, end_date  ISO dates; DATE_FIXES corrects seven typos
  sample_size, sample_type  sample_type upper-cased (LV, RV, A; one AV)
  dem_pct, gop_pct, other_pct  the source's dem, rep, other, except where an independent counts as the Democrat
                      (INDEPENDENTS_AS_DEM, as in targets.csv): the independent is dem_pct and the Democratic nominee
                      other_pct. other_pct is blank unless RCP listed a third candidate
  margin_raw          dem_pct - gop_pct; blank where either is blank (no D or no R candidate polled)
  margin_2p_norm      100 * (dem_pct - gop_pct) / (dem_pct + gop_pct)
  in_earlier_table    the same poll (pollster without *, start and end dates) is in an earlier table of the race
                      and stage, so counting both would count one poll twice
  quality_flags       independent_counted_as_dem, date_fixed, sample_fields_shifted, alaska_four_candidates
  source_line         the row's line number in the raw file (header = line 1)
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from senate_schema import KEY, OUTPUT_DIR, POLLS, write_outputs
from senate_targets import CAUCUS_DEMOCRATS

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/All-US-Senate-polls-2006-2024/2006_to_2024_senate_polls.csv"
INCUMBENCY_PATH = ROOT / "data/interim/senate_incumbency.csv"

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

# AK 2022 (RCV): Murkowski and Tshibaka are both Republicans, so rep is one of them and the D-R margin isn't the
# race's margin; the margins are left blank
ALASKA_FOUR_CANDIDATES = {(2022, "AK")}


def election_day(year: int) -> dt.date:
    """The Tuesday after the first Monday in November."""
    day = dt.date(year, 11, 2)
    while day.weekday() != 1:
        day += dt.timedelta(days=1)
    return day


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


def build_polls(raw: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    df = assign_races(fix_rows(raw), races)
    df["source_line"] = df.index
    df["start_date"] = pd.to_datetime(df["start_date"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d")
    df["end_date"] = pd.to_datetime(df["end_date"], format="%Y/%m/%d").dt.strftime("%Y-%m-%d")
    df["stage"] = ["runoff" if end > election_day(year).isoformat() else "general"
                   for year, end in zip(df["year"], df["end_date"])]
    df["sample_size"] = pd.to_numeric(df["sample_size"]).astype("Int64")
    df["sample_type"] = df["sample_type"].str.upper()

    df = df.rename(columns={"dem": "dem_pct", "rep": "gop_pct", "other": "other_pct"})
    swap = pd.Series(list(zip(df["year"], df["state"])), index=df.index).isin(INDEPENDENTS_AS_DEM)
    df.loc[swap, ["dem_pct", "other_pct"]] = df.loc[swap, ["other_pct", "dem_pct"]].to_numpy()
    add_flag(df, swap, "independent_counted_as_dem")

    df["margin_raw"] = df["dem_pct"] - df["gop_pct"]
    df["margin_2p_norm"] = 100 * df["margin_raw"] / (df["dem_pct"] + df["gop_pct"])
    alaska = pd.Series(list(zip(df["year"], df["state"])), index=df.index).isin(ALASKA_FOUR_CANDIDATES)
    df.loc[alaska, ["margin_raw", "margin_2p_norm"]] = None
    add_flag(df, alaska, "alaska_four_candidates")

    race = [*KEY, "stage"]
    df["rcp_table"] = df.groupby(race)["table_line"].rank(method="dense").astype(int)
    poll_id = [*race, "pollster_id", "start_date", "end_date"]
    df["pollster_id"] = df["pollster"].str.rstrip("*").str.strip()
    first_table = df.groupby(poll_id)["rcp_table"].transform("min")
    df["in_earlier_table"] = df["rcp_table"] > first_table
    df["quality_flags"] = df["quality_flags"].replace("", None)
    return df.sort_values([*race, "source_line"])[list(POLLS)].reset_index(drop=True)


def report(polls: pd.DataFrame, raw: pd.DataFrame, races: pd.DataFrame) -> None:
    print(f"{len(raw)} source rows -> {len(polls)} polls ({len(raw) - len(polls)} in dropped tables: "
          f"{sorted(DROPPED_TABLES.values())})")
    print("per year: " + str(polls.groupby("year").size().to_dict()))
    print("stage: " + str(polls["stage"].value_counts().to_dict()))
    print(f"in an earlier table of the race: {int(polls['in_earlier_table'].sum())}")
    print(f"blank margin: {int(polls['margin_raw'].isna().sum())}, in "
          + ", ".join(f"{y} {s}" for y, s in polls.loc[polls["margin_raw"].isna(), ["year", "state"]]
                      .drop_duplicates().itertuples(index=False)))
    print("flags: " + str(polls["quality_flags"].str.split(";").explode().value_counts().to_dict()))
    polled = set(zip(polls["year"], polls["state"], polls["class"]))
    cycle = races[races["year"].between(2006, 2024)]
    missing = [f"{y} {s}-{c}" for y, s, c in zip(cycle["year"], cycle["state"], cycle["class"])
               if (y, s, c) not in polled]
    print(f"races 2006-2024 with polls: {len(polled)} of {len(cycle)}; without ({len(missing)}): {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    raw = pd.read_csv(RAW_PATH, dtype={"sample_size": str})
    raw.index = raw.index + 2  # file line numbers
    races = pd.read_csv(INCUMBENCY_PATH)
    polls = build_polls(raw, races)
    report(polls, raw, races)
    write_outputs({"polls.csv": polls}, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
