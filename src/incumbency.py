"""Build House and Senate incumbency (2006-2026) into house_incumbency.csv and senate_incumbency.csv.

inc        1 if an incumbent is running in the general election, 0 for an open seat
inc_signed +1 for a running Democratic incumbent, -1 for a running Republican, 0 for an open seat, and blank (NA) when
           that isn't a single major party and no override settles it

2006-2024: Cha, Kuriwaki & Snyder, "Candidates in American General Elections" (CAGE), Harvard Dataverse,
           doi:10.7910/DVN/DGDRDT, v4.1 (CC0). Its `inc` codes each general-election candidate: 0 not an incumbent,
           1 regular, 2 won a special election, 3 appointed. Codes 1 and 2 count as incumbents; an appointed incumbent
           (3, Senate only) never won the seat, so that race is coded as open.
2026:      House: Cook Political Report's 2026 PVI list (data/raw/pvi.sources/cook_pvi_public_2026.csv), which names each
           2026 district's incumbent and party, or "OPEN (...)", on the 2026 lines.
           Senate: data/raw/incumbency_2026/senate_2026.csv, one row per race: state, class, special (0/1), inc (0/1,
           unsigned), and party (the running incumbent's D/R/I; blank for an open seat). The sign comes from party.
"""

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

import acs

ROOT = Path(__file__).resolve().parents[1]
CAGE_URL = "https://dataverse.harvard.edu/api/access/datafile/13461690"  # candidates_2006-2024.tab
CAGE_SHA256 = "abf13d31e8ab3c8ba897a003e9f59865175e1fb0a95b1da35134f9699ddd56f5"
CAGE_PATH = ROOT / "data/raw/cage/candidates_2006-2024.tab"  # download cache (gitignored)
CAGE_SOURCE = "CAGE v4.1 (Cha, Kuriwaki & Snyder), doi:10.7910/DVN/DGDRDT"
COOK_2026_PATH = ROOT / "data/raw/pvi.sources/cook_pvi_public_2026.csv"
COOK_SOURCE = "Cook Political Report 2026 PVI list (incumbent / OPEN), https://www.cookpolitical.com/cook-pvi"
SENATE_2026_PATH = ROOT / "data/raw/incumbency_2026/senate_2026.csv"
SENATE_2026_SOURCE = "data/raw/incumbency_2026/senate_2026.csv"

HISTORICAL_YEARS = list(range(2006, 2025, 2))
YEAR_2026 = 2026
SIGN = {"D": 1, "R": -1}
ELECTED_INC = {1, 2}  # CAGE inc codes that count as a running incumbent: regular, and won a special election
APPOINTED_INC = 3  # appointed and never elected to the seat: coded as open
FIELDS = ["inc", "inc_signed", "incumbent_party", "incumbent_names", "status", "source"]
HOUSE_KEY = ["year", "state", "district"]
SENATE_KEY = ["year", "state", "class"]
HOUSE_COLUMNS = HOUSE_KEY + FIELDS
SENATE_COLUMNS = SENATE_KEY + ["special"] + FIELDS

# key -> (inc, inc_signed, incumbent_names, reason): corrections to what the sources code, applied after all years are
# combined. House keys are (year, state, district).
HOUSE_OVERRIDES = {
    (2024, "TX", 34): (1, 1, "GONZALEZ, VICENTE",
                       "CAGE codes Mayra Flores (R) as an incumbent, but she lost TX-34 in 2022; Gonzalez (D) is"),
    (2006, "MI", 7): (0, 0, "", "incumbent Schwarz (R) lost the primary and ran as a write-in; the R nominee wasn't the incumbent"),
    (2016, "KS", 1): (0, 0, "", "incumbent Huelskamp (R) lost the primary and ran as a write-in; the R nominee wasn't the incumbent"),
    # incumbents of both parties in one redrawn district, resolved by hand
    (2012, "IA", 3): (1, -1, "LATHAM, THOMAS P. (TOM)", "Boswell (D) vs. Latham (R) after redistricting; coded as an R incumbent"),
    (2012, "OH", 16): (1, -1, "RENACCI, JAMES B. (JIM)", "Sutton (D) vs. Renacci (R) after redistricting; coded as an R incumbent"),
    (2018, "PA", 17): (0, 0, "", "Lamb (D) vs. Rothfus (R) on the court-drawn 2018 map; coded as open"),
    (2022, "FL", 2): (0, 0, "", "Lawson (D) vs. Dunn (R) after redistricting; coded as open"),
    (2022, "TX", 34): (0, 0, "", "Gonzalez (D) vs. Flores (R, won the 2022 special) after redistricting; coded as open"),
    # Cook lists Kiley as an independent
    (2026, "CA", 6): (1, -1, "Kevin Kiley", "Cook lists Kiley as an independent; coded as an R incumbent"),
}

# Senate keys are (year, state, class). Independents and non-nominee incumbents are coded by caucus or original party.
SENATE_OVERRIDES = {
    (2012, "ME", 1): (0, 0, "", "CAGE codes King as an incumbent, but Snowe (R) retired and King was first elected in 2012"),
    (2006, "CT", 1): (1, 1, "LIEBERMAN, JOSEPH I.",
                      "D incumbent who lost the primary and won as Connecticut for Lieberman; coded by his original party"),
    (2010, "AK", 3): (1, -1, "MURKOWSKI, LISA ANN",
                      "R incumbent who lost the primary and won as a write-in; coded by her original party"),
    **{(year, "VT", 1): (1, 1, "SANDERS, BERNARD (BERNIE)", "independent who caucuses with Democrats")
       for year in (2012, 2018, 2024)},
    (2024, "ME", 1): (1, 1, "KING, ANGUS S., JR.", "independent who caucuses with Democrats"),
    # appointed incumbents are open (as CAGE's inc = 3 is for 2006-2024); senate_2026.csv doesn't mark appointments
    (2026, "OH", 3): (0, 0, "", "special for Vance's seat; the appointed incumbent (Husted) is coded as open"),
    (2026, "FL", 3): (0, 0, "", "special for Rubio's seat; the appointed incumbent (Moody) is coded as open"),
}

# Senate seat class from the year the seat is next up: class I is up in 2024, 2030, ...; II in 2026; III in 2028.
CLASS_BY_NEXTUP_MOD_6 = {2: 1, 4: 2, 0: 3}
CLASS_SIZES = {1: 33, 2: 33, 3: 34}
# (year, state) -> corrected `nextup`: CAGE codes LA's 2014 runoff (Landrieu vs. Cassidy, class II) as next up in 2016
NEXTUP_FIXES = {(2014, "LA"): 2020}
# specials that weren't on the November ballot: Massachusetts filled Kennedy's seat in January 2010
NON_NOVEMBER_SPECIALS = {(2010, "MA", 1)}


def download_cage(path: Path = CAGE_PATH) -> Path:
    """Cached path to the CAGE candidates file, downloading it on first use; raises if its checksum isn't the pinned one."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with urllib.request.urlopen(CAGE_URL, timeout=300) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        os.replace(tmp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CAGE_SHA256:
        raise ValueError(f"{path} sha256 {digest} does not match the pinned {CAGE_SHA256}; delete it to re-download")
    return path


def classify(incumbents: pd.DataFrame) -> tuple[int, object, str, str]:
    """(inc, inc_signed, incumbent_party, status) for one race, from its incumbent candidates' rows."""
    if incumbents.empty:
        return 0, 0, "", "open"
    parties = set(incumbents["party"])
    if len(parties) == 1 and parties <= set(SIGN):
        party = parties.pop()
        return 1, SIGN[party], party, "incumbent"
    if parties <= set(SIGN):  # one Democratic and one Republican incumbent, e.g. after redistricting
        return 1, pd.NA, "D; R", "review: incumbents of both parties"
    return 1, pd.NA, "; ".join(sorted(parties)), "review: non-major-party incumbent"


def race_rows(candidates: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """One classified row per race (unique key) from CAGE candidate rows."""
    rows = []
    for values, race in candidates.groupby(key):
        incumbents = race[race["inc"].isin(ELECTED_INC)]
        inc, signed, party, status = classify(incumbents)
        appointed = race.loc[race["inc"] == APPOINTED_INC, "name_snyder"]
        if inc == 0 and len(appointed):
            status = f"open: appointed incumbent {'; '.join(sorted(appointed))}"
        rows.append({**dict(zip(key, values)), "inc": inc, "inc_signed": signed, "incumbent_party": party,
                     "incumbent_names": "; ".join(sorted(incumbents["name_snyder"])), "status": status,
                     "source": CAGE_SOURCE})
    return pd.DataFrame(rows)


def apply_overrides(df: pd.DataFrame, overrides: dict, key: list[str]) -> pd.DataFrame:
    """Replace the rows named in overrides, raising if any override doesn't match exactly one row."""
    out = df.copy()
    for values, (inc, signed, names, reason) in overrides.items():
        hit = pd.concat([out[col] == v for col, v in zip(key, values)], axis=1).all(axis=1)
        if hit.sum() != 1:
            raise ValueError(f"Override {values} matches {hit.sum()} rows")
        party = {1: "D", -1: "R", 0: ""}[signed]
        out.loc[hit, ["inc", "inc_signed", "incumbent_party", "incumbent_names", "status"]] = [
            inc, signed, party, names, f"override: {reason}"]
    return out


def finish(df: pd.DataFrame, columns: list[str], key: list[str]) -> pd.DataFrame:
    """Typed, checked and sorted output: inc is 0/1, inc_signed a nullable int, and both agree on which seats are open."""
    out = df.copy()
    out["inc"] = out["inc"].astype(int)
    out["inc_signed"] = pd.array(out["inc_signed"], dtype="Int64")
    if ((out["inc"] == 0) != (out["inc_signed"] == 0).fillna(False)).any():
        raise ValueError("inc and inc_signed disagree on which seats are open")
    if out.duplicated(key).any():
        raise ValueError(f"Duplicate {key} keys")
    return out[columns].sort_values(key, ignore_index=True)


# ---------------------------------------------------------------- House

def expected_districts(counts: dict[str, int]) -> set[tuple[str, int]]:
    """Every (state, district) for one year's seat counts, with at-large states as district 0."""
    return {(state, d) for state, n in counts.items() for d in ([0] if n == 1 else range(1, n + 1))}


def check_districts(df: pd.DataFrame, year: int, counts: dict[str, int]) -> None:
    """Raise unless one year's rows are exactly its districts, once each."""
    found = list(zip(df["state"], df["district"]))
    if len(found) != len(set(found)) or set(found) != expected_districts(counts):
        missing, extra = expected_districts(counts) - set(found), set(found) - expected_districts(counts)
        raise ValueError(f"{year} districts don't match the seat counts: missing {sorted(missing)[:5]}, extra {sorted(extra)[:5]}")


def build_house_historical(cage: pd.DataFrame, counts_by_year: dict) -> pd.DataFrame:
    """One row per district for each historical year, from CAGE's House general-election candidates."""
    house = cage[(cage["office"] == "H") & (cage["type"] == "G") & cage["year"].isin(HISTORICAL_YEARS)].copy()
    if house["inc"].isna().any() or house["dist"].isna().any():
        raise ValueError("CAGE House rows with a missing incumbency code or district")
    house["year"] = house["year"].astype(int)
    house["district"] = house["dist"].astype(int)
    years = []
    for year in HISTORICAL_YEARS:
        counts = acs.normalize_counts(counts_by_year, year)
        races = house[house["year"] == year].copy()
        # CAGE numbers at-large seats 1; the repo convention is 0
        at_large = races["state"].map(counts) == 1
        if not (races.loc[at_large, "district"] == 1).all():
            raise ValueError(f"{year}: at-large races not numbered 1 in CAGE")
        races.loc[at_large, "district"] = 0
        year_rows = race_rows(races, HOUSE_KEY)
        check_districts(year_rows, year, counts)
        years.append(year_rows)
    return pd.concat(years, ignore_index=True)


def read_cook_2026(path: Path = COOK_2026_PATH) -> pd.DataFrame:
    """Cook's 2026 list as (state, district, incumbent, party); its header names five columns but rows carry a sixth (rank)."""
    cook = pd.read_csv(path, header=0, names=["Dist", "Incumbent", "Party", "2025 PVI", "2026 PVI", "Rank"],
                       dtype=str, index_col=False)
    dist = cook["Dist"].str.extract(r"^([A-Z]{2})-(AL|\d{2})$")
    if dist.isna().any().any():
        raise ValueError(f"Unexpected Dist values: {cook.loc[dist.isna().any(axis=1), 'Dist'].tolist()}")
    return pd.DataFrame({"state": dist[0], "district": dist[1].replace("AL", "0").astype(int),
                         "incumbent": cook["Incumbent"].str.strip(), "party": cook["Party"].str.strip()})


def rows_2026(table: pd.DataFrame, is_open: pd.Series, key: dict, names: pd.Series, source: str) -> pd.DataFrame:
    """2026 rows from a list of races: open seats are 0, else an incumbent runs, signed by the listed party."""
    unknown = set(table.loc[~is_open, "party"]) - set(SIGN) - {"I"}
    if unknown:
        raise ValueError(f"Unexpected 2026 incumbent parties: {sorted(unknown)}")
    out = pd.DataFrame({
        "year": YEAR_2026, **key,
        "inc": (~is_open).astype(int),
        "inc_signed": table["party"].map(SIGN).astype("Int64").where(~is_open, 0),
        "incumbent_party": table["party"].where(~is_open, ""),
        "incumbent_names": names.where(~is_open, ""),
        "status": "incumbent",
        "source": source,
    })
    out.loc[is_open, "status"] = "open"
    out.loc[~is_open & out["inc_signed"].isna(), "status"] = "review: non-major-party incumbent"
    return out


def build_house_2026(cook: pd.DataFrame, counts_by_year: dict) -> pd.DataFrame:
    """2026 rows from Cook's list: "OPEN (...)" is an open seat (its Party is the departing member's)."""
    out = rows_2026(cook, cook["incumbent"].str.startswith("OPEN"),
                    {"state": cook["state"], "district": cook["district"]}, cook["incumbent"], COOK_SOURCE)
    check_districts(out, YEAR_2026, acs.normalize_counts(counts_by_year, YEAR_2026))
    return out


def build_house(cage: pd.DataFrame, cook: pd.DataFrame, counts_by_year: dict) -> pd.DataFrame:
    """All House years in the house_incumbency.csv schema."""
    combined = pd.concat([build_house_historical(cage, counts_by_year), build_house_2026(cook, counts_by_year)],
                         ignore_index=True)
    return finish(apply_overrides(combined, HOUSE_OVERRIDES, HOUSE_KEY), HOUSE_COLUMNS, HOUSE_KEY)


# ---------------------------------------------------------------- Senate

def regular_class(year: int) -> int:
    """The Senate class whose regular election falls in `year` (class I in 2024, II in 2026, III in 2028)."""
    return CLASS_BY_NEXTUP_MOD_6[(year + 6) % 6]


def senate_candidates(cage: pd.DataFrame) -> pd.DataFrame:
    """CAGE Senate candidates in November general and special elections, with each race's seat class.

    Drops non-November specials and same-day specials for a seat's last weeks (nextup == year, e.g. CA 2022 and
    2024), which repeat that seat's full-term race.
    """
    senate = cage[(cage["office"] == "S") & cage["year"].isin(HISTORICAL_YEARS)].copy()
    if senate["inc"].isna().any() or senate["nextup"].isna().any():
        raise ValueError("CAGE Senate rows with a missing incumbency code or next-up year")
    senate["year"] = senate["year"].astype(int)
    senate["nextup"] = senate["nextup"].astype(int)
    for (year, state), nextup in NEXTUP_FIXES.items():
        hit = (senate["year"] == year) & (senate["state"] == state)
        if not hit.any():
            raise ValueError(f"NEXTUP_FIXES {(year, state)} matches no rows")
        senate.loc[hit, "nextup"] = nextup
    senate = senate[~((senate["type"] == "S") & (senate["nextup"] == senate["year"]))].copy()
    senate["class"] = (senate["nextup"] % 6).map(CLASS_BY_NEXTUP_MOD_6)
    senate["special"] = (senate["type"] == "S").astype(int)
    non_november = pd.Series(list(zip(senate["year"], senate["state"], senate["class"])), index=senate.index)
    return senate[~non_november.isin(NON_NOVEMBER_SPECIALS)]


def check_senate(races: pd.DataFrame) -> None:
    """Raise unless each class has the same, right-sized set of states in every cycle, every state sits in exactly two
    classes, and each year's regular races are its own class (specials are for other classes' seats)."""
    regular = races[races["special"] == 0]
    if (regular["class"] != regular["year"].map(regular_class)).any():
        raise ValueError(f"Regular races in the wrong class:\n{regular[regular['class'] != regular['year'].map(regular_class)]}")
    if (races.loc[races["special"] == 1, "class"] == races.loc[races["special"] == 1, "year"].map(regular_class)).any():
        raise ValueError("A special election shares its year's regular class")
    members = {}
    for (cls, year), states in regular.groupby(["class", "year"])["state"]:
        members.setdefault(cls, {})[year] = frozenset(states)
    for cls, by_year in members.items():
        if len(set(by_year.values())) != 1 or len(next(iter(by_year.values()))) != CLASS_SIZES[cls]:
            raise ValueError(f"Class {cls} states differ across cycles or aren't {CLASS_SIZES[cls]}: "
                             f"{ {y: len(s) for y, s in by_year.items()} }")
    seats = pd.Series([s for states in members.values() for s in next(iter(states.values()))]).value_counts()
    if set(seats) != {2} or set(seats.index) != set(acs.STATE_FIPS.values()):
        raise ValueError("Every state must sit in exactly two Senate classes")


def build_senate_historical(cage: pd.DataFrame) -> pd.DataFrame:
    """One row per Senate race (year, state, class) for each historical year."""
    candidates = senate_candidates(cage)
    out = race_rows(candidates, SENATE_KEY + ["special"])
    check_senate(out)
    return out


def read_senate_2026(path: Path = SENATE_2026_PATH) -> pd.DataFrame:
    """The 2026 Senate races: state, class, special (0/1), inc (0/1, unsigned), party (blank when inc is 0).

    Raises unless every running incumbent (inc 1) has a party and no open seat (inc 0) does.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: one row per 2026 Senate race with columns state, class, special, inc, party")
    races = pd.read_csv(path, dtype={"state": str, "party": str})
    if list(races.columns) != ["state", "class", "special", "inc", "party"]:
        raise ValueError(f"{path} columns {list(races.columns)}, expected state, class, special, inc, party")
    races["party"] = races["party"].fillna("").str.strip()
    if not races["inc"].isin([0, 1]).all() or not races["special"].isin([0, 1]).all():
        raise ValueError(f"{path}: inc and special must be 0 or 1")
    mismatch = (races["inc"] == 1) != (races["party"] != "")
    if mismatch.any():
        raise ValueError(f"{path}: party must be given exactly when inc is 1:\n{races[mismatch]}")
    return races


def build_senate_2026(races: pd.DataFrame, classes: dict[int, frozenset]) -> pd.DataFrame:
    """2026 Senate rows; the regular races must be exactly class II's states, as in CAGE's earlier cycles."""
    regular = races[races["special"] == 0]
    if set(regular["class"]) != {regular_class(YEAR_2026)} or frozenset(regular["state"]) != classes[regular_class(YEAR_2026)]:
        raise ValueError("2026 regular races must be exactly the class II states")
    out = rows_2026(races, races["inc"] == 0,
                    {"state": races["state"], "class": races["class"], "special": races["special"]},
                    pd.Series("", index=races.index), SENATE_2026_SOURCE)  # the file has no names
    return out


def build_senate(cage: pd.DataFrame, races_2026: pd.DataFrame) -> pd.DataFrame:
    """All Senate years in the senate_incumbency.csv schema."""
    historical = build_senate_historical(cage)
    regular = historical[historical["special"] == 0]
    classes = {cls: frozenset(g["state"]) for cls, g in regular.groupby("class")}
    combined = pd.concat([historical, build_senate_2026(races_2026, classes)], ignore_index=True)
    check_senate(combined)  # 2026's class II states and specials must agree with every earlier cycle
    return finish(apply_overrides(combined, SENATE_OVERRIDES, SENATE_KEY), SENATE_COLUMNS, SENATE_KEY)


def summarize(df: pd.DataFrame, name: str) -> str:
    counts = df.groupby("year")["inc_signed"].value_counts(dropna=False).unstack(fill_value=0)
    review = df[df["status"].str.startswith("review")]
    return f"{name}: {len(df)} rows\n{counts}\nNeeds review: {len(review)}\n{review.to_string() if len(review) else ''}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cage", type=Path, help="local CAGE candidates file (default: download and cache it)")
    parser.add_argument("--counts", default=ROOT / "data/reference/district_counts.json", type=Path)
    parser.add_argument("--house-output", default=ROOT / "data/interim/house_incumbency.csv", type=Path)
    parser.add_argument("--senate-output", default=ROOT / "data/interim/senate_incumbency.csv", type=Path)
    args = parser.parse_args()
    with open(args.counts) as f:
        counts_by_year = json.load(f)
    cage = pd.read_csv(args.cage or download_cage(), sep="\t", low_memory=False)

    house = build_house(cage, read_cook_2026(), counts_by_year)
    acs.write_csv_atomic(house, args.house_output)
    print(summarize(house, f"House -> {args.house_output}"))
    senate = build_senate(cage, read_senate_2026())  # raises, after the House file is written, if 2026 is missing
    acs.write_csv_atomic(senate, args.senate_output)
    print(summarize(senate, f"Senate -> {args.senate_output}"))


if __name__ == "__main__":
    main()
