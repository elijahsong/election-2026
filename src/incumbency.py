"""Build House incumbency (one row per election year x district, 2006-2026) into incumbency.csv.

inc        1 if an incumbent is running in the district's general election, 0 for an open seat
inc_signed +1 for a running Democratic incumbent, -1 for a running Republican, 0 for an open seat, and blank (NA) when
           that isn't a single major party (two incumbents of different parties, or an independent incumbent)

2006-2024: Cha, Kuriwaki & Snyder, "Candidates in American General Elections" (CAGE), Harvard Dataverse,
           doi:10.7910/DVN/DGDRDT, v4.1 (CC0). Its `inc` codes each general-election candidate: 0 not an incumbent,
           1 regular, 2 won a special election, 3 appointed. Any code above 0 counts as an incumbent.
2026:      Cook Political Report's 2026 PVI list (data/raw/pvi.sources/cook_pvi_public_2026.csv), which names each 2026
           district's incumbent and party, or "OPEN (...)", on the 2026 lines.
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

HISTORICAL_YEARS = list(range(2006, 2025, 2))
COOK_YEAR = 2026
SIGN = {"D": 1, "R": -1}
COLUMNS = ["year", "state", "district", "inc", "inc_signed", "incumbent_party", "incumbent_names", "status", "source"]

# (year, state, district) -> (inc, inc_signed, incumbent_names, reason): corrections to what the sources code, applied
# to every year after both sources are combined
OVERRIDES = {
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


def expected_districts(counts: dict[str, int]) -> set[tuple[str, int]]:
    """Every (state, district) for one year's seat counts, with at-large states as district 0."""
    return {(state, d) for state, n in counts.items() for d in ([0] if n == 1 else range(1, n + 1))}


def check_districts(df: pd.DataFrame, year: int, counts: dict[str, int]) -> None:
    """Raise unless one year's rows are exactly its districts, once each."""
    found = list(zip(df["state"], df["district"]))
    if len(found) != len(set(found)) or set(found) != expected_districts(counts):
        missing, extra = expected_districts(counts) - set(found), set(found) - expected_districts(counts)
        raise ValueError(f"{year} districts don't match the seat counts: missing {sorted(missing)[:5]}, extra {sorted(extra)[:5]}")


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


def build_historical(cage: pd.DataFrame, counts_by_year: dict) -> pd.DataFrame:
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
        rows = []
        for (state, district), race in races.groupby(["state", "district"]):
            incumbents = race[race["inc"] > 0]
            inc, signed, party, status = classify(incumbents)
            rows.append({"year": year, "state": state, "district": district, "inc": inc, "inc_signed": signed,
                         "incumbent_party": party, "incumbent_names": "; ".join(sorted(incumbents["name_snyder"])),
                         "status": status, "source": CAGE_SOURCE})
        year_rows = pd.DataFrame(rows)
        check_districts(year_rows, year, counts)
        years.append(year_rows)
    return pd.concat(years, ignore_index=True)


def apply_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the rows named in OVERRIDES, raising if any override doesn't match exactly one row."""
    out = df.copy()
    for (year, state, district), (inc, signed, names, reason) in OVERRIDES.items():
        hit = (out["year"] == year) & (out["state"] == state) & (out["district"] == district)
        if hit.sum() != 1:
            raise ValueError(f"Override {(year, state, district)} matches {hit.sum()} rows")
        party = {1: "D", -1: "R", 0: ""}[signed]
        out.loc[hit, ["inc", "inc_signed", "incumbent_party", "incumbent_names", "status"]] = [
            inc, signed, party, names, f"override: {reason}"]
    return out


def read_cook_2026(path: Path = COOK_2026_PATH) -> pd.DataFrame:
    """Cook's 2026 list as (state, district, incumbent, party); its header names five columns but rows carry a sixth (rank)."""
    cook = pd.read_csv(path, header=0, names=["Dist", "Incumbent", "Party", "2025 PVI", "2026 PVI", "Rank"],
                       dtype=str, index_col=False)
    dist = cook["Dist"].str.extract(r"^([A-Z]{2})-(AL|\d{2})$")
    if dist.isna().any().any():
        raise ValueError(f"Unexpected Dist values: {cook.loc[dist.isna().any(axis=1), 'Dist'].tolist()}")
    return pd.DataFrame({"state": dist[0], "district": dist[1].replace("AL", "0").astype(int),
                         "incumbent": cook["Incumbent"].str.strip(), "party": cook["Party"].str.strip()})


def build_2026(cook: pd.DataFrame, counts_by_year: dict) -> pd.DataFrame:
    """2026 rows from Cook's list: "OPEN (...)" is an open seat (its Party is the departing member's), else the incumbent runs."""
    is_open = cook["incumbent"].str.startswith("OPEN")
    unknown = set(cook.loc[~is_open, "party"]) - set(SIGN) - {"I"}
    if unknown:
        raise ValueError(f"Unexpected Cook 2026 incumbent parties: {sorted(unknown)}")
    signed = cook["party"].map(SIGN).astype("Int64").where(~is_open, 0)
    out = pd.DataFrame({
        "year": COOK_YEAR, "state": cook["state"], "district": cook["district"],
        "inc": (~is_open).astype(int), "inc_signed": signed,
        "incumbent_party": cook["party"].where(~is_open, ""),
        "incumbent_names": cook["incumbent"].where(~is_open, ""),
        "status": "incumbent",
        "source": COOK_SOURCE,
    })
    out.loc[is_open, "status"] = "open"
    out.loc[~is_open & out["inc_signed"].isna(), "status"] = "review: non-major-party incumbent"
    check_districts(out, COOK_YEAR, acs.normalize_counts(counts_by_year, COOK_YEAR))
    return out


def build_incumbency(cage: pd.DataFrame, cook: pd.DataFrame, counts_by_year: dict) -> pd.DataFrame:
    """All years, sorted, in the incumbency.csv schema."""
    out = apply_overrides(pd.concat([build_historical(cage, counts_by_year), build_2026(cook, counts_by_year)],
                                    ignore_index=True))
    out["inc"] = out["inc"].astype(int)
    out["inc_signed"] = pd.array(out["inc_signed"], dtype="Int64")
    if ((out["inc"] == 0) != (out["inc_signed"] == 0).fillna(False)).any():
        raise ValueError("inc and inc_signed disagree on which seats are open")
    return out[COLUMNS].sort_values(["year", "state", "district"], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cage", type=Path, help="local CAGE candidates file (default: download and cache it)")
    parser.add_argument("--counts", default=ROOT / "data/reference/district_counts.json", type=Path)
    parser.add_argument("--output", default=ROOT / "data/interim/incumbency.csv", type=Path)
    args = parser.parse_args()
    with open(args.counts) as f:
        counts_by_year = json.load(f)
    cage = pd.read_csv(args.cage or download_cage(), sep="\t", low_memory=False)
    out = build_incumbency(cage, read_cook_2026(), counts_by_year)
    acs.write_csv_atomic(out, args.output)
    summary = out.groupby("year")["inc_signed"].value_counts(dropna=False).unstack(fill_value=0)
    print(f"{len(out)} rows -> {args.output}\n{summary}\n\nNeeds review:\n"
          f"{out[out['status'].str.startswith('review')][['year', 'state', 'district', 'incumbent_names', 'status']]}")


if __name__ == "__main__":
    main()
