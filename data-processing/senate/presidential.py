"""Build state and county presidential results, 2000-2024, with raw votes, into data/processed/senate/.

  pres_state_votes.csv   year, state: dem/gop/3p/total_votes, dem_2p_norm, gop_2p_norm, margin_2p_norm
                         from MEDSL U.S. President 1976-2024 (data/raw/medsl_pres_state)
  pres_county_votes.csv  year, state, county_id, county_fips, county_name and the same vote columns, from MEDSL County
                         Presidential Returns 2000-2024 (data/raw/medsl_pres_county)

The repo had no state or county presidential results, only the national baseline (data/interim/presidential.csv,
which this script doesn't read or change). These two files are the presidential inputs of senate_pvi.py.

Votes:
  State: a nominee's votes are summed across all their ballot lines (e.g. NY's Working Families line), as in the
    House targets; dem_votes is the ballot-listed (not write-in) candidate on the DEMOCRAT line, gop_votes the one on
    the REPUBLICAN line, and every other candidate, write-in and scattering row is 3p. total_votes = dem + gop + 3p.
  County: MEDSL's county file has only D, R, Libertarian, Green and an OTHER row per county, with the party already
    summed. Where a county reports a TOTAL mode row, only the TOTAL rows are used (UT 2020 and nine 2024 states list
    TOTAL beside the by-mode rows, which add up to it); otherwise its modes are summed.
  dem_2p_norm = 100 * D / (D + R), gop_2p_norm = 100 - dem_2p_norm, margin_2p_norm = dem_2p_norm - gop_2p_norm.

County rows:
  county_fips is MEDSL's code zero-padded to five digits, and county_id is the same code. FIPS_RECODES harmonizes one
  county MEDSL codes two ways: Oglala Lakota County, SD (Shannon County until 2015; Census code 46113, then 46102)
  is 46113 in 2000-2012 and 2024 but 46102 in 2016 and 2020. It is 46113 in every year here, as in Heideman's
  county Senate results, so the two join. MEDSL's Kansas City, MO rows (county_fips 2938000 in 2000-2020, 36000 in
  2024: a jurisdiction Missouri tabulates apart from its counties) are added to Jackson County (29095): the Kansas
  City Election Board serves only the part of the city inside Jackson County (data/raw/kansas_city_mo/SOURCE.md).
  In 2000 the Kansas City row has 0 votes and MEDSL's Jackson row already includes the city. Kept: the 48 contiguous
  states and Hawaii. Left out, and counted in the report:
    - Alaska: MEDSL reports it by state house district, coded 02001-02040, which collide with real borough FIPS. Its
      statewide result is in pres_state_votes.csv; senate_pvi.py gives Alaska one At-Large geography.
    - DC, which has no Senate seat.
    - rows without a county: CT "STATEWIDE WRITEIN", ME "MAINE UOCAVA" and RI "FEDERAL PRECINCT".
    - placeholder rows with no D or R votes: Bedford city, VA (51515) in 2016, after it merged into Bedford County.
  County sums are compared with the state file (report only), so source differences stay visible.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from senate_schema import OUTPUT_DIR, PRES_COUNTY, PRES_STATE, write_outputs
import acs

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "data/raw/medsl_pres_state/1976-2024-president.csv"
COUNTY_PATH = ROOT / "data/raw/medsl_pres_county/countypres_2000-2024.csv"
FIRST_YEAR = 2000
STATES = set(acs.STATE_FIPS.values())  # the 50 states; DC isn't in it
COUNTY_EXCLUDED_STATES = {"AK", "DC"}
FIPS_RECODES = {"46102": "46113"}  # Oglala Lakota County, SD, coded as in Heideman's Senate files
JACKSON_COUNTY_MO = "29095"  # gets MEDSL's Kansas City, MO rows


def shares(df: pd.DataFrame) -> pd.DataFrame:
    """df with total_votes and the two-party share columns, from dem/gop/3p_votes."""
    df = df.copy()
    df["total_votes"] = df["dem_votes"] + df["gop_votes"] + df["3p_votes"]
    two_party = (df["dem_votes"] + df["gop_votes"]).astype("float64")
    df["dem_2p_norm"] = 100 * df["dem_votes"].astype("float64") / two_party.where(two_party > 0)
    df["gop_2p_norm"] = 100 - df["dem_2p_norm"]
    df["margin_2p_norm"] = df["dem_2p_norm"] - df["gop_2p_norm"]
    for c in ["dem_votes", "gop_votes", "3p_votes", "total_votes"]:
        df[c] = df[c].astype("Int64")
    return df


def state_votes(path: Path = STATE_PATH) -> pd.DataFrame:
    """One row per year and state (50 states, 2000 on), each nominee's lines summed."""
    df = pd.read_csv(path)
    df = df[(df["year"] >= FIRST_YEAR) & df["state_po"].isin(STATES)].drop(columns="state").rename(columns={"state_po": "state"})
    df["candidate"] = df["candidate"].fillna("(no name)")
    nominee = {}
    for party, column in [("DEMOCRAT", "dem"), ("REPUBLICAN", "gop")]:
        # the nominee is the line's ballot-listed candidate; write-ins on the line (MD 2004, AZ 2016) are 3p
        lines = df[(df["party_simplified"] == party) & ~df["writein"].astype(bool)] \
            .groupby(["year", "state"])["candidate"].unique()
        if (lines.str.len() != 1).any():
            raise ValueError(f"State-years with several {party} candidates: {lines[lines.str.len() != 1].to_dict()}")
        nominee[column] = lines.str[0]
    df = df.join(pd.DataFrame(nominee), on=["year", "state"])
    df["bucket"] = np.select([df["candidate"] == df["dem"], df["candidate"] == df["gop"]], ["dem", "gop"], "3p")
    out = df.pivot_table(index=["year", "state"], columns="bucket", values="candidatevotes", aggfunc="sum",
                         fill_value=0).reindex(columns=["dem", "gop", "3p"], fill_value=0)
    out = out.rename(columns={"dem": "dem_votes", "gop": "gop_votes", "3p": "3p_votes"}).reset_index()
    return shares(out)[list(PRES_STATE)]


def county_votes(path: Path = COUNTY_PATH) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per year and county (2000 on), and the number of source rows (or, for placeholders, counties) left
    out, by reason."""
    df = pd.read_csv(path, dtype={"county_fips": str})
    df = df.drop(columns="state").rename(columns={"state_po": "state"})
    excluded = {
        "Alaska (state house districts) and DC": df["state"].isin(COUNTY_EXCLUDED_STATES),
        "no county (statewide write-in, UOCAVA, federal precinct)": df["county_fips"].isna(),
    }
    drop = np.logical_or.reduce(list(excluded.values()))
    counts = {reason: int(mask.sum()) for reason, mask in excluded.items()}
    df = df[~drop].copy()
    df["county_fips"] = df["county_fips"].str.zfill(5).replace(FIPS_RECODES)
    kansas_city = (df["state"] == "MO") & (df["county_name"] == "KANSAS CITY")
    df.loc[kansas_city, ["county_fips", "county_name"]] = [JACKSON_COUNTY_MO, "JACKSON"]
    has_total = df.groupby(["year", "county_fips"])["mode"].transform(lambda m: (m == "TOTAL").any())
    df = df[(df["mode"] == "TOTAL") | ~has_total]
    df["bucket"] = df["party"].map({"DEMOCRAT": "dem", "REPUBLICAN": "gop"}).fillna("3p")
    out = df.pivot_table(index=["year", "state", "county_fips"], columns="bucket", values="candidatevotes",
                         aggfunc="sum", fill_value=0).reindex(columns=["dem", "gop", "3p"], fill_value=0)
    out = out.rename(columns={"dem": "dem_votes", "gop": "gop_votes", "3p": "3p_votes"}).reset_index()
    names = df.drop_duplicates(["year", "county_fips"]).set_index(["year", "county_fips"])["county_name"]
    out["county_name"] = names.reindex(pd.MultiIndex.from_frame(out[["year", "county_fips"]])).to_numpy()
    out["county_id"] = out["county_fips"]
    no_votes = out["dem_votes"] + out["gop_votes"] == 0
    counts["placeholder county with no D or R votes"] = int(no_votes.sum())
    out = shares(out[~no_votes])
    return out[list(PRES_COUNTY)].sort_values(["year", "state", "county_fips"], ignore_index=True), counts


def compare(state: pd.DataFrame, county: pd.DataFrame) -> pd.DataFrame:
    """Per year and state with counties: county sums minus the state file, for D, R and total votes."""
    sums = county.groupby(["year", "state"])[["dem_votes", "gop_votes", "total_votes"]].sum()
    diff = (sums - state.set_index(["year", "state"])[["dem_votes", "gop_votes", "total_votes"]]).dropna()
    return diff[(diff != 0).any(axis=1)].reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    state = state_votes()
    county, excluded = county_votes()
    if (state.groupby("year").size() != 50).any():
        raise ValueError(f"State rows per year: {state.groupby('year').size().to_dict()}")
    print(f"pres_state_votes: {len(state)} rows, {state['year'].min()}-{state['year'].max()}")
    print(f"pres_county_votes: {len(county)} rows; counties per year "
          f"{county.groupby('year').size().to_dict()}")
    print("Source rows left out: " + "; ".join(f"{k} {v}" for k, v in excluded.items()))
    diff = compare(state, county)
    print(f"County sums differing from the state file (votes, county minus state): {len(diff)} state-years; largest "
          "by |D| difference:")
    print(diff.reindex(diff["dem_votes"].abs().sort_values(ascending=False).index).head(15).to_string(index=False))
    write_outputs({"pres_state_votes.csv": state, "pres_county_votes.csv": county}, args.output_dir)


if __name__ == "__main__":
    main()
