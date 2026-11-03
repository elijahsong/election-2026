"""State and county presidential PVI for Senate years 2006-2026, into data/processed/senate/.

  state_pvi.csv   year, state (50 states)
  county_pvi.csv  year, state, county_id, county_fips (counties of the 48 contiguous states; one At-Large row for
                  each of AK and HI)

For Senate year `year`, prev and anteprev are the two latest presidential elections strictly before it (so 2008 uses
2004 and 2000: the target year's own presidential result is never used). With dem_2p_norm the geography's two-party
Democratic share and national_dem_2p_norm the existing national baseline (data/interim/presidential.csv
dem_pct_2p_norm, read as is):

  rel_dem_2p(p)  = dem_2p_norm(geography, p) - national_dem_2p_norm(p)
  pvi            = 0.75 * rel_dem_2p(prev) + 0.25 * rel_dem_2p(anteprev)
  partisan_lean  = 2 * pvi

All in percentage points, positive = more Democratic than the nation. This is the House formula
(data-processing/house/03b-pvi-calculation.ipynb). States and counties use the same national baseline.

Inputs are pres_state_votes.csv and pres_county_votes.csv (presidential.py). A county gets a row for a year only when
it has a result, under the same county_fips, in both presidential years. One present in only one of them (VA's
Bedford city after 2013, SD's Shannon County renumbered as Oglala Lakota) is reported and left out; no code or result
is invented. Alaska (no counties in the source) and Hawaii each get a single At-Large county-equivalent (county_id and
county_name "At-Large", county_fips 02000 / 15000: synthetic state-aggregate codes, not county FIPS) whose PVI is the
statewide PVI.
"""

import argparse
from pathlib import Path

import pandas as pd

from senate_schema import AT_LARGE, COUNTY_PVI, OUTPUT_DIR, STATE_PVI, read_output, write_outputs

ROOT = Path(__file__).resolve().parents[2]
NATIONAL_PATH = ROOT / "data/interim/presidential.csv"
YEARS = range(2006, 2027, 2)
WEIGHTS = (0.75, 0.25)  # (prev, anteprev) presidential election


def prior_pres_years(year: int, pres_years) -> tuple[int, int]:
    """The two latest presidential years strictly before `year`, latest first."""
    earlier = sorted(p for p in pres_years if p < year)
    if len(earlier) < 2:
        raise ValueError(f"Fewer than two presidential elections before {year}")
    return earlier[-1], earlier[-2]


def read_national(path: Path = NATIONAL_PATH) -> pd.Series:
    """The national two-party Democratic share by presidential year, exactly as stored."""
    national = pd.read_csv(path)
    return national.set_index("pres_year")["dem_pct_2p_norm"]


def build_pvi(results: pd.DataFrame, national: pd.Series, geo: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """PVI per year in YEARS and geography (geo columns), and the geographies left out because they lack a result in
    one of the two presidential years."""
    results = results.assign(rel=results["dem_2p_norm"] - results["year"].map(national))
    if results["rel"].isna().any():
        raise ValueError("Presidential years missing from the national baseline")
    frames, dropped = [], []
    for year in YEARS:
        prev, ante = prior_pres_years(year, national.index)
        a = results[results["year"] == prev].set_index(geo)
        b = results[results["year"] == ante].set_index(geo)
        both = a.index.intersection(b.index)
        dropped += [f"{year}: {key} only in {prev if key in a.index else ante}"
                    for key in a.index.symmetric_difference(b.index)]
        out = pd.DataFrame(index=both).reset_index()
        out["year"], out["prev_pres_year"], out["anteprev_pres_year"] = year, prev, ante
        out["prev_rel_dem_2p"] = a.loc[both, "rel"].to_numpy()
        out["anteprev_rel_dem_2p"] = b.loc[both, "rel"].to_numpy()
        if "county_name" in a.columns:
            out["county_name"] = a.loc[both, "county_name"].to_numpy()
        frames.append(out)
    out = pd.concat(frames, ignore_index=True)
    out["pvi"] = WEIGHTS[0] * out["prev_rel_dem_2p"] + WEIGHTS[1] * out["anteprev_rel_dem_2p"]
    out["partisan_lean"] = 2 * out["pvi"]
    return out, dropped


def at_large_rows(state_pvi: pd.DataFrame) -> pd.DataFrame:
    """One At-Large county-equivalent per year for each AT_LARGE state, with its statewide PVI."""
    rows = state_pvi[state_pvi["state"].isin(AT_LARGE)].copy()
    rows["county_fips"] = rows["state"].map(AT_LARGE)
    rows["county_id"] = rows["county_name"] = "At-Large"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    national = read_national()
    state, dropped_states = build_pvi(read_output("pres_state_votes.csv", args.output_dir), national, ["state"])
    if dropped_states or (state.groupby("year").size() != 50).any():
        raise ValueError(f"Every year must have all 50 states: {dropped_states}")
    state = state.sort_values(["year", "state"], ignore_index=True)[list(STATE_PVI)]

    pres_county = read_output("pres_county_votes.csv", args.output_dir)
    pres_county = pres_county[~pres_county["state"].isin(AT_LARGE)]  # HI's counties too: AK/HI are At-Large
    county, dropped = build_pvi(pres_county, national, ["state", "county_id", "county_fips"])
    county["is_at_large"] = False
    county = pd.concat([county, at_large_rows(state).assign(is_at_large=True)], ignore_index=True)
    county = county.sort_values(["year", "state", "county_fips"], ignore_index=True)[list(COUNTY_PVI)]

    print(f"state_pvi: {len(state)} rows, {state['year'].min()}-{state['year'].max()}; "
          f"pvi range {state['pvi'].min():+.1f} to {state['pvi'].max():+.1f}")
    print(f"county_pvi: {len(county)} rows; per year {county.groupby('year').size().to_dict()}")
    print(f"Counties left out (no result under the same FIPS in one of the two presidential years): {len(dropped)}")
    for line in dropped:
        print("  " + line)
    write_outputs({"state_pvi.csv": state, "county_pvi.csv": county}, args.output_dir)


if __name__ == "__main__":
    main()
