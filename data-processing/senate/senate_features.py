"""Build the Senate feature tables in data/processed/senate/. Run after senate_pvi.py.

basic_f.csv          year, state, class (every race in data/interim/senate_incumbency.csv, 2006-2026):
                     gen_margin_raw  the same year's national generic-ballot margin (data/reference/generic_ballot.csv
                                     margin_raw, as in the House basic_f.csv); not the Senate result. Blank for 2026,
                                     which has no generic-ballot row yet
                     partisan_lean   the state's 2 * pvi (state_pvi.csv)
                     inc_signed      senate_incumbency.csv (incumbency.py): +1 D incumbent, -1 R, 0 open; appointed
                                     incumbents are open, and independents are coded by caucus, as in the House file
prices_f.csv         year, state, class: census_region, price_data_month and the CPI-U / gasoline features of the
                     state's Census region, from data/processed/cpi_gas_region.csv (data-processing/cpi_gas.py), then
                     the state's unemployment features, from data/processed/unemployment_state.csv
                     (data-processing/unemployment.py); both shared with the House prices_f.csv
acs_f.csv            year, state, class: acs_vintage through rel_housing_burden, the state's ACS values
                     (data/interim/senate/acs_state.csv, senate_acs.py)
fec_f.csv            year, state, class: the House fec_f columns from data/processed/senate_fec.csv
                     (data-processing/fec_finance.py), prepared by house_features.fec_features (the same +-5 cap; no
                     Senate race needs a known-gap blank)
nat_political_f.csv  the House nat_political_f.csv, unchanged (it is checked to be identical)
modeling.csv         basic_f, acs_f, fec_f and prices_f joined on year, state, class, then nat_political_f joined on
                     year (house_features.join_districts and add_national, as the House modeling.csv)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from senate_schema import ACS_F, BASIC_F, KEY, OUTPUT_DIR, PRICES_F, read_output, write_outputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cpi_gas
import house_features  # on the path via senate_schema (data-processing/house)
import unemployment

ROOT = Path(__file__).resolve().parents[2]
GENERIC_BALLOT_PATH = ROOT / "data/reference/generic_ballot.csv"
INCUMBENCY_PATH = ROOT / "data/interim/senate_incumbency.csv"
HOUSE_NATIONAL_PATH = ROOT / "data/processed/house/nat_political_f.csv"
PRICES_PATH = ROOT / "data/processed/cpi_gas_region.csv"
UNEMPLOYMENT_PATH = ROOT / "data/processed/unemployment_state.csv"
ACS_PATH = ROOT / "data/interim/senate/acs_state.csv"
FEC_PATH = ROOT / "data/processed/senate_fec.csv"


def basic_features(races: pd.DataFrame, state_pvi: pd.DataFrame, gb: pd.DataFrame) -> pd.DataFrame:
    gen = gb[["election", "margin_raw"]].rename(columns={"election": "year", "margin_raw": "gen_margin_raw"})
    out = races[[*KEY, "inc_signed"]].merge(gen, on="year", how="left", validate="many_to_one") \
        .merge(state_pvi[["year", "state", "partisan_lean"]], on=["year", "state"], how="left",
               validate="many_to_one")
    if out["partisan_lean"].isna().any() or out["inc_signed"].isna().any():
        raise ValueError("Races without a partisan lean or inc_signed")
    out["inc_signed"] = out["inc_signed"].astype("int64")
    return out.sort_values(KEY, ignore_index=True)[list(BASIC_F)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    races = pd.read_csv(INCUMBENCY_PATH)
    basic = basic_features(races, read_output("state_pvi.csv", args.output_dir), pd.read_csv(GENERIC_BALLOT_PATH))
    national = pd.read_csv(HOUSE_NATIONAL_PATH)
    prices = unemployment.attach_states(cpi_gas.attach_regions(basic[KEY], pd.read_csv(PRICES_PATH)),
                                        pd.read_csv(UNEMPLOYMENT_PATH))[list(PRICES_F)]
    acs_f = basic[KEY].merge(pd.read_csv(ACS_PATH), on=["year", "state"], how="left",
                             validate="many_to_one")[list(ACS_F)]
    fec_f = house_features.fec_features(pd.read_csv(FEC_PATH), key=KEY, known_gaps=[])
    modeling = house_features.add_national(
        house_features.join_districts({"basic_f": basic, "acs_f": acs_f, "fec_f": fec_f, "prices_f": prices}, key=KEY),
        national)
    print(f"basic_f: {len(basic)} races, {basic['year'].min()}-{basic['year'].max()}; per year "
          f"{basic.groupby('year').size().to_dict()}; gen_margin_raw blank for "
          f"{sorted(basic.loc[basic['gen_margin_raw'].isna(), 'year'].unique())}")
    print("inc_signed: " + ", ".join(f"{k:+d} {v}" for k, v in basic["inc_signed"].value_counts().sort_index().items()))
    write_outputs({"basic_f.csv": basic, "acs_f.csv": acs_f, "fec_f.csv": fec_f, "prices_f.csv": prices,
                   "nat_political_f.csv": national, "modeling.csv": modeling}, args.output_dir)
    if not read_output("nat_political_f.csv", args.output_dir).equals(national):
        raise ValueError("nat_political_f.csv differs from the House file")


if __name__ == "__main__":
    main()
