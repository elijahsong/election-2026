"""Build the House feature tables and the joined modeling table in data/processed/house/.

basic_f.csv          year, state, district: gen_margin_raw (generic_ballot.csv margin_raw, election = year),
                     partisan_lean (house_partisan_lean.csv), inc_signed (house_incumbency.csv)
nat_political_f.csv  year: every generic_ballot.csv field prefixed gb_, with election -> year and
                     dem / gop / und_estimate -> gb_dem_pct / gb_gop_pct / gb_und_pct
acs_f.csv            year, state, district: acs_vintage through rel_state_housing_burden from acs_supplement.csv
modeling.csv         basic_f and acs_f joined on year, state, district, then nat_political_f joined on year

District-level tables must have unique keys and exactly the same key set, and no column may appear in two of them.
A year with no generic-ballot row (2026, until one is added) keeps blank generic-ballot fields; nothing is filled in.
"""

import argparse
from pathlib import Path

import pandas as pd

import acs

ROOT = Path(__file__).resolve().parents[2]
GENERIC_BALLOT_PATH = ROOT / "data/reference/generic_ballot.csv"
PARTISAN_LEAN_PATH = ROOT / "data/interim/house/house_partisan_lean.csv"
INCUMBENCY_PATH = ROOT / "data/interim/house/house_incumbency.csv"
ACS_PATH = ROOT / "data/interim/house/acs_supplement.csv"
OUTPUT_DIR = ROOT / "data/processed/house"

KEY = ["year", "state", "district"]
GB_RENAMES = {"election": "year", "dem_estimate": "gb_dem_pct", "gop_estimate": "gb_gop_pct", "und_estimate": "gb_und_pct"}
ACS_FIELDS = ("acs_vintage", "rel_state_housing_burden")  # first and last acs_supplement.csv column kept


def check_unique(df: pd.DataFrame, key: list[str], name: str) -> None:
    dup = df[df.duplicated(key, keep=False)]
    if len(dup):
        raise ValueError(f"{name}: {len(dup)} rows share a key, e.g. {dup[key].head(3).values.tolist()}")


def join_districts(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join district-level tables on KEY; they must cover exactly the same keys and share no other column."""
    keys = {name: set(df[KEY].itertuples(index=False, name=None)) for name, df in frames.items()}
    union = set().union(*keys.values())
    for name, df in frames.items():
        check_unique(df, KEY, name)
        missing = sorted(union - keys[name])
        if missing:
            raise ValueError(f"{name} lacks {len(missing)} keys the other tables have, e.g. {missing[:3]}")
    columns = [c for df in frames.values() for c in df.columns if c not in KEY]
    shared = sorted({c for c in columns if columns.count(c) > 1})
    if shared:
        raise ValueError(f"Columns in more than one table: {shared}")
    out, *rest = frames.values()
    for df in rest:
        out = out.merge(df, on=KEY, validate="one_to_one")
    return out.sort_values(KEY, ignore_index=True)


def add_national(df: pd.DataFrame, national: pd.DataFrame) -> pd.DataFrame:
    """df plus the year-level columns of national; years national lacks keep NA."""
    check_unique(national, ["year"], "national")
    return df.merge(national, on="year", how="left", validate="many_to_one")


def nat_political_features(gb: pd.DataFrame) -> pd.DataFrame:
    out = gb.rename(columns={c: GB_RENAMES.get(c, f"gb_{c}") for c in gb.columns})
    return out.sort_values("year", ignore_index=True)


def basic_features(gb: pd.DataFrame, lean: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    districts = join_districts({"partisan_lean": lean[KEY + ["partisan_lean"]], "incumbency": inc[KEY + ["inc_signed"]]})
    gen = gb[["election", "margin_raw"]].rename(columns={"election": "year", "margin_raw": "gen_margin_raw"})
    return add_national(districts, gen)[KEY + ["gen_margin_raw", "partisan_lean", "inc_signed"]]


def acs_features(acs_supplement: pd.DataFrame) -> pd.DataFrame:
    first, last = ACS_FIELDS
    return pd.concat([acs_supplement[KEY], acs_supplement.loc[:, first:last]], axis=1).sort_values(KEY, ignore_index=True)


def modeling_table(basic: pd.DataFrame, acs_f: pd.DataFrame, nat: pd.DataFrame) -> pd.DataFrame:
    return add_national(join_districts({"basic_f": basic, "acs_f": acs_f}), nat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    gb = pd.read_csv(GENERIC_BALLOT_PATH)
    basic = basic_features(gb, pd.read_csv(PARTISAN_LEAN_PATH), pd.read_csv(INCUMBENCY_PATH))
    nat = nat_political_features(gb)
    acs_f = acs_features(pd.read_csv(ACS_PATH))
    modeling = modeling_table(basic, acs_f, nat)

    for name, df in [("basic_f", basic), ("nat_political_f", nat), ("acs_f", acs_f), ("modeling", modeling)]:
        acs.write_csv_atomic(df, args.output_dir / f"{name}.csv")
        print(f"{name}: {len(df)} rows, {df['year'].nunique()} years -> {args.output_dir / name}.csv")
    no_gb = sorted(set(modeling["year"]) - set(nat["year"]))
    print(f"Years with no generic-ballot row (gen_margin_raw and gb_* left blank): {no_gb}")


if __name__ == "__main__":
    main()
