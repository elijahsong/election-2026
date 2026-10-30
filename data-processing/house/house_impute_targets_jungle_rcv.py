"""Set the jungle and RCV targets of data/processed/house/targets.csv, in place. Run after house_targets.py, which
leaves them blank; rerunning gives the same file.

dr_jungle and dr_rcv (IMPUTE_RACE_TYPES): the target is the combined D vs R vote of the November (first) round,
  which the row's vote columns already hold, summed across each party's candidates. That is an observed result, not
  an imputation: adj_margin_2p_norm = margin_2p_norm, is_imputed False and needs_imputation False, with
  dem/gop_votes_imputed and imputation_method left blank. The ME-2 2018, 2022 and 2024 rows (flag
  rcv_first_round_not_in_source) are set from the source's rows as they are, though those aren't first-round counts.

Also writes data-processing/house/exceptions.csv: for each race_type of the resulting targets.csv, its Rows, the Blank
adj_margin_2p_norm count, and the keys it Applies to (a list of (year, state, district); blank for dr, nearly
every row). house_exceptions.ipynb looks a race_type up in it.

Every value set must equal targets_raw.csv's observed target for that key. targets_raw.csv, targets_supp.csv and
targets_candidates.csv aren't changed: the first keeps these targets as observed, and the other two have no target
or imputation columns.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import acs
from house_targets import IMPUTE_RACE_TYPES, KEY, OUTPUT_DIR

ROOT = Path(__file__).resolve().parents[2]
EXCEPTIONS_PATH = Path(__file__).resolve().parent / "exceptions.csv"
sys.path.insert(0, str(ROOT / "src"))
from house.regression.schema import TARGET_COLUMNS, TARGET_PROVENANCE_COLUMNS, TARGETS_SCHEMA, validate_schema

# integer columns with blanks, which read_csv would otherwise read as floats and write back as e.g. 93879.0
INT_COLUMNS = ["dem_votes", "gop_votes", "3p_votes", "total_votes", "margin_votes", "dem_candidate_count",
               "gop_candidate_count", "3p_candidate_count", "dem_votes_imputed", "gop_votes_imputed",
               "source_totalvotes", "nonvote_votes"]


def read_targets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={c: "Int64" for c in INT_COLUMNS})
    # object, so a blank can be set: read_csv reads a column without blanks as bool (is_imputed) or float
    for c in ["is_imputed", "imputation_method"]:
        df[c] = df[c].astype(object)
    return df


def impute_jungle_rcv(targets: pd.DataFrame) -> pd.DataFrame:
    """targets with each IMPUTE_RACE_TYPES row's target set, as observed, from its own first-round party totals."""
    out = targets.copy()
    rows = out["race_type"].isin(IMPUTE_RACE_TYPES)
    no_votes = rows & out["margin_2p_norm"].isna()
    if no_votes.any():
        raise ValueError(f"{int(no_votes.sum())} jungle / RCV rows have no reported votes, e.g. "
                         f"{out.loc[no_votes, KEY].head(3).values.tolist()}")
    out.loc[rows, "adj_margin_2p_norm"] = out.loc[rows, "margin_2p_norm"]
    out.loc[rows, ["dem_votes_imputed", "gop_votes_imputed", "imputation_method"]] = None
    out.loc[rows, "is_imputed"] = False
    out.loc[rows, "needs_imputation"] = False
    return out


def check_against_raw(targets: pd.DataFrame, raw: pd.DataFrame) -> list[str]:
    """Jungle / RCV rows whose target differs from targets_raw.csv's observed one (or that targets_raw.csv lacks)."""
    rows = targets.loc[targets["race_type"].isin(IMPUTE_RACE_TYPES), [*KEY, "adj_margin_2p_norm"]]
    merged = rows.merge(raw[[*KEY, "adj_margin_2p_norm"]], on=KEY, how="left", suffixes=("", "_raw"))
    bad = merged["adj_margin_2p_norm_raw"].isna() | ~np.isclose(
        merged["adj_margin_2p_norm"], merged["adj_margin_2p_norm_raw"].fillna(np.inf), rtol=0, atol=1e-9)
    return [f"{tuple(k)}: set {a}, targets_raw.csv {b}"
            for k, a, b in zip(merged.loc[bad, KEY].values.tolist(), merged.loc[bad, "adj_margin_2p_norm"],
                               merged.loc[bad, "adj_margin_2p_norm_raw"])]


def race_type_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per race_type, most rows first: its row count, blank adj_margin_2p_norm count, and keys as a
    list of (year, state, district). The keys of dr, nearly every row, are left blank."""
    out = []
    for race_type, group in df.groupby("race_type"):
        keys = [(int(y), s, int(d)) for y, s, d in group[KEY].itertuples(index=False)]
        out.append({"race_type": race_type, "Rows": len(group),
                    "Blank": int(group["adj_margin_2p_norm"].isna().sum()),
                    "Applies to": None if race_type == "dr" else str(keys)})
    return pd.DataFrame(out).sort_values("Rows", ascending=False, kind="stable", ignore_index=True)


def report(before: pd.DataFrame, after: pd.DataFrame) -> str:
    rows = after["race_type"].isin(IMPUTE_RACE_TYPES)
    target = after["adj_margin_2p_norm"]
    return "\n".join([
        "set: " + ", ".join(f"{k} {v}" for k, v in after.loc[rows, "race_type"].value_counts().items()),
        (f"adj_margin_2p_norm set on {before['adj_margin_2p_norm'].notna().sum()} rows before, "
         f"{target.notna().sum()} after; blank on {target.isna().sum()} ("
         + ", ".join(f"{k} {v}" for k, v in after.loc[target.isna(), "race_type"].value_counts().items()) + ")"),
        "race_type | Rows | Blank | Applies to",
        *(" | ".join("" if pd.isna(v) else str(v) for v in row)
          for row in race_type_table(after).itertuples(index=False)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets-dir", default=OUTPUT_DIR, type=Path)
    parser.add_argument("--exceptions", default=EXCEPTIONS_PATH, type=Path)
    args = parser.parse_args()

    path = args.targets_dir / "targets.csv"
    targets = read_targets(path)
    out = impute_jungle_rcv(targets)[TARGET_COLUMNS + TARGET_PROVENANCE_COLUMNS]
    problems = check_against_raw(out, read_targets(args.targets_dir / "targets_raw.csv"))
    validation = validate_schema(out, schema=TARGETS_SCHEMA)
    print(report(targets, out))
    print("targets.csv schema check (nulls in placeholder and blank-target fields are expected):\n"
          + validation.summary())
    validation.raise_if_errors()
    if problems:
        raise ValueError("Jungle / RCV targets differ from targets_raw.csv:\n" + "\n".join(problems))
    acs.write_csv_atomic(out, path)
    print(f"-> {path}")
    acs.write_csv_atomic(race_type_table(out), args.exceptions)
    print(f"-> {args.exceptions}")


if __name__ == "__main__":
    main()
