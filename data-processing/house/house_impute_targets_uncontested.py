"""Impute the uncontested targets of data/processed/house/targets.csv, in place. Run after house_targets.py (and
house_impute_targets_jungle_rcv.py); rerunning gives the same file.

UNCONTESTED_RACE_TYPES (dem_vs_3p through no_major_party) have no observed D vs R result. Within each (year, state),
the reference races are the REFERENCE_RACE_TYPES rows with reported D and R votes (jungle and RCV rows use their
party totals). They are never imputed rows, nor the ME-2 RCV rows whose source counts aren't the first round
(NOT_FIRST_ROUND_FLAG). With L the district's D two-party share in a neutral year,
0.5 + cook_pvi / 100 from house_partisan_lean.csv (built only from presidential elections before the year, so it
carries no House result):

  turnout_hat       = round(mean(D_j + R_j))                       over the state's references j
  state_adjustment  = mean(D_j / (D_j + R_j) - L_j)
  share             = clip(L + state_adjustment, 0, 1)
  D_hat             = round(turnout_hat * share), R_hat = turnout_hat - D_hat

dem/gop_votes_imputed are D_hat and R_hat, adj_margin_2p_norm is 100 * (D_hat - R_hat) / turnout_hat, is_imputed
True and imputation_method IMPUTATION_METHOD; needs_imputation stays True. The vote columns, race_type and every other
column are left as they are.

Fallback, for a row with no reference race in its state and year: Split Ticket's national SHAVE for the year, from
data/reference/shave.csv. Its margin_2p_norm is the national D two-party margin in points, so the national two-party
share minus 0.5 is margin_2p_norm / 200:

  share               = clip(L + margin_2p_norm / 200, 0, 1)
  adj_margin_2p_norm  = 100 * (2 * share - 1)

with is_imputed True and imputation_method FALLBACK_METHOD. Without a turnout reference, dem/gop_votes_imputed stay
blank and the row keeps NO_REFERENCE_FLAG. Only margin_2p_norm is used: shave.csv rounds each column on its own, so
50 + margin_2p_norm / 2 can differ from its dem_2p_norm by about 0.005 points. A row with no partisan lean, or a
fallback row whose year isn't in shave.csv, stays blank and is flagged in quality_flags (NO_LEAN_FLAG, NO_SHAVE_FLAG).
Both methods are in schema.MODELED_IMPUTATION_METHODS, so the regression neither trains on these rows nor scores
them.

Also refreshes data-processing/house/exceptions.csv (see house_impute_targets_jungle_rcv.py). targets_raw.csv,
targets_supp.csv and targets_candidates.csv aren't changed.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import acs
from house_impute_targets_jungle_rcv import EXCEPTIONS_PATH, race_type_table, read_targets
from house_targets import KEY, OUTPUT_DIR

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from house.regression.schema import (MODELED_IMPUTATION_METHODS, RACE_TYPES, TARGET_COLUMNS,
                                     TARGET_PROVENANCE_COLUMNS, TARGETS_SCHEMA, validate_schema)

LEAN_PATH = ROOT / "data/interim/house/house_partisan_lean.csv"
SHAVE_PATH = ROOT / "data/reference/shave.csv"
UNCONTESTED_RACE_TYPES = RACE_TYPES[RACE_TYPES.index("dem_vs_3p"):]
REFERENCE_RACE_TYPES = ["dr", "dr_multi", "dr_jungle", "dr_rcv"]
IMPUTATION_METHOD = "state_mean_residual"
FALLBACK_METHOD = "national_shave"
assert {IMPUTATION_METHOD, FALLBACK_METHOD} <= set(MODELED_IMPUTATION_METHODS)
NO_REFERENCE_FLAG = "imputation_no_reference"
NO_LEAN_FLAG = "imputation_no_partisan_lean"
NO_SHAVE_FLAG = "imputation_no_shave"
OWN_FLAGS = (NO_REFERENCE_FLAG, NO_LEAN_FLAG, NO_SHAVE_FLAG)
# house_targets.py's flag on the ME-2 RCV rows whose source counts aren't the first round; never references
NOT_FIRST_ROUND_FLAG = "rcv_first_round_not_in_source"
# the vote columns, which must come out exactly as they went in
VOTE_COLUMNS = ["dem_votes", "gop_votes", "3p_votes", "total_votes"]


def read_lean(path: Path = LEAN_PATH) -> pd.DataFrame:
    """KEY + lean, the D two-party share (0-1) in a neutral year."""
    lean = pd.read_csv(path)
    return lean[KEY].assign(lean=0.5 + lean["cook_pvi"] / 100)


def read_shave(path: Path = SHAVE_PATH) -> pd.DataFrame:
    """year + shave_shift, the national D two-party share minus 0.5: SHAVE's margin_2p_norm / 200."""
    shave = pd.read_csv(path)
    if shave["year"].duplicated().any() or not shave["margin_2p_norm"].between(-100, 100).all():
        raise ValueError(f"{path}: years must be unique and margin_2p_norm within -100..100")
    return shave[["year"]].assign(shave_shift=shave["margin_2p_norm"] / 200)


def without_flags(flags: pd.Series) -> pd.Series:
    """quality_flags with this script's own flags removed (blank when none is left)."""
    return flags.fillna("").str.split(";").map(
        lambda f: ";".join(x for x in f if x and x not in OWN_FLAGS) or None)


def with_flag(flags: pd.Series, flag: str) -> pd.Series:
    return flags.fillna("").map(lambda f: f"{f};{flag}" if f else flag)


def state_references(targets: pd.DataFrame, lean: pd.DataFrame) -> pd.DataFrame:
    """Per (year, state): n_references, turnout_hat and state_adjustment, from the reference races."""
    ref = targets[targets["race_type"].isin(REFERENCE_RACE_TYPES) & targets["dem_votes"].notna()
                  & targets["gop_votes"].notna()
                  & ~targets["quality_flags"].fillna("").str.contains(NOT_FIRST_ROUND_FLAG)]
    ref = ref[ref["dem_votes"] + ref["gop_votes"] > 0].merge(lean, on=KEY, how="left", validate="one_to_one")
    if ref["lean"].isna().any():
        raise ValueError(f"Reference races without a partisan lean: {ref.loc[ref['lean'].isna(), KEY].values.tolist()}")
    two_party = (ref["dem_votes"] + ref["gop_votes"]).astype(float)
    ref = ref.assign(two_party=two_party, residual=ref["dem_votes"].astype(float) / two_party - ref["lean"])
    return ref.groupby(["year", "state"]).agg(n_references=("residual", "size"), turnout_hat=("two_party", "mean"),
                                               state_adjustment=("residual", "mean")) \
        .assign(turnout_hat=lambda g: np.round(g["turnout_hat"]).astype("int64")).reset_index()


def impute_uncontested(targets: pd.DataFrame, lean: pd.DataFrame, shave: pd.DataFrame) -> pd.DataFrame:
    """targets with each UNCONTESTED_RACE_TYPES row's target imputed from its state's reference races, or from the
    national SHAVE (read_shave) where the state has none."""
    out = targets.copy()
    rows = out["race_type"].isin(UNCONTESTED_RACE_TYPES)
    observed = rows & out["is_imputed"].eq(False)
    if observed.any():
        raise ValueError(f"{int(observed.sum())} uncontested rows have an observed target, e.g. "
                         f"{out.loc[observed, KEY].head(3).values.tolist()}")
    out.loc[rows, ["adj_margin_2p_norm", "dem_votes_imputed", "gop_votes_imputed", "is_imputed",
                   "imputation_method"]] = None
    out.loc[rows, "quality_flags"] = without_flags(out.loc[rows, "quality_flags"])

    states = state_references(out, lean)
    found = out.loc[rows, KEY].merge(lean, on=KEY, how="left", validate="one_to_one") \
        .merge(states, on=["year", "state"], how="left", validate="many_to_one") \
        .merge(shave, on="year", how="left", validate="many_to_one").set_index(out.index[rows])
    no_lean, no_reference = found["lean"].isna(), found["n_references"].isna()
    fallback = ~no_lean & no_reference
    no_shave = fallback & found["shave_shift"].isna()
    for flag, missing in [(NO_LEAN_FLAG, no_lean), (NO_REFERENCE_FLAG, no_reference), (NO_SHAVE_FLAG, no_shave)]:
        index = found.index[missing]
        out.loc[index, "quality_flags"] = with_flag(out.loc[index, "quality_flags"], flag)

    primary = found[~no_lean & ~no_reference]
    turnout = primary["turnout_hat"].astype("int64")
    if (turnout <= 0).any():
        raise ValueError(f"Zero turnout estimate for {primary.loc[turnout <= 0, KEY].values.tolist()}")
    share = (primary["lean"] + primary["state_adjustment"]).clip(0, 1)
    dem = np.round(turnout * share).astype("int64")
    gop = turnout - dem
    out.loc[primary.index, "dem_votes_imputed"] = dem
    out.loc[primary.index, "gop_votes_imputed"] = gop
    out.loc[primary.index, "adj_margin_2p_norm"] = 100 * (dem - gop) / turnout
    out.loc[primary.index, "is_imputed"] = True
    out.loc[primary.index, "imputation_method"] = IMPUTATION_METHOD
    for c in ["dem_votes_imputed", "gop_votes_imputed"]:
        out[c] = out[c].astype("Int64")

    # fallback: the national SHAVE in place of the state adjustment; no turnout, so no vote counts
    shaved = found[fallback & ~no_shave]
    share = (shaved["lean"] + shaved["shave_shift"]).clip(0, 1)
    out.loc[shaved.index, "adj_margin_2p_norm"] = 100 * (2 * share - 1)
    out.loc[shaved.index, "is_imputed"] = True
    out.loc[shaved.index, "imputation_method"] = FALLBACK_METHOD
    return out


def check_votes_unchanged(targets: pd.DataFrame, raw: pd.DataFrame) -> list[str]:
    """Keys whose vote columns differ from targets_raw.csv's (or that targets_raw.csv lacks)."""
    merged = targets[[*KEY, *VOTE_COLUMNS]].merge(raw[[*KEY, *VOTE_COLUMNS]], on=KEY, how="left",
                                                  suffixes=("", "_raw"), indicator=True)
    bad = merged["_merge"] != "both"
    for c in VOTE_COLUMNS:
        bad |= merged[c].isna() != merged[f"{c}_raw"].isna()
        bad |= (merged[c] != merged[f"{c}_raw"]).fillna(False)
    return [str(tuple(k)) for k in merged.loc[bad, KEY].values.tolist()]


def report(after: pd.DataFrame, lean: pd.DataFrame) -> str:
    imputed = after["imputation_method"] == IMPUTATION_METHOD
    shaved = after["imputation_method"] == FALLBACK_METHOD
    flagged = after[after["race_type"].isin(UNCONTESTED_RACE_TYPES) & after["adj_margin_2p_norm"].isna()]
    states = state_references(after, lean)
    used = states.merge(after.loc[imputed, ["year", "state"]].drop_duplicates(), on=["year", "state"])

    def spread(s: pd.Series, fmt: str) -> str:
        return f"min {s.min():{fmt}}, median {s.median():{fmt}}, max {s.max():{fmt}}"

    return "\n".join([
        f"imputed ({IMPUTATION_METHOD}): {int(imputed.sum())} rows; "
        + ", ".join(f"{k} {v}" for k, v in after.loc[imputed, "race_type"].value_counts().items()),
        f"imputed ({FALLBACK_METHOD}, no vote counts): {int(shaved.sum())} rows; "
        + ", ".join(f"{(int(y), s, int(d))} {r} {m:+.1f}" for y, s, d, r, m
                    in after.loc[shaved, [*KEY, "race_type", "adj_margin_2p_norm"]].itertuples(index=False)),
        f"left blank and flagged: {len(flagged)} rows; "
        + ", ".join(f"{(int(y), s, int(d))} {r} {f}" for y, s, d, r, f
                    in flagged[[*KEY, "race_type", "quality_flags"]].itertuples(index=False)),
        f"over the {len(used)} (year, state) pairs imputed by {IMPUTATION_METHOD}: references {spread(used['n_references'], '.0f')}; "
        f"state_adjustment {spread(used['state_adjustment'], '+.4f')}; turnout_hat {spread(used['turnout_hat'], ',.0f')}",
        "race_type | Rows | Blank | Applies to",
        *(" | ".join("" if pd.isna(v) else str(v) for v in row)
          for row in race_type_table(after).itertuples(index=False)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets-dir", default=OUTPUT_DIR, type=Path)
    parser.add_argument("--lean", default=LEAN_PATH, type=Path)
    parser.add_argument("--shave", default=SHAVE_PATH, type=Path)
    parser.add_argument("--exceptions", default=EXCEPTIONS_PATH, type=Path)
    args = parser.parse_args()

    path = args.targets_dir / "targets.csv"
    lean = read_lean(args.lean)
    out = impute_uncontested(read_targets(path), lean, read_shave(args.shave))[TARGET_COLUMNS + TARGET_PROVENANCE_COLUMNS]
    changed = check_votes_unchanged(out, read_targets(args.targets_dir / "targets_raw.csv"))
    validation = validate_schema(out, schema=TARGETS_SCHEMA)
    print(report(out, lean))
    print("targets.csv schema check (nulls in placeholder and blank-target fields are expected):\n"
          + validation.summary())
    validation.raise_if_errors()
    if changed:
        raise ValueError("Vote columns differ from targets_raw.csv for " + ", ".join(changed))
    acs.write_csv_atomic(out, path)
    print(f"-> {path}")
    acs.write_csv_atomic(race_type_table(out), args.exceptions)
    print(f"-> {args.exceptions}")


if __name__ == "__main__":
    main()
