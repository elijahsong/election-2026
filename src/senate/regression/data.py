"""Senate Stage 1 data: one row per race (year, state, class), predictors joined to targets.

The race key includes `class`, so the two races of a state-year (a regular election and a special) are separate
rows with their own predictors and targets. `special` comes from senate_incumbency.csv.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data/processed/senate"
INCUMBENCY_PATH = ROOT / "data/interim/senate_incumbency.csv"

KEY = ["year", "state", "class"]
TARGET = "adj_margin_2p_norm"  # Democratic two-party margin, points (King and Sanders count as D)
# Observed D-vs-R contests: scored, and the default training rows
EVAL_RACE_TYPES = ["dr", "dr_jungle", "dr_multi", "dr_rcv"]
TARGET_COLUMNS = [TARGET, "margin_2p_norm", "race_type", "is_imputed", "winner_party"]
# Races dropped entirely: neither trained on nor scored
EXCLUDED_RACES = {
    (2022, "AK", 3): "Murkowski vs Tshibaka, both Republicans: the D-vs-R margin (-78.5) isn't comparable",
}


def load_data() -> pd.DataFrame:
    """modeling.csv (every race 2006-2026) with the targets (2006-2024) and `special`. Adds:
    observed  an observed D-vs-R target: finite, not imputed, race_type in EVAL_RACE_TYPES
    imputed   an imputed target (uncontested and third-party races; see targets.csv)
    excluded  in EXCLUDED_RACES: never observed or imputed"""
    predictors = pd.read_csv(PROCESSED / "modeling.csv")
    targets = pd.read_csv(PROCESSED / "targets.csv", usecols=[*KEY, *TARGET_COLUMNS])
    special = pd.read_csv(INCUMBENCY_PATH, usecols=[*KEY, "special"])
    data = predictors.merge(targets, on=KEY, how="left", validate="one_to_one") \
        .merge(special, on=KEY, how="left", validate="one_to_one")
    data["special"] = data["special"].astype(int)
    imputed = data["is_imputed"].fillna(False).astype(bool)
    data["excluded"] = pd.Series(list(zip(*(data[k] for k in KEY)))).isin(EXCLUDED_RACES)
    data["observed"] = data[TARGET].notna() & ~imputed & data["race_type"].isin(EVAL_RACE_TYPES) & ~data["excluded"]
    data["imputed"] = data[TARGET].notna() & imputed & ~data["excluded"]
    return data.sort_values(KEY, ignore_index=True)


def inventory(data: pd.DataFrame) -> pd.DataFrame:
    """Races per year: all, observed D-vs-R targets, excluded, imputed, no usable target, and specials."""
    return data.groupby("year").agg(
        races=("state", "size"), observed=("observed", "sum"), excluded=("excluded", "sum"), imputed=("imputed", "sum"),
        no_target=(TARGET, lambda s: int(s.isna().sum())), specials=("special", "sum"))
