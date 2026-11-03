"""Senate Stage 2: a fixed heuristic blend of fundamentals, polling and expert ratings, in Democratic-positive
two-party margin points (deterministic; no simulation).

    M = w_F * M_F + w_P * M_P + w_E * M_E

    M_F  fundamentals: the Senate Stage 1 prediction (stage1.SENATE_S0 or SENATE_S1), out of sample for past years
    M_P  polling: polls.csv's poll_margin_dem, the 30-day RCP mean for 2006-2024 and FiftyPlusOne's average for 2026,
         both D-positive two-party margins known before Election Day
    M_E  expert ratings: RATING_MARGIN of each of Sabato, Cook and Inside Elections (expert-ratings.csv), combined
         40/30/30 (PROVIDER_WEIGHTS) and renormalized over the providers that rated the race. The ratings' margins
         are averaged; their probability encodings (<provider>_prob_dem) are never used here

Weights (WEIGHTS), fixed whatever the number of polls; never fitted:

    case          fundamentals  polling  experts
    polling           0.37        0.54     0.09
    no_polling        0.65        0.00     0.35

Missing polling is never a 0 margin: it selects the no_polling weights. Ratings exist for 2016-2026 only. A race
without ratings keeps the other components' weights in the same proportion (renormalized to sum to 1), and
fallback_reason says so; nothing is substituted for the missing rating.

Rows are races, keyed (year, state, class); special must agree across the three inputs, and every join is
one-to-one, so the two races of a state-year never mix.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import senate.regression as sr

ROOT = Path(__file__).resolve().parents[2]
POLLS_PATH = ROOT / "data/processed/senate/polls.csv"
RATINGS_PATH = ROOT / "data/processed/senate/expert-ratings.csv"
KEY = sr.KEY

RATING_MARGIN = {
    "Solid D": 31.0, "Likely D": 15.0, "Lean D": 10.0, "Tilt D": 5.0, "Tossup": 0.0,
    "Tilt R": -5.0, "Lean R": -10.0, "Likely R": -15.0, "Solid R": -31.0,
}
PROVIDER_WEIGHTS = {"sabato": 0.40, "cook": 0.30, "ie": 0.30}  # within the expert component only
COMPONENTS = ["fundamentals", "polling", "experts"]
WEIGHTS = {
    "polling": {"fundamentals": 0.37, "polling": 0.54, "experts": 0.09},  # revised 2026-11-03 (was 0.45/0.40/0.15)
    "no_polling": {"fundamentals": 0.65, "polling": 0.00, "experts": 0.35},
}


def expert_margin(ratings: pd.DataFrame) -> pd.Series:
    """The 40/30/30 weighted mean of the providers' RATING_MARGIN, renormalized over the providers present; NaN for
    a race no provider rated."""
    margins = pd.DataFrame({p: ratings[f"{p}_rating"].map(RATING_MARGIN) for p in PROVIDER_WEIGHTS})
    labels = ratings[[f"{p}_rating" for p in PROVIDER_WEIGHTS]].to_numpy().ravel()
    unknown = {label for label in labels if pd.notna(label)} - set(RATING_MARGIN)
    if unknown:
        raise ValueError(f"Ratings without a margin: {sorted(unknown)}")
    weights = margins.notna() * pd.Series(PROVIDER_WEIGHTS)
    total = weights.sum(axis=1)
    return (margins.fillna(0) * weights).sum(axis=1) / total.where(total > 0)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """polls.csv and expert-ratings.csv, the columns Stage 2 uses."""
    polls = pd.read_csv(POLLS_PATH, usecols=[*KEY, "special", "poll_margin_dem", "poll_available", "poll_source"])
    ratings = pd.read_csv(RATINGS_PATH, usecols=[*KEY, "special", *(f"{p}_rating" for p in PROVIDER_WEIGHTS),
                                                 "n_experts"])
    return polls, ratings


def _join(left: pd.DataFrame, right: pd.DataFrame, name: str) -> pd.DataFrame:
    out = left.merge(right, on=KEY, how="left", validate="one_to_one", suffixes=("", f"_{name}"))
    other = out[f"special_{name}"]
    if (other.notna() & (other.astype("boolean") != out["special"].astype(bool))).any():
        raise ValueError(f"special disagrees between Stage 1 and {name}")
    return out.drop(columns=[f"special_{name}"])


def blend(stage1_predictions: pd.DataFrame, polls: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """One row per Stage 1 prediction (KEY, special, prediction, actual, ...), with the three component margins,
    their availability, the applied weights, the case and any fallback, the blended prediction and its residual."""
    if stage1_predictions.duplicated(KEY).any():
        raise ValueError("Stage 1 predictions repeat a race")
    out = stage1_predictions.rename(columns={"prediction": "margin_fundamentals"})
    out = _join(out, polls, "polls")
    out = _join(out, ratings, "ratings")
    out["margin_polling"] = out["poll_margin_dem"].astype(float)
    out["margin_experts"] = expert_margin(out)
    available = pd.DataFrame({"fundamentals": out["margin_fundamentals"].notna(),
                              "polling": out["margin_polling"].notna(), "experts": out["margin_experts"].notna()})
    if not available["fundamentals"].all():
        raise ValueError("Every race needs a Stage 1 prediction")
    out["polling_available"], out["experts_available"] = available["polling"], available["experts"]
    out["weight_case"] = np.where(available["polling"], "polling", "no_polling")

    base = pd.DataFrame([WEIGHTS[case] for case in out["weight_case"]], index=out.index)
    applied = base.where(available, 0.0)
    applied = applied.div(applied.sum(axis=1), axis=0)  # renormalize only when experts are missing
    for c in COMPONENTS:
        out[f"w_{c}"] = applied[c]
    out["fallback_reason"] = np.where(available["experts"], None,
                                      "no expert ratings: other weights renormalized")
    margins = out[[f"margin_{c}" for c in COMPONENTS]].fillna(0).to_numpy()  # zero only where the weight is 0
    out["prediction"] = (margins * applied[COMPONENTS].to_numpy()).sum(axis=1)
    out["residual"] = out["actual"] - out["prediction"]
    return out[[*KEY, "special", "prediction", "actual", "residual", "margin_fundamentals", "margin_polling",
                "margin_experts", "polling_available", "experts_available", "poll_source", "n_experts",
                "weight_case", *(f"w_{c}" for c in COMPONENTS), "fallback_reason",
                *(c for c in ["train_start", "train_end"] if c in out)]]


def score_table(rows: pd.DataFrame, data: pd.DataFrame, predictions: dict[str, str]) -> pd.DataFrame:
    """{label: column} of predictions scored on rows with an actual: MAE, bias (actual - prediction), ex-ante
    competitive MAE and n (|partisan_lean + gb_margin_2p_norm| <= 10), winner accuracy, and the bias by incumbent
    party (inc_signed -1 R, 0 open, +1 D)."""
    rows = rows.merge(data[[*KEY, "partisan_lean", "gb_margin_2p_norm", "inc_signed"]], on=KEY, how="left",
                      validate="one_to_one")
    ex_ante = (rows["partisan_lean"] + rows["gb_margin_2p_norm"]).abs() <= sr.EX_ANTE_THRESHOLD
    table = {}
    for label, column in predictions.items():
        residual = rows["actual"] - rows[column]
        called = (rows[column] != 0) & (rows["actual"] != 0)
        table[label] = {
            "n": len(rows), "mae": residual.abs().mean(), "bias": residual.mean(),
            "ex_ante_n": int(ex_ante.sum()), "ex_ante_mae": residual[ex_ante].abs().mean(),
            "accuracy_pct": 100 * (np.sign(rows.loc[called, column]) == np.sign(rows.loc[called, "actual"])).mean(),
            "ex_ante_accuracy_pct": 100 * (np.sign(rows.loc[called & ex_ante, column])
                                           == np.sign(rows.loc[called & ex_ante, "actual"])).mean(),
            **{f"bias_inc_{name}": residual[rows["inc_signed"] == v].mean()
               for name, v in [("R", -1), ("open", 0), ("D", 1)]},
        }
    return pd.DataFrame(table).T
