"""The two adjustment layers of the 2026 forecast, both off by default.

1. Generic-ballot shift (an input). shift_generic_ballot moves 2026's gen_margin_raw (FiftyPlusOne's election-eve
   D - R margin) by `shift` points, positive toward Democrats. Each chamber's Stage 1 then re-predicts, so a race moves
   by the model's own pass-through; nothing imposes it (scenarios/generic_ballot.py).

2. Margin adjustments (an output, before the Monte Carlo). data/reference/margin_adjustments_2026.csv, edited by hand,
   one row per (race, category, party); a race's rows are summed:

       race_id            the submission's race_id: "PA-10", "AK-AL", "AK-SEN"
       category           a category of the scale (below), or blank for a manual adjustment
       party              with a category: the party of the candidate it applies to (D, R, or I for the independent in
                          an independent-vs-R Senate race)
       margin_adjustment  a manual adjustment's points, Democratic-positive (independent-positive in an
                          independent-vs-R Senate race). Blank with a category: the scale sets it
       reason             why, with the evidence (the candidate, the scandal, the source)

   The scale (data/reference/adjustment_scale.csv): category, vote_share_points, margin_points, description, source.
   margin_points is what the category costs the named candidate's side in two-party margin (twice the vote share), so
   a category row adjusts the margin by -margin_points when party is D or I and +margin_points when it is R. The
   starting values are Basinger (2013)'s scandal penalties for House incumbents, 1973-2010; edit the file to change
   the scale, never a row. House races are meant to use the scale (they have no expert ratings or other subjective
   input); a manual House row is allowed but flagged (manual_house).

   A race not in the file is adjusted by 0. The adjustment is added to the House Stage 1 margin of a simulated race,
   or to the Senate race's center, before simulation, so probabilities, seats and generic-ballot scenarios all carry
   it. The unadjusted margin is kept beside it (prediction_model / center_model).
   Magnitudes are judgment, from candidate history and local election evidence, never fitted or taken from markets.
   Election rules (top-two, ranked choice, Louisiana, Kiley's ballot line) are not adjustments: they stay in the race
   handling (house/simulation/races.py, resolvers.py).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MARGIN_ADJUSTMENTS_PATH = ROOT / "data/reference/margin_adjustments_2026.csv"
ADJUSTMENT_SCALE_PATH = ROOT / "data/reference/adjustment_scale.csv"
GENERIC_BALLOT = "gen_margin_raw"  # the generic-ballot term of House Model A and Senate Stage 1
FORECAST_YEAR = 2026
ADJUSTMENT_COLUMNS = ["race_id", "category", "party", "margin_adjustment", "reason"]
SCALE_COLUMNS = ["category", "vote_share_points", "margin_points", "description", "source"]
AGAINST_DEMOCRATIC_MARGIN = {"D": -1.0, "I": -1.0, "R": 1.0}  # the sign of a category's points, by the party it hits


def shift_generic_ballot(frame: pd.DataFrame, shift: float, year: int = FORECAST_YEAR) -> pd.DataFrame:
    """frame with `year`'s gen_margin_raw moved by `shift` points (positive = toward Democrats); other years as is."""
    out = frame.copy()
    out.loc[out["year"] == year, GENERIC_BALLOT] += shift
    return out


def generic_ballot(frame: pd.DataFrame, year: int = FORECAST_YEAR) -> float:
    """The (national, single-valued) generic ballot of `year`."""
    values = frame.loc[frame["year"] == year, GENERIC_BALLOT].unique()
    if len(values) != 1:
        raise ValueError(f"{year} has {len(values)} generic-ballot values; expected one")
    return float(values[0])


def load_adjustment_scale(path=ADJUSTMENT_SCALE_PATH) -> pd.DataFrame:
    """The category scale, indexed by category, checked: one row per category, margin_points = 2 x
    vote_share_points, both non-negative."""
    scale = pd.read_csv(path)
    if list(scale.columns) != SCALE_COLUMNS:
        raise ValueError(f"{path}: columns must be {SCALE_COLUMNS}; got {list(scale.columns)}")
    if scale["category"].duplicated().any() or (scale[["vote_share_points", "margin_points"]] < 0).any().any():
        raise ValueError(f"{path}: categories must be unique and points non-negative")
    if not (scale["margin_points"] == 2 * scale["vote_share_points"]).all():
        raise ValueError(f"{path}: margin_points must be twice vote_share_points (two-party margin)")
    return scale.set_index("category")


def load_margin_adjustments(path=MARGIN_ADJUSTMENTS_PATH, template: pd.DataFrame | None = None,
                            scale: pd.DataFrame | None = None) -> pd.DataFrame:
    """The adjustments file, checked, with margin_adjustment filled from the scale for category rows, source (scale or
    manual), chamber (given the template) and manual_house. Refuses: other columns, a row repeated (same race, category
    and party), a row without a reason, a category not in the scale, a category row without a party or with its own
    points, a manual row without points, and (given the template) races not in it."""
    scale = load_adjustment_scale() if scale is None else scale
    table = pd.read_csv(path, dtype={"race_id": str, "category": str, "party": str, "reason": str})
    if list(table.columns) != ADJUSTMENT_COLUMNS:
        raise ValueError(f"{path}: columns must be {ADJUSTMENT_COLUMNS}; got {list(table.columns)}")
    table["margin_adjustment"] = pd.to_numeric(table["margin_adjustment"], errors="raise").astype(float)
    scaled = table["category"].notna()
    problems = []

    def rows(mask) -> list:
        return table.loc[mask, "race_id"].tolist()

    if table.duplicated(["race_id", "category", "party"]).any():
        problems.append(f"rows repeated: {rows(table.duplicated(['race_id', 'category', 'party']))}")
    if table["reason"].fillna("").str.strip().eq("").any():
        problems.append(f"no reason: {rows(table['reason'].fillna('').str.strip().eq(''))}")
    if (scaled & ~table["category"].isin(scale.index)).any():
        problems.append(f"categories not in the scale {sorted(scale.index)}: "
                        f"{rows(scaled & ~table['category'].isin(scale.index))}")
    if (scaled & ~table["party"].isin(AGAINST_DEMOCRATIC_MARGIN)).any():
        problems.append(f"category rows need party D, R or I: {rows(scaled & ~table['party'].isin(AGAINST_DEMOCRATIC_MARGIN))}")
    if (scaled & table["margin_adjustment"].notna()).any():
        problems.append(f"category rows take their points from the scale, so leave margin_adjustment blank: "
                        f"{rows(scaled & table['margin_adjustment'].notna())}")
    if (~scaled & table["margin_adjustment"].isna()).any():
        problems.append(f"manual rows (no category) need margin_adjustment: {rows(~scaled & table['margin_adjustment'].isna())}")
    if template is not None:
        unknown = sorted(set(table["race_id"]) - set(template["race_id"]))
        if unknown:
            problems.append(f"not in the template: {unknown}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))

    points = table["category"].map(scale["margin_points"]) * table["party"].map(AGAINST_DEMOCRATIC_MARGIN)
    table["margin_adjustment"] = table["margin_adjustment"].where(~scaled, points)
    table["source"] = pd.Series(["scale" if s else "manual" for s in scaled], index=table.index)
    if template is not None:
        table = table.merge(template[["race_id", "chamber"]], on="race_id", how="left", validate="many_to_one")
        table["manual_house"] = (table["chamber"] == "House") & (table["source"] == "manual")
    return table


def by_race(adjustments: pd.DataFrame | None, chamber: str) -> pd.Series:
    """{race_id: summed margin_adjustment} for one chamber's rows (needs load_margin_adjustments' chamber column);
    empty without adjustments."""
    if adjustments is None:
        return pd.Series(dtype=float)
    rows = adjustments[adjustments["chamber"] == chamber]
    return rows.groupby("race_id")["margin_adjustment"].sum()
