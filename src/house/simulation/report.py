"""Stage 2 tables for display: stacking calibrations, consistent number formats, and seat summaries.

    stack(calibrations, method)        one Calibration method's table for every configuration (and backtest), stacked
    styled(table)                      the table with FORMATS applied to whichever of its columns they name
    seat_table(draws, scenarios, ...)  seats by party and scenario: median, central intervals, P(majority)
    track_table(forecast, ...)         the House rows of the congress-track submission (TRACK_TEMPLATE_PATH)

Formatting only: nothing here computes a metric (calibration.py and races.py do).
"""

import pandas as pd

from house.regression.data import ROOT
from house.regression.schema import KEY

from .calibration import seat_summary
from .races import DEM, GOP, OTHER, RaceDraws

TRACK_TEMPLATE_PATH = ROOT / "data/reference/full congress track template.csv"
TRACK_PARTY = {"D": "D", "R": "R", "other": "I"}  # ballot party -> the submission's code; I is any non-major party

PARTIES = {"D": DEM, "R": GOP, "other": OTHER}
MAJORITY = 218
FORMATS = {
    **{c: "{:.4f}" for c in ["Brier", "Brier (all races)", "contested Brier", "competitive Brier"]},
    **{c: "{:.1%}" for c in ["80% coverage", "95% coverage", "accuracy", "p_dem", "p_gop", "p_other", "forecast",
                             "observed", f"P({MAJORITY}+)"]},
    **{c: "{:.1f}" for c in ["actual-seat percentile", "80% mean width", "95% mean width"]},
    **{c: "{:+.1f}" for c in ["prediction", "actual"]},
    **{c: "{:+d}" for c in ["median − actual", "called − actual"]},
    **{c: "{:.0f}" for c in ["n", "races scored", "observed margins"]},
}


def stack(calibrations: dict, method: str, outer: str = "backtest") -> pd.DataFrame:
    """Calibration.<method>() for each configuration, stacked with a config level in front. calibrations is
    {name: Calibration}, or {backtest: {name: Calibration}} for side-by-side backtests (an outer level too, named
    `outer`: e.g. "hierarchy" for {hierarchy: {config: Calibration}}). method may also be a function of one
    Calibration (e.g. lambda cal: simulation.group_seats(cal, "competitive_ex_ante"))."""
    run = (lambda cal: getattr(cal, method)()) if isinstance(method, str) else method
    nested = all(isinstance(v, dict) for v in calibrations.values())
    if nested:
        frames = {(backtest, name): run(cal) for backtest, cals in calibrations.items() for name, cal in cals.items()}
        names = [outer, "config"]
    else:
        frames = {name: run(cal) for name, cal in calibrations.items()}
        names = ["config"]
    out = pd.concat(frames, names=names)
    if isinstance(out.index, pd.MultiIndex) and out.index.names[-1] is None:  # a plain row number: drop it
        out = out.droplevel(-1)
    return out


def styled(table: pd.DataFrame):
    """A Styler with FORMATS on the columns it names (flat or the last level of MultiIndex columns)."""
    formats = {c: FORMATS[c[-1] if isinstance(c, tuple) else c] for c in table.columns
               if (c[-1] if isinstance(c, tuple) else c) in FORMATS}
    return table.style.format(formats, precision=3)


def seat_table(draws: RaceDraws, scenarios: dict, levels=(0.8, 0.95), majority: int = MAJORITY) -> pd.DataFrame:
    """Seats per ballot party under each scenario ({name: alignment or None}; None is the official count), with the
    median, central intervals and P(seats >= majority). Checks that every draw's seats sum to the races simulated."""
    rows = {}
    for scenario, alignment in scenarios.items():
        by_party = {party: draws.seats(code, alignment=alignment) for party, code in PARTIES.items()}
        assert (sum(by_party.values()) == draws.winners.shape[1]).all(), "a draw's seats don't sum to the races"
        for party, seats in by_party.items():
            summary = seat_summary(seats, levels=levels)
            rows[(scenario, party)] = {
                "median": summary["median"],
                **{f"{round(100 * level)}% interval": f"{lower}–{upper}"
                   for level, (lower, upper) in summary["intervals"].items()},
                f"P({majority}+)": float((seats >= majority).mean()),
            }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["scenario", "party"])


def race_id(state: str, district: int) -> str:
    """The submission's House race_id: AL-01, or AK-AL for an at-large district (district 0)."""
    return f"{state}-AL" if district == 0 else f"{state}-{district:02d}"


def track_table(forecast: pd.DataFrame, top_two: pd.DataFrame, fec_slots: pd.DataFrame,
                template: pd.DataFrame) -> pd.DataFrame:
    """The template's House rows, in its order: race_id, chamber, predicted_winner_party, candidate_name,
    win_probability.

    predicted_winner_party   forecast_table's predicted_party (the most likely ballot party), coded TRACK_PARTY
    candidate_name           forecast_table's predicted_candidate; where it is unresolved (Louisiana's several
                             candidates per party), the party's model slot in house_fec.csv (fec_slots: its top FEC
                             fundraiser, fec_finance.TOP_FUNDRAISER_STATES_2026), the candidate Stage 1 simulated
    win_probability          the predicted party's share of draws; in a same-party race (party certain), the named
                             candidate's probability from top_two (top_two.predict_top_two)"""
    out = forecast[[*KEY, "treatment", "predicted_party", "predicted_candidate", "p_dem", "p_gop", "p_other"]].copy()
    out["win_probability"] = out[["p_dem", "p_gop", "p_other"]].max(axis=1)

    same = top_two.set_index(KEY)
    for i in out.index[out["treatment"] == "same_party"]:
        race = same.loc[tuple(out.loc[i, KEY])]
        out.loc[i, "win_probability"] = (race["p_leader"] if out.loc[i, "predicted_candidate"] == race["leader"]
                                         else race["p_runner_up"])

    slots = fec_slots[fec_slots["year"] == out["year"].iloc[0]].set_index(KEY)
    for i in out.index[out["predicted_candidate"].isna() & out["predicted_party"].isin(["D", "R"])]:
        column = "dem_name" if out.loc[i, "predicted_party"] == "D" else "gop_name"
        out.loc[i, "predicted_candidate"] = slots.loc[tuple(out.loc[i, KEY]), column]

    out = pd.DataFrame({
        "race_id": [race_id(s, d) for s, d in zip(out["state"], out["district"])],
        "chamber": "House",
        "predicted_winner_party": out["predicted_party"].map(TRACK_PARTY),
        "candidate_name": out["predicted_candidate"],
        "win_probability": out["win_probability"],
    })
    house = template.loc[template["chamber"] == "House", ["race_id"]]
    if set(house["race_id"]) != set(out["race_id"]) or len(out) != len(house):
        raise ValueError("the forecast's districts don't match the template's House rows")
    out = house.merge(out, on="race_id", how="left", validate="1:1")
    if out[["predicted_winner_party", "candidate_name", "win_probability"]].isna().any().any():
        raise ValueError("a House row has no party, candidate or probability")
    return out
