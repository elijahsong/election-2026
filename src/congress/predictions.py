"""predictions/final_preds.csv: one row per race of the submission, with its 80% margin interval, its win probability
under the generic-ballot stress tests and Polymarket's price (notebooks/full-forecast/full_forecast_2026.ipynb).

Columns (every probability and margin is the submitted candidate's, from the adjusted forecast = the submission):
    race_id, chamber, predicted_winner_party, candidate_name    the submission's row
    predicted_margin                  the candidate's predicted two-party margin over the runner-up, in points
                                      (positive: the candidate wins): the adjusted margin the Monte Carlo simulates
                                      around (House Stage 1 + adjustment; Senate blend center + adjustment; same-party
                                      top-two races, the top-two model's mean), signed toward the candidate. Blank for
                                      races decided by rule
    predicted_80_low, pred_80_high    the 10th and 90th percentiles of the candidate's simulated two-party margin over
                                      the runner-up, in points (positive: the candidate wins)
                                      - House D-vs-R and Senate races: the Monte Carlo's margin draws, signed toward the
                                        candidate (the Senate's independent-only races: the independent's margin over
                                        the Republican)
                                      - same-party top-two races: top_two.predict_top_two's 80% interval of the
                                        leader's margin, flipped when the runner-up is the pick
                                      - races decided by rule (uncontested, a major party against only third parties):
                                        blank, since no margin is simulated
    win_probability                   the submission's
    gen_ballot_less4, gen_ballot_plus4    the candidate's win probability with the generic ballot 4 points less / more
                                      Democratic (the full forecast's R+4 / D+4 runs, same draws and adjustments).
                                      Same-party top-two races don't depend on it and keep win_probability
    polymarket_odd                    Polymarket's current probability for the candidate (polymarket/odds.py)
    polymarket_delta                  win_probability - polymarket_odd: positive when the model is more confident in
                                      the candidate than Polymarket
"""

import numpy as np
import pandas as pd

from congress.report import PICK_CODES
from house import simulation as house_sim

COLUMNS = ["race_id", "chamber", "predicted_winner_party", "candidate_name", "predicted_margin", "predicted_80_low",
           "pred_80_high", "win_probability", "gen_ballot_less4", "gen_ballot_plus4", "polymarket_odd",
           "polymarket_delta"]
LEVELS = (0.10, 0.90)


def _house_columns(house) -> dict[str, int]:
    return {house_sim.race_id(r.state, r.district): j for j, r in enumerate(house.draws.races.itertuples(index=False))}


def _senate_columns(senate) -> dict[str, int]:
    return {f"{state}-SEN": j for j, state in enumerate(senate.point["state"])}


def pick_sign(chamber: str, party: str) -> int:
    """+1 when the pick is the margin's positive side (D, or the non-Republican in the Senate), else -1."""
    return 1 if party == "D" or (chamber == "Senate" and party == "I") else -1


def margin_intervals(submission: pd.DataFrame, house, senate, top_two: pd.DataFrame) -> pd.DataFrame:
    """race_id, predicted_margin, predicted_80_low, pred_80_high (module docstring). top_two: house_inputs.top_two."""
    house_j, senate_j = _house_columns(house), _senate_columns(senate)
    treatment = dict(zip(house_j, house.draws.races["treatment"]))
    leaders = top_two.assign(race_id=[house_sim.race_id(s, d) for s, d in zip(top_two["state"], top_two["district"])])
    leaders = leaders.set_index("race_id")
    rows = []
    house_point = house.draws.races["prediction"].to_numpy()  # the adjusted D - R margin each race is drawn around
    for pick in submission.itertuples(index=False):
        point = low = high = np.nan
        if pick.chamber == "Senate":
            sign = pick_sign("Senate", pick.predicted_winner_party)
            j = senate_j[pick.race_id]
            point = sign * senate.point["center"].iloc[j]
            low, high = np.quantile(sign * senate.sims[:, j], LEVELS)
        elif treatment[pick.race_id] == "same_party":
            race = leaders.loc[pick.race_id]
            point, low, high = race["mean"], race["interval_80_low"], race["interval_80_high"]
            if pick.candidate_name != race["leader"]:
                point, low, high = -point, -high, -low
        else:
            j = house_j[pick.race_id]
            draws = house.draws.margins[:, j]
            if not np.isnan(draws).all():
                sign = pick_sign("House", pick.predicted_winner_party)
                point = sign * house_point[j]
                low, high = np.quantile(sign * draws, LEVELS)
        rows.append({"race_id": pick.race_id, "predicted_margin": point, "predicted_80_low": low, "pred_80_high": high})
    return pd.DataFrame(rows)


def shifted_win_probability(submission: pd.DataFrame, house, senate) -> pd.Series:
    """The submitted candidate's win probability in another run (a generic-ballot shift), by race_id. Same-party
    top-two races keep their submitted probability: the shift doesn't touch them."""
    house_j, senate_j = _house_columns(house), _senate_columns(senate)
    treatment = dict(zip(house_j, house.draws.races["treatment"]))
    out = {}
    for pick in submission.itertuples(index=False):
        if pick.chamber == "Senate":
            non_r = senate.sims[:, senate_j[pick.race_id]] > 0
            out[pick.race_id] = (non_r if pick.predicted_winner_party in ("D", "I") else ~non_r).mean()
        elif treatment[pick.race_id] == "same_party":
            out[pick.race_id] = pick.win_probability
        else:
            winners = house.draws.winners[:, house_j[pick.race_id]]
            out[pick.race_id] = (winners == PICK_CODES[pick.predicted_winner_party]).mean()
    return pd.Series(out)


def final_predictions(submission: pd.DataFrame, house_runs: dict, senate_runs: dict, top_two: pd.DataFrame,
                      polymarket: pd.DataFrame, *, baseline: str = "adjusted", less: str = "R+4",
                      more: str = "D+4") -> pd.DataFrame:
    """The final_preds.csv table in the submission's row order. house_runs / senate_runs: the full forecast's runs by
    name (generic_ballot.scenario_runs); polymarket: polymarket.odds.track_odds's race_id, polymarket_odd."""
    house, senate = house_runs[baseline], senate_runs[baseline]
    out = submission[["race_id", "chamber", "predicted_winner_party", "candidate_name", "win_probability"]].merge(
        margin_intervals(submission, house, senate, top_two), on="race_id", validate="one_to_one")
    out["gen_ballot_less4"] = out["race_id"].map(shifted_win_probability(submission, house_runs[less],
                                                                         senate_runs[less]))
    out["gen_ballot_plus4"] = out["race_id"].map(shifted_win_probability(submission, house_runs[more],
                                                                         senate_runs[more]))
    out = out.merge(polymarket[["race_id", "polymarket_odd"]], on="race_id", how="left", validate="one_to_one")
    out["polymarket_delta"] = out["win_probability"] - out["polymarket_odd"]
    return out[COLUMNS]
