"""The full forecast's tables, for both chambers: one row per race with probabilities and a rating, chamber
summaries, and comparisons across runs (model vs adjusted, generic-ballot shifts).

Race rows (race_table) take each chamber's Forecast.races() (house/forecast.py, senate/forecast.py), which share
these columns:
    race_id, chamber, race_type          the submission's race id; House race_type, Senate seat_type
    dem_candidate, gop_candidate         each side's candidate (house.forecast.candidate_names; in a Senate
                                         independent-vs-R race, dem_candidate is the independent, marked (I))
    margin_model, margin_adjustment,     the model's margin (House Stage 1; Senate blend or rating center), the
    margin                               manual adjustment (congress/adjustments.py), and their sum, which is
                                         simulated. Blank for a House race decided by rule
    dem_2p_norm, gop_2p_norm             the two-party vote shares that margin implies: 50 + margin / 2 and
                                         50 - margin / 2 (in an independent-vs-R Senate race, dem_2p_norm is the
                                         independent's)
    p_dem, p_gop, p_other                each side's share of draws; other is a third party or independent
    party_probability                    the most likely side's share of draws
    predicted_winner_party,              the submission's row (D, R or I). win_probability is the named candidate's,
    candidate_name, win_probability      which differs from party_probability only in same-party top-two races

Ratings (rate) come from party_probability, the chance the predicted winner's party (or independent) wins:
    Toss-up      below RATING_THRESHOLDS["Toss-up"]
    Lean <side>  below RATING_THRESHOLDS["Lean"]
    Likely       below RATING_THRESHOLDS["Likely"]
    Safe         otherwise
The thresholds are a reporting convention, not a model input: nothing upstream reads them.
"""

import numpy as np
import pandas as pd

from house import simulation as house_sim
from house.regression.schema import KEY

RATING_THRESHOLDS = {"Toss-up": 0.60, "Lean": 0.75, "Likely": 0.95}
RATING_GROUP_COLUMNS = ["race_id", "race_type", "dem_candidate", "gop_candidate", "margin_model", "margin_adjustment",
                        "p_dem", "dem_2p_norm", "gop_2p_norm"]
RATING_ORDER = ["Safe D", "Likely D", "Lean D", "Toss-up", "Lean I", "Likely I", "Safe I", "Lean R", "Likely R",
                "Safe R"]
HOUSE_MAJORITY = 218
SENATE_CONTROL = 51  # the Vice President is a Republican, so a 50-50 Senate is Republican


def senate_dem_caucus(counts: pd.DataFrame) -> pd.Series:
    """Seats caucusing with the Democrats in each simulation: Democrats, King and Sanders, and every new independent
    (NE, ID, SD). Assumed for the full forecast; senate.simulation.chamber.control keeps both caucus scenarios."""
    return counts["dem_aligned"] + counts["new_independents"]


def rate(probability: pd.Series, party: pd.Series, thresholds: dict = RATING_THRESHOLDS) -> pd.Series:
    """The rating of each race from its predicted side's probability and party (D, R or I)."""
    label = np.select([probability < thresholds["Toss-up"], probability < thresholds["Lean"],
                       probability < thresholds["Likely"]], ["Toss-up", "Lean", "Likely"], "Safe")
    return pd.Series(np.where(label == "Toss-up", label, label + " " + party.astype(str)), index=probability.index)


def race_table(house, senate, template: pd.DataFrame, thresholds: dict = RATING_THRESHOLDS) -> pd.DataFrame:
    """Every race of both chambers (house and senate are Forecast objects), in the template's order, with its
    rating."""
    races = pd.concat([house.races(), senate.races()], ignore_index=True)
    races["rating"] = rate(races["party_probability"], races["predicted_winner_party"], thresholds)
    races["dem_2p_norm"] = 50 + races["margin"] / 2
    races["gop_2p_norm"] = 50 - races["margin"] / 2
    if len(races) != len(template) or set(races["race_id"]) != set(template["race_id"]):
        raise ValueError("The forecast's races don't match the template's")
    return template[["race_id"]].merge(races, on="race_id", how="left", validate="one_to_one")


def threshold_line(thresholds: dict = RATING_THRESHOLDS) -> str:
    """The rating thresholds in words, on the predicted side's win probability."""
    t = {k: round(100 * v) for k, v in thresholds.items()}
    return (f"Toss-up < {t['Toss-up']}% · Lean {t['Toss-up']}–{t['Lean']}% · Likely {t['Lean']}–{t['Likely']}% · "
            f"Safe ≥ {t['Likely']}% (the predicted side's win probability)")


def rating_table(races: pd.DataFrame, *, chamber: str, rating: str, columns=RATING_GROUP_COLUMNS) -> pd.DataFrame:
    """One chamber's races with one rating, most Democratic first: race_id, type, the model margin and adjustment,
    p_dem and the implied two-party shares."""
    rows = races[(races["chamber"] == chamber) & (races["rating"] == rating)]
    return (rows.sort_values("p_dem", ascending=False)[list(columns)].rename(columns={"race_type": "type"})
            .reset_index(drop=True))


def rating_counts(races: pd.DataFrame) -> pd.DataFrame:
    """Races per rating (columns, in RATING_ORDER) and chamber (rows)."""
    counts = pd.crosstab(races["chamber"], races["rating"])
    return counts.reindex(columns=[r for r in RATING_ORDER if r in counts.columns], fill_value=0)


def compare(tables: dict[str, pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """{run name: race_table} side by side: `columns` of every run, indexed by race_id, grouped by column."""
    wide = pd.concat({name: t.set_index("race_id")[columns] for name, t in tables.items()}, axis=1)
    return wide.swaplevel(axis=1)[columns]


def changed(tables: dict[str, pd.DataFrame], baseline: str, column: str) -> pd.Series:
    """Per race, whether `column` differs from `baseline` in any other run."""
    wide = pd.concat({name: t.set_index("race_id")[column] for name, t in tables.items()}, axis=1)
    return wide.ne(wide[baseline], axis=0).any(axis=1)


def chamber_summary(house: dict, senate: dict, baseline: str) -> pd.DataFrame:
    """One row per run ({name: Forecast} for each chamber, same names): the generic ballot, how far the race margins
    moved from `baseline` on average, and each chamber's seats and control. Senate seats and control count the new
    independents with the Democrats (senate_dem_caucus)."""
    base_house = house[baseline].point.set_index(KEY)["prediction"]
    base_senate = senate[baseline].point["center"]
    rows = {}
    for name, h in house.items():
        s = senate[name]
        house_moved = (h.point.set_index(KEY)["prediction"] - base_house).dropna()
        dem_nominee = s.point["seat_type"] == "dem_nominee"
        dem_seats = h.draws.seats(house_sim.DEM)
        dem_caucus = senate_dem_caucus(s.counts)
        rows[name] = {
            "generic ballot (D - R)": h.generic_ballot,
            "House: mean race shift": house_moved.mean(),
            "House: median D seats": float(np.median(dem_seats)),
            "House: D seats 10%-90%": f"{np.quantile(dem_seats, 0.1):.0f}–{np.quantile(dem_seats, 0.9):.0f}",
            f"House: P(D {HOUSE_MAJORITY}+)": float((dem_seats >= HOUSE_MAJORITY).mean()),
            "House: races called D": int((h.track["predicted_winner_party"] == "D").sum()),
            "Senate: mean race shift (D nominees)": (s.point["center"] - base_senate)[dem_nominee].mean(),
            "Senate: median D-caucus seats": float(dem_caucus.median()),
            "Senate: D-caucus seats 10%-90%": f"{dem_caucus.quantile(0.1):.0f}–{dem_caucus.quantile(0.9):.0f}",
            "Senate: P(D control)": float((dem_caucus >= SENATE_CONTROL).mean()),
            "Senate: races called D": int((s.track["predicted_winner_party"] == "D").sum()),
        }
    return pd.DataFrame(rows).T


def flipped_races(tracks: dict[str, pd.DataFrame], baseline: str) -> pd.DataFrame:
    """Races whose predicted winner differs from `baseline` in some run ({name: track rows}): each run's party,
    candidate and win probability."""
    wide = pd.concat({name: t.set_index("race_id")[["predicted_winner_party", "candidate_name", "win_probability"]]
                      for name, t in tracks.items()}, axis=1)
    party = wide.xs("predicted_winner_party", axis=1, level=1)
    flipped = party.ne(party[baseline], axis=0).any(axis=1)
    return wide[flipped].swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)


PICK_CODES = {"D": house_sim.DEM, "R": house_sim.GOP, "I": house_sim.OTHER}  # submission party -> House draw code


def pick_outcomes(house, senate, picks: pd.DataFrame, seed: int) -> tuple[list[str], np.ndarray]:
    """(race_ids in picks' order, (n_sims, n_races) bool): whether each race's picked candidate (picks: race_id,
    predicted_winner_party, win_probability, e.g. the submission) wins in each simulation of the house and senate
    Forecasts. The picks can come from another run than the draws (a stress test).

    House: the draw's winning party is the pick's (I = OTHER, e.g. Kiley). Senate: the pick is D or I, and the
    non-Republican wins the draw (margin > 0), or the pick is R and loses it. Same-party top-two races: the party is
    certain and the draws carry no candidate, so the pick wins with its own win_probability, drawn from `seed`
    (independent of everything else, as the top-two model is)."""
    n = house.draws.winners.shape[0]
    if senate.sims.shape[0] != n:
        raise ValueError("House and Senate need the same number of simulations")
    rng = np.random.default_rng(seed)
    party = picks.set_index("race_id")["predicted_winner_party"]
    probability = picks.set_index("race_id")["win_probability"]
    columns = {}
    races = house.draws.races
    for j, race in enumerate(races.itertuples(index=False)):
        race_id = house_sim.race_id(race.state, race.district)
        if race.treatment == "same_party":
            columns[race_id] = rng.random(n) < probability[race_id]
        else:
            columns[race_id] = house.draws.winners[:, j] == PICK_CODES[party[race_id]]
    for j, state in enumerate(senate.point["state"]):
        race_id = f"{state}-SEN"
        non_r_wins = senate.sims[:, j] > 0
        columns[race_id] = non_r_wins if party[race_id] in ("D", "I") else ~non_r_wins
    race_ids = picks["race_id"].tolist()
    return race_ids, np.column_stack([columns[r] for r in race_ids])
