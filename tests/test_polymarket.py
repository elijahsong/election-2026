"""Polymarket race matching, prices and the join to the submission (synthetic events, no API call), and the
final_preds signs."""

import numpy as np
import pandas as pd
import pytest

from congress import predictions
from polymarket import odds


def market(title, question, bid, ask, last=None, active=True):
    return {"groupItemTitle": title, "question": question, "bestBid": bid, "bestAsk": ask, "lastTradePrice": last,
            "outcomePrices": None, "active": active, "closed": False}


EVENTS = [
    {"title": "NY-17 House Election Winner", "slug": "ny-17", "markets": [
        market("Mike Lawler (R)", "Will the Republican Party win the NY-17 House seat?", 0.40, 0.42),
        market("Cait Conley (D)", "Will the Democratic Party win the NY-17 House seat?", 0.57, 0.59),
        market("Other", "Will another party win the NY-17 House seat?", 0, 1, active=False)]},
    {"title": "CA-11 House Election Winner (Individual)", "slug": "ca-11-ind", "markets": [
        market("Scott Wiener", "Will Scott Wiener win the CA-11 House seat?", 0.79, 0.81),
        market("Saikat Chakrabarti", "Will Saikat Chakrabarti win the CA-11 House seat?", None, 0.21, last=0.2)]},
    {"title": "Alaska Senate Election Winner", "slug": "ak-sen", "markets": [
        market("Sen. Dan Sullivan", "Will Dan Sullivan win the Alaska Senate race in 2026?", 0.25, 0.26),
        market("Daniel J. Sullivan Jr.", "Will Daniel J. Sullivan Jr. win?", None, 0.001, last=0.001),
        market("Mary Peltola", "Will Mary Peltola win the Alaska Senate race in 2026?", 0.74, 0.75)]},
    {"title": "NY-17 House Election Margin of Victory", "slug": "ny-17-mov", "markets": []},
]


def test_titles_and_prices():
    assert odds.race_id_from_title("NY-17 House Election Winner") == ("NY-17", False)
    assert odds.race_id_from_title("AK-AL House Election Winner") == ("AK-AL", False)
    assert odds.race_id_from_title("CA-04 House Election Winner (by individual)") == ("CA-04", True)
    assert odds.race_id_from_title("Nebraska Senate Election Winner  ") == ("NE-SEN", False)
    assert odds.race_id_from_title("NY-17 House Election Margin of Victory") is None
    m = odds.race_markets(EVENTS)
    assert len(m) == 7  # the inactive placeholder is dropped
    ny = m[m["race_id"] == "NY-17"].set_index("party")
    assert ny.loc["D", "price"] == pytest.approx(0.58)  # bid-ask midpoint
    # normalized over the race: 0.58 / (0.58 + 0.41)
    assert ny["polymarket_prob"].sum() == pytest.approx(1.0) and ny.loc["D", "polymarket_prob"] == pytest.approx(0.58 / 0.99)
    ca = m[m["race_id"] == "CA-11"].set_index("candidate")
    assert ca.loc["Saikat Chakrabarti", "price"] == 0.2  # no bid: the last trade


def test_join_by_party_and_by_name():
    m = odds.race_markets(EVENTS)
    submission = pd.DataFrame({"race_id": ["NY-17", "CA-11", "AK-SEN"], "predicted_winner_party": ["D", "D", "R"],
                               "candidate_name": ["Cait Conley", "Scott Wiener", "Dan S. Sullivan"]})
    out = odds.track_odds(submission, m, same_party={"CA-11"}).set_index("race_id")["polymarket_odd"]
    assert out["NY-17"] == pytest.approx(0.58 / 0.99)
    assert out["CA-11"] == pytest.approx(0.8 / 1.0)
    # two Sullivans: the first name picks the senator, not Daniel J. Sullivan Jr.
    ak = m[m["race_id"] == "AK-SEN"]
    assert out["AK-SEN"] == pytest.approx(0.255 / ak["price"].sum())


def test_margin_sign_is_the_candidates():
    assert predictions.pick_sign("House", "D") == 1 and predictions.pick_sign("House", "R") == -1
    assert predictions.pick_sign("House", "I") == -1  # Kiley fills the GOP slot of the D - R margin
    assert predictions.pick_sign("Senate", "I") == 1  # independent-only races: the independent's margin over R
    draws = np.array([-6.0, -2.0, 1.0, 3.0, 8.0])
    low, high = np.quantile(predictions.pick_sign("House", "R") * draws, predictions.LEVELS)
    assert low < 0 < high and high == pytest.approx(-np.quantile(draws, 0.10))
