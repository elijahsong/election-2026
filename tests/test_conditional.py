"""Conditional forecasts (congress/conditional.py) on a small synthetic baseline: calls, probability targets, observed
margins, the input parser, and refusals."""

import numpy as np
import pandas as pd
import pytest

from congress import conditional as cond
from house import simulation as house_sim


def synthetic_baseline(n: int = 4_000, seed: int = 0) -> cond.Baseline:
    """SYNTHETIC: two House races sharing a national error (XX-01 simulated, XX-02 same-party) and one Senate race."""
    rng = np.random.default_rng(seed)
    national = rng.normal(0, 3, n)
    margins = np.column_stack([1.0 + national + rng.normal(0, 4, n), np.full(n, np.nan)]).astype(np.float32)
    winners = np.column_stack([np.where(margins[:, 0] > 0, house_sim.DEM, house_sim.GOP),
                               np.full(n, house_sim.DEM)]).astype(np.int8)
    senate = (-2.0 + rng.normal(0, 3, n) + rng.normal(0, 4, n)).astype(np.float32)[:, None]
    picks = pd.DataFrame({"race_id": ["XX-01", "XX-02", "XX-SEN"], "chamber": ["House", "House", "Senate"],
                          "predicted_winner_party": ["D", "D", "R"], "candidate_name": ["A", "B", "C"],
                          "win_probability": [(winners[:, 0] == house_sim.DEM).mean(), 0.7, (senate[:, 0] <= 0).mean()]})
    return cond.Baseline(picks=picks, house_ids=np.array(["XX-01", "XX-02"]),
                         house_treatment=np.array(["simulated", "same_party"]), house_winners=winners,
                         house_margins=margins, house_dem_seats=(winners == house_sim.DEM).sum(axis=1).astype(np.int16),
                         house_turnout=np.array([3.0, 1.0]),
                         house_loser_code=np.array([house_sim.GOP, 0], dtype=np.int8),
                         house_fixed_margin=np.array([np.nan, 100.0], dtype=np.float32),
                         senate_ids=np.array(["XX-SEN"]), senate_seat_type=np.array(["dem_nominee"]),
                         senate_sims=senate, senate_dem_caucus=(senate[:, 0] > 0).astype(np.int16))


def test_parse_forms():
    assert cond.parse("call", "ky-06=d") == cond.Condition("call", "KY-06", party="D")
    assert cond.parse("prob", "NC-11=D:0.93") == cond.Condition("prob", "NC-11", party="D", value=0.93)
    assert cond.parse("margin", "VA-02=R+2.5:3").value == -2.5
    assert cond.parse("margin", "NE-SEN=I+4:2").party == "I"
    assert cond.parse("national", "D+12:1.5") == cond.Condition("national", cond.NATIONAL_ID, value=12.0, sd=1.5)
    assert cond.parse("national", "R+2:1").value == -2.0
    with pytest.raises(cond.ConditionError):
        cond.parse("prob", "NC-11=D:1.5")


def test_call_prob_and_margin_weights():
    b = synthetic_baseline()
    w = cond.weights(b, [cond.Condition("call", "XX-01", party="R")])
    assert w["House"][b.house_winners[:, 0] == house_sim.DEM].sum() == 0  # every D win dropped
    assert np.allclose(w["Senate"], 1 / b.n_sims)  # the other chamber is untouched
    w = cond.weights(b, [cond.Condition("prob", "XX-01", party="D", value=0.9)])
    assert w["House"][b.house_winners[:, 0] == house_sim.DEM].sum() == pytest.approx(0.9, abs=1e-6)
    # an observed margin pulls the race's weighted mean margin toward it
    w = cond.weights(b, [cond.Condition("margin", "XX-01", party="D", value=6.0, sd=1.0)])
    assert (w["House"] * b.house_margins[:, 0]).sum() == pytest.approx(6.0, abs=0.5)
    table = cond.race_table(b, b, w).set_index("race_id")
    assert table.loc["XX-02", "conditioned_probability"] == 0.7  # same-party candidate picks never move
    assert table.loc["XX-01", "change"] > 0


def test_refusals():
    b = synthetic_baseline()
    with pytest.raises(cond.ConditionError, match="same-party"):
        cond.weights(b, [cond.Condition("call", "XX-02", party="D")])
    with pytest.raises(cond.ConditionError, match="No House simulation"):
        cond.weights(b, [cond.Condition("call", "XX-01", party="D"), cond.Condition("call", "XX-01", party="R")])
    with pytest.raises(cond.ConditionError, match="impossible"):
        cond.weights(b, [cond.Condition("call", "XX-01", party="D"),
                         cond.Condition("prob", "XX-01", party="D", value=0.5)])
    with pytest.raises(cond.ConditionError, match="No race"):
        cond.weights(b, [cond.Condition("call", "ZZ-09", party="D")])


def test_effective_sims_and_quantiles():
    assert cond.effective_sims(np.full(100, 0.01)) == pytest.approx(100)
    assert cond.weighted_quantile(np.array([1, 2, 3, 4]), np.array([0.1, 0.1, 0.1, 0.7]), 0.5) == 4


def test_national_popular_vote_moves_every_simulated_race():
    b = synthetic_baseline()
    pv = cond.implied_popular_vote(b)  # 3/4 of XX-01's margin + 1/4 of XX-02's +100
    assert np.allclose(pv, 0.75 * b.house_margins[:, 0] + 25, atol=1e-4)
    moved, w = cond.condition(b, [cond.Condition("national", cond.NATIONAL_ID, value=40.0, sd=1.0)])
    after = cond.implied_popular_vote(moved)
    assert after.mean() == pytest.approx(40.0, abs=0.1) and after.std() == pytest.approx(1.0, abs=0.1)
    assert cond.effective_sims(w["House"]) == pytest.approx(b.n_sims)  # applied as the national error, not weights
    dem = moved.house_margins[:, 0] > 0
    assert (moved.house_winners[dem, 0] == house_sim.DEM).all() and (moved.house_winners[~dem, 0] == house_sim.GOP).all()
    assert np.array_equal(moved.house_dem_seats, (moved.house_winners == house_sim.DEM).sum(axis=1))
    assert np.array_equal(moved.house_winners[:, 1], b.house_winners[:, 1])  # a race decided by rule doesn't move
    with pytest.raises(cond.ConditionError, match="One national"):
        cond.condition(b, [cond.Condition("national", cond.NATIONAL_ID, value=1.0, sd=1.0)] * 2)
