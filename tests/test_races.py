import numpy as np
import pandas as pd
import pytest

from house import simulation
from house.simulation import DEM, GOP, OTHER

SCALES = {"national": 2.0, "state": 3.0, "district": 4.0}
KEY = ["year", "state", "district"]
# SYNTHETIC lineup mix for one 435-district election
MIX = {"dr": 395, "dr_rcv": 3, "dr_jungle": 6, "dem_uncontested": 8, "gop_uncontested": 6, "dem_vs_3p": 6,
       "gop_vs_3p": 4, "dem_vs_dem": 6, "gop_vs_gop": 1}


def synthetic_races(seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC 2026-style races (attach_predictions output): MIX lineups over 435 districts in 50 states, with a
    Stage 1 margin on the simulated ones only."""
    rng = np.random.default_rng(seed)
    types = np.repeat(list(MIX), list(MIX.values()))
    races = pd.DataFrame({"year": 2026, "state": [f"S{i % 50:02d}" for i in range(435)],
                          "district": np.arange(435) // 50 + 1, "race_type": types})
    races["treatment"] = races["race_type"].map(simulation.TREATMENTS)
    simulated = races["treatment"] == "simulated"
    races["election_method"] = races["race_type"].map({"dr_rcv": "rcv", "dr_jungle": "jungle"}).fillna("standard")
    races["loser_party"] = np.where(simulated, "R", None)
    races["winner_party"], races["winner_resolved"], races["actual"], races["lineup_restored"] = None, False, np.nan, False
    races["prediction"] = np.where(simulated, rng.uniform(-30, 30, 435), np.nan)
    races["prediction_source"] = np.where(simulated, "ensemble", races["treatment"])
    return races


def test_every_district_gets_one_outcome_per_draw_and_overrides_hold():
    races = synthetic_races()
    draws = simulation.simulate_races(races, SCALES, 5, 4_000, np.random.default_rng(1))
    assert draws.winners.shape == (4_000, 435)
    assert np.isin(draws.winners, [DEM, GOP, OTHER]).all()
    assert (draws.seats(DEM) + draws.seats(GOP) + draws.seats(OTHER) == 435).all()

    by_type = {t: draws.winners[:, (races["race_type"] == t).to_numpy()] for t in MIX}
    assert (by_type["dem_uncontested"] == DEM).all() and (by_type["gop_uncontested"] == GOP).all()
    assert (by_type["dem_vs_dem"] == DEM).all() and (by_type["gop_vs_gop"] == GOP).all()
    # third parties only: the major party loses 0.5% of draws, always to OTHER, never to the other major party
    for t, party, rival in [("dem_vs_3p", DEM, GOP), ("gop_vs_3p", GOP, DEM)]:
        w = by_type[t]
        assert not (w == rival).any() and set(np.unique(w)) <= {party, OTHER}
        n = w.size
        upset = (w == OTHER).mean()
        sd = np.sqrt(simulation.THIRD_PARTY_UPSET_PROB * (1 - simulation.THIRD_PARTY_UPSET_PROB) / n)
        assert abs(upset - simulation.THIRD_PARTY_UPSET_PROB) < 4 * sd
    # no margin is invented for a race decided by rule; the simulated ones are exactly simulate_margins
    decided = (races["treatment"] != "simulated").to_numpy()
    assert np.isnan(draws.margins[:, decided]).all() and not np.isnan(draws.margins[:, ~decided]).any()
    sims, _ = simulation.simulate_margins(races[~decided], SCALES, 5, 4_000, np.random.default_rng(1))
    assert np.array_equal(draws.margins[:, ~decided], sims)
    # RCV and jungle keep the simulated party result by default
    rcv = (races["race_type"] == "dr_rcv").to_numpy()
    assert np.array_equal(draws.winners[:, rcv], np.where(draws.margins[:, rcv] > 0, DEM, GOP))

    again = simulation.simulate_races(races, SCALES, 5, 4_000, np.random.default_rng(1))
    assert np.array_equal(draws.winners, again.winners)  # one seed, one result


def test_rule_decided_races_never_carry_a_margin():
    races = synthetic_races().drop(columns=["prediction", "prediction_source"])
    primary = races[KEY].assign(prediction=1.0)  # Stage 1 predicts every race, the rule-decided ones too
    point = simulation.attach_predictions(races, primary, primary_source="ensemble")
    assert point.loc[point["treatment"] != "simulated", "prediction"].isna().all()
    # a simulated race the primary can't predict takes the fallback, labeled; without one it raises
    gap = primary.iloc[1:]
    point = simulation.attach_predictions(races, gap, primary_source="ensemble", fallback=primary)
    assert point.loc[0, "prediction_source"] == "house_model_a_fallback"
    with pytest.raises(ValueError, match="without a Stage 1 prediction"):
        simulation.attach_predictions(races, gap, primary_source="ensemble")


def test_an_independent_gop_proxy_wins_as_other_and_counts_as_r_only_in_a_scenario():
    races = synthetic_races().iloc[:2].copy()
    races["race_type"], races["treatment"], races["prediction"] = "dr", "simulated", [-50.0, 50.0]
    races["loser_party"] = ["other", "R"]  # race 0: the GOP proxy is an independent on the ballot (CA-6 2026)
    draws = simulation.simulate_races(races, SCALES, 5, 1_000, np.random.default_rng(2))
    assert (draws.winners[:, 0] == OTHER).mean() > 0.99 and not (draws.winners[:, 0] == GOP).any()
    key = tuple(races.iloc[0][KEY])
    assert draws.seats(GOP).max() == 0
    assert (draws.seats(GOP, alignment={key: GOP}) == draws.seats(OTHER)).all()


def test_resolvers_run_within_each_draw():
    class Coin(simulation.JungleSameParty):
        def resolve(self, races, winners, rng):  # SYNTHETIC rule: a same-party race goes OTHER in odd draws
            out = winners.copy()
            out[1::2] = OTHER
            return out

    races = synthetic_races()
    draws = simulation.simulate_races(races, SCALES, 5, 100, np.random.default_rng(3), resolvers=[Coin()])
    same = (races["treatment"] == "same_party").to_numpy()
    assert (draws.winners[0::2, same] != OTHER).all() and (draws.winners[1::2, same] == OTHER).all()
    default = simulation.simulate_races(races, SCALES, 5, 100, np.random.default_rng(3),
                                        resolvers=[simulation.JungleSameParty(), simulation.RankedChoice()])
    plain = simulation.simulate_races(races, SCALES, 5, 100, np.random.default_rng(3))
    assert np.array_equal(default.winners, plain.winners)  # the shipped resolvers keep the default behavior


def test_candidates_resolve_only_when_unambiguous():
    candidates = pd.DataFrame([
        {"candidate_name": "Ann", "party": "D"},
        {"candidate_name": "Bea", "party": "D"},
        {"candidate_name": "Kevin", "party": "other"},  # an independent, even if he is a model GOP proxy
        {"candidate_name": "Ray", "party": "R"},
    ])
    race = pd.Series({"race_type": "dem_vs_dem"})
    assert simulation.resolve_candidate(race, "R", candidates) == ("Ray", "resolved")
    assert simulation.resolve_candidate(race, "D", candidates) == (None, "several_D_candidates")
    assert simulation.resolve_candidate(race, "other", candidates) == ("Kevin", "resolved")
    assert simulation.resolve_candidate(race, "R", candidates.iloc[:3]) == (None, "no_registered_candidate")

    class PickFirst(simulation.JungleSameParty):
        def candidate(self, race, party, candidates):
            return candidates["candidate_name"].iloc[0]

    assert simulation.resolve_candidate(race, "D", candidates, [PickFirst()]) == ("Ann", "resolved_by_PickFirst")


def test_historical_lineups_use_pre_election_information():
    # SYNTHETIC targets rows: a result-based reclassification, a voided election, a write-in-only opponent
    targets = pd.DataFrame([
        {"year": 2022, "state": "MT", "district": 2, "race_type": "gop_vs_3p", "winner_party": "R",
         "adj_margin_2p_norm": -23.4, "imputation_method": "state_mean_residual",
         "quality_flags": "major_party_third_treated_uncontested"},
        {"year": 2018, "state": "NC", "district": 9, "race_type": "gop_vs_3p", "winner_party": "R",
         "adj_margin_2p_norm": -7.2, "imputation_method": "state_mean_residual",
         "quality_flags": "voided_election_treated_uncontested"},
        {"year": 2022, "state": "AZ", "district": 8, "race_type": "gop_uncontested", "winner_party": "R",
         "adj_margin_2p_norm": -40.0, "imputation_method": "state_mean_residual",
         "quality_flags": "writein_treated_uncontested"},
        {"year": 2022, "state": "AZ", "district": 1, "race_type": "dr", "winner_party": "R",
         "adj_margin_2p_norm": -1.0, "imputation_method": None, "quality_flags": None},
    ])
    raw = targets[KEY].assign(race_type=["dr", "dr", "dr_writein", "dr"])
    races = simulation.historical_races(targets, raw).set_index(["state", "district"])
    assert races.loc[("MT", 2), ["race_type", "treatment", "winner_party"]].tolist() == ["dr", "simulated", "R"]
    assert races.loc[("NC", 9), "race_type"] == "dr" and not races.loc[("NC", 9), "winner_resolved"]
    assert races.loc[("AZ", 8), "treatment"] == "fixed"  # write-ins aren't on the ballot
    # observed margins only where the target was observed, never for a restored lineup
    assert races["actual"].notna().tolist() == [False, False, False, True]
    blanked = simulation.lineup_targets(targets).set_index(["state", "district"])
    assert blanked.loc[[("MT", 2), ("NC", 9)], ["adj_margin_2p_norm", "imputation_method"]].isna().all().all()
    assert blanked.loc[("AZ", 8), "imputation_method"] == "state_mean_residual"


def test_top_two_blend_weights_and_pick():
    # SYNTHETIC finalists: race 1 has a poll that flips the primary leader, race 2 has none
    finalists = pd.DataFrame([
        {"year": 2026, "state": "S1", "district": 1, "candidate_name": "Ann", "primary_rank": 1, "primary_share": 30.0,
         "primary_share_top_two": 52.0, "individual_contribution_advantage": 0.0, "poll_share_top_two": 30.0},
        {"year": 2026, "state": "S1", "district": 1, "candidate_name": "Bea", "primary_rank": 2, "primary_share": 27.7,
         "primary_share_top_two": 48.0, "individual_contribution_advantage": 0.0, "poll_share_top_two": 70.0},
        {"year": 2026, "state": "S1", "district": 2, "candidate_name": "Cal", "primary_rank": 1, "primary_share": 60.0,
         "primary_share_top_two": 60.0, "individual_contribution_advantage": 1.0, "poll_share_top_two": np.nan},
        {"year": 2026, "state": "S1", "district": 2, "candidate_name": "Dee", "primary_rank": 2, "primary_share": 40.0,
         "primary_share_top_two": 40.0, "individual_contribution_advantage": -1.0, "poll_share_top_two": np.nan},
    ])
    p = simulation.predict_top_two(finalists, weights_with_poll={"primary": 0.6, "fundraising": 0.3, "poll": 0.1},
                                   weights_no_poll={"primary": 0.6, "fundraising": 0.4}, points_per_log=5.0,
                                   base_sd=8.0, split_sd=0.15, nu=5).set_index("district")
    assert p.loc[1, "mean"] == pytest.approx(0.6 * 4 + 0.1 * -40)        # the poll flips it to Bea
    assert p.loc[2, "mean"] == pytest.approx(0.6 * 20 + 0.4 * 5)         # no poll: 60/40
    assert p.loc[2, "sd"] == pytest.approx(8.0)                          # nobody backed a third candidate
    assert p.loc[1, "sd"] > p.loc[2, "sd"]                               # 42% uncommitted widens it
    resolver = simulation.TopTwoSameParty(p.reset_index())
    race = pd.Series({"year": 2026, "state": "S1", "district": 1, "race_type": "dem_vs_dem"})
    candidates = pd.DataFrame({"candidate_name": ["Ann", "Bea"], "party": ["D", "D"]})
    assert simulation.resolve_candidate(race, "D", candidates, [resolver]) == ("Bea", "resolved_by_TopTwoSameParty")
