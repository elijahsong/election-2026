import numpy as np
import pandas as pd
import pytest

from senate.simulation import calibration, chamber, errors, track


def synthetic_oos(seed=0):
    # SYNTHETIC out-of-sample blend residuals, 2016-2022, 30 races a year, half polled
    rng = np.random.default_rng(seed)
    rows = [{"year": y, "state": f"S{i:02d}", "class": 1, "polling_available": i % 2 == 0} for y in range(2016, 2024, 2)
            for i in range(30)]
    df = pd.DataFrame(rows)
    df["prediction"] = rng.normal(0, 15, len(df))
    df["residual"] = rng.normal(0, 6, len(df)) + df["year"].map({2016: 2, 2018: -3, 2020: 1, 2022: 0})
    df["actual"] = df["prediction"] + df["residual"]
    return df.assign(center=df["prediction"], competitive_ex_ante=df["prediction"].abs() <= 10, config="test")


def test_national_variance_is_the_regularized_moment_and_s_m1_shrinks_to_pooled():
    oos = synthetic_oos()
    sds, _ = errors.estimate_error_sds(oos, "S-M1")
    yearly = oos.groupby("year")["residual"].mean()
    assert sds["national"] ** 2 == pytest.approx((2 * 9 + (yearly ** 2).sum()) / (2 + 4))
    one_year, _ = errors.estimate_error_sds(oos[oos["year"] == 2016], "S-M0")
    assert one_year["national"] == pytest.approx(np.sqrt((2 * 9 + yearly[2016] ** 2) / 3))  # prior-dominated
    for group in ["polled", "unpolled"]:
        mask = oos["polling_available"] if group == "polled" else ~oos["polling_available"]
        dev = oos["residual"] - oos["year"].map(yearly)
        raw = np.sqrt((dev[mask] ** 2).mean() * len(oos) / (len(oos) - 4))
        assert min(raw, sds["race"]) <= sds[f"race_{group}"] <= max(raw, sds["race"])


def test_backtest_fits_each_year_on_earlier_years_only():
    oos = synthetic_oos()
    base = calibration.backtest(oos, test_years=[2020], n_sims=500)
    later = oos.assign(residual=oos["residual"].where(oos["year"] < 2020, 99.0))
    assert base["sds"].loc[2020, "national"] == calibration.backtest(later, test_years=[2020],
                                                                     n_sims=500)["sds"].loc[2020, "national"]
    assert base["sds"].loc[2020, "train_years"] == "2016-2018"


def test_national_draw_is_shared_and_sds_match():
    point = pd.DataFrame({"center": np.zeros(3), "polling_available": True})
    sims = errors.simulate_margins(point, {"national": 3.0, "race": 0.0}, 200_000, np.random.default_rng(1))
    assert np.allclose(sims[:, 0], sims[:, 2]) and sims[:, 0].std() == pytest.approx(3.0, rel=0.03)


def test_independent_wins_are_never_democratic_seats():
    point = pd.DataFrame({"seat_type": ["dem_nominee"] * 34 + ["independent_only"]})
    sims = np.full((1, 35), -1.0)
    sims[0, 0], sims[0, 34] = 1.0, 1.0  # one D win, one independent win
    counts = chamber.count_seats(sims, point).iloc[0]
    assert counts["dem_party"] == 33 and counts["dem_aligned"] == 35 and counts["new_independents"] == 1
    assert counts["dem_aligned"] + counts["new_independents"] + counts["gop"] == 100
    best = chamber.count_seats(np.ones((1, 35)), point).iloc[0]
    assert best["dem_aligned"] == chamber.DEM_ALIGNED_HOLDOVER + 34


def test_s_m2_finds_a_shared_regional_shock_and_keeps_total_race_variance():
    # SYNTHETIC: 3 regions x 10 states, a regional shock per region-year plus race noise
    rng = np.random.default_rng(2)
    rows = []
    for y in range(2016, 2026, 2):
        shocks = rng.normal(0, 4, 3)
        rows += [{"year": y, "state": f"S{i:02d}", "class": 1, "region": f"R{i % 3}",
                  "residual": shocks[i % 3] + rng.normal(0, 5)} for i in range(30)]
    oos = pd.DataFrame(rows)
    s_m0, _ = errors.estimate_error_sds(oos, "S-M0")
    s_m2, _ = errors.estimate_error_sds(oos, "S-M2")
    assert s_m2["regional"] > 1.0
    assert s_m2["race"] ** 2 + s_m2["regional"] ** 2 == pytest.approx(s_m0["race"] ** 2)
    point = pd.DataFrame({"center": np.zeros(3), "region": ["A", "A", "B"], "polling_available": True})
    sims = errors.simulate_margins(point, {"national": 0.0, "race": 0.0, "regional": 2.0}, 1000,
                                   np.random.default_rng(3))
    assert np.allclose(sims[:, 0], sims[:, 1]) and not np.allclose(sims[:, 0], sims[:, 2])


def test_track_rows_name_the_predicted_winner_and_refuse_unresolved_independents():
    point = pd.DataFrame({"year": 2026, "state": ["AA", "BB", "CC"], "class": 2,
                          "seat_type": ["dem_nominee", "independent_only", "independent_only"]})
    nominees = pd.DataFrame({"year": 2026, "state": ["AA", "BB", "CC"], "class": 2, "dem_name": ["Dee", None, None],
                             "gop_name": ["Rae", "Rob", "Ron"], "ind_name": [None, None, None]})
    ballot = pd.DataFrame({"year": 2026, "state": ["BB", "CC", "CC"], "class": 2, "party_label": "Independent",
                           "name": ["Ivy", "Ian", "Ida"]})
    resolved = track.resolve_candidates(point, nominees, ballot)
    assert resolved["ind_candidate"].tolist()[1] == "Ivy" and resolved["independent_resolution"][2].startswith("unr")
    template = pd.DataFrame({"race_id": ["AA-SEN", "BB-SEN", "CC-SEN"], "chamber": "Senate"})
    sims = np.array([[1.0, 2.0, -1.0]] * 3 + [[-1.0, 2.0, -1.0]])  # AA: D 75%; BB: I 100%; CC: R 100%
    out = track.track_table(resolved, sims, template).set_index("race_id")
    assert out.loc["AA-SEN", ["predicted_winner_party", "candidate_name"]].tolist() == ["D", "Dee"]
    assert out.loc["AA-SEN", "win_probability"] == 0.75
    assert out.loc["BB-SEN", ["predicted_winner_party", "candidate_name"]].tolist() == ["I", "Ivy"]
    assert out.loc["CC-SEN", "predicted_winner_party"] == "R"
    with pytest.raises(ValueError, match="no resolved candidate"):
        track.track_table(resolved, np.array([[1.0, 2.0, 1.0]]), template)  # CC's independent can't be named
