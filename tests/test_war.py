"""Split Ticket WAR: processed data (signs, matching), the lagged predictor (alignment, signs, leakage) and the neutral
missing value that leaves production Stage 1 unchanged."""

import warnings

import numpy as np
import pandas as pd
import pytest
import split_ticket_war as stw

import house.regression as hr
from congress import war
from house import stage1

KEY = ["year", "state", "district"]


@pytest.fixture(scope="module")
def war_races():
    return war.load_war()


def test_signs_are_democratic_positive(war_races):
    by_race = war_races.set_index(["year", "chamber", "geography"])
    assert by_race.loc[(2022, "House", "OH-09"), "war_dem"] == 15.6  # Kaptur over Majewski, "D+15.6"
    assert by_race.loc[(2022, "House", "NE-02"), "war_dem"] == -3.0  # Bacon, "R+3"
    assert by_race.loc[(2020, "Senate", "Maine"), "war_dem"] == -12.5  # Collins, "R+12.5"
    sign = np.where(war_races["war_label"].str.startswith("D"), 1, -1)
    assert np.allclose(war_races["war_dem"], sign * war_races["war_label"].str[2:].astype(float))


def test_matching_invariants(war_races):
    for side in ["dem", "gop"]:
        usable = war_races[war_races[f"{side}_usable"]]
        assert usable[f"{side}_candidate_id"].notna().all()
        assert usable[f"{side}_match"].isin(stw.GOOD_MATCHES).all()
    # an independent in WAR's Democratic column (no D nominee) makes the race unusable on both sides
    osborn = war_races.set_index(["year", "geography"]).loc[(2024, "Nebraska")]
    assert osborn["dem_slot"] == "not_dem_nominee" and not osborn["dem_usable"] and not osborn["gop_usable"]
    people = pd.read_csv(war.PEOPLE_PATH)
    assert not people["candidate_id"].isin(stw.DISTINCT_PEOPLE).any()  # father and son never linked
    rose = people.set_index("candidate_id")["person_id"]
    assert rose["H8NY11113"] == rose["H2NY11140"]  # Max Rose re-registered for 2022
    for chamber in ["House", "Senate"]:
        war.appearances(war_races, chamber)  # raises if a person has two usable races in one year


def synthetic_apps():
    """Person P (a Democrat) in 2018, 2020 and 2022; Q (a Republican) in 2022 against P (a rematch) and in 2024; S a
    Democrat in 2018 who runs as a Republican later."""
    return pd.DataFrame({
        "year": [2018, 2020, 2022, 2022, 2024, 2018],
        "geography": ["XX-01", "XX-01", "XX-01", "XX-01", "XX-01", "XX-02"],
        "side": ["dem", "dem", "dem", "gop", "gop", "dem"],
        "person_id": ["P", "P", "P", "Q", "Q", "S"],
        "war_centered": [4.0, 2.0, -1.0, -1.0, 5.0, 3.0],
    })


def lag_for(nominees):
    races = pd.DataFrame(nominees, columns=["year", "state", "district", "dem_person_id", "gop_person_id"])
    return war.lagged_war(races, synthetic_apps(), KEY).set_index(["year", "district"])


def test_lag_alignment_signs_and_rematches():
    out = lag_for([(2018, "XX", 1, "P", None), (2020, "XX", 1, "P", None), (2022, "XX", 1, "P", None),
                   (2024, "XX", 1, "P", "Q"), (2020, "XX", 2, None, "S"), (2026, "XX", 1, None, "Q")])
    assert np.isnan(out.loc[(2018, 1), war.LAG]) and not out.loc[(2018, 1), war.AVAILABLE]  # nothing earlier
    assert out.loc[(2020, 1), war.LAG] == 4.0 and out.loc[(2020, 1), "war_n_years"] == 1  # one earlier year
    assert out.loc[(2022, 1), war.LAG] == pytest.approx(0.75 * 2.0 + 0.25 * 4.0)  # 2020 most recent, then 2018
    # 2024 rematch: the shared 2022 race counts once (-1), then P's 2020 race; Q's own 2024 race is not earlier
    assert out.loc[(2024, 1), war.LAG] == pytest.approx(0.75 * -1.0 + 0.25 * 2.0)
    assert out.loc[(2024, 1), "war_from"] == "both"
    # a Republican's D-positive WAR stays D-positive; a party switcher's flips (S: +3 as a Democrat, now R)
    assert out.loc[(2026, 1), war.LAG] == pytest.approx(0.75 * 5.0 + 0.25 * -1.0)
    assert out.loc[(2020, 2), war.LAG] == -3.0 and out.loc[(2020, 2), "war_party_switch"]


def test_same_year_races_of_both_nominees_are_one_observation():
    apps = pd.DataFrame({"year": [2022, 2022], "geography": ["XX-01", "XX-02"], "side": ["dem", "gop"],
                         "person_id": ["P", "Q"], "war_centered": [4.0, -2.0]})
    races = pd.DataFrame({"year": [2024], "state": ["XX"], "district": [3], "dem_person_id": ["P"],
                          "gop_person_id": ["Q"]})
    out = war.lagged_war(races, apps, KEY).iloc[0]
    assert out[war.LAG] == 1.0 and out["war_n_years"] == 1  # mean of +4 and -2, one year


def test_no_leakage_from_the_race_year_or_later():
    base = lag_for([(2022, "XX", 1, "P", "Q")])
    apps = synthetic_apps()
    apps.loc[apps["year"] >= 2022, "war_centered"] += 100  # change every WAR of 2022 and later
    races = pd.DataFrame([(2022, "XX", 1, "P", "Q")], columns=["year", "state", "district", "dem_person_id",
                                                              "gop_person_id"])
    moved = war.lagged_war(races, apps, KEY).set_index(["year", "district"])
    assert moved.loc[(2022, 1), war.LAG] == base.loc[(2022, 1), war.LAG]
    real = war.chamber_lags("Senate")
    assert (real["war_recent_year"].dropna() < real.loc[real["war_recent_year"].notna(), "year"]).all()


def test_centering_is_by_chamber_year():
    rows = pd.DataFrame({"year": 2020, "chamber": "Senate", "geography": ["A", "B", "C"], "war_dem": [1.0, 3.0, 8.0],
                         "dem_usable": [True, True, False], "gop_usable": True,
                         "dem_person_id": ["a", "b", "c"], "gop_person_id": ["x", "y", "z"]})
    apps = war.appearances(rows, "Senate").set_index(["geography", "side"])
    assert apps.loc[("A", "dem"), "war_centered"] == -3.0  # 1 - mean(1, 3, 8): unusable rows count in the mean
    assert ("C", "dem") not in apps.index and ("C", "gop") in apps.index  # usability is per side


def test_neutral_missing_war_keeps_every_row():
    frame = pd.DataFrame({"year": [2024, 2024], "state": "XX", "district": [1, 2]})
    lags = pd.DataFrame({"year": [2024, 2024], "state": "XX", "district": [1, 2], war.LAG: [12.0, np.nan],
                         war.AVAILABLE: [True, False], "war_flag": ["one year (2022)", "neutral"], "war_from": "dem"})
    out = war.with_war_lag(frame, lags, KEY)
    assert len(out) == 2 and out[war.LAG].tolist() == [12.0, 0.0] and out[war.AVAILABLE].tolist() == [True, False]
    assert war.with_war_lag(frame, lags, KEY, cap=10)[war.LAG].tolist() == [10.0, 0.0]
    assert war.house_war_features(lags)[war.LAG].tolist() == [12.0, 0.0]


def test_production_model_a_uses_the_stored_lag_and_drops_no_rows():
    with warnings.catch_warnings(action="ignore"):
        predictors, targets = hr.load_predictors(), hr.load_targets()
    # modeling.csv's war_f tranche is src/congress/war.py's lag, the neutral 0 included, for every district-year
    stored = predictors[[*hr.KEY, war.LAG, war.AVAILABLE]].merge(
        war.house_war_features(war.chamber_lags("House")), on=hr.KEY, suffixes=("", "_src"), validate="one_to_one")
    assert len(stored) == len(predictors) and np.allclose(stored[war.LAG], stored[f"{war.LAG}_src"])
    assert stored[war.AVAILABLE].equals(stored[f"{war.AVAILABLE}_src"])

    assert "war_lag" in stage1.MODEL_A.formula
    pre_war = stage1.Component("A_polling", stage1.MODEL_A.formula.replace(" + war_lag", ""), fixed={"inc_signed": 2.5})
    window = stage1.expanding_window(stage1.HOUSE_MODEL_A, 2024)
    with warnings.catch_warnings(action="ignore"):
        production = stage1.fit_predict(stage1.HOUSE_MODEL_A, window, 2024, predictors=predictors, targets=targets)
        before = stage1.fit_predict(stage1.Stage1Config("pre_war", ((pre_war, 1.0),)), window, 2024,
                                    predictors=predictors, targets=targets)
    train = lambda r: r.components["A_polling"].predictions.query("role == 'train'")[hr.KEY].reset_index(drop=True)
    assert train(production).equals(train(before))  # the neutral 0 drops no training row
    assert production.predictions[hr.KEY].equals(before.predictions[hr.KEY])
    assert 0.4 < production.components["A_polling"].coefficients.loc["war_lag", "coef"] < 0.8
