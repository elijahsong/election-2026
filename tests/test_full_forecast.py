"""The full forecast's adjustment layers, custom Senate weights, ratings, and the deterministic submission writer.
All data here is SYNTHETIC."""

import congress_track
import numpy as np
import pandas as pd
import pytest

from congress import adjustments, report
from house import forecast as house_forecast
from senate import forecast as senate_forecast
from senate import stage2

TEMPLATE = pd.DataFrame({"race_id": ["AA-01", "AA-02", "BB-SEN"], "chamber": ["House", "House", "Senate"]})


def test_custom_senate_weights_are_checked_and_used():
    key = {"year": 2026, "state": "AA", "class": 1, "special": 0}
    stage1 = pd.DataFrame([{**key, "prediction": 6.0, "actual": np.nan}])
    polls = pd.DataFrame([{**key, "poll_margin_dem": 2.0, "poll_available": True, "poll_source": "test"}])
    rated = pd.DataFrame([{**key, "sabato_rating": "Tossup", "cook_rating": "Tossup", "ie_rating": "Tossup",
                           "n_experts": 3}])
    weights = {"polling": {"fundamentals": 0.5, "polling": 0.5, "experts": 0.0},
               "no_polling": {"fundamentals": 1.0, "polling": 0.0, "experts": 0.0}}
    assert stage2.blend(stage1, polls, rated, weights=weights)["prediction"].item() == pytest.approx(4.0)
    for bad in [{"polling": weights["polling"]},  # a case missing
                {**weights, "polling": {"fundamentals": 0.5, "polling": 0.6, "experts": 0.0}},  # sums to 1.1
                {**weights, "no_polling": {"fundamentals": 0.5, "polling": 0.5, "experts": 0.0}}]:  # polls w/o polls
        with pytest.raises(ValueError):
            stage2.check_weights(bad)


SCALE = pd.DataFrame({"category": ["corruption", "sexual"], "vote_share_points": [8, 5], "margin_points": [16, 10],
                      "description": "", "source": "test"}).set_index("category")
HEADER = "race_id,category,party,margin_adjustment,reason\n"


def test_category_rows_take_signed_points_from_the_scale_and_sum_per_race(tmp_path):
    path = tmp_path / "adjustments.csv"
    path.write_text(HEADER + "AA-01,corruption,R,,R incumbent indicted\n"   # hurts R: +16 to the D margin
                             "AA-01,sexual,D,,D nominee harassment case\n"  # hurts D: -10
                             "AA-02,,,1.5,manual\n"
                             "BB-SEN,sexual,I,,the independent\n")          # independent-positive margin: -10
    table = adjustments.load_margin_adjustments(path, template=TEMPLATE, scale=SCALE)
    assert adjustments.by_race(table, "House").to_dict() == {"AA-01": 6.0, "AA-02": 1.5}
    assert adjustments.by_race(table, "Senate").to_dict() == {"BB-SEN": -10.0}
    assert table["manual_house"].tolist() == [False, False, True, False]


def test_adjustments_file_is_validated(tmp_path):
    path = tmp_path / "adjustments.csv"
    for body in ["AA-01,corruption,R,,x\nAA-01,corruption,R,,y\n",  # repeated
                 "AA-01,,,1,\n",                # no reason
                 "ZZ-09,,,1,x\n",               # not in the template
                 "AA-01,bribes,R,,x\n",         # not in the scale
                 "AA-01,corruption,,,x\n",      # a category without a party
                 "AA-01,corruption,R,3,x\n",    # a category with its own points
                 "AA-01,,,,x\n"]:               # a manual row without points
        path.write_text(HEADER + body)
        with pytest.raises(ValueError):
            adjustments.load_margin_adjustments(path, template=TEMPLATE, scale=SCALE)


def test_the_repo_scale_and_adjustments_load():
    scale = adjustments.load_adjustment_scale()
    assert (scale["margin_points"] == 2 * scale["vote_share_points"]).all()
    template = pd.read_csv(congress_track.TEMPLATE_PATH)
    adjustments.load_margin_adjustments(template=template, scale=scale)


def test_house_adjustments_move_simulated_margins_only():
    point = pd.DataFrame({"year": 2026, "state": "AA", "district": [1, 2], "treatment": ["simulated", "fixed"],
                          "prediction": [-1.0, np.nan]})
    out = house_forecast.adjust_margins(point, pd.Series({"AA-01": 3.0}))
    assert out["prediction"].tolist()[0] == 2.0 and out["prediction_model"].tolist()[0] == -1.0
    assert out["margin_adjustment"].tolist() == [3.0, 0.0] and np.isnan(out["prediction"].iloc[1])
    with pytest.raises(ValueError, match="decided by rule"):
        house_forecast.adjust_margins(point, pd.Series({"AA-02": 1.0}))
    with pytest.raises(ValueError, match="aren't in the forecast"):
        house_forecast.adjust_margins(point, pd.Series({"AA-03": 1.0}))


def test_senate_adjustments_move_the_center_and_keep_the_model():
    point = pd.DataFrame({"state": ["AK", "NE"], "center": [-0.6, -8.5]})
    out = senate_forecast.adjust_centers(point, pd.Series({"AK-SEN": 2.0}))
    assert out["center"].tolist() == pytest.approx([1.4, -8.5]) and out["center_model"].tolist() == [-0.6, -8.5]


def test_generic_ballot_shift_moves_only_the_forecast_year():
    frame = pd.DataFrame({"year": [2024, 2026, 2026], adjustments.GENERIC_BALLOT: [0.7, 8.7, 8.7]})
    shifted = adjustments.shift_generic_ballot(frame, -4.0)
    assert shifted[adjustments.GENERIC_BALLOT].tolist() == pytest.approx([0.7, 4.7, 4.7])
    assert adjustments.generic_ballot(shifted) == pytest.approx(4.7)


def test_ratings_follow_the_thresholds():
    probability = pd.Series([0.55, 0.60, 0.80, 0.95, 0.99])
    party = pd.Series(["D", "R", "D", "I", "R"])
    assert report.rate(probability, party).tolist() == ["Toss-up", "Lean R", "Likely D", "Safe I", "Safe R"]


def test_submission_is_validated_rounded_and_identical_on_rerun(tmp_path):
    rows = pd.DataFrame({"race_id": ["AA-01", "AA-02"], "chamber": "House", "predicted_winner_party": ["D", "R"],
                         "candidate_name": ["A", "B"], "win_probability": [0.6123456789, 0.9]})
    senate = pd.DataFrame({"race_id": ["BB-SEN"], "chamber": "Senate", "predicted_winner_party": ["R"],
                           "candidate_name": ["C"], "win_probability": [0.7]})
    paths = (tmp_path / "a.csv", tmp_path / "b.csv")
    out = congress_track.write_submission({"House": rows, "Senate": senate}, template=TEMPLATE, paths=paths)
    assert out["win_probability"].tolist()[0] == 0.612346
    first = paths[0].read_bytes()
    congress_track.write_submission({"House": rows, "Senate": senate}, template=TEMPLATE, paths=paths)
    assert paths[0].read_bytes() == first == paths[1].read_bytes()
    with pytest.raises(ValueError):  # a House race missing
        congress_track.write_submission({"House": rows.iloc[:1], "Senate": senate}, template=TEMPLATE, paths=paths)
