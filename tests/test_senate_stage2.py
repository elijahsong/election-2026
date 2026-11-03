import numpy as np
import pandas as pd
import pytest

from senate import stage2


def _inputs(poll=3.0, ratings=("Lean D", "Tossup", "Likely R"), special=0):
    # SYNTHETIC: one race, 2024 AA class 1
    key = {"year": 2024, "state": "AA", "class": 1}
    stage1 = pd.DataFrame([{**key, "special": special, "prediction": 6.0, "actual": 4.0}])
    polls = pd.DataFrame([{**key, "special": special, "poll_margin_dem": poll, "poll_available": poll is not None,
                           "poll_source": "test"}])
    rated = pd.DataFrame([{**key, "special": special, "sabato_rating": ratings[0], "cook_rating": ratings[1],
                           "ie_rating": ratings[2], "n_experts": sum(r is not None for r in ratings)}])
    return stage1, polls, rated


def test_rating_margins_are_democratic_positive_and_weighted_40_30_30():
    _, _, rated = _inputs()
    assert stage2.expert_margin(rated).iloc[0] == pytest.approx(0.4 * 10 + 0.3 * 0 + 0.3 * -15)
    _, _, one_missing = _inputs(ratings=("Solid R", None, "Tilt D"))
    assert stage2.expert_margin(one_missing).iloc[0] == pytest.approx((0.4 * -31 + 0.3 * 5) / 0.7)


def test_weights_sum_to_one_and_missing_polls_use_the_no_polling_weights():
    polled = stage2.blend(*_inputs(poll=3.0)).iloc[0]
    unpolled = stage2.blend(*_inputs(poll=None)).iloc[0]
    for row in (polled, unpolled):
        assert row[["w_fundamentals", "w_polling", "w_experts"]].sum() == pytest.approx(1.0)
    experts = 0.4 * 10 + 0.3 * -15
    assert polled["prediction"] == pytest.approx(0.37 * 6 + 0.54 * 3 + 0.09 * experts)
    # no poll: 65/35, never a 0 poll margin weighted at 54%
    assert unpolled["weight_case"] == "no_polling" and unpolled["w_polling"] == 0
    assert np.isnan(unpolled["margin_polling"])
    assert unpolled["prediction"] == pytest.approx(0.65 * 6 + 0.35 * experts)
    # no ratings either: fundamentals only, flagged
    bare = stage2.blend(*_inputs(poll=None, ratings=(None, None, None))).iloc[0]
    assert bare["prediction"] == pytest.approx(6.0) and bare["fallback_reason"]


def test_joins_are_one_to_one_on_the_race_and_check_special():
    stage1, polls, rated = _inputs()
    with pytest.raises(pd.errors.MergeError):
        stage2.blend(stage1, pd.concat([polls, polls]), rated)
    with pytest.raises(ValueError, match="special"):
        stage2.blend(stage1, polls.assign(special=1), rated)
