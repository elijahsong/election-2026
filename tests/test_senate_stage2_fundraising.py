"""The experimental Senate fundraising signal (senate.stage2_fundraising): cutoff integrity, sign conventions, missing
fundraising, and an unchanged baseline."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from senate import stage2
from senate import stage2_fundraising as sf


def _fec(**overrides):
    # SYNTHETIC: one 2024 race; D raised $2M, R $1M, both through the 12G report (Oct 16)
    row = {"year": 2024, "state": "AA", "class": 1, "ind_name": None,
           "dem_match_status": "name_full", "gop_match_status": "name_full",
           "dem_fec_missing": False, "gop_fec_missing": False,
           "dem_receipts": 2e6, "gop_receipts": 1e6,
           "dem_coverage_end_date": "2024-10-16", "gop_coverage_end_date": "2024-10-16"}
    return pd.DataFrame([{**row, **overrides}])


def test_election_day():
    assert [sf.election_day(y) for y in (2016, 2022, 2026)] == [date(2016, 11, 8), date(2022, 11, 8),
                                                                  date(2026, 11, 3)]


def test_fundraising_advantage_is_democratic_positive_and_stabilized():
    row = sf.fundraising_table(_fec()).iloc[0]
    assert row["fund_status"] == "ok"
    assert row["fund_adv"] == pytest.approx(np.log(2.1e6 / 1.1e6)) and row["fund_adv"] > 0
    flipped = sf.fundraising_table(_fec(dem_receipts=1e6, gop_receipts=2e6)).iloc[0]
    assert flipped["fund_adv"] == pytest.approx(-row["fund_adv"])
    # a $0 opponent stays finite: log(2.1M / 0.1M)
    assert sf.fundraising_table(_fec(gop_receipts=0.0)).iloc[0]["fund_adv"] == pytest.approx(np.log(21))


def test_only_reports_ending_before_election_day_are_used():
    post = sf.fundraising_table(_fec(gop_coverage_end_date="2024-12-31")).iloc[0]
    assert post["fund_status"] == "post_election_only" and np.isnan(post["fund_adv"])
    assert sf.fundraising_table(_fec(dem_coverage_end_date="2024-11-05")).iloc[0]["fund_status"] == \
        "post_election_only"  # Election Day itself is too late
    # a summary that ends before Election Day (2026's) counts; so does a nominee who never filed ($0, no dates)
    assert sf.fundraising_table(_fec(gop_coverage_end_date="2024-10-30")).iloc[0]["fund_status"] == "ok"
    never_filed = sf.fundraising_table(_fec(gop_receipts=0.0, gop_coverage_end_date=None)).iloc[0]
    assert never_filed["fund_status"] == "ok"
    # but an amount without dates isn't
    assert sf.fundraising_table(_fec(gop_coverage_end_date=None)).iloc[0]["fund_status"] == "post_election_only"


def test_missing_fundraising_is_flagged_never_zero_money():
    cases = {
        "independent_race": _fec(ind_name="ORMAN", dem_match_status="no_nominee", dem_fec_missing=True,
                                 dem_receipts=0.0),
        "no_nominee": _fec(gop_match_status="no_nominee", gop_fec_missing=True, gop_receipts=0.0),
        "unresolved": _fec(dem_match_status="nominee_unresolved", dem_fec_missing=None, dem_receipts=None),
        "no_fec_record": _fec(dem_match_status="no_fec_record", dem_fec_missing=True, dem_receipts=0.0),
    }
    for status, fec in cases.items():
        row = sf.fundraising_table(fec).iloc[0]
        assert row["fund_status"] == status and np.isnan(row["fund_adv"]), status


def test_capped_adjustment_signs_caps_and_missing():
    fund_adv = pd.Series([3.0, -3.0, 0.1, np.nan, 3.0])
    polled = pd.Series([True, True, True, True, False])
    delta, uncapped = sf.capped_adjustment(fund_adv, slope=1.5, polling_available=polled)
    assert delta.tolist() == pytest.approx([0.5, -0.5, 0.15, 0.0, 1.0])
    assert np.isnan(uncapped.iloc[3])


def _baseline():
    # SYNTHETIC: stage2.blend output for one polled and one unpolled race
    stage1 = pd.DataFrame({"year": 2024, "state": ["AA", "BB"], "class": 1, "special": 0,
                           "prediction": [2.0, -20.0], "actual": [1.0, -25.0]})
    polls = stage1[["year", "state", "class", "special"]].assign(poll_margin_dem=[4.0, np.nan],
                                                                 poll_available=[True, False], poll_source="test")
    ratings = stage1[["year", "state", "class", "special"]].assign(sabato_rating=["Tossup", "Solid R"],
                                                                   cook_rating=None, ie_rating=None, n_experts=1)
    return stage2.blend(stage1, polls, ratings)


def test_four_component_blend_weights_and_missing_fundraising_keeps_the_baseline():
    base = _baseline()
    fundraising = pd.Series([10.0, -30.0])
    pred = sf.four_component_blend(base, fundraising)
    assert pred.iloc[0] == pytest.approx(0.35 * 2 + 0.54 * 4 + 0.09 * 0 + 0.02 * 10)
    assert pred.iloc[1] == pytest.approx(0.60 * -20 + 0.35 * -31 + 0.05 * -30)
    unchanged = sf.four_component_blend(base, pd.Series([np.nan, np.nan]))
    assert unchanged.tolist() == base["prediction"].tolist()


def test_calibration_uses_earlier_years_only():
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({"year": np.repeat([2018, 2020, 2022], 30), "fund_adv": rng.normal(size=90)})
    rows["residual"] = 1.2 * rows["fund_adv"] + rng.normal(size=90)
    rows["actual"] = 10 * rows["fund_adv"] + rng.normal(size=90)
    later = rows.assign(residual=np.where(rows["year"] >= 2022, 99.0, rows["residual"]),
                        actual=np.where(rows["year"] >= 2022, -99.0, rows["actual"]))
    assert sf.fit_adjustment_slope(later, 2022).params.equals(sf.fit_adjustment_slope(rows, 2022).params)
    assert sf.fit_fundraising_margin(later, 2022).params.equals(sf.fit_fundraising_margin(rows, 2022).params)
    assert sf.fit_adjustment_slope(rows, 2022).nobs == 60
    with pytest.raises(ValueError, match="No calibration rows"):
        sf.fit_adjustment_slope(rows, 2018)
