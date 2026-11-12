from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import senate.regression as sr
from senate import stage1
from senate.regression import lagged

PROCESSED = Path(__file__).resolve().parents[1] / "data/processed/senate"


def test_lags_are_the_two_prior_state_years_and_missing_is_not_zero():
    # SYNTHETIC state AA: elections 2010, 2012 (a regular and a special), 2016 (regular + special), 2018
    races = pd.DataFrame({"year": [2010, 2012, 2012, 2016, 2016, 2018], "state": "AA",
                          "class": [1, 2, 3, 1, 3, 2]})
    residuals = races.assign(residual=[0.0, 2.0, 4.0, 8.0, np.nan, 1.0])  # 2016's special has no residual
    state_years = lagged.state_year_residuals(races, residuals)
    assert state_years.set_index("year").loc[2012, "residual"] == 3.0  # the mean of the year's two races
    assert np.isnan(state_years.set_index("year").loc[2016, "residual"])  # one of 2016's races is missing

    out = lagged.lagged_performance(races, state_years).set_index(["year", "class"])
    # 2016 (either race): lags are 2012 and 2010, never the same-day 2016 race; a genuine 0 residual is kept
    assert (out.loc[(2016, 1), ["recent_year", "previous_year"]] == [2012, 2010]).all()
    assert out.loc[(2016, 3), lagged.LAG] == pytest.approx(0.75 * 3.0 + 0.25 * 0.0)
    # 2018: the recent election (2016) has no residual, so L is missing with a reason, never 0 or an older year
    assert np.isnan(out.loc[(2018, 2), lagged.LAG]) and "2016" in out.loc[(2018, 2), "lag_missing_reason"]
    assert not out.loc[(2010, 1), "lag_available"]  # no earlier elections


@pytest.mark.skipif(not (PROCESSED / "modeling.csv").exists(), reason="Senate outputs not built")
def test_lagged_performance_never_uses_the_target_year_or_later():
    data = sr.load_data()

    def lag_for(frame, year):
        res = lagged.oos_residuals(stage1, stage1.SENATE_S1, frame, range(2010, 2026, 2))
        lp = lagged.lagged_performance(frame, lagged.state_year_residuals(frame[frame["year"] <= 2024], res))
        return lp[lp["year"] == year].set_index(sr.KEY)[lagged.LAG]

    base = lag_for(data, 2018)
    shifted = data.assign(**{sr.TARGET: data[sr.TARGET].where(data["year"] < 2018, 99.0)})
    pd.testing.assert_series_equal(base, lag_for(shifted, 2018))  # results from 2018 on change nothing
    assert base.notna().sum() == 34


@pytest.mark.skipif(not (PROCESSED / "modeling.csv").exists(), reason="Senate outputs not built")
def test_neutral_lag_trains_on_s1s_rows_and_flags_the_zeros():
    data = sr.load_data()
    res = lagged.oos_residuals(stage1, stage1.SENATE_S1, data, range(2010, 2026, 2))
    lp = lagged.lagged_performance(data, lagged.state_year_residuals(data[data["year"] <= 2024], res))
    d = data.merge(lp[[*sr.KEY, lagged.LAG, "lag_available"]], on=sr.KEY, validate="one_to_one")
    s1 = stage1.rolling_predictions(stage1.SENATE_S1, eval_years=[2020], data=data)
    preds, coefs = lagged.fit_predict_lagged(d, stage1.SENATE_S1.formula, dict(stage1.SENATE_S1.fixed),
                                             s1[2020].predictions, [2020], missing_lag="neutral")
    assert coefs.loc[2020, "n_train"] == s1[2020].model.nobs  # the same training rows as S1
    assert not preds["lag_fallback"].any() and preds["lag_neutral"].any()
    assert d[lagged.LAG].isna().any()  # the stored lag stays missing; 0 only inside the fit


def test_incumbent_lag_follows_the_candidate_not_the_state():
    # SYNTHETIC: state AA. Candidate D1 wins 2010 and 2016 (class 1); the 2012 class-2 race (the other seat,
    # candidate D2) must not count for D1. In 2022 D1 is the incumbent again; R9 challenged in 2016, ran open in 2020.
    races = pd.DataFrame({"year": [2010, 2012, 2016, 2020, 2022], "state": "AA", "class": [1, 2, 1, 2, 1],
                          "inc_signed": [0, 0, 1, 0, 1]})
    nominees = races[["year", "state", "class"]].assign(
        dem_name=["d1", "d2", "d1", "d3", "d1"], dem_candidate_id=["D1", "D2", "D1", "D3", "D1"],
        gop_name=["r1", "r2", "r9", "r9", "r5"], gop_candidate_id=["R1", "R2", "R9", "R9", "R5"])
    residuals = races[["year", "state", "class"]].assign(residual=[4.0, -10.0, 8.0, 3.0, 1.0])
    out = lagged.incumbent_lag(races, nominees, residuals).set_index("year")
    assert out.loc[2022, lagged.INCUMBENT_LAG] == pytest.approx(0.75 * 8.0 + 0.25 * 4.0)  # D1's own 2016 and 2010
    assert out.loc[2022, "inc_prev_years"] == "2010,2016" and out.loc[2022, "inc_lag_flag"] == "two elections"
    assert out.loc[2016, "inc_lag_flag"] == "one election (2010)"  # only one earlier race, flagged
    assert out.loc[2016, lagged.INCUMBENT_LAG] == 4.0
    # open seats get no lag even when a nominee ran before (R9 in 2020), and the missing lag is NaN, not 0
    assert out.loc[2020, "inc_lag_flag"].startswith("neutral: open seat")
    assert np.isnan(out.loc[2020, lagged.INCUMBENT_LAG])
    # an earlier race without a residual doesn't count
    no_res = lagged.incumbent_lag(races, nominees, residuals.assign(residual=[np.nan, -10.0, np.nan, 3.0, 1.0]))
    assert no_res.set_index("year").loc[2022, "inc_lag_flag"] == "neutral: earlier races without a residual"


@pytest.mark.skipif(not (PROCESSED / "modeling.csv").exists(), reason="Senate outputs not built")
def test_incumbent_lag_never_uses_the_target_year_or_later():
    data = sr.load_data()
    nominees = pd.read_csv(PROCESSED.parent / "senate_fec.csv")

    def lag_for(frame):
        res = lagged.oos_residuals(stage1, stage1.SENATE_S1, frame, range(2010, 2026, 2))
        out = lagged.incumbent_lag(frame, nominees, res)
        return out[out["year"] == 2020].set_index(sr.KEY)[lagged.INCUMBENT_LAG]

    shifted = data.assign(**{sr.TARGET: data[sr.TARGET].where(data["year"] < 2020, 99.0)})
    base = lag_for(data)
    pd.testing.assert_series_equal(base, lag_for(shifted))
    assert base.notna().sum() == 28
