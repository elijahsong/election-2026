import math

import numpy as np
import pandas as pd
import pytest

import unemployment


def synthetic_rates(start="2020-01", end="2022-10"):
    """Monthly rates for every state and US: state s is base_s + 0.1 per month, US 4.0 + 0.05 per month."""
    months = pd.period_range(start, end, freq="M")
    t = np.arange(len(months))
    states = pd.DataFrame({f"{s}UR": 3.0 + i * 0.05 + 0.1 * t for i, s in enumerate(unemployment.STATES)},
                          index=months)
    national = pd.Series(4.0 + 0.05 * t, index=months)
    return unemployment.monthly_rates(states.round(1), national.round(1))


def test_monthly_rates_cover_every_state_dc_and_us():
    monthly = synthetic_rates()
    assert set(monthly["area"]) == set(unemployment.STATES) | {"US"} and len(unemployment.STATES) == 51
    assert monthly["month"].min() == unemployment.cpi_gas.FIRST_MONTH  # padded back to the first month, blank
    assert monthly.loc[monthly["month"] < pd.Period("2020-01", "M"), "unemployment_rate"].isna().all()


def test_features_use_september_and_compare_september_with_september():
    monthly = synthetic_rates()
    out = unemployment.state_features(monthly, {2022: "2022-11-08"}).set_index("state")
    rates = monthly.pivot(index="month", columns="area", values="unemployment_rate")
    sep, sep_before, jun = (rates.loc[pd.Period(m, "M")] for m in ["2022-09", "2021-09", "2022-06"])
    ca = out.loc["CA"]
    assert ca["unemp_data_month"] == "2022-09"
    assert ca["unemp_state_rate"] == sep["CA"] and ca["unemp_us_rate"] == sep["US"] == out.loc["TX", "unemp_us_rate"]
    assert ca["unemp_state_minus_us"] == round(sep["CA"] - sep["US"], 1)
    assert ca["log_unemp_state"] == round(math.log(sep["CA"]), 4)
    assert ca["log_rel_unemp"] == pytest.approx(math.log(sep["CA"] / sep["US"]), abs=1e-4)
    assert ca["unemp_change_12m"] == round(sep["CA"] - sep_before["CA"], 1)
    assert ca["unemp_change_3m"] == round(sep["CA"] - jun["CA"], 1)
    assert ca["unemp_change_12m_minus_us"] == round((sep["CA"] - sep_before["CA"]) - (sep["US"] - sep_before["US"]), 1)
    # an October jump, released after Election Day, changes nothing
    spiked = monthly.copy()
    spiked.loc[spiked["month"] == pd.Period("2022-10", "M"), "unemployment_rate"] += 5
    pd.testing.assert_frame_equal(unemployment.state_features(spiked, {2022: "2022-11-08"}),
                                  unemployment.state_features(monthly, {2022: "2022-11-08"}))


def test_rounding_removes_float_noise_and_negative_zero():
    months = pd.period_range("2022-06", "2022-09", freq="M")
    states = pd.DataFrame({f"{s}UR": [4.1, 4.1, 4.1, 4.3] for s in unemployment.STATES}, index=months)
    out = unemployment.state_features(unemployment.monthly_rates(states, pd.Series(4.3, index=months)),
                                      {2022: "2022-11-08"})
    assert out["unemp_change_3m"].eq(0.2).all()  # 4.3 - 4.1 is 0.19999... before rounding
    assert out["unemp_state_minus_us"].eq(0.0).all() and not np.signbit(out["unemp_state_minus_us"]).any()


def test_missing_months_leave_only_their_features_blank():
    monthly = synthetic_rates()
    gone = (monthly["area"] == "OH") & (monthly["month"] == pd.Period("2021-09", "M"))  # a month BLS didn't publish
    monthly.loc[gone, "unemployment_rate"] = np.nan
    out = unemployment.state_features(monthly, {2022: "2022-11-08"}).set_index("state")
    assert math.isnan(out.loc["OH", "unemp_change_12m"]) and math.isnan(out.loc["OH", "unemp_change_12m_minus_us"])
    assert out.loc["OH", unemployment.FEATURES].drop(["unemp_change_12m", "unemp_change_12m_minus_us"]).notna().all()
    assert out.drop(index="OH")[unemployment.FEATURES].notna().all().all()
    # a blank national month blanks the national and relative features of every state
    monthly.loc[(monthly["area"] == "US") & (monthly["month"] == pd.Period("2022-09", "M")), "unemployment_rate"] = np.nan
    out = unemployment.state_features(monthly, {2022: "2022-11-08"})
    assert out[["unemp_us_rate", "unemp_state_minus_us", "log_rel_unemp"]].isna().all().all()
    assert out["unemp_state_rate"].notna().all()


def test_attach_states_raises_for_missing_year_states():
    state_f = unemployment.state_features(synthetic_rates(), {2022: "2022-11-08"})
    keys = pd.DataFrame({"year": [2022, 2022], "state": ["AK", "AK"], "district": [0, 1]})
    out = unemployment.attach_states(keys, state_f)
    assert len(out) == 2 and out["unemp_state_rate"].nunique() == 1
    with pytest.raises(ValueError, match="2024"):
        unemployment.attach_states(keys.assign(year=2024), state_f)
