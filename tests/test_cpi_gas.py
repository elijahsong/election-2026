import math

import numpy as np
import pandas as pd
import pytest

import acs
import cpi_gas


def test_every_state_has_one_census_region():
    states = set(acs.STATE_FIPS.values()) | {"DC"}
    assert set(cpi_gas.STATE_REGION) == states
    assert sum(map(len, cpi_gas.CENSUS_REGION_STATES.values())) == len(states)  # no state in two regions
    assert {r: len(s) for r, s in cpi_gas.CENSUS_REGION_STATES.items()} == \
        {"northeast": 9, "midwest": 12, "south": 17, "west": 13}
    assert [cpi_gas.STATE_REGION[s] for s in ["PA", "NJ", "MO", "ND", "DE", "MD", "TX", "OK", "AK", "HI", "NM"]] == \
        ["northeast", "northeast", "midwest", "midwest", "south", "south", "south", "south", "west", "west", "west"]


def test_data_month_is_the_last_month_public_before_election_day():
    # month M is published mid-month M + 1, so the October CPI (out mid-November) is never available on Election Day
    assert cpi_gas.data_month("2026-11-03") == pd.Period("2026-09", "M")
    assert cpi_gas.data_month("2022-11-08") == pd.Period("2022-09", "M")
    with pytest.raises(ValueError, match="after the first"):
        cpi_gas.data_month("2026-11-01")


def synthetic_monthly(start="2020-01", end="2022-10"):
    """Levels growing a fixed % a month per region (gas: a fixed $ a month), so every feature has a closed form."""
    months = pd.period_range(start, end, freq="M")
    t = np.arange(len(months))
    cpi_growth = {"us": 0.002, "northeast": 0.003, "midwest": 0.001, "south": 0.002, "west": 0.004}
    gas_step = {"us": 0.02, "northeast": 0.03, "midwest": 0.01, "south": 0.0, "west": 0.05}
    rows = [pd.DataFrame({"month": months, "region": r, "cpi": 100 * (1 + cpi_growth[r]) ** t,
                          "gas": 3.0 + gas_step[r] * t}) for r in cpi_gas.CPI_SERIES]
    return pd.concat(rows, ignore_index=True)


def test_features_from_the_data_month(monkeypatch):
    monkeypatch.setattr(cpi_gas, "REAL_PRICE_REFERENCE_MONTH", pd.Period("2022-09", "M"))
    monkeypatch.setattr(cpi_gas, "DEFLATOR_BASE_MONTH", pd.Period("2020-01", "M"))  # month 0: CPI 100, gas $3
    monthly = synthetic_monthly()
    out = cpi_gas.region_features(monthly, {2022: "2022-11-08"}).set_index("census_region")
    ne = out.loc["northeast"]
    assert ne["price_data_month"] == "2022-09" and ne["election_date"] == "2022-11-08"

    def growth(rate, months):
        return round(100 * ((1 + rate) ** months - 1), cpi_gas.DECIMALS)

    assert ne["cpi_region_yoy"] == growth(0.003, 12)
    assert ne["cpi_us_yoy"] == growth(0.002, 12) == out.loc["west", "cpi_us_yoy"]
    assert ne["cpi_region_minus_us_yoy"] == pytest.approx(growth(0.003, 12) - growth(0.002, 12), abs=1e-4)
    assert ne["cpi_region_24m"] == growth(0.003, 24)
    assert ne["cpi_region_6m"] == growth(0.003, 6)

    # September 2022 is month 32 of the series (January 2020 = 0): NE gas is 3 + 0.03 * 32
    sep = 3.0 + 0.03 * 32
    assert ne["gas_region_price_3m"] == pytest.approx(3.0 + 0.03 * 31)  # July-September mean
    assert ne["gas_region_yoy"] == round(100 * (sep / (sep - 0.03 * 12) - 1), cpi_gas.DECIMALS)
    assert ne["gas_region_6m_change"] == round(100 * (sep / (sep - 0.03 * 6) - 1), cpi_gas.DECIMALS)
    assert ne["gas_region_minus_us"] == pytest.approx(sep - (3.0 + 0.02 * 32))
    us_yoy = 100 * ((3.0 + 0.02 * 32) / (3.0 + 0.02 * 20) - 1)  # the national 12-month change
    assert ne["gas_region_minus_us_yoy"] == pytest.approx(100 * (sep / (sep - 0.03 * 12) - 1) - us_yoy, abs=1e-4)
    assert ne["gas_region_real_price"] == pytest.approx(sep)  # the reference month itself: real = nominal
    # an earlier month is restated in reference-month dollars with the region's own CPI
    early = cpi_gas.region_features(monthly, {2021: "2021-11-02"}).set_index("census_region").loc["northeast"]
    assert early["gas_region_real_price"] == pytest.approx((3.0 + 0.03 * 20) * 1.003 ** 12, abs=1e-4)

    # log features: real prices in base-month dollars, regional gas by regional CPI and national gas by national CPI
    us_sep = 3.0 + 0.02 * 32
    real_ne, real_us = sep / 1.003 ** 32, us_sep / 1.002 ** 32
    assert ne["cpi_region_log_yoy"] == pytest.approx(1200 * math.log(1.003), abs=1e-4)
    assert ne["cpi_region_log_diff_us"] == pytest.approx(1200 * (math.log(1.003) - math.log(1.002)), abs=1e-4)
    assert ne["log_real_gas_price"] == pytest.approx(math.log(real_ne), abs=1e-4)
    assert ne["log_rel_gas_price"] == pytest.approx(math.log(sep / us_sep), abs=1e-4)
    assert ne["log_rel_real_gas"] == pytest.approx(math.log(real_ne / real_us), abs=1e-4)
    # the real gap is the nominal gap less the regional - national inflation accumulated since the base month
    assert ne["log_rel_real_gas"] == pytest.approx(
        ne["log_rel_gas_price"] - 32 * (math.log(1.003) - math.log(1.002)), abs=2e-4)
    assert ne["log_gas_yoy"] == pytest.approx(100 * math.log(sep / (sep - 0.03 * 12)), abs=1e-4)
    assert ne["log_real_gas_yoy"] == pytest.approx(ne["log_gas_yoy"] - ne["cpi_region_log_yoy"], abs=2e-4)
    # the same region's log and percent changes agree to first order
    assert ne["cpi_region_log_yoy"] < ne["cpi_region_yoy"] < ne["cpi_region_log_yoy"] + 0.1


def test_every_year_compares_september_with_september():
    # a December election would move the data month to October and break year-to-year comparability
    with pytest.raises(ValueError, match="must use September"):
        cpi_gas.region_features(synthetic_monthly(), {2022: "2022-12-06"})


def test_october_levels_are_never_used():
    # an October spike, published after Election Day, must not move any feature
    monthly = synthetic_monthly()
    spiked = monthly.copy()
    spiked.loc[spiked["month"] == pd.Period("2022-10", "M"), ["cpi", "gas"]] *= 2
    a = cpi_gas.region_features(monthly, {2022: "2022-11-08"})
    b = cpi_gas.region_features(spiked, {2022: "2022-11-08"})
    pd.testing.assert_frame_equal(a, b)


def test_missing_months_leave_only_their_features_blank(monkeypatch):
    monkeypatch.setattr(cpi_gas, "REAL_PRICE_REFERENCE_MONTH", pd.Period("2022-09", "M"))
    monkeypatch.setattr(cpi_gas, "DEFLATOR_BASE_MONTH", pd.Period("2021-01", "M"))
    monthly = synthetic_monthly(start="2021-01")
    # an unpublished month (as October 2025): the 12-month base month and one month of the gas average
    missing = (monthly["region"] == "south") & (monthly["month"].isin([pd.Period("2021-09", "M"),
                                                                       pd.Period("2022-08", "M")]))
    monthly.loc[missing, ["cpi", "gas"]] = np.nan
    out = cpi_gas.region_features(monthly, {2022: "2022-11-08"}).set_index("census_region")
    south = out.loc["south"]
    assert math.isnan(south["cpi_region_yoy"]) and math.isnan(south["gas_region_yoy"])
    assert math.isnan(south["cpi_region_minus_us_yoy"]) and math.isnan(south["gas_region_price_3m"])
    assert math.isnan(south["gas_region_minus_us_yoy"])
    for column in ["cpi_region_log_yoy", "cpi_region_log_diff_us", "log_gas_yoy", "log_real_gas_yoy"]:
        assert math.isnan(south[column])
    assert south[["log_real_gas_price", "log_rel_gas_price", "log_rel_real_gas"]].notna().all()
    assert not math.isnan(south["cpi_region_6m"]) and not math.isnan(south["gas_region_real_price"])
    # 24 months back (September 2020) is before the data start: blank for every region, nothing else affected
    assert out["cpi_region_24m"].isna().all()
    assert out.loc["northeast", cpi_gas.FEATURES].drop("cpi_region_24m").notna().all()


def test_read_bls_series_report_keeps_blank_months(tmp_path):
    path = tmp_path / "SeriesReport.csv"
    path.write_text(
        "﻿Consumer Price Index for All Urban Consumers (CPI-U),,,\n"
        "Series Id:,CUUR0000SA0,,\n"
        ",,,\n"
        "Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,HALF1,HALF2\n"
        "2025,317.671,319.082,319.799,320.795,321.465,322.561,323.048,323.976,324.800,,324.122,324.054,320.229,324.000\n",
        encoding="utf-8")
    series = cpi_gas.read_bls_series_report(path, "CUUR0000SA0")
    assert len(series) == 12 and series[pd.Period("2025-09", "M")] == 324.8
    assert math.isnan(series[pd.Period("2025-10", "M")])
    with pytest.raises(ValueError, match="expected CUUR0100SA0"):
        cpi_gas.read_bls_series_report(path, "CUUR0100SA0")


def test_monthly_levels_fill_gaps_with_nan_and_trim_unpublished_months(monkeypatch):
    monkeypatch.setattr(cpi_gas, "FIRST_MONTH", pd.Period("2005-01", "M"))
    months = pd.PeriodIndex(["2005-01", "2005-03"], freq="M")
    national = pd.Series([190.7, np.nan], index=pd.PeriodIndex(["2005-01", "2005-04"], freq="M"))
    regional = pd.DataFrame({s: [100.0, 101.0] for r, s in cpi_gas.CPI_SERIES.items() if r != "us"}, index=months)
    gas = pd.DataFrame({s: [1.8, 1.9] for s in cpi_gas.GAS_SERIES.values()}, index=months)
    out = cpi_gas.monthly_levels(national, regional, gas)
    assert out["month"].astype(str).unique().tolist() == ["2005-01", "2005-02", "2005-03"]  # 2005-04 is all blank
    feb = out[out["month"] == pd.Period("2005-02", "M")]
    assert feb[["cpi", "gas"]].isna().all().all()
    assert out.loc[(out["region"] == "us") & (out["month"] == pd.Period("2005-01", "M")), "cpi"].item() == 190.7


def test_attach_regions_maps_states_and_raises_on_gaps():
    region_f = cpi_gas.region_features(synthetic_monthly(), {2022: "2022-11-08"})
    keys = pd.DataFrame({"year": [2022, 2022, 2022], "state": ["AK", "PA", "TX"], "district": [0, 1, 2]})
    out = cpi_gas.attach_regions(keys, region_f)
    assert out["census_region"].tolist() == ["west", "northeast", "south"] and len(out) == 3
    assert out.loc[0, "cpi_region_yoy"] == region_f.set_index("census_region").loc["west", "cpi_region_yoy"]
    with pytest.raises(ValueError, match="no Census region"):
        cpi_gas.attach_regions(keys.assign(state=["PR", "PA", "TX"]), region_f)
    with pytest.raises(ValueError, match="2024"):
        cpi_gas.attach_regions(keys.assign(year=2024), region_f)


def test_month_ranges():
    months = list(pd.period_range("2004-01", "2004-03", freq="M")) + [pd.Period("2025-10", "M")]
    assert cpi_gas.month_ranges(months) == "2004-01 to 2004-03, 2025-10"


def test_missing_deflator_base_month_blanks_only_the_real_log_features(monkeypatch):
    monkeypatch.setattr(cpi_gas, "DEFLATOR_BASE_MONTH", pd.Period("2019-01", "M"))  # before the synthetic data
    out = cpi_gas.region_features(synthetic_monthly(), {2022: "2022-11-08"})
    assert out[["log_real_gas_price", "log_rel_real_gas", "log_real_gas_yoy"]].isna().all().all()
    assert out[["log_rel_gas_price", "log_gas_yoy", "cpi_region_log_yoy"]].notna().all().all()


def test_reference_months_are_septembers():
    assert cpi_gas.DEFLATOR_BASE_MONTH.month == 9 and cpi_gas.REAL_PRICE_REFERENCE_MONTH.month == 9
