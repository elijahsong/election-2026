import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from house import simulation


def synthetic_residuals(n_years=40, sd_state=3.0, sd_district=5.0, seed=0) -> pd.DataFrame:
    """SYNTHETIC residuals, not election data: 30 states of 1-8 districts per year, normal state and district errors,
    and a national error of exactly +2 or -2 points (so its estimator has a hand-checkable value)."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 9, size=30)
    rows = []
    for y in range(n_years):
        national = 2.0 if y % 2 else -2.0
        for s, size in enumerate(sizes):
            state = rng.normal(0, sd_state)
            for d in range(1, size + 1):
                rows.append({"year": 2000 + 2 * y, "state": f"S{s:02d}", "district": d,
                             "residual": national + state + rng.normal(0, sd_district)})
    df = pd.DataFrame(rows)
    # recenter each year's mean at exactly +-2, so the national estimate is hand-checkable
    year_mean = df.groupby("year")["residual"].transform("mean")
    df["residual"] += np.where(df["year"] % 4 == 2, 2.0, -2.0) - year_mean
    return df


def test_t_scale_is_the_scale_with_the_given_standard_deviation():
    assert simulation.t_scale(2.0, 5) == pytest.approx(2.0 * np.sqrt(3 / 5))
    assert simulation.t_scale(2.0, 1e9) == pytest.approx(2.0)  # large nu: the normal, where scale = SD
    with pytest.raises(ValueError, match="nu must exceed 2"):
        simulation.t_scale(2.0, 2)


def test_error_sds_use_one_observation_per_election_and_recover_state_and_district_spread():
    oos = synthetic_residuals()
    sds, details = simulation.estimate_error_sds(oos, national_prior_sd=3.0, national_prior_elections=2)
    k = 40
    # yearly means of +-2: (2 * 3^2 + K * 2^2) / (2 + K), the prior counted as 2 pseudo-elections
    assert sds["national"] == pytest.approx(np.sqrt((2 * 9 + k * 4) / (2 + k)))
    assert details.loc["national", ["observations", "df"]].tolist() == [k, k]
    assert details.loc["national", "unregularized sd"] == pytest.approx(2.0)
    assert sds["state"] == pytest.approx(3.0, rel=0.15) and sds["district"] == pytest.approx(5.0, rel=0.05)
    unregularized, _ = simulation.estimate_error_sds(oos, national_prior_sd=3.0, national_prior_elections=0)
    assert unregularized["national"] == pytest.approx(2.0)
    one_election, _ = simulation.estimate_error_sds(oos[oos["year"] == 2000])  # defined from a single election
    assert one_election["national"] == pytest.approx(np.sqrt((2 * 9 + 4) / 3))


def test_simulated_errors_are_shared_at_the_right_level():
    point = pd.DataFrame({"state": ["CA", "CA", "AK", "NY", "CA", "NY"], "district": [1, 2, 0, 1, 3, 2],
                          "prediction": [10.0, -5.0, -12.0, 20.0, 1.0, -3.0]})
    scales = {"national": 2.0, "state": 3.0, "district": 4.0}
    sims, draws = simulation.simulate_margins(point, scales, 5, 20_000, np.random.default_rng(42))

    assert sims.shape == (20_000, 6) and draws.national.shape == (20_000, 1)
    assert draws.state.shape == (20_000, 3) and list(draws.states) == ["AK", "CA", "NY"]
    assert np.allclose(sims, point["prediction"].to_numpy() + draws.national + draws.state_by_district()
                       + draws.district)
    by_state = draws.state_by_district()
    assert np.array_equal(by_state[:, 0], by_state[:, 1]) and np.array_equal(by_state[:, 0], by_state[:, 4])
    assert not np.allclose(by_state[:, 0], by_state[:, 3])  # CA and NY draw separately
    corr = np.corrcoef(draws.district, rowvar=False)
    assert np.abs(corr[~np.eye(6, dtype=bool)]).max() < 0.05
    # the t draws have SD = scale * sqrt(nu / (nu - 2)), not the scale
    assert draws.district.std() == pytest.approx(4.0 * np.sqrt(5 / 3), rel=0.05)

    again, _ = simulation.simulate_margins(point, scales, 5, 20_000, np.random.default_rng(42))
    assert np.array_equal(sims, again)  # one seed, one result


def synthetic_oos(config="house_model_a") -> pd.DataFrame:
    """SYNTHETIC Stage 1 out-of-sample rows for 2018-2024: predictions plus the synthetic residuals."""
    oos = synthetic_residuals(n_years=4).assign(year=lambda d: d["year"] + 18)  # 2018, 2020, 2022, 2024
    rng = np.random.default_rng(1)
    oos["prediction"] = rng.uniform(-30, 30, len(oos))
    oos["actual"] = oos["prediction"] + oos["residual"]
    oos["competitive_ex_ante"] = oos["prediction"].abs() <= 10
    oos["competitive_ex_post"] = oos["actual"].abs() <= 10
    return oos.assign(config=config)


def synthetic_races(oos: pd.DataFrame) -> pd.DataFrame:
    """SYNTHETIC race table for synthetic_oos: every race D vs R, won by the sign of its actual margin."""
    return oos[["year", "state", "district"]].assign(
        race_type="dr", treatment="simulated", election_method="standard", loser_party="R",
        winner_party=np.where(oos["actual"] > 0, "D", "R"), winner_resolved=True, actual=oos["actual"],
        lineup_restored=False)


def backtest(oos, test_years, races=None, **settings):
    """simulation.backtest of synthetic_oos rows, with their synthetic_races unless races are given."""
    races = synthetic_races(oos) if races is None else races
    return simulation.backtest(oos, test_years, races=races, predictions=oos, **settings)


def test_backtest_trains_on_earlier_elections_only_and_scores_matched_panels():
    oos = synthetic_oos()
    settings = {"nu": 5, "n_sims": 2_000, "seed": 42, "house_size": 90, "national_prior_sd": 3.0,
                "national_prior_elections": 2}
    calibration = backtest(oos, [2020, 2022, 2024], **settings)

    for year, result in calibration.years.items():
        assert result.training_years == [y for y in [2018, 2020, 2022] if y < year]  # never the test year
        expected, _ = simulation.estimate_error_sds(oos[oos["year"] < year], national_prior_sd=3.0,
                                                    national_prior_elections=2)
        assert result.error_sd == expected
        test = oos[oos["year"] == year].reset_index(drop=True)
        assert result.districts[["state", "district"]].equals(test[["state", "district"]])  # alignment
        assert result.districts["p_dem"].between(0, 1).all()
        assert result.panel_dem_seats.shape == (2_000,) and result.panel_dem_seats.max() <= len(test)
        assert result.panel_actual_dem_seats == int((test["actual"] > 0).sum())  # the same districts

    # leakage: changing 2022's results leaves 2022's own fit alone, but changes 2024's
    shifted = oos.copy()
    shifted.loc[shifted["year"] == 2022, ["actual", "residual"]] += 5
    again = backtest(shifted, [2022, 2024], **settings)
    assert again.years[2022].error_sd == calibration.years[2022].error_sd
    assert np.array_equal(again.years[2022].panel_dem_seats, calibration.years[2022].panel_dem_seats)
    assert again.years[2024].error_sd != calibration.years[2024].error_sd

    rerun = backtest(oos, [2022], **settings)  # one seed, one result, whatever else runs
    assert np.array_equal(rerun.years[2022].districts["p_dem"], calibration.years[2022].districts["p_dem"])
    assert list(calibration.table().index) == [2020, 2022, 2024]

    with pytest.raises(ValueError, match="No election before 2018"):
        backtest(oos, [2018], **settings)
    with pytest.raises(ValueError, match="mixes Stage 1 configurations"):
        backtest(pd.concat([oos, synthetic_oos("other").assign(year=2026)]), [2020], **settings)


def test_calibration_metrics():
    assert simulation.brier([1.0, 0.0, 0.5], [True, False, True]) == pytest.approx(0.25 / 3)
    assert simulation.seat_percentile(np.array([1, 2, 2, 3]), 2) == pytest.approx(50.0)  # mid-rank
    assert simulation.seat_percentile(np.array([1, 2, 2, 3]), 3) == pytest.approx(100 * (0.75 + 0.5 * 0.25))
    assert simulation.seat_percentile(np.array([1, 2, 2, 3]), 0) == 0.0
    assert simulation.seat_percentile(np.array([1, 2, 2, 3]), 4) == 100.0
    lower, upper = simulation.central_interval(np.arange(101, dtype=float)[:, None], 0.8)
    assert (lower[0], upper[0]) == pytest.approx((10.0, 90.0))


def test_seat_summary_uses_drawn_counts():
    draws = np.repeat(np.arange(100, 110), 10)  # 100 draws, ten each of 100..109
    summary = simulation.seat_summary(draws, actual=104)
    # inverted CDF: the smallest count with at least q of the draws at or below it; never a half seat
    assert summary["median"] == 104
    assert summary["intervals"] == {0.8: (100, 108), 0.95: (100, 109)}
    assert summary["percentile"] == pytest.approx(100 * (0.4 + 0.5 * 0.1))
    assert simulation.seat_summary(draws)["percentile"] is None


def test_seat_plot_matches_the_calibration_table():
    calibration = backtest(synthetic_oos(), [2022], nu=5, n_sims=2_000, seed=42, house_size=90,
                                      national_prior_sd=3.0, national_prior_elections=2)
    year = calibration.years[2022]
    row = calibration.table().loc[2022]
    _, ax = simulation.plot_seat_distribution(year.panel_dem_seats, year=2022,
                                                actual_seats=year.panel_actual_dem_seats,
                                                panel_size=year.panel_size, house_size=year.house_size)
    lines = {line.get_gid(): line.get_xdata()[0] for line in ax.lines}
    assert lines == {"median": row["median panel_dem_seats"], "actual": row["panel_actual_dem_seats"]}
    assert row["matched panel"] == f"{year.panel_size} of 90"

    # bars: one per drawn seat count, heights are the draws' shares, colored by the table's intervals
    bars = [b for b in ax.patches if b.get_gid() == "seat_bars"]
    seats = np.array([round(b.get_x() + b.get_width() / 2) for b in bars])
    heights = np.array([b.get_height() for b in bars])
    assert np.array_equal(seats, np.unique(year.panel_dem_seats))
    assert heights.sum() == pytest.approx(1.0)
    lo80, hi80 = map(int, row["80% panel seat interval"].split("–"))
    lo95, hi95 = map(int, row["95% panel seat interval"].split("–"))
    face = {s: matplotlib.colors.to_hex(b.get_facecolor()) for s, b in zip(seats, bars)}
    assert {face[s] for s in seats if lo80 <= s <= hi80} == {simulation.visualization.COLORS["inside_80"]}
    assert {face[s] for s in seats if lo95 <= s <= hi95 and not lo80 <= s <= hi80} <= {
        simulation.visualization.COLORS["inside_95"]}
    summary = next(t for t in ax.texts if t.get_gid() == "summary").get_text()
    assert f"Matched panel: {year.panel_size} of 90 districts" in summary
    assert f"percentile {row['actual-seat percentile']:.0f}" in summary

    # before the election: no actual line or percentile, and the panel is a forecast panel
    _, ax = simulation.plot_seat_distribution(year.panel_dem_seats, year=2026, panel_size=year.panel_size,
                                              house_size=90)
    assert [line.get_gid() for line in ax.lines] == ["median"]
    assert "Forecast panel" in next(t for t in ax.texts if t.get_gid() == "summary").get_text()
    with pytest.raises(ValueError, match="integer seat counts"):
        simulation.plot_seat_distribution(year.panel_dem_seats + 0.5, year=2022)


def test_reliability_bins_districts_by_win_probability():
    districts = pd.DataFrame({"p_dem": [0.0, 0.05, 0.2, 0.6, 0.65, 1.0],
                              "dem_won": [False, False, True, True, False, True]})
    table = simulation.reliability(districts, edges=(0, 0.1, 0.5, 1))
    assert table["n"].tolist() == [2, 1, 3]  # p_dem = 0 lands in the first bin
    assert table["forecast"].tolist() == pytest.approx([0.025, 0.2, 0.75])
    assert table["observed"].tolist() == pytest.approx([0.0, 1.0, 2 / 3])


def test_backtest_views_draw_the_seat_summaries():
    settings = {"nu": 5, "n_sims": 2_000, "seed": 42, "house_size": 90, "national_prior_sd": 3.0,
                "national_prior_elections": 2}
    calibration = {name: backtest(synthetic_oos(name), [2020, 2022], **settings)
                   for name in ["config_a", "config_b"]}

    _, ax = simulation.plot_seat_errors(calibration)
    lines = {line.get_gid(): line for line in ax.lines}
    for name, cal in calibration.items():
        for year, result in cal.years.items():
            seats, actual = result.seat_summary(), result.panel_actual_dem_seats
            for level in (0.8, 0.95):
                lower, upper = seats["intervals"][level]
                assert list(lines[f"{round(100 * level)}:{year}:{name}"].get_xdata()) == [lower - actual, upper - actual]
            assert lines[f"median:{year}:{name}"].get_xdata()[0] == seats["median"] - actual

    _, axes = simulation.plot_backtest_seat_distributions(calibration)
    assert axes.shape == (2, 2) and axes[0, 1].get_title(loc="left") == "2022 · config_a"
    assert all(ax.get_legend() is None and not [t for t in ax.texts if t.get_gid() != "summary"] for ax in axes.flat)
    assert axes[0, 0].get_xlim() == axes[1, 0].get_xlim()  # one x range per year

    _, axes = simulation.plot_reliability(calibration)
    assert axes.shape == (1, 2)


def test_fixed_training_years_for_the_separate_2018_test():
    oos = synthetic_oos()  # 2018-2024
    settings = {"nu": 5, "n_sims": 2_000, "seed": 42, "house_size": 90, "national_prior_sd": 3.0,
                "national_prior_elections": 2}
    early = backtest(oos, [2022], train_years=[2018], **settings)
    assert early.years[2022].training_years == [2018]
    expected, _ = simulation.estimate_error_sds(oos[oos["year"] == 2018], national_prior_sd=3.0,
                                                national_prior_elections=2)
    assert early.years[2022].error_sd == expected
    assert early.table().loc[2022, "stage 2 training"] == "2018"
    # same seed, same test year: the draws differ from the default fit only through the scales
    default = backtest(oos, [2022], **settings)
    assert default.years[2022].training_years == [2018, 2020]
    assert default.years[2022].districts[["state", "district"]].equals(early.years[2022].districts[["state", "district"]])

    with pytest.raises(ValueError, match="must all precede"):
        backtest(oos, [2020], train_years=[2018, 2020], **settings)
    with pytest.raises(ValueError, match="no rows in oos"):
        backtest(oos, [2020], train_years=[2016], **settings)


def test_calls_confusion_matrix_and_seats_agree():
    calibration = backtest(synthetic_oos(), [2020, 2022], nu=5, n_sims=2_000, seed=42, house_size=90,
                                      national_prior_sd=3.0, national_prior_elections=2)
    stats, wrong, seats = calibration.competitive_stats(), calibration.wrong_calls(), calibration.seats()
    for year, result in calibration.years.items():
        d = result.districts
        everything = stats.loc[(year, "all")]
        cells = ["called D, won D", "called D, won R", "called R, won D", "called R, won R"]
        assert everything[cells].sum() == everything["n"] == len(d)
        assert everything["wrong calls"] == everything["called D, won R"] + everything["called R, won D"]
        assert everything["wrong calls"] == (wrong["year"] == year).sum()
        assert (wrong["called"] != wrong["won"]).all()
        assert stats.loc[(year, "competitive_ex_post"), "n"] == d["competitive_ex_post"].sum()
        # seats: the calls add up to called D seats; the actual count is the D-won column total
        assert seats.loc[year, "called D seats"] == everything["called D, won D"] + everything["called D, won R"]
        assert seats.loc[year, "panel_actual_dem_seats"] == everything["called D, won D"] + everything["called R, won D"]
        assert seats.loc[year, "median panel_dem_seats"] == result.seat_summary()["median"]


def test_backtest_results_never_reach_the_draws_and_unresolved_races_are_not_scored():
    oos = synthetic_oos()
    settings = {"nu": 5, "n_sims": 2_000, "seed": 42, "house_size": 90, "national_prior_sd": 3.0,
                "national_prior_elections": 2}
    races = synthetic_races(oos)
    # a rule-decided race (no Stage 1 prediction needed) and an unresolved winner in 2022
    first = races.index[races["year"] == 2022][0]
    races.loc[first, ["race_type", "treatment", "loser_party", "actual"]] = ["dem_uncontested", "fixed", None, np.nan]
    second = races.index[races["year"] == 2022][1]
    races.loc[second, ["winner_party", "winner_resolved"]] = [None, False]
    calibration = backtest(oos, [2022], races=races, **settings)
    year = calibration.years[2022]
    d = year.districts
    assert year.panel_size == len(d) - 1 and year.unresolved == [f"{d.loc[1, 'state']}-{d.loc[1, 'district']}"]
    assert np.isnan(d.loc[0, ["prediction", "lower_80", "upper_80"]].astype(float)).all() and d.loc[0, "p_dem"] == 1
    assert year.panel_actual_dem_seats == int(d.loc[d["winner_resolved"], "dem_won"].sum())
    assert calibration.table().loc[2022, "matched panel"] == f"{len(d) - 1} of 90"
    assert calibration.competitive_stats().loc[(2022, "all"), "n"] == len(d) - 1
    assert calibration.competitive_stats().loc[(2022, "contested"), "n"] == len(d) - 2
    # leakage: the test year's winners and actual margins only score the draws, they never shape them
    flipped = races.copy()
    in_2022 = flipped["year"] == 2022
    flipped.loc[in_2022, "winner_party"] = flipped.loc[in_2022, "winner_party"].map({"D": "R", "R": "D"})
    flipped.loc[in_2022, "actual"] = -flipped.loc[in_2022, "actual"]
    again = backtest(oos, [2022], races=flipped, **settings).years[2022]
    assert np.array_equal(again.house_seats["D"], year.house_seats["D"])
    assert np.array_equal(again.districts["p_dem"], d["p_dem"])


def test_shared_checks_pass_on_a_valid_backtest_and_catch_an_invented_margin():
    oos = synthetic_oos().assign(train_end=lambda d: d["year"] - 2)  # each year fit through the election before
    settings = {"nu": 5, "n_sims": 2_000, "seed": 42, "house_size": len(oos[oos["year"] == 2022]),
                "national_prior_sd": 3.0, "national_prior_elections": 2}
    calibration = backtest(oos, [2022], **settings)
    inputs = simulation.Stage1Inputs(config=type("C", (), {"name": "house_model_a"})(), rolling={}, oos=oos,
                                     predictions=oos)
    races = synthetic_races(oos)
    kw = {"house_size": settings["house_size"], "n_sims": 2_000, "national_prior_sd": 3.0, "national_prior_elections": 2}
    assert simulation.check_backtest(calibration, inputs, races, **kw).startswith("backtest checks passed")
    assert simulation.check_engine_draws(calibration.years[2022], nu=5, n_sims=2_000, seed=42).startswith("engine")
    assert simulation.check_same_draws(calibration.years[2022], backtest(oos, [2022], **settings).years[2022])
    calibration.years[2022].districts.loc[0, "treatment"] = "fixed"  # a rule-decided race that still has a margin
    with pytest.raises(AssertionError, match="invented margin"):
        simulation.check_backtest(calibration, inputs, races, **kw)


def test_report_stacks_and_formats_calibration_tables():
    oos = synthetic_oos()
    settings = {"nu": 5, "n_sims": 500, "seed": 1, "house_size": 90, "national_prior_sd": 3.0,
                "national_prior_elections": 2}
    cals = {"a": backtest(oos, [2022], **settings), "b": backtest(oos, [2022], **settings)}
    flat = simulation.stack(cals, "table")
    assert flat.index.names == ["config", "test year"] and len(flat) == 2
    nested = simulation.stack({"main": cals}, "wrong_calls")
    assert nested.index.names == ["backtest", "config"]
    html = simulation.styled(flat).to_html()
    assert f"{flat.iloc[0]['Brier (all races)']:.4f}" in html
