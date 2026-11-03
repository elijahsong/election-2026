"""The regional and demographic error factors (house/simulation/factors.py) and their place in the engine and backtest.
All data here is SYNTHETIC unless a test says otherwise."""

import numpy as np
import pandas as pd
import pytest

from house import simulation

# 50 synthetic states in 10 regions of unequal size, 2-9 states each
STATES_PER_REGION = [2, 3, 4, 5, 6, 7, 8, 9, 3, 3]


def synthetic_panel(n_years, *, sd_region=0.0, sd_gamma=0.0, z_region_sd=0.0, sd_state=3.0, sd_district=5.0,
                    giant_region=False, seed=0):
    """SYNTHETIC residuals with known components: (oos, regions, loadings). States have 1-40 districts (or 40 for
    every state of the largest region, giant_region=True); residual = N + R_g + S_s + g Z_d + D_d, all normal, with a
    3-point national SD. Z is fixed across years and clustered by region when z_region_sd > 0."""
    rng = np.random.default_rng(seed)
    region = np.repeat(np.arange(len(STATES_PER_REGION)), STATES_PER_REGION)
    size = rng.integers(1, 41, len(region))
    if giant_region:
        size[region == 7] = 40
    names = [f"S{s:02d}" for s in range(len(region))]
    regions = pd.Series([f"G{g}" for g in region], index=names)
    state = np.repeat(np.arange(len(region)), size)
    z = (rng.normal(0, z_region_sd, 10)[region[state]] + rng.normal(0, 0.4, len(region))[state]
         + rng.normal(0, 0.8, len(state)))
    z = z - z.mean()
    key = pd.DataFrame({"state": np.array(names)[state], "district": np.concatenate([np.arange(1, n + 1) for n in size])})
    frames, loadings = [], []
    for y in range(n_years):
        residual = (rng.normal(0, 3) + rng.normal(0, sd_region, 10)[region[state]]
                    + rng.normal(0, sd_state, len(region))[state] + rng.normal(0, sd_gamma) * z
                    + rng.normal(0, sd_district, len(state)))
        frames.append(key.assign(year=2000 + 2 * y, residual=residual))
        loadings.append(key.assign(year=2000 + 2 * y, demographic_loading=z))
    order = ["year", "state", "district"]
    oos = pd.concat(frames, ignore_index=True)[[*order, "residual"]]
    return oos, regions, pd.concat(loadings, ignore_index=True)[[*order, "demographic_loading"]]


def mean_and_se(x: pd.Series) -> tuple[float, float]:
    return float(x.mean()), float(x.std() / np.sqrt(len(x)))


# --- inputs ----------------------------------------------------------------------------------------------------------

def test_the_real_region_file_assigns_each_of_the_50_states_once(tmp_path):
    regions = simulation.load_regions()  # data/reference/regional_stage2.csv
    assert len(regions) == 50 and regions.index.is_unique and regions.nunique() == 10
    table = pd.read_csv(simulation.REGIONS_PATH)
    with pytest.raises(ValueError, match="repeated \\['CA'\\]"):
        pd.concat([table, table[table["state"] == "CA"]]).to_csv(tmp_path / "dup.csv", index=False)
        simulation.load_regions(tmp_path / "dup.csv")
    with pytest.raises(ValueError, match="missing \\['WY'\\]"):
        table[table["state"] != "WY"].to_csv(tmp_path / "missing.csv", index=False)
        simulation.load_regions(tmp_path / "missing.csv")


def test_loadings_are_centered_each_year_keyed_by_district_and_only_set_units():
    rng = np.random.default_rng(0)
    predictors = pd.DataFrame({"year": np.repeat([2022, 2024], 20), "state": "AA",
                               "district": np.tile(np.arange(1, 21), 2), "college": rng.uniform(15, 60, 40)})
    loadings = simulation.demographic_loadings(predictors, column="college", reference_sd=10.0)
    assert loadings.groupby("year")["demographic_loading"].sum().abs().max() < 1e-12  # no national shift
    first = predictors[predictors["year"] == 2022]
    assert np.allclose(loadings["demographic_loading"].iloc[:20], (first["college"] - first["college"].mean()) / 10)
    shuffled = simulation.demographic_loadings(predictors.sample(frac=1, random_state=1), reference_sd=10.0)
    assert shuffled.merge(loadings, on=["year", "state", "district"])[
        ["demographic_loading_x", "demographic_loading_y"]].pipe(lambda d: np.allclose(d.iloc[:, 0], d.iloc[:, 1]))

    # the reference SD only rescales g: sigma_g doubles when Z halves, and nothing else moves
    oos, regions, z = synthetic_panel(4, sd_gamma=1.5, seed=3)
    settings = {"national_prior_sd": 3.0, "national_prior_elections": 2}
    sds, _ = simulation.estimate_factor_sds(oos, loadings=z, **settings)
    half, _ = simulation.estimate_factor_sds(oos, loadings=z.assign(demographic_loading=z["demographic_loading"] / 2),
                                             **settings)
    assert half["demographic"] == pytest.approx(2 * sds["demographic"])
    assert half["state"] == pytest.approx(sds["state"]) and half["district"] == pytest.approx(sds["district"])


# --- the engine --------------------------------------------------------------------------------------------------------

def factor_point() -> pd.DataFrame:
    return pd.DataFrame({"state": ["CA", "CA", "OR", "TX", "TX", "NY"], "district": [1, 2, 1, 1, 2, 1],
                         "region": ["West", "West", "West", "South", "South", "East"],
                         "demographic_loading": [1.2, -0.4, 0.3, -1.0, 0.5, 0.9],
                         "prediction": [10.0, -5.0, 2.0, -12.0, 1.0, 20.0]})


def test_factor_shocks_are_shared_at_the_right_level():
    point = factor_point()
    scales = {"national": 2.0, "state": 3.0, "district": 4.0, "region": 1.5, "demographic": 0.8}
    sds = {level: scale * np.sqrt(5 / 3) for level, scale in scales.items()}
    sims, draws = simulation.simulate_margins(point, scales, 5, 20_000, np.random.default_rng(42))
    print(simulation.check_engine(sims, draws, point, sds, scales, 5))

    region = draws.region_by_district()
    assert np.array_equal(region[:, 0], region[:, 2])  # CA and OR: different states, one region
    assert not np.allclose(region[:, 0], region[:, 3])  # West and South draw separately
    demographic = draws.demographic_by_district()
    assert draws.demographic.shape == (20_000, 1)  # one national coefficient per simulation
    assert np.allclose(demographic / point["demographic_loading"].to_numpy(), draws.demographic)
    state = draws.state_by_district()
    assert np.array_equal(state[:, 0], state[:, 1]) and not np.allclose(state[:, 0], state[:, 2])

    again, _ = simulation.simulate_margins(point, scales, 5, 20_000, np.random.default_rng(42))
    assert np.array_equal(sims, again)  # one seed, one result


def test_factor_draws_follow_each_district_when_rows_are_reordered():
    point = factor_point()
    scales = {"national": 2.0, "state": 3.0, "district": 4.0, "region": 1.5, "demographic": 0.8}
    order = [5, 3, 0, 4, 2, 1]
    _, draws = simulation.simulate_margins(point, scales, 5, 1_000, np.random.default_rng(7))
    _, shuffled = simulation.simulate_margins(point.iloc[order], scales, 5, 1_000, np.random.default_rng(7))
    # regions and states are drawn in sorted order, so each district keeps its own draws whatever the row order
    assert np.array_equal(shuffled.region_by_district(), draws.region_by_district()[:, order])
    assert np.array_equal(shuffled.state_by_district(), draws.state_by_district()[:, order])
    assert np.array_equal(shuffled.demographic_by_district(), draws.demographic_by_district()[:, order])


def test_h0_draws_are_exactly_the_reference_three_level_draws():
    point = factor_point().drop(columns=["region", "demographic_loading"])
    scales = {"national": 2.0, "state": 3.0, "district": 4.0}
    sims, _ = simulation.simulate_margins(point, scales, 5, 2_000, np.random.default_rng(11))
    # the pre-factor engine, written out: national, state (sorted), district, in that order from one generator
    rng = np.random.default_rng(11)
    states, index = np.unique(point["state"], return_inverse=True)
    national = 2.0 * rng.standard_t(5, size=(2_000, 1))
    state = 3.0 * rng.standard_t(5, size=(2_000, len(states)))
    district = 4.0 * rng.standard_t(5, size=(2_000, len(point)))
    assert np.array_equal(sims, point["prediction"].to_numpy() + national + state[:, index] + district)

    # the factors come after: H2 reuses H0's raw national, state and district draws (common random numbers)
    _, with_region = simulation.simulate_margins(factor_point(), {**scales, "region": 1.0}, 5, 2_000,
                                                 np.random.default_rng(11))
    assert np.array_equal(with_region.national, national) and np.array_equal(with_region.district, district)


# --- estimation: the regional moment is checked for bias, never assumed unbiased --------------------------------------

@pytest.mark.parametrize("giant_region", [False, True])
def test_regional_moment_recovers_a_known_regional_variance_with_unequal_states_and_regions(giant_region):
    oos, regions, _ = synthetic_panel(300, sd_region=2.0, giant_region=giant_region)
    moments = simulation.factor_moments(oos, regions=regions)
    corrected, se = mean_and_se(moments["region_moment"])
    raw, _ = mean_and_se(moments["region_c"])
    assert abs(corrected - 4.0) < 3 * se  # unbiased within simulation error
    assert abs(corrected - 4.0) < abs(raw - 4.0)  # the uncorrected pair moment runs low (5-17% here)
    assert (moments["region_a"] > 0).all()  # big regions are over-represented in same-region pairs


def test_regional_moment_is_zero_without_regional_shocks():
    oos, regions, _ = synthetic_panel(300)
    corrected, se = mean_and_se(simulation.factor_moments(oos, regions=regions)["region_moment"])
    assert abs(corrected) < 3 * se


def test_demographic_only_shocks_load_on_the_coefficient_not_the_region_in_h3():
    oos, regions, loadings = synthetic_panel(300, sd_gamma=1.5, z_region_sd=0.7)  # Z clustered by region
    h2 = simulation.factor_moments(oos, regions=regions)
    h3 = simulation.factor_moments(oos, regions=regions, loadings=loadings)
    gamma, gamma_se = mean_and_se(h3["gamma_moment"])
    assert abs(gamma - 1.5 ** 2) < 3 * gamma_se
    region_h3, se_h3 = mean_and_se(h3["region_moment"])
    region_h2, se_h2 = mean_and_se(h2["region_moment"])
    assert abs(region_h3) < 3 * se_h3  # net of the demographic fit: no regional variance
    assert region_h2 > region_h3 + 2 * se_h2  # H2 alone absorbs the regionally clustered part of the swing


def test_zero_priors_shrink_the_factor_variances_by_k_over_n0_plus_k():
    oos, regions, loadings = synthetic_panel(4, sd_region=2.0, sd_gamma=1.0, seed=5)
    settings = {"regions": regions, "loadings": loadings, "national_prior_sd": 3.0, "national_prior_elections": 2}
    free, _ = simulation.estimate_factor_sds(oos, regional_prior_elections=0, demographic_prior_elections=0,
                                             **settings)
    shrunk, details = simulation.estimate_factor_sds(oos, regional_prior_elections=3, demographic_prior_elections=3,
                                                     **settings)
    assert (details["cap"] == "").all()
    assert shrunk["region"] ** 2 == pytest.approx(free["region"] ** 2 * 4 / 7)
    assert shrunk["demographic"] ** 2 == pytest.approx(free["demographic"] ** 2 * 4 / 7)
    assert shrunk["national"] == free["national"]  # the national estimator and its prior are H0's


# --- the variance budget -----------------------------------------------------------------------------------------------

def test_factors_redistribute_h0_variance_and_never_add_to_it():
    oos, regions, loadings = synthetic_panel(4, sd_region=2.0, sd_gamma=1.0, seed=2)
    settings = {"national_prior_sd": 3.0, "national_prior_elections": 2}
    h0, h0_details = simulation.estimate_factor_sds(oos, **settings)
    reference, _ = simulation.estimate_error_sds(oos, **settings)
    assert h0 == reference  # no factors: H0's estimator itself
    for r, z in [(None, loadings), (regions, None), (regions, loadings)]:
        sds, details = simulation.estimate_factor_sds(oos, regions=r, loadings=z, **settings)
        print(simulation.check_variance_budget(sds, details, h0))
        assert sds["state"] < h0["state"]  # the factors' share came out of the state level


def test_caps_shrink_the_factors_and_keep_the_budget_exact():
    budget = 9.0 + 25.0
    for var_region, var_gamma, cut in [(2.0, 0.5, False), (12.0, 0.5, True), (0.0, 40.0, True), (8.0, 10.0, True)]:
        var_state, var_district, region, gamma, caps = simulation.allocate_budget(
            9.0, 25.0, var_region, var_gamma, between=0.2, within=0.8)
        assert var_state >= 0 and var_district >= 0
        assert var_state + var_district + region + gamma * (0.2 + 0.8) == pytest.approx(budget, rel=1e-12)
        assert bool(caps["region"] or caps["demographic"]) == cut
        assert region <= var_region and gamma <= var_gamma  # a cap only ever reduces the factors


def test_simulated_district_variance_is_the_same_under_every_hierarchy():
    oos, regions, loadings = synthetic_panel(4, sd_region=2.0, sd_gamma=1.0, seed=4)
    point = (oos[oos["year"] == 2006].assign(prediction=0.0)
             .pipe(simulation.attach_factors, regions=regions, loadings=loadings))
    settings = {"national_prior_sd": 3.0, "national_prior_elections": 2}
    variance = {}
    for name, r, z in [("H0", None, None), ("H1", None, loadings), ("H2", regions, None), ("H3", regions, loadings)]:
        sds, _ = simulation.estimate_factor_sds(oos[oos["year"] < 2006], regions=r, loadings=z, **settings)
        scales = {level: simulation.t_scale(sd, 5) for level, sd in sds.items()}
        sims, _ = simulation.simulate_margins(point, scales, 5, 20_000, np.random.default_rng(0))
        variance[name] = sims.var(axis=0).mean()  # an average district's simulated error variance
    for name in ["H1", "H2", "H3"]:
        assert variance[name] == pytest.approx(variance["H0"], rel=0.02), name


# --- the backtest --------------------------------------------------------------------------------------------------------

def synthetic_backtest_inputs():
    """SYNTHETIC 2018-2024 oos rows, races, regions and loadings for simulation.backtest."""
    oos, regions, loadings = synthetic_panel(4, sd_region=2.0, sd_gamma=1.0, seed=9)
    oos["year"] += 18
    loadings["year"] += 18
    rng = np.random.default_rng(1)
    oos["prediction"] = rng.uniform(-30, 30, len(oos))
    oos["actual"] = oos["prediction"] + oos["residual"]
    oos["competitive_ex_ante"] = oos["prediction"].abs() <= 10
    oos["competitive_ex_post"] = oos["actual"].abs() <= 10
    oos["config"] = "house_model_a"
    races = oos[["year", "state", "district"]].assign(
        race_type="dr", treatment="simulated", election_method="standard", loser_party="R",
        winner_party=np.where(oos["actual"] > 0, "D", "R"), winner_resolved=True, actual=oos["actual"],
        lineup_restored=False)
    return oos, races, regions, loadings


def run(oos, races, test_years, **factors):
    return simulation.backtest(oos, test_years, races=races, predictions=oos, nu=5, n_sims=2_000, seed=42,
                               house_size=len(races[races["year"] == 2024]), national_prior_sd=3.0,
                               national_prior_elections=2, **factors)


def test_backtest_factors_use_earlier_elections_only_and_h0_is_the_default():
    oos, races, regions, loadings = synthetic_backtest_inputs()
    h0 = run(oos, races, [2022, 2024])
    for year, result in h0.years.items():
        expected, _ = simulation.estimate_error_sds(oos[oos["year"] < year], national_prior_sd=3.0,
                                                    national_prior_elections=2)
        assert result.error_sd == expected
    h3 = run(oos, races, [2022, 2024], regions=regions, loadings=loadings)
    assert h3.components == ("national", "state", "district", "region", "demographic")
    result = h3.years[2024]
    assert result.districts[["state", "district"]].equals(
        oos.loc[oos["year"] == 2024, ["state", "district"]].reset_index(drop=True))  # every race in order
    assert result.districts["region"].equals(result.districts["state"].map(regions))
    print(simulation.check_engine_draws(result, nu=5, n_sims=2_000, seed=42))
    assert "sim sd of mean error" in h3.diagnostics()

    # changing 2022's residuals leaves 2022's own fit and draws alone, but moves 2024's
    shifted = oos.copy()
    in_2022 = shifted["year"] == 2022
    for column in ["actual", "residual"]:
        shifted.loc[in_2022, column] += np.linspace(-6, 6, in_2022.sum())
    again = run(shifted, races, [2022, 2024], regions=regions, loadings=loadings)
    assert again.years[2022].error_sd == h3.years[2022].error_sd
    assert np.array_equal(again.years[2022].panel_dem_seats, h3.years[2022].panel_dem_seats)
    assert again.years[2024].error_sd != h3.years[2024].error_sd

    # 2024's loadings never reach 2022
    moved = loadings.copy()
    moved.loc[moved["year"] == 2024, "demographic_loading"] *= -1
    flipped = run(oos, races, [2022, 2024], regions=regions, loadings=moved)
    assert np.array_equal(flipped.years[2022].districts["p_dem"], h3.years[2022].districts["p_dem"])
    assert not np.array_equal(flipped.years[2024].districts["p_dem"], h3.years[2024].districts["p_dem"])

    rerun = run(oos, races, [2024], regions=regions, loadings=loadings)
    print(simulation.check_same_draws(rerun.years[2024], h3.years[2024]))


def test_group_seats_score_the_same_races_as_the_panel():
    oos, races, regions, loadings = synthetic_backtest_inputs()
    cal = run(oos, races, [2024], regions=regions)
    groups = races[["year", "state", "district"]].assign(region=races["state"].map(regions))
    table = simulation.group_seats(cal, "region", groups)
    result = cal.years[2024]
    assert table["races"].sum() == result.panel_size
    assert table["actual D seats"].sum() == result.panel_actual_dem_seats
    by_draw = sum((result.winners[:, (result.districts["region"] == g).to_numpy()] == simulation.DEM).sum(axis=1)
                  for g in regions.unique())
    assert np.array_equal(by_draw, result.panel_dem_seats)  # the groups partition the panel draw by draw
