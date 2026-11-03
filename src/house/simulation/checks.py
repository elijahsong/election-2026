"""Stage 2 validation: the invariants every run must hold, checked the same way in every notebook.

Each check raises AssertionError (with what failed) and returns a one-line summary when everything holds.

    check_engine(sims, draws, point, error_sd, error_scale, nu)   the raw hierarchical draws (hierarchical.py),
                                                                  regional and demographic factors included
    check_variance_budget(sds, details, h0_sds)                   the factors redistribute H0's variance, never add
    check_lineup_run(inputs, predictors, targets)                 lineup_targets only adds predictions
    check_backtest(calibration, inputs, races, ...)               no future information, every race once, no
                                                                  invented margins, scored seats match
    check_engine_draws(result, nu, n_sims, seed)                  a backtest year's simulated races are exactly
                                                                  simulate_margins' draws
    check_same_draws(a, b)                                        reproducibility: two runs, identical draws
    check_forecast(table, draws, candidates, house_size)          the 2026 table: one outcome per race and draw, the
                                                                  overrides, ballot identities

Rerunning for a reproducibility check stays in the notebook, so its settings are visible there.
"""

import warnings

import numpy as np
import pandas as pd

from house import stage1
from house.regression.schema import KEY

from .calibration import Calibration, YearCalibration, out_of_sample, win_probability
from .factors import DEMOGRAPHIC_PRIOR_ELECTIONS, REGIONAL_PRIOR_ELECTIONS, estimate_factor_sds
from .hierarchical import simulate_margins
from .inputs import Stage1Inputs
from .races import RaceDraws


def check_engine(sims: np.ndarray, draws, point: pd.DataFrame, error_sd: dict, error_scale: dict, nu: float) -> str:
    """The raw draws of hierarchical.simulate_margins: shapes, the N + S + D (+ R + g Z) decomposition, sharing at each
    level (one national value per draw, one per state, one per region, one coefficient g per draw times each
    district's loading), independent district errors, and SDs near their targets (when the t has a finite fourth
    moment)."""
    n_sims, n = sims.shape
    assert n == len(point) and draws.national.shape == (n_sims, 1)
    assert draws.state.shape == (n_sims, point["state"].nunique()) and draws.district.shape == (n_sims, n)
    for level in error_scale:  # scales and SDs are distinct quantities
        assert np.isclose(error_scale[level] * np.sqrt(nu / (nu - 2)), error_sd[level]), level
    errors = sims - point["prediction"].to_numpy()
    assert np.allclose(errors, draws.total())  # exactly prediction + every component
    state_part = draws.state_by_district()
    national_part = errors - state_part - draws.district - draws.region_by_district() - draws.demographic_by_district()
    assert np.allclose(national_part, national_part[:, [0]])  # one national value per draw
    for s in range(len(draws.states)):  # identical within a state
        in_state = state_part[:, draws.state_of_district == s]
        assert np.array_equal(in_state, in_state[:, [0]].repeat(in_state.shape[1], axis=1))
    assert len(np.unique(draws.state[0])) == len(draws.states)  # a separate draw per state
    if "region" in error_scale:
        assert draws.region.shape == (n_sims, point["region"].nunique())
        assert (draws.regions[draws.region_of_district] == point["region"].to_numpy()).all()  # each district's own
        region_part = draws.region_by_district()
        for g in range(len(draws.regions)):  # identical within a region, across its states
            in_region = region_part[:, draws.region_of_district == g]
            assert np.array_equal(in_region, in_region[:, [0]].repeat(in_region.shape[1], axis=1))
        assert len(np.unique(draws.region[0])) == len(draws.regions)  # a separate draw per region
    if "demographic" in error_scale:
        assert draws.demographic.shape == (n_sims, 1)  # one national coefficient per draw
        assert np.array_equal(draws.loading, point["demographic_loading"].to_numpy(dtype=float))  # row-aligned
        assert np.allclose(draws.demographic_by_district(), draws.demographic * draws.loading[None, :])
    corr = np.corrcoef(draws.district[:, :50], rowvar=False)  # within 5 standard errors of 0
    assert np.abs(corr[~np.eye(len(corr), dtype=bool)]).max() < 5 / np.sqrt(n_sims)
    if nu > 4:
        for level in error_scale:
            if error_sd[level] > 0:
                assert np.isclose(getattr(draws, level).std(), error_sd[level], rtol=0.05), level
    regions = f", {len(draws.regions)} regions" if draws.regions is not None else ""
    return (f"engine checks passed: {n_sims:,} simulations x {n} districts in {len(draws.states)} states{regions}; "
            f"components {', '.join(error_scale)}")


def check_variance_budget(sds: dict, details: pd.DataFrame, h0_sds: dict) -> str:
    """factors.estimate_factor_sds against H0 on the same residuals: the national SD is H0's, every level is
    non-negative, and the variance an average district's error gets from state, district, region and demographic
    together is exactly H0's state + district variance. A binding cap is named in the summary."""
    assert sds["national"] == h0_sds["national"], "the national SD differs from H0's"
    assert all(sd >= 0 for sd in sds.values())
    if "district variance" not in details:  # H0's own details: no factors, the budget is its state and district
        details = details.assign(**{"district variance": pd.Series(sds) ** 2, "cap": ""})
    sub_national = details.loc[[c for c in details.index if c != "national"], "district variance"].sum()
    budget = h0_sds["state"] ** 2 + h0_sds["district"] ** 2
    assert np.isclose(sub_national, budget, rtol=1e-12), f"variance {sub_national:.4f} vs H0 budget {budget:.4f}"
    caps = details.loc[details["cap"].fillna("") != "", "cap"]
    cap_note = "; caps: " + "; ".join(f"{c} {t}" for c, t in caps.items()) if len(caps) else "; no cap binds"
    return (f"variance budget checks passed: components {', '.join(sds)}, sub-national variance {sub_national:.3f} "
            f"= H0's {budget:.3f}{cap_note}")


def check_lineup_run(inputs: Stage1Inputs, *, predictors: pd.DataFrame, targets: pd.DataFrame) -> str:
    """The lineup_targets run against a plain Stage 1 run: identical residuals (so the same Stage 2 fit) and the same
    prediction for every race both predict; the lineup run only adds predictions."""
    with warnings.catch_warnings(action="ignore"):
        plain = stage1.rolling_predictions(inputs.config, tuple(inputs.rolling), predictors=predictors,
                                           targets=targets)
    assert inputs.oos.equals(out_of_sample(plain)), "lineup_targets changed a residual"
    added = 0
    for year, result in plain.items():
        lineup = inputs.rolling[year].predictions
        both = result.predictions.merge(lineup, on=KEY, suffixes=("", "_lineup"))
        assert len(both) == len(result.predictions), f"{year}: the lineup run lost a prediction"
        assert np.array_equal(both["prediction"], both["prediction_lineup"]), f"{year}: a prediction changed"
        added += len(lineup) - len(result.predictions)
    return f"lineup run checks passed: {inputs.config.name}, residuals unchanged, {added} predictions added"


def check_backtest(calibration: Calibration, inputs: Stage1Inputs, races: pd.DataFrame, *, house_size: int,
                   n_sims: int, national_prior_sd: float, national_prior_elections: float,
                   train_years=None, regions: pd.Series | None = None, loadings: pd.DataFrame | None = None,
                   regional_prior_elections: float = REGIONAL_PRIOR_ELECTIONS,
                   demographic_prior_elections: float = DEMOGRAPHIC_PRIOR_ELECTIONS) -> str:
    """Every test year of a backtest:

    - no future information: Stage 2 trained on earlier elections only (train_years if given), its SDs are exactly
      estimate_factor_sds of those residuals (estimate_error_sds for H0, with the same regions and loadings), and
      every Stage 1 prediction was fit on earlier years;
    - every simulated race carries its own state's region and its own district-year's loading, when used;
    - every race of the year once, in races' order, with probabilities summing to 1 and seats summing to house_size;
    - margins only for simulated races (none invented for races decided by rule), and a prediction for each;
    - the observed seats are the resolved D winners."""
    oos, predictions = inputs.oos, inputs.predictions
    assert calibration.config == inputs.config.name and set(oos["config"]) == {calibration.config}
    for year, result in calibration.years.items():
        train = oos[oos["year"].isin(train_years)] if train_years is not None else oos[oos["year"] < year]
        assert result.training_years == sorted(int(y) for y in train["year"].unique())
        assert result.training_years and max(result.training_years) < year, f"{year}: trained on a later election"
        expected, _ = estimate_factor_sds(train, regions=regions, loadings=loadings,
                                          national_prior_sd=national_prior_sd,
                                          national_prior_elections=national_prior_elections,
                                          regional_prior_elections=regional_prior_elections,
                                          demographic_prior_elections=demographic_prior_elections)
        assert result.error_sd == expected, f"{year}: SDs aren't those of the training residuals"
        assert (predictions.loc[predictions["year"] == year, "train_end"] < year).all(), f"{year}: Stage 1 leak"

        d = result.districts
        year_races = races.loc[races["year"] == year, KEY].reset_index(drop=True)
        assert len(d) == house_size and d[KEY].equals(year_races), f"{year}: races misaligned"
        assert np.allclose(d[["p_dem", "p_gop", "p_other"]].sum(axis=1), 1)
        seats = result.house_seats
        assert seats["D"].shape == (n_sims,) and (seats["D"] + seats["R"] + seats["other"] == house_size).all()
        simulated = d["treatment"] == "simulated"
        assert d.loc[~simulated, ["prediction", "lower_80", "actual"]].isna().all().all(), f"{year}: invented margin"
        assert d.loc[simulated, "prediction"].notna().all()
        if regions is not None:
            assert d.loc[simulated, "region"].equals(d.loc[simulated, "state"].map(regions)), f"{year}: region"
        if loadings is not None:
            own = d.loc[simulated, KEY].merge(loadings, on=KEY, how="left", validate="one_to_one")
            assert np.array_equal(own["demographic_loading"], d.loc[simulated, "demographic_loading"]), f"{year}: Z"
        assert result.panel_actual_dem_seats == int((d["winner_party"].eq("D") & d["winner_resolved"]).sum())
    return (f"backtest checks passed: {calibration.config} ({', '.join(calibration.components)}), "
            f"{', '.join(map(str, calibration.years))}")


def check_engine_draws(result: YearCalibration, *, nu: float, n_sims: int, seed: int) -> str:
    """A backtest year's simulated races have exactly hierarchical.simulate_margins' win probabilities for the
    year's seed: the race handling leaves the engine's draws untouched."""
    contested = result.districts[result.districts["treatment"] == "simulated"]
    sims, _ = simulate_margins(contested, result.error_scale, nu, n_sims, np.random.default_rng([seed, result.year]))
    assert np.array_equal(win_probability(sims), contested["p_dem"]), f"{result.year}: draws differ from the engine"
    return f"engine draw checks passed: {result.year}, {len(contested)} simulated races"


def check_same_draws(a, b) -> str:
    """Two runs of the same settings gave identical draws: YearCalibration or RaceDraws pairs."""
    if isinstance(a, RaceDraws):
        assert np.array_equal(a.winners, b.winners), "winners differ"
    else:
        assert np.array_equal(a.districts["p_dem"], b.districts["p_dem"]), "p_dem differs"
        assert np.array_equal(a.panel_dem_seats, b.panel_dem_seats), "seat draws differ"
    return "reproducibility check passed: identical draws"


def check_forecast(table: pd.DataFrame, draws: RaceDraws, candidates: pd.DataFrame, *, house_size: int) -> str:
    """A forecast_table: one row per race and one outcome per race and draw, probabilities summing to 1, no margin for
    a race decided by rule, the fixed / same-party / third-party overrides, independent GOP proxies never counted R,
    and every resolved candidate on the ballot for the winning party, exactly."""
    assert len(table) == house_size and draws.winners.shape[1] == house_size
    assert np.allclose(table[["p_dem", "p_gop", "p_other"]].sum(axis=1), 1)
    decided = (table["treatment"] != "simulated").to_numpy()
    assert table.loc[decided, "prediction"].isna().all() and np.isnan(draws.margins[:, decided]).all()
    for race_type, column in [("dem_uncontested", "p_dem"), ("gop_uncontested", "p_gop"), ("dem_vs_dem", "p_dem"),
                              ("gop_vs_gop", "p_gop")]:
        assert table.loc[table["race_type"] == race_type, column].eq(1).all(), race_type
    for race_type, party, rival in [("dem_vs_3p", "p_dem", "p_gop"), ("gop_vs_3p", "p_gop", "p_dem")]:
        rows = table[table["race_type"] == race_type]
        assert rows[rival].eq(0).all() and rows[party].between(0.99, 1).all(), race_type  # upsets go to OTHER
    proxy = draws.races["loser_party"].eq("other").to_numpy()
    assert table.loc[proxy, "p_gop"].eq(0).all(), "an independent GOP proxy was counted R"
    resolved = table[table["candidate_status"] == "resolved"]
    named = resolved.merge(candidates, left_on=[*KEY, "predicted_candidate"], right_on=[*KEY, "candidate_name"],
                           validate="one_to_one")
    assert len(named) == len(resolved) and (named["party"] == named["predicted_party"]).all()
    return f"forecast checks passed: {draws.winners.shape[0]:,} draws x {house_size} races"
