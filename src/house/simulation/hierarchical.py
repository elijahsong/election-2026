"""Three-level hierarchical error model for House Monte Carlo simulations.

For district d in state s, one simulated margin (D - R points) is

    M_d_sim = M_d_hat + N + S_s + D_d

    N    national error: one draw per simulation, shared by every district
    S_s  state error: one draw per state per simulation, shared by the state's districts
    D_d  district error: one draw per district per simulation, independent

Each component is tau * t_nu: a standard Student-t draw with nu degrees of freedom, centered at 0, times a scale tau.
A t scale is not a standard deviation: SD = tau * sqrt(nu / (nu - 2)), defined only for nu > 2. The standard
deviations are what estimate_error_sds measures from residuals; t_scale converts them to the scales simulate_margins
draws with.

Estimation (estimate_error_sds), from Stage 1's out-of-sample residuals r = actual - predicted, one per district-year:

- National: each election's mean residual rbar_y is ONE observation of N, so K elections give K observations, not one
  per district. N is centered at 0, and its variance is a pseudo-sample-size-weighted regularized moment estimator:

      sigma_N^2 = (n0 * s0^2 + sum_y rbar_y^2) / (n0 + K)

  a fixed prior SD s0 (NATIONAL_PRIOR_SD, 3 points) counted as n0 pseudo-elections (NATIONAL_PRIOR_ELECTIONS, 2),
  pooled with the K observed squared yearly means. It is not an exact Bayesian posterior. The prior is fixed, never
  tuned on backtest outcomes, and keeps the estimate defined and conservative with as few as one election. A yearly
  mean residual also averages the state and district noise of that year's districts (about 0.8 points SD in
  2018-2024), which is not subtracted, so rbar_y approximates rather than isolates the national shock.
- State and district: a one-way random-effects ANOVA of r - rbar_y, with state-years as the groups, pooled over years
  and states. sigma_D^2 is the within-group mean square. sigma_S^2 = (between mean square - within mean square) / n0,
  truncated at 0, where n0 is the average group size adjusted for unequal sizes. Each year's state means average to 0
  around its own mean, so the between mean square has G - K degrees of freedom (G state-years). With roughly 200
  state-years and 1,400 within degrees of freedom both are well determined, so neither is regularized further.

Optional factors (factors.py estimates them; H0 above is the reference and is unchanged by them). simulate_margins
draws a component for each key of `scales`, always national, state, district first, so with those three alone its
draws are exactly H0's. Two more may follow, in this order:

    R_g    "region": one draw per political region per simulation, shared by its states' districts (point["region"])
    g Z_d  "demographic": ONE national coefficient g per simulation times each district's fixed loading Z_d
           (point["demographic_loading"])
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVELS = ("national", "state", "district")
FACTORS = ("region", "demographic")  # optional components (factors.py)
NATIONAL_PRIOR_SD = 3.0       # points; fixed, not tuned on backtest outcomes
NATIONAL_PRIOR_ELECTIONS = 2  # the prior's weight, in pseudo-elections


def t_scale(sd: float, nu: float) -> float:
    """The Student-t scale whose standard deviation is `sd`: sd * sqrt((nu - 2) / nu)."""
    if not nu > 2:
        raise ValueError(f"nu must exceed 2 for a Student-t distribution to have a standard deviation; got {nu}")
    return sd * np.sqrt((nu - 2) / nu)


def estimate_error_sds(oos: pd.DataFrame, *, national_prior_sd: float = NATIONAL_PRIOR_SD,
                       national_prior_elections: float = NATIONAL_PRIOR_ELECTIONS) -> tuple[dict[str, float], pd.DataFrame]:
    """(standard deviation per level, a table of how each was estimated) from out-of-sample residuals.

    oos has one row per district-year with year, state, and residual (actual - predicted), e.g. the stacked
    Stage1Result.predictions of stage1.rolling_predictions, restricted to rows with an actual. See the module docstring
    for the estimators."""
    if oos["residual"].isna().any():
        raise ValueError("Residuals must be observed; drop rows without an actual first")
    residual, year, state = oos["residual"], oos["year"], oos["state"]

    # National: one observation per election
    yearly = residual.groupby(year).mean()
    k = len(yearly)
    if k < 1 or national_prior_elections < 0 or national_prior_elections + k <= 0:
        raise ValueError(f"Need at least one election and a non-negative prior weight; got {k} and "
                         f"{national_prior_elections}")
    var_national = ((national_prior_elections * national_prior_sd ** 2 + (yearly ** 2).sum())
                    / (national_prior_elections + k))

    # State and district: random-effects ANOVA on the residual net of its year's mean, groups = state-years
    deviation = residual - year.map(yearly)
    groups = deviation.groupby([year, state])
    size = groups.size()
    n, g = len(deviation), len(size)
    ms_within = ((deviation - groups.transform("mean")) ** 2).sum() / (n - g)
    ms_between = (size * groups.mean() ** 2).sum() / (g - k)
    by_year = size.groupby(level=0)
    n0 = (by_year.sum() - by_year.apply(lambda m: (m ** 2).sum()) / by_year.sum()).sum() / (g - k)
    var_state = max((ms_between - ms_within) / n0, 0.0)
    var_district = ms_within

    sds = {"national": float(np.sqrt(var_national)), "state": float(np.sqrt(var_state)),
           "district": float(np.sqrt(var_district))}
    details = pd.DataFrame({
        "unit": ["election", "state-year", "district-year"],
        "observations": [k, g, n],
        "df": [k, g - k, n - g],
        "unregularized sd": [float(np.sqrt((yearly ** 2).mean())), sds["state"], sds["district"]],
        "regularization": [f"prior sd {national_prior_sd:g} x {national_prior_elections:g} pseudo-elections", "none",
                           "none"],
        "sd": list(sds.values()),
    }, index=pd.Index(LEVELS, name="level"))
    return sds, details


@dataclass
class Draws:
    """The raw error draws behind one simulate_margins call, for checks and decompositions. The region and
    demographic fields are None unless those components were drawn."""
    national: np.ndarray           # (n_sims, 1)
    state: np.ndarray              # (n_sims, len(states))
    states: np.ndarray             # the state codes, in the column order of `state`
    state_of_district: np.ndarray  # for each district (point's row order), its column in `state`
    district: np.ndarray           # (n_sims, n_districts)
    region: np.ndarray | None = None              # (n_sims, len(regions))
    regions: np.ndarray | None = None             # the region names, in the column order of `region`
    region_of_district: np.ndarray | None = None  # for each district, its column in `region`
    demographic: np.ndarray | None = None         # (n_sims, 1): the national coefficient g of each simulation
    loading: np.ndarray | None = None             # (n_districts,): each district's fixed loading Z_d

    def state_by_district(self) -> np.ndarray:
        """(n_sims, n_districts): each district's state error."""
        return self.state[:, self.state_of_district]

    def region_by_district(self) -> np.ndarray:
        """(n_sims, n_districts): each district's regional error (0 when no region component was drawn)."""
        if self.region is None:
            return np.zeros_like(self.district)
        return self.region[:, self.region_of_district]

    def demographic_by_district(self) -> np.ndarray:
        """(n_sims, n_districts): g * Z_d (0 when no demographic component was drawn)."""
        if self.demographic is None:
            return np.zeros_like(self.district)
        return self.demographic * self.loading

    def total(self) -> np.ndarray:
        """(n_sims, n_districts): the summed error N + S + D (+ R + g Z)."""
        return (self.national + self.state_by_district() + self.district + self.region_by_district()
                + self.demographic_by_district())


def simulate_margins(point: pd.DataFrame, scales: dict[str, float], nu: float, n_sims: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, Draws]:
    """(sims, draws): sims[i, j] = point's prediction j + N_i + S_i,state(j) + D_ij [+ R_i,region(j) + g_i Z_j], shape
    (n_sims, len(point)), columns in point's row order.

    point needs state and prediction (e.g. Stage1Result.predictions), plus region if scales has "region" and
    demographic_loading if it has "demographic". scales are Student-t scales per component (t_scale of the estimated
    SDs); national, state and district are required. Every draw comes from `rng`: national, state, district, then
    region, then demographic, so the first three are the same draws whichever optional components follow."""
    unknown = set(scales) - {*LEVELS, *FACTORS}
    if unknown:
        raise ValueError(f"Unknown error components {sorted(unknown)}")
    prediction = point["prediction"].to_numpy(dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Every district needs a finite prediction")
    states, state_of_district = np.unique(point["state"].to_numpy(), return_inverse=True)
    draws = Draws(
        national=scales["national"] * rng.standard_t(nu, size=(n_sims, 1)),
        state=scales["state"] * rng.standard_t(nu, size=(n_sims, len(states))),
        states=states,
        state_of_district=state_of_district,
        district=scales["district"] * rng.standard_t(nu, size=(n_sims, len(prediction))),
    )
    if "region" in scales:
        draws.regions, draws.region_of_district = np.unique(point["region"].to_numpy(), return_inverse=True)
        draws.region = scales["region"] * rng.standard_t(nu, size=(n_sims, len(draws.regions)))
    if "demographic" in scales:
        draws.loading = point["demographic_loading"].to_numpy(dtype=float)
        if not np.isfinite(draws.loading).all():
            raise ValueError("Every district needs a finite demographic_loading")
        draws.demographic = scales["demographic"] * rng.standard_t(nu, size=(n_sims, 1))
    sims = prediction + draws.national + draws.state_by_district() + draws.district  # H0, summed as it always was
    if draws.region is not None:
        sims = sims + draws.region_by_district()
    if draws.demographic is not None:
        sims = sims + draws.demographic_by_district()
    return sims, draws
