"""Optional error factors for the hierarchical simulation: a political-region shock and a demographic shock.

    H0  baseline     N + S + D                 (hierarchical.estimate_error_sds, the reference implementation)
    H1  demographic  H0 + g Z_d
    H2  regional     H0 + R_g
    H3  full         H0 + R_g + g Z_d

For district d in state s, region g(s), election y: e_dsy = N_y + S_sy + D_dy [+ R_g(s),y] [+ g_y Z_d]. Each
component is a Student-t draw times a scale (hierarchical.t_scale); hierarchical.simulate_margins draws them.

Inputs
- Regions (load_regions): the user's 10 custom political regions, data/reference/regional_stage2.csv. Every one of the
  50 states is assigned exactly once. One common regional variance is estimated for all regions; no region gets its
  own scale.
- Loading (demographic_loadings): district college attainment (ACS `college`, % of adults 25+ with a bachelor's
  degree), Z_dy = (college_dy - mean of the year's 435 districts) / COLLEGE_REFERENCE_SD. Centering on the year's
  equally weighted district mean makes sum_d Z_d = 0, so a demographic draw moves the House-wide average by nothing and
  can't stand in for a national shift. The scale is a fixed 10 points (district SD 9.7-11.2 in 2006-2026); it only
  sets g's units (points of margin per 10 points of college share) and never changes a simulated margin, because
  sigma_g is estimated in the same units. The ACS vintage is the one Stage 1 uses for that election (composition,
  never results).

Estimation (estimate_factor_sds), from Stage 1's rolling OUT-OF-SAMPLE residuals of the training elections only:

1. The H0 budget. estimate_error_sds gives sigma_N, sigma_S0, sigma_D0 exactly as in H0. sigma_N, with its fixed
   3-point prior counted as 2 pseudo-elections, is used unchanged in every hierarchy.
2. Demographic. Per training year, g_hat_y is the slope of the residual on Z within state-years (state-year fixed
   effects, so a state or regional swing can't load onto it), with sampling variance se_y^2 = (within-fit residual
   mean square) / sum (Z - Zbar_sy)^2. g_y is one draw per election, so K training years give K observations:

       sigma_g^2 = max(0, sum_y (g_hat_y^2 - se_y^2)) / (n0_g + K)

   a prior at 0 counted as n0_g pseudo-elections (DEMOGRAPHIC_PRIOR_ELECTIONS, 3), the national estimator's form.
3. Regional. Residuals net of the year mean (and, in H3, net of g_hat_y (Z - Zbar_y)) are averaged by state-year:
   ubar_s, over n_s districts. In the model, two districts in DIFFERENT states share R only when their states are in
   the same region, so per year

       c_y = (district-pair-weighted mean of ubar_s ubar_s' over different-state pairs in the same region)
             - (the same over different-state pairs in different regions)

   measures how much more two states of a region co-move than two states of different regions. It is NOT unbiased
   on its own: centering on the year mean pulls every state toward the House-wide average, which over-represents big
   regions and big states. Exactly, E[c_y] = sigma_R^2 (1 - a_y) - sigma_S^2 b_y, where a_y and b_y are the same
   same-minus-different pair means of (w_g + w_g') and (v_s + v_s'), w_g and v_s the region's and state's shares of
   the year's districts (design constants: no outcome enters them). With sigma_S^2 = sigma_S0^2 - sigma_R^2 (H0's
   state level holds the regional one), each year's corrected moment is

       m_y = (c_y + sigma_S0^2 b_y) / (1 - a_y + b_y)

   (in H3 the small b_y term still uses sigma_S0^2, ignoring the demographic share of it). On synthetic data with
   1-40 districts per state and 2-9 states per region, c_y alone ran 5-17% below sigma_R^2; m_y is unbiased within
   simulation error (tests/test_simulation_factors.py). Then sigma_R^2 = max(0, sum_y m_y) / (n0_R + K), with
   n0_R = REGIONAL_PRIOR_ELECTIONS (3): region-years are not independent evidence the way their count suggests, so
   the effective sample is the K elections.
4. Redistribution. The factors take their variance OUT of the H0 levels they were hiding in, never on top of them:

       sigma_S^2 = sigma_S0^2 - sigma_R^2 - sigma_g^2 B      sigma_D^2 = sigma_D0^2 - sigma_g^2 W

   B and W are Z's between- and within-state-year mean squares over the training rows (T = B + W, its mean square
   around the year mean): the share of g Z that H0 attributed to states and to districts. So
   sigma_S^2 + sigma_D^2 + sigma_R^2 + sigma_g^2 T = sigma_S0^2 + sigma_D0^2 exactly, every time: an average district's
   error variance is H0's, only its correlations change. What shrinkage takes from a factor stays in its H0 level.
   Caps: if sigma_g^2 W > sigma_D0^2, sigma_g^2 is cut to fit; then if sigma_R^2 + sigma_g^2 B > sigma_S0^2, both
   are scaled down by the same factor to fit. `details` reports every cap.

Approximation, not corrected: the national estimator uses each election's mean residual, which also contains that
year's district-weighted average of the regional shocks, sum_g w_g R_g (the demographic one averages to about 0, Z
being centered). So part of the regional variance is already counted in sigma_N, and H0's state level correspondingly
holds a little less than the full sigma_S^2 + sigma_R^2. A regional draw likewise moves the simulated House-wide
average by sum_g w_g R_g, adding roughly sigma_R^2 sum_g w_g^2 (about 0.1-0.15 sigma_R^2 for these regions) to its
variance. sigma_N is left exactly as in H0 rather than reduced, so H2 and H3 carry slightly more national-average
variance than H0; calibration.Calibration.diagnostics reports the simulated SD of the House-wide mean error
("sim sd of mean error") for every hierarchy so the overlap is visible.
"""

import json

import numpy as np
import pandas as pd

from house.regression.data import ROOT
from house.regression.schema import KEY

from .hierarchical import estimate_error_sds

REGIONS_PATH = ROOT / "data/reference/regional_stage2.csv"
STATES = tuple(json.loads((ROOT / "data/reference/district_counts.json").read_text())["2024"])  # the 50 states
COLLEGE_REFERENCE_SD = 10.0       # points of college share per unit of Z; fixed, sets units only
REGIONAL_PRIOR_ELECTIONS = 3      # the zero prior's weight, in pseudo-elections; fixed, not tuned on backtests
DEMOGRAPHIC_PRIOR_ELECTIONS = 3


def load_regions(path=REGIONS_PATH) -> pd.Series:
    """state -> political region, from a CSV with columns state, region. Raises unless each of the 50 states is
    assigned exactly once and nothing else is listed."""
    table = pd.read_csv(path)
    repeated = sorted(table.loc[table["state"].duplicated(), "state"])
    missing, extra = sorted(set(STATES) - set(table["state"])), sorted(set(table["state"]) - set(STATES))
    if repeated or missing or extra or table["region"].isna().any():
        raise ValueError(f"Every state needs exactly one region: repeated {repeated}, missing {missing}, "
                         f"unknown {extra}, blank regions {int(table['region'].isna().sum())}")
    return table.set_index("state")["region"]


def demographic_loadings(predictors: pd.DataFrame, *, column: str = "college",
                         reference_sd: float = COLLEGE_REFERENCE_SD) -> pd.DataFrame:
    """KEY + demographic_loading: (column - the year's mean over its districts) / reference_sd, one row per
    district-year of predictors (e.g. house.regression.load_predictors(), every district 2006-2026)."""
    value = predictors[column]
    if value.isna().any():
        raise ValueError(f"{int(value.isna().sum())} district-years have no {column}")
    out = predictors[KEY].copy()
    out["demographic_loading"] = (value - value.groupby(predictors["year"]).transform("mean")) / reference_sd
    return out.reset_index(drop=True)


def college_terciles(loadings: pd.DataFrame) -> pd.DataFrame:
    """KEY + college_tercile (low, middle, high): each year's districts split into thirds by demographic_loading. A
    grouping for calibration.group_seats, known before the election."""
    out = loadings[KEY].copy()
    out["college_tercile"] = loadings.groupby("year")["demographic_loading"].transform(
        lambda z: pd.qcut(z, 3, labels=["1 low", "2 middle", "3 high"]).astype(str))
    return out


def attach_factors(races: pd.DataFrame, *, regions: pd.Series | None = None,
                   loadings: pd.DataFrame | None = None) -> pd.DataFrame:
    """races plus region (from the state) and/or demographic_loading (by KEY), for simulate_margins. Raises if a
    simulated race lacks one (rows without a treatment column count as simulated)."""
    out = races.copy()
    simulated = out["treatment"].eq("simulated") if "treatment" in out else pd.Series(True, index=out.index)
    if regions is not None:
        out["region"] = out["state"].map(regions)
        if out.loc[simulated, "region"].isna().any():
            raise ValueError(f"No region for states {sorted(set(out.loc[simulated & out['region'].isna(), 'state']))}")
    if loadings is not None:
        out = out.merge(loadings[[*KEY, "demographic_loading"]], on=KEY, how="left", validate="many_to_one")
        if out.loc[simulated.to_numpy(), "demographic_loading"].isna().any():
            raise ValueError("A simulated race has no demographic_loading")
    return out


def factor_moments(oos: pd.DataFrame, *, regions: pd.Series | None = None,
                   loadings: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per training election: the per-year observations behind estimate_factor_sds.

    demographic (with loadings): gamma (within-state-year slope of the residual on Z), gamma_se, gamma_moment
    (gamma^2 - se^2), and Z's B and W (between- and within-state-year mean squares). regional (with regions):
    region_c (same-region minus different-region state-pair covariance, net of the demographic fit when loadings are
    given too), its centering-bias constants region_a and region_b, the corrected region_moment, and region_years
    (regions observed). The correction's sigma_S0^2 is H0's state variance on the same residuals."""
    data = oos[[*KEY, "residual"]].copy()
    var_state0 = estimate_error_sds(oos)[0]["state"] ** 2  # the state SD doesn't depend on the national prior
    if regions is not None:
        data["region"] = data["state"].map(regions)
        if data["region"].isna().any():
            raise ValueError(f"No region for states {sorted(set(data.loc[data['region'].isna(), 'state']))}")
    if loadings is not None:
        data = data.merge(loadings[[*KEY, "demographic_loading"]], on=KEY, how="left", validate="one_to_one")
        if data["demographic_loading"].isna().any():
            raise ValueError("A training residual has no demographic_loading")

    rows = {}
    for year, d in data.groupby("year"):
        row = {"districts": len(d)}
        u = d["residual"] - d["residual"].mean()  # net of the year's mean, as for H0's state and district levels
        if loadings is not None:
            z = d["demographic_loading"]
            z_state = z.groupby(d["state"]).transform("mean")
            zc, rc = z - z_state, d["residual"] - d["residual"].groupby(d["state"]).transform("mean")
            sxx = float((zc ** 2).sum())
            gamma = float((zc * rc).sum() / sxx)
            df = len(d) - d["state"].nunique() - 1
            se2 = float(((rc - gamma * zc) ** 2).sum() / df / sxx)
            row |= {"gamma": gamma, "gamma_se": np.sqrt(se2), "gamma_moment": gamma ** 2 - se2,
                    "B": float(((z_state - z.mean()) ** 2).mean()), "W": float((zc ** 2).mean())}
            u = u - gamma * (z - z.mean())  # H3: the region moment is taken net of the year's demographic swing
        if regions is not None:
            states = pd.DataFrame({"u": u, "state": d["state"], "region": d["region"]}).groupby("state").agg(
                n=("u", "size"), mean=("u", "mean"), region=("region", "first"))
            share = states["n"].to_numpy() / len(d)                                    # v_s
            region_share = states["region"].map(states.groupby("region")["n"].sum() / len(d)).to_numpy()  # w_g
            ubar, region, n = states["mean"].to_numpy(), states["region"].to_numpy(), states["n"].to_numpy()
            i, j = np.nonzero(~np.eye(len(states), dtype=bool))  # every ordered pair of different states
            weight, same = n[i] * n[j], region[i] == region[j]  # district pairs between the two states

            def same_minus_different(x):
                return (np.average(x[same], weights=weight[same]) - np.average(x[~same], weights=weight[~same]))

            c = same_minus_different(ubar[i] * ubar[j])
            a = same_minus_different(region_share[i] + region_share[j])
            b = same_minus_different(share[i] + share[j])
            row |= {"region_c": c, "region_a": a, "region_b": b,
                    "region_moment": (c + var_state0 * b) / (1 - a + b), "region_years": int(len(set(region)))}
        rows[int(year)] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("year")


def allocate_budget(var_state0: float, var_district0: float, var_region: float, var_gamma: float, *,
                    between: float, within: float) -> tuple[float, float, float, float, dict[str, str]]:
    """(var_state, var_district, var_region, var_gamma, caps): H0's state and district variances with the factors'
    shares taken out, var_state = var_state0 - var_region - var_gamma * between and
    var_district = var_district0 - var_gamma * within. A factor never takes more than its level holds: var_gamma is
    cut first to fit within var_district0, then var_region and var_gamma are scaled by one factor to fit within
    var_state0, so var_state + var_district + var_region + var_gamma * (between + within) = var_state0 +
    var_district0 in every case. caps names each binding cap ("" when none)."""
    caps = {"region": "", "demographic": ""}
    if var_gamma * within > var_district0:
        var_gamma = var_district0 / within
        caps["demographic"] = "cut to fit the district budget"
    taken = var_region + var_gamma * between
    if taken > var_state0:
        shrink = var_state0 / taken
        var_region, var_gamma = var_region * shrink, var_gamma * shrink
        for name in caps:
            caps[name] = "; ".join(filter(None, [caps[name], f"scaled by {shrink:.3f} to fit the state budget"]))
    var_state = max(var_state0 - var_region - var_gamma * between, 0.0)  # 0 up to rounding when the cap binds
    return var_state, var_district0 - var_gamma * within, var_region, var_gamma, caps


def estimate_factor_sds(oos: pd.DataFrame, *, regions: pd.Series | None = None, loadings: pd.DataFrame | None = None,
                        national_prior_sd: float, national_prior_elections: float,
                        regional_prior_elections: float = REGIONAL_PRIOR_ELECTIONS,
                        demographic_prior_elections: float = DEMOGRAPHIC_PRIOR_ELECTIONS,
                        ) -> tuple[dict[str, float], pd.DataFrame]:
    """(SD per component, how each was estimated). With neither regions nor loadings this IS estimate_error_sds (H0).
    Otherwise region and/or demographic are added and the state and district SDs give up exactly the variance the
    factors take (module docstring). sds["demographic"] is sigma_g, the SD of the coefficient g, in points per unit Z.

    details, one row per component: unit, observations, unregularized sd (the factors' raw moment estimate), h0 sd,
    regularized sd (after the zero prior, before any cap), sd (allocated), district variance (its share of an
    average district's error variance: sd^2, or sigma_g^2 T), regularization, and cap (blank unless one bound)."""
    h0, h0_details = estimate_error_sds(oos, national_prior_sd=national_prior_sd,
                                        national_prior_elections=national_prior_elections)
    if regions is None and loadings is None:
        return h0, h0_details
    moments = factor_moments(oos, regions=regions, loadings=loadings)
    k = len(moments)
    var_state0, var_district0 = h0["state"] ** 2, h0["district"] ** 2

    var_region = raw_region = 0.0
    if regions is not None:
        raw_region = max(float(moments["region_moment"].mean()), 0.0)
        var_region = max(float(moments["region_moment"].sum()), 0.0) / (regional_prior_elections + k)
    var_gamma = raw_gamma = between = within = 0.0
    if loadings is not None:
        raw_gamma = max(float(moments["gamma_moment"].mean()), 0.0)
        var_gamma = max(float(moments["gamma_moment"].sum()), 0.0) / (demographic_prior_elections + k)
        weights = moments["districts"] / moments["districts"].sum()  # pooled over training rows, like the ANOVA
        between, within = float((weights * moments["B"]).sum()), float((weights * moments["W"]).sum())
    regularized = {"region": np.sqrt(var_region), "demographic": np.sqrt(var_gamma)}

    var_state, var_district, var_region, var_gamma, caps = allocate_budget(
        var_state0, var_district0, var_region, var_gamma, between=between, within=within)

    sds = {"national": h0["national"], "state": float(np.sqrt(var_state)), "district": float(np.sqrt(var_district))}
    rows = {level: {"unit": h0_details.loc[level, "unit"], "observations": h0_details.loc[level, "observations"],
                    "unregularized sd": h0_details.loc[level, "unregularized sd"], "h0 sd": h0[level],
                    "regularized sd": h0[level], "regularization": h0_details.loc[level, "regularization"], "cap": ""}
            for level in ("national", "state", "district")}
    rows["state"]["regularization"] = "H0 state budget, less the factors' between-state share"
    rows["district"]["regularization"] = "H0 district budget, less the demographic within-state share"
    if regions is not None:
        sds["region"] = float(np.sqrt(var_region))
        rows["region"] = {"unit": "region-year", "observations": int(moments["region_years"].sum()),
                          "unregularized sd": np.sqrt(raw_region), "h0 sd": np.nan,
                          "regularized sd": regularized["region"],
                          "regularization": f"prior sd 0 x {regional_prior_elections:g} pseudo-elections",
                          "cap": caps["region"]}
    if loadings is not None:
        sds["demographic"] = float(np.sqrt(var_gamma))
        rows["demographic"] = {"unit": "election", "observations": k, "unregularized sd": np.sqrt(raw_gamma),
                               "h0 sd": np.nan, "regularized sd": regularized["demographic"],
                               "regularization": f"prior sd 0 x {demographic_prior_elections:g} pseudo-elections; "
                                                 f"B {between:.3f}, W {within:.3f}",
                               "cap": caps["demographic"]}
    details = pd.DataFrame.from_dict(rows, orient="index").rename_axis("component")
    details["sd"] = pd.Series(sds)
    details["district variance"] = details["sd"] ** 2
    if loadings is not None:
        details.loc["demographic", "district variance"] = var_gamma * (between + within)
    return sds, details[["unit", "observations", "unregularized sd", "h0 sd", "regularized sd", "sd",
                         "district variance", "regularization", "cap"]]
