"""Senate Monte Carlo errors: a national error and race errors, both Student-t, around the Stage 2 blend.

For race r in election y, one simulated margin (D-positive points) is

    M_r = prediction_r + N + e_r

    N    national error: one draw per simulation, shared by every race
    e_r  race error: one independent draw per race

S-M0 (the primary baseline) uses one race scale. S-M1 (an experiment) uses separate scales for polled and unpolled
races (polling_available, known before the election), both strongly shrunk toward the pooled scale. S-M2 (a
sensitivity experiment) is S-M0 plus a regional error R_g, one draw per region per simulation, shared by the region's
races (the 10 political regions of data/reference/regional_stage2.csv):

    M_r = prediction_r + N + R_g(r) + e_r

Estimation (estimate_error_sds), from the Stage 2 blend's out-of-sample residuals r = actual - prediction:
- National: each election's mean residual rbar_y is ONE observation, so K elections give K observations. Its
  variance is the House Stage 2 estimator, with the Senate's own residuals (the scales aren't assumed equal):
      sigma_N^2 = (n0 * s0^2 + sum_y rbar_y^2) / (n0 + K),
  a fixed prior SD s0 (NATIONAL_PRIOR_SD, 3 points) counted as n0 pseudo-elections (NATIONAL_PRIOR_ELECTIONS, 2).
  The prior is fixed, not tuned on the backtest. rbar_y also averages race noise (~ sigma_race / sqrt(33)), which isn't
  subtracted.
- Race (pooled): sum (r - rbar_y)^2 / (N - K).
- S-M1, for g in {polled, unpolled}: s_g^2 = (n_g * v_g + k0 * s_pool^2) / (n_g + k0), with
  v_g = mean((r - rbar_y)^2 over g) * N / (N - K) and k0 = RACE_PRIOR_RACES (30 pseudo-races, about one cycle; fixed).

- S-M2, regional: with d_r = r - rbar_y, per election
      c_y = mean(d_r d_r' over pairs of races in different states of the same region)
            - mean(d_r d_r' over pairs of races in different regions),
  how much more two races of a region co-move than two races of different regions (the shared negative covariance
  that centering on rbar_y adds to every pair cancels in the difference). sigma_R^2 = max(0, sum_y c_y / (K + n0_R)):
  a zero prior worth n0_R = REGIONAL_PRIOR_ELECTIONS (3) pseudo-elections, as in the House. The race variance is then
  max(s_pool^2 - sigma_R^2, 0), so a race's total error variance stays S-M0's and only its correlation changes.
  With ~33 races in 10 regions (about 3 per region-year), the estimate is noisy; it's an approximation, fixed a priori.

Every SD is converted to a Student-t scale (t_scale) so each draw's SD equals the estimate; nu = NU = 5.
"""

from pathlib import Path

import numpy as np
import pandas as pd

NU = 5
NATIONAL_PRIOR_SD = 3.0       # points; fixed, not tuned on backtest outcomes
NATIONAL_PRIOR_ELECTIONS = 2  # the prior's weight, in pseudo-elections
RACE_PRIOR_RACES = 30         # S-M1's shrinkage toward the pooled race SD, in pseudo-races; fixed
REGIONAL_PRIOR_ELECTIONS = 3  # S-M2's zero prior on the regional variance, in pseudo-elections; fixed
MODELS = ("S-M0", "S-M1", "S-M2")
REGIONS_PATH = Path(__file__).resolve().parents[3] / "data/reference/regional_stage2.csv"


def load_regions(path=REGIONS_PATH) -> pd.Series:
    """state -> political region; raises unless each of the 50 states appears once."""
    table = pd.read_csv(path)
    if len(table) != 50 or table["state"].duplicated().any() or table["region"].isna().any():
        raise ValueError("Every state needs exactly one region")
    return table.set_index("state")["region"]


def regional_moment(deviation: pd.Series, year: pd.Series, state: pd.Series, region: pd.Series) -> pd.Series:
    """c_y per election: the same-region (different-state) minus different-region mean of d_r * d_r'."""
    out = {}
    for y in sorted(year.unique()):
        m = (year == y).to_numpy()
        d, st, rg = deviation[m].to_numpy(), state[m].to_numpy(), region[m].to_numpy()
        prod = np.outer(d, d)
        upper = np.triu(np.ones_like(prod, dtype=bool), k=1)
        same = upper & (rg[:, None] == rg[None, :]) & (st[:, None] != st[None, :])
        diff = upper & (rg[:, None] != rg[None, :])
        out[y] = prod[same].mean() - prod[diff].mean() if same.any() and diff.any() else np.nan
    return pd.Series(out)


def t_scale(sd: float, nu: float = NU) -> float:
    """The Student-t scale whose standard deviation is `sd`: sd * sqrt((nu - 2) / nu)."""
    if not nu > 2:
        raise ValueError(f"nu must exceed 2; got {nu}")
    return sd * np.sqrt((nu - 2) / nu)


def estimate_error_sds(oos: pd.DataFrame, model: str = "S-M0") -> tuple[dict[str, float], pd.DataFrame]:
    """(SD per component, a table of how each was estimated) from out-of-sample residuals. oos has one row per race
    with year, residual and (for S-M1) polling_available."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if oos["residual"].isna().any():
        raise ValueError("Residuals must be observed")
    residual, year = oos["residual"], oos["year"]
    yearly = residual.groupby(year).mean()
    k, n = len(yearly), len(residual)
    if n - k <= 0:
        raise ValueError("Need more races than elections")
    n0, s0 = NATIONAL_PRIOR_ELECTIONS, NATIONAL_PRIOR_SD
    var_national = (n0 * s0 ** 2 + (yearly ** 2).sum()) / (n0 + k)
    deviation = residual - year.map(yearly)
    var_pool = (deviation ** 2).sum() / (n - k)

    sds = {"national": float(np.sqrt(var_national)), "race": float(np.sqrt(var_pool))}
    rows = [
        {"component": "national", "observations": k, "raw_sd": float(np.sqrt((yearly ** 2).mean())),
         "sd": sds["national"], "regularization": f"prior sd {s0:g} x {n0:g} pseudo-elections"},
        {"component": "race (pooled)", "observations": n, "raw_sd": sds["race"], "sd": sds["race"],
         "regularization": "none" if model == "S-M0" else "the shrinkage target"},
    ]
    if model == "S-M1":
        polled = oos["polling_available"].astype(bool)
        for group, mask in {"polled": polled, "unpolled": ~polled}.items():
            n_g = int(mask.sum())
            v_g = (deviation[mask] ** 2).mean() * n / (n - k) if n_g else np.nan
            var_g = (n_g * v_g + RACE_PRIOR_RACES * var_pool) / (n_g + RACE_PRIOR_RACES) if n_g else var_pool
            sds[f"race_{group}"] = float(np.sqrt(var_g))
            rows.append({"component": f"race ({group})", "observations": n_g,
                         "raw_sd": float(np.sqrt(v_g)) if n_g else np.nan, "sd": sds[f"race_{group}"],
                         "regularization": f"shrunk to pooled with {RACE_PRIOR_RACES} pseudo-races"})
    if model == "S-M2":
        moments = regional_moment(deviation, year, oos["state"], oos["region"]).dropna()
        var_regional = max(moments.sum() / (len(moments) + REGIONAL_PRIOR_ELECTIONS), 0.0)
        sds["regional"] = float(np.sqrt(var_regional))
        sds["race"] = float(np.sqrt(max(var_pool - var_regional, 0.0)))
        rows[1]["sd"], rows[1]["regularization"] = sds["race"], "pooled minus the regional variance"
        rows.append({"component": "regional", "observations": len(moments),
                     "raw_sd": float(np.sqrt(max(moments.mean(), 0.0))) if len(moments) else np.nan,
                     "sd": sds["regional"],
                     "regularization": f"zero prior x {REGIONAL_PRIOR_ELECTIONS} pseudo-elections; truncated at 0"})
    return sds, pd.DataFrame(rows).set_index("component")


def race_sds(point: pd.DataFrame, sds: dict[str, float]) -> np.ndarray:
    """Each race's error SD: the pooled race SD (S-M0), or the polled / unpolled SD (S-M1, when sds has them)."""
    if "race_polled" not in sds:
        return np.full(len(point), sds["race"])
    return np.where(point["polling_available"].astype(bool), sds["race_polled"], sds["race_unpolled"])


def simulate_margins(point: pd.DataFrame, sds: dict[str, float], n_sims: int, rng: np.random.Generator,
                     nu: float = NU) -> np.ndarray:
    """(n_sims, n_races) simulated margins: point["center"] + one national t draw per simulation [+ one regional t
    draw per region per simulation, when sds has "regional"; point needs region] + an independent t draw per race.
    Draws come from `rng`: national, then race, then regional."""
    center = point["center"].to_numpy(dtype=float)
    if not np.isfinite(center).all():
        raise ValueError("Every race needs a finite center")
    national = t_scale(sds["national"], nu) * rng.standard_t(nu, size=(n_sims, 1))
    race = np.array([t_scale(s, nu) for s in race_sds(point, sds)]) * rng.standard_t(nu, size=(n_sims, len(center)))
    sims = center + national + race
    if "regional" in sds:
        regions, index = np.unique(point["region"].to_numpy(), return_inverse=True)
        regional = t_scale(sds["regional"], nu) * rng.standard_t(nu, size=(n_sims, len(regions)))
        sims = sims + regional[:, index]
    return sims
