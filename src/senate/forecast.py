"""The 2026 Senate production forecast: Stage 1 -> the Stage 2 blend -> the Monte Carlo (S-M0) -> track rows.

This is notebooks/senate/stage2/03_monte_carlo.ipynb's 2026 forecast as two functions, so the full forecast
(notebooks/full-forecast/) and the scenario runs share one pipeline:

    inputs = forecast.load_inputs()
    run = forecast.run(inputs)                                  # the model's forecast, production weights
    run = forecast.run(inputs, weights=MY_WEIGHTS, shift=4.0, adjustments=by_race)

03_monte_carlo.ipynb stays the calibration and backtest notebook. run(inputs) with the production weights, no shift
and no adjustments reproduces its primary track file (data/processed/senate/congress_track_senate_2026.csv) exactly.

Production settings (03_monte_carlo.ipynb's):
    CONFIG          Stage 1: senate_s1 (incumbency fixed at +2.5)
    MODEL           the error model, S-M0: one national and one race Student-t error
    N_SIMS, SEED    40,000 draws from np.random.default_rng(SEED)
    stage2.WEIGHTS  the blend weights (fundamentals / polls / experts), with polls and without

Run-time inputs, all production by default:
    weights       blend weights in stage2.WEIGHTS' shape (stage2.check_weights). The error SDs are re-estimated from
                  the 2016-2024 out-of-sample residuals of the blend with these same weights, so the spread matches
                  the centers it is drawn around
    shift         points added to 2026's gen_margin_raw (congress/adjustments.py). It moves only the fundamentals;
                  polls and ratings are held fixed, and the three independent-vs-R races (centered on ratings) don't
                  move
    adjustments   {race_id: points}, added to the race's center before the Monte Carlo (in an independent-vs-R race,
                  independent-positive). center_model keeps the unadjusted center, margin_adjustment the points added
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import senate.regression as sr
from congress import adjustments as adj
from senate import stage1, stage2
from senate.simulation import calibration, chamber, errors, track

YEAR = 2026
CONFIG = stage1.SENATE_S1
MODEL = "S-M0"
N_SIMS = 40_000
SEED = 0
MODEL_TRACK_PATH = track.TRACK_OUTPUT_PATH  # 03_monte_carlo.ipynb's primary (S-M0) model rows


@dataclass
class Inputs:
    data: pd.DataFrame
    polls: pd.DataFrame
    ratings: pd.DataFrame
    nominees: pd.DataFrame
    ballot: pd.DataFrame
    template: pd.DataFrame


def load_inputs() -> Inputs:
    polls, ratings = stage2.load_inputs()
    return Inputs(
        data=sr.load_data(),
        polls=polls,
        ratings=ratings,
        nominees=pd.read_csv(track.NOMINEES_PATH),
        ballot=pd.read_csv(track.BALLOT_PATH),
        template=pd.read_csv(track.TRACK_TEMPLATE_PATH),
    )


@dataclass
class Forecast:
    shift: float
    generic_ballot: float
    weights: dict
    error_sd: dict
    error_details: pd.DataFrame
    point: pd.DataFrame          # every race: the blend, center_model, margin_adjustment, center (what is simulated),
                                 # and the candidates (track.resolve_candidates)
    sims: np.ndarray             # (n_sims, 35) margins
    counts: pd.DataFrame         # seats per simulation (chamber.count_seats)
    track: pd.DataFrame          # the congress-track Senate rows

    def races(self) -> pd.DataFrame:
        """One row per race in the full forecast's shared columns (congress/report.py). The non-Republican side is
        D in a Democratic nominee's race and other (an independent) in an independent-only race."""
        p_non_r = (self.sims > 0).mean(axis=0)
        independent = (self.point["seat_type"] == "independent_only").to_numpy()
        out = pd.DataFrame({
            "race_id": self.point["state"] + "-SEN", "chamber": "Senate", "race_type": self.point["seat_type"],
            "dem_candidate": np.where(independent, self.point["ind_candidate"] + " (I)", self.point["dem_candidate"]),
            "gop_candidate": self.point["gop_candidate"],
            "margin_model": self.point["center_model"], "margin_adjustment": self.point["margin_adjustment"],
            "margin": self.point["center"], "p_dem": np.where(independent, 0.0, p_non_r), "p_gop": 1 - p_non_r,
            "p_other": np.where(independent, p_non_r, 0.0), "party_probability": np.maximum(p_non_r, 1 - p_non_r),
        })
        return out.merge(self.track.drop(columns="chamber"), on="race_id", validate="one_to_one")


def run(inputs: Inputs, *, weights: dict = stage2.WEIGHTS, shift: float = 0.0, adjustments: pd.Series | None = None,
        n_sims: int = N_SIMS, seed: int = SEED) -> Forecast:
    """The 2026 Senate forecast with blend `weights`, the generic ballot moved by `shift`, and `adjustments`
    ({race_id: points}) added to the races' centers."""
    oos = calibration.blend_out_of_sample(CONFIG, data=inputs.data, polls=inputs.polls, ratings=inputs.ratings,
                                          weights=weights)
    error_sd, error_details = errors.estimate_error_sds(oos, MODEL)
    data = adj.shift_generic_ballot(inputs.data, shift, YEAR)
    point = chamber.forecast_points(CONFIG, data=data, polls=inputs.polls, ratings=inputs.ratings, weights=weights)
    point = adjust_centers(point, adjustments)
    sims, counts = chamber.simulate_2026(point, error_sd, n_sims=n_sims, seed=seed)
    named = track.resolve_candidates(point, inputs.nominees, inputs.ballot)
    return Forecast(shift, adj.generic_ballot(data, YEAR), weights, error_sd, error_details, named, sims, counts,
                    track.track_table(named, sims, inputs.template))


def adjust_centers(point: pd.DataFrame, adjustments: pd.Series | None) -> pd.DataFrame:
    """point with center_model (the blend or rating center), margin_adjustment (0 where none) and center = their
    sum. Refuses an adjustment to a race id that isn't here."""
    adjustments = pd.Series(dtype=float) if adjustments is None else adjustments
    ids = point["state"] + "-SEN"
    unknown = sorted(set(adjustments.index) - set(ids))
    if unknown:
        raise ValueError(f"Senate adjustments for races that aren't in the forecast: {unknown}")
    out = point.assign(center_model=point["center"], margin_adjustment=ids.map(adjustments).fillna(0.0).astype(float))
    out["center"] = out["center_model"] + out["margin_adjustment"]
    return out
