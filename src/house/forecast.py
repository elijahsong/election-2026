"""The 2026 House production forecast: Stage 1 -> race handling -> Stage 2 (H0) -> the congress-track House rows.

This is section 5 of notebooks/house/stage2/02_monte_carlo.ipynb as two functions, so the full forecast
(notebooks/full-forecast/) and the scenario runs share one pipeline:

    inputs = forecast.load_inputs()                  # data, the error model, the same-party top-two predictions
    run = forecast.run(inputs)                       # the model's forecast
    run = forecast.run(inputs, shift=-4.0, adjustments=by_race)   # a generic-ballot shift, margin adjustments

02_monte_carlo.ipynb stays the calibration and backtest notebook. run(inputs) with no shift and no adjustments
reproduces its 2026 track file (data/processed/house/congress_track_house_2026.csv) exactly.

Production settings (02_monte_carlo.ipynb's; changing one here changes the production forecast):
    CONFIG, FALLBACK_CONFIG       Stage 1: the 85/15 ensemble, Model A where the ensemble lacks inputs
    NU, NATIONAL_PRIOR_*          the Student-t error model, fit on 2018-2024 out-of-sample residuals
    N_SIMS, SEED                  draws come from np.random.default_rng([SEED, 2026])
    TOP_TWO_*                     the same-party top-two resolver (house/simulation/top_two.py), set by hand

Run-time inputs (congress/adjustments.py), both none by default:
    shift         points added to 2026's gen_margin_raw. The error model and top-two predictions don't depend on it
    adjustments   {race_id: points}, added to a simulated race's Stage 1 margin before the Monte Carlo. A race decided
                  by rule has no margin, so an adjustment to one is refused. prediction_model keeps the unadjusted
                  margin, margin_adjustment the points added
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import house.regression as hr
from congress import adjustments as adj
from house import simulation, stage1
from house.regression.data import ROOT
from house.regression.schema import KEY

YEAR = 2026
CONFIG = stage1.HOUSE_ENSEMBLE_AB_85_15
FALLBACK_CONFIG = stage1.HOUSE_MODEL_A
NU = 5
NATIONAL_PRIOR_SD = 3.0
NATIONAL_PRIOR_ELECTIONS = 2
N_SIMS = 40_000
SEED = 42
TOP_TWO_WEIGHTS_WITH_POLL = {"primary": 0.6, "fundraising": 0.3, "poll": 0.1}
TOP_TWO_WEIGHTS_NO_POLL = {"primary": 0.6, "fundraising": 0.4}
TOP_TWO_POINTS_PER_LOG = 4.46  # Model B's fundraising_advantage coefficient (2006-2024 fit), points per log unit
TOP_TWO_BASE_SD = 8.0
TOP_TWO_SPLIT_SD = 0.15
HOUSE_SEATS = 435
MAJORITY = 218
FEC_SLOTS_PATH = ROOT / "data/processed/house_fec.csv"
MODEL_TRACK_PATH = ROOT / "data/processed/house/congress_track_house_2026.csv"  # 02_monte_carlo.ipynb's model rows


@dataclass
class Inputs:
    """Everything run() needs that no shift or adjustment changes."""
    predictors: pd.DataFrame
    targets: pd.DataFrame
    candidates: pd.DataFrame
    template: pd.DataFrame
    fec_slots: pd.DataFrame
    top_two: pd.DataFrame
    error_sd: dict             # national, state, district SDs
    error_details: pd.DataFrame
    error_scale: dict          # the Student-t scales the draws use

    @property
    def resolvers(self) -> list:
        return [simulation.TopTwoSameParty(self.top_two)]


def load_inputs() -> Inputs:
    """The data, the error model (from CONFIG's 2018-2024 rolling out-of-sample residuals) and the top-two
    predictions, with the production settings."""
    with warnings.catch_warnings(action="ignore"):  # loading warns about known optional-column gaps
        predictors = hr.load_predictors()
        targets = hr.load_targets()
    oos = simulation.stage1_inputs(CONFIG, predictors=predictors, targets=targets).oos
    error_sd, error_details = simulation.estimate_error_sds(oos, national_prior_sd=NATIONAL_PRIOR_SD,
                                                            national_prior_elections=NATIONAL_PRIOR_ELECTIONS)
    top_two = simulation.predict_top_two(simulation.load_top_two(), weights_with_poll=TOP_TWO_WEIGHTS_WITH_POLL,
                                         weights_no_poll=TOP_TWO_WEIGHTS_NO_POLL,
                                         points_per_log=TOP_TWO_POINTS_PER_LOG, base_sd=TOP_TWO_BASE_SD,
                                         split_sd=TOP_TWO_SPLIT_SD, nu=NU)
    return Inputs(
        predictors=predictors,
        targets=targets,
        candidates=simulation.load_candidates_2026(),
        template=pd.read_csv(simulation.TRACK_TEMPLATE_PATH),
        fec_slots=pd.read_csv(FEC_SLOTS_PATH),
        top_two=top_two,
        error_sd=error_sd,
        error_details=error_details,
        error_scale={level: simulation.t_scale(sd, NU) for level, sd in error_sd.items()},
    )


@dataclass
class Forecast:
    shift: float
    generic_ballot: float
    point: pd.DataFrame           # every race: prediction_model, margin_adjustment, prediction (what is simulated)
    draws: simulation.RaceDraws
    table: pd.DataFrame           # forecast_table: p_dem, p_gop, p_other, predicted party and candidate
    track: pd.DataFrame           # the congress-track House rows
    names: pd.DataFrame           # KEY, dem_candidate, gop_candidate (candidate_names)

    def races(self) -> pd.DataFrame:
        """One row per race in the full forecast's shared columns (congress/report.py)."""
        table = self.table.merge(self.point[[*KEY, "prediction_model", "margin_adjustment"]], on=KEY,
                                 validate="one_to_one").merge(self.names, on=KEY, how="left", validate="one_to_one")
        table["race_id"] = [simulation.race_id(s, d) for s, d in zip(table["state"], table["district"])]
        party_p = table[["p_dem", "p_gop", "p_other"]].max(axis=1)
        out = pd.DataFrame({
            "race_id": table["race_id"], "chamber": "House", "race_type": table["race_type"],
            "dem_candidate": table["dem_candidate"], "gop_candidate": table["gop_candidate"],
            "margin_model": table["prediction_model"], "margin_adjustment": table["margin_adjustment"],
            "margin": table["prediction"], "p_dem": table["p_dem"], "p_gop": table["p_gop"],
            "p_other": table["p_other"], "party_probability": party_p,
        })
        return out.merge(self.track.drop(columns="chamber"), on="race_id", validate="one_to_one")


def run(inputs: Inputs, *, shift: float = 0.0, adjustments: pd.Series | None = None,
        n_sims: int = N_SIMS, seed: int = SEED) -> Forecast:
    """The 2026 House forecast with the generic ballot moved by `shift` and `adjustments` ({race_id: points}) added
    to the simulated races' Stage 1 margins."""
    predictors = adj.shift_generic_ballot(inputs.predictors, shift, YEAR)
    forecast = simulation.stage1_forecast(CONFIG, predictors=predictors, targets=inputs.targets, year=YEAR)
    fallback = simulation.stage1_forecast(FALLBACK_CONFIG, predictors=predictors, targets=inputs.targets, year=YEAR)
    point = simulation.attach_predictions(simulation.races_2026(inputs.candidates), forecast.predictions,
                                          primary_source=CONFIG.name, fallback=fallback.predictions)
    point = adjust_margins(point, adjustments)
    draws = simulation.simulate_races(point, inputs.error_scale, NU, n_sims, np.random.default_rng([seed, YEAR]),
                                      resolvers=inputs.resolvers)
    table = simulation.forecast_table(draws, inputs.candidates, resolvers=inputs.resolvers)
    simulation.check_forecast(table, draws, inputs.candidates, house_size=HOUSE_SEATS)
    track = simulation.track_table(table, inputs.top_two, fec_slots=inputs.fec_slots, template=inputs.template)
    names = candidate_names(point, inputs.candidates, inputs.fec_slots)
    return Forecast(shift, adj.generic_ballot(predictors, YEAR), point, draws, table, track, names)


def candidate_names(point: pd.DataFrame, candidates: pd.DataFrame, fec_slots: pd.DataFrame) -> pd.DataFrame:
    """KEY, dem_candidate, gop_candidate: each side's general-election candidate (model_party D or R in the
    registry). Several candidates of one party are joined with " / " (a same-party top-two race lists both
    finalists), except in a simulated race (Louisiana's all-candidate round), where the party's FEC model slot names
    the candidate Stage 1 simulated. A GOP proxy who isn't a Republican on the ballot is marked (I): Kiley, CA-6."""
    rows = candidates[candidates["model_party"].isin(["D", "R"])]
    proxy = rows["model_party"].eq("R") & rows["party"].ne("R")
    rows = rows.assign(label=rows["candidate_name"].where(~proxy, rows["candidate_name"] + " (I)"))
    by_party = rows.groupby([*KEY, "model_party"])["label"]
    joined = by_party.agg(" / ".join).unstack("model_party")
    several = by_party.size().unstack("model_party").fillna(0) > 1
    simulated = point.set_index(KEY)["treatment"].eq("simulated").reindex(joined.index, fill_value=False)
    slots = fec_slots[fec_slots["year"] == YEAR].set_index(KEY).reindex(joined.index)
    out = pd.DataFrame(index=joined.index)
    for column, party, slot in [("dem_candidate", "D", "dem_name"), ("gop_candidate", "R", "gop_name")]:
        use_slot = several.get(party, False) & simulated & slots[slot].notna()
        out[column] = joined.get(party).where(~use_slot, slots[slot])
    return out.reset_index()


def adjust_margins(point: pd.DataFrame, adjustments: pd.Series | None) -> pd.DataFrame:
    """point with prediction_model (Stage 1's margin), margin_adjustment (0 where none) and prediction = their sum.
    Refuses an adjustment to a race id that isn't here or to a race decided by rule (it has no margin)."""
    adjustments = pd.Series(dtype=float) if adjustments is None else adjustments
    ids = pd.Series([simulation.race_id(s, d) for s, d in zip(point["state"], point["district"])], index=point.index)
    unknown = sorted(set(adjustments.index) - set(ids))
    if unknown:
        raise ValueError(f"House adjustments for races that aren't in the forecast: {unknown}")
    out = point.assign(prediction_model=point["prediction"],
                       margin_adjustment=ids.map(adjustments).fillna(0.0).astype(float))
    by_rule = out["margin_adjustment"].ne(0) & out["treatment"].ne("simulated")
    if by_rule.any():
        raise ValueError(f"House adjustments to races decided by rule (no margin): {ids[by_rule].tolist()}")
    out["prediction"] = out["prediction_model"] + out["margin_adjustment"]
    return out
