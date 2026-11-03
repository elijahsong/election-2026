"""Race handling around the Monte Carlo: classify each district by its general-election lineup, simulate only the
D-vs-R races from Stage 1 margins, and give every other race its party outcome by rule, within each draw.

    race_type (lineup)                     treatment     party outcome per draw                          margin
    dem_ / gop_uncontested                 fixed         D / R in every draw                             none
    dem_vs_3p / gop_vs_3p                  third_party   D / R, else OTHER with THIRD_PARTY_UPSET_PROB   none
    dem_vs_dem / gop_vs_gop                same_party    D / R in every draw (resolvers run per draw)     none
    dr, dr_multi, dr_writein, dr_jungle,   simulated     D if the simulated margin > 0, else the race's  simulated
    dr_rcv                                               loser_party (R, or OTHER for an independent
                                                         GOP proxy)

Every district gives exactly one party outcome per draw. Party codes: DEM 1, GOP -1, OTHER 0 (a third party or an
independent, kept distinct: never folded into D or R). An upset is OTHER, never the other major party, and no margin
is ever made up for a race that isn't simulated.

Lineups. History (historical_races) uses targets.csv's race_type, the ballot's candidates by party, except where
house_targets.TREATED_UNCONTESTED reclassified a race from its result (RESULTS_BASED_FLAGS: a major party finishing
third, MT-2 2022; a voided election, NC-9 2018). Those take targets_raw.csv's race_type, their pre-election lineup.
Write-in-only opponents (writein_treated_uncontested) stay uncontested: write-ins aren't on the ballot. A voided
election's winner is unresolved (blank winner_party), so it is left out of observed-winner scoring and observed seat
totals. 2026 (races_2026) uses the candidate registry (data/processed/house/candidates_2026.csv).

Predictions. A simulated race needs a Stage 1 margin: the configuration's, or Model A's as the fallback when the
configuration lacks an input (attach_predictions). The restored lineups have no Stage 1 prediction in a plain run,
because their targets were imputed; lineup_targets blanks those targets, so Stage 1 predicts them as unscored eval
rows. Training rows don't change, so neither do the coefficients.

CA-6 2026: Kiley, an independent on the ballot, is the GOP proxy for Stage 1 (fec_finance.INDEPENDENTS_AS_GOP). The
race is simulated, and a draw he wins is OTHER. RaceDraws.seats(..., alignment=...) can count such a seat with a party
for a caucus scenario, only when asked.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from house.regression.schema import KEY

from .hierarchical import simulate_margins
from .resolvers import resolve_candidate

DEM, GOP, OTHER = 1, -1, 0
PARTY_CODES = {"D": DEM, "R": GOP, "other": OTHER}
PARTY_NAMES = {DEM: "D", GOP: "R", OTHER: "other"}
THIRD_PARTY_UPSET_PROB = 0.005  # a major party against third parties only loses 0.5% of draws (0 of 205 in 2006-2024)

SIMULATED_TYPES = ("dr", "dr_multi", "dr_writein", "dr_jungle", "dr_rcv")
TREATMENTS = {**{t: "simulated" for t in SIMULATED_TYPES},
              "dem_uncontested": "fixed", "gop_uncontested": "fixed",
              "dem_vs_3p": "third_party", "gop_vs_3p": "third_party",
              "dem_vs_dem": "same_party", "gop_vs_gop": "same_party"}
FAVORED = {"dem_uncontested": DEM, "gop_uncontested": GOP, "dem_vs_3p": DEM, "gop_vs_3p": GOP,
           "dem_vs_dem": DEM, "gop_vs_gop": GOP}
# house_targets.TREATED_UNCONTESTED reasons that come from the result, so are undone for the pre-election lineup
RESULTS_BASED_FLAGS = ("major_party_third_treated_uncontested", "voided_election_treated_uncontested")
UNRESOLVED_WINNER_FLAGS = ("voided_election_treated_uncontested",)  # never certified: no observed winner
RACE_COLUMNS = [*KEY, "race_type", "treatment", "election_method", "loser_party", "winner_party", "winner_resolved",
                "actual", "lineup_restored"]


def _treatment(race_type: pd.Series) -> pd.Series:
    treatment = race_type.map(TREATMENTS)
    unknown = sorted(set(race_type[treatment.isna()]))
    if unknown:
        raise ValueError(f"No treatment for race types {unknown}")
    return treatment


def _flagged(flags: pd.Series, names) -> pd.Series:
    return flags.fillna("").str.split(";").map(lambda f: bool(set(f) & set(names)))


def historical_races(targets: pd.DataFrame, targets_raw: pd.DataFrame) -> pd.DataFrame:
    """One row per district-year of targets (RACE_COLUMNS): the pre-election lineup's race_type and treatment, the
    observed winner (winner_party, blank when unresolved), and actual, the observed two-party margin, only for
    simulated races with a target that wasn't imputed."""
    out = targets[[*KEY, "race_type", "winner_party", "adj_margin_2p_norm", "imputation_method",
                   "quality_flags"]].copy()
    restored = _flagged(out["quality_flags"], RESULTS_BASED_FLAGS)
    raw = out[KEY].merge(targets_raw[[*KEY, "race_type"]], on=KEY, how="left", validate="one_to_one")["race_type"]
    out.loc[restored, "race_type"] = raw[restored].to_numpy()
    out["lineup_restored"] = restored
    out["treatment"] = _treatment(out["race_type"])
    out["election_method"] = out["race_type"].map({"dr_rcv": "rcv", "dr_jungle": "jungle"}).fillna("standard")
    out["loser_party"] = np.where(out["treatment"] == "simulated", "R", None)
    unresolved = _flagged(out["quality_flags"], UNRESOLVED_WINNER_FLAGS)
    out["winner_party"] = out["winner_party"].where(~unresolved)
    out["winner_resolved"] = out["winner_party"].notna()
    observed = (out["treatment"] == "simulated") & out["imputation_method"].isna() & ~restored
    out["actual"] = out["adj_margin_2p_norm"].where(observed)
    return out[RACE_COLUMNS].reset_index(drop=True)


def lineup_targets(targets: pd.DataFrame) -> pd.DataFrame:
    """targets with the target and imputation blanked on the races historical_races restores to their pre-election
    D-vs-R lineup, so Stage 1 predicts them (as eval rows without a target: predicted, never scored or trained on)."""
    out = targets.copy()
    restored = _flagged(out["quality_flags"], RESULTS_BASED_FLAGS)
    out.loc[restored, ["adj_margin_2p_norm", "imputation_method"]] = np.nan
    return out


def races_2026(candidates: pd.DataFrame) -> pd.DataFrame:
    """One row per 2026 district (RACE_COLUMNS) from the candidate registry. A simulated race's loser_party is R, or
    other when its GOP proxy (model_party R) is not an R on the ballot."""
    first = candidates.drop_duplicates(KEY)[[*KEY, "race_type", "election_method"]].reset_index(drop=True)
    if candidates.groupby(KEY)["race_type"].nunique().gt(1).any():
        raise ValueError("A race has several race_type values in the registry")
    proxy = candidates[(candidates["model_party"] == "R") & (candidates["party"] != "R")][KEY].assign(proxy=True)
    out = first.merge(proxy, on=KEY, how="left", validate="one_to_one")
    out["treatment"] = _treatment(out["race_type"])
    out["loser_party"] = np.where(out["treatment"] != "simulated", None,
                                  np.where(out["proxy"].eq(True), "other", "R"))
    out["winner_party"], out["winner_resolved"], out["actual"], out["lineup_restored"] = None, False, np.nan, False
    return out[RACE_COLUMNS]


def attach_predictions(races: pd.DataFrame, primary: pd.DataFrame, *, primary_source: str,
                       fallback: pd.DataFrame | None = None,
                       fallback_source: str = "house_model_a_fallback") -> pd.DataFrame:
    """races plus state-level inputs for the simulation: prediction (the Stage 1 margin, simulated races only) and
    prediction_source (primary_source; fallback_source where only `fallback` predicts the race; otherwise the
    treatment, for races decided by rule). Raises if a simulated race has no prediction. primary and fallback have
    KEY and prediction (e.g. stacked Stage1Result.predictions)."""
    out = races.merge(primary[[*KEY, "prediction"]], on=KEY, how="left", validate="one_to_one")
    simulated = out["treatment"] == "simulated"
    out["prediction_source"] = np.where(simulated & out["prediction"].notna(), primary_source, out["treatment"])
    if fallback is not None:
        backup = out[KEY].merge(fallback[[*KEY, "prediction"]], on=KEY, how="left", validate="one_to_one")["prediction"]
        use = simulated & out["prediction"].isna() & backup.notna()
        out.loc[use, "prediction"] = backup[use]
        out.loc[use, "prediction_source"] = fallback_source
    missing = out[simulated & out["prediction"].isna()]
    if not missing.empty:
        raise ValueError(f"{len(missing)} simulated races without a Stage 1 prediction, e.g. "
                         f"{list(missing[KEY].head().itertuples(index=False, name=None))}")
    out.loc[~simulated, "prediction"] = np.nan  # a race decided by rule never carries a margin
    return out


@dataclass
class RaceDraws:
    """Party outcomes for every race in every draw."""
    races: pd.DataFrame     # RACE_COLUMNS plus prediction, prediction_source; row order = columns below
    winners: np.ndarray     # (n_sims, n_races) int8 party codes: the official, ballot-party winner
    margins: np.ndarray     # (n_sims, n_races) simulated D - R margins; NaN for races decided by rule

    def win_probability(self, party: int) -> np.ndarray:
        """Per race, the share of draws `party` (DEM, GOP or OTHER) wins."""
        return (self.winners == party).mean(axis=0)

    def seats(self, party: int, columns=None, alignment: dict | None = None) -> np.ndarray:
        """(n_sims,) seats `party` wins, over the races in `columns` (a boolean mask; default every race).
        alignment, a caucus scenario, counts an OTHER win in a listed race ({(year, state, district): party code})
        for that party; it is never applied unless given."""
        winners = self.winners
        if alignment:
            winners = winners.copy()
            keys = list(self.races[KEY].itertuples(index=False, name=None))
            for key, code in alignment.items():
                j = keys.index(key)
                winners[:, j] = np.where(winners[:, j] == OTHER, code, winners[:, j])
        if columns is not None:
            winners = winners[:, np.asarray(columns, dtype=bool)]
        return (winners == party).sum(axis=1)


def simulate_races(races: pd.DataFrame, scales: dict[str, float], nu: float, n_sims: int, rng: np.random.Generator,
                   resolvers=()) -> RaceDraws:
    """Every race's party outcome in each of n_sims draws (races from attach_predictions).

    Order of draws from `rng`: hierarchical.simulate_margins on the simulated races (unchanged; the same draws it gives
    those rows alone), then one uniform per third-party race and draw for the upsets. Then the fixed and same-party
    outcomes, then each resolver on its race types, in the order given."""
    n = len(races)
    treatment = races["treatment"].to_numpy()
    winners = np.zeros((n_sims, n), dtype=np.int8)
    margins = np.full((n_sims, n), np.nan)

    simulated = np.flatnonzero(treatment == "simulated")
    if len(simulated):
        sims, _ = simulate_margins(races.iloc[simulated], scales, nu, n_sims, rng)
        margins[:, simulated] = sims
        loser = races["loser_party"].iloc[simulated].map(PARTY_CODES).to_numpy(dtype=np.int8)
        winners[:, simulated] = np.where(sims > 0, DEM, loser)

    favored = races["race_type"].map(FAVORED)
    third = np.flatnonzero(treatment == "third_party")
    if len(third):
        upset = rng.random((n_sims, len(third))) < THIRD_PARTY_UPSET_PROB
        winners[:, third] = np.where(upset, OTHER, favored.iloc[third].to_numpy(dtype=np.int8))
    decided = np.flatnonzero(np.isin(treatment, ["fixed", "same_party"]))
    winners[:, decided] = favored.iloc[decided].to_numpy(dtype=np.int8)

    for resolver in resolvers:
        columns = np.flatnonzero(races["race_type"].isin(resolver.race_types).to_numpy())
        if len(columns):
            winners[:, columns] = resolver.resolve(races.iloc[columns], winners[:, columns].copy(), rng)
    return RaceDraws(races=races.reset_index(drop=True), winners=winners, margins=margins)


def forecast_table(draws: RaceDraws, candidates: pd.DataFrame, resolvers=()) -> pd.DataFrame:
    """One row per race: race_type, election_method, treatment, prediction_source, prediction (Stage 1's median
    margin, blank for races decided by rule), p_dem, p_gop, p_other, predicted_party (the most likely ballot party),
    and predicted_candidate with candidate_status (resolvers.resolve_candidate) from the registry `candidates`."""
    out = draws.races[[*KEY, "race_type", "election_method", "treatment", "prediction_source", "prediction"]].copy()
    probabilities = {code: draws.win_probability(code) for code in (DEM, GOP, OTHER)}
    out["p_dem"], out["p_gop"], out["p_other"] = probabilities[DEM], probabilities[GOP], probabilities[OTHER]
    out["predicted_party"] = pd.DataFrame({"D": out["p_dem"], "R": out["p_gop"], "other": out["p_other"]}).idxmax(axis=1)
    by_race = dict(iter(candidates.groupby(KEY)))
    names, statuses = [], []
    for race in out.itertuples(index=False):
        key = tuple(getattr(race, k) for k in KEY)
        name, status = resolve_candidate(pd.Series(race._asdict()), race.predicted_party,
                                         by_race.get(key, candidates.iloc[:0]), resolvers)
        names.append(name)
        statuses.append(status)
    out["predicted_candidate"], out["candidate_status"] = names, statuses
    return out
