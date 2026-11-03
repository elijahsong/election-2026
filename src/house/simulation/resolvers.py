"""Election-rule resolvers: hooks that run after each draw's party outcomes, for races whose rules need them.

A Resolver names the race types it handles. simulate_races calls resolve() on those races' winners, draw by draw (the
whole (n_sims, n_races) block at once), after the overrides and the margin draws, so a resolver may change a draw's
party outcome. forecast_table calls candidate() when the winning party has several candidates on the ballot.

The defaults change nothing:

    race type             default party outcome                 resolver hook
    dem_vs_dem / gop_vs_gop  the party's seat in every draw     JungleSameParty
    dr_rcv                the simulated D-vs-R result           RankedChoice
    other D-vs-R types    the simulated result                  none

JungleSameParty and RankedChoice keep the default behavior. They are the places for later candidate-level models
(which of two Democrats wins a top-two general; RCV transfers); no such model is implemented here.
"""

import numpy as np
import pandas as pd


class Resolver:
    """Base resolver: party outcomes unchanged, and no candidate picked among several of one party."""
    race_types: tuple[str, ...] = ()

    def resolve(self, races: pd.DataFrame, winners: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """(n_sims, len(races)) party codes for these races, given their winners so far (same shape)."""
        return winners

    def candidate(self, race: pd.Series, party: str, candidates: pd.DataFrame) -> str | None:
        """The winning candidate's name among `candidates` (the race's candidates of `party`), or None."""
        return None


class JungleSameParty(Resolver):
    """Same-party general elections (top-two D vs D, R vs R). Hook for choosing between the party's candidates."""
    race_types = ("dem_vs_dem", "gop_vs_gop")


class RankedChoice(Resolver):
    """Ranked-choice generals (ME, AK). Hook for transfer-based party outcomes and candidate picks."""
    race_types = ("dr_rcv",)


def resolve_candidate(race: pd.Series, party: str, candidates: pd.DataFrame, resolvers=()) -> tuple[str | None, str]:
    """(name, status) of the winner of `race` given its winning ballot `party` (D, R, or other), from the race's
    registry rows `candidates`. One candidate of that ballot party: resolved. Several: the first resolver for the race
    type that picks one, else unresolved (several_<party>_candidates); never a guess. None: no_registered_candidate.
    Ballot parties are matched exactly, so an independent is never resolved as a D or an R."""
    eligible = candidates[candidates["party"] == party]
    if len(eligible) == 1:
        return eligible["candidate_name"].iloc[0], "resolved"
    if eligible.empty:
        return None, "no_registered_candidate"
    for resolver in resolvers:
        if race["race_type"] in resolver.race_types:
            name = resolver.candidate(race, party, eligible)
            if name is not None:
                if name not in set(eligible["candidate_name"]):
                    raise ValueError(f"{type(resolver).__name__} picked {name!r}, not a {party} candidate of the race")
                return name, f"resolved_by_{type(resolver).__name__}"
    return None, f"several_{party}_candidates"
