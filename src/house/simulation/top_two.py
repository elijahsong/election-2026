"""Which candidate wins a same-party top-two race (dem_vs_dem, gop_vs_gop): a weighted blend, with no training data.

The party outcome is already decided (the party's seat in every draw); this picks the candidate and says how sure
that pick is. Input: data/processed/house/toptwosameparty.csv (data-processing/house/top_two_same_party.py), one row
per finalist. Every quantity is the two-way November margin of the primary leader over the runner-up, in points:

    primary       2 * primary_share_top_two - 100          the June result, eliminated candidates' voters split in
                                                           proportion to the finalists' votes
    fundraising   points_per_log * individual_contribution_advantage
                                                           Model B's own prediction for a same-party race: partisan
                                                           lean, the intercept and the national terms cancel when both
                                                           finalists share a party and a district, leaving only its
                                                           fundraising_advantage coefficient (4.46 points per log unit,
                                                           2006-2024 fit). Individual contributions, not receipts, so
                                                           self-funding and loans (CA-4 Jones, CA-7 Matsui) don't count
    poll          2 * poll_share_top_two - 100             undecideds dropped; only where both finalists have a share

    mean = sum of weight * component                       WEIGHTS_WITH_POLL where the race has a poll, else
                                                           WEIGHTS_NO_POLL (both set by the user, not estimated)

Because the three components are all predictions of the same quantity on the same scale, the weights mean what they
say: a blend of forecasts, as Stage 1 blends Models A and B.

Uncertainty, also assumed rather than fitted:

    sd = sqrt(base_sd^2 + (200 * uncommitted * split_sd)^2)

    base_sd      the shift from June to November that any race has (turnout, campaigns): a floor
    uncommitted  the fraction of June voters who backed neither finalist (other-party voters, eliminated candidates).
                 They choose between the finalists in November. If the leader's share of them is the proportional
                 split +/- split_sd, the margin moves by 200 * uncommitted * split_sd points (1 SD). CA-7, where 40% of
                 June voters backed neither, is far less certain than CA-12, where almost no one did.

    p_leader = P(margin > 0) for a Student-t with nu degrees of freedom and that sd (hierarchical.t_scale), the same
    error form as the D-vs-R simulation.
"""

import numpy as np
import pandas as pd
from scipy import stats

from house.regression.data import ROOT
from house.regression.schema import KEY

from .hierarchical import t_scale
from .resolvers import JungleSameParty

TOP_TWO_PATH = ROOT / "data/processed/house/toptwosameparty.csv"
WEIGHTS_WITH_POLL = {"primary": 0.6, "fundraising": 0.3, "poll": 0.1}
WEIGHTS_NO_POLL = {"primary": 0.6, "fundraising": 0.4}
FUNDRAISING_POINTS_PER_LOG = 4.46  # stage1.MODEL_B's fundraising_advantage coefficient, fit on 2006-2024
BASE_SD = 8.0    # points; assumed
SPLIT_SD = 0.15  # SD of the leader's share of the uncommitted June voters; assumed
NU = 5           # as the Stage 2 notebook's D-vs-R draws


def load_top_two(path=TOP_TWO_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def predict_top_two(finalists: pd.DataFrame, *, weights_with_poll: dict, weights_no_poll: dict,
                    points_per_log: float, base_sd: float, split_sd: float, nu: float) -> pd.DataFrame:
    """One row per race: leader, runner_up, each component margin, the weights used, mean, sd, p_leader, p_runner_up
    and an 80% interval for the leader's margin (all in points)."""
    for weights in (weights_with_poll, weights_no_poll):
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError(f"weights must sum to 1; got {weights}")
    rows = []
    for key, race in finalists.sort_values("primary_rank").groupby(KEY, sort=True):
        leader, runner_up = race.iloc[0], race.iloc[1]
        margins = {
            "primary": 2 * leader["primary_share_top_two"] - 100,
            "fundraising": points_per_log * leader["individual_contribution_advantage"],
            "poll": 2 * leader["poll_share_top_two"] - 100,  # NaN without a poll for both
        }
        weights = weights_no_poll if np.isnan(margins["poll"]) else weights_with_poll
        mean = sum(w * margins[k] for k, w in weights.items())
        uncommitted = 1 - (leader["primary_share"] + runner_up["primary_share"]) / 100
        sd = np.hypot(base_sd, 200 * uncommitted * split_sd)
        scale = t_scale(sd, nu)
        p_leader = stats.t.cdf(mean / scale, nu)
        half_width = stats.t.ppf(0.9, nu) * scale
        rows.append({**dict(zip(KEY, key)), "leader": leader["candidate_name"], "runner_up": runner_up["candidate_name"],
                     **{f"margin_{k}": v for k, v in margins.items()},
                     "weights": ", ".join(f"{k} {w:g}" for k, w in weights.items()),
                     "uncommitted_share": 100 * uncommitted, "mean": mean, "sd": sd,
                     "interval_80_low": mean - half_width, "interval_80_high": mean + half_width,
                     "p_leader": p_leader, "p_runner_up": 1 - p_leader})
    return pd.DataFrame(rows)


class TopTwoSameParty(JungleSameParty):
    """JungleSameParty with a candidate pick: the finalist with p >= 0.5 in predict_top_two's table. The party
    outcome per draw is unchanged (the party's seat either way)."""

    def __init__(self, predictions: pd.DataFrame):
        self.predictions = predictions.set_index(KEY)

    def candidate(self, race: pd.Series, party: str, candidates: pd.DataFrame) -> str | None:
        key = tuple(race[k] for k in KEY)
        if key not in self.predictions.index:
            return None
        row = self.predictions.loc[key]
        return row["leader"] if row["p_leader"] >= 0.5 else row["runner_up"]
