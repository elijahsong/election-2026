"""Conditional forecasts: the submitted forecast's simulations, conditioned on race results as they come in.

    uv run python src/congress/conditional.py                                   the baseline forecast
    uv run python src/congress/conditional.py --call KY-06=D --prob NC-11=D:0.93 --margin VA-02=D+3.5:2
    uv run python src/congress/conditional.py -i                                interactive (type inputs one by one)

Inputs (any number, by the submission's race_id; party D, R or I):
    --call RACE=PARTY              PARTY won RACE: keep only the simulations where it does
    --prob RACE=PARTY:P            P(PARTY wins RACE) = P (e.g. a call you half-believe, or a market price): the
                                   simulations are reweighted so the race hits P, and every other race follows
    --margin RACE=MARGIN:SD        an observed margin with its uncertainty (partial returns): MARGIN as D+3.5, R+2 or
                                   I+4 (the independent over the Republican in NE / ID / SD), SD in points. Each
                                   simulation is weighted by the normal likelihood of the observation given its margin
    --national MARGIN:SD           the national House popular vote, e.g. D+12:1.5 (House only; see below)

The national House popular vote. Each simulation implies one: its 435 district margins weighted by the district's
2024 total votes (targets.csv), races decided by rule counted at +-100 (a Democrat or Republican unopposed by the
other party). In 2024 this weighting gives the actual contested popular vote exactly; for 2026 it is an
approximation (2024 turnout on 2026 lines, two-party margins). An observed popular vote far from the baseline's would
leave reweighting with few simulations, so --national is applied as the national error instead: in every simulation
all simulated House margins move by the same amount, so that its implied popular vote equals a draw from
N(MARGIN, SD), while its state and district errors are kept. Winners and seats are recomputed from the moved margins,
and the other inputs are then applied by weighting, as above. One --national at a time.

How it works. The baseline is the production forecast exactly as submitted (house.forecast and senate.forecast with
the production settings and data/reference/margin_adjustments_2026.csv; checked against the submission file). Its
simulations already move together: every House race shares a national error, and races in one state a state error;
every Senate race shares the Senate's national error. Conditioning weights the simulations:
    w = product of the calls' indicators x product of the margins' likelihoods, then raked (iterative proportional
        fitting) until every --prob target holds
and every race probability and seat count is recomputed with w.

Limits (printed with the results):
- The chambers are separate: their national errors are drawn independently, so a House input moves no Senate race
  and vice versa.
- Effective simulations, 1 / sum(w^2) for normalized w, shrink with every surprise. Below about 1,000 the numbers are
  noisy; --sims draws more (costly: about 1 GB of memory per 100,000 House simulations).
- Inputs on races decided by rule (uncontested, same-party top-two) carry no information about other races, and the
  same-party top-two candidate picks never change.

Cache: the baseline's simulations are saved to data/interim/conditional/baseline_<key>.npz on first use and reused,
so the interactive tool starts in seconds. The key hashes the submission file, the adjustments file and --sims, so
rerunning the full forecast (which rewrites the submission) makes a new cache. One cache is kept per --sims value, so
switching between sizes doesn't rebuild. --rebuild forces one. With --sims other than the production 40,000 the
draws differ from the submission's: "before" numbers then come from the same draws, not the submission.
"""

import argparse
import hashlib
import re
import shlex
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from congress import adjustments, report
from house import forecast as house_forecast
from house import simulation as house_sim
from senate import forecast as senate_forecast
from senate import stage2

SUBMISSION_PATH = ROOT / "data/processed/full congress track template.csv"
CACHE_DIR = ROOT / "data/interim/conditional"
LOW_ESS = 1_000
CACHE_VERSION = "2"  # bump when Baseline's fields change
TARGETS_PATH = ROOT / "data/processed/house/targets.csv"
NATIONAL_ID = "HOUSE-PV"
NATIONAL_SEED = 20261103
HOUSE_CODES = {"D": house_sim.DEM, "R": house_sim.GOP, "I": house_sim.OTHER}


class ConditionError(ValueError):
    """An input that can't be applied (unknown race, impossible outcome, uninformative race)."""


# ---------------------------------------------------------------- the baseline (cached simulations)

@dataclass
class Baseline:
    picks: pd.DataFrame          # the submission: race_id, chamber, predicted_winner_party, candidate_name, win_probability
    house_ids: np.ndarray        # race_id of each House draw column
    house_treatment: np.ndarray  # fixed, third_party, same_party or simulated
    house_winners: np.ndarray    # (n_sims, 435) int8 party codes
    house_margins: np.ndarray    # (n_sims, 435) float32 D - R margins, NaN for races decided by rule
    house_dem_seats: np.ndarray  # (n_sims,) Democratic seats
    house_turnout: np.ndarray    # (435,) the district's 2024 total votes (the median where missing): popular-vote weights
    house_loser_code: np.ndarray  # (435,) int8 party code of a simulated race's winner when the Democrat loses
    house_fixed_margin: np.ndarray  # (435,) +100 / -100 for races decided by rule (a D or R seat), NaN when simulated
    senate_ids: np.ndarray       # race_id of each Senate column
    senate_seat_type: np.ndarray  # dem_nominee or independent_only
    senate_sims: np.ndarray      # (n_sims, 35) float32 margins: the D (or independent) nominee over the Republican
    senate_dem_caucus: np.ndarray  # (n_sims,) seats caucusing with Democrats (report.senate_dem_caucus)

    @property
    def n_sims(self) -> int:
        return len(self.house_dem_seats)


def cache_key(n_sims: int) -> str:
    digest = hashlib.sha256()
    for path in [SUBMISSION_PATH, adjustments.MARGIN_ADJUSTMENTS_PATH]:
        digest.update(path.read_bytes())
    digest.update(f"{n_sims}:{CACHE_VERSION}".encode())
    return digest.hexdigest()[:16]


def build_baseline(n_sims: int = house_forecast.N_SIMS) -> Baseline:
    """Run the production forecast (both chambers, the adjustments file) and check it reproduces the submission."""
    with warnings.catch_warnings(action="ignore"):
        house_inputs = house_forecast.load_inputs()
        senate_inputs = senate_forecast.load_inputs()
    margin = adjustments.load_margin_adjustments(template=house_inputs.template)
    house = house_forecast.run(house_inputs, adjustments=adjustments.by_race(margin, "House"), n_sims=n_sims)
    senate = senate_forecast.run(senate_inputs, weights=stage2.WEIGHTS, adjustments=adjustments.by_race(margin, "Senate"),
                                 n_sims=n_sims)
    picks = pd.read_csv(SUBMISSION_PATH)
    if n_sims == house_forecast.N_SIMS:  # the production draws: the runs must be the submission
        track = pd.concat([house.track, senate.track], ignore_index=True).set_index("race_id")
        sub = picks.set_index("race_id")
        same = (track.loc[sub.index, "predicted_winner_party"].eq(sub["predicted_winner_party"]).all()
                and np.allclose(track.loc[sub.index, "win_probability"], sub["win_probability"], atol=1e-6))
        if not same:
            raise RuntimeError("The production forecast no longer reproduces the submission file: rerun "
                               "notebooks/full-forecast/full_forecast_2026.ipynb first")
    races = house.draws.races
    house_ids = np.array([house_sim.race_id(s, d) for s, d in zip(races["state"], races["district"])])
    targets = pd.read_csv(TARGETS_PATH, usecols=["year", "state", "district", "total_votes"])
    t24 = targets[targets["year"] == 2024]
    votes = pd.Series(t24["total_votes"].to_numpy(),
                      index=[house_sim.race_id(s, d) for s, d in zip(t24["state"], t24["district"])])
    turnout = pd.Series(house_ids).map(votes)
    fixed = np.where(races["race_type"].str.startswith("dem_"), 100.0,
                     np.where(races["race_type"].str.startswith("gop_"), -100.0, np.nan))
    loser = races["loser_party"].map({"R": house_sim.GOP, "other": house_sim.OTHER}).fillna(0)
    return Baseline(
        picks=picks[["race_id", "chamber", "predicted_winner_party", "candidate_name", "win_probability"]],
        house_ids=house_ids,
        house_treatment=races["treatment"].to_numpy().astype(str),
        house_winners=house.draws.winners.astype(np.int8),
        house_margins=house.draws.margins.astype(np.float32),
        house_dem_seats=house.draws.seats(house_sim.DEM).astype(np.int16),
        house_turnout=turnout.fillna(turnout.median()).to_numpy(dtype=float),
        house_loser_code=loser.to_numpy(dtype=np.int8),
        house_fixed_margin=fixed.astype(np.float32),
        senate_ids=(senate.point["state"] + "-SEN").to_numpy().astype(str),
        senate_seat_type=senate.point["seat_type"].to_numpy().astype(str),
        senate_sims=senate.sims.astype(np.float32),
        senate_dem_caucus=np.asarray(report.senate_dem_caucus(senate.counts)).astype(np.int16),
    )


def load_baseline(n_sims: int = house_forecast.N_SIMS, rebuild: bool = False) -> tuple[Baseline, str]:
    """(baseline, where it came from): the cached simulations for the current submission, else a fresh run, cached."""
    path = CACHE_DIR / f"baseline_{n_sims}_{cache_key(n_sims)}.npz"
    picks = pd.read_csv(SUBMISSION_PATH)[["race_id", "chamber", "predicted_winner_party", "candidate_name",
                                          "win_probability"]]
    if path.exists() and not rebuild:
        arrays = np.load(path)
        return Baseline(picks=picks, **{k: arrays[k] for k in arrays.files}), f"cache {path.relative_to(ROOT)}"
    baseline = build_baseline(n_sims)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for old in CACHE_DIR.glob("baseline_*.npz"):  # one cache per simulation count: the current submission's
        stale = old.stem.startswith(f"baseline_{n_sims}_") or old.stem.count("_") == 1  # or an older naming
        if stale and old != path:
            old.unlink()
    np.savez_compressed(path, **{k: v for k, v in vars(baseline).items() if k != "picks"})
    return baseline, f"fresh run, cached to {path.relative_to(ROOT)}"


# ---------------------------------------------------------------- inputs

@dataclass(frozen=True)
class Condition:
    kind: str                 # "call", "prob", "margin" or "national"
    race_id: str
    party: str | None = None  # call and prob
    value: float | None = None  # prob: the probability; margin: the observed margin, positive for the non-Republican
    sd: float | None = None     # margin and national

    def __str__(self) -> str:
        if self.kind == "national":
            side = "R" if self.value < 0 else "D"
            return f"national House popular vote: {side}+{abs(self.value):g} ± {self.sd:g}"
        if self.kind == "call":
            return f"call {self.race_id} = {self.party}"
        if self.kind == "prob":
            return f"prob {self.race_id}: P({self.party}) = {self.value:g}"
        side = "R" if self.value < 0 else ("I" if self.party == "I" else "D")
        return f"margin {self.race_id}: {side}+{abs(self.value):g} ± {self.sd:g}"


def parse(kind: str, text: str) -> Condition:
    """A Condition from the CLI's forms: call "KY-06=D", prob "KY-06=D:0.95", margin "VA-02=D+3.5:2", national
    "D+12:1.5"."""
    text = text.strip().upper().replace(" ", "")
    if kind == "call":
        m = re.fullmatch(r"([A-Z]{2}-(?:\d{2}|AL|SEN))=([DRI])", text)
        if m:
            return Condition("call", m[1], party=m[2])
    elif kind == "prob":
        m = re.fullmatch(r"([A-Z]{2}-(?:\d{2}|AL|SEN))=([DRI]):(0?\.\d+|1(?:\.0*)?|0)", text)
        if m:
            return Condition("prob", m[1], party=m[2], value=float(m[3]))
    elif kind == "margin":
        m = re.fullmatch(r"([A-Z]{2}-(?:\d{2}|AL|SEN))=([DRI])?\+?(-?\d+(?:\.\d+)?):(\d+(?:\.\d+)?)", text)
        if m and float(m[4]) > 0:
            sign = -1.0 if m[2] == "R" else 1.0
            return Condition("margin", m[1], party=m[2] or "D", value=sign * float(m[3]), sd=float(m[4]))
    elif kind == "national":
        m = re.fullmatch(r"([DR])?\+?(-?\d+(?:\.\d+)?):(\d+(?:\.\d+)?)", text)
        if m and float(m[3]) > 0:
            sign = -1.0 if m[1] == "R" else 1.0
            return Condition("national", NATIONAL_ID, value=sign * float(m[2]), sd=float(m[3]))
    raise ConditionError(f"Can't read {kind} {text!r}: expected call KY-06=D, prob KY-06=D:0.95, "
                         "margin VA-02=D+3.5:2 or national D+12:1.5")


def _house_column(baseline: Baseline, race_id: str) -> int:
    hits = np.flatnonzero(baseline.house_ids == race_id)
    if not len(hits):
        raise ConditionError(f"No race {race_id}")
    return int(hits[0])


def _senate_column(baseline: Baseline, race_id: str) -> int:
    hits = np.flatnonzero(baseline.senate_ids == race_id)
    if not len(hits):
        raise ConditionError(f"No race {race_id}")
    return int(hits[0])


def event(baseline: Baseline, race_id: str, party: str) -> tuple[str, np.ndarray]:
    """(chamber, the simulations in which `party` wins `race_id`)."""
    if race_id.endswith("-SEN"):
        j = _senate_column(baseline, race_id)
        independent = baseline.senate_seat_type[j] == "independent_only"
        if (party == "D" and independent) or (party == "I" and not independent):
            raise ConditionError(f"{race_id}'s non-Republican is {'an independent (I)' if independent else 'a Democrat (D)'}")
        non_r = baseline.senate_sims[:, j] > 0
        return "Senate", non_r if party in ("D", "I") else ~non_r
    j = _house_column(baseline, race_id)
    if baseline.house_treatment[j] == "same_party":
        raise ConditionError(f"{race_id} is a same-party top-two race: its party is certain, and it carries no "
                             "information about other races")
    return "House", baseline.house_winners[:, j] == HOUSE_CODES[party]


def margin_draws(baseline: Baseline, race_id: str) -> tuple[str, np.ndarray]:
    """(chamber, each simulation's margin of the non-Republican over the Republican)."""
    if race_id.endswith("-SEN"):
        return "Senate", baseline.senate_sims[:, _senate_column(baseline, race_id)].astype(float)
    margins = baseline.house_margins[:, _house_column(baseline, race_id)].astype(float)
    if np.isnan(margins).all():
        raise ConditionError(f"{race_id} is decided by rule: it has no simulated margin")
    return "House", margins


def implied_popular_vote(baseline: Baseline) -> np.ndarray:
    """(n_sims,) each simulation's House popular vote, D - R points (module docstring)."""
    margins = np.where(np.isnan(baseline.house_margins), baseline.house_fixed_margin[None, :], baseline.house_margins)
    return margins.astype(float) @ baseline.house_turnout / baseline.house_turnout.sum()


def apply_national(baseline: Baseline, c: Condition, seed: int = NATIONAL_SEED) -> Baseline:
    """The baseline with every simulation's House national error set so its implied popular vote is a draw from
    N(c.value, c.sd): each simulated margin moves by the simulation's shift; winners and seats follow."""
    simulated = np.isnan(baseline.house_fixed_margin)
    target = c.value + c.sd * np.random.default_rng(seed).standard_normal(baseline.n_sims)
    share = baseline.house_turnout[simulated].sum() / baseline.house_turnout.sum()  # rule races don't move
    shift = (target - implied_popular_vote(baseline)) / share
    margins = baseline.house_margins.copy()
    margins[:, simulated] += shift[:, None].astype(np.float32)
    winners = baseline.house_winners.copy()
    winners[:, simulated] = np.where(margins[:, simulated] > 0, house_sim.DEM,
                                     baseline.house_loser_code[simulated][None, :])
    dem = (baseline.house_dem_seats - (baseline.house_winners[:, simulated] == house_sim.DEM).sum(axis=1)
           + (winners[:, simulated] == house_sim.DEM).sum(axis=1))
    return replace(baseline, house_margins=margins, house_winners=winners, house_dem_seats=dem.astype(np.int16))


def condition(baseline: Baseline, conditions: list[Condition]) -> tuple[Baseline, dict[str, np.ndarray]]:
    """(the baseline with any national input applied, the weights of the other inputs)."""
    national = [c for c in conditions if c.kind == "national"]
    if len(national) > 1:
        raise ConditionError("One national input at a time: drop the earlier one first (drop national)")
    moved = apply_national(baseline, national[0]) if national else baseline
    return moved, weights(moved, [c for c in conditions if c.kind != "national"])


def weights(baseline: Baseline, conditions: list[Condition], tol: float = 1e-6,
            max_iter: int = 1_000) -> dict[str, np.ndarray]:
    """{"House": w, "Senate": w}, each normalized to sum to 1 (module docstring)."""
    n = baseline.n_sims
    w = {"House": np.ones(n), "Senate": np.ones(n)}
    targets = {"House": [], "Senate": []}
    for c in conditions:
        if c.kind == "margin":
            chamber, m = margin_draws(baseline, c.race_id)
            w[chamber] = w[chamber] * np.exp(-0.5 * ((m - c.value) / c.sd) ** 2)
        else:
            chamber, won = event(baseline, c.race_id, c.party)
            if c.kind == "call":
                w[chamber] = w[chamber] * won
            else:
                targets[chamber].append((c, won))
    for chamber in ["House", "Senate"]:
        if w[chamber].sum() <= 0:
            raise ConditionError(f"No {chamber} simulation fits every call and margin together")
        for _ in range(max_iter):
            worst = 0.0
            for c, won in targets[chamber]:
                total, inside = w[chamber].sum(), w[chamber][won].sum()
                q = inside / total
                if (q == 0 and c.value > 0) or (q == 1 and c.value < 1):
                    raise ConditionError(f"{c} is impossible given the other {chamber} inputs")
                if 0 < q < 1:
                    w[chamber] = np.where(won, w[chamber] * c.value / q, w[chamber] * (1 - c.value) / (1 - q))
                worst = max(worst, abs(q - c.value))
            if worst < tol:
                break
        else:
            raise ConditionError(f"The {chamber} probability targets can't all hold together")
        w[chamber] = w[chamber] / w[chamber].sum()
    return w


# ---------------------------------------------------------------- results

def effective_sims(w: np.ndarray) -> float:
    return float(1.0 / np.sum(w ** 2))


def weighted_quantile(values: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(w[order])
    return float(values[order][np.searchsorted(cumulative, q * cumulative[-1])])


def chambers(baseline: Baseline, w: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per chamber: P(control), median seats and their 10-90% range, effective simulations."""
    rows = {}
    for chamber, seats, majority in [("House", baseline.house_dem_seats, report.HOUSE_MAJORITY),
                                     ("Senate", baseline.senate_dem_caucus, report.SENATE_CONTROL)]:
        ww = w[chamber]
        rows[chamber] = {
            "P(D control)": float(ww[seats >= majority].sum()),
            "median D seats": weighted_quantile(seats, ww, 0.5),
            "D seats 10%-90%": f"{weighted_quantile(seats, ww, 0.1):.0f}–{weighted_quantile(seats, ww, 0.9):.0f}",
            "effective sims": effective_sims(ww),
        }
    return pd.DataFrame(rows).T


def pick_probabilities(baseline: Baseline, w: dict[str, np.ndarray]) -> pd.Series:
    """Each submitted candidate's win probability under weights w, by race_id. Same-party top-two races keep the
    submission's (their candidate draws are independent of everything else)."""
    house = baseline.picks[baseline.picks["chamber"] == "House"].set_index("race_id")
    column = {r: j for j, r in enumerate(baseline.house_ids)}
    out = {}
    for race_id, pick in house.iterrows():
        j = column[race_id]
        if baseline.house_treatment[j] == "same_party":
            out[race_id] = pick["win_probability"]
        else:
            out[race_id] = float(w["House"][baseline.house_winners[:, j] == HOUSE_CODES[pick["predicted_winner_party"]]]
                                 .sum())
    senate = baseline.picks[baseline.picks["chamber"] == "Senate"].set_index("race_id")
    for j, race_id in enumerate(baseline.senate_ids):
        non_r = baseline.senate_sims[:, j] > 0
        wins = non_r if senate.loc[race_id, "predicted_winner_party"] in ("D", "I") else ~non_r
        out[race_id] = float(w["Senate"][wins].sum())
    return pd.Series(out)[baseline.picks["race_id"]]


def race_table(baseline: Baseline, conditioned: Baseline, w: dict[str, np.ndarray]) -> pd.DataFrame:
    """The submission's picks with each pick's win probability in the baseline's own draws (win_probability; with the
    production 40,000 the submission's exactly, checked at build), conditioned, and the change. Measuring both from
    the same draws keeps Monte Carlo noise out of the change when --sims differs from production."""
    out = baseline.picks.copy()
    n = baseline.n_sims
    out["win_probability"] = pick_probabilities(baseline, {"House": np.full(n, 1 / n),
                                                           "Senate": np.full(n, 1 / n)}).to_numpy()
    out["conditioned_probability"] = pick_probabilities(conditioned, w).to_numpy()
    out["change"] = out["conditioned_probability"] - out["win_probability"]
    out["flips"] = out["conditioned_probability"] < 0.5
    return out


def report_text(baseline: Baseline, conditions: list[Condition], top: int = 15) -> str:
    """The printed result: inputs, chambers before and after, the conditioned races and the largest moves."""
    uniform = weights(baseline, [])
    base = chambers(baseline, uniform)
    moved, w = condition(baseline, conditions)
    now = chambers(moved, w)
    lines = ["Inputs: " + (" | ".join(map(str, conditions)) if conditions else "none (the submitted forecast)"), ""]
    pv_before, pv_after = implied_popular_vote(baseline), implied_popular_vote(moved)

    def pv(values, ww):
        q = [weighted_quantile(values, ww, x) for x in (0.1, 0.5, 0.9)]
        return f"D{q[1]:+.1f} (80%: D{q[0]:+.1f} to D{q[2]:+.1f})"

    lines.append(f"House popular vote, implied: {pv(pv_before, uniform['House'])} -> {pv(pv_after, w['House'])}")
    for chamber in ["House", "Senate"]:
        b, a = base.loc[chamber], now.loc[chamber]
        lines.append(f"{chamber:6s}  P(D control) {b['P(D control)']:.3f} -> {a['P(D control)']:.3f}   "
                     f"D seats median {b['median D seats']:.0f} ({b['D seats 10%-90%']}) -> "
                     f"{a['median D seats']:.0f} ({a['D seats 10%-90%']})   "
                     f"effective sims {a['effective sims']:,.0f} of {baseline.n_sims:,}")
        if a["effective sims"] < LOW_ESS:
            lines.append(f"        ! fewer than {LOW_ESS:,} effective simulations: noisy. Rebuild with more "
                         "(--sims 200000) or drop an input")
    if any(c.race_id.endswith("-SEN") for c in conditions) != any(not c.race_id.endswith("-SEN") for c in conditions):
        lines.append("        (the chambers are simulated separately: inputs in one move no race in the other)")
    table = race_table(baseline, moved, w)
    given = {c.race_id for c in conditions if c.kind != "national"}
    columns = ["race_id", "predicted_winner_party", "candidate_name", "win_probability", "conditioned_probability",
               "change", "flips"]
    if given:
        lines += ["", "Conditioned races:", table[table["race_id"].isin(given)][columns].round(3).to_string(index=False)]
    for chamber in ["House", "Senate"]:
        moved = table[~table["race_id"].isin(given) & (table["chamber"] == chamber)]
        moved = moved.reindex(moved["change"].abs().sort_values(ascending=False).index).head(top)
        moved = moved[moved["change"].abs() > 0.0005]
        if len(moved):
            lines += ["", f"{chamber}: largest moves (the submitted pick's win probability; flips = now below 50%):",
                      moved[columns].round(3).to_string(index=False)]
    return "\n".join(lines)


# ---------------------------------------------------------------- the CLI

HELP = """Commands (party D, R or I; race ids as in the submission: KY-06, AK-AL, ME-SEN):
  call RACE PARTY            PARTY won RACE                       e.g. call KY-06 D
  prob RACE PARTY P          P(PARTY wins RACE) = P               e.g. prob NC-11 D 0.9
  margin RACE MARGIN SD      observed margin, SD in points        e.g. margin VA-02 D+3.5 2
  national MARGIN SD         the national House popular vote      e.g. national D+12 1.5
  list                       the inputs so far
  undo                       drop the last input
  drop RACE                  drop every input on RACE (drop national: the popular vote)
  reset                      drop every input
  show [N]                   the results again, with the N largest moves
  race RACE                  one race's pick, before and after
  save PATH                  write the conditioned race table as CSV
  help, quit"""


def interactive(baseline: Baseline, conditions: list[Condition], top: int) -> None:
    print(report_text(baseline, conditions, top))
    print("\n" + HELP)
    while True:
        try:
            line = input("\ncondition> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        words = shlex.split(line)
        command, args = words[0].lower(), words[1:]
        try:
            if command in ("quit", "exit", "q"):
                return
            if command == "help":
                print(HELP)
                continue
            if command == "list":
                print("\n".join(f"  {i + 1}. {c}" for i, c in enumerate(conditions)) or "  none")
                continue
            if command == "race" and len(args) == 1:
                table = race_table(baseline, *condition(baseline, conditions)).set_index("race_id")
                print(table.loc[[args[0].upper()]].round(3).to_string())
                continue
            if command == "save" and len(args) == 1:
                race_table(baseline, *condition(baseline, conditions)).round(6).to_csv(args[0], index=False)
                print(f"-> {args[0]}")
                continue
            if command == "show":
                top = int(args[0]) if args else top
            elif command == "undo":
                conditions = conditions[:-1]
            elif command == "reset":
                conditions = []
            elif command == "drop" and len(args) == 1:
                target = NATIONAL_ID if args[0].lower() == "national" else args[0].upper()
                conditions = [c for c in conditions if c.race_id != target]
            elif command == "call" and len(args) == 2:
                conditions = conditions + [parse("call", f"{args[0]}={args[1]}")]
            elif command == "prob" and len(args) == 3:
                conditions = conditions + [parse("prob", f"{args[0]}={args[1]}:{args[2]}")]
            elif command == "margin" and len(args) == 3:
                conditions = conditions + [parse("margin", f"{args[0]}={args[1]}:{args[2]}")]
            elif command == "national" and len(args) == 2:
                conditions = conditions + [parse("national", f"{args[0]}:{args[1]}")]
            else:
                print("Not a command. Type help.")
                continue
            condition(baseline, conditions)  # refuse an input that can't hold, keeping the earlier ones
            print(report_text(baseline, conditions, top))
        except ConditionError as error:
            if command in ("call", "prob", "margin", "national"):
                conditions = conditions[:-1]
            print(f"! {error}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Condition the submitted forecast on race results.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=HELP)
    parser.add_argument("--call", action="append", default=[], metavar="RACE=PARTY")
    parser.add_argument("--prob", action="append", default=[], metavar="RACE=PARTY:P")
    parser.add_argument("--margin", action="append", default=[], metavar="RACE=MARGIN:SD")
    parser.add_argument("--national", metavar="MARGIN:SD", help="the House popular vote, e.g. D+12:1.5")
    parser.add_argument("-i", "--interactive", action="store_true", help="type inputs one by one")
    parser.add_argument("--top", type=int, default=15, help="largest moves to list")
    parser.add_argument("--out", type=Path, help="also write the conditioned race table to this CSV")
    parser.add_argument("--sims", type=int, default=house_forecast.N_SIMS, help="simulations (production: 40,000)")
    parser.add_argument("--rebuild", action="store_true", help="rerun the forecast instead of using the cache")
    args = parser.parse_args(argv)
    try:
        conditions = ([parse("call", t) for t in args.call] + [parse("prob", t) for t in args.prob]
                      + [parse("margin", t) for t in args.margin]
                      + ([parse("national", args.national)] if args.national else []))
    except ConditionError as error:
        parser.error(str(error))

    print("Loading the baseline forecast ...", flush=True)
    baseline, source = load_baseline(args.sims, rebuild=args.rebuild)
    print(f"Baseline: {baseline.n_sims:,} simulations ({source})\n")
    if args.interactive:
        interactive(baseline, conditions, args.top)
        return
    try:
        print(report_text(baseline, conditions, args.top))
        if args.out:
            race_table(baseline, *condition(baseline, conditions)).round(6).to_csv(args.out, index=False)
            print(f"\n-> {args.out}")
    except ConditionError as error:
        raise SystemExit(f"! {error}")


if __name__ == "__main__":
    main()
