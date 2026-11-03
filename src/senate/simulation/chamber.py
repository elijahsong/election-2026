"""2026 Senate: simulate the 35 races and count the chamber, keeping party and caucus apart.

Seat categories (never merged):
    dem_party          Democrats: the 32 Democratic holdovers plus Democratic nominees who win
    dem_aligned        dem_party plus the independents who caucus with Democrats: King (ME) and Sanders (VT), both
                       holdovers, plus any winner flagged independent_counted_as_dem (none in 2026)
    new_independents   independents elected in 2026 whose caucus is unknown: NE (Osborn), and the independent
                       opponents in ID and SD, where there is no Democratic nominee
    gop                the 31 Republican holdovers plus Republican winners

Holdovers (seats not up in 2026; 119th Congress: 53 R, 45 D, 2 independents caucusing with D):
    D-party 32 = 45 - the 13 D-held seats up (DEM_HELD_UP_2026); D-aligned 34 = 32 + King + Sanders;
    R 31 = 53 - the 22 R-held seats up. 32 + 2 + 31 + 35 = 100 is checked.

Race centers:
- Races with a Democratic nominee: the Stage 2 blend (D-positive D-vs-R margin).
- Independent-vs-Republican races (quality_flags independent_vs_gop or no_dem_nominee): the D-vs-R blend doesn't
  apply, so the center is the documented heuristic INDEPENDENT_RACE_CENTER: the expert-rating margin alone, since
  Cook, Sabato and Inside Elections rate the actual matchup (positive = the independent ahead). A race without
  ratings has no matched information and stops the simulation. matchup_note flags races where senate_fec.csv names
  no independent opponent (ID, SD: the ratings cover the Republican against the whole field). These races get the
  national error too, read as an anti-Republican swing, and the unpolled race scale. Their scale was estimated on D-vs-R
  blend residuals, not expert-only centers, so it is likely too narrow there.
A non-Republican win in an independent race is a new independent seat, never a Democratic one.

Control (the Vice President is a Republican, so a 50-50 Senate is Republican):
    "new independents with R or neither": D control iff dem_aligned >= 51
    "new independents with D":            D control iff dem_aligned + new_independents >= 51
"""

import numpy as np
import pandas as pd

import senate.regression as sr
from senate import stage1, stage2

from .errors import load_regions, simulate_margins

YEAR = 2026
DEM_HELD_UP_2026 = frozenset({"CO", "DE", "GA", "MA", "NJ", "NM", "OR", "RI", "VA", "IL", "MI", "MN", "NH"})
DEM_PARTY_HOLDOVER = 45 - len(DEM_HELD_UP_2026)  # 32
CAUCUSING_INDEPENDENT_HOLDOVER = 2               # King (ME, class 1), Sanders (VT, class 1)
DEM_ALIGNED_HOLDOVER = DEM_PARTY_HOLDOVER + CAUCUSING_INDEPENDENT_HOLDOVER  # 34
GOP_HOLDOVER = 31
INDEPENDENT_RACE_FLAGS = ("independent_vs_gop", "no_dem_nominee")
INDEPENDENT_RACE_CENTER = "expert ratings only (independent-vs-R heuristic)"
CONTROL_SCENARIOS = ("new independents with R or neither", "new independents with D")


def forecast_points(config=stage1.SENATE_S1, data=None, polls=None, ratings=None) -> pd.DataFrame:
    """One row per 2026 race: the Stage 2 blend, seat_type (dem_nominee, caucusing_independent or independent_only),
    center and center_source, and matchup_note."""
    data = sr.load_data() if data is None else data
    if polls is None or ratings is None:
        polls, ratings = stage2.load_inputs()
    point = stage2.blend(stage1.forecast(config, YEAR, data=data).predictions, polls, ratings)
    flags = pd.read_csv(stage2.RATINGS_PATH, usecols=[*sr.KEY, "quality_flags"])
    point = point.merge(flags, on=sr.KEY, how="left", validate="one_to_one")
    point = point.merge(data[[*sr.KEY, "inc_signed"]], on=sr.KEY, how="left", validate="one_to_one")
    has = lambda flag: point["quality_flags"].fillna("").str.contains(flag)  # noqa: E731
    independent = np.logical_or.reduce([has(f) for f in INDEPENDENT_RACE_FLAGS])
    point["seat_type"] = np.select([independent, has("independent_counted_as_dem")],
                                   ["independent_only", "caucusing_independent"], "dem_nominee")
    if point.loc[independent, "margin_experts"].isna().any():
        raise ValueError("An independent-vs-R race has no expert rating: no matched matchup information")
    point["center"] = np.where(independent, point["margin_experts"], point["prediction"])
    point["center_source"] = np.where(independent, INDEPENDENT_RACE_CENTER, "Stage 2 blend")
    point["matchup_note"] = np.where(has("no_dem_nominee"),
                                     "no named independent opponent in senate_fec.csv; ratings cover the field", None)
    point.loc[independent, "polling_available"] = False  # no matched independent-vs-R polls
    point["region"] = point["state"].map(load_regions())
    _check_holdovers(point)
    return point


def _check_holdovers(point: pd.DataFrame) -> None:
    if len(point) != 35 or DEM_ALIGNED_HOLDOVER + GOP_HOLDOVER + len(point) != 100:
        raise ValueError("Holdovers and races don't make 100 seats")
    if not set(point.loc[point["inc_signed"] == 1, "state"]) <= DEM_HELD_UP_2026:
        raise ValueError("A Democratic incumbent's seat is missing from DEM_HELD_UP_2026")
    if DEM_HELD_UP_2026 & set(point.loc[point["inc_signed"] == -1, "state"]):
        raise ValueError("DEM_HELD_UP_2026 lists a Republican incumbent's seat")


def count_seats(sims: np.ndarray, point: pd.DataFrame) -> pd.DataFrame:
    """One row per simulation: dem_party, dem_aligned, new_independents and gop seats (they sum to 100)."""
    win = sims > 0
    seat_type = point["seat_type"].to_numpy()
    dem_wins = win[:, seat_type == "dem_nominee"].sum(axis=1)
    caucusing = win[:, seat_type == "caucusing_independent"].sum(axis=1)
    new_ind = win[:, seat_type == "independent_only"].sum(axis=1)
    counts = pd.DataFrame({
        "dem_party": DEM_PARTY_HOLDOVER + dem_wins,
        "dem_aligned": DEM_ALIGNED_HOLDOVER + dem_wins + caucusing,
        "new_independents": new_ind,
        "gop": GOP_HOLDOVER + (~win).sum(axis=1),
    })
    if not (counts["dem_aligned"] + counts["new_independents"] + counts["gop"] == 100).all():
        raise ValueError("Seat counts don't sum to 100")
    return counts


def control(counts: pd.DataFrame) -> pd.Series:
    """P(Democratic control) under each CONTROL_SCENARIOS entry, and P(Republicans hold 50+ seats outright)."""
    return pd.Series({
        f"P(D control): {CONTROL_SCENARIOS[0]}": (counts["dem_aligned"] >= 51).mean(),
        f"P(D control): {CONTROL_SCENARIOS[1]}": (counts["dem_aligned"] + counts["new_independents"] >= 51).mean(),
        "P(R seats >= 50)": (counts["gop"] >= 50).mean(),
    })


def simulate_2026(point: pd.DataFrame, sds: dict[str, float], n_sims: int = 20_000,
                  seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """(margins (n_sims, 35), seat counts per simulation)."""
    sims = simulate_margins(point, sds, n_sims, np.random.default_rng(seed))
    return sims, count_seats(sims, point)


def plot_seat_distribution(counts: pd.DataFrame, ax, title: str = ""):
    """D-aligned seats, and D-aligned plus new independents, with the 51-seat control line."""
    bins = np.arange(counts["dem_aligned"].min() - 0.5, counts["dem_aligned"].max() + counts["new_independents"].max()
                     + 1.5)
    ax.hist(counts["dem_aligned"], bins=bins, color="#2a78d6", alpha=0.85, label="D-aligned (D + King, Sanders)")
    ax.hist(counts["dem_aligned"] + counts["new_independents"], bins=bins, histtype="step", color="#888", lw=1.5,
            label="D-aligned + new independents")
    ax.axvline(50.5, color="#c44", lw=1)
    ax.set(xlabel="seats", ylabel="simulations", title=title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ax
