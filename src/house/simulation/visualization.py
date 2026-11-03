"""Charts of Monte Carlo output.

    plot_seat_distribution             one election's histogram of simulated Democratic seats
    plot_backtest_seat_distributions   that histogram for every backtest config (rows) and test year (columns)
    plot_seat_errors                   every backtest election's seat median and intervals, minus the actual count
    plot_reliability                   district win probabilities against how often Democrats won, per test year

The seat charts take their median and central 80% and 95% intervals from calibration.seat_summary, the same function
behind Calibration.table, so charts and table always agree; plot_reliability draws Calibration.reliability. Only
actual draws are plotted: no smoothing and no fitted distribution. Every function returns the figure and axes and
never saves.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter

from .calibration import (
    RELIABILITY_EDGES,
    Calibration,
    reliability,
    seat_quantile,
    seat_summary,
)

INTERVAL_LEVELS = (0.8, 0.95)  # the central intervals shaded
X_VIEW_TAIL = 0.001  # share of draws per tail that may fall outside the x view

# Light-surface colors: interval bands are one blue ramp (darker = more central), the actual result a contrasting
# orange, text in ink grays rather than series colors.
COLORS = {
    "inside_80": "#2a78d6",
    "inside_95": "#9ec5f4",
    "outside_95": "#d3d2cd",
    "median": "#0b0b0b",
    "actual": "#eb6834",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e6e5e1",
    "surface": "#fcfcfb",
}
# The two backtest configurations, in the order given: categorical slots 1 and 3, the first filled circles, the
# second hollow squares, so coinciding points stay visible
CONFIG_COLORS = ("#2a78d6", "#1baf7a")
CONFIG_MARKERS = ("o", "s")
CONFIG_FILLED = (True, False)


def _style(ax, grid_axis="y"):
    """The shared look: surface background, recessive grid, no box."""
    ax.figure.set_facecolor(COLORS["surface"])
    ax.set_facecolor(COLORS["surface"])
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["muted"], length=0)


def _panel_label(panel_size, house_size, matched: bool) -> str | None:
    if panel_size is None:
        return None
    if panel_size == house_size:
        return f"All {house_size} districts"
    kind = "Matched panel" if matched else "Forecast panel"
    return f"{kind}: {panel_size} of {house_size} districts"


def plot_seat_distribution(seat_draws, *, year, actual_seats=None, panel_size=None, house_size=435, title=None,
                           ax=None, compact=False):
    """(fig, ax): the histogram of simulated Democratic seats for one election.

    seat_draws are integer seat counts, one per simulation (e.g. YearCalibration.panel_dem_seats, or
    (sims > 0).sum(axis=1) for the forecast). actual_seats is the observed count over the SAME districts, or None
    before the election. panel_size is how many districts the draws count over; when it is below house_size, the
    counts are panel seats, not House seats, and the chart says so. compact=True (for grids, which show one legend)
    drops the legend and the label beside the actual line; the summary still gives the actual count and percentile.
    Nothing is saved; the caller decides."""
    draws = np.asarray(seat_draws)
    if not np.issubdtype(draws.dtype, np.integer):
        raise ValueError(f"seat_draws must be integer seat counts from the simulations; got dtype {draws.dtype}")
    summary = seat_summary(draws, actual_seats, INTERVAL_LEVELS)
    (lo80, hi80), (lo95, hi95) = summary["intervals"][0.8], summary["intervals"][0.95]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    else:
        fig = ax.figure
    _style(ax)

    # One bar per integer seat count, height = share of simulations
    seats, counts = np.unique(draws, return_counts=True)
    share = counts / len(draws)
    colors = np.where((seats >= lo80) & (seats <= hi80), COLORS["inside_80"],
                      np.where((seats >= lo95) & (seats <= hi95), COLORS["inside_95"], COLORS["outside_95"]))
    ax.bar(seats, share, width=0.85, color=colors, linewidth=0, gid="seat_bars")

    # The median draws over the actual line, so both show when they coincide
    ax.axvline(summary["median"], color=COLORS["median"], linewidth=1.5, zorder=4, gid="median")
    peak = share.max()
    if actual_seats is not None:
        ax.axvline(actual_seats, color=COLORS["actual"], linewidth=2.2, zorder=3, gid="actual")
    if actual_seats is not None and not compact:
        right = actual_seats >= summary["median"]
        ax.annotate(f"Actual {actual_seats}\npercentile {summary['percentile']:.0f}",
                    xy=(actual_seats, peak * 1.02), xytext=(6 if right else -6, 0), textcoords="offset points",
                    ha="left" if right else "right", va="bottom", fontsize=9.5, color=COLORS["text"],
                    fontweight="bold")

    # Compact summary, top left
    lines = [f"Median  {summary['median']}",
             f"80%  {lo80}–{hi80}",
             f"95%  {lo95}–{hi95}"]
    if actual_seats is not None:
        lines.append(f"Actual  {actual_seats}  (percentile {summary['percentile']:.0f})")
    panel = _panel_label(panel_size, house_size, matched=actual_seats is not None)
    if panel:
        lines.append(panel)
    lines.append(f"{len(draws):,} simulations")
    ax.text(0.015, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color=COLORS["muted"], linespacing=1.5, gid="summary")

    if not compact:
        ax.legend(handles=_seat_legend(actual_seats is not None), loc="upper right", frameon=False, fontsize=9,
                  labelcolor=COLORS["text"])

    # Room above the tallest bar for the summary, legend, and annotation. The x view spans the central 99.8% of
    # draws (the Student-t tails would otherwise squeeze the bars) and the actual count; every draw is still plotted.
    ax.set_ylim(0, peak * (1.75 if compact else 1.45))  # small grid panels need more room for the summary
    x_min, x_max = seat_quantile(draws, [X_VIEW_TAIL, 1 - X_VIEW_TAIL])
    if actual_seats is not None:
        x_min, x_max = min(x_min, actual_seats), max(x_max, actual_seats)
    ax.set_xlim(x_min - 2.5, x_max + 2.5)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    seats_label = "Democratic seats won"
    if panel_size is not None and panel_size < house_size:
        seats_label += f" (of {panel_size} panel districts)"
    ax.set_xlabel(seats_label, color=COLORS["text"])
    ax.set_ylabel("Share of simulations", color=COLORS["text"])
    ax.set_title(title or f"{year}: simulated Democratic House seats", loc="left", fontsize=12,
                 color=COLORS["text"], fontweight="bold")
    return fig, ax


def _seat_legend(with_actual: bool) -> list:
    handles = [Patch(color=COLORS["inside_80"], label="Central 80%"),
               Patch(color=COLORS["inside_95"], label="Central 95%"),
               Patch(color=COLORS["outside_95"], label="Outside 95%"),
               Line2D([], [], color=COLORS["median"], linewidth=1.5, label="Median")]
    if with_actual:
        handles.append(Line2D([], [], color=COLORS["actual"], linewidth=2.2, label="Actual"))
    return handles


def plot_backtest_seat_distributions(calibration: dict[str, Calibration], years=None):
    """(fig, axes): plot_seat_distribution for every configuration (rows, in calibration's order) and test year
    (columns), from the stored backtest draws. Panels in a column share their x range, so the configurations compare
    directly; each column's panel size is in its panels' summaries and x labels."""
    names = list(calibration)
    years = list(years or next(iter(calibration.values())).years)
    fig, axes = plt.subplots(len(names), len(years), figsize=(max(5.2 * len(years), 9), 3.9 * len(names)),
                             layout="constrained", squeeze=False)
    for i, name in enumerate(names):
        for j, year in enumerate(years):
            result = calibration[name].years[year]
            plot_seat_distribution(result.panel_dem_seats, year=year, actual_seats=result.panel_actual_dem_seats,
                                   panel_size=result.panel_size, house_size=result.house_size,
                                   title=f"{year} · {name}", ax=axes[i, j], compact=True)
            axes[i, j].title.set_fontsize(10.5)
            if j:
                axes[i, j].set_ylabel("")
            if i < len(names) - 1:
                axes[i, j].set_xlabel("")
    for j in range(len(years)):  # one x range per column
        lows, highs = zip(*(ax.get_xlim() for ax in axes[:, j]))
        for ax in axes[:, j]:
            ax.set_xlim(min(lows), max(highs))
    fig.legend(handles=_seat_legend(True), loc="outside lower center", ncols=5, frameon=False, fontsize=9,
               labelcolor=COLORS["text"])
    fig.suptitle("Backtest: simulated Democratic seats vs the actual count (matched panels, not all 435 seats)",
                 x=0.01, ha="left", fontsize=12.5, color=COLORS["text"], fontweight="bold")
    return fig, axes


def plot_seat_errors(calibration: dict[str, Calibration], years=None, ax=None):
    """(fig, ax): for every test year and configuration, the simulated seat median and central 80% / 95% intervals
    minus the actual count, all from seat_summary. 0 is the actual result; a bar entirely right of 0 means the
    simulations expected more Democratic seats than Democrats won. Each row is labeled with the actual count's
    mid-rank percentile and its matched panel."""
    names = list(calibration)
    years = list(years or next(iter(calibration.values())).years)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) * len(years) + 1.6), layout="constrained")
    else:
        fig = ax.figure
    _style(ax, grid_axis="x")

    labels, rows, y = [], [], 0
    for year in years:
        for name in names:
            result = calibration[name].years[year]
            seats = result.seat_summary()
            actual = seats["actual"]
            (lo80, hi80), (lo95, hi95) = seats["intervals"][0.8], seats["intervals"][0.95]
            ax.plot([lo95 - actual, hi95 - actual], [y, y], color=COLORS["inside_95"], linewidth=4,
                    solid_capstyle="round", gid=f"95:{year}:{name}")
            ax.plot([lo80 - actual, hi80 - actual], [y, y], color=COLORS["inside_80"], linewidth=8,
                    solid_capstyle="round", gid=f"80:{year}:{name}")
            ax.plot([seats["median"] - actual], [y], marker="|", markersize=14, markeredgewidth=2.5,
                    color=COLORS["median"], zorder=4, gid=f"median:{year}:{name}")  # above the actual line
            ax.annotate(f"percentile {seats['percentile']:.0f}", xy=(1, y), xycoords=("axes fraction", "data"),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=COLORS["muted"])
            labels.append(f"{year} · {name}  ({result.panel_size} of {result.house_size})")
            rows.append(y)
            y -= 1
        y -= 0.6  # a gap between years
    ax.axvline(0, color=COLORS["actual"], linewidth=2, gid="actual")
    ax.set_yticks(rows, labels, fontsize=9, color=COLORS["text"])
    ax.set_xlabel("Simulated − actual Democratic seats (matched panel)", color=COLORS["text"])
    ax.set_title("Seat forecast errors by election", loc="left", fontsize=12, color=COLORS["text"],
                 fontweight="bold")
    fig.legend(handles=[Patch(color=COLORS["inside_80"], label="Central 80%"),
                       Patch(color=COLORS["inside_95"], label="Central 95%"),
                       Line2D([], [], color=COLORS["median"], marker="|", linestyle="", markersize=12,
                              markeredgewidth=2, label="Median"),
                       Line2D([], [], color=COLORS["actual"], linewidth=2, label="Actual (0)")],
              loc="outside lower center", ncols=4, frameon=False, fontsize=9, labelcolor=COLORS["text"])
    return fig, ax


def plot_reliability(calibration: dict[str, Calibration], years=None, edges=RELIABILITY_EDGES):
    """(fig, axes): one panel per test year of Calibration.reliability: each bin's mean simulated Democratic win
    probability (x) against the share of its districts Democrats won (y), one series per configuration, marker area
    proportional to the bin's district count (labeled, from the first configuration). Up to two configurations.
    Points on the diagonal are calibrated. The middle bins hold few districts, so their observed shares are noisy."""
    names = list(calibration)
    years = list(years or next(iter(calibration.values())).years)
    fig, axes = plt.subplots(1, len(years), figsize=(4.4 * len(years), 4.6), layout="constrained", sharey=True,
                             squeeze=False)
    for j, year in enumerate(years):
        ax = axes[0, j]
        _style(ax, grid_axis="both")
        ax.plot([0, 1], [0, 1], color=COLORS["muted"], linewidth=1, linestyle=(0, (4, 3)), gid="diagonal")
        for k, name in enumerate(names):
            table = reliability(calibration[name].years[year].districts, edges)
            filled = CONFIG_FILLED[k]
            ax.scatter(table["forecast"], table["observed"], s=20 + 0.6 * table["n"], marker=CONFIG_MARKERS[k],
                       facecolor=CONFIG_COLORS[k] if filled else "none",
                       edgecolor=COLORS["surface"] if filled else CONFIG_COLORS[k],
                       linewidth=1.2 if filled else 1.8, label=name, zorder=3 + k, gid=f"reliability:{name}")
            if k == 0:  # bin sizes, once per panel (both configurations bin nearly the same districts)
                for _, row in table.iterrows():
                    ax.annotate(f"n={row['n']:.0f}", xy=(row["forecast"], row["observed"]), xytext=(7, -10),
                                textcoords="offset points", fontsize=8, color=COLORS["muted"])
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(f"{year}", loc="left", fontsize=11, color=COLORS["text"], fontweight="bold")
        ax.set_xlabel("Simulated D win probability (bin mean)", color=COLORS["text"])
    axes[0, 0].set_ylabel("Share of districts D won", color=COLORS["text"])
    fig.legend(handles=axes[0, 0].collections[:len(names)], labels=names, loc="outside lower center",
               ncols=len(names), frameon=False, fontsize=9, labelcolor=COLORS["text"])
    fig.suptitle("District win-probability reliability", x=0.01, ha="left", fontsize=12.5, color=COLORS["text"],
                 fontweight="bold")
    return fig, axes
