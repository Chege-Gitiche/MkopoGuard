"""House style for MkopoGuard charts: one look across every notebook and the README.

Colours follow a validated colour-blind-safe palette: blue marks Kenya (or the first
series), orange the second series, grey everything else. Text is never coloured.
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

BLUE = "#2a78d6"  # Kenya / series 1
ORANGE = "#eb6834"  # series 2
GREY = "#c9c8c2"  # other countries
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e6e5e0"


def apply_style() -> None:
    """Set matplotlib defaults once per notebook."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_MUTED,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "text.color": TEXT,
            "font.size": 10.5,
            "legend.frameon": False,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _finish(ax, title: str, subtitle: str, percent_axis: str) -> None:
    ax.set_title(title, pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=TEXT_MUTED, fontsize=9.5)
    if percent_axis in ("x", "y"):
        axis = ax.xaxis if percent_axis == "x" else ax.yaxis
        axis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_axisbelow(True)


def ranked_bars(
    shares: pd.Series,
    title: str,
    subtitle: str,
    highlight: str | None = "KEN",
    labels: dict | None = None,
    percent: bool = True,
    baseline: float = 0.0,
):
    """Horizontal bars, largest at the top, with one highlighted entry.

    percent=False shows plain decimals (e.g. AUC). baseline sets where bars start, for
    measures whose "nothing" point isn't zero (AUC's is 0.5).
    """
    shares = shares.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(shares) + 1.4))
    if highlight is None:
        colours = [BLUE] * len(shares)
    else:
        colours = [BLUE if code == highlight else GREY for code in shares.index]
    names = [labels.get(code, code) if labels else code for code in shares.index]
    values = shares.to_numpy()
    ax.barh(names, values - baseline, left=baseline, color=colours, height=0.7)
    ax.grid(axis="y", visible=False)
    span = shares.max() - baseline
    for y, value in enumerate(values):
        text = f"{value:.0%}" if percent else f"{value:.3f}"
        ax.text(value + span * 0.02, y, text, va="center", fontsize=9, color=TEXT_MUTED)
    ax.set_xlim(baseline, min(1.0, shares.max() + span * 0.15))
    _finish(ax, title, subtitle, "x" if percent else "none")
    return fig


def paired_bars(
    table: pd.DataFrame,
    title: str,
    subtitle: str,
    xlabel: str = "",
):
    """Vertical bars for 1-2 series (table columns) across categories (table index)."""
    series = list(table.columns)
    colours = [BLUE, ORANGE][: len(series)]
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(table))
    for i, (name, colour) in enumerate(zip(series, colours, strict=True)):
        positions = [p + (i - (len(series) - 1) / 2) * width for p in x]
        values = table[name].to_numpy()
        ax.bar(positions, values, width=width - 0.04, color=colour, label=name)
        for pos, value in zip(positions, values, strict=True):
            ax.text(pos, value + 0.01, f"{value:.0%}", ha="center", fontsize=8.5, color=TEXT_MUTED)
    ax.set_xticks(list(x), [textwrap.fill(str(i), 14) for i in table.index])
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, min(1.0, table.to_numpy().max() + 0.12))
    if len(series) > 1:
        ax.legend(loc="upper left", ncols=len(series), bbox_to_anchor=(0, -0.16))
    _finish(ax, title, subtitle, "y")
    return fig


def save(fig, path: Path) -> Path:
    """Save a chart as PNG (creating the folder) and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def histogram(
    series: dict[str, pd.Series],
    title: str,
    subtitle: str,
    xlabel: str,
    bins,
    log_x: bool = False,
):
    """Distribution of one or two series, as the share of applicants in each bin.

    One series is drawn as filled bars; two are drawn as outlines so both stay visible.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colours = [BLUE, ORANGE]
    for (name, values), colour in zip(series.items(), colours, strict=False):
        values = values.dropna()
        weights = [1 / len(values)] * len(values)
        if len(series) == 1:
            ax.hist(values, bins=bins, weights=weights, color=colour, edgecolor=SURFACE)
        else:
            ax.hist(
                values,
                bins=bins,
                weights=weights,
                histtype="step",
                linewidth=2,
                color=colour,
                label=name,
            )
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", visible=False)
    if len(series) > 1:
        ax.legend(loc="upper right")
    _finish(ax, title, subtitle, "y")
    return fig
