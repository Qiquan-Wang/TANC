"""visualisation_utils.py — shared helpers for TANC visualisations."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def make_figure(
    ax: Axes | None,
    figsize: tuple[float, float] | None,
    default_figsize: tuple[float, float] = (6, 5),
) -> tuple[Figure, Axes]:
    """Create or reuse a figure/axes pair.

    Parameters
    ----------
    ax : Axes or None
        If provided, reuse this axes.
    figsize : (w, h) or None
        If None and ax is None, use ``default_figsize``.

    Returns
    -------
    (Figure, Axes)
    """
    if ax is not None:
        return ax.get_figure(), ax
    if figsize is None:
        figsize = default_figsize
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def annotate_ph_stats(
    ax: Axes,
    statistics: dict,
    dim: int = 0,
    x: float = 0.98,
    y: float = 0.98,
) -> None:
    """Add a text box with key PH statistics to a figure axes.

    Parameters
    ----------
    ax : Axes
    statistics : dict
        Output of ``ph_tool.compute_statistics``.
    dim : int
        Homology dimension to display stats for.
    x, y : float
        Axes-coordinate position (top-right by default).
    """
    prefix = f"H{dim}"
    lines: list[str] = []
    for stat in ["total_persistence", "persistence_norm", "persistence_entropy"]:
        key = f"{prefix}_{stat}"
        if key in statistics:
            val = statistics[key]
            lines.append(f"{stat}: {val:.4f}")

    if lines:
        text = "\n".join(lines)
        ax.text(
            x, y, text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )


def format_paper_reference(paper_reference: str | None) -> str:
    """Format a paper reference string for plot titles/annotations."""
    if not paper_reference:
        return ""
    return f"Method: {paper_reference}"
