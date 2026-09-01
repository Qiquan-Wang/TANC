"""pipeline_diagram.py — render a TDAPipeline as a labelled flowchart.

Cosmetic but very explainer-friendly: drops into a paper-preset example
and gives the reader a one-glance picture of what the pipeline does.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _fmt_kwargs(kwargs: dict, max_items: int = 6) -> list[str]:
    if not kwargs:
        return ["(no kwargs)"]
    items = list(kwargs.items())
    lines = [f"{k} = {v!r}" for k, v in items[:max_items]]
    if len(items) > max_items:
        lines.append(f"... (+{len(items) - max_items} more)")
    return lines


def _box(ax, x, y, w, h, title: str, body_lines: list[str],
         facecolor: str, edgecolor: str = "black") -> None:
    """Draw one labelled box."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.06, title,
            ha="center", va="top", fontsize=11, weight="bold")
    body = "\n".join(body_lines)
    ax.text(x + 0.03, y + h - 0.18, body,
            ha="left", va="top", fontsize=8, family="monospace")


def _arrow(ax, x1, y1, x2, y2) -> None:
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.2, color="black",
    )
    ax.add_patch(arr)


def plot_pipeline_diagram(
    pipeline: Any,
    ax=None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Render a :class:`TDAPipeline` as a flowchart of its configured stages.

    Parameters
    ----------
    pipeline : TDAPipeline
        Any object that exposes ``builder``, ``builder_kwargs``, ``tool``,
        ``tool_kwargs``, ``visualisation``, ``paper_reference``.
    ax : matplotlib.Axes or None
    figsize : (w, h) or None.  Default ``(10, 4)``.
    title : str or None.  Default uses ``pipeline.paper_reference``.

    Returns
    -------
    matplotlib Figure
    """
    if figsize is None:
        figsize = (10, 4)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # --- Stage 1: builder ----------------------------------------------------
    builder_name = pipeline.builder
    if builder_name is None:
        builder_label = "(none — direct dimension)"
    elif callable(builder_name):
        builder_label = f"<callable {getattr(builder_name, '__name__', '?')}>"
    else:
        builder_label = str(builder_name)
    _box(ax, 0.2, 0.7, 3.2, 2.6,
         title=f"Builder\n{builder_label}",
         body_lines=_fmt_kwargs(pipeline.builder_kwargs or {}),
         facecolor="#dbe9f6")

    # --- Stage 2: tool -------------------------------------------------------
    _box(ax, 4.0, 0.7, 3.2, 2.6,
         title=f"Tool\n{pipeline.tool}",
         body_lines=_fmt_kwargs(pipeline.tool_kwargs or {}),
         facecolor="#e9f6db")

    # --- Stage 3: visualisation ---------------------------------------------
    vis = pipeline.visualisation or "(user picks)"
    _box(ax, 7.8, 0.9, 2.0, 2.2,
         title="Default plot",
         body_lines=[f"kind = {vis!r}"],
         facecolor="#f6e9db")

    _arrow(ax, 3.4, 2.0, 4.0, 2.0)
    _arrow(ax, 7.2, 2.0, 7.8, 2.0)

    if title is None:
        title = pipeline.paper_reference or "TDAPipeline"
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig