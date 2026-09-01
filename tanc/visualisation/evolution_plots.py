"""evolution_plots.py — small-multiples views of how PH evolves over training."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tanc.model_extractor._snapshot import TrainingView, ModelSnapshot
from tanc.graph_builder._bundle import GraphBundle
from tanc.visualisation.representations import (
    plot_persistence_diagram,
    plot_barcode,
)
from tanc.visualisation.trajectory_plots import _default_weight_bundle


def _select_checkpoints(view: TrainingView, n_panels: int) -> list[int]:
    n = len(view.snapshots)
    if n_panels >= n:
        return list(range(n))
    return list(np.linspace(0, n - 1, n_panels, dtype=int))


def plot_diagram_evolution(
    view: TrainingView,
    n_panels: int = 6,
    kind: str = "diagram",
    dims: list[int] | None = None,
    layer: str | None = None,
    builder: Callable[[ModelSnapshot], GraphBundle] | None = None,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Small-multiples of persistence diagrams (or barcodes) across training.

    Parameters
    ----------
    view : TrainingView
    n_panels : int
        Number of evenly-spaced epoch checkpoints to show.
    kind : str
        ``"diagram"`` or ``"barcode"``.
    dims : list[int] or None
        PH dimensions to display.  Defaults to all available.
    layer : str or None
        If given (and no custom builder), only this layer's weights feed
        the default weight graph.
    builder : callable or None
        Custom ``snapshot → GraphBundle``.  ``None`` → default weight graph.
    ncols : int
    """
    from tanc.topo_tools.ph_tool import run_ph

    if kind not in {"diagram", "barcode"}:
        raise ValueError(f"kind must be 'diagram' or 'barcode', got '{kind}'.")
    plot_fn = plot_persistence_diagram if kind == "diagram" else plot_barcode

    idxs = _select_checkpoints(view, n_panels)
    n = len(idxs)
    ncols = max(1, ncols)
    nrows = math.ceil(n / ncols)
    if figsize is None:
        per = 4 if kind == "diagram" else 5
        figsize = (per * ncols, per * nrows)

    max_dim = max(dims) if dims else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, snap_idx in enumerate(idxs):
        snap = view.snapshots[snap_idx]
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        try:
            bundle = (builder(snap) if builder is not None
                      else _default_weight_bundle(snap, layer=layer))
            res = run_ph(bundle, max_dim=max_dim)
            label = (f"epoch {snap.epoch}" if snap.epoch is not None
                     else f"snap {snap_idx}")
            plot_fn(res.ph_result, dims=dims, ax=ax, title=label)
        except Exception as e:
            ax.text(0.5, 0.5, f"PH failed:\n{type(e).__name__}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.suptitle(title or f"{kind.capitalize()} evolution across training")
    fig.tight_layout()
    return fig