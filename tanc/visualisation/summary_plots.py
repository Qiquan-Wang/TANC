"""summary_plots.py — multi-panel dashboards and animations of training runs."""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tanc.model_extractor._snapshot import TrainingView, ModelSnapshot
from tanc.graph_builder._bundle import GraphBundle
from tanc.visualisation.trajectory_plots import _default_weight_bundle


# ─────────────────────────────────────────────────────────────────────────────
# Training-summary dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_summary(
    view: TrainingView,
    layer: str | None = None,
    builder: Callable[[ModelSnapshot], GraphBundle] | None = None,
    figsize: tuple[float, float] = (15, 9),
    title: str | None = None,
) -> Figure:
    """One-call 2x3 dashboard summarising a training run.

    Panels (left-to-right, top-to-bottom):

    1. Loss & accuracy curves.
    2. Per-layer weight Frobenius-norm trajectory.
    3. Intrinsic dimension across layers at the final snapshot.
    4. PH total-persistence (H0) over training.
    5. Betti_1 trajectory.
    6. Wasserstein distance to the final diagram (H1).

    Panels that fail (missing data, optional dep, ripser unavailable, ...)
    fall back to a small placeholder message rather than aborting the whole
    figure.
    """
    from tanc.visualisation.training_plots import (
        plot_training_curves,
        plot_weight_norm_trajectory,
    )
    from tanc.visualisation.trajectory_plots import (
        plot_id_trajectory_all_layers,
        plot_ph_statistic_trajectory,
        plot_betti_trajectory,
        plot_diagram_distance_trajectory,
    )

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    panels: list[tuple[str, Callable[[], None]]] = [
        ("Loss & accuracy",
         lambda: plot_training_curves(view, twin_axes=True, ax=axes[0][0])),
        ("Weight norm per layer",
         lambda: plot_weight_norm_trajectory(view, ax=axes[0][1])),
        ("ID over training",
         lambda: plot_id_trajectory_all_layers(
             view, method="global", layout="overlay", figsize=None) or None),
        ("H0 total persistence",
         lambda: plot_ph_statistic_trajectory(
             view, stat="total_persistence", dim=0,
             layer=layer, builder=builder, ax=axes[1][0])),
        ("Betti_1 trajectory",
         lambda: plot_betti_trajectory(
             view, dim=1, layer=layer, builder=builder, ax=axes[1][1])),
        ("Distance to final diagram",
         lambda: plot_diagram_distance_trajectory(
             view, ref="final", metric="wasserstein", dim=1,
             layer=layer, builder=builder, ax=axes[1][2])),
    ]
    coords = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for (label, fn), (r, c) in zip(panels, coords):
        ax = axes[r][c]
        try:
            # Functions that take ax= draw into it directly; functions that
            # build their own figure (ID trajectory) are handled separately.
            if label == "ID over training":
                # No ax-kwarg version exists; build standalone then copy.
                _draw_id_into_axes(view, ax)
            else:
                fn()
        except Exception as exc:
            ax.clear()
            ax.text(0.5, 0.5, f"{label}\n[{type(exc).__name__}]",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, alpha=0.7)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title or "Training summary", y=1.02)
    fig.tight_layout()
    return fig


def _draw_id_into_axes(view: TrainingView, ax) -> None:
    """ID-over-training overlay drawn into an existing axes."""
    from tanc.visualisation.trajectory_plots import _id_single_layer

    layers = list(view.final_snapshot.activations.keys())
    epochs = np.array(
        [s.epoch if s.epoch is not None else i for i, s in enumerate(view.snapshots)],
        dtype=float,
    )
    for name in layers:
        vals = np.array(
            [_id_single_layer(s, name, "global") for s in view.snapshots],
            dtype=float,
        )
        ax.plot(epochs, vals, marker="o", markersize=3, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Intrinsic dim")
    ax.legend(fontsize=7)
    ax.set_title("ID over training (global 2NN)")


# ─────────────────────────────────────────────────────────────────────────────
# Training animation
# ─────────────────────────────────────────────────────────────────────────────

def make_training_animation(
    view: TrainingView,
    kind: str = "diagram",
    out: str | None = None,
    dim: int = 1,
    layer: str | None = None,
    builder: Callable[[ModelSnapshot], GraphBundle] | None = None,
    fps: int = 4,
    dpi: int = 100,
):
    """Save (or return) an animation of PH evolving across training snapshots.

    Each frame is computed with the per-snapshot logic shared by
    ``plot_diagram_evolution``: build a GraphBundle, run PH, render.

    Parameters
    ----------
    view : TrainingView
    kind : str
        ``"diagram"`` | ``"barcode"`` | ``"landscape"`` | ``"image"``.
    out : str or None
        If provided, save to this path.  ``.gif`` uses the pillow writer;
        ``.mp4`` requires ffmpeg.  When ``None`` the FuncAnimation object
        is returned without saving.
    dim : int
        Homology dimension.
    fps : int
    dpi : int

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    import matplotlib.animation as manim
    from tanc.topo_tools.ph_tool import run_ph
    from tanc.visualisation.representations import (
        plot_persistence_diagram, plot_barcode,
        plot_persistence_landscape, plot_persistence_image,
    )

    plot_fn = {
        "diagram":   plot_persistence_diagram,
        "barcode":   plot_barcode,
        "landscape": plot_persistence_landscape,
        "image":     plot_persistence_image,
    }.get(kind)
    if plot_fn is None:
        raise ValueError(
            f"kind must be 'diagram', 'barcode', 'landscape' or 'image' (got {kind!r})."
        )

    # Pre-compute every frame's PH so we know the global axis range.
    per_frame: list = []
    for snap in view.snapshots:
        try:
            bundle = (builder(snap) if builder is not None
                      else _default_weight_bundle(snap, layer=layer))
            res = run_ph(bundle, max_dim=max(dim, 1))
            per_frame.append((snap, res.ph_result))
        except Exception:
            per_frame.append((snap, None))

    fig, ax = plt.subplots(figsize=(6, 5))

    def _render(i: int):
        ax.clear()
        snap, ph = per_frame[i]
        label = (f"epoch {snap.epoch}" if snap.epoch is not None else f"snap {i}")
        if ph is None:
            ax.text(0.5, 0.5, "PH failed", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            if kind in {"landscape", "image"}:
                plot_fn(ph, dim=dim, ax=ax, title=label)
            else:
                plot_fn(ph, dims=[dim], ax=ax, title=label)

    anim = manim.FuncAnimation(
        fig, _render, frames=len(per_frame), interval=int(1000 / fps),
    )
    if out is not None:
        if out.lower().endswith(".gif"):
            writer = manim.PillowWriter(fps=fps)
        else:
            writer = manim.FFMpegWriter(fps=fps)
        anim.save(out, writer=writer, dpi=dpi)
    return anim