"""training_plots.py — plots driven directly by a ``TrainingView``.

Lightweight one-call helpers for the kinds of curves you usually want
when staring at a training run: loss/accuracy, per-layer weight norm,
and per-layer activation statistics.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from tanc.model_extractor._snapshot import TrainingView
from tanc.visualisation.visualisation_utils import make_figure


# ─────────────────────────────────────────────────────────────────────────────
# Loss & accuracy
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    view: TrainingView,
    twin_axes: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot training loss and validation accuracy across the run.

    Parameters
    ----------
    view : TrainingView
    twin_axes : bool
        ``True`` → one axes with loss on the left y-axis and accuracy on a
        twin right y-axis.  ``False`` → two stacked subplots.
    ax : Axes or None
        Only honoured in ``twin_axes=True`` mode.
    figsize : (w, h) or None
    title : str or None

    Returns
    -------
    matplotlib Figure
    """
    epochs = np.array(
        [s.epoch if s.epoch is not None else i for i, s in enumerate(view.snapshots)],
        dtype=float,
    )
    losses = view.loss_trajectory()
    accs = view.accuracy_trajectory()
    has_loss = np.isfinite(losses).any()
    has_acc = np.isfinite(accs).any()

    if not has_loss and not has_acc:
        raise ValueError("TrainingView contains neither loss nor accuracy data.")

    if twin_axes:
        fig, ax = make_figure(ax, figsize, default_figsize=(7, 4))
        ax2 = None
        if has_loss:
            ax.plot(epochs, losses, color="tab:red", label="loss")
            ax.set_ylabel("Loss", color="tab:red")
            ax.tick_params(axis="y", labelcolor="tab:red")
        if has_acc:
            ax2 = ax.twinx() if has_loss else ax
            ax2.plot(epochs, accs, color="tab:blue", label="accuracy")
            ax2.set_ylabel("Accuracy", color="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_xlabel("Epoch")
        ax.set_title(title or "Training curves")
        fig.tight_layout()
        return fig

    n_panels = int(has_loss) + int(has_acc)
    if figsize is None:
        figsize = (7, 3 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]
    i = 0
    if has_loss:
        axes[i].plot(epochs, losses, color="tab:red")
        axes[i].set_ylabel("Loss")
        i += 1
    if has_acc:
        axes[i].plot(epochs, accs, color="tab:blue")
        axes[i].set_ylabel("Accuracy")
    axes[-1].set_xlabel("Epoch")
    fig.suptitle(title or "Training curves")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Weight norm trajectory
# ─────────────────────────────────────────────────────────────────────────────

def plot_weight_norm_trajectory(
    view: TrainingView,
    layers: list[str] | None = None,
    norm: str = "fro",
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot ‖W_l‖ per layer across training.

    Parameters
    ----------
    view : TrainingView
    layers : list[str] or None
        Layer names.  ``None`` → all weight layers in the first snapshot.
    norm : str
        ``"fro"`` Frobenius norm, ``"l1"`` sum of absolute values,
        ``"l2"`` 2-norm (largest singular value), ``"nuc"`` nuclear norm.
    """
    if not view.snapshots:
        raise ValueError("TrainingView is empty.")

    if layers is None:
        layers = list(view.snapshots[0].weights.keys())
    if not layers:
        raise ValueError("No weight layers available in the TrainingView.")

    epochs = np.array(
        [s.epoch if s.epoch is not None else i for i, s in enumerate(view.snapshots)],
        dtype=float,
    )

    norm_kind = {"fro": "fro", "l1": 1, "l2": 2, "nuc": "nuc"}.get(norm)
    if norm_kind is None:
        raise ValueError(f"Unknown norm '{norm}'. Valid: 'fro','l1','l2','nuc'.")

    fig, ax = make_figure(ax, figsize, default_figsize=(7, 4))
    for name in layers:
        series = []
        for s in view.snapshots:
            W = s.weights.get(name)
            if W is None:
                series.append(np.nan)
                continue
            if norm == "l1":
                series.append(float(np.sum(np.abs(W))))
            else:
                series.append(float(np.linalg.norm(W, ord=norm_kind)))
        ax.plot(epochs, series, marker="o", markersize=3, label=name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"‖W‖_{norm}")
    ax.legend(fontsize=8)
    ax.set_title(title or f"Weight {norm}-norm trajectory")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Activation statistic trajectory
# ─────────────────────────────────────────────────────────────────────────────

def plot_activation_stats_trajectory(
    view: TrainingView,
    layer: str,
    stats: tuple[str, ...] = ("mean", "std", "sparsity"),
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot summary statistics of one layer's activations across training.

    ``"mean"``     — mean activation value across all units and samples.
    ``"std"``      — standard deviation of activations.
    ``"sparsity"`` — fraction of zero activations (post-ReLU sparsity proxy).
    ``"max"``      — maximum absolute activation.

    Parameters
    ----------
    view : TrainingView
    layer : str
        Layer name whose activations to summarise.
    stats : tuple of str
    """
    valid = {"mean", "std", "sparsity", "max"}
    bad = set(stats) - valid
    if bad:
        raise ValueError(f"Unknown stats {bad}. Valid: {valid}.")

    epochs: list[float] = []
    series: dict[str, list[float]] = {s: [] for s in stats}
    for i, snap in enumerate(view.snapshots):
        if layer not in snap.activations:
            continue
        a = snap.activations[layer]
        epochs.append(snap.epoch if snap.epoch is not None else i)
        for stat in stats:
            if stat == "mean":
                series[stat].append(float(np.mean(a)))
            elif stat == "std":
                series[stat].append(float(np.std(a)))
            elif stat == "sparsity":
                series[stat].append(float(np.mean(a == 0.0)))
            elif stat == "max":
                series[stat].append(float(np.max(np.abs(a))))

    if not epochs:
        raise ValueError(f"Layer '{layer}' not captured in any snapshot's activations.")

    fig, ax = make_figure(ax, figsize, default_figsize=(7, 4))
    for stat in stats:
        ax.plot(epochs, series[stat], marker="o", markersize=3, label=stat)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Activation statistic")
    ax.legend()
    ax.set_title(title or f"Activation stats — {layer}")
    fig.tight_layout()
    return fig