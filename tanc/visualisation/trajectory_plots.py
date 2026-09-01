"""trajectory_plots.py — scalar topological summaries over training epochs.

Each function takes a ``TrainingView`` and produces one plot tracking
some topological quantity along the training trajectory:

* ``plot_id_over_training``               — intrinsic dimension of one layer's activations vs epoch
* ``plot_id_trajectory_all_layers``       — same, overlaid or grid layout across layers
* ``plot_ph_dimension_over_training``     — sliding-window PH fractal dimension of the weight trajectory
* ``plot_magnitude_dimension_over_training`` — sliding-window magnitude dimension of the weight trajectory
* ``plot_ph_statistic_trajectory``        — total persistence / entropy / norm vs epoch
* ``plot_ph_statistics_panel``           — a grid of many diagram statistics vs epoch
* ``plot_betti_trajectory``               — Betti_d at a fixed filtration value vs epoch
* ``plot_diagram_distance_trajectory``    — Wasserstein/Bottleneck distance to a reference diagram vs epoch
  (``ref="previous"`` gives the consecutive-snapshot churn signal)
* ``detect_settle``                       — changepoint: the epoch where a trajectory reaches its plateau
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from tanc.model_extractor._snapshot import TrainingView, ModelSnapshot
from tanc.graph_builder._bundle import GraphBundle
from tanc.visualisation.visualisation_utils import make_figure

if TYPE_CHECKING:
    from tanc.pipeline.pipeline import TDAPipeline
    from tanc.topo_tools._result import TopoResult

# A "measure" is a key into a TopoResult: a statistics name
# (e.g. "H1_total_persistence"), the literal "dimension" (the scalar
# TopoResult.dimension), or a callable ``TopoResult -> float``.
Measure = "str | Callable[[TopoResult], float]"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _epochs_of(view: TrainingView) -> np.ndarray:
    return np.array(
        [s.epoch if s.epoch is not None else i for i, s in enumerate(view.snapshots)],
        dtype=float,
    )


def _smooth_series(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred moving average with edge padding (window <= 1 is a no-op)."""
    x = np.asarray(x, dtype=float)
    if window <= 1 or x.size < 2:
        return x
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(window) / window, mode="valid")


def detect_settle(
    values: Sequence[float],
    epochs: Sequence[float] | None = None,
    frac: float = 0.1,
    late_n: int = 15,
    smooth_window: int = 5,
) -> float:
    """First epoch from which a trajectory stays at its late-training plateau.

    A generic changepoint rule for scalar-over-epochs series (diagram
    distances, PH statistics, intrinsic dimension, …): smooth the series,
    take the early-training level (median of the first eighth) and the
    late-training baseline (median of the last ``late_n`` points), and return
    the first epoch after which the smoothed signal stays within ``frac`` of
    the baseline, relative to the early-to-late range.  Works for signals
    that decrease to a plateau or increase to one.

    Returns ``nan`` when no plateau is found (or the series is flat).
    NaN entries are ignored.

    Note: the rule is *relative* — a signal that collapses immediately and
    then keeps drifting slowly will report a late settle epoch; compare
    absolute magnitudes when that distinction matters.
    """
    v = np.asarray(values, dtype=float)
    if epochs is None:
        ep = np.arange(1, v.size + 1, dtype=float)
    else:
        ep = np.asarray(epochs, dtype=float)
    keep = np.isfinite(v)
    v, ep = v[keep], ep[keep]
    if v.size < max(6, late_n // 2):
        return float("nan")
    s = _smooth_series(v, smooth_window)
    early = float(np.median(s[: max(5, s.size // 8)]))
    late = float(np.median(s[-late_n:]))
    if early == late:
        return float("nan")
    z = (s - late) / (early - late)          # 1 at early level, 0 at plateau
    below = z <= frac
    for i in range(below.size):
        if below[i:].all():
            return float(ep[i])
    return float("nan")


def _mark_settle(ax: Axes, epochs, values, color: str = "0.3") -> float:
    """Draw a dotted vertical line at the detected settle epoch (if any)."""
    s = detect_settle(values, epochs)
    if np.isfinite(s):
        ax.axvline(s, color=color, lw=1.0, ls=":")
        ax.annotate(f"settles ~{s:.0f}", xy=(s, 0.95), xycoords=("data", "axes fraction"),
                    fontsize=8, color=color, ha="left", va="top", xytext=(4, 0),
                    textcoords="offset points")
    return s


def _prune_persistence(dgm: np.ndarray, min_persistence: float) -> np.ndarray:
    """Drop diagram points whose lifetime ``death - birth`` is below a threshold.

    Low-persistence points are topological noise; removing them before an
    (expensive) Wasserstein/bottleneck distance both denoises the comparison and
    slashes its cost — these metrics scale super-linearly in the number of points,
    and a dense weight-graph flag complex can emit hundreds of thousands of
    near-zero-lifetime H1 bars.  ``min_persistence <= 0`` is a no-op.
    """
    if not min_persistence or min_persistence <= 0 or dgm.size == 0:
        return dgm
    life = dgm[:, 1] - dgm[:, 0]
    return dgm[life >= min_persistence]


def _default_weight_bundle(snapshot: ModelSnapshot, layer: str | None = None) -> GraphBundle:
    """Build a default weight-graph GraphBundle for one snapshot."""
    from tanc.graph_builder.weight_graphs import build_weight_graph

    if layer is not None:
        Ws = [snapshot.weights[layer]]
    else:
        Ws = list(snapshot.weights.values())
    if not Ws:
        raise ValueError("Snapshot has no weight matrices.")
    return build_weight_graph(Ws, edge_weight="normalized", graph_scope="multipartite")


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsic dimension over training
# ─────────────────────────────────────────────────────────────────────────────

def _id_point_cloud(
    snapshot: ModelSnapshot,
    layer: str,
    source: str,
    orientation: str,
) -> "np.ndarray | None":
    """The (points, dims) array whose intrinsic dimension is estimated.

    ``source="activations"``: the layer's activation matrix
    ``(N_samples, N_neurons)`` — ``orientation="rows"`` keeps the samples as
    points; ``"cols"`` transposes so each **neuron's response profile** across
    the probe batch is one point.

    ``source="weights"``: the layer's weight matrix in the toolkit's
    ``(N_in, N_out)`` convention — ``orientation="rows"`` gives the input
    features as points in output space; ``"cols"`` the output neurons
    (receptive fields) as points in input space.
    """
    if source == "activations":
        X = snapshot.activations.get(layer)
    elif source == "weights":
        X = snapshot.weights.get(layer)
    else:
        raise ValueError(f"source must be 'activations' or 'weights', got {source!r}.")
    if X is None:
        return None
    if orientation not in ("rows", "cols"):
        raise ValueError(f"orientation must be 'rows' or 'cols', got {orientation!r}.")
    return X if orientation == "rows" else X.T


def _id_single_layer(
    snapshot: ModelSnapshot,
    layer: str,
    method: str,
    source: str = "activations",
    orientation: str = "rows",
    **method_kwargs,
) -> float:
    from tanc.topo_tools.dimension_tool import run_activation_id

    X = _id_point_cloud(snapshot, layer, source, orientation)
    if X is None:
        return float("nan")
    result = run_activation_id(
        [X],
        method=method,
        layer_labels=[layer],
        **method_kwargs,
    )
    return float(result["id_estimates"][0])


def plot_id_over_training(
    view: TrainingView,
    layer: str,
    methods: tuple[str, ...] | str = ("global",),
    source: str = "activations",
    orientation: str = "rows",
    mark_settle: bool = False,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    **method_kwargs,
) -> Figure:
    """Plot the intrinsic dimension of one layer's point cloud vs epoch.

    Parameters
    ----------
    view : TrainingView
    layer : str
        Layer name whose activations (or weights) to score.
    methods : tuple of str or str
        Estimator(s) to run.  Each is plotted as a separate line.
        Accepted values: ``"global"`` (Ong et al. 2026 global 2NN) and
        ``"local"`` (Ruppik et al. 2025 local 2NN).
    source : str
        ``"activations"`` (default) — the layer's response to the probe batch;
        ``"weights"`` — the layer's weight matrix (``(N_in, N_out)``
        convention; a SoRO layer contributes its effective weight).
    orientation : str
        Which side of the matrix forms the point cloud.  For activations:
        ``"rows"`` = probe samples as points (the usual data-manifold ID),
        ``"cols"`` = individual neurons as points (each neuron's response
        profile across the probe batch).  For weights: ``"rows"`` = input
        features as points in output space, ``"cols"`` = output neurons
        (receptive fields) as points in input space.
    mark_settle : bool
        Annotate the epoch where each series reaches its late-training
        plateau (see :func:`detect_settle`).
    **method_kwargs
        Forwarded to ``run_activation_id``.
    """
    if isinstance(methods, str):
        methods = (methods,)

    epochs = _epochs_of(view)
    fig, ax = make_figure(ax, figsize, default_figsize=(7, 4))

    for m in methods:
        ids = np.array(
            [_id_single_layer(s, layer, m, source=source, orientation=orientation,
                              **method_kwargs)
             for s in view.snapshots],
            dtype=float,
        )
        ax.plot(epochs, ids, marker="o", markersize=3, label=f"{m} 2NN")
        if mark_settle:
            _mark_settle(ax, epochs, ids)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Intrinsic dimension")
    ax.legend()
    src_lab = f"{source} {orientation}"
    ax.set_title(title or f"ID over training — {layer} ({src_lab})")
    fig.tight_layout()
    return fig


def plot_id_trajectory_all_layers(
    view: TrainingView,
    method: str = "global",
    layout: str = "overlay",
    layers: list[str] | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    **method_kwargs,
) -> Figure:
    """Plot ID vs epoch for every (selected) layer.

    Parameters
    ----------
    view : TrainingView
    method : str
        ``"global"`` or ``"local"``.
    layout : str
        ``"overlay"`` → all layers on one axes, one line each.
        ``"grid"``    → small-multiples, one subplot per layer.
    layers : list[str] or None
        Defaults to all activation layers in the final snapshot.
    ncols : int
        Grid columns (ignored for overlay).
    """
    if layers is None:
        layers = list(view.final_snapshot.activations.keys())
    if not layers:
        raise ValueError("No activation layers available in the TrainingView.")

    epochs = _epochs_of(view)
    series: dict[str, np.ndarray] = {}
    for name in layers:
        series[name] = np.array(
            [_id_single_layer(s, name, method, **method_kwargs) for s in view.snapshots],
            dtype=float,
        )

    if layout == "overlay":
        fig, ax = make_figure(None, figsize, default_figsize=(7, 4))
        for name, vals in series.items():
            ax.plot(epochs, vals, marker="o", markersize=3, label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Intrinsic dimension")
        ax.legend(fontsize=8)
        ax.set_title(title or f"ID trajectory ({method} 2NN)")
        fig.tight_layout()
        return fig

    if layout != "grid":
        raise ValueError(f"Unknown layout '{layout}'. Valid: 'overlay', 'grid'.")

    n = len(layers)
    ncols = max(1, ncols)
    nrows = math.ceil(n / ncols)
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, squeeze=False)
    for i, name in enumerate(layers):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.plot(epochs, series[name], marker="o", markersize=3)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ID")
    # blank unused panels
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    fig.suptitle(title or f"ID trajectory ({method} 2NN)")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory dimensions (PH dim, magnitude dim) via sliding window
# ─────────────────────────────────────────────────────────────────────────────

def _windowed_traj_dim(
    view: TrainingView,
    method: str,
    window: int,
    stride: int,
    layers: list[str] | None,
    distance: str,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute traj dimension on rolling windows of snapshots.

    Returns (epoch_centres, dim_values).
    """
    from tanc.graph_builder.weight_trajectory import build_weight_trajectory
    from tanc.topo_tools.dimension_tool import (
        compute_ph_dimension,
        compute_magnitude_dimension,
    )

    snaps = view.snapshots
    n = len(snaps)
    if window > n:
        raise ValueError(f"window={window} > snapshots available ({n}).")

    epoch_indices = _epochs_of(view)
    centres: list[float] = []
    dims: list[float] = []
    for start in range(0, n - window + 1, max(1, stride)):
        end = start + window
        sub = snaps[start:end]
        traj = [s.weight_matrices(layers) for s in sub]
        loss_values = None
        if distance == "loss_difference":
            loss_values = np.array(
                [s.loss if s.loss is not None else np.nan for s in sub], dtype=float
            )
            if not np.isfinite(loss_values).all():
                dims.append(float("nan"))
                centres.append(float(epoch_indices[start + window // 2]))
                continue
        bundle = build_weight_trajectory(
            traj, distance=distance, loss_values=loss_values,
        )
        try:
            if method == "ph_dimension":
                res = compute_ph_dimension(bundle.matrix, **kwargs)
                dims.append(float(res.get("ph_dimension", float("nan"))))
            else:
                res = compute_magnitude_dimension(bundle.matrix, **kwargs)
                dims.append(float(res.get("magnitude_dimension", float("nan"))))
        except (ValueError, np.linalg.LinAlgError):
            dims.append(float("nan"))
        centres.append(float(epoch_indices[start + window // 2]))

    return np.array(centres), np.array(dims)


def plot_ph_dimension_over_training(
    view: TrainingView,
    window: int = 20,
    stride: int = 1,
    layers: list[str] | None = None,
    distance: str = "euclidean",
    alpha: float = 1.0,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Sliding-window PH fractal dimension of the weight trajectory.

    For each window of consecutive snapshots, builds a weight-trajectory
    distance matrix and runs ``compute_ph_dimension``.

    Parameters
    ----------
    view : TrainingView
    window : int
        Number of snapshots per window (>=10 recommended by the estimator).
    stride : int
    layers : list[str] or None
        Weight layers to include in the trajectory (concatenated/flattened).
    distance : str
        ``"euclidean"`` | ``"cosine"`` | ``"loss_difference"``.
    alpha : float
        Lifetime exponent forwarded to ``compute_ph_dimension``.
    """
    centres, dims = _windowed_traj_dim(
        view, "ph_dimension", window, stride, layers, distance, alpha=alpha,
    )
    fig, ax = make_figure(ax, figsize, default_figsize=(7, 4))
    ax.plot(centres, dims, marker="o", markersize=3)
    ax.set_xlabel("Epoch (window centre)")
    ax.set_ylabel(f"PH dimension (α={alpha})")
    ax.set_title(title or f"PH dimension over training (window={window}, {distance})")
    fig.tight_layout()
    return fig


def plot_magnitude_dimension_over_training(
    view: TrainingView,
    window: int = 20,
    stride: int = 1,
    layers: list[str] | None = None,
    distance: str = "euclidean",
    t_range: tuple[float, float] = (0.1, 10.0),
    n_scale: int = 30,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Sliding-window magnitude dimension of the weight trajectory."""
    centres, dims = _windowed_traj_dim(
        view, "magnitude_dimension", window, stride, layers, distance,
        t_range=t_range, n_scale=n_scale,
    )
    fig, ax = make_figure(ax, figsize, default_figsize=(7, 4))
    ax.plot(centres, dims, marker="o", markersize=3, color="tab:purple")
    ax.set_xlabel("Epoch (window centre)")
    ax.set_ylabel("Magnitude dimension")
    ax.set_title(title or f"Magnitude dimension over training (window={window})")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-snapshot PH summaries vs epoch
# ─────────────────────────────────────────────────────────────────────────────

def _scale_normalized(bundle: GraphBundle) -> GraphBundle:
    """Return a copy of the bundle with unit mean off-diagonal magnitude.

    Divides the matrix by the mean absolute off-diagonal entry, making the
    subsequent persistence diagrams scale-free.  Useful for over-training
    comparisons where the *scale* of a representation grows with training
    (e.g. activation/logit norms growing as a classifier sharpens) and would
    otherwise dominate diagram statistics and distances.
    """
    M = np.asarray(bundle.matrix, dtype=float)
    off = M[~np.eye(M.shape[0], dtype=bool)] if M.shape[0] == M.shape[1] else M
    scale = float(np.mean(np.abs(off)))
    if not np.isfinite(scale) or scale == 0:
        return bundle
    return GraphBundle(
        matrix=M / scale,
        matrix_type=bundle.matrix_type,
        node_features=bundle.node_features,
        node_labels=bundle.node_labels,
        n_nodes=bundle.n_nodes,
        metadata={**(bundle.metadata or {}), "scale_normalized": scale},
    )


def _snapshot_ph_result(
    snapshot: ModelSnapshot,
    builder: Callable[[ModelSnapshot], GraphBundle] | None,
    layer: str | None,
    max_dim: int,
    normalize: bool = False,
    sparse: bool = False,
):
    from tanc.topo_tools.ph_tool import run_ph

    if builder is None:
        bundle = _default_weight_bundle(snapshot, layer=layer)
    else:
        bundle = builder(snapshot)
    if normalize:
        bundle = _scale_normalized(bundle)
    return run_ph(bundle, max_dim=max_dim, sparse=sparse)


def plot_ph_statistic_trajectory(
    view: TrainingView,
    stat: str = "total_persistence",
    dim: int = 0,
    layer: str | None = None,
    builder: Callable[[ModelSnapshot], GraphBundle] | None = None,
    normalize: bool = False,
    mark_settle: bool = False,
    sparse: bool = False,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot a per-snapshot PH summary statistic vs epoch.

    Parameters
    ----------
    view : TrainingView
    stat : str
        ``"total_persistence"`` | ``"persistence_norm"`` |
        ``"persistence_entropy"`` | ``"convex_hull_area"``.
    dim : int
        Homology dimension.
    layer : str or None
        If given (and no custom builder), only this layer's weight matrix
        feeds the default weight graph.  ``None`` → all weight layers.
    builder : callable or None
        Custom ``snapshot → GraphBundle`` builder.  Lets you swap in
        activation graphs, kernel graphs, etc.  ``None`` → default
        ``build_weight_graph`` over all (or one) weight layers.
    normalize : bool
        Scale-normalise each snapshot's bundle before PH (unit mean
        off-diagonal magnitude) so the statistic tracks shape, not the
        representation's growing scale.
    mark_settle : bool
        Annotate the epoch where the series reaches its late-training
        plateau (see :func:`detect_settle`).
    """
    valid_stats = {
        "total_persistence", "persistence_norm",
        "persistence_entropy", "convex_hull_area",
    }
    if stat not in valid_stats:
        raise ValueError(f"Unknown stat '{stat}'. Valid: {valid_stats}.")

    epochs = _epochs_of(view)
    values: list[float] = []
    for snap in view.snapshots:
        try:
            res = _snapshot_ph_result(snap, builder, layer, max_dim=max(dim, 1),
                                      normalize=normalize, sparse=sparse)
            values.append(float(res.statistics.get(f"H{dim}_{stat}", float("nan"))))
        except Exception:
            values.append(float("nan"))

    fig, ax = make_figure(ax, figsize, default_figsize=(7, 4))
    ax.plot(epochs, values, marker="o", markersize=3)
    if mark_settle:
        _mark_settle(ax, epochs, values)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"H{dim} {stat}")
    ax.set_title(title or f"{stat} (H{dim}) over training"
                 + (" (scale-normalised)" if normalize else ""))
    fig.tight_layout()
    return fig


def plot_betti_trajectory(
    view: TrainingView,
    dim: int = 1,
    epsilon: float | None = None,
    layer: str | None = None,
    builder: Callable[[ModelSnapshot], GraphBundle] | None = None,
    sparse: bool = False,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Betti_d at a fixed filtration value vs epoch.

    Parameters
    ----------
    view : TrainingView
    dim : int
    epsilon : float or None
        Filtration value at which to count bars.  ``None`` → median birth
        of the dimension's diagram computed independently per snapshot
        (good first look; pass an explicit value for comparability).
    layer, builder
        Same semantics as ``plot_ph_statistic_trajectory``.
    """
    from tanc.topo_tools.ph_tool import betti_number

    epochs = _epochs_of(view)
    counts: list[float] = []
    for snap in view.snapshots:
        try:
            res = _snapshot_ph_result(snap, builder, layer, max_dim=max(dim, 1),
                                      sparse=sparse)
            dgm = res.ph_result.diagrams.get(dim, np.empty((0, 2)))
            if dgm.shape[0] == 0:
                counts.append(0.0)
                continue
            eps = epsilon if epsilon is not None else float(np.median(dgm[:, 0]))
            counts.append(float(betti_number(dgm, eps)))
        except Exception:
            counts.append(float("nan"))

    fig, ax = make_figure(ax, figsize, default_figsize=(7, 4))
    ax.plot(epochs, counts, marker="o", markersize=3)
    eps_label = f"ε={epsilon}" if epsilon is not None else "ε=median(birth)"
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Betti_{dim}")
    ax.set_title(title or f"Betti_{dim} trajectory ({eps_label})")
    fig.tight_layout()
    return fig


def plot_diagram_distance_matrix(
    view: TrainingView,
    metric: str = "wasserstein",
    dim: int = 1,
    layer: str | None = None,
    builder: Callable[[ModelSnapshot], GraphBundle] | None = None,
    sparse: bool = False,
    min_persistence: float = 0.0,
    cmap: str = "magma",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Heatmap of pairwise diagram distances between every pair of snapshots.

    Computes PH at each snapshot, then a symmetric ``(T, T)`` matrix of
    Wasserstein or bottleneck distances.  Reveals phase transitions and
    stable regions during training.

    Parameters
    ----------
    view : TrainingView
    metric : str
        ``"wasserstein"`` | ``"bottleneck"`` (requires ``persim``).
    dim : int
    layer, builder
        Same semantics as ``plot_ph_statistic_trajectory``.
    min_persistence : float
        Drop diagram points with lifetime ``death - birth < min_persistence``
        before computing distances (see :func:`_prune_persistence`).  Strongly
        recommended for dense weight-graph constructions, whose diagrams carry
        hundreds of thousands of near-zero-lifetime H1 bars that make the
        distance both noisy and very slow.  ``0`` (default) keeps all points.
    """
    if metric == "wasserstein":
        from tanc.topo_tools.ph_tool import wasserstein_distance as dist_fn
    elif metric == "bottleneck":
        from tanc.topo_tools.ph_tool import bottleneck_distance as dist_fn
    else:
        raise ValueError(f"metric must be 'wasserstein' or 'bottleneck', got '{metric}'.")

    epochs = _epochs_of(view)
    diagrams: list[np.ndarray] = []
    for snap in view.snapshots:
        try:
            res = _snapshot_ph_result(snap, builder, layer, max_dim=max(dim, 1),
                                      sparse=sparse)
            dgm = res.ph_result.diagrams.get(dim, np.empty((0, 2)))
            diagrams.append(_prune_persistence(dgm, min_persistence))
        except Exception:
            diagrams.append(np.empty((0, 2)))

    T = len(diagrams)
    D = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            if diagrams[i].size == 0 or diagrams[j].size == 0:
                D[i, j] = float("nan")
            else:
                D[i, j] = dist_fn(diagrams[i], diagrams[j])
            D[j, i] = D[i, j]

    if figsize is None:
        figsize = (6, 5)
    fig, ax = plt.subplots(figsize=figsize)
    extent = (epochs[0], epochs[-1], epochs[-1], epochs[0])
    im = ax.imshow(D, cmap=cmap, aspect="auto", extent=extent)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=f"{metric} distance (H{dim})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epoch")
    ax.set_title(title or f"Pairwise diagram {metric} distances (H{dim})")
    fig.tight_layout()
    return fig


def plot_ph_statistic_pairplot(
    view: TrainingView,
    stats: tuple[str, ...] = ("total_persistence", "persistence_norm",
                              "persistence_entropy"),
    dim: int = 0,
    layer: str | None = None,
    builder: Callable[[ModelSnapshot], GraphBundle] | None = None,
    sparse: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Scatter matrix of multiple PH summary statistics across snapshots.

    Points are coloured by epoch so you can see the trajectory through
    statistic-space.  Diagonal panels show histograms.

    Parameters
    ----------
    view : TrainingView
    stats : tuple of str
        Any subset of ``"total_persistence"``, ``"persistence_norm"``,
        ``"persistence_entropy"``, ``"convex_hull_area"``.
    dim : int
        Homology dimension to read from per-snapshot statistics.
    layer, builder
        Same semantics as ``plot_ph_statistic_trajectory``.
    """
    valid = {"total_persistence", "persistence_norm",
             "persistence_entropy", "convex_hull_area"}
    bad = set(stats) - valid
    if bad:
        raise ValueError(f"Unknown stats {bad}. Valid: {valid}.")

    epochs = _epochs_of(view)
    rows: list[list[float]] = []
    for snap in view.snapshots:
        row: list[float] = []
        try:
            res = _snapshot_ph_result(snap, builder, layer, max_dim=max(dim, 1),
                                      sparse=sparse)
            for s in stats:
                row.append(float(res.statistics.get(f"H{dim}_{s}", float("nan"))))
        except Exception:
            row = [float("nan")] * len(stats)
        rows.append(row)
    M = np.array(rows)

    n = len(stats)
    if figsize is None:
        figsize = (3 * n, 3 * n)
    fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj((epochs - epochs.min()) / max(epochs.max() - epochs.min(), 1e-9))

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            xs, ys = M[:, j], M[:, i]
            if i == j:
                finite = np.isfinite(xs)
                if finite.any():
                    ax.hist(xs[finite], bins=15, color="tab:gray", alpha=0.8)
                ax.set_yticks([])
            else:
                ax.scatter(xs, ys, c=colors, s=18, edgecolors="none")
            if i == n - 1:
                ax.set_xlabel(stats[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(stats[i], fontsize=9)
    # Colourbar for epoch
    sm = plt.cm.ScalarMappable(cmap=cmap_obj,
                               norm=plt.Normalize(vmin=epochs.min(),
                                                  vmax=epochs.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04, label="Epoch")
    fig.suptitle(title or f"PH-statistic pairplot (H{dim})")
    return fig


def plot_diagram_distance_trajectory(
    view: TrainingView,
    ref: str = "final",
    metric: str = "wasserstein",
    dim: int = 1,
    layer: str | None = None,
    builder: Callable[[ModelSnapshot], GraphBundle] | None = None,
    normalize: bool = False,
    mark_settle: bool = False,
    sparse: bool = False,
    min_persistence: float = 0.0,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Distance from each snapshot's persistence diagram to a reference.

    Parameters
    ----------
    view : TrainingView
    ref : str
        ``"final"`` (last snapshot) | ``"initial"`` (first snapshot) |
        ``"previous"`` (the immediately preceding snapshot — a *churn*
        signal: how much the representation's topology changed between
        consecutive captures; it collapses to ~0 when training reaches a
        steady state).
    metric : str
        ``"wasserstein"`` | ``"bottleneck"`` (requires ``persim``).
    dim : int
    layer, builder
        Same semantics as ``plot_ph_statistic_trajectory``.
    normalize : bool
        Scale-normalise each snapshot's bundle (unit mean off-diagonal
        magnitude) before PH, so distances track *shape* change rather than
        the growth of the representation's scale over training.
    mark_settle : bool
        Annotate the epoch where the series reaches its late-training
        plateau (see :func:`detect_settle`).
    min_persistence : float
        Drop diagram points with lifetime ``death - birth < min_persistence``
        before computing distances (see :func:`_prune_persistence`).  Strongly
        recommended for dense weight-graph constructions, whose diagrams carry
        hundreds of thousands of near-zero-lifetime H1 bars that make the
        distance both noisy and very slow.  ``0`` (default) keeps all points.
    """
    if ref not in {"final", "initial", "previous"}:
        raise ValueError(f"ref must be 'final', 'initial' or 'previous', got '{ref}'.")
    if metric == "wasserstein":
        from tanc.topo_tools.ph_tool import wasserstein_distance as dist_fn
    elif metric == "bottleneck":
        from tanc.topo_tools.ph_tool import bottleneck_distance as dist_fn
    else:
        raise ValueError(f"metric must be 'wasserstein' or 'bottleneck', got '{metric}'.")

    epochs = _epochs_of(view)
    diagrams: list[np.ndarray] = []
    for snap in view.snapshots:
        try:
            res = _snapshot_ph_result(snap, builder, layer, max_dim=max(dim, 1),
                                      normalize=normalize, sparse=sparse)
            dgm = res.ph_result.diagrams.get(dim, np.empty((0, 2)))
            diagrams.append(_prune_persistence(dgm, min_persistence))
        except Exception:
            diagrams.append(np.empty((0, 2)))

    if ref == "previous":
        distances = [float("nan")] + [
            dist_fn(a, b) if a.size and b.size else float("nan")
            for a, b in zip(diagrams[:-1], diagrams[1:])
        ]
        ylabel = f"{metric} distance to previous diagram (H{dim})"
    else:
        ref_dgm = diagrams[-1] if ref == "final" else diagrams[0]
        distances = [dist_fn(d, ref_dgm) if d.size and ref_dgm.size else float("nan")
                     for d in diagrams]
        ylabel = f"{metric} distance to {ref} diagram (H{dim})"

    fig, ax = make_figure(ax, figsize, default_figsize=(7, 4))
    ax.plot(epochs, distances, marker="o", markersize=3)
    if mark_settle:
        _mark_settle(ax, epochs, distances)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Diagram distance trajectory — {metric}, H{dim}"
                 + (" (scale-normalised)" if normalize else ""))
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-driven trajectories: run *any* configured TDAPipeline over a
# TrainingView and track one-or-more scalar measures per epoch, optionally
# split per layer.  This is the generic counterpart to the hand-rolled
# per-snapshot ``.fit()`` loop — you configure a pipeline once (including
# ``TDAPipeline.from_paper(...)``) and get the whole time series back.
# ─────────────────────────────────────────────────────────────────────────────

def _measure_label(measure) -> str:
    """Human-readable column name for a measure spec."""
    if callable(measure):
        return getattr(measure, "__name__", "measure")
    return str(measure)


def _extract_measure(result: "TopoResult", measure) -> float:
    """Pull a single scalar for *measure* out of a TopoResult.

    ``callable`` → ``measure(result)``; ``"dimension"`` → ``result.dimension``;
    any other string → ``result.statistics[measure]``.  Missing values (and the
    ``None`` from an empty ``dimension``) come back as ``nan``.
    """
    if callable(measure):
        val = measure(result)
        return float(val) if val is not None else float("nan")
    if measure == "dimension":
        val = getattr(result, "dimension", None)
        return float(val) if val is not None else float("nan")
    stats = getattr(result, "statistics", None) or {}
    val = stats.get(measure, float("nan"))
    return float(val) if val is not None else float("nan")


def _per_layer_representation(representation, layer: str):
    """Build a single-arg ``snapshot -> array`` representation for one layer.

    ``representation`` may be ``"weights"``, ``"activations"``, or a two-arg
    callable ``(snapshot, layer) -> array``.
    """
    if callable(representation):
        return lambda s, _l=layer: representation(s, _l)
    if representation in (None, "weights"):
        # 1-element list → the pipeline builds a graph from this layer alone.
        return lambda s, _l=layer: s.weight_matrices([_l])
    if representation == "activations":
        # 1-element list to match the weights convention: dimension estimators
        # iterate a list of layer matrices, and per-layer PH builders accept it.
        return lambda s, _l=layer: [s.activation_matrix(_l)]
    raise ValueError(
        f"representation={representation!r} is not supported with per-layer "
        "trajectories; use 'weights', 'activations', or a (snapshot, layer) "
        "callable."
    )


@dataclass
class TrajectorySeries:
    """Scalar topological measures tracked across a training trajectory.

    Produced by :func:`pipeline_trajectory` / ``TDAPipeline.over_training``.

    Attributes
    ----------
    epochs : ndarray, shape ``(n_snapshots,)``
    values : dict[str, dict[str, ndarray]]
        ``measure name → {series label → per-epoch array}``.  The series label
        is a layer name when split per layer, else ``"all"``.
    paper_reference : str or None
    """

    epochs: np.ndarray
    values: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    paper_reference: str | None = None

    @property
    def measures(self) -> list[str]:
        return list(self.values.keys())

    @property
    def series_labels(self) -> list[str]:
        labels: list[str] = []
        for per_series in self.values.values():
            for label in per_series:
                if label not in labels:
                    labels.append(label)
        return labels

    def to_frame(self):
        """Return a tidy ``(epoch, measure, series, value)`` pandas DataFrame."""
        import pandas as pd

        rows = []
        for measure, per_series in self.values.items():
            for label, arr in per_series.items():
                for epoch, value in zip(self.epochs, arr):
                    rows.append({"epoch": epoch, "measure": measure,
                                 "series": label, "value": value})
        return pd.DataFrame(rows)

    def plot(
        self,
        layout: str = "grid",
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        marker: str = "o",
    ) -> Figure:
        """Plot the tracked measures vs epoch.

        One line per series (e.g. per layer).  With several measures,
        ``layout="grid"`` (default) gives one subplot per measure; ``"overlay"``
        draws them all on a single axis.
        """
        measures = self.measures
        if not measures:
            raise ValueError("TrajectorySeries has no measures to plot.")

        # Single measure, or an explicit overlay → one shared axis.
        if len(measures) == 1 or layout == "overlay":
            fig, ax = make_figure(ax, figsize, default_figsize=(7, 4))
            multi_series = False
            for measure in measures:
                for label, arr in self.values[measure].items():
                    lbl = label if len(measures) == 1 else f"{measure} — {label}"
                    ax.plot(self.epochs, arr, marker=marker, markersize=3, label=lbl)
                    multi_series = multi_series or label != "all"
            ax.set_xlabel("Epoch")
            ax.set_ylabel(measures[0] if len(measures) == 1 else "value")
            if multi_series or len(measures) > 1:
                ax.legend()
            ax.set_title(title or self._auto_title())
            fig.tight_layout()
            return fig

        # Several measures → a grid, one subplot each.
        n = len(measures)
        ncols = min(n, 2)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize or (6.5 * ncols, 3.6 * nrows),
            squeeze=False,
        )
        flat = axes.flat
        for cell, measure in zip(flat, measures):
            per_series = self.values[measure]
            for label, arr in per_series.items():
                cell.plot(self.epochs, arr, marker=marker, markersize=3, label=label)
            cell.set_xlabel("Epoch")
            cell.set_ylabel(measure)
            cell.set_title(measure)
            if any(label != "all" for label in per_series):
                cell.legend()
        for extra in list(flat)[n:]:
            extra.set_visible(False)
        fig.suptitle(title or self._auto_title())
        fig.tight_layout()
        return fig

    def _auto_title(self) -> str:
        base = "Topology over training"
        return f"{base} — {self.paper_reference}" if self.paper_reference else base


def pipeline_trajectory(
    view: TrainingView,
    pipe: "TDAPipeline",
    measures,
    layers: Sequence[str] | None = None,
    representation=None,
    on_error: str = "nan",
) -> TrajectorySeries:
    """Run *pipe* on every snapshot of *view* and track scalar measures per epoch.

    One ``pipe.fit`` per snapshot (per layer, when ``layers`` is given); every
    requested measure is then read off that same :class:`TopoResult`.

    Parameters
    ----------
    view : TrainingView
    pipe : TDAPipeline
        Any configured pipeline, e.g. ``TDAPipeline.from_paper("watanabe2021")``.
    measures : str | callable | sequence of them
        What to extract per snapshot.  A statistics key
        (``"H1_total_persistence"``), the literal ``"dimension"``, or a callable
        ``TopoResult -> float``.
    layers : sequence of str or None
        When given, the pipeline is run on each layer independently and the
        result is one line per layer (concatenating layers is often meaningless).
        ``None`` → a single ``"all"`` series using the pipeline's own handling.
    representation : str | callable or None
        How to turn each snapshot into pipeline input.  With ``layers`` set:
        ``"weights"`` (default), ``"activations"``, or a ``(snapshot, layer)``
        callable.  With ``layers=None``: forwarded to ``pipe.fit`` as-is (any
        representation keyword / layer name / ``snapshot`` callable), ``None``
        letting the pipeline decide.
    on_error : str
        ``"nan"`` (default) records ``nan`` for a snapshot whose fit raises;
        ``"raise"`` propagates the exception.

    Returns
    -------
    TrajectorySeries
    """
    if on_error not in {"nan", "raise"}:
        raise ValueError(f"on_error must be 'nan' or 'raise', got {on_error!r}.")

    if isinstance(measures, str) or callable(measures):
        measures = [measures]
    measures = list(measures)
    if not measures:
        raise ValueError("Provide at least one measure to track.")

    epochs = _epochs_of(view)
    n = len(view.snapshots)

    if layers is None:
        series_specs: list[tuple[str, object]] = [("all", representation)]
    else:
        series_specs = [
            (layer, _per_layer_representation(representation, layer))
            for layer in layers
        ]

    labels = [_measure_label(m) for m in measures]
    values: dict[str, dict[str, np.ndarray]] = {
        lbl: {series: np.full(n, np.nan) for series, _ in series_specs}
        for lbl in labels
    }

    for i, snap in enumerate(view.snapshots):
        for series, rep in series_specs:
            try:
                result = pipe.fit(snap, representation=rep)
            except Exception:
                if on_error == "raise":
                    raise
                continue  # leave nan
            if isinstance(result, list):
                # Some builders return one result per layer; per-layer specs
                # already isolate a single layer, so take the first.
                if not result:
                    continue
                result = result[0]
            for measure, lbl in zip(measures, labels):
                try:
                    values[lbl][series][i] = _extract_measure(result, measure)
                except Exception:
                    if on_error == "raise":
                        raise
                    # leave nan

    return TrajectorySeries(
        epochs=epochs,
        values=values,
        paper_reference=getattr(pipe, "paper_reference", None),
    )

# Default panel: H1 (loops) first, then H0 — the non-redundant, intuition-rich
# subset of diagram_statistics for watching topology evolve over training.
_PH_PANEL_DEFAULT = (
    "n_bars", "total", "max", "mean", "min", "std",
    "entropy", "gini", "birth_mean",
)


def plot_ph_statistics_panel(
    view: TrainingView,
    dim: int = 1,
    stats: Sequence[str] = _PH_PANEL_DEFAULT,
    layer: str | None = None,
    builder: Callable[[ModelSnapshot], GraphBundle] | None = None,
    normalize: bool = False,
    mark_settle: bool = False,
    sparse: bool = False,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Grid of persistence-diagram statistics vs epoch — a fuller picture than
    :func:`plot_ph_statistic_trajectory` (one stat) of how a layer's topology
    evolves during training.

    Computes the ``dim``-homology diagram at each snapshot (default: the
    weight graph; pass a ``builder`` for a point cloud / activation graph),
    then plots each of ``stats`` (keys of
    :data:`~tanc.topo_tools.ph_tool.DIAGRAM_STAT_LABELS`) in its own
    subplot with an optional settle marker.

    Parameters mirror :func:`plot_ph_statistic_trajectory`; ``stats`` selects
    which summaries to show.
    """
    from tanc.topo_tools.ph_tool import diagram_statistics, DIAGRAM_STAT_LABELS

    bad = set(stats) - set(DIAGRAM_STAT_LABELS)
    if bad:
        raise ValueError(f"Unknown stats {bad}. Valid: {set(DIAGRAM_STAT_LABELS)}.")

    epochs = _epochs_of(view)
    rows = []
    for snap in view.snapshots:
        try:
            res = _snapshot_ph_result(snap, builder, layer, max_dim=max(dim, 1),
                                      normalize=normalize, sparse=sparse)
            dgm = res.ph_result.diagrams.get(dim, np.empty((0, 2)))
            rows.append(diagram_statistics(dgm))
        except Exception:
            rows.append({k: float("nan") for k in DIAGRAM_STAT_LABELS})

    n = len(stats)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.0 * ncols, 2.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, squeeze=False)
    axf = axes.ravel()
    for ax, stat in zip(axf, stats):
        vals = np.array([r.get(stat, float("nan")) for r in rows], dtype=float)
        ax.plot(epochs, vals, marker="o", markersize=2.5, lw=1.4)
        if mark_settle:
            _mark_settle(ax, epochs, vals)
        ax.set_title(f"H{dim} · {DIAGRAM_STAT_LABELS[stat]}", fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
    for ax in axf[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    fig.suptitle(title or f"H{dim} persistence-diagram statistics over training", y=1.0)
    fig.tight_layout()
    return fig
