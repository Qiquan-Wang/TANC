"""graph_plots.py — visualisations for GraphBundles and TU scores."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from tanc.graph_builder._bundle import GraphBundle
from tanc.visualisation.visualisation_utils import make_figure


# ─────────────────────────────────────────────────────────────────────────────
# Polyhedral (ReLU linear-region) decomposition of a 2-D input space
# ─────────────────────────────────────────────────────────────────────────────

def plot_polyhedral_regions(
    xx: np.ndarray,
    yy: np.ndarray,
    patterns: np.ndarray,
    points: np.ndarray | None = None,
    point_labels: np.ndarray | None = None,
    decision: np.ndarray | None = None,
    fill: bool = True,
    boundary_color: str = "0.6",
    boundary_alpha: float = 0.4,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6.0, 6.0),
    title: str | None = None,
    seed: int = 0,
) -> Figure:
    """Draw a ReLU network's **polyhedral decomposition** over a 2-D input grid.

    Each distinct binary activation pattern is one linear (polyhedral) region of
    input space — the object Liu et al. (2023) run persistent homology on.  This
    plots those regions directly: fill each region a distinct colour and draw the
    region boundaries (the network's piecewise-linear cuts).  Optionally overlay
    the classifier's **decision boundary**, which is built from straight pieces
    that kink at region boundaries — showing how the regions serve the task.

    Parameters
    ----------
    xx, yy : (H, W) meshgrid arrays of input coordinates (``np.meshgrid``).
    patterns : (H*W, n_neurons) binary activation patterns at the grid points
        (hidden ReLU units, ``(pre_activation > 0)``), row-ordered like
        ``xx.ravel()``.
    points : (M, 2) optional data points to overlay.
    point_labels : (M,) optional labels colouring the overlaid points.
    decision : (H*W,) optional predicted class per grid point; drawn as a bold
        decision-boundary contour.
    fill : bool
        Fill regions with cycling colours (else just draw boundaries).
    boundary_color, boundary_alpha : str, float
        Colour/opacity of the region boundary lines.  Kept light (grey, faint)
        by default so the bold black decision boundary stays readable.
    seed : int
        Shuffles region→colour so neighbouring regions differ.

    Returns
    -------
    matplotlib Figure
    """
    H, W = xx.shape
    uniq, ids = np.unique(np.asarray(patterns), axis=0, return_inverse=True)
    region = ids.reshape(H, W)

    fig, ax = make_figure(ax, figsize, default_figsize=figsize)
    extent = (float(xx.min()), float(xx.max()), float(yy.min()), float(yy.max()))

    if fill:
        perm = np.random.default_rng(seed).permutation(len(uniq))
        ax.imshow(perm[region], origin="lower", extent=extent, aspect="auto",
                  cmap="tab20", interpolation="nearest", alpha=0.55)

    # Region boundaries = grid cells whose region differs from a neighbour.
    # Drawn light/faint so they don't compete with the bold decision boundary.
    from matplotlib.colors import ListedColormap
    boundary = np.zeros((H, W), dtype=bool)
    boundary[:, :-1] |= region[:, :-1] != region[:, 1:]
    boundary[:-1, :] |= region[:-1, :] != region[1:, :]
    ax.imshow(np.where(boundary, 1.0, np.nan), origin="lower", extent=extent,
              aspect="auto", cmap=ListedColormap([boundary_color]),
              alpha=boundary_alpha, interpolation="nearest")

    # Decision boundary: bold contour where the predicted class changes.
    if decision is not None:
        dec = np.asarray(decision).reshape(H, W).astype(float)
        levels = np.arange(np.floor(dec.min()) + 0.5, np.ceil(dec.max()))
        if len(levels):
            ax.contour(xx, yy, dec, levels=levels, colors="black", linewidths=2.2)

    if points is not None:
        points = np.asarray(points)
        ax.scatter(points[:, 0], points[:, 1],
                   c=("k" if point_labels is None else np.asarray(point_labels)),
                   cmap="coolwarm", s=8, edgecolors="white", linewidths=0.2, zorder=3)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(title or f"Polyhedral decomposition — {len(uniq)} linear regions")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# GraphBundle heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_graph_matrix(
    bundle: GraphBundle,
    reorder_by_label: bool = False,
    log: bool = False,
    cmap: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Heatmap of a GraphBundle's matrix.

    Parameters
    ----------
    bundle : GraphBundle
    reorder_by_label : bool
        If ``True`` and ``bundle.node_labels`` is present, sort rows/cols by
        class label.  Often makes block structure visible.
    log : bool
        Apply ``log1p`` to the matrix before plotting (helpful for
        heavy-tailed distance matrices).
    cmap : str or None
        Defaults to ``"viridis"`` for distance/similarity, ``"Greys"`` for
        adjacency.
    """
    M = bundle.matrix.astype(float).copy()
    if reorder_by_label:
        if bundle.node_labels is None:
            raise ValueError("reorder_by_label=True but bundle.node_labels is None.")
        order = np.argsort(bundle.node_labels)
        M = M[np.ix_(order, order)]
    if log:
        M = np.log1p(np.abs(M))

    if cmap is None:
        cmap = "Greys" if bundle.matrix_type == "adjacency" else "viridis"

    fig, ax = make_figure(ax, figsize, default_figsize=(6, 5))
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=bundle.matrix_type)
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")
    ax.set_title(title or f"GraphBundle matrix ({bundle.matrix_type})")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Node embedding
# ─────────────────────────────────────────────────────────────────────────────

def plot_node_embedding(
    bundle: GraphBundle,
    method: str = "mds",
    seed: int = 0,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    point_size: int = 25,
    alpha: float = 0.8,
) -> Figure:
    """2-D embedding of GraphBundle nodes, coloured by ``node_labels``.

    Parameters
    ----------
    bundle : GraphBundle
    method : str
        ``"mds"`` (metric MDS on the distance matrix) or ``"tsne"``.
    seed : int
    """
    matrix_type = bundle.matrix_type
    if matrix_type == "distance":
        D = bundle.matrix
    elif matrix_type == "similarity":
        D = float(bundle.matrix.max()) - bundle.matrix
        np.fill_diagonal(D, 0.0)
    elif matrix_type == "adjacency":
        D = 1.0 - bundle.matrix.astype(float)
        np.fill_diagonal(D, 0.0)
    else:
        raise ValueError(f"Unknown matrix_type '{matrix_type}'.")

    if method == "mds":
        from sklearn.manifold import MDS
        emb = MDS(
            n_components=2, dissimilarity="precomputed",
            random_state=seed, normalized_stress="auto",
        ).fit_transform(D)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        emb = TSNE(
            n_components=2, metric="precomputed",
            init="random", random_state=seed,
            perplexity=min(30, max(5, bundle.n_nodes // 3)),
        ).fit_transform(D)
    else:
        raise ValueError(f"method must be 'mds' or 'tsne', got '{method}'.")

    fig, ax = make_figure(ax, figsize, default_figsize=(6, 5))
    if bundle.node_labels is not None:
        labels = bundle.node_labels
        for c in np.unique(labels):
            mask = labels == c
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       s=point_size, alpha=alpha, label=f"class {c}")
        ax.legend(fontsize=8)
    else:
        ax.scatter(emb[:, 0], emb[:, 1], s=point_size, alpha=alpha)

    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(title or f"Node embedding ({method.upper()})")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsic-dimension with uncertainty bands (bootstrap)
# ─────────────────────────────────────────────────────────────────────────────

def plot_id_with_uncertainty(
    activations: list[np.ndarray],
    method: str = "global",
    n_bootstraps: int = 20,
    subsample_frac: float = 0.8,
    layer_labels: list[str] | None = None,
    seed: int = 0,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    **method_kwargs,
) -> Figure:
    """Per-layer ID with mean +/- 1 SD shading from bootstrap subsampling.

    For each layer, draws ``n_bootstraps`` random subsamples of size
    ``subsample_frac * N`` from the activation array and runs the chosen
    2NN estimator on each.  Plots the mean across bootstraps as a line and
    +/- 1 SD as a shaded band.

    Parameters
    ----------
    activations : list of (N, D) ndarrays
        One activation array per layer (raw activations, not distance
        matrices).
    method : str
        ``"global"`` or ``"local"``.
    n_bootstraps : int
    subsample_frac : float
        Fraction of points to keep in each bootstrap draw.
    layer_labels : list[str] or None
    seed : int
    **method_kwargs
        Forwarded to the underlying 2NN estimator.
    """
    from tanc.topo_tools.dimension_tool import (
        estimate_id_global, estimate_id_local,
    )
    from sklearn.metrics import pairwise_distances

    estimator = estimate_id_global if method == "global" else estimate_id_local
    if layer_labels is None:
        layer_labels = [f"Layer {i}" for i in range(len(activations))]

    rng = np.random.default_rng(seed)
    means: list[float] = []
    stds: list[float] = []
    for arr in activations:
        a = np.asarray(arr)
        N = a.shape[0]
        n_sub = max(3, int(round(subsample_frac * N)))
        samples: list[float] = []
        for _ in range(n_bootstraps):
            idx = rng.choice(N, size=n_sub, replace=False)
            sub = a[idx]
            D = pairwise_distances(sub, metric="euclidean")
            try:
                res = estimator(D, **method_kwargs)
                samples.append(float(
                    res.get("id_estimate", res.get("id_mean", float("nan")))
                ))
            except Exception:
                samples.append(float("nan"))
        samples = np.array(samples)
        valid = samples[np.isfinite(samples)]
        means.append(float(valid.mean()) if valid.size else float("nan"))
        stds.append(float(valid.std()) if valid.size else float("nan"))

    means_arr = np.array(means)
    stds_arr = np.array(stds)
    x = np.arange(len(layer_labels))

    fig, ax = make_figure(ax, figsize, default_figsize=(7, 4))
    ax.plot(x, means_arr, marker="o", linewidth=2, label="mean")
    ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr,
                    alpha=0.3, label="+/- 1 SD")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Intrinsic dim ({method} 2NN)")
    ax.legend()
    ax.set_title(title or f"ID per layer with bootstrap uncertainty "
                          f"(n={n_bootstraps})")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# QQ plot of log(log mu) ratios — tail-behaviour diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def plot_id_qq(
    data,
    dist: str = "norm",
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """QQ plot of the global-2NN log(log mu) values against a theoretical distribution.

    Useful as a tail-behaviour diagnostic: heavy upper tail in the
    empirical quantiles inflates the mean log-log ratio and therefore the
    ID estimate.

    Parameters
    ----------
    data
        One of:

        * A ``dict`` returned by ``estimate_id_global`` — its
          ``loglog_ratios`` field is used.
        * A ``(N, N)`` distance matrix — ratios computed via
          ``compute_2nn_ratios`` then transformed.
        * A 1-D array of ``log(log mu)`` values (used directly).
    dist : str
        ``scipy.stats`` distribution name.  ``"norm"`` (default) is the
        usual sanity check; ``"gumbel_r"`` matches the asymptotic
        distribution of the maximum log-ratio.
    """
    from scipy import stats

    arr = None
    if isinstance(data, dict):
        arr = np.asarray(data.get("loglog_ratios", []), dtype=float)
    else:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            from tanc.topo_tools.dimension_tool import compute_2nn_ratios
            ratios = compute_2nn_ratios(arr)
            ratios = ratios[ratios > 1.0]
            arr = np.log(np.log(ratios)) if ratios.size else np.array([])

    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        raise ValueError(
            f"Need >= 3 finite log(log mu) values for a QQ plot, got {arr.size}."
        )

    fig, ax = make_figure(ax, figsize, default_figsize=(6, 5))
    stats.probplot(arr, dist=dist, plot=ax)
    ax.set_title(title or f"QQ plot of log(log mu) vs {dist}")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Topological-uncertainty score distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_tu_roc(
    scores: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """ROC curve treating *misclassification* as the positive class.

    Companion to :func:`plot_tu_score_distribution`.  A useful diagnostic
    for the Lacombe et al. (2021) TU score: a good TU score should be
    higher for misclassified inputs, so the misclassification ROC curve
    should rise above the diagonal.

    Parameters
    ----------
    scores : (N,) ndarray
        Topological-uncertainty scores.
    y_true, y_pred : (N,) integer arrays
        Used to derive the binary "misclassified" label.
    """
    from sklearn.metrics import roc_curve, auc

    scores = np.asarray(scores, dtype=float).ravel()
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if not (scores.shape == y_true.shape == y_pred.shape):
        raise ValueError(
            "scores, y_true, y_pred must have the same shape; got "
            f"{scores.shape}, {y_true.shape}, {y_pred.shape}."
        )

    misclassified = (y_true != y_pred).astype(int)
    if misclassified.sum() == 0 or misclassified.sum() == len(misclassified):
        raise ValueError(
            "ROC curve is undefined: y_true == y_pred everywhere (or never)."
        )

    fpr, tpr, _ = roc_curve(misclassified, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = make_figure(ax, figsize, default_figsize=(6, 5))
    ax.plot(fpr, tpr, color="tab:blue", linewidth=2,
            label=f"TU (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--",
            linewidth=1, label="chance")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend()
    ax.set_title(title or "TU score — misclassification ROC")
    fig.tight_layout()
    return fig


def h0_signal_pathways(
    edge_matrices: list[np.ndarray],
    n_pathways: int = 6,
):
    """Extract the H0 representative *signal pathways* of a layered NN graph.

    Gebhart et al. (2019) interpret long-lifetime ``H0`` generators of the
    activation graph (edge weight ``|w·h|``) as the network's prominent,
    *distinct* signal pathways.  For a graph, ``H0`` persistence is exactly
    single-linkage clustering: the merge tree is the **maximum spanning
    forest**, and the ``n_pathways - 1`` weakest tree edges are the death
    simplices of the longest-lifetime ``H0`` bars.  Removing them splits the
    network into the ``n_pathways`` most persistent clusters — the pathways.

    Parameters
    ----------
    edge_matrices : list of (N_in, N_out) non-negative edge-weight matrices,
        one per consecutive layer (use ``|w·h|`` for Gebhart).
    n_pathways : int
        How many distinct pathways (top H0 clusters) to extract.

    Returns
    -------
    dict with keys
        ``"cluster_of"``      : (N_nodes,) int cluster id per node,
        ``"tree_edges"``      : list of (layer, i, j, weight, cluster) — the
                                merge-tree edges *within* a pathway,
        ``"death_edges"``     : list of (layer, i, j, weight) — the cut edges
                                that separate pathways (the H0 deaths),
                                weakest first = longest lifetime,
        ``"layer_sizes"``     : list[int].
    """
    Ws = [np.abs(np.asarray(W, dtype=float)) for W in edge_matrices]
    sizes = [Ws[0].shape[0]] + [W.shape[1] for W in Ws]
    offsets = [sum(sizes[:l]) for l in range(len(sizes))]
    N = sum(sizes)

    def gid(l, i):
        return offsets[l] + i

    # All edges, strongest first (superlevel filtration on similarity).
    edges = [(float(W[i, j]), l, i, j)
             for l, W in enumerate(Ws)
             for i in range(W.shape[0]) for j in range(W.shape[1])]
    edges.sort(key=lambda e: -e[0])

    parent = list(range(N))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    # Maximum spanning forest = the H0 merge tree; its edges are the H0 reps.
    tree: list[tuple] = []
    for w, l, i, j in edges:
        u, v = find(gid(l, i)), find(gid(l + 1, j))
        if u != v:
            parent[u] = v
            tree.append((l, i, j, w))

    # The (n_pathways-1) weakest tree edges are the deaths of the longest bars.
    tree_by_w = sorted(tree, key=lambda e: e[3])          # ascending weight
    n_cut = max(0, min(n_pathways - 1, len(tree_by_w)))
    death = tree_by_w[:n_cut]
    death_set = {(l, i, j) for l, i, j, _w in death}

    # Cluster assignment = connected components after removing the cut edges.
    parent2 = list(range(N))

    def find2(x):
        root = x
        while parent2[root] != root:
            root = parent2[root]
        while parent2[x] != root:
            parent2[x], x = root, parent2[x]
        return root

    for l, i, j, _w in tree:
        if (l, i, j) in death_set:
            continue
        parent2[find2(gid(l, i))] = find2(gid(l + 1, j))

    roots: dict[int, int] = {}
    cluster_of = np.empty(N, dtype=int)
    for node in range(N):
        r = find2(node)
        if r not in roots:
            roots[r] = len(roots)
        cluster_of[node] = roots[r]

    tree_edges = [(l, i, j, w, int(cluster_of[gid(l, i)]))
                  for l, i, j, w in tree if (l, i, j) not in death_set]
    death_edges = [(l, i, j, w) for l, i, j, w in death]   # weakest first

    return {
        "cluster_of": cluster_of,
        "tree_edges": tree_edges,
        "death_edges": death_edges,
        "layer_sizes": sizes,
    }


def plot_pathways_on_network(
    weight_matrices: list[np.ndarray],
    mode: str = "magnitude",
    top_frac: float = 0.05,
    n_pathways: int = 6,
    annotate_ph: "PersistenceResult | None" = None,
    cmap: str = "viridis",
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Layered-network diagram with prominent pathways highlighted.

    Parameters
    ----------
    weight_matrices : list of (N_in, N_out) ndarrays
        Per-layer edge weights.  For ``mode="h0"`` pass the **activation-
        weighted** edges ``|w·h|`` (Gebhart's graph), not the raw weights.
    mode : str
        ``"h0"`` — highlight the network's **H0 representative signal
        pathways** (Gebhart et al. 2019): the maximum-spanning-tree edges of
        the activation graph, coloured by which of the top ``n_pathways``
        clusters (persistent H0 components) they belong to, with the H0
        *death* edges that separate the pathways drawn dashed.  This is the
        faithful topological pathway plot.

        ``"magnitude"`` (default) — a quick visual proxy: the top
        ``top_frac`` fraction of edges per layer by absolute weight, coloured
        by magnitude.  *Not* topological; kept for backward compatibility.
    top_frac : float
        Edge fraction to highlight in ``mode="magnitude"``.
    n_pathways : int
        Number of H0 pathways (top clusters) in ``mode="h0"``.
    annotate_ph : PersistenceResult or None
        If given, the top-5 H0 lifetimes are listed in a sidebar (context).
    """
    Ws = [np.asarray(W) for W in weight_matrices]
    if not Ws:
        raise ValueError("weight_matrices is empty.")
    if mode not in ("h0", "magnitude"):
        raise ValueError(f"mode must be 'h0' or 'magnitude', got {mode!r}.")
    n_layers = len(Ws) + 1
    layer_sizes = [Ws[0].shape[0]] + [W.shape[1] for W in Ws]
    if figsize is None:
        figsize = (max(6, 1.7 * n_layers), 5.5)
    fig, ax = make_figure(ax, figsize, default_figsize=figsize)

    # Node positions: one column per layer.
    positions: list[np.ndarray] = []
    for li, n in enumerate(layer_sizes):
        y = np.linspace(0.05, 0.95, n) if n > 1 else np.array([0.5])
        x = np.full(n, li / max(n_layers - 1, 1))
        positions.append(np.column_stack([x, y]))
    total_edges = sum(W.size for W in Ws)

    if mode == "magnitude":
        cmap_obj = plt.get_cmap(cmap)
        for li, W in enumerate(Ws):
            srcs, dsts = positions[li], positions[li + 1]
            absW = np.abs(W)
            thresh = np.quantile(absW, 1 - top_frac) if absW.size else 0.0
            vmax = float(absW.max()) or 1.0
            for i, j in np.argwhere(absW < thresh):
                ax.plot([srcs[i, 0], dsts[j, 0]], [srcs[i, 1], dsts[j, 1]],
                        color="grey", linewidth=0.4, alpha=0.15, zorder=1)
            for i, j in np.argwhere(absW >= thresh):
                ax.plot([srcs[i, 0], dsts[j, 0]], [srcs[i, 1], dsts[j, 1]],
                        color=cmap_obj(absW[i, j] / vmax),
                        linewidth=1.3, alpha=0.85, zorder=2)
        default_title = f"Network pathways — top {int(top_frac * 100)}% edges by |w| (magnitude proxy)"

    else:  # mode == "h0"
        paths = h0_signal_pathways(Ws, n_pathways=n_pathways)
        qual = plt.get_cmap("tab10")
        # Faint background (only when the graph is small enough to be legible).
        if total_edges <= 4000:
            for li, W in enumerate(Ws):
                srcs, dsts = positions[li], positions[li + 1]
                for i, j in np.argwhere(np.abs(W) > 0):
                    ax.plot([srcs[i, 0], dsts[j, 0]], [srcs[i, 1], dsts[j, 1]],
                            color="grey", linewidth=0.3, alpha=0.08, zorder=1)
        # Pathway (merge-tree) edges, coloured by cluster.
        wmax = max((w for *_x, w, _c in paths["tree_edges"]), default=1.0) or 1.0
        for l, i, j, w, c in paths["tree_edges"]:
            s, d = positions[l][i], positions[l + 1][j]
            ax.plot([s[0], d[0]], [s[1], d[1]], color=qual(c % 10),
                    linewidth=0.6 + 2.4 * w / wmax, alpha=0.9, zorder=2)
        # H0 death edges that separate pathways: dashed.
        for l, i, j, w in paths["death_edges"]:
            s, d = positions[l][i], positions[l + 1][j]
            ax.plot([s[0], d[0]], [s[1], d[1]], color="black",
                    linewidth=1.0, alpha=0.6, linestyle="--", zorder=2)
        default_title = (f"Top {n_pathways} H0 signal pathways "
                         f"(|w·h| graph; dashed = H0 death edges)")
        if annotate_ph is None and paths["death_edges"]:
            lines = ["H0 deaths (separations)"]
            for k, (_l, _i, _j, w) in enumerate(paths["death_edges"]):
                lines.append(f"  #{k+1}: {w:.4f}")
            ax.text(1.02, 0.5, "\n".join(lines), transform=ax.transAxes,
                    va="center", ha="left", fontsize=8, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    # Nodes (smaller when a layer is large).
    for p in positions:
        s = 42 if len(p) <= 80 else max(6, 1200 // len(p))
        ax.scatter(p[:, 0], p[:, 1], s=s, color="white", edgecolors="black",
                   linewidths=0.6, zorder=3)
    for li in range(n_layers):
        ax.text(li / max(n_layers - 1, 1), -0.05, f"L{li}",
                ha="center", va="top", fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.05)
    ax.axis("off")
    ax.set_title(title or default_title)

    # Optional PH sidebar (context).
    if annotate_ph is not None:
        dgm0 = annotate_ph.diagrams.get(0, np.empty((0, 2)))
        if dgm0.shape[0] > 0:
            lt = dgm0[:, 1] - dgm0[:, 0]
            lt = lt[np.isfinite(lt)]
            top = np.argsort(-lt)[:5]
            lines = [f"Top {len(top)} H0 lifetimes"] + [
                f"  #{k+1}: {lt[idx]:.4f}" for k, idx in enumerate(top)]
            ax.text(1.02, 0.5, "\n".join(lines), transform=ax.transAxes,
                    va="center", ha="left", fontsize=8, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    return fig


def plot_tu_score_distribution(
    scores: np.ndarray,
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    bins: int = 30,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Histogram of TopologicalUncertainty scores.

    If both ``y_true`` and ``y_pred`` are given, the histogram is split
    into correctly and incorrectly classified samples — a quick visual
    check of whether high TU correlates with misclassification.

    Parameters
    ----------
    scores : (N,) array of TU scores.
    y_true, y_pred : optional integer label arrays.
    bins : int
    """
    scores = np.asarray(scores, dtype=float).ravel()
    fig, ax = make_figure(ax, figsize, default_figsize=(7, 4))

    if y_true is not None and y_pred is not None:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if scores.shape[0] != y_true.shape[0] or scores.shape[0] != y_pred.shape[0]:
            raise ValueError("scores, y_true, y_pred must have the same length.")
        correct = y_true == y_pred
        edges = np.histogram_bin_edges(scores, bins=bins)
        ax.hist(scores[correct], bins=edges, alpha=0.6,
                color="tab:green", label=f"correct (n={correct.sum()})")
        ax.hist(scores[~correct], bins=edges, alpha=0.6,
                color="tab:red", label=f"misclassified (n={(~correct).sum()})")
        ax.legend()
    else:
        ax.hist(scores, bins=bins, edgecolor="black", alpha=0.75)

    ax.set_xlabel("Topological uncertainty score")
    ax.set_ylabel("Count")
    ax.set_title(title or "TU score distribution")
    fig.tight_layout()
    return fig