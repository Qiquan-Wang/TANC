"""sweep_plots.py — reading a Mapper sweep, and colouring the graphs it produced.

Two jobs.

**Colouring.** A Mapper node is a set of points, so any per-point quantity can
colour it.  The obvious choice — how many source models a node contains — is
confounded with node size: a node holding 400 points from a population of 100
networks contains almost all of them by chance.  :func:`node_colour` therefore
offers an ``observed / expected`` normalisation alongside the raw value, and the
expectation is computed against the null that points are assigned to nodes
independently of their source.

**Diagnostics.** The plots here are built around one question: *did Mapper find
structure, or reproduce its cover?*  A graph and its nerve are drawn together
rather than separately, because the interesting quantity is the difference.
They deliberately show the negative result clearly — a flat ``b1_excess`` across
a whole parameter range is a finding, and should look like one.

Recolouring is free: it reads the node membership saved with each graph, so a
finished sweep can be re-examined without recomputing any Mapper.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

__all__ = [
    "COLOURINGS",
    "node_colour",
    "plot_mapper_sweep_graph",
    "plot_graph_panel",
    "plot_stability_heatmap",
    "plot_cover_degeneracy",
    "plot_node_size_distribution",
    "plot_filter_sweep",
    "plot_population_summary",
]

_EPS = 1e-12

#: Colourings that need only the graph, versus those that also need the cloud.
COLOURINGS: dict[str, str] = {
    "lens":      "mean lens position of each node (needs only the graph)",
    "size":      "number of points in the node (needs only the graph)",
    "degree":    "node degree in the Mapper graph (needs only the graph)",
    "label":     "dominant label among the node's points (needs labels)",
    "diversity": "distinct labels present, e.g. how many models a node mixes",
    "density":   "mean codensity — is the node in the core or the periphery?",
    "scalar":    "mean of a supplied per-point value",
}


def node_colour(
    graph,
    by: Any = "lens",
    *,
    labels: np.ndarray | None = None,
    values: np.ndarray | None = None,
    cloud: np.ndarray | None = None,
    normalise: str = "raw",
) -> np.ndarray:
    """One colour value per Mapper node.

    Parameters
    ----------
    graph : MapperGraph
        As returned by :func:`~tanc.topo_tools.mapper_sweep.mapper_graph`
        or :func:`~tanc.topo_tools.mapper_sweep.load_graph`.
    by : str or callable
        A key of :data:`COLOURINGS`, or ``f(member_indices) -> float``.
    labels : (N,) ndarray, optional
        Per-point labels, for ``"label"`` and ``"diversity"``.  For a population
        this is usually
        :meth:`~tanc.model_extractor.population.TrainedPopulation.member_index`.
    values : (N,) ndarray, optional
        Per-point scalar, for ``"scalar"``.
    cloud : (N, D) ndarray, optional
        The point cloud, needed for ``"density"``.
    normalise : {"raw", "vs_expected", "zscore"}
        ``"vs_expected"`` divides by what the value would be if points were
        assigned to nodes at random, which removes the node-size confound.  It
        is defined for ``"diversity"``; for other colourings it falls back to
        ``"raw"`` with no error, since there is no meaningful null.

    Returns
    -------
    (n_nodes,) ndarray

    Notes
    -----
    For ``"diversity"`` the expectation under independent assignment is
    ``M (1 - (1 - 1/M)**n)`` for a node of ``n`` points drawn from ``M``
    sources.  A ratio near 1 is the *pass*: a group cannot hold more sources
    than it has members, so exceeding the null is impossible, and falling well
    below it means a few sources contributed most of the node.
    """
    nodes = graph.nodes
    if not nodes:
        return np.zeros(0)

    if callable(by):
        out = np.array([float(by(idx)) for idx in nodes])
        return _post_normalise(out, normalise, None)

    if by not in COLOURINGS:
        raise ValueError(
            f"Unknown colouring {by!r}. Known:\n  "
            + "\n  ".join(f"{k:<10} {v}" for k, v in COLOURINGS.items())
            + "\n  or pass a callable f(member_indices) -> float."
        )

    if by == "size":
        out = graph.node_sizes.astype(float)
    elif by == "degree":
        deg = np.zeros(len(nodes))
        if graph.n_edges:
            for u, v in graph.edges:
                deg[u] += 1
                deg[v] += 1
        out = deg
    elif by == "lens":
        if graph.lens is None:
            raise ValueError("Colouring by 'lens' needs a graph built with lens values.")
        out = graph.node_lens().mean(axis=1)
    elif by in ("label", "diversity"):
        if labels is None:
            raise ValueError(f"Colouring by {by!r} needs `labels` (one per point).")
        lab = np.asarray(labels)
        if by == "label":
            out = np.array([_dominant(lab[idx]) for idx in nodes], dtype=float)
        else:
            out = np.array([len(np.unique(lab[idx])) for idx in nodes], dtype=float)
            if normalise == "vs_expected":
                M = len(np.unique(lab))
                exp = np.array([_expected_distinct(len(idx), M) for idx in nodes])
                return out / (exp + _EPS)
    elif by == "density":
        if cloud is None:
            raise ValueError("Colouring by 'density' needs the point `cloud`.")
        from tanc.topo_tools.mapper_sweep import codensity
        rho = codensity(np.asarray(cloud))
        out = np.array([rho[idx].mean() for idx in nodes])
    elif by == "scalar":
        if values is None:
            raise ValueError("Colouring by 'scalar' needs `values` (one per point).")
        v = np.asarray(values, dtype=float)
        out = np.array([v[idx].mean() for idx in nodes])
    else:                                             # pragma: no cover
        raise AssertionError(f"unhandled colouring {by!r}")

    return _post_normalise(out, normalise, None)


def _dominant(v: np.ndarray) -> float:
    vals, counts = np.unique(v, return_counts=True)
    return float(vals[int(np.argmax(counts))])


def _expected_distinct(n: int, M: int) -> float:
    """Distinct sources expected in a node of *n* points drawn from *M* sources."""
    if M <= 0 or n <= 0:
        return 1.0
    return M * (1.0 - (1.0 - 1.0 / M) ** n)


def _post_normalise(out: np.ndarray, how: str, _null) -> np.ndarray:
    if how in ("raw", "vs_expected"):      # vs_expected handled where a null exists
        return out
    if how == "zscore":
        return (out - out.mean()) / (out.std() + _EPS)
    raise ValueError(f"normalise must be 'raw', 'vs_expected' or 'zscore'; got {how!r}.")


# ─────────────────────────────────────────────────────────────────────────────
# Drawing one graph
# ─────────────────────────────────────────────────────────────────────────────

def plot_mapper_sweep_graph(
    graph,
    *,
    colour_by: Any = "lens",
    labels: np.ndarray | None = None,
    values: np.ndarray | None = None,
    cloud: np.ndarray | None = None,
    normalise: str = "raw",
    layout: str = "lens",
    max_nodes: int = 4000,
    ax=None,
    title: str | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (7.0, 6.0),
) -> Figure:
    """Draw a Mapper graph, sized by node membership and coloured as requested.

    Parameters
    ----------
    graph : MapperGraph
    colour_by, labels, values, cloud, normalise
        Passed to :func:`node_colour`.
    layout : {"lens", "spring"}
        ``"lens"`` places nodes at their mean lens position, which is
        deterministic, instant, and interpretable — the axes mean something.
        ``"spring"`` is the usual force-directed layout, and is slow above a
        few thousand nodes.
    max_nodes : int
        Refuse to draw beyond this, rather than hanging.  A graph this large is
        usually a shattered one, which the sweep table diagnoses better than any
        picture.
    ax, title, cmap, figsize

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = graph.n_nodes
    if n == 0:
        raise ValueError("This graph has no nodes — nothing to draw.")
    if n > max_nodes:
        raise ValueError(
            f"This graph has {n} nodes, above max_nodes={max_nodes}. Graphs this "
            f"large are usually shattered rather than informative — check "
            f"cpc_mean and node_median in the sweep table. Raise max_nodes to "
            f"override."
        )

    colours = node_colour(graph, colour_by, labels=labels, values=values,
                          cloud=cloud, normalise=normalise)
    sizes = graph.node_sizes.astype(float)
    pos = _positions(graph, layout)

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)

    if graph.n_edges:
        from matplotlib.collections import LineCollection
        seg = np.array([[pos[u], pos[v]] for u, v in graph.edges])
        ax.add_collection(LineCollection(seg, colors="0.75", linewidths=0.5, zorder=1))

    marker = 20.0 * sizes / (sizes.max() + _EPS) + 4.0
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=colours, s=marker, cmap=cmap,
                    zorder=2, edgecolors="none")
    label = colour_by if isinstance(colour_by, str) else "custom"
    if normalise != "raw":
        label = f"{label} ({normalise})"
    fig.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.04)

    ax.set_title(title or f"{n} nodes, {graph.n_edges} edges")
    if layout == "lens":
        ax.set_xlabel("lens 1")
        ax.set_ylabel("lens 2" if _lens_dim(graph) > 1 else "node index")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.autoscale_view()
    fig.tight_layout()
    return fig


def plot_graph_panel(
    graphs,
    *,
    titles: Sequence[str] | None = None,
    ncol: int = 6,
    size: float = 2.5,
    colour_by: Any = "size",
    layout: str = "spring",
    labels: Sequence[np.ndarray] | np.ndarray | None = None,
    normalise: str = "raw",
    cmap: str = "viridis",
    suptitle: str | None = None,
    title_size: float = 7.0,
    max_nodes: int = 4000,
) -> Figure:
    """Draw many Mapper graphs as one contact sheet.

    A sweep produces one graph per configuration, and the question a sweep exists
    to answer — does this structure survive a *range* of parameters, or is it an
    accident of one setting? — is answered by looking at them together.  A single
    graph, however carefully chosen, cannot answer it.

    Per-graph colourbars are suppressed: at contact-sheet scale they consume more
    of each tile than the graph does, and relative node size is already legible
    from the marker areas.  Use :func:`plot_mapper_sweep_graph` for a full-size
    figure with its scale bar.

    Parameters
    ----------
    graphs : sequence of MapperGraph
        The graphs to draw, in the order they should appear.  Load them from a
        store with ``result.graph(row)`` — nothing is recomputed.
    titles : sequence of str, optional
        One caption per graph.  Keep them short; they are drawn at
        ``title_size`` points.  A newline splits a caption over two lines.
    ncol : int
        Tiles per row.  The number of rows follows from the count.
    size : float
        Side length of one tile in inches.
    colour_by : str or callable
        Any key of :data:`COLOURINGS`, or a callable taking node indices.
        ``"size"`` is the default because it needs no extra input and reads
        clearly when small.
    layout : str
        ``"spring"`` (default here, unlike the single-graph plotter) or
        ``"lens"``.  A lens layout collapses a 1-D lens to a line, which is
        hard to read at tile scale.
    labels : ndarray or sequence of ndarray, optional
        Needed by the ``"label"`` and ``"diversity"`` colourings.  Pass one
        array to use for every graph, or one per graph.
    normalise : str
        Passed to :func:`node_colour` for the diversity colouring.
    suptitle : str, optional
        Heading for the whole sheet — typically the cloud name.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``graphs`` is empty, or ``titles``/``labels`` are the wrong length.
    """
    graphs = list(graphs)
    if not graphs:
        raise ValueError(
            "plot_graph_panel needs at least one graph. If these came from a "
            "sweep, check that the rows were filtered to status == 'ok'."
        )
    if titles is not None and len(titles) != len(graphs):
        raise ValueError(
            f"titles has {len(titles)} entries for {len(graphs)} graphs."
        )
    if labels is not None and isinstance(labels, (list, tuple)):
        if len(labels) != len(graphs):
            raise ValueError(
                f"labels has {len(labels)} entries for {len(graphs)} graphs. "
                "Pass a single array to reuse it for every graph."
            )
        per_graph_labels = list(labels)
    else:
        per_graph_labels = [labels] * len(graphs)

    ncol = max(1, int(ncol))
    nrow = int(np.ceil(len(graphs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * size, nrow * size),
                             squeeze=False)
    flat = np.ravel(axes)
    for ax in flat:
        ax.axis("off")

    # Anything the single-graph plotter adds beyond these tiles is a colourbar.
    tile_ids = {id(a) for a in flat}
    for i, (ax, g) in enumerate(zip(flat, graphs)):
        plot_mapper_sweep_graph(
            g, colour_by=colour_by, labels=per_graph_labels[i],
            normalise=normalise, layout=layout, ax=ax, cmap=cmap,
            max_nodes=max_nodes,
        )
        for extra in [a for a in fig.axes if id(a) not in tile_ids]:
            extra.remove()
        if titles is not None:
            ax.set_title(titles[i], fontsize=title_size)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, y=1.0, fontsize=11)
    fig.tight_layout()
    return fig


def _lens_dim(graph) -> int:
    if graph.lens is None:
        return 0
    L = graph.lens
    return 1 if L.ndim == 1 else L.shape[1]


def _positions(graph, layout: str) -> np.ndarray:
    if layout == "lens":
        if graph.lens is None:
            raise ValueError("layout='lens' needs a graph built with lens values.")
        L = graph.node_lens()
        if L.shape[1] == 1:
            # A one-dimensional lens gives no second axis; spread nodes
            # vertically by index so overlapping nodes stay distinguishable.
            return np.column_stack([L[:, 0], np.arange(len(L), dtype=float)])
        return L[:, :2]
    if layout == "spring":
        import networkx as nx
        g = graph.to_networkx()
        p = nx.spring_layout(g, seed=0)
        return np.array([p[i] for i in range(graph.n_nodes)])
    raise ValueError(f"layout must be 'lens' or 'spring'; got {layout!r}.")


# ─────────────────────────────────────────────────────────────────────────────
# Reading a sweep
# ─────────────────────────────────────────────────────────────────────────────

def _rows_to_arrays(rows: Sequence[dict], x: str, y: str, z: str):
    """Pivot sweep rows into a grid of *z* over *x* and *y*."""
    ok = [r for r in rows if r.get("status") == "ok" and z in r]
    if not ok:
        raise ValueError(
            f"No successful rows carry {z!r}. Available measures: "
            f"{sorted(set().union(*(set(r) for r in rows))) if rows else '[]'}"
        )
    xs = sorted({_num(r[x]) for r in ok})
    ys = sorted({_num(r[y]) for r in ok})
    M = np.full((len(ys), len(xs)), np.nan)
    for r in ok:
        M[ys.index(_num(r[y])), xs.index(_num(r[x]))] = float(r[z])
    return np.array(xs), np.array(ys), M


def _num(v):
    """Sweep rows come back as strings when they came from the config record."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def plot_stability_heatmap(
    rows: Sequence[dict],
    measure: str = "b1_excess",
    *,
    x: str = "n_intervals",
    y: str = "overlap",
    ax=None,
    cmap: str = "RdBu_r",
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> Figure:
    """A measure across two cover parameters — the plateau check.

    A structure that is real shows a *region* of stable values, not a single
    bright cell.  Read this before reading any individual graph: an isolated
    extreme is the signature of a parameter accident.

    Parameters
    ----------
    rows : sequence of dict
        From ``store.rows()``.
    measure : str
        Any recorded measure; ``b1_excess`` by default, since ``b1`` alone
        cannot distinguish structure from the cover's own topology.
    x, y : str
        Axis parameters.
    """
    xs, ys, M = _rows_to_arrays(rows, x, y, measure)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)

    lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    kw = dict(vmin=-lim, vmax=lim) if measure.endswith("excess") else {}
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap, **kw)
    ax.set_xticks(range(len(xs)), [f"{v:g}" for v in xs])
    ax.set_yticks(range(len(ys)), [f"{v:g}" for v in ys])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        fontsize=7, color="0.2")
    fig.colorbar(im, ax=ax, label=measure, fraction=0.046, pad=0.04)
    ax.set_title(title or f"{measure} across the cover")
    fig.tight_layout()
    return fig


def plot_cover_degeneracy(
    rows: Sequence[dict],
    *,
    x: str = "n_intervals",
    group: str = "overlap",
    ax=None,
    title: str | None = None,
    figsize: tuple[float, float] = (11.0, 4.2),
) -> Figure:
    """Where Mapper stops finding structure and starts returning its cover.

    Three panels, all reading the same failure from different sides:

    ``node_ratio``
        Nodes per non-empty cover cell.  At 1 the graph *is* the nerve.
    ``cpc_frac_1``
        Fraction of cells yielding exactly one cluster.  At 1, likewise.
    ``node_median``
        Median node size.  Falling to 1 means the opposite failure — the graph
        has shattered, and its large ``b1`` is ``E - V + b0`` for a dust of
        singletons rather than any topology.

    The two failures sit at opposite ends of the resolution axis and a single
    criterion cannot catch both, which is why all three are drawn together.
    """
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        raise ValueError("No successful rows to plot.")
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panels = [
        ("node_ratio", "nodes per non-empty cell", 1.0, "= 1: graph is its nerve"),
        ("cpc_frac_1", "cells giving exactly 1 cluster", 1.0, "= 1: lattice"),
        ("node_median", "median node size", 1.0, "= 1: shattered"),
    ]
    groups = sorted({_num(r[group]) for r in ok if group in r})
    for ax_, (meas, ylab, ref, note) in zip(axes, panels):
        for g in groups:
            sel = [r for r in ok if _num(r.get(group)) == g and meas in r]
            if not sel:
                continue
            sel.sort(key=lambda r: _num(r[x]))
            ax_.plot([_num(r[x]) for r in sel], [float(r[meas]) for r in sel],
                     marker="o", ms=3.5, lw=1.2, label=f"{group}={g:g}")
        ax_.axhline(ref, color="crimson", ls="--", lw=1.0)
        ax_.annotate(note, xy=(0.98, ref), xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=7, color="crimson")
        ax_.set_xlabel(x)
        ax_.set_ylabel(ylab)
        ax_.set_yscale("log" if meas != "cpc_frac_1" else "linear")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(title or "Is the graph telling us about the data or the cover?")
    fig.tight_layout()
    return fig


def plot_node_size_distribution(
    rows: Sequence[dict],
    *,
    ax=None,
    title: str | None = None,
    figsize: tuple[float, float] = (6.5, 5.0),
) -> Figure:
    """Median against maximum node size, with the illegible region marked.

    Mean node size hides the failure that matters: a handful of huge nodes can
    carry the mean while the median sits at 1 and the graph is dust.  Plotting
    median against maximum separates the two immediately.
    """
    ok = [r for r in rows
          if r.get("status") == "ok" and "node_median" in r and "node_max" in r]
    if not ok:
        raise ValueError("No successful rows carry node_median and node_max.")
    med = np.array([float(r["node_median"]) for r in ok])
    mx = np.array([float(r["node_max"]) for r in ok])
    b1x = np.array([float(r.get("b1_excess", np.nan)) for r in ok])

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
    sc = ax.scatter(med, mx, c=b1x, cmap="RdBu_r", s=26, edgecolors="0.3", linewidths=0.3,
                    vmin=-np.nanmax(np.abs(b1x)) if np.isfinite(b1x).any() else -1,
                    vmax=np.nanmax(np.abs(b1x)) if np.isfinite(b1x).any() else 1)
    ax.axvspan(0.5, 2.0, color="crimson", alpha=0.10)
    ax.annotate("median ≤ 2:\nnodes are dust", xy=(0.9, 0.05), xycoords="axes fraction",
                ha="right", fontsize=8, color="crimson")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("median node size")
    ax.set_ylabel("largest node")
    fig.colorbar(sc, ax=ax, label="b1_excess", fraction=0.046, pad=0.04)
    ax.set_title(title or "Node sizes — is this graph legible?")
    fig.tight_layout()
    return fig


def plot_filter_sweep(
    rows: Sequence[dict],
    *,
    x: str = "preprocess",
    measures: Sequence[str] = ("b1", "nerve_b1", "b1_excess"),
    ax=None,
    title: str | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> Figure:
    """Topology against filter strength, with the nerve drawn alongside.

    Plotting ``b1`` and the nerve's ``b1`` on the same axes makes the only
    interesting quantity — the gap between them — visible directly.  If the gap
    closes as the periphery is removed, the structure lived in the periphery.
    """
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        raise ValueError("No successful rows to plot.")
    keys = sorted({str(r.get(x)) for r in ok})
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
    for meas in measures:
        vals = []
        for k in keys:
            sel = [float(r[meas]) for r in ok if str(r.get(x)) == k and meas in r]
            vals.append(np.median(sel) if sel else np.nan)
        style = dict(ls="--", lw=1.2) if meas.startswith("nerve") else dict(lw=1.8)
        ax.plot(range(len(keys)), vals, marker="o", ms=4, label=meas, **style)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel("median over configurations")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title or "Does the structure survive the filter?")
    fig.tight_layout()
    return fig


def plot_population_summary(
    population,
    *,
    ax=None,
    title: str | None = None,
    figsize: tuple[float, float] = (6.5, 4.0),
) -> Figure:
    """Accuracy across the population, so a reader can see the members compare.

    A population statement is only as good as the population: if accuracies
    scatter widely, a difference between members may be a difference in how
    well they trained rather than in what they learned.
    """
    acc = np.asarray(population.accuracies, dtype=float)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
    if not np.isfinite(acc).any():
        ax.text(0.5, 0.5, "no validation accuracy recorded\n(pass val_data to train_population)",
                ha="center", va="center", transform=ax.transAxes, color="0.4")
        ax.set_axis_off()
        return fig
    finite = acc[np.isfinite(acc)]
    ax.bar(np.arange(len(acc)), acc, color="steelblue")
    ax.axhline(finite.mean(), color="crimson", ls="--", lw=1.2,
               label=f"mean {finite.mean():.3f} ± {finite.std():.3f}")
    ax.set_xlabel("member")
    ax.set_ylabel("validation accuracy")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title or f"Population of {len(acc)} networks")
    fig.tight_layout()
    return fig
