"""mapper_tool.py — Mapper pipeline using giotto-tda.

Source papers
-------------
* Rathore et al. — TopoAct (2021): activation vectors, L2-norm filter,
  70 intervals, 30% overlap.
* Zhou et al. (2023): activation vectors, L2-norm filter, 40 intervals,
  20% overlap.
* Gabrielsson & Carlsson (2019): conv kernels, PCA 2D, VNE metric,
  30 intervals, overlap≈0.667.
* Gabella (2021): weight trajectories, L2-norm filter (primary);
  PCA 3-component (secondary surface analysis).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from tanc.graph_builder._bundle import GraphBundle
from tanc.topo_tools._result import TopoResult


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_mapper(
    bundle: GraphBundle,
    filter_fn: str | callable = "pca",
    n_components: int = 2,
    n_intervals: int = 10,
    overlap_frac: float = 0.3,
    clusterer=None,
    compute_ph: bool = False,
    ph_max_dim: int = 1,
    ph_backend: str = "ripser",
    ph_weight_to_distance: str = "max_minus",
    stat_names: list[str] | None = None,
) -> TopoResult:
    """Full Mapper pipeline using giotto-tda.

    Parameters
    ----------
    bundle : GraphBundle
        ``bundle.node_features`` is used as the input point cloud ``X``.
        Raises ``ValueError`` if ``node_features is None``.
    filter_fn : str or callable
        ``"pca"`` | ``"l2_norm"`` | ``"eccentricity"`` | ``"entropy"`` |
        callable.  ``"l2_norm"`` computes the L2-norm of each point
        (the lens function used in Rathore et al. 2021; Gabella 2021).
    n_components : int
        PCA components; used only when ``filter_fn="pca"``.
    n_intervals : int
        Intervals per filter dimension (CubicalCover parameter).
    overlap_frac : float
        Fractional overlap between adjacent intervals; must be in ``(0, 1)``.
    clusterer
        Clustering step, as a name or an sklearn-compatible object. Names:
        ``"dbscan"`` (default), ``"agglomerative"`` (single-linkage hierarchical
        clustering with an automatic first-gap threshold — the clusterer used by
        Gabrielsson & Carlsson 2019 via Ayasdi; no ``eps``/``min_samples``), or
        ``"histogram_gap"``. ``None`` → ``DBSCAN()``. Pass a configured object for
        other linkages, e.g. ``FirstSimpleGap(linkage="ward")``.
    compute_ph : bool
        If True, also compute PH of the Mapper graph.
    ph_max_dim : int
        Max homology dimension for Mapper-graph PH.
    ph_backend : str
        Backend for Mapper-graph PH.
    ph_weight_to_distance : str
        Conversion scheme for edge weights → distances before PH.
    stat_names : list[str] or None
        Statistics to compute when ``compute_ph=True``.

    Returns
    -------
    TopoResult
        ``tool="mapper"``.
    """
    _check_giotto()

    if bundle.node_features is None:
        raise ValueError(
            "Mapper requires node features (bundle.node_features is None). "
            "Run graph_builder with node_feature_fn='laplacian_eigenvectors' "
            "or provide a custom node_feature_fn to generate node embeddings."
        )

    X = bundle.node_features
    labels = bundle.node_labels

    graph, node_members, filter_values = _mapper_pipeline(
        X, filter_fn, n_components, n_intervals, overlap_frac, clusterer, labels
    )

    graph_stats = mapper_graph_stats(graph)

    ph_result = None
    statistics = None
    if compute_ph:
        ph_result_obj = compute_mapper_ph(
            graph,
            weight_to_distance=ph_weight_to_distance,
            max_dim=ph_max_dim,
            backend=ph_backend,
        )
        ph_result = ph_result_obj.ph_result
        from tanc.topo_tools.ph_tool import compute_statistics
        statistics = compute_statistics(ph_result, stat_names=stat_names)

    return TopoResult(
        tool="mapper",
        ph_result=ph_result,
        statistics=statistics,
        mapper_graph=graph,
        mapper_node_members=node_members,
        mapper_filter_values=filter_values,
        mapper_graph_stats=graph_stats,
        config={
            "filter_fn": str(filter_fn),
            "n_components": n_components,
            "n_intervals": n_intervals,
            "overlap_frac": overlap_frac,
            "compute_ph": compute_ph,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Filter functions
# ─────────────────────────────────────────────────────────────────────────────

def apply_filter(
    X: np.ndarray,
    filter_fn: str | callable,
    n_components: int = 2,
) -> np.ndarray:
    """Apply a filter function to X and return ``(N, k)`` filter values.

    Parameters
    ----------
    X : (N, D) ndarray
    filter_fn : str or callable
        ``"pca"`` | ``"eccentricity"`` | ``"entropy"`` | callable.
    n_components : int
        Components for PCA; ignored for other filters.

    Returns
    -------
    (N, k) ndarray

    Raises
    ------
    ValueError : unknown filter_fn string.
    """
    import pandas as pd
    from gtda.mapper import Projection, Eccentricity, Entropy
    from gtda.mapper.utils.pipeline import transformer_from_callable_on_rows
    from sklearn.decomposition import PCA

    X_df = pd.DataFrame(X)
    if filter_fn == "pca":
        return PCA(n_components=n_components).fit_transform(X_df)
    elif filter_fn == "l2_norm":
        # L2-norm lens: ||x||_2 per sample (Rathore et al. 2021; Gabella 2021)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return norms
    elif filter_fn == "eccentricity":
        return Eccentricity().fit_transform(X_df)
    elif filter_fn == "entropy":
        return Entropy().fit_transform(X_df)
    elif callable(filter_fn):
        return transformer_from_callable_on_rows(filter_fn).fit_transform(X_df)
    else:
        raise ValueError(
            f"Unknown filter_fn '{filter_fn}'. "
            "Valid strings: 'pca', 'l2_norm', 'eccentricity', 'entropy'. "
            "Or pass a callable."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Graph statistics
# ─────────────────────────────────────────────────────────────────────────────

def mapper_graph_stats(graph) -> dict:
    """Compute summary statistics for the Mapper graph.

    Keys: ``n_nodes``, ``n_edges``, ``mean_degree``, ``density``,
    ``n_connected_components``, ``is_connected``,
    ``mean_clustering_coefficient``, ``diameter``.
    """
    import networkx as nx
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    if n_nodes == 0:
        return {
            "n_nodes": 0, "n_edges": 0, "mean_degree": 0.0, "density": 0.0,
            "n_connected_components": 0, "is_connected": False,
            "mean_clustering_coefficient": 0.0, "diameter": np.inf,
        }
    degrees = [d for _, d in graph.degree()]
    mean_degree = float(np.mean(degrees)) if degrees else 0.0
    density = nx.density(graph)
    n_cc = nx.number_connected_components(graph)
    is_connected = nx.is_connected(graph)
    mean_cc = nx.average_clustering(graph)
    if is_connected:
        diameter = nx.diameter(graph)
    else:
        diameter = np.inf
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_degree": mean_degree,
        "density": density,
        "n_connected_components": n_cc,
        "is_connected": is_connected,
        "mean_clustering_coefficient": mean_cc,
        "diameter": diameter,
    }


def cycle_score(graph) -> int:
    """First Betti number: n_edges - n_nodes + n_connected_components."""
    import networkx as nx
    return (graph.number_of_edges()
            - graph.number_of_nodes()
            + nx.number_connected_components(graph))


def class_separation_score(
    graph,
    node_members: dict[int, list[int]],
    labels: np.ndarray,
) -> float:
    """Fraction of edges connecting nodes of different dominant classes.

    Dominant class per node = argmax of label_fractions.

    Returns float in [0, 1].
    """
    labels = np.asarray(labels)
    dominant: dict[int, int] = {}
    for node in graph.nodes():
        members = node_members.get(node, [])
        if not members:
            dominant[node] = -1
        else:
            node_labels = labels[members]
            classes, counts = np.unique(node_labels, return_counts=True)
            dominant[node] = int(classes[np.argmax(counts)])

    edges = list(graph.edges())
    if not edges:
        return 0.0
    cross = sum(
        1 for u, v in edges if dominant.get(u, -1) != dominant.get(v, -1)
    )
    return cross / len(edges)


def mapper_graph_adjacency(graph, weight_attr: str = "weight") -> np.ndarray:
    """Weighted adjacency matrix of shape (N_nodes, N_nodes)."""
    import networkx as nx
    nodes = sorted(graph.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    adj = np.zeros((n, n))
    for u, v, data in graph.edges(data=True):
        w = data.get(weight_attr, 1.0)
        adj[idx[u], idx[v]] = w
        adj[idx[v], idx[u]] = w
    return adj


def node_exemplars(
    result,
    data: np.ndarray | None = None,
    k: int = 5,
    rank_by: str = "centroid",
) -> dict[int, np.ndarray]:
    """Return the ``k`` most-representative data-point indices per Mapper node.

    Designed for paper-style "exemplar" visualisations (e.g. TopoAct's
    per-node sample images, Gabrielsson & Carlsson's per-node conv
    kernels) — the toolkit is deliberately not image-aware, so this
    helper returns *indices into the original data* and lets the caller
    render whatever exemplars they want.

    Parameters
    ----------
    result : TopoResult
        Result of :func:`run_mapper`.
    data : (N, D) ndarray or None
        The original data used to build the Mapper graph (rows align with
        ``mapper_node_members`` indices).  Required when
        ``rank_by="centroid"``; ignored when ``rank_by="random"``.
    k : int
        Number of exemplars per node.
    rank_by : str
        ``"centroid"`` — rank members by Euclidean distance to the node's
        cluster centroid in ``data``-space.
        ``"random"`` — uniformly sample ``k`` members per node.

    Returns
    -------
    dict[int, ndarray]
        ``{mapper_node_id: array_of_data_indices}``.  Each value is at
        most ``k`` long (smaller for sparsely-populated nodes).
    """
    members = getattr(result, "mapper_node_members", None)
    if members is None:
        raise ValueError(
            "result.mapper_node_members is None — pass a TopoResult from "
            "run_mapper(...)."
        )

    out: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(0)

    if rank_by == "centroid":
        if data is None:
            raise ValueError(
                "rank_by='centroid' requires the original `data` array."
            )
        X = np.asarray(data)
        for node_id, idxs in members.items():
            idxs = np.asarray(idxs, dtype=int)
            if idxs.size == 0:
                out[node_id] = idxs
                continue
            centroid = X[idxs].mean(axis=0)
            dists = np.linalg.norm(X[idxs] - centroid, axis=1)
            order = np.argsort(dists)[:k]
            out[node_id] = idxs[order]
        return out

    if rank_by == "random":
        for node_id, idxs in members.items():
            idxs = np.asarray(idxs, dtype=int)
            take = min(k, idxs.size)
            if take == 0:
                out[node_id] = idxs
                continue
            chosen = rng.choice(idxs, size=take, replace=False)
            out[node_id] = chosen
        return out

    raise ValueError(
        f"Unknown rank_by '{rank_by}'. Valid: 'centroid', 'random'."
    )


def _node_member_sets(members) -> list[set]:
    """Coerce a Mapper's node→samples memberships to a list of sets.

    Accepts a ``TopoResult`` (uses ``mapper_node_members``), a mapping
    ``{node: [sample_idx, ...]}``, or a list of member lists.
    """
    if hasattr(members, "mapper_node_members"):
        members = members.mapper_node_members
    if members is None:
        raise ValueError(
            "No Mapper node memberships available.  Pass a TopoResult from "
            "run_mapper (which carries mapper_node_members), a {node: [idx,...]} "
            "mapping, or a list of member lists."
        )
    if isinstance(members, dict):
        return [set(v) for v in members.values()]
    return [set(v) for v in members]


def _hypernetwork(sets: list[set], node_metric: str, disconnected: str):
    """Build the hypernetwork ``(X, mu, Y, nu, omega)`` of Zhou et al. (2023).

    Parameters
    ----------
    sets : list of set
        One member set per Mapper node -- the hyperedges ``Y``.
    node_metric : {"jaccard", "intersection"}
        Edge weight on the line graph.  ``"jaccard"`` is the paper's
        ``w(v, v') = 1 - |y ∩ y'| / |y ∪ y'|``; ``"intersection"`` is its stated
        alternative ``w(v, v') = 1 / |y ∩ y'|``.
    disconnected : {"max_finite", "error"}
        The line graph need not be connected -- two Mapper nodes may share no
        sample even transitively -- leaving ``d(y, y')`` infinite.  The paper
        does not define this case.  ``"max_finite"`` substitutes twice the
        largest finite path length, matching how the toolkit's other graph
        distances handle disconnection; ``"error"`` refuses instead.

    Returns
    -------
    omega : (|X|, |Y|) ndarray
        ``omega(x, y) = 0`` when ``x`` is in ``y``, else the length of the
        shortest path from any hyperedge containing ``x`` to ``y`` (their Eq. 2).
    mu : (|X|,) ndarray
        Vertex measure, proportional to vertex degree.
    nu : (|Y|,) ndarray
        Hyperedge measure, proportional to the summed degree of its vertices.
    """
    import networkx as nx

    universe = sorted(set().union(*sets)) if sets else []
    if not universe:
        raise ValueError("Every Mapper node is empty; there is nothing to compare.")
    x_index = {x: i for i, x in enumerate(universe)}
    n_x, n_y = len(universe), len(sets)

    # deg(x): how many hyperedges contain x
    deg = np.zeros(n_x)
    for y in sets:
        for x in y:
            deg[x_index[x]] += 1.0

    mu = deg / deg.sum() if deg.sum() else np.full(n_x, 1.0 / n_x)

    # nu_hat(y) = sum of vertex degrees within y
    nu_hat = np.array([sum(deg[x_index[x]] for x in y) for y in sets], dtype=float)
    nu = nu_hat / nu_hat.sum() if nu_hat.sum() else np.full(n_y, 1.0 / n_y)

    # Line graph L(H): one node per hyperedge, an edge where they intersect.
    L = nx.Graph()
    L.add_nodes_from(range(n_y))
    for i in range(n_y):
        for j in range(i + 1, n_y):
            inter = len(sets[i] & sets[j])
            if inter == 0:
                continue                      # no edge in the line graph
            if node_metric == "intersection":
                w = 1.0 / inter
            else:                              # jaccard
                w = 1.0 - inter / len(sets[i] | sets[j])
            L.add_edge(i, j, weight=max(w, 0.0))

    d = np.full((n_y, n_y), np.inf)
    for i, lengths in nx.all_pairs_dijkstra_path_length(L, weight="weight"):
        for j, val in lengths.items():
            d[i, j] = val
    np.fill_diagonal(d, 0.0)

    if not np.isfinite(d).all():
        if disconnected == "error":
            raise ValueError(
                "The line graph is disconnected, so some hyperedge distances are "
                "infinite and Zhou et al. do not define this case. Pass "
                "disconnected='max_finite' to substitute twice the largest finite "
                "path length, or compare Mappers built on a shared sample set."
            )
        finite = d[np.isfinite(d)]
        d[~np.isfinite(d)] = 2.0 * float(finite.max()) if finite.size else 1.0

    # omega(x, y): 0 if x in y, else min over hyperedges containing x (Eq. 2)
    contains = [[] for _ in range(n_x)]
    for j, y in enumerate(sets):
        for x in y:
            contains[x_index[x]].append(j)
    omega = np.zeros((n_x, n_y))
    for i in range(n_x):
        rows = contains[i]
        omega[i, :] = d[rows, :].min(axis=0) if rows else d.max()
    return omega, mu, nu


def mapper_hypernetwork_gw_distance(
    members_a,
    members_b,
    node_metric: str = "jaccard",
    n_restarts: int = 1,
    disconnected: str = "max_finite",
    seed: int = 0,
) -> float:
    """Co-optimal-transport distance between two Mapper graphs (Zhou et al. 2023).

    This implements the paper's construction rather than approximating it.  Each
    Mapper graph becomes a **hypernetwork** ``H = (X, mu, Y, nu, omega)``: ``X``
    is the sample set, ``Y`` the Mapper nodes read as hyperedges, and ``omega``
    the ``|X| x |Y|`` relation given by their Eq. 2 -- the shortest-path distance,
    on the Jaccard-weighted line graph, from any hyperedge containing a sample to
    the hyperedge in question.  Measures are degree-based: ``mu`` proportional to
    vertex degree, ``nu(y)`` to the summed degree of the vertices in ``y``.

    Their Eq. 3 then minimises over **two couplings at once** -- one on samples,
    one on nodes -- which is a co-optimal transport problem, solved here with
    POT's ``co_optimal_transport2``.

    What the returned number is
    ---------------------------
    That objective is non-convex and is solved by block-coordinate descent, so
    **the value is an upper bound** on the true distance: a minimum over however
    many local optima the restarts happen to reach.  Three practical consequences,
    all shared with the authors' reference implementation:

    * ``d(H, H)`` is not guaranteed to be ``0`` at a low restart count.  On a
      six-node example it stays near ``0.14`` until roughly sixty restarts, then
      drops to exactly zero.  A non-zero self-distance means the solver has not
      found the identity coupling, not that the construction is wrong.
    * the value is not exactly symmetric in its arguments -- around a percent is
      typical -- so build a distance matrix in a fixed argument order.
    * more restarts can only lower the value, never raise it.

    The paper's text describes 20 randomly initialised instances keeping the
    smallest, while the authors' reference implementation
    (``tdavislab/mapper-compare``) uses a single **uniform** initialisation.  The
    default here follows the reference code; raise ``n_restarts`` for a tighter
    bound, at proportional cost.

    Unlike a plain Gromov-Wasserstein comparison, this applies even when the two
    Mappers are built over *different* sample universes.

    Parameters
    ----------
    members_a, members_b : TopoResult | dict | list
        Node→samples memberships for each Mapper.
    node_metric : {"jaccard", "intersection"}
        Line-graph edge weight; see :func:`_hypernetwork`.
    n_restarts : int
        Initialisations, smallest kept.  ``1`` (default) is the single uniform
        start used by the authors' reference code; higher values add randomised
        restarts, as the paper's text describes.
    disconnected : {"max_finite", "error"}
        How to treat an unreachable hyperedge pair; see :func:`_hypernetwork`.
    seed : int
        Seeds the restart initialisations, so the result is reproducible.

    Returns
    -------
    float
        The distance, square-rooted from the squared COOT objective so it is on
        the same scale as the input relation.

    See Also
    --------
    mapper_hypergraph_gw_distance : cheaper single-coupling approximation.
    mapper_gw_distance : adjacency-only proxy, ignoring memberships.
    """
    try:
        import ot
        from ot.coot import co_optimal_transport2
    except ImportError as exc:
        raise ImportError(
            "mapper_hypernetwork_gw_distance requires POT (pip install pot)."
        ) from exc
    if node_metric not in {"jaccard", "intersection"}:
        raise ValueError(
            f"node_metric must be 'jaccard' or 'intersection'; got {node_metric!r}."
        )
    if n_restarts < 1:
        raise ValueError(f"n_restarts must be at least 1; got {n_restarts}.")

    A = _node_member_sets(members_a)
    B = _node_member_sets(members_b)
    if len(A) == 0 or len(B) == 0:
        raise ValueError("Both Mappers must have at least one node.")

    wa, mu_a, nu_a = _hypernetwork(A, node_metric, disconnected)
    wb, mu_b, nu_b = _hypernetwork(B, node_metric, disconnected)

    def _random_coupling(p_, q_, rng_):
        """A random coupling of two marginals, by Sinkhorn-scaling noise."""
        K = rng_.random((len(p_), len(q_))) + 1e-3
        for _ in range(60):
            K *= (p_ / (K.sum(axis=1) + 1e-15))[:, None]
            K *= (q_ / (K.sum(axis=0) + 1e-15))[None, :]
        return K

    rng = np.random.default_rng(seed)
    best = np.inf
    for r in range(n_restarts):
        # r == 0 is the uniform (product) coupling, which is what the authors'
        # reference implementation uses; later restarts are randomised.  Note
        # that ``duals_*`` would be ignored here: those seed the Sinkhorn solver,
        # and this runs exact EMD.
        if r == 0:
            warmstart = None
        else:
            # POT requires all four keys when warmstart is given.  The duals only
            # matter to its Sinkhorn path (this runs exact EMD), so they are zero;
            # the couplings are what actually randomise the start.
            warmstart = {
                "duals_sample": (np.zeros(wa.shape[0]), np.zeros(wb.shape[0])),
                "duals_feature": (np.zeros(wa.shape[1]), np.zeros(wb.shape[1])),
                "pi_sample": _random_coupling(mu_a, mu_b, rng),
                "pi_feature": _random_coupling(nu_a, nu_b, rng),
            }
        val = co_optimal_transport2(
            wa, wb, mu_a, nu_a, mu_b, nu_b, warmstart=warmstart,
        )
        best = min(best, float(val))
    if not np.isfinite(best):
        raise RuntimeError(
            "Every co-optimal-transport restart failed. Check that both Mappers "
            "have non-empty nodes, or reduce n_restarts."
        )
    return float(max(best, 0.0)) ** 0.5


def mapper_hypergraph_gw_distance(
    members_a,
    members_b,
    mass: str = "cluster_size",
    node_metric: str = "jaccard",
    loss_fun: str = "square_loss",
) -> float:
    """Gromov-Wasserstein distance between two Mapper outputs **as hypergraphs**.

    An **approximation** to Zhou et al. (2023), not their full construction.
    Each Mapper node is treated as a **hyperedge** — the set of original
    sample indices it contains — so the comparison uses node-to-sample
    *memberships*, not merely the Mapper adjacency graph (which the plain
    :func:`mapper_gw_distance` uses).

    How it differs from the paper
    -----------------------------
    Zhou et al. model a Mapper graph as a hypernetwork ``(X, mu, Y, nu, omega)``
    and compare two of them with a **co-optimal transport** problem (their Eq. 3)
    over *two* couplings at once — one on samples ``X``, one on nodes ``Y``.
    This function instead solves a standard single-coupling Gromov-Wasserstein
    problem on the nodes alone.  Concretely:

    * ``omega`` in the paper is a rectangular ``|X| x |Y|`` sample-to-node
      relation; here the structure matrix is square, node-to-node.
    * the paper's entries are **shortest-path** distances on the Jaccard-weighted
      line graph; here they are raw pairwise Jaccard distances, so two nodes
      sharing no samples score 1.0 rather than a path length through nodes that
      do overlap.
    * the paper's measures are degree-based (``mu`` proportional to vertex
      degree, ``nu(y)`` to the sum of vertex degrees in ``y``); here the default
      is cluster size.

    The result is a membership-aware **summary** that follows the paper's intent
    rather than a reproduction of its metric.  Values lie on the bounded Jaccard
    scale and are not comparable with the paper's published numbers.

    Two ingredients follow the hypergraph view:

    * **Structural cost** — the intra-hypergraph distance between two nodes is a
      metric on their member sets (default ``"jaccard"``:
      ``1 - |A∩B| / |A∪B|``), i.e. how the clusters relate through shared
      samples.
    * **Node masses** — proportional to **cluster size** (``mass="cluster_size"``)
      rather than uniform, so larger regions carry more weight.

    Gromov-Wasserstein (via POT) then compares the two metric-measure
    hypergraphs, so it applies even when the two Mappers live over *different*
    sample universes (e.g. different layers).

    Parameters
    ----------
    members_a, members_b : TopoResult | dict | list
        Node→samples memberships for each Mapper (see :func:`_node_member_sets`).
    mass : str
        ``"cluster_size"`` (default) | ``"uniform"``.
    node_metric : str
        ``"jaccard"`` (default) | ``"overlap"`` (``1 - |A∩B|/min(|A|,|B|)``).
    loss_fun : str
        POT GW loss: ``"square_loss"`` | ``"kl_loss"``.

    Returns
    -------
    float
        The GW distance (square-rooted from POT's squared GW, a proper metric).
    """
    try:
        import ot
    except ImportError as exc:
        raise ImportError(
            "mapper_hypergraph_gw_distance requires POT (pip install pot)."
        ) from exc

    A = _node_member_sets(members_a)
    B = _node_member_sets(members_b)
    if len(A) == 0 or len(B) == 0:
        raise ValueError("Both Mappers must have at least one node.")

    def _structure(sets_: list[set]) -> np.ndarray:
        n = len(sets_)
        C = np.zeros((n, n))
        for i in range(n):
            for k in range(i + 1, n):
                inter = len(sets_[i] & sets_[k])
                if node_metric == "overlap":
                    denom = min(len(sets_[i]), len(sets_[k]))
                else:  # jaccard
                    denom = len(sets_[i] | sets_[k])
                d = 1.0 - (inter / denom if denom else 0.0)
                C[i, k] = C[k, i] = d
        return C

    def _mass(sets_: list[set]) -> np.ndarray:
        if mass == "uniform":
            return np.full(len(sets_), 1.0 / len(sets_))
        w = np.array([len(s) for s in sets_], dtype=float)
        return w / w.sum() if w.sum() > 0 else np.full(len(sets_), 1.0 / len(sets_))

    C1, C2 = _structure(A), _structure(B)
    p, q = _mass(A), _mass(B)
    sq = ot.gromov.gromov_wasserstein2(C1, C2, p, q, loss_fun=loss_fun)
    return float(max(sq, 0.0)) ** 0.5


def mapper_gw_distance(
    graph_a,
    graph_b,
    weight_attr: str = "weight",
    structure: str = "shortest_path",
    loss_fun: str = "square_loss",
) -> float:
    """Gromov-Wasserstein distance between two Mapper **adjacency graphs**.

    A *plain-graph* approximation: each graph is summarised by its node-to-node
    shortest-path distance matrix (default) with **uniform** node masses.  It
    ignores node-to-sample memberships entirely.  For a membership-aware
    comparison see :func:`mapper_hypergraph_gw_distance`; note that neither
    function implements the full co-optimal-transport construction of
    Zhou et al. (2023).

    Requires the POT library (``pip install pot``).  GW has no good
    pure-numpy implementation; the toolkit raises ``ImportError`` with an
    install hint rather than ship a buggy fallback.

    Parameters
    ----------
    graph_a, graph_b : networkx.Graph
        Mapper graphs as returned by :func:`run_mapper`.
    weight_attr : str
        Edge attribute used as the path length.
    structure : str
        ``"shortest_path"`` (default) — geodesic distances on the graph.
        ``"adjacency"`` — raw weighted adjacency (use only when graphs
        are well connected; can give degenerate matrices otherwise).
    loss_fun : str
        Passed to POT's ``gromov_wasserstein2`` (``"square_loss"`` or
        ``"kl_loss"``).

    Returns
    -------
    float
        The GW distance.  Square-rooted from the squared GW that POT
        returns, so that the value is a proper metric.
    """
    try:
        import ot  # POT
    except ImportError as exc:
        raise ImportError(
            "mapper_gw_distance requires the POT (Python Optimal Transport) "
            "library. Install with: pip install pot"
        ) from exc

    import networkx as nx

    def _structure_matrix(g):
        if structure == "shortest_path":
            try:
                sp = dict(nx.all_pairs_dijkstra_path_length(
                    g, weight=weight_attr
                ))
            except Exception:
                sp = dict(nx.all_pairs_shortest_path_length(g))
            nodes = sorted(g.nodes())
            n = len(nodes)
            idx = {v: i for i, v in enumerate(nodes)}
            D = np.full((n, n), np.inf)
            for u, neighbours in sp.items():
                for v, d in neighbours.items():
                    D[idx[u], idx[v]] = d
            # Replace inf (disconnected components) with twice the
            # finite diameter so GW remains well-defined.
            finite = D[np.isfinite(D)]
            if finite.size:
                D[~np.isfinite(D)] = 2.0 * float(finite.max())
            else:
                D[~np.isfinite(D)] = 1.0
            np.fill_diagonal(D, 0.0)
            return D
        elif structure == "adjacency":
            return mapper_graph_adjacency(g, weight_attr=weight_attr)
        raise ValueError(f"Unknown structure '{structure}'.")

    C1 = _structure_matrix(graph_a)
    C2 = _structure_matrix(graph_b)
    p = np.full(C1.shape[0], 1.0 / C1.shape[0])
    q = np.full(C2.shape[0], 1.0 / C2.shape[0])
    sq = ot.gromov.gromov_wasserstein2(C1, C2, p, q, loss_fun=loss_fun)
    return float(max(sq, 0.0)) ** 0.5


def compute_mapper_ph(
    graph,
    weight_attr: str = "weight",
    weight_to_distance: str = "max_minus",
    max_dim: int = 1,
    backend: str = "ripser",
) -> TopoResult:
    """Compute persistent homology of the Mapper graph.

    .. warning::
        PH of the Mapper graph is **not topologically stable**.  Small
        perturbations of the input data can change the Mapper graph
        structure substantially.  Interpret descriptively, not as a stable
        topological invariant.

    Parameters
    ----------
    graph : networkx.Graph
    weight_attr : str
        Edge attribute name for weights.
    weight_to_distance : str
        ``"max_minus"``  → ``d = max(W) - w``
        ``"reciprocal"`` → ``d = 1 / w``
        ``"one_minus"``  → ``d = 1 - w``
        ``"raw"``        → ``d = w``
    max_dim : int
    backend : str

    Raises
    ------
    ValueError : graph has no edges.
    ValueError : invalid weight_to_distance for given weights.
    ValueError : unknown weight_to_distance string.
    """
    if graph.number_of_edges() == 0:
        raise ValueError("Mapper graph has no edges; cannot compute PH.")

    W = mapper_graph_adjacency(graph, weight_attr)

    # Extract edge weights (non-zero upper triangle)
    iu = np.triu_indices_from(W, k=1)
    weights = W[iu]
    weights_nonzero = weights[weights > 0]

    if weight_to_distance == "max_minus":
        max_w = W.max()
        D = max_w - W
        np.fill_diagonal(D, 0.0)
    elif weight_to_distance == "reciprocal":
        if (weights_nonzero == 0).any():
            raise ValueError(
                "weight_to_distance='reciprocal' but some edge weights are 0."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.where(W > 0, 1.0 / W, 0.0)
        np.fill_diagonal(D, 0.0)
    elif weight_to_distance == "one_minus":
        if weights_nonzero.min() < 0 or weights_nonzero.max() > 1:
            raise ValueError(
                "weight_to_distance='one_minus' but some weights are outside [0, 1]."
            )
        D = np.where(W > 0, 1.0 - W, 0.0)
        np.fill_diagonal(D, 0.0)
    elif weight_to_distance == "raw":
        D = W.copy()
        np.fill_diagonal(D, 0.0)
    else:
        raise ValueError(
            f"Unknown weight_to_distance '{weight_to_distance}'. "
            "Valid: 'max_minus', 'reciprocal', 'one_minus', 'raw'."
        )

    from tanc.graph_builder._bundle import GraphBundle
    from tanc.topo_tools.ph_tool import compute_persistence

    bundle = GraphBundle(
        matrix=D,
        matrix_type="distance",
        node_features=None,
        node_labels=None,
        n_nodes=D.shape[0],
    )
    return compute_persistence(bundle, input_type="distance_matrix",
                               max_dim=max_dim, backend=backend)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_mapper_graph(
    graph,
    node_members: dict[int, list[int]] | None,
    filter_values: np.ndarray | None,
    color_by: np.ndarray | None = None,
    node_size_by: str = "size",
    layout: str = "spring",
    title: str | None = None,
    ax=None,
    figsize: tuple[float, float] | None = None,
    render: str = "matplotlib",
    gtda_pipeline=None,
    gtda_data: np.ndarray | None = None,
    color_variable: np.ndarray | None = None,
):
    """Visualise the Mapper graph.

    Parameters
    ----------
    render : str
        ``"matplotlib"`` (default) — networkx + matplotlib Figure.
        ``"giotto"`` — plotly Figure via giotto-tda.

    Returns
    -------
    matplotlib.figure.Figure or plotly Figure.
    """
    if render == "matplotlib":
        return _plot_mapper_matplotlib(
            graph, node_members, filter_values, color_by,
            node_size_by, layout, title, ax, figsize
        )
    elif render == "giotto":
        if gtda_pipeline is None or gtda_data is None:
            raise ValueError(
                "render='giotto' requires gtda_pipeline and gtda_data to be "
                "passed explicitly."
            )
        # Warn if matplotlib-only params are non-default
        for param_name, param_val, default in [
            ("ax", ax, None), ("figsize", figsize, None),
            ("layout", layout, "spring"), ("node_size_by", node_size_by, "size"),
        ]:
            if param_val != default:
                warnings.warn(
                    f"render='giotto': parameter '{param_name}' is ignored.",
                    UserWarning,
                    stacklevel=2,
                )
        try:
            from gtda.mapper.visualization import plot_static_mapper_graph
        except ImportError:
            raise ImportError("giotto-tda is required: pip install giotto-tda")
        return plot_static_mapper_graph(
            gtda_pipeline, gtda_data, color_variable=color_variable
        )
    else:
        raise ValueError(
            f"Unknown render='{render}'. Valid: 'matplotlib', 'giotto'."
        )


def _plot_mapper_matplotlib(
    graph,
    node_members,
    filter_values,
    color_by,
    node_size_by,
    layout,
    title,
    ax,
    figsize,
):
    import matplotlib.pyplot as plt
    import networkx as nx

    if figsize is None:
        figsize = (8, 6)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Layout
    if layout == "spring":
        pos = nx.spring_layout(graph, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.spring_layout(graph, seed=42)

    # Node sizes
    nodes = list(graph.nodes())
    if node_size_by == "size" and node_members is not None:
        sizes = [max(1, len(node_members.get(n, []))) * 20 for n in nodes]
    else:
        sizes = [200] * len(nodes)

    # Node colours
    if color_by is not None and node_members is not None:
        node_colors = [
            float(np.mean(color_by[node_members.get(n, [0])]))
            if node_members.get(n)
            else 0.0
            for n in nodes
        ]
        vmin, vmax = min(node_colors), max(node_colors)
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )
        sm.set_array([])
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes,
                               node_size=sizes, node_color=node_colors,
                               cmap="viridis", ax=ax)
        plt.colorbar(sm, ax=ax)
    else:
        node_sizes_arr = [max(1, len(node_members.get(n, []))) for n in nodes] if node_members else [1] * len(nodes)
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes,
                               node_size=sizes,
                               node_color=node_sizes_arr,
                               cmap="Blues", ax=ax)

    # Edge widths
    edge_weights = [graph[u][v].get("weight", 1) for u, v in graph.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [1.0 + 3.0 * w / max_w for w in edge_weights]
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.7, ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.axis("off")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Internal pipeline
# ─────────────────────────────────────────────────────────────────────────────

# Named clusterers accepted by ``run_mapper(clusterer=...)``.  DBSCAN is the giotto
# default; the "agglomerative"/"histogram_gap" options are single-linkage hierarchical
# clusterers with an *automatic* gap threshold — the family Gabrielsson & Carlsson (2019)
# use via Ayasdi ("single linkage clustering with a fixed heuristic for the threshold").
_CLUSTERER_ALIASES = {
    "dbscan": "dbscan",
    "agglomerative": "single_linkage",
    "single_linkage": "single_linkage",
    "single-linkage": "single_linkage",
    "single": "single_linkage",
    "histogram_gap": "histogram_gap",
    "histogram-gap": "histogram_gap",
    "histogram": "histogram_gap",
}


def _resolve_clusterer(clusterer):
    """Resolve ``clusterer`` (a name, ``None``, or an object) to an sklearn-compatible clusterer.

    Named options (case-insensitive strings):

    * ``"dbscan"`` (also the default for ``None``) — density-based ``DBSCAN()``.
    * ``"agglomerative"`` / ``"single_linkage"`` — agglomerative (single-linkage) hierarchical
      clustering with an automatic *first-gap* threshold (giotto's ``FirstSimpleGap``). This is
      the clusterer Gabrielsson & Carlsson (2019) use (single-linkage via Ayasdi); it takes no
      ``eps`` / ``min_samples``.
    * ``"histogram_gap"`` — agglomerative single-linkage with a histogram-gap threshold
      (giotto's ``FirstHistogramGap``).

    For a different linkage (``"complete"``/``"average"``/``"ward"``) or tuned gap parameters,
    pass a configured object directly, e.g. ``clusterer=FirstSimpleGap(linkage="ward")``.
    Any object that is not a string is returned unchanged.
    """
    if clusterer is None:
        from sklearn.cluster import DBSCAN
        return DBSCAN()
    if not isinstance(clusterer, str):
        return clusterer

    key = _CLUSTERER_ALIASES.get(clusterer.strip().lower())
    if key == "dbscan":
        from sklearn.cluster import DBSCAN
        return DBSCAN()
    if key == "single_linkage":
        from gtda.mapper import FirstSimpleGap
        return FirstSimpleGap()
    if key == "histogram_gap":
        from gtda.mapper import FirstHistogramGap
        return FirstHistogramGap()
    raise ValueError(
        f"Unknown clusterer '{clusterer}'. Valid names: 'dbscan', 'agglomerative' "
        "(single-linkage with an auto gap threshold — the paper's clusterer), 'histogram_gap'. "
        "Or pass an sklearn-compatible clusterer object (e.g. FirstSimpleGap(linkage='ward'))."
    )


def _mapper_pipeline(
    X: np.ndarray,
    filter_fn,
    n_components: int,
    n_intervals: int,
    overlap_frac: float,
    clusterer,
    labels,
):
    """Run the full giotto-tda Mapper pipeline and return (graph, node_members, filter_values)."""
    import pandas as pd
    import networkx as nx
    from sklearn.decomposition import PCA
    from gtda.mapper import (
        make_mapper_pipeline,
        CubicalCover,
        Projection,
        Eccentricity,
        Entropy,
    )
    from gtda.mapper import transformer_from_callable_on_rows

    clusterer = _resolve_clusterer(clusterer)

    # Resolve filter object
    X_df = pd.DataFrame(X)
    if filter_fn == "pca":
        filter_obj = PCA(n_components=n_components)
    elif filter_fn == "l2_norm":
        # L2-norm lens (Rathore et al. 2021; Gabella 2021).
        # FunctionTransformer operates on the full matrix at once, avoiding
        # the row-vs-column ambiguity of transformer_from_callable_on_rows.
        from sklearn.preprocessing import FunctionTransformer
        filter_obj = FunctionTransformer(
            lambda X: np.linalg.norm(np.asarray(X), axis=1, keepdims=True)
        )
    elif filter_fn == "eccentricity":
        filter_obj = Eccentricity()
    elif filter_fn == "entropy":
        filter_obj = Entropy()
    elif callable(filter_fn):
        filter_obj = transformer_from_callable_on_rows(filter_fn)
    else:
        raise ValueError(
            f"Unknown filter_fn '{filter_fn}'. "
            "Valid strings: 'pca', 'l2_norm', 'eccentricity', 'entropy'."
        )

    pipe = make_mapper_pipeline(
        filter_func=filter_obj,
        cover=CubicalCover(n_intervals=n_intervals, overlap_frac=overlap_frac),
        clusterer=clusterer,
    )

    gtda_graph = pipe.fit_transform(X_df)
    filter_values = apply_filter(X, filter_fn, n_components)
    graph = _igraph_to_networkx(gtda_graph, X, filter_values, labels)
    node_members = {n: list(graph.nodes[n]["members"]) for n in graph.nodes}

    return graph, node_members, filter_values


def _igraph_to_networkx(
    gtda_graph,
    X: np.ndarray,
    filter_values: np.ndarray,
    labels,
):
    """Convert giotto igraph.Graph to nx.Graph with standard node attributes."""
    import networkx as nx

    G = nx.Graph()

    # Vertices
    for v in gtda_graph.vs:
        node_id = v.index
        members = list(v["node_elements"])
        size = len(members)
        mean_filter = float(np.mean(filter_values[members], axis=0).mean()) if members else 0.0
        attrs = {
            "members": members,
            "size": size,
            "mean_filter": mean_filter,
        }
        if labels is not None:
            node_labels = np.asarray(labels)[members]
            classes, counts = np.unique(node_labels, return_counts=True)
            label_fractions = {int(c): float(cnt / size) for c, cnt in zip(classes, counts)}
            attrs["label_fractions"] = label_fractions
        G.add_node(node_id, **attrs)

    # Edges
    for e in gtda_graph.es:
        u, v = e.source, e.target
        # Shared members
        members_u = set(gtda_graph.vs[u]["node_elements"])
        members_v = set(gtda_graph.vs[v]["node_elements"])
        shared = len(members_u & members_v)
        G.add_edge(u, v, weight=shared)

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _check_giotto() -> None:
    try:
        import gtda  # noqa: F401
    except ImportError:
        raise ImportError(
            "giotto-tda is required for the Mapper tool. "
            "Install with: pip install giotto-tda"
        )

# Public alias for __init__.py exports
mapper_pipeline = _mapper_pipeline
