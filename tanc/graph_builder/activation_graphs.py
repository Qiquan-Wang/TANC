"""activation_graphs.py — distance matrices from activation point clouds.

Source papers
-------------
* Naitzat et al. (2020): ``distance="geodesic"`` — Betti numbers decrease
  layer by layer as the network learns.  Uses graph geodesic (hop count)
  on a kNN graph built from Euclidean distances.
* Karuppiah et al. (2025): ``distance="euclidean"`` — direct VR on
  Euclidean distances between sample activation vectors.
* Ballester et al. (2024): ``distance="correlation"`` — correlation distance
  ``d = 1 - |corr|`` between neurons reconstructs whole-network topology
  via functional graphs.
* Gabrielsson & Carlsson (2019): ``distance="vne"`` — Variance Normalised
  Euclidean metric on flattened weight/kernel vectors from multiple models.
  Used with Mapper to cluster models by task or architecture.
* Mapper compatibility — all distance modes store the Euclidean distance
  matrix and raw activations in metadata so the output can be fed to
  ``run_mapper``.  Activation-space Mapper approaches include:

  - Rathore et al. (2021, TopoAct): activation vectors, L2-norm lens,
    Euclidean metric.
  - Zhou et al. (2023): activation vectors, L2-norm lens, Euclidean metric.
  - Gabrielsson & Carlsson (2019): flattened kernels, PCA lens, VNE metric.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.csgraph as sc
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph, NearestNeighbors

from tanc.graph_builder._bundle import GraphBundle
from tanc.graph_builder.node_features import compute_node_features


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_activation_graph(
    activations: np.ndarray | list[np.ndarray],
    distance: str = "euclidean",
    k: int | None = None,
    node_feature_fn: str | callable | None = None,
    n_node_features: int = 8,
    drop_constant: bool = False,
    node_sampling: str | None = None,
    max_neurons: int | None = None,
    sampling_seed: int = 0,
) -> GraphBundle:
    """Build a distance matrix from an activation point cloud.

    Parameters
    ----------
    activations : (N_samples, N_neurons) ndarray or list[ndarray]
        Activation matrix for a single layer, or a list of per-layer
        activation matrices (each ``(N_samples, N_neurons_l)``).  When a
        list is provided the matrices are concatenated horizontally into
        a single ``(N_samples, total_neurons)`` array — this allows
        passing ``snapshot.all_activation_matrices()`` directly.
    distance : str
        ``"euclidean"`` | ``"geodesic"`` | ``"correlation"`` | ``"vne"``
    k : int or None
        k-neighbours for geodesic approximation.  Required when
        ``distance="geodesic"``.
    node_feature_fn : str, callable, or None
        Passed to ``compute_node_features``.  For
        ``distance="correlation"`` the natural node features are the
        activation columns (one trace per neuron), set automatically
        unless overridden.
    n_node_features : int
        Number of node feature dimensions.
    drop_constant : bool
        Drop zero-variance (constant-activation) neurons before building
        the graph.  Ballester et al. (2024) discard such neurons — and a
        correlation distance is undefined for them — so this is the safe
        default whenever ``distance="correlation"``.
    node_sampling : str or None
        Sub-sample neurons (graph nodes) to at most ``max_neurons`` before
        building the distance matrix.  Only meaningful when nodes *are*
        neurons, i.e. ``distance="correlation"``.

        * ``None`` — keep every neuron (default).
        * ``"importance"`` — Ballester et al. (2024) importance sampling
          (Eqs. 4.2–4.3): a neuron's weight is how often it is the
          argmax-|activation| over inputs, with a small floor so every
          neuron is reachable.
        * ``"uniform"`` — sample neurons uniformly at random.
    max_neurons : int or None
        Neuron budget for ``node_sampling``.  Ignored when the network
        already has ``≤ max_neurons`` neurons.
    sampling_seed : int
        Seed for ``node_sampling`` (deterministic per call).

    Returns
    -------
    GraphBundle
        ``matrix_type="distance"``.

        For **all** distance modes the metadata dict contains:

        * ``"euclidean_distance_matrix"`` — (N_samples, N_samples) Euclidean
          distance matrix between sample activation vectors, suitable for
          passing to ``run_mapper`` (see Rathore et al., TopoAct 2021).
        * ``"activations"`` — the (N_samples, N_neurons) activation
          matrix (concatenated if a list was provided), so a
          mapper-compatible ``GraphBundle`` can be constructed with
          ``node_features=metadata["activations"]`` and
          ``matrix=metadata["euclidean_distance_matrix"]``.

    Raises
    ------
    ValueError
        ``distance="geodesic"`` and ``k is None``.

    Notes
    -----
    **Mapper workflows (activation-space).**
    Several papers apply Mapper to activation point clouds.  The choice
    of filter function and metric varies:

    * **Rathore et al. (2021, TopoAct)** — L2-norm of each activation
      vector as the lens function, Euclidean metric.
    * **Zhou et al. (2023)** — 1-D PCA projection as the lens function,
      Euclidean metric; used to study adversarial training.

    When ``distance="euclidean"``, the bundle can be used directly::

        bundle = build_activation_graph(activations, distance="euclidean")
        result = run_mapper(bundle, filter_fn="l2_norm")   # TopoAct
        result = run_mapper(bundle, filter_fn="pca", n_components=1)  # Zhou

    When ``distance="correlation"`` or ``distance="geodesic"``, the primary
    matrix has a different entity space (neurons for correlation, geodesic
    distances for geodesic).  Build a sample-level mapper bundle from the
    metadata::

        bundle = build_activation_graph(activations, distance="correlation")
        mapper_bundle = GraphBundle(
            matrix=bundle.metadata["euclidean_distance_matrix"],
            matrix_type="distance",
            node_features=bundle.metadata["activations"],
            node_labels=None,
            n_nodes=activations.shape[0],
        )
        result = run_mapper(mapper_bundle, filter_fn="pca")

    For weight-space Mapper (Carlsson & Gabrielsson 2020), see
    :func:`~tanc.graph_builder.kernel_graphs.build_kernel_graph`.
    For training-trajectory Mapper (Gabella 2021), see
    :func:`~tanc.graph_builder.weight_trajectory.build_weight_trajectory`.
    """
    activations = _concat_activations(activations)

    # ── Neuron (node) preprocessing — correlation/functional-graph only ──
    if (drop_constant or node_sampling is not None) and distance != "correlation":
        raise ValueError(
            "drop_constant/node_sampling operate on neuron-nodes and are only "
            f"valid for distance='correlation', not '{distance}'."
        )
    if distance == "correlation" and not drop_constant:
        constant = np.flatnonzero(activations.var(axis=0) <= 1e-12)
        if constant.size:
            shown = constant[:8].tolist()
            more = ", ..." if constant.size > 8 else ""
            raise ValueError(
                f"{constant.size} neuron(s) have zero variance (columns {shown}{more}). "
                f"A correlation distance to a constant vector is undefined, so the "
                f"matrix would fill with NaN and the filtration would be undefined. "
                f"Pass drop_constant=True to discard them (Ballester et al. 2024), or "
                f"use a distance that tolerates them, e.g. distance='euclidean'."
            )

    selected_neurons: np.ndarray | None = None
    if distance == "correlation" and (drop_constant or node_sampling is not None):
        activations, selected_neurons = _select_neurons(
            activations, drop_constant, node_sampling, max_neurons, sampling_seed
        )

    dist_matrix = _build_distance_matrix(activations, distance, k)
    N = dist_matrix.shape[0]

    # ── Euclidean distance matrix for Mapper ──
    # Always compute sample-level Euclidean distances so that downstream
    # Mapper pipelines have access regardless of the primary distance mode.
    if distance == "euclidean":
        euclidean_dm = dist_matrix
    else:
        euclidean_dm = pairwise_distances(activations, metric="euclidean")
        euclidean_dm = euclidean_dm.astype(float)

    # ── Node features ──
    node_features = None
    if distance == "correlation" and node_feature_fn is None:
        # Each neuron's activation trace across samples (Ballester et al.)
        node_features = activations.T  # (N_neurons, N_samples)
    elif node_feature_fn is not None:
        node_features = compute_node_features(
            dist_matrix, "distance", node_feature_fn, n_node_features
        )

    # For euclidean/geodesic/vne, activations themselves are natural node
    # features — each sample is one node, each neuron is one feature
    # dimension.  Useful for Mapper (Rathore et al., TopoAct; Gabrielsson 2019).
    if node_features is None and distance in ("euclidean", "geodesic", "vne"):
        node_features = activations  # (N_samples, N_neurons)

    return GraphBundle(
        matrix=dist_matrix,
        matrix_type="distance",
        node_features=node_features,
        node_labels=None,
        n_nodes=N,
        metadata={
            "builder": "activation_graph",
            "distance": distance,
            "k": k,
            "input_shape": activations.shape,
            "euclidean_distance_matrix": euclidean_dm,
            "activations": activations,
            "selected_neurons": selected_neurons,
        },
    )


def find_optimal_k(
    activations: np.ndarray,
    k_range: tuple[int, int] = (2, 20),
    resolution: int = 100,
    criterion: str = "betti_stability",
) -> int:
    """Select the optimal k for geodesic distance computation.

    Selects the k that yields stable Betti-0 numbers across a range of
    values.  Inspired by the kNN-geodesic approach in Naitzat et al.
    (2020), though the paper does not prescribe this specific selection
    procedure.

    Parameters
    ----------
    activations : (N_samples, N_neurons) ndarray
    k_range : (int, int)
        Range ``[k_min, k_max]`` to search.
    resolution : int
        Unused but kept for API compatibility.
    criterion : str
        Only ``"betti_stability"`` is currently implemented.

    Returns
    -------
    int
        Optimal k value.
    """
    if criterion != "betti_stability":
        raise ValueError(
            f"Unknown criterion '{criterion}'. Valid: 'betti_stability'."
        )

    activations = np.asarray(activations, dtype=float)
    k_min, k_max = k_range
    betti_history: list[int] = []
    k_values = list(range(k_min, k_max + 1))

    # Deferred import to avoid circular dependency
    from tanc.topo_tools.ph_tool import compute_persistence

    for k in k_values:
        D = _geodesic_distances(activations, k)
        bundle = GraphBundle(
            matrix=D,
            matrix_type="distance",
            node_features=None,
            node_labels=None,
            n_nodes=D.shape[0],
        )
        result = compute_persistence(bundle, max_dim=0)

        # Betti number at epsilon = median of all pairwise distances
        epsilon = float(np.median(D[D > 0])) if (D > 0).any() else 0.0
        dgm = result.ph_result.diagrams.get(0, np.empty((0, 2)))
        from tanc.topo_tools.ph_tool import betti_number
        # One implementation of the Betti count, in ph_tool.  It uses the
        # half-open convention birth <= eps < death: a bar that dies AT eps
        # is no longer alive there.  The copy that used to live here closed
        # the interval and so over-counted at exactly the death value.
        betti = betti_number(dgm, epsilon)
        betti_history.append(betti)

    # Find first k where Betti number is unchanged for 3 consecutive values
    for i in range(len(betti_history) - 2):
        if (betti_history[i] == betti_history[i + 1] == betti_history[i + 2]):
            return k_values[i]

    # Fallback: k with minimum variance over sliding window of 3
    best_k_idx = 0
    best_var = float("inf")
    for i in range(len(betti_history) - 2):
        var = float(np.var(betti_history[i : i + 3]))
        if var < best_var:
            best_var = var
            best_k_idx = i
    return k_values[best_k_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _concat_activations(
    activations: np.ndarray | list[np.ndarray],
) -> np.ndarray:
    """Normalise activations input to a single (N_samples, N_neurons) array.

    If a list of per-layer matrices is provided, they are concatenated
    along axis 1.  All matrices must share the same number of samples.
    """
    if isinstance(activations, list):
        if len(activations) == 0:
            raise ValueError("activations list is empty.")
        arrays = [np.asarray(a, dtype=float) for a in activations]
        n_samples = arrays[0].shape[0]
        for i, a in enumerate(arrays):
            if a.ndim != 2:
                raise ValueError(
                    f"Each activation matrix must be 2-D, but element {i} "
                    f"has ndim={a.ndim}."
                )
            if a.shape[0] != n_samples:
                raise ValueError(
                    f"All activation matrices must have the same number of "
                    f"samples.  Element 0 has {n_samples} samples but "
                    f"element {i} has {a.shape[0]}."
                )
        return np.hstack(arrays)
    return np.asarray(activations, dtype=float)


def _importance_sample_neurons(A: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of ``k`` neurons sampled by Ballester et al. (2024) importance.

    Importance of a neuron (Eq. 4.2) = how often it is the argmax-|activation|
    over the input samples; the sampling distribution (Eq. 4.3) gives zero-
    importance neurons a small uniform floor so every neuron stays reachable.
    """
    winners = np.argmax(np.abs(A), axis=1)
    I = np.bincount(winners, minlength=A.shape[1]).astype(float)      # Eq. 4.2
    n = A.shape[0]
    nz = I > 0
    n_zero = int(np.sum(~nz))
    p = np.empty(A.shape[1])
    p[nz] = I[nz] / (n + 1)                                           # Eq. 4.3
    p[~nz] = 1.0 / ((n + 1) * max(n_zero, 1))
    p /= p.sum()
    return rng.choice(A.shape[1], size=k, replace=False, p=p)


def _select_neurons(
    activations: np.ndarray,
    drop_constant: bool,
    node_sampling: str | None,
    max_neurons: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop constant neurons and/or sub-sample neurons (columns).

    Returns the reduced ``(N_samples, N_kept)`` activation matrix and the
    indices of the kept neurons (into the original column order).
    """
    idx = np.arange(activations.shape[1])
    if drop_constant:
        keep = activations.var(axis=0) > 1e-12
        activations, idx = activations[:, keep], idx[keep]

    if node_sampling is not None and max_neurons is not None and activations.shape[1] > max_neurons:
        rng = np.random.default_rng(seed)
        if node_sampling == "importance":
            sel = _importance_sample_neurons(activations, max_neurons, rng)
        elif node_sampling == "uniform":
            sel = rng.choice(activations.shape[1], size=max_neurons, replace=False)
        else:
            raise ValueError(
                f"Unknown node_sampling '{node_sampling}'. "
                "Valid: 'importance', 'uniform', or None."
            )
        activations, idx = activations[:, sel], idx[sel]

    return activations, idx


def _build_distance_matrix(
    activations: np.ndarray,
    distance: str,
    k: int | None,
) -> np.ndarray:
    if distance == "euclidean":
        D = pairwise_distances(activations, metric="euclidean")
        return D.astype(float)

    elif distance == "correlation":
        # Correlation between neurons (columns), not samples
        D = 1.0 - np.abs(np.corrcoef(activations.T))
        np.fill_diagonal(D, 0.0)
        return D.astype(float)

    elif distance == "geodesic":
        if k is None:
            raise ValueError(
                "distance='geodesic' requires k to be specified. "
                "Use find_optimal_k() to select k automatically."
            )
        return _geodesic_distances(activations, k)

    elif distance == "vne":
        # Variance Normalised Euclidean (Gabrielsson & Carlsson 2019):
        # divide each feature by its std across the dataset, then take Euclidean.
        std = activations.std(axis=0)
        std[std == 0] = 1.0
        X_norm = activations / std
        D = pairwise_distances(X_norm, metric="euclidean")
        return D.astype(float)

    else:
        raise ValueError(
            f"Unknown distance '{distance}'. "
            "Valid: 'euclidean', 'correlation', 'geodesic', 'vne'."
        )


def _geodesic_distances(activations: np.ndarray, k: int) -> np.ndarray:
    """Compute geodesic (shortest-path) distance matrix via kNN graph."""
    G_sparse = kneighbors_graph(activations, k, mode="connectivity")
    G_arr = G_sparse.toarray()
    # Symmetrise: keep edge if either direction has it
    G_sym = np.maximum(G_arr, G_arr.T)
    # All-pairs shortest paths
    D = sc.shortest_path(G_sym, directed=False)
    # Replace inf (disconnected) with 2 * max_finite
    finite_vals = D[np.isfinite(D)]
    if len(finite_vals) > 0:
        max_finite = float(finite_vals.max())
        D[~np.isfinite(D)] = 2.0 * max_finite
    np.fill_diagonal(D, 0.0)
    return D.astype(float)


