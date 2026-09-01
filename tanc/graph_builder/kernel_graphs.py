"""kernel_graphs.py — distance matrices from convolutional filter weights.

Treats each convolutional filter (kernel) as a point in parameter space
and computes pairwise distances, producing a ``GraphBundle`` that can be
passed to ``run_mapper``, ``compute_persistence``, or
``compute_dimension``.

Source papers
-------------
* Carlsson & Gabrielsson (2020): convolutional filter weights as a point
  cloud, PCA lens, Variance Normalized Euclidean (VNE) metric, with
  optional density filtration before Mapper.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from tanc.graph_builder._bundle import GraphBundle
from tanc.graph_builder.node_features import compute_node_features


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_kernel_graph(
    weights: np.ndarray | list[np.ndarray],
    distance: str = "vne",
    density_filter: bool = False,
    density_k: int = 5,
    density_quantile: float = 0.1,
    node_feature_fn: str | callable | None = None,
    n_node_features: int = 8,
) -> GraphBundle:
    """Build a distance matrix from convolutional spatial filters.

    Each **spatial filter** ``(kH, kW)`` is treated as one point in
    a ``kH * kW``-dimensional space (Carlsson & Gabrielsson 2020).
    The resulting ``GraphBundle`` can be passed to ``run_mapper``,
    ``compute_persistence``, or ``compute_dimension``.

    Parameters
    ----------
    weights : ndarray or list[ndarray]
        Convolutional filter weights.  Accepted formats:

        * ``(out_ch, in_ch, kH, kW)`` ndarray — a conv layer's raw
          weight tensor (Conv2d, or Conv1d promoted to 4-D with
          ``kH=1``).  Reshaped to ``(out_ch * in_ch, kH * kW)``:
          each spatial filter is one point.
          Use ``snapshot.kernel_weight(layer_name)`` to get Conv1d
          weights already promoted to this 4-D form.
        * ``(N, kH, kW)`` ndarray — N spatial filters, each flattened
          to ``kH * kW`` dimensions.
        * ``(N, D)`` ndarray — N already-flattened spatial filters.
        * ``list[ndarray]`` — spatial filters from multiple model
          instances (e.g. different random seeds).  Each element is
          processed as above and the results are stacked vertically,
          pooling all spatial filters into one point cloud.  All
          elements must produce the same ``D`` after flattening.

    distance : str
        ``"euclidean"`` | ``"vne"``

        ``"vne"`` is Variance Normalized Euclidean (Carlsson &
        Gabrielsson 2020): each dimension is scaled by the inverse of
        its standard deviation across all filters before computing
        Euclidean distances.  This prevents high-variance dimensions
        from dominating.

    density_filter : bool
        If True, remove low-density points before computing the
        distance matrix (Carlsson & Gabrielsson 2020).  Density is
        estimated as ``1 / mean_kNN_distance``.  Points below the
        ``density_quantile`` threshold are removed.

    density_k : int
        Number of neighbours for the density estimate.  Only used when
        ``density_filter=True``.

    density_quantile : float
        Fraction of lowest-density points to remove, in ``(0, 1)``.
        Only used when ``density_filter=True``.

    node_feature_fn : str, callable, or None
        Passed to ``compute_node_features``.  If ``None``, the raw
        flattened spatial filter vectors are used as node features.

    n_node_features : int
        Number of node feature dimensions.

    Returns
    -------
    GraphBundle
        ``matrix_type="distance"``.

        Metadata contains:

        * ``"filter_weights"`` — the ``(N_filters, spatial_dim)``
          (possibly density-filtered) spatial filter matrix, where
          ``spatial_dim`` is ``kW`` for Conv1d or ``kH*kW`` for Conv2d.
        * ``"removed_indices"`` — indices of filters removed by density
          filtering (empty array if ``density_filter=False``).
        * ``"n_sources"`` — number of input arrays (1 for a single
          array, len(list) for a list).

    Raises
    ------
    ValueError
        Unknown ``distance`` string.
    ValueError
        ``density_quantile`` not in ``(0, 1)``.

    Notes
    -----
    **Spatial filter convention (Carlsson & Gabrielsson 2020).**
    A conv layer with weight tensor ``(out_ch, in_ch, kH, kW)``
    contains ``out_ch * in_ch`` spatial filters, each ``kH * kW``-
    dimensional.  Conv1d layers use the same convention with ``kH=1``
    (promoted to 4-D by the model extractor).  The paper pools
    spatial filters across multiple trained networks to study the
    topology of the learned filter space — pass a list of weight
    tensors to replicate this.

    **Mapper workflow.**
    The recommended filter function is ``"pca"`` with 1 or 2
    components::

        # Conv2d — raw weight tensor: (out_ch, in_ch, kH, kW)
        w = model.conv2d_layer.weight.detach().numpy()
        bundle = build_kernel_graph(w, distance="vne",
                                    density_filter=True)
        result = run_mapper(bundle, filter_fn="pca", n_components=2)

        # Conv1d — via snapshot (auto-promoted to 4-D)
        w = snapshot.kernel_weight("conv1d_layer")
        bundle = build_kernel_graph(w, distance="vne")

        # Pool across 100 trained networks
        ws = [m.conv1.weight.detach().numpy() for m in models]
        bundle = build_kernel_graph(ws, distance="vne",
                                    density_filter=True,
                                    density_quantile=0.3)
    """
    weights = _prepare_spatial_filters(weights)

    removed_indices = np.array([], dtype=int)

    # ── Density filtration ──
    if density_filter:
        if not (0 < density_quantile < 1):
            raise ValueError(
                f"density_quantile must be in (0, 1), got {density_quantile}."
            )
        weights, removed_indices = _density_filter(
            weights, density_k, density_quantile
        )

    # ── Distance matrix ──
    dist_matrix = _build_kernel_distance(weights, distance)
    N = dist_matrix.shape[0]

    # ── Node features ──
    node_features = None
    if node_feature_fn is not None:
        node_features = compute_node_features(
            dist_matrix, "distance", node_feature_fn, n_node_features
        )
    else:
        # Raw filter weights as node features (natural for Mapper PCA lens)
        node_features = weights

    return GraphBundle(
        matrix=dist_matrix,
        matrix_type="distance",
        node_features=node_features,
        node_labels=None,
        n_nodes=N,
        metadata={
            "builder": "kernel_graph",
            "distance": distance,
            "density_filter": density_filter,
            "density_k": density_k,
            "density_quantile": density_quantile,
            "input_shape": weights.shape,
            "filter_weights": weights,
            "removed_indices": removed_indices,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_to_spatial(arr: np.ndarray) -> np.ndarray:
    """Reshape a single weight array to ``(N_filters, spatial_dim)``.

    * ``(N, D)`` — already flattened, return as-is.
    * ``(N, kH, kW)`` — N spatial filters → ``(N, kH*kW)``.
    * ``(out_ch, in_ch, kH, kW)`` — Conv2d (or Conv1d promoted to 4-D)
      raw weights → ``(out_ch * in_ch, kH * kW)``.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # (N_channels, kH, kW) → (N_channels, kH*kW)
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 4:
        # Conv raw: (out_ch, in_ch, kH, kW) → (out_ch * in_ch, kH*kW)
        return arr.reshape(arr.shape[0] * arr.shape[1], -1)
    raise ValueError(
        f"Cannot interpret array with ndim={arr.ndim} as spatial "
        f"filters.  Expected 2-D (N, D), 3-D (N, kH, kW), "
        f"or 4-D (out_ch, in_ch, kH, kW)."
    )


def _prepare_spatial_filters(
    weights: np.ndarray | list[np.ndarray],
) -> np.ndarray:
    """Normalise weights input to a ``(N_filters, spatial_dim)`` array.

    Accepts 4-D ``(out_ch, in_ch, kH, kW)`` raw conv tensors (Conv2d,
    or Conv1d promoted with ``kH=1``), 3-D ``(N, kH, kW)`` spatial
    filter batches, or pre-flattened 2-D ``(N, D)`` arrays.

    If a list is provided, each element is flattened to spatial filters
    and the results are stacked vertically.  All elements must produce
    the same number of columns (same spatial kernel size).
    """
    if isinstance(weights, list):
        if len(weights) == 0:
            raise ValueError("weights list is empty.")
        arrays = [_flatten_to_spatial(w) for w in weights]
        D = arrays[0].shape[1]
        for i, a in enumerate(arrays):
            if a.shape[1] != D:
                raise ValueError(
                    f"All weight arrays must have the same spatial filter "
                    f"dimensionality.  Element 0 has D={D} but "
                    f"element {i} has D={a.shape[1]}."
                )
        return np.vstack(arrays)
    return _flatten_to_spatial(weights)


def _build_kernel_distance(
    weights: np.ndarray,
    distance: str,
) -> np.ndarray:
    """Compute pairwise distance matrix between filters."""
    if distance == "euclidean":
        D = pairwise_distances(weights, metric="euclidean")
        return D.astype(float)

    elif distance == "vne":
        return _vne_distances(weights)

    else:
        raise ValueError(
            f"Unknown distance '{distance}'. Valid: 'euclidean', 'vne'."
        )


def _vne_distances(weights: np.ndarray) -> np.ndarray:
    """Variance Normalized Euclidean distance (Carlsson & Gabrielsson 2020).

    Each dimension is scaled by ``1 / std`` before Euclidean distance
    computation.  Zero-variance dimensions are left unscaled.
    """
    stds = weights.std(axis=0)
    # Avoid division by zero for constant dimensions
    stds = np.where(stds > 0, stds, 1.0)
    weights_scaled = weights / stds
    D = pairwise_distances(weights_scaled, metric="euclidean")
    return D.astype(float)


def _density_filter(
    weights: np.ndarray,
    k: int,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove low-density points from the point cloud.

    Density is estimated as ``1 / mean_kNN_distance``.  Points in the
    lowest ``quantile`` fraction are removed.

    Returns
    -------
    (filtered_weights, removed_indices)
    """
    N = weights.shape[0]
    k_eff = min(k, N - 1)
    if k_eff < 1:
        return weights, np.array([], dtype=int)

    nn = NearestNeighbors(n_neighbors=k_eff + 1)
    nn.fit(weights)
    distances, _ = nn.kneighbors(weights)
    # Exclude self-distance (column 0)
    mean_knn_dist = distances[:, 1:].mean(axis=1)

    # Density = 1 / mean_knn_dist (higher is denser)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(mean_knn_dist > 0, 1.0 / mean_knn_dist, np.inf)

    threshold = np.quantile(density, quantile)
    keep_mask = density >= threshold
    removed_indices = np.where(~keep_mask)[0]

    return weights[keep_mask], removed_indices
