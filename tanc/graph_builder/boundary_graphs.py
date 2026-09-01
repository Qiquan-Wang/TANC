"""boundary_graphs.py — distance matrices for boundary topology.

Two builders:
* ``build_labelled_complex_graph``: label-masked distance matrix for the
  labelled Čech / VR complex (Ramamurthy, Varshney & Mody, 2019).
* ``build_polyhedral_graph``: Hamming distance on unique ReLU activation
  patterns (Liu, Cole, Peterson & Kirby, 2023).
"""

from __future__ import annotations

import numpy as np
import scipy.spatial.distance as sp_dist
from sklearn.metrics import pairwise_distances

from tanc.graph_builder._bundle import GraphBundle


# ─────────────────────────────────────────────────────────────────────────────
# Labelled complex
# ─────────────────────────────────────────────────────────────────────────────

def build_labelled_complex_graph(
    points: np.ndarray,
    labels: np.ndarray,
    distance_metric: str = "euclidean",
    source_class: int | None = None,
    gamma: float | None = None,
    gamma_quantile: float = 0.5,
    scale: str = "global",
    k_local: int = 5,
) -> GraphBundle:
    """Distance matrix for the **labelled Vietoris–Rips complex** (Ramamurthy,
    Varshney & Mody, 2019).

    The complex vertices are the points of one class, ``S`` (``source_class``);
    the opposite-class points form the reference set ``W``.  Following the paper,
    a same-class vertex is kept only when it is **within ``gamma`` of at least one
    point in ``W``** (i.e. it lies near the decision boundary).  The returned
    ``(M, M)`` distance matrix is the pairwise distance among those *eligible*
    ``S`` vertices, so that standard VR persistent homology on it realises the
    labelled complex — **same-class simplices filtered by proximity to the
    opposite class**, not a cross-class distance graph.

    Parameters
    ----------
    points : (N, D) ndarray
    labels : (N,) ndarray of int
        Class label per point.
    distance_metric : str
        ``"euclidean"`` | ``"cosine"`` | ``"correlation"``.
    source_class : int or None
        The class used as the complex vertices ``S``.  ``None`` → the first
        label in ``np.unique(labels)``.  All other classes form ``W``.
    gamma : float or None
        Absolute proximity-to-``W`` threshold.  ``None`` → set from
        ``gamma_quantile`` of the per-vertex nearest-``W`` distances.
    gamma_quantile : float
        Used only when ``gamma is None``: keep vertices whose distance to ``W``
        is at or below this quantile (default ``0.5`` → the half of ``S`` nearest
        the boundary).  ``1.0`` keeps all of ``S``.
    scale : str
        ``"global"`` — plain VR distances (default).
        ``"local"``  — *locally-scaled* labelled VR: each pairwise distance is
        divided by ``sqrt(r_k(u) · r_k(v))`` where ``r_k`` is the vertex's
        ``k_local``-th nearest-neighbour radius among the eligible vertices.
    k_local : int
        Neighbour rank for the local scale (``scale="local"``).

    Returns
    -------
    GraphBundle
        ``matrix_type="distance"``; ``node_labels`` all equal ``source_class``.

    Notes
    -----
    This is the labelled **VR** complex (ripser-computable from a distance
    matrix).  The labelled **Čech** variant replaces the pairwise VR condition
    with a ball-intersection (α-complex) condition and requires GUDHI.
    """
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)

    # Flatten image/conv inputs (N, C, H, W) → (N, C*H*W); the labelled complex
    # computes pairwise distances, which require 2-D points.
    if points.ndim > 2:
        points = points.reshape(points.shape[0], -1)

    if len(points) != len(labels):
        raise ValueError(f"len(points)={len(points)} != len(labels)={len(labels)}.")
    uniq = np.unique(labels)
    if len(uniq) < 2:
        raise ValueError(
            f"At least 2 distinct class labels are required; found {len(uniq)}."
        )
    valid_metrics = {"euclidean", "cosine", "correlation"}
    if distance_metric not in valid_metrics:
        raise ValueError(
            f"Unknown distance_metric '{distance_metric}'. Valid: {valid_metrics}."
        )
    if scale not in {"global", "local"}:
        raise ValueError(f"Unknown scale '{scale}'. Valid: 'global', 'local'.")

    # ── Split into vertices S (source class) and reference W (the rest) ──
    s_label = uniq[0] if source_class is None else source_class
    s_mask = labels == s_label
    if s_mask.sum() < 3:
        raise ValueError(
            f"source_class={s_label} has only {int(s_mask.sum())} points; "
            "need >= 3 vertices for the complex."
        )
    if (~s_mask).sum() < 1:
        raise ValueError("No opposite-class reference points (W is empty).")
    S_pts, W_pts = points[s_mask], points[~s_mask]

    # ── Keep S vertices near the opposite class: d(v, W) <= gamma ──
    d_to_W = pairwise_distances(S_pts, W_pts, metric=distance_metric).min(axis=1)
    if gamma is None:
        gamma = float(np.quantile(d_to_W, gamma_quantile))
    eligible = d_to_W <= gamma
    S_elig = S_pts[eligible]
    if len(S_elig) < 3:
        raise ValueError(
            f"Only {len(S_elig)} eligible vertices within gamma={gamma:.4g} of W; "
            "increase gamma or gamma_quantile."
        )

    # ── Pairwise distances among eligible vertices (optionally locally scaled) ──
    D = pairwise_distances(S_elig, metric=distance_metric).astype(float)
    if scale == "local":
        kk = min(k_local, len(S_elig) - 1)
        r_local = np.sort(D, axis=1)[:, kk]
        r_local = np.where(r_local > 0, r_local, 1.0)
        D = D / np.sqrt(np.outer(r_local, r_local))
    np.fill_diagonal(D, 0.0)

    M = len(S_elig)
    return GraphBundle(
        matrix=D,
        matrix_type="distance",
        node_features=S_elig,
        node_labels=np.full(M, s_label),
        n_nodes=M,
        metadata={
            "builder": "labelled_complex_graph",
            "distance_metric": distance_metric,
            "source_class": int(s_label),
            "n_vertices_total": int(s_mask.sum()),
            "n_eligible": M,
            "gamma": float(gamma),
            "scale": scale,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Polyhedral graph
# ─────────────────────────────────────────────────────────────────────────────

def build_polyhedral_graph(
    activations: np.ndarray,
    input_type: str = "auto",
) -> GraphBundle:
    """Build a Hamming distance matrix over unique ReLU activation patterns.

    Parameters
    ----------
    activations : (N, n_neurons) ndarray
        Either binary patterns (0/1) or continuous activation values.
        When continuous, binarised as ``(activations > 0)``.
        Typically obtained by concatenating hidden-layer activations across
        all ReLU layers.  Exclude the output layer.
    input_type : str
        ``"auto"``       — infer: if all values are 0/1, treat as binary;
                           else binarise via ``(activations > 0)``.
        ``"binary"``     — pre-binarised; raises ValueError if non-binary.
        ``"continuous"`` — always binarise via ``(activations > 0)``.

    Returns
    -------
    GraphBundle
        ``matrix_type="distance"``.
        ``matrix`` is the ``(n_regions, n_regions)`` normalised Hamming
        distance matrix over unique patterns.
        ``node_features`` holds the unique binary pattern for each node.
        ``node_labels=None``.

    Raises
    ------
    ValueError
        ``input_type="binary"`` and data contains non-binary values.
    ValueError
        Fewer than 3 unique patterns found.
    ValueError
        ``activations.ndim != 2``.
    ValueError
        Unknown ``input_type`` string.
    """
    # A multi-layer recording arrives as a list of per-layer matrices (one
    # (N, n_l) array per layer).  Concatenate along the neuron axis — this is the
    # documented input ("concatenating hidden-layer activations across all ReLU
    # layers") — rather than letting np.asarray choke on the ragged list.
    if isinstance(activations, (list, tuple)):
        mats = [np.asarray(a, dtype=float) for a in activations]
        if len(mats) == 1:
            activations = mats[0]
        elif all(m.ndim == 2 and m.shape[0] == mats[0].shape[0] for m in mats):
            activations = np.concatenate(mats, axis=1)
        else:
            raise ValueError(
                "Polyhedral graph received multiple activation matrices that do "
                "not share the sample axis (shapes "
                f"{[m.shape for m in mats]}); record layers with a common batch, "
                "or select a single layer."
            )

    activations = np.asarray(activations)

    if activations.ndim != 2:
        raise ValueError(
            f"activations must be 2-dimensional, got ndim={activations.ndim}."
        )

    binary_patterns = _binarise(activations, input_type)

    # Deduplication
    pattern_strings_all = ["".join(map(str, row)) for row in binary_patterns]
    seen: dict[str, int] = {}
    for idx, s in enumerate(pattern_strings_all):
        if s not in seen:
            seen[s] = idx
    unique_indices = list(seen.values())
    unique_patterns = binary_patterns[unique_indices]
    pattern_strings = list(seen.keys())

    n_regions = len(unique_patterns)
    if n_regions < 3:
        raise ValueError(
            f"Fewer than 3 unique activation patterns found ({n_regions}). "
            "At least 3 are required for meaningful polyhedral PH."
        )

    # Pairwise normalised Hamming distance
    hamming_dist = sp_dist.cdist(
        unique_patterns.astype(float),
        unique_patterns.astype(float),
        metric="hamming",
    )

    return GraphBundle(
        matrix=hamming_dist,
        matrix_type="distance",
        node_features=unique_patterns.astype(float),
        node_labels=None,
        n_nodes=n_regions,
        metadata={
            "builder": "polyhedral_graph",
            "input_type": input_type,
            "n_regions": n_regions,
            "n_neurons": activations.shape[1],
            "n_samples": activations.shape[0],
            "pattern_strings": pattern_strings,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _binarise(activations: np.ndarray, input_type: str) -> np.ndarray:
    """Binarise activation matrix according to input_type."""
    if input_type == "binary":
        if not np.isin(activations, [0, 1]).all():
            raise ValueError(
                "input_type='binary' but activations contain values other "
                "than 0 and 1.  Pass input_type='continuous' or 'auto' to "
                "binarise automatically."
            )
        return activations.astype(int)
    elif input_type == "continuous":
        return (activations > 0).astype(int)
    elif input_type == "auto":
        if np.isin(activations, [0, 1]).all():
            return activations.astype(int)
        else:
            return (activations > 0).astype(int)
    else:
        raise ValueError(
            f"Unknown input_type '{input_type}'. "
            "Valid: 'auto', 'binary', 'continuous'."
        )
