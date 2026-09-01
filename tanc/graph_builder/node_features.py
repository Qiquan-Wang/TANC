"""Node feature computation for graph_builder.

Computes per-node embeddings when no natural node features are available
(e.g. weight-only architecture graphs).

Future methods (e.g. ``"node2vec"``, ``"deepwalk"``) can be registered
without changing the function signature by passing a callable via the
``method`` parameter.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import networkx as nx


def compute_node_features(
    matrix: np.ndarray,
    matrix_type: str,
    method: str | callable = "laplacian_eigenvectors",
    n_components: int = 8,
    **method_kwargs,
) -> np.ndarray:
    """Compute per-node feature embeddings from a graph matrix.

    Parameters
    ----------
    matrix : (N, N) ndarray
        Adjacency, similarity, or distance matrix.
    matrix_type : str
        ``"distance"`` | ``"similarity"`` | ``"adjacency"`` — controls
        conversion to a weight matrix before embedding.
    method : str or callable
        ``"laplacian_eigenvectors"`` (default) | ``"degree_features"`` |
        custom callable ``(matrix) -> (N, k) ndarray``.
    n_components : int
        Number of embedding dimensions (ignored for ``"degree_features"``
        and custom callables).

    Returns
    -------
    (N, n_components) ndarray

    Raises
    ------
    ValueError
        If ``method`` is an unknown string, or ``n_components >= N``, or
        ``method="degree_features"`` and ``n_components != 4``.
    TypeError
        If a callable ``method`` returns the wrong shape.
    """
    N = matrix.shape[0]

    if callable(method):
        result = method(matrix)
        if not isinstance(result, np.ndarray) or result.ndim != 2 or result.shape[0] != N:
            raise TypeError(
                f"Custom method callable must return an (N, k) ndarray with "
                f"N={N}, but got shape {getattr(result, 'shape', 'unknown')}."
            )
        return result

    if method == "laplacian_eigenvectors":
        return _laplacian_eigenvectors(matrix, matrix_type, n_components)
    elif method == "degree_features":
        return _degree_features(matrix, matrix_type, n_components)
    else:
        raise ValueError(
            f"Unknown node feature method '{method}'. "
            "Valid strings: 'laplacian_eigenvectors', 'degree_features'. "
            "Pass a callable for custom methods."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_weight_matrix(matrix: np.ndarray, matrix_type: str) -> np.ndarray:
    """Convert any matrix type to a non-negative weight matrix."""
    if matrix_type == "adjacency":
        W = matrix.astype(float).copy()
    elif matrix_type == "similarity":
        W = matrix.astype(float).copy()
        np.fill_diagonal(W, 0.0)
    elif matrix_type == "distance":
        std = matrix.std()
        if std == 0:
            std = 1.0
        W = np.exp(-matrix / std).astype(float)
        np.fill_diagonal(W, 0.0)
    else:
        raise ValueError(
            f"Unknown matrix_type '{matrix_type}'. "
            "Valid: 'distance', 'similarity', 'adjacency'."
        )
    return W


def _laplacian_eigenvectors(
    matrix: np.ndarray,
    matrix_type: str,
    n_components: int,
) -> np.ndarray:
    """Laplacian spectral embedding.

    Steps
    -----
    1. Convert matrix to non-negative weight matrix W.
    2. Degree matrix D = diag(W.sum(axis=1)).
    3. Normalised Laplacian L = I - D^{-1/2} W D^{-1/2}.
       Zero-degree rows/cols set to zero.
    4. Eigenvectors via scipy.linalg.eigh.
    5. Return n_components eigenvectors for smallest non-trivial eigenvalues.
    """
    N = matrix.shape[0]
    if n_components >= N:
        raise ValueError(
            f"n_components={n_components} must be < N={N}."
        )

    W = _to_weight_matrix(matrix, matrix_type)

    degrees = W.sum(axis=1)
    # D^{-1/2}: handle zero-degree nodes
    with np.errstate(divide="ignore", invalid="ignore"):
        d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)

    # Normalised Laplacian L = I - D^{-1/2} W D^{-1/2}
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt

    # Zero out rows/cols of zero-degree nodes (numeric stability)
    zero_deg = degrees == 0
    L[zero_deg, :] = 0.0
    L[:, zero_deg] = 0.0

    # Eigendecomposition: eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = scipy.linalg.eigh(L)

    # Skip the first (near-zero) eigenvector; take the next n_components
    start = 1
    features = eigenvectors[:, start : start + n_components]

    # If not enough non-trivial vectors, pad with zeros
    if features.shape[1] < n_components:
        pad = np.zeros((N, n_components - features.shape[1]))
        features = np.hstack([features, pad])

    return features.astype(float)


def _degree_features(
    matrix: np.ndarray,
    matrix_type: str,
    n_components: int,
) -> np.ndarray:
    """Lightweight node features: degree, clustering, betweenness, closeness.

    Always returns (N, 4).  Raises ValueError if n_components != 4.
    """
    if n_components != 4:
        raise ValueError(
            f"'degree_features' always returns 4 features "
            f"[degree, clustering_coeff, betweenness_centrality, "
            f"closeness_centrality], but n_components={n_components} was "
            f"requested.  Set n_components=4 to use this method."
        )

    W = _to_weight_matrix(matrix, matrix_type)

    # Build networkx graph
    G = nx.from_numpy_array(W)

    N = matrix.shape[0]
    nodes = list(range(N))

    degrees = np.array([G.degree(n, weight="weight") for n in nodes], dtype=float)
    clustering = np.array(
        [nx.clustering(G, n, weight="weight") for n in nodes], dtype=float
    )
    betweenness = np.array(
        [nx.betweenness_centrality(G, weight="weight")[n] for n in nodes],
        dtype=float,
    )
    closeness = np.array(
        [nx.closeness_centrality(G, n) for n in nodes], dtype=float
    )

    return np.column_stack([degrees, clustering, betweenness, closeness])
