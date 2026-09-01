"""weight_trajectory.py — distance matrices from weight training trajectories.

Each training step is a node; distances represent movement in parameter
space (or loss space).

Source papers
-------------
* Birdal et al. (2021): ``distance="euclidean"``
* Andreeva et al. (2024): ``distance="cosine"``
* Dupuis et al. (2023): ``distance="loss_difference"``
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances

from tanc.graph_builder._bundle import GraphBundle
from tanc.graph_builder.node_features import compute_node_features


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_weight_trajectory(
    trajectory: np.ndarray | list[np.ndarray] | list[list[np.ndarray]],
    distance: str = "euclidean",
    loss_values: np.ndarray | None = None,
    node_feature_fn: str | callable | None = None,
    n_node_features: int = 8,
) -> GraphBundle:
    """Build a distance matrix from a weight training trajectory.

    Parameters
    ----------
    trajectory
        The sequence of weights over T training steps:

        * ``(T, P)`` ndarray — use directly.
        * ``list[ndarray]`` — each array is flattened and stacked → ``(T, P)``.
        * ``list[list[ndarray]]`` — layers over time: each time step is a
          list of layer arrays, which are flattened and concatenated → ``(T, P)``.

    distance : str
        ``"euclidean"`` | ``"cosine"`` | ``"loss_difference"``
    loss_values : (T,) or (T, n_samples) ndarray, or None
        Required when ``distance="loss_difference"``.

        * ``(T, n_samples)`` — **per-sample** losses; the distance is the Dupuis
          et al. (2023) pseudo-metric ``rho(w_i, w_j) = mean_s |ell_i,s - ell_j,s|``
          (mean over samples of the absolute per-sample loss difference).
        * ``(T,)`` — a single scalar loss per step; the distance falls back to
          ``|mean-loss_i - mean-loss_j|``, which is *not* the Dupuis pseudo-metric
          (a warning is emitted).
    node_feature_fn : str, callable, or None
        Passed to ``compute_node_features``.  If ``"weights"``, the
        flattened parameter vectors ``(T, P)`` are used as node features.
        ``None`` → use ``loss_values`` as node features if available, else None.
    n_node_features : int
        Number of node feature dimensions.

    Returns
    -------
    GraphBundle
        ``matrix_type="distance"``.
        Symmetric ``(T, T)`` distance matrix.

    Raises
    ------
    ValueError
        ``distance="loss_difference"`` and ``loss_values is None``.
    ValueError
        ``len(loss_values) != T``.
    """
    trajectory_flat = _flatten_trajectory(trajectory)
    T, P = trajectory_flat.shape

    if distance == "loss_difference":
        if loss_values is None:
            raise ValueError(
                "distance='loss_difference' requires loss_values to be provided."
            )
        loss_values = np.asarray(loss_values, dtype=float)
        if loss_values.shape[0] != T:
            raise ValueError(
                f"loss_values first axis ({loss_values.shape[0]}) != T={T}."
            )
        if loss_values.ndim == 2:
            # Dupuis et al. (2023) per-sample pseudo-metric:
            #   rho(w_i, w_j) = (1/n) * sum_s | ell(w_i, z_s) - ell(w_j, z_s) |
            # (the absolute value is INSIDE the mean over samples).  Computed
            # row-by-row so we never allocate the full (T, T, n) tensor.
            dist_matrix = np.zeros((T, T), dtype=float)
            for i in range(T):
                dist_matrix[i] = np.abs(loss_values - loss_values[i]).mean(axis=1)
        else:
            import warnings
            warnings.warn(
                "distance='loss_difference' received a 1-D (per-step) loss "
                "vector, so the distance is |mean-loss_i - mean-loss_j| — NOT "
                "the Dupuis et al. per-sample pseudo-metric "
                "mean_s|ell_i,s - ell_j,s|.  Pass a (T, n_samples) per-sample "
                "loss matrix for the paper-faithful metric.",
                UserWarning, stacklevel=2,
            )
            dist_matrix = np.abs(loss_values[:, None] - loss_values[None, :])
    elif distance == "euclidean":
        dist_matrix = pairwise_distances(trajectory_flat, metric="euclidean")
    elif distance == "cosine":
        dist_matrix = pairwise_distances(trajectory_flat, metric="cosine")
    else:
        raise ValueError(
            f"Unknown distance '{distance}'. "
            "Valid: 'euclidean', 'cosine', 'loss_difference'."
        )

    dist_matrix = dist_matrix.astype(float)
    np.fill_diagonal(dist_matrix, 0.0)

    # Node features
    node_features = None
    if node_feature_fn == "weights":
        node_features = trajectory_flat  # (T, P)
    elif node_feature_fn is not None:
        node_features = compute_node_features(
            dist_matrix, "distance", node_feature_fn, n_node_features
        )
    elif loss_values is not None and np.ndim(loss_values) == 1:
        node_features = np.asarray(loss_values).reshape(-1, 1)

    return GraphBundle(
        matrix=dist_matrix,
        matrix_type="distance",
        node_features=node_features,
        node_labels=None,
        n_nodes=T,
        metadata={
            "builder": "weight_trajectory",
            "distance": distance,
            "t_steps": T,
            "n_params": P,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_trajectory(trajectory) -> np.ndarray:
    """Parse trajectory into a flat ``(T, P)`` array.

    * ``(T, P)`` ndarray — returned as-is.
    * ``list[ndarray]`` — each array is ravelled and stacked.
    * ``list[list[ndarray]]`` — each time step's list of layer arrays is
      ravelled, concatenated, then stacked.
    """
    if isinstance(trajectory, np.ndarray):
        if trajectory.ndim != 2:
            raise ValueError(
                f"If trajectory is an ndarray it must be 2-D (T, P), "
                f"got ndim={trajectory.ndim}."
            )
        return trajectory.astype(float)

    elif not isinstance(trajectory, list) or len(trajectory) == 0:
        raise ValueError("trajectory must be a 2-D ndarray or a non-empty list.")

    first = trajectory[0]

    if isinstance(first, np.ndarray):
        # list[ndarray]: each element is one time step
        rows = [np.asarray(t).ravel() for t in trajectory]
        return np.vstack(rows).astype(float)

    if isinstance(first, list):
        # list[list[ndarray]]: each element is a list of layer arrays
        rows = [
            np.concatenate([np.asarray(layer).ravel() for layer in step])
            for step in trajectory
        ]
        return np.vstack(rows).astype(float)

    raise ValueError(
        f"Cannot parse trajectory with element type {type(first)}."
    )
