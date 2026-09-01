"""point_cloud_graphs.py — treat a weight matrix as a point cloud of rows or columns.

The :func:`~tanc.graph_builder.weight_graphs.build_weight_graph` builder turns a
weight matrix into a *bipartite / multipartite neuron graph* (input and output
neurons are both nodes; weights are edges).  This builder takes the complementary
view: it reads the matrix as a **point cloud**, where each row — or each column —
is a single point, and returns the pairwise-distance ``GraphBundle`` that persistent
homology, Mapper and the dimension estimators consume directly.

For a weight matrix of the toolkit's ``(N_in, N_out)`` convention:

* ``orientation="rows"`` → the ``N_in`` rows are points in ``R^{N_out}``: each
  **input feature** as a point in output-activation space.
* ``orientation="cols"`` → the ``N_out`` columns are points in ``R^{N_in}``: each
  **output neuron** (its receptive field) as a point in input space.

The two clouds carry complementary information: pairwise row distances are invariant
to an orthogonal map applied on the *output* side, and column distances to one applied
on the *input* side — the coordinate-free counterpart of separating the left and right
singular subspaces of the weight matrix.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from tanc.graph_builder._bundle import GraphBundle

_VALID_ORIENTATIONS = ("rows", "cols")
_VALID_METRICS = ("euclidean", "cosine", "correlation", "cityblock")


def build_point_cloud_graph(
    data: np.ndarray | list[np.ndarray],
    orientation: str = "rows",
    metric: str = "euclidean",
    attach_points: bool = True,
) -> GraphBundle:
    """Build a pairwise-distance ``GraphBundle`` from the rows or columns of a matrix.

    Parameters
    ----------
    data : (M, N) ndarray or length-1 list of one
        A single weight matrix.  ``representation="weights"`` yields a *list* of
        per-layer matrices; select one layer (e.g.
        ``layer_selection=["fc1"]``) so a single matrix reaches this builder, or
        pass the array directly.
    orientation : str
        ``"rows"`` — the ``M`` rows become points in ``R^N``;
        ``"cols"`` — the ``N`` columns become points in ``R^M``.
    metric : str
        Any of ``"euclidean"``, ``"cosine"``, ``"correlation"``, ``"cityblock"``
        (passed to :func:`scipy.spatial.distance.cdist`).
    attach_points : bool
        Store the point coordinates as ``node_features`` so the bundle is
        Mapper-ready.  Set ``False`` to keep the bundle lightweight.

    Returns
    -------
    GraphBundle
        ``matrix`` is the ``(P, P)`` distance matrix (``P = M`` for rows,
        ``N`` for columns), ``matrix_type="distance"`` — ready for
        :func:`~tanc.topo_tools.ph_tool.run_ph`, ``run_mapper`` or the
        dimension estimators.
    """
    if isinstance(data, (list, tuple)):
        if len(data) != 1:
            raise ValueError(
                f"build_point_cloud_graph needs a single matrix, got {len(data)}. "
                "A point cloud is defined per weight matrix — select one layer, e.g. "
                "layer_selection=['fc1'], or pass the array directly."
            )
        data = data[0]

    W = np.asarray(data, dtype=float)
    if W.ndim != 2:
        raise ValueError(f"data must be a 2-D matrix, got shape {W.shape}.")
    if orientation not in _VALID_ORIENTATIONS:
        raise ValueError(
            f"orientation must be one of {_VALID_ORIENTATIONS}, got {orientation!r}."
        )
    if metric not in _VALID_METRICS:
        raise ValueError(f"metric must be one of {_VALID_METRICS}, got {metric!r}.")

    points = W if orientation == "rows" else W.T          # (P, D)
    D = cdist(points, points, metric=metric)

    return GraphBundle(
        matrix=D,
        matrix_type="distance",
        node_features=points if attach_points else None,
        node_labels=None,
        n_nodes=len(D),
        metadata={
            "builder": "point_cloud_graph",
            "orientation": orientation,
            "metric": metric,
            "point_dim": int(points.shape[1]),
            "n_points": int(points.shape[0]),
            "source_matrix_shape": tuple(W.shape),
        },
    )
