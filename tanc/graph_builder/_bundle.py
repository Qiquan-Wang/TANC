"""GraphBundle — the contract object flowing from Module 1 to Module 2."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tanc._serialization import SaveLoadMixin


@dataclass
class GraphBundle(SaveLoadMixin):
    """Standardised container produced by all graph_builder functions.

    Parameters
    ----------
    matrix : (N, N) ndarray
        Square matrix encoding graph structure.
    matrix_type : str
        ``"distance"`` | ``"similarity"`` | ``"adjacency"``.
        Controls how Module 2 interprets and filtrates the matrix.
    node_features : (N, d) ndarray or None
        Per-node feature vectors.  Required by the Mapper tool; optional
        for PH.
    node_labels : (N,) ndarray or None
        Integer class labels per node, if available.
    n_nodes : int
        Number of nodes; must equal ``matrix.shape[0]``.
    metadata : dict
        Provenance record: builder name, builder params, input shapes.

    Matrix-type semantics
    ---------------------
    ``"distance"``
        Entries are distances; 0 on the diagonal.  Passed directly to
        ripser with ``distance_matrix=True``.
    ``"similarity"``
        Entries are similarities; maximum on the diagonal.  Converted via
        superlevel-set filtration (``max - d``) before PH.
    ``"adjacency"``
        Unweighted 0/1 graph.  Converted via ``1 - A`` (non-edges become
        distance 1) before PH.
    """

    matrix: np.ndarray
    matrix_type: str
    node_features: np.ndarray | None
    node_labels: np.ndarray | None
    n_nodes: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError(
                f"matrix must be 2-dimensional, got ndim={self.matrix.ndim}."
            )
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"matrix must be square, got shape {self.matrix.shape}."
            )
        if self.n_nodes != self.matrix.shape[0]:
            raise ValueError(
                f"n_nodes={self.n_nodes} does not match matrix.shape[0]="
                f"{self.matrix.shape[0]}."
            )
        valid_types = {"distance", "similarity", "adjacency"}
        if self.matrix_type not in valid_types:
            raise ValueError(
                f"matrix_type must be one of {valid_types}, "
                f"got '{self.matrix_type}'."
            )
        if self.node_features is not None:
            if self.node_features.shape[0] != self.n_nodes:
                raise ValueError(
                    f"node_features.shape[0]={self.node_features.shape[0]} "
                    f"does not match n_nodes={self.n_nodes}."
                )
        if self.node_labels is not None:
            if self.node_labels.shape[0] != self.n_nodes:
                raise ValueError(
                    f"node_labels.shape[0]={self.node_labels.shape[0]} "
                    f"does not match n_nodes={self.n_nodes}."
                )
