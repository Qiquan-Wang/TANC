"""Topological Uncertainty (Lacombe et al. 2021).

Provides the per-sample monitoring score from Lacombe et al. 2021 -- a
post-training scalar that flags inputs the network responds to in an
unusual way, even when the softmax confidence is high.

Public API
----------
* ``bipartite_mst_diagram(W, x)`` -- sorted MST edge weights of the
  bipartite graph with edge weight ``|W[i, j] * x[i]|``.  This is the
  paper's persistence diagram in the superlevel-set construction
  (Section 2.2 + Appendix A.1).
* ``frechet_mean_diagram(diagrams)`` -- elementwise mean of equal-length
  sorted diagrams (the L2-Wasserstein Frechet mean for 1-D MST
  diagrams; Section 2.2 + Appendix A.2).
* ``diagram_l2_distance(d1, d2)`` -- the metric ``Dist`` used by TU.
* ``TopologicalUncertainty`` -- streaming ``partial_fit`` / ``score``
  monitor that accumulates per-(layer, class) Frechet means from
  training data, then scores new inputs.

Implementation notes
--------------------
* The MST is computed via ``scipy.sparse.csgraph.minimum_spanning_tree``
  on a sparse CSR adjacency, so large layers (e.g. ``9216 x 128`` =
  ~1.2 M edges) are tractable without instantiating the ~700 MB dense
  matrix.
* Streaming accumulators (``partial_fit``) hold only one sorted vector
  per ``(layer, class)`` pair, never the full per-sample diagrams, so
  fitting on the full MNIST training set (60 k samples) is memory-flat.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


# ---------------------------------------------------------------------------
# Functional primitives
# ---------------------------------------------------------------------------

def bipartite_mst_diagram(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Sorted (descending) MST edge weights of the complete bipartite
    graph with edge weight ``|W[i, j] * x[i]|``.

    This is the persistence diagram in the superlevel-set construction
    used by Lacombe et al. (2021) for the activation graph of a single
    layer ``l`` and a single observation ``x``.  The returned vector
    has length ``N_in + N_out - 1`` and *is* the diagram
    ``D_l(x, F)`` in the paper's notation.

    Parameters
    ----------
    W : (N_in, N_out) ndarray
        Layer weight matrix.
    x : (N_in,) ndarray
        Pre-synaptic activation feeding into the layer.

    Returns
    -------
    (N_in + N_out - 1,) ndarray
        Sorted descending MST edge weights.

    Notes
    -----
    Uses sparse CSR adjacency so the call costs ~scales linearly with
    ``N_in * N_out`` instead of building a dense ``(N_in + N_out)^2``
    matrix.
    """
    W = np.asarray(W)
    x = np.asarray(x)
    if W.ndim != 2 or x.ndim != 1 or W.shape[0] != x.shape[0]:
        raise ValueError(
            f"Expected W shape (N_in, N_out) and x shape (N_in,); "
            f"got W {W.shape}, x {x.shape}."
        )
    N_in, N_out = W.shape
    edge_w = np.abs(W * x[:, None]).ravel()
    if edge_w.size == 0:
        return np.empty(0, dtype=float)
    max_w = float(edge_w.max())
    big = max_w + 1.0 if max_w > 0 else 1.0
    shifted = big - edge_w
    row_idx = np.repeat(np.arange(N_in), N_out)
    col_idx = N_in + np.tile(np.arange(N_out), N_in)
    N = N_in + N_out
    M = coo_matrix((shifted, (row_idx, col_idx)), shape=(N, N)).tocsr()
    mst = minimum_spanning_tree(M)
    actual = big - mst.data
    return np.sort(actual)[::-1]


def frechet_mean_diagram(diagrams) -> np.ndarray:
    """Elementwise mean of equal-length sorted persistence diagrams.

    For 1-D persistence diagrams with the same number of points -- which
    is the case for MST diagrams of fixed-size bipartite graphs -- the
    L2-Wasserstein Frechet mean reduces to the elementwise mean of the
    sorted edge-weight vectors (Lacombe et al. 2021 Section 2.2;
    originally Turner et al. 2014).

    Parameters
    ----------
    diagrams : list[(N,) ndarray] or (M, N) ndarray
        A non-empty collection of equal-length sorted diagrams.

    Returns
    -------
    (N,) ndarray
    """
    arr = np.asarray(diagrams)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(
            f"diagrams must be a non-empty 2-D array; got shape {arr.shape}."
        )
    return arr.mean(axis=0)


def diagram_l2_distance(d1: np.ndarray, d2: np.ndarray) -> float:
    """L2 distance between two equal-length sorted persistence diagrams.

    Implements the metric ``Dist`` defined in Lacombe et al. (2021)
    Section 2.2 / Appendix A.2:

    .. math::

        \\mathrm{Dist}(\\mu, \\nu)^2 = \\frac{1}{N} \\sum_i (w_i - w'_i)^2

    where ``w``, ``w'`` are the sorted edge-weight vectors of the two
    diagrams.

    Parameters
    ----------
    d1, d2 : (N,) ndarray
        Two equal-length sorted diagrams.

    Returns
    -------
    float
    """
    d1 = np.asarray(d1)
    d2 = np.asarray(d2)
    if d1.shape != d2.shape:
        raise ValueError(
            f"diagrams must have the same shape; got {d1.shape} and {d2.shape}."
        )
    return float(np.sqrt(np.mean((d1 - d2) ** 2)))


# ---------------------------------------------------------------------------
# TopologicalUncertainty monitor
# ---------------------------------------------------------------------------

class TopologicalUncertainty:
    """Per-sample monitoring score from Lacombe et al. (2021).

    Fits per-(layer, predicted-class) Frechet-mean diagrams from
    training activations and predictions, then scores new inputs by
    their average per-layer L2 distance to the mean of their predicted
    class.

    Streaming-friendly: ``partial_fit`` can be called repeatedly on
    different batches and produces the same Frechet means as a single
    call on the concatenated data.

    Parameters
    ----------
    weight_matrices : list[ndarray]
        Per-layer weight matrices ``[W_1, ..., W_L]``.  ``W_l`` has
        shape ``(N_in_l, N_out_l)``.
    n_classes : int
        Number of network output classes.

    Examples
    --------
    >>> from tanc.topo_tools import TopologicalUncertainty
    >>>
    >>> tu = TopologicalUncertainty(
    ...     weight_matrices=[W_fc1, W_fc2],
    ...     n_classes=10,
    ... )
    >>>
    >>> # Streaming fit on training data
    >>> for X_batch in train_batches:
    ...     activations_per_layer = extract_pre_activations(X_batch)
    ...     predictions = network_predictions(X_batch)
    ...     tu.partial_fit(activations_per_layer, predictions)
    >>>
    >>> # Score new inputs
    >>> scores = tu.score(test_activations_per_layer, test_predictions)

    Notes
    -----
    TU is defined as

    .. math::

        \\mathrm{TU}(x, F) = \\frac{1}{L} \\sum_{l=1}^{L}
            \\mathrm{Dist}\\!\\left( D_l(x, F),
                                     \\overline{D^{\\mathrm{train}}_{l, k(x)}} \\right),

    where ``k(x)`` is the network's predicted class for ``x`` and
    ``D_l(x, F)`` is the MST diagram of layer ``l``'s activation graph
    for input ``x``.
    """

    def __init__(self, weight_matrices, n_classes: int) -> None:
        weight_matrices = [np.asarray(W) for W in weight_matrices]
        if not weight_matrices:
            raise ValueError("weight_matrices must be non-empty.")
        for l, W in enumerate(weight_matrices):
            if W.ndim != 2:
                raise ValueError(
                    f"weight_matrices[{l}] must be 2-D; got shape {W.shape}."
                )
        n_classes = int(n_classes)
        if n_classes <= 0:
            raise ValueError("n_classes must be > 0.")

        self.weight_matrices = weight_matrices
        self.n_classes = n_classes
        self._L = len(weight_matrices)
        self._diag_lengths = [
            W.shape[0] + W.shape[1] - 1 for W in weight_matrices
        ]
        self._sums = [
            np.zeros((n_classes, n), dtype=np.float64)
            for n in self._diag_lengths
        ]
        self._counts = np.zeros(n_classes, dtype=np.int64)

    # ----- fitting -----

    def partial_fit(self, activations_per_layer, predictions):
        """Accumulate per-class means from one batch.

        Parameters
        ----------
        activations_per_layer : sequence of (B, N_in_l) ndarrays
            Pre-synaptic activations entering each of the ``L`` layers,
            for the same batch of ``B`` inputs.
        predictions : (B,) array-like of int
            Network predicted class for each input.

        Returns
        -------
        self
        """
        self._check_activations(activations_per_layer)
        predictions = np.asarray(predictions, dtype=int)
        B = predictions.shape[0]
        for a in activations_per_layer:
            if a.shape[0] != B:
                raise ValueError(
                    "All activation arrays must have the same batch size as predictions."
                )
        if B > 0 and (predictions.min() < 0 or predictions.max() >= self.n_classes):
            raise ValueError(
                f"predictions out of range [0, {self.n_classes - 1}]."
            )
        for i in range(B):
            k = int(predictions[i])
            for l in range(self._L):
                d = bipartite_mst_diagram(
                    self.weight_matrices[l], activations_per_layer[l][i]
                )
                self._sums[l][k] += d
            self._counts[k] += 1
        return self

    def fit(self, activations_per_layer, predictions):
        """Convenience alias for ``partial_fit`` on a single batch."""
        return self.partial_fit(activations_per_layer, predictions)

    # ----- means -----

    @property
    def mean_diagrams(self) -> dict:
        """Dict ``(layer_idx, class_idx) -> Frechet-mean diagram``.

        Classes that have not seen any training sample yet map to a
        zero diagram of the appropriate length.
        """
        means: dict = {}
        for l in range(self._L):
            for k in range(self.n_classes):
                if self._counts[k] > 0:
                    means[(l, k)] = self._sums[l][k] / self._counts[k]
                else:
                    means[(l, k)] = self._sums[l][k].copy()
        return means

    @property
    def class_counts(self) -> np.ndarray:
        """Per-class training sample count (1-D array of length ``n_classes``)."""
        return self._counts.copy()

    # ----- scoring -----

    def score(self, activations_per_layer, predictions) -> np.ndarray:
        """Compute Topological Uncertainty for a batch.

        Parameters
        ----------
        activations_per_layer : sequence of (B, N_in_l) ndarrays
        predictions : (B,) array-like of int

        Returns
        -------
        (B,) ndarray of TU scores -- higher means more anomalous.
        """
        if int(self._counts.sum()) == 0:
            raise RuntimeError(
                "TopologicalUncertainty has not been fit yet; call partial_fit first."
            )
        self._check_activations(activations_per_layer)
        predictions = np.asarray(predictions, dtype=int)
        B = predictions.shape[0]
        for a in activations_per_layer:
            if a.shape[0] != B:
                raise ValueError(
                    "All activation arrays must have the same batch size as predictions."
                )
        if B > 0 and (predictions.min() < 0 or predictions.max() >= self.n_classes):
            raise ValueError(
                f"predictions out of range [0, {self.n_classes - 1}]."
            )
        means = self.mean_diagrams
        out = np.zeros(B)
        for i in range(B):
            k = int(predictions[i])
            total = 0.0
            for l in range(self._L):
                d = bipartite_mst_diagram(
                    self.weight_matrices[l], activations_per_layer[l][i]
                )
                total += diagram_l2_distance(d, means[(l, k)])
            out[i] = total / self._L
        return out

    # ----- helpers -----

    def _check_activations(self, activations_per_layer) -> None:
        if len(activations_per_layer) != self._L:
            raise ValueError(
                f"Expected {self._L} activation arrays (one per layer); "
                f"got {len(activations_per_layer)}."
            )
        for l, a in enumerate(activations_per_layer):
            a = np.asarray(a)
            expected_in = self.weight_matrices[l].shape[0]
            if a.ndim != 2 or a.shape[1] != expected_in:
                raise ValueError(
                    f"activations_per_layer[{l}] must have shape (B, {expected_in}); "
                    f"got {a.shape}."
                )

    def __repr__(self) -> str:
        sizes = " -> ".join(
            f"{W.shape[0]}x{W.shape[1]}" for W in self.weight_matrices
        )
        n_fit = int(self._counts.sum())
        return (
            f"TopologicalUncertainty(layers=[{sizes}], "
            f"n_classes={self.n_classes}, n_train={n_fit})"
        )


__all__ = [
    "bipartite_mst_diagram",
    "frechet_mean_diagram",
    "diagram_l2_distance",
    "TopologicalUncertainty",
]
