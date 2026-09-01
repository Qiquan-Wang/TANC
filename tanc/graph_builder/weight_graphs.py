"""weight_graphs.py — weighted adjacency matrices from NN weights/activations.

Converts any combination of weight matrices and activation matrices into a
``GraphBundle`` with ``matrix_type="similarity"``.

Source papers
-------------
* Watanabe & Yamana (2021): ``edge_weight="relevance"``,
  ``graph_scope="full"``, ``induced_paths=True``, ``max_dim=1``.
* Rieck et al. — Neural Persistence (2019): ``edge_weight="global_normalized"``,
  ``graph_scope="bipartite"``, ``induced_paths=False``, ``max_dim=0``.
  (``"global_normalized"`` divides by the max absolute weight across all
  selected layers, matching the paper's single normalisation constant.)
* Gebhart et al. (2019): ``edge_weight="weighted_activation"``,
  ``graph_scope="multipartite"``, ``induced_paths=False``, ``max_dim=0``.
* Lacombe et al. — Topological Uncertainty (2021): same as Gebhart but
  ``graph_scope="bipartite"`` (one graph per layer) with user-specified
  ``layer_subset``.
"""

from __future__ import annotations

import numpy as np

from tanc.graph_builder._bundle import GraphBundle
from tanc.graph_builder.node_features import compute_node_features


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_weight_graph(
    data: (
        np.ndarray
        | list[np.ndarray]
        | list[tuple[np.ndarray, np.ndarray]]
    ),
    edge_weight: str = "normalized",
    graph_scope: str = "multipartite",
    induced_paths: bool = False,
    layer_subset: list[int] | None = None,
    node_feature_fn: str | callable | None = "laplacian_eigenvectors",
    n_node_features: int = 8,
) -> GraphBundle:
    """Build a weighted adjacency matrix from NN weights and/or activations.

    Parameters
    ----------
    data
        One of:

        * Single ``(N_in, N_out)`` ndarray — one weight matrix.
        * ``list[ndarray]`` — weight matrices, each ``(N_l, M_l)``.
        * ``list[tuple[ndarray, ndarray]]`` — coupled
          ``(weight (N_in, N_out), activation (batch, N_in))`` per layer.
          Required for ``edge_weight="correlation"`` and
          ``edge_weight="weighted_activation"``.

    edge_weight : str
        ``"absolute"`` | ``"normalized"`` | ``"global_normalized"`` |
        ``"relevance"`` | ``"correlation"`` | ``"weighted_activation"``.
        ``"global_normalized"`` divides all layers by the single maximum
        absolute weight across all selected layers (Rieck et al. 2019).
    graph_scope : str
        ``"bipartite"`` | ``"multipartite"`` | ``"full"``
    induced_paths : bool
        When True, add max-product induced-path connections.
        Requires ``graph_scope="full"`` (raises ValueError otherwise).
    layer_subset : list[int] or None
        0-indexed layer indices to include.  None = all layers.
    node_feature_fn : str, callable, or None
        Passed to ``compute_node_features``.  ``None`` → no node features.
    n_node_features : int
        Number of node feature dimensions.

    Returns
    -------
    GraphBundle
        ``matrix_type="similarity"``.
    """
    if induced_paths and graph_scope != "full":
        raise ValueError(
            "induced_paths=True requires graph_scope='full', "
            f"got graph_scope='{graph_scope}'."
        )

    input_type = _infer_input_type(data)
    _check_edge_weight_compat(input_type, edge_weight)

    # Normalise data into list of (weight, activation) tuples
    layers = _normalise_data(data, input_type)

    # Apply layer_subset
    if layer_subset is not None:
        layers = [layers[i] for i in layer_subset]

    if len(layers) == 0:
        raise ValueError("No layers remain after applying layer_subset.")

    # Build per-layer raw weight matrices (before bipartite wrapping)
    per_layer_raw = [_layer_weight_matrix(W, A, edge_weight) for W, A in layers]

    # global_normalized: divide all layers by the single maximum across all layers
    if edge_weight == "global_normalized":
        global_max = max(m.max() for m in per_layer_raw)
        if global_max == 0:
            raise ValueError(
                "edge_weight='global_normalized' but max(|W|) == 0 across "
                "all selected layers.  All weights are zero; cannot normalise."
            )
        per_layer_raw = [m / global_max for m in per_layer_raw]

    # Assemble multi-layer adjacency
    adj = _assemble_adjacency(per_layer_raw, graph_scope, edge_weight)

    # Induced paths — requires layer sizes for DAG-ordered propagation
    if induced_paths:
        layer_sizes = _compute_layer_sizes(per_layer_raw)
        adj = _max_product_paths(adj, layer_sizes)

    N = adj.shape[0]

    # Node features
    node_features = None
    if node_feature_fn is not None:
        node_features = compute_node_features(
            adj, "similarity", node_feature_fn, n_node_features
        )

    return GraphBundle(
        matrix=adj,
        matrix_type="similarity",
        node_features=node_features,
        node_labels=None,
        n_nodes=N,
        metadata={
            "builder": "weight_graph",
            "edge_weight": edge_weight,
            "graph_scope": graph_scope,
            "induced_paths": induced_paths,
            "layer_subset": layer_subset,
            "input_type": input_type,
            "n_layers": len(layers),
            # Raw signed per-layer weight matrices, kept so downstream tools
            # (e.g. the Watanabe directed-clique-complex PH) can recompute
            # relevance with their own sign convention.
            "per_layer_weights": [W for W, _A in layers if W is not None],
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Input inference
# ─────────────────────────────────────────────────────────────────────────────

def _infer_input_type(data) -> str:
    """Return one of ``"single_weight"``, ``"list_weight"``,
    ``"list_coupled"``.

    A plain ``list[ndarray]`` is always treated as weight matrices.
    Activations only enter via the coupled ``list[tuple]`` path.
    """
    if isinstance(data, np.ndarray):
        return "single_weight"
    elif not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            "data must be a numpy ndarray or a non-empty list."
        )
    first = data[0]
    if isinstance(first, tuple):
        return "list_coupled"
    if isinstance(first, np.ndarray):
        return "list_weight"
    raise ValueError(
        f"Cannot infer input type from data element of type {type(first)}."
    )


def _check_edge_weight_compat(input_type: str, edge_weight: str) -> None:
    compat = {
        "single_weight": {"absolute", "normalized", "global_normalized", "relevance"},
        "list_weight": {"absolute", "normalized", "global_normalized", "relevance"},
        "list_coupled": {"absolute", "normalized", "global_normalized", "relevance", "correlation", "weighted_activation"},
    }
    if edge_weight not in compat[input_type]:
        raise ValueError(
            f"edge_weight='{edge_weight}' is incompatible with inferred "
            f"input_type='{input_type}'.  "
            f"Compatible edge_weight values: {compat[input_type]}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Data normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_data(data, input_type: str) -> list[tuple[np.ndarray | None, np.ndarray | None]]:
    """Return list of (weight_matrix_or_None, activation_matrix_or_None)."""
    if input_type == "single_weight":
        return [(np.asarray(data), None)]
    if input_type == "list_weight":
        return [(np.asarray(w), None) for w in data]
    if input_type == "list_coupled":
        return [(np.asarray(w), np.asarray(a)) for w, a in data]
    raise ValueError(f"Unknown input_type '{input_type}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer weight matrix
# ─────────────────────────────────────────────────────────────────────────────

def _layer_weight_matrix(
    W: np.ndarray | None,
    A: np.ndarray | None,
    edge_weight: str,
) -> np.ndarray:
    """Return the raw per-layer weight matrix.

    For weight-based methods (``"absolute"``, ``"normalized"``,
    ``"weighted_activation"``) this is a rectangular ``(N_in, N_out)`` matrix.
    For ``"correlation"`` this is a square ``(N_neurons, N_neurons)`` matrix.
    Bipartite-block wrapping is deferred to the assembly step.
    """
    if edge_weight == "absolute":
        return np.abs(W)

    elif edge_weight == "normalized":
        raw = np.abs(W)
        max_val = raw.max()
        if max_val == 0:
            raise ValueError(
                "edge_weight='normalized' but max(|W|) == 0.  "
                "All weights are zero; cannot normalise."
            )
        return raw / max_val

    elif edge_weight == "global_normalized":
        # Raw absolute values; global normalisation is applied after all
        # layers are collected (in build_weight_graph).
        return np.abs(W)

    elif edge_weight == "relevance":
        # Watanabe & Yamana (2021), Eq. 1: R_ij = w⁺_ij / Σ_k w⁺_kj
        # Positive-part weights, column-normalised per output neuron.
        # Neurons with no positive incoming weights get zero relevance.
        pos = np.maximum(W, 0.0)
        col_sums = pos.sum(axis=0)
        safe_sums = np.where(col_sums == 0, 1.0, col_sums)
        result = pos / safe_sums
        result[:, col_sums == 0] = 0.0
        return result

    elif edge_weight == "correlation":
        # A is (batch, N_neurons); correlation between neurons (columns)
        corr = np.corrcoef(A.T)
        sim = np.abs(corr)
        np.fill_diagonal(sim, 1.0)
        return sim

    elif edge_weight == "weighted_activation":
        # Gebhart et al. (2019): φ(u,v) = |w_{u→v} · h_u|
        # A is pre-synaptic activation (batch, N_in), W is (N_in, N_out)
        # Average over the batch to get a single representative graph.
        mean_act = A.mean(axis=0)   # (N_in,)
        return np.abs(W * mean_act[:, None])

    else:
        raise ValueError(
            f"Unknown edge_weight '{edge_weight}'. "
            "Valid: 'absolute', 'normalized', 'global_normalized', 'relevance', "
            "'correlation', 'weighted_activation'."
        )


def _bipartite_block(W: np.ndarray) -> np.ndarray:
    """Convert (N_in, N_out) weight matrix into square (N_in+N_out, N_in+N_out)
    bipartite adjacency block."""
    N_in, N_out = W.shape
    N = N_in + N_out
    adj = np.zeros((N, N))
    adj[:N_in, N_in:] = W
    adj[N_in:, :N_in] = W.T
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Multi-layer assembly
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_adjacency(
    per_layer_raw: list[np.ndarray],
    graph_scope: str,
    edge_weight: str,
) -> np.ndarray:
    """Assemble per-layer raw weight matrices into one adjacency matrix.

    ``per_layer_raw`` contains raw weight matrices: rectangular ``(N_in, N_out)``
    for weight-based methods, or square ``(N, N)`` for ``"correlation"``.

    Scopes
    ------
    ``"bipartite"``
        Block-diagonal: each layer forms its own bipartite block.
    ``"multipartite"``
        All layers share a common node set; only consecutive layers connected.
    ``"full"``
        Same as multipartite; matrix is ready for induced-path extension.
    """
    if len(per_layer_raw) == 1:
        r = per_layer_raw[0]
        return _bipartite_block(r) if edge_weight != "correlation" else r

    if graph_scope == "bipartite":
        if edge_weight == "correlation":
            return _block_diagonal(per_layer_raw)
        return _block_diagonal([_bipartite_block(r) for r in per_layer_raw])

    elif graph_scope in ("multipartite", "full"):
        if edge_weight == "correlation":
            # Correlation matrices are per-layer square; fall back to block-diagonal
            return _block_diagonal(per_layer_raw)
        return _multipartite_assembly(per_layer_raw)

    else:
        raise ValueError(
            f"Unknown graph_scope '{graph_scope}'. "
            "Valid: 'bipartite', 'multipartite', 'full'."
        )


def _block_diagonal(blocks: list[np.ndarray]) -> np.ndarray:
    """Stack square matrices along the diagonal."""
    total = sum(b.shape[0] for b in blocks)
    result = np.zeros((total, total))
    offset = 0
    for b in blocks:
        n = b.shape[0]
        result[offset:offset + n, offset:offset + n] = b
        offset += n
    return result


def _multipartite_assembly(per_layer_raw: list[np.ndarray]) -> np.ndarray:
    """Build a single adjacency matrix where nodes are ordered by layer.

    Parameters
    ----------
    per_layer_raw : list of (N_l, N_{l+1}) ndarrays
        Raw (non-negative) weight matrices, one per consecutive layer pair.
    """
    # Determine layer sizes directly from raw weight matrix shapes.
    layer_sizes: list[int] = [per_layer_raw[0].shape[0]]
    for i, bw in enumerate(per_layer_raw):
        if i > 0 and layer_sizes[-1] != bw.shape[0]:
            raise ValueError(
                f"Inconsistent layer sizes between consecutive weight matrices: "
                f"layer {i-1} output size {layer_sizes[-1]} != "
                f"layer {i} input size {bw.shape[0]}."
            )
        layer_sizes.append(bw.shape[1])

    total_nodes = sum(layer_sizes)
    adj = np.zeros((total_nodes, total_nodes))

    # Fill inter-layer edges
    offsets = [sum(layer_sizes[:i]) for i in range(len(layer_sizes))]
    for l_idx, bw in enumerate(per_layer_raw):
        r0 = offsets[l_idx]
        c0 = offsets[l_idx + 1]
        N_in, N_out = bw.shape
        adj[r0:r0 + N_in, c0:c0 + N_out] = bw
        adj[c0:c0 + N_out, r0:r0 + N_in] = bw.T

    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Induced paths (Watanabe & Yamana, 2021)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_layer_sizes(per_layer_raw: list[np.ndarray]) -> list[int]:
    """Derive layer sizes from per-layer rectangular weight matrices."""
    sizes = [per_layer_raw[0].shape[0]]
    for bw in per_layer_raw:
        sizes.append(bw.shape[1])
    return sizes


def _max_product_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Max-product matrix multiplication: C[i,k] = max_j A[i,j] * B[j,k]."""
    # A: (M, J), B: (J, K)  →  C: (M, K)
    return np.max(A[:, :, np.newaxis] * B[np.newaxis, :, :], axis=1)


def _max_product_paths(
    adj: np.ndarray,
    layer_sizes: list[int],
) -> np.ndarray:
    """Add edges between all non-consecutive layer nodes via max-product paths.

    For each pair (i, k) that are not directly connected, compute
    (Watanabe & Yamana 2021, Eq. 2):

        R̃_ik = max over all paths (i → m₁ → … → mₖ → k):
                    R_{i,m₁} · R_{m₁,m₂} · … · R_{mₖ,k}

    Paths respect the feedforward layer ordering: a path from layer *a*
    to layer *b* passes through layers a, a+1, …, b in order (each
    layer visited exactly once).  This is enforced by chaining the
    per-layer weight blocks with max-product matrix multiplication
    rather than generic all-pairs iteration.

    Parameters
    ----------
    adj : (N, N) ndarray
        Existing multipartite adjacency (non-negative similarities).
    layer_sizes : list[int]
        Number of neurons in each layer (length = number of layers).

    Returns
    -------
    (N, N) ndarray with induced-path weights added.
    """
    n_layers = len(layer_sizes)
    offsets = [sum(layer_sizes[:i]) for i in range(n_layers)]
    best = adj.copy()

    # For each starting layer, chain max-product paths forward through
    # successive layers.
    for i in range(n_layers - 2):
        ri = offsets[i]
        ei = ri + layer_sizes[i]

        # M accumulates the max-product path from layer i to the
        # current frontier layer.  Start with direct edges to layer i+1.
        rj = offsets[i + 1]
        ej = rj + layer_sizes[i + 1]
        M = best[ri:ei, rj:ej].copy()          # (size_i, size_{i+1})

        # Extend one layer at a time: i → i+2, i → i+3, …
        for k in range(i + 2, n_layers):
            rk = offsets[k]
            ek = rk + layer_sizes[k]

            # Direct edges from the previous frontier (k-1) to layer k
            rprev = offsets[k - 1]
            eprev = rprev + layer_sizes[k - 1]
            R_prev_k = best[rprev:eprev, rk:ek]    # (size_{k-1}, size_k)

            M = _max_product_multiply(M, R_prev_k)  # (size_i, size_k)

            # Fill the induced-path weights into the adjacency
            best[ri:ei, rk:ek] = M
            best[rk:ek, ri:ei] = best[ri:ei, rk:ek].T

    np.fill_diagonal(best, 0.0)
    return best
