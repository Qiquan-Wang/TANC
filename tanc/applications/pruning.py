"""pruning.py — persistent-homology network pruning (PHPM) + GMP baseline.

Reproduces Watanabe & Yamana, "Deep Neural Network Pruning Using Persistent
Homology" (PHPM, AIKE 2020).  PHPM prunes a fully-connected sub-network by
*keeping the edges that carry its strongest H1 loops* and zeroing the rest:

1. Build the directed clique complex of the FCN and its H1 diagram
   (:func:`tanc.topo_tools.clique_complex.compute_watanabe_ph`).
2. Sort the H1 classes by ``birth + death`` (ascending = strongest combinational
   effects first), and walk them, keeping every real weight-edge touched by each
   class's **birth edge** and **death triangle**, until the target number of
   kept edges is reached.  A magnitude fallback tops up the keep-set so the
   requested ratio is always met.
3. Zero the un-kept weights.

The classic **global magnitude pruning (GMP)** baseline is provided for
comparison, plus helpers to apply masks, diagnose disconnected neurons, and
score accuracy.

All functions operate on a plain ``list`` of consecutive ``(N_in, N_out)``
weight matrices — the FCN head of any framework's model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tanc.topo_tools._result import TopoResult


# ─────────────────────────────────────────────────────────────────────────────
# Map a simplex of the clique complex back to real weight edges
# ─────────────────────────────────────────────────────────────────────────────

def _offsets(layer_sizes: list[int]) -> list[int]:
    return [sum(layer_sizes[:l]) for l in range(len(layer_sizes))]


def _locate(gid: int, offsets: list[int]) -> tuple[int, int]:
    for l in range(len(offsets) - 1, -1, -1):
        if gid >= offsets[l]:
            return l, gid - offsets[l]
    return 0, gid


def simplex_to_real_edges(
    simplex: tuple[int, ...],
    rel_matrices: list[np.ndarray],
    layer_sizes: list[int],
) -> list[tuple[int, int, int]]:
    """Real weight edges ``(layer, i, j)`` underlying a complex simplex.

    * A **direct edge** between consecutive layers maps to that one weight.
    * An **induced edge** between layers two apart maps to the two weights of
      its strongest 2-hop path (the same ``argmax`` path that gave its birth).
    * A **triangle** maps to its two real weights (input→mid, mid→output).
    """
    offsets = _offsets(layer_sizes)
    nodes = sorted((_locate(v, offsets) for v in simplex), key=lambda t: t[0])
    out: list[tuple[int, int, int]] = []

    if len(nodes) == 2:
        (la, ia), (lb, ib) = nodes
        if lb == la + 1:                                   # direct edge
            out.append((la, ia, ib))
        elif lb == la + 2:                                 # induced edge → best path
            j = int((rel_matrices[la][ia, :] * rel_matrices[la + 1][:, ib]).argmax())
            out += [(la, ia, j), (la + 1, j, ib)]
    elif len(nodes) == 3:                                  # triangle i-j-k
        (la, i), (_, j), (_, k) = nodes
        out += [(la, i, j), (la + 1, j, k)]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PHPM mask
# ─────────────────────────────────────────────────────────────────────────────

def phpm_pruning_mask(
    weight_matrices: list[np.ndarray],
    ph_result: TopoResult,
    pruning_ratio: float,
    verbose: bool = False,
) -> list[np.ndarray]:
    """PHPM keep-masks (``True`` = keep) for each weight matrix.

    Parameters
    ----------
    weight_matrices : list of ``(N_in, N_out)`` FCN weight matrices.
    ph_result : TopoResult
        Output of ``compute_watanabe_ph(weight_matrices, ...)`` — carries the
        H1 representatives, layout and relevance matrices in
        ``ph_result.ph_result.metadata["watanabe"]``.
    pruning_ratio : float
        Fraction of edges to remove, ``(m − n) / m`` in the paper.

    Returns
    -------
    list[np.ndarray] of boolean keep-masks, one per weight matrix.
    """
    meta = ph_result.ph_result.metadata["watanabe"]
    reps = meta["h1_representatives"]
    rels = meta["relevance_matrices"]
    sizes = meta["layer_sizes"]

    total = sum(W.size for W in weight_matrices)
    n_keep = int(round((1.0 - pruning_ratio) * total))

    keep: set[tuple[int, int, int]] = set()
    # Strongest loops first: ascending birth + death.
    for b, d, s_birth, s_death in sorted(reps, key=lambda t: t[0] + t[1]):
        if len(keep) >= n_keep:
            break
        for e in simplex_to_real_edges(s_birth, rels, sizes):
            keep.add(e)
        for e in simplex_to_real_edges(s_death, rels, sizes):
            keep.add(e)

    # Magnitude fallback: top up with the largest-|w| remaining edges.
    if len(keep) < n_keep:
        ranked = sorted(
            ((l, i, j, abs(W[i, j]))
             for l, W in enumerate(weight_matrices)
             for i in range(W.shape[0]) for j in range(W.shape[1])),
            key=lambda e: -e[3],
        )
        for l, i, j, _ in ranked:
            if len(keep) >= n_keep:
                break
            keep.add((l, i, j))

    masks = [np.zeros(W.shape, dtype=bool) for W in weight_matrices]
    for l, i, j in keep:
        masks[l][i, j] = True

    if verbose:
        kept = sum(int(m.sum()) for m in masks)
        print(f"  PHPM kept {kept} edges → ratio {1 - kept / total:.3f} "
              f"(target {pruning_ratio})")
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# GMP baseline
# ─────────────────────────────────────────────────────────────────────────────

def _mag_keep(W: np.ndarray, ratio: float) -> np.ndarray:
    k = int(round(ratio * W.size))
    if k <= 0:
        return np.ones(W.shape, dtype=bool)
    thr = np.partition(np.abs(W).ravel(), k - 1)[k - 1]
    return np.abs(W) > thr


def gmp_pruning_mask(
    weight_matrices: list[np.ndarray],
    pruning_ratio: float,
    per_layer: bool = False,
) -> list[np.ndarray]:
    """Global magnitude pruning keep-masks (``True`` = keep).

    ``per_layer=False`` (paper's wording) ranks every weight together and
    removes the globally smallest ``|w|`` — which can wipe out the small output
    matrix and disconnect classes.  ``per_layer=True`` removes the same fraction
    *within* each matrix (the robust variant most reproductions use).
    """
    if per_layer:
        return [_mag_keep(W, pruning_ratio) for W in weight_matrices]

    total = sum(W.size for W in weight_matrices)
    k = int(round(pruning_ratio * total))
    if k <= 0:
        return [np.ones(W.shape, dtype=bool) for W in weight_matrices]
    allw = np.concatenate([np.abs(W).ravel() for W in weight_matrices])
    thr = np.partition(allw, k - 1)[k - 1]
    return [np.abs(W) > thr for W in weight_matrices]


# ─────────────────────────────────────────────────────────────────────────────
# Apply / diagnose / evaluate
# ─────────────────────────────────────────────────────────────────────────────

def apply_masks(
    weight_matrices: list[np.ndarray],
    masks: list[np.ndarray],
) -> list[np.ndarray]:
    """Return copies of the weights with un-kept entries zeroed."""
    return [W * m for W, m in zip(weight_matrices, masks)]


def prune_diagnostics(masks: list[np.ndarray]) -> dict[str, Any]:
    """Summarise a keep-mask set: kept/total per layer and dead target neurons.

    A target neuron is 'dead' when its weight-matrix column is all-zero, i.e.
    it receives no incoming signal after pruning (for the last matrix this
    means a class becomes unpredictable).
    """
    total = sum(m.size for m in masks)
    kept = sum(int(m.sum()) for m in masks)
    dead = [int((m.sum(axis=0) == 0).sum()) for m in masks]
    return {
        "kept": kept,
        "total": total,
        "achieved_ratio": 1 - kept / total,
        "kept_per_layer": [int(m.sum()) for m in masks],
        "dead_target_neurons_per_layer": dead,
    }


def fcn_accuracy(
    forward,
    weight_matrices: list[np.ndarray],
    masks: list[np.ndarray] | None,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Top-1 accuracy of ``forward(X, weights)`` with optional masking applied.

    ``forward`` is a callable ``(X, weight_matrices) -> logits`` so this module
    stays framework-agnostic; pass the masked weights' effect via ``masks``.
    """
    w = apply_masks(weight_matrices, masks) if masks is not None else weight_matrices
    logits = forward(X, w)
    return float((np.asarray(logits).argmax(axis=1) == np.asarray(y)).mean())
