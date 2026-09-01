"""_activations.py — framework-neutral activation pooling.

A captured layer output is reduced to a ``(N_samples, features)`` cloud before
it enters the snapshot.  ``pool_activation`` implements the reduction so the
PyTorch and TensorFlow extractors share identical semantics.

Pure numpy — no framework import — so both backends can use it freely.
"""

from __future__ import annotations

import numpy as np

# Valid ``activation_pooling`` values, in documentation order.
ACTIVATION_POOLINGS = ("flatten", "last", "mean", "max", "first")


def pool_activation(act: np.ndarray, pooling: str) -> np.ndarray:
    """Reduce a captured activation tensor to a ``(N_samples, features)`` cloud.

    ``pooling`` controls how the non-sample axes collapse:

    * ``"flatten"`` (default) — reshape ``(N, …) → (N, -1)``; the historical
      behaviour, and the only one applied to 2-D or >3-D tensors, so CNN/MLP
      extraction is unchanged.
    * ``"last"`` / ``"first"`` / ``"mean"`` / ``"max"`` — **token pooling** for
      transformer-style ``(batch, seq_len, hidden)`` activations: reduce over the
      *sequence* axis, keeping ``hidden``.  ``"last"`` takes the final token (a
      summary of the whole sequence for a causal decoder); ``"first"`` the first
      (≈ BERT ``[CLS]``); ``"mean"`` / ``"max"`` pool across all tokens.

    The token-pool modes assume the ``(batch, seq, hidden)`` layout and only
    apply to 3-D tensors; anything else falls back to ``"flatten"``.
    """
    act = np.asarray(act)
    if act.ndim <= 2:
        return act
    if pooling == "flatten":
        return act.reshape(act.shape[0], -1)
    if act.ndim == 3:
        if pooling == "last":
            return act[:, -1, :]
        if pooling == "first":
            return act[:, 0, :]
        if pooling == "mean":
            return act.mean(axis=1)
        if pooling == "max":
            return act.max(axis=1)
    # >3-D (e.g. conv feature maps): token pooling is undefined → flatten.
    return act.reshape(act.shape[0], -1)
