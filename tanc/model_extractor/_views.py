"""_views.py — turning a weight or activation matrix into a point cloud.

A single matrix admits many readings, and which one carries the structure is
usually not known in advance.  A convolutional kernel ``W`` of shape
``(freq, time)`` can be one point (flattened), one point per row, one point per
column, a profile of row norms, or its own Gram matrix.  Each is a different
question about the same weights, and each produces a differently shaped cloud.

This module names those readings so they can be swept as a parameter rather
than hand-written per experiment.  The same names apply to activation matrices
``(n_samples, n_units)``, where ``rows`` are samples, ``cols`` are units, and
``gram_cols`` is the unit covariance — a standard object in network analysis.

Every view returns a 2-D ``(n_points, n_features)`` array, including views that
yield a single point, so results stack unambiguously across a population.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["VIEWS", "CONV_VIEWS", "matrix_view", "conv_view", "conv_view_shape",
           "view_shape", "describe_view", "stack_views"]

_EPS = 1e-12

#: Short descriptions of each view, keyed by name.  ``matrix_view`` accepts any
#: of these; anything else raises with this mapping in the message.
VIEWS: dict[str, str] = {
    "full":      "the whole matrix as one point (flattened)",
    "rows":      "one point per row",
    "cols":      "one point per column",
    "row_norm":  "one point: the L2 norm of each row",
    "col_norm":  "one point: the L2 norm of each column",
    "row_sum":   "one point: the signed sum along each row",
    "col_sum":   "one point: the signed sum along each column",
    "gram_rows": "Gram matrix of the rows, M @ M.T",
    "gram_cols": "Gram matrix of the columns, M.T @ M",
    "gram_diag": "mean of each Gram diagonal — the lag or interval profile",
}

_GRAM_VIEWS = ("gram_rows", "gram_cols")

#: Views for a convolutional weight ``(out, in, h, w)``.  A 4-D tensor has no
#: single "row", so each view names **which axes index points**; the remaining
#: axes are flattened into that point's coordinates.  Conv1d is promoted to
#: ``(out, in, 1, w)`` on extraction, so the same names apply with ``h = 1``.
CONV_VIEWS: dict[str, str] = {
    "rows":        "one point per kernel row      (points: out,in,h  dim: w)",
    "cols":        "one point per kernel column   (points: out,in,w  dim: h)",
    "kernel":      "one point per (out,in) kernel (points: out,in    dim: h*w)",
    "out_channel": "one point per output channel  (points: out       dim: in*h*w)",
    "in_channel":  "one point per input channel   (points: in        dim: out*h*w)",
    "tap":         "one point per spatial position(points: h,w       dim: out*in)",
}

# ``full`` is deliberately NOT a conv view.  Its meaning on a convolutional layer
# already depends on ``per_filter`` — one point per filter by default, one point
# per layer when False — and routing it here would silently change results for
# anyone already using it.  ``out_channel`` is the explicit name for the default
# reading, and ``per_filter=False`` remains the way to get one point per layer.


def conv_view(W: np.ndarray, view: str = "kernel") -> np.ndarray:
    """Read a convolutional weight tensor as a point cloud.

    A ``(out, in, h, w)`` tensor admits several readings and "row" alone does
    not name one, so every view here says which axes index points.  ``rows``
    and ``cols`` are the rows and columns **of the kernel itself**, which is the
    reading that matches how a kernel is drawn and discussed.

    Parameters
    ----------
    W : (out, in, h, w) ndarray
        A convolutional weight.  Conv1d weights are promoted to
        ``(out, in, 1, w)`` at extraction, so they arrive 4-D too.
    view : str
        A key of :data:`CONV_VIEWS`.

    Returns
    -------
    (n_points, n_features) ndarray

    Raises
    ------
    ValueError
        Unknown *view*, a tensor that is not 4-D, or a view that would give
        points of dimension 1 — a 1-D cloud carries no geometry worth a
        Mapper graph or a persistence diagram, and the failure is otherwise
        silent.

    Examples
    --------
    >>> W = np.zeros((16, 1, 5, 5))              # Conv2d(1, 16, kernel_size=5)
    >>> conv_view(W, "rows").shape               # 16 kernels x 5 rows each
    (80, 5)
    >>> conv_view(W, "kernel").shape
    (16, 25)
    >>> conv_view(W, "out_channel").shape
    (16, 25)
    """
    A = np.asarray(W, dtype=np.float64)
    if A.ndim == 3:                       # tolerate an un-promoted Conv1d
        A = A[:, :, None, :]
    if A.ndim != 4:
        raise ValueError(
            f"conv_view needs a 4-D (out, in, h, w) tensor; got shape {A.shape}. "
            f"For a 2-D weight use matrix_view instead."
        )
    if view not in CONV_VIEWS:
        raise ValueError(
            f"Unknown conv view {view!r}. Known views:\n  "
            + "\n  ".join(f"{k:<12} {v}" for k, v in CONV_VIEWS.items())
        )

    o, i, h, w = A.shape
    if view == "rows":
        out = A.reshape(o * i * h, w)
    elif view == "cols":
        out = A.transpose(0, 1, 3, 2).reshape(o * i * w, h)
    elif view == "kernel":
        out = A.reshape(o * i, h * w)
    elif view == "out_channel":
        out = A.reshape(o, i * h * w)
    elif view == "in_channel":
        out = A.transpose(1, 0, 2, 3).reshape(i, o * h * w)
    else:                                  # "tap"
        out = A.transpose(2, 3, 0, 1).reshape(h * w, o * i)

    if out.shape[1] < 2:
        alt = "kernel" if view in ("rows", "cols") else "out_channel"
        raise ValueError(
            f"view={view!r} on a weight of shape {A.shape} gives points of "
            f"dimension {out.shape[1]}, which has no geometry to analyse. This "
            f"happens when the axis being kept is a singleton — a Conv1d kernel "
            f"has h = 1, so 'cols' degenerates. Try {alt!r} instead, or a view "
            f"that keeps a longer axis: {sorted(CONV_VIEWS)}."
        )
    return out


def conv_view_shape(shape: tuple[int, ...], view: str) -> tuple[int, int]:
    """Cloud shape :func:`conv_view` would produce, without building it."""
    if len(shape) == 3:
        shape = (shape[0], shape[1], 1, shape[2])
    if len(shape) != 4:
        raise ValueError(f"conv_view_shape needs a 4-D shape; got {shape}.")
    if view not in CONV_VIEWS:
        raise ValueError(f"Unknown conv view {view!r}. Known: {sorted(CONV_VIEWS)}.")
    o, i, h, w = shape
    return {
        "rows":        (o * i * h, w),
        "cols":        (o * i * w, h),
        "kernel":      (o * i, h * w),
        "out_channel": (o, i * h * w),
        "in_channel":  (i, o * h * w),
        "tap":         (h * w, o * i),
    }[view]


def matrix_view(
    M: np.ndarray,
    view: str = "full",
    *,
    part: str = "upper",
    normalise: str | None = None,
) -> np.ndarray:
    """Read a 2-D matrix as a point cloud.

    Parameters
    ----------
    M : (r, c) ndarray
        A weight matrix, a convolutional kernel flattened to two axes, or an
        activation matrix.
    view : str
        A key of :data:`VIEWS`.
    part : {"upper", "full"}
        For the Gram views only.  ``"upper"`` keeps the upper triangle
        including the diagonal, which loses nothing — a Gram matrix is
        symmetric — and roughly halves the dimensionality.
    normalise : {None, "rows", "correlation"}
        ``"rows"`` L2-normalises each row before the view is taken, so a Gram
        matrix becomes cosine similarities and ``gram_diag`` becomes the mean
        cosine at each lag.  ``"correlation"`` centres each row first as well.
        Applied before the view, never after.

    Returns
    -------
    (n_points, n_features) ndarray
        Always 2-D.  Views that describe the matrix as a whole return exactly
        one row.

    Raises
    ------
    ValueError
        Unknown *view*, or a matrix that is not 2-D.

    Examples
    --------
    >>> W = np.arange(12, dtype=float).reshape(3, 4)
    >>> matrix_view(W, "full").shape
    (1, 12)
    >>> matrix_view(W, "rows").shape
    (3, 4)
    >>> matrix_view(W, "row_norm").shape
    (1, 3)
    >>> matrix_view(W, "gram_rows", part="upper").shape
    (1, 6)
    """
    A = np.asarray(M, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(
            f"matrix_view needs a 2-D matrix; got shape {A.shape}. Reshape a "
            f"convolutional kernel to (out_channels, -1) or (freq, time) first."
        )
    if view not in VIEWS:
        raise ValueError(
            f"Unknown view {view!r}. Known views:\n  "
            + "\n  ".join(f"{k:<10} {v}" for k, v in VIEWS.items())
        )
    if part not in {"upper", "full"}:
        raise ValueError(f"part must be 'upper' or 'full'; got {part!r}.")

    A = _normalise(A, normalise)

    if view == "full":
        return A.reshape(1, -1)
    if view == "rows":
        return A
    if view == "cols":
        return A.T
    if view == "row_norm":
        return np.linalg.norm(A, axis=1)[None, :]
    if view == "col_norm":
        return np.linalg.norm(A, axis=0)[None, :]
    if view == "row_sum":
        return A.sum(axis=1)[None, :]
    if view == "col_sum":
        return A.sum(axis=0)[None, :]
    if view in _GRAM_VIEWS:
        G = A @ A.T if view == "gram_rows" else A.T @ A
        if part == "upper":
            iu = np.triu_indices(G.shape[0])
            return G[iu][None, :]
        return G.reshape(1, -1)
    if view == "gram_diag":
        G = A @ A.T
        n = G.shape[0]
        return np.array([[np.trace(G, offset=t) / max(n - t, 1) for t in range(n)]])
    raise AssertionError(f"unhandled view {view!r}")   # pragma: no cover


def _normalise(A: np.ndarray, how: str | None) -> np.ndarray:
    if how is None:
        return A
    if how == "rows":
        return A / (np.linalg.norm(A, axis=1, keepdims=True) + _EPS)
    if how == "correlation":
        C = A - A.mean(axis=1, keepdims=True)
        return C / (np.linalg.norm(C, axis=1, keepdims=True) + _EPS)
    raise ValueError(
        f"normalise must be None, 'rows' or 'correlation'; got {how!r}."
    )


def view_shape(matrix_shape: tuple[int, int], view: str, part: str = "upper") -> tuple[int, int]:
    """Cloud shape a view would produce, without building it.

    Useful for costing a grid before running it — the Gram views in particular
    can be far larger than the matrix they come from.

    Examples
    --------
    >>> view_shape((84, 15), "gram_rows")          # 84*85/2
    (1, 3570)
    >>> view_shape((84, 15), "gram_rows", "full")
    (1, 7056)
    >>> view_shape((84, 15), "rows")
    (84, 15)
    """
    r, c = matrix_shape
    if view not in VIEWS:
        raise ValueError(f"Unknown view {view!r}. Known: {sorted(VIEWS)}.")
    if view == "full":
        return (1, r * c)
    if view == "rows":
        return (r, c)
    if view == "cols":
        return (c, r)
    if view in ("row_norm", "row_sum"):
        return (1, r)
    if view in ("col_norm", "col_sum"):
        return (1, c)
    if view == "gram_rows":
        return (1, r * (r + 1) // 2 if part == "upper" else r * r)
    if view == "gram_cols":
        return (1, c * (c + 1) // 2 if part == "upper" else c * c)
    if view == "gram_diag":
        return (1, r)
    raise AssertionError(f"unhandled view {view!r}")   # pragma: no cover


def describe_view(view: str) -> str:
    """One-line description of a view, for messages and printed summaries."""
    if view not in VIEWS:
        raise ValueError(f"Unknown view {view!r}. Known: {sorted(VIEWS)}.")
    return VIEWS[view]


def stack_views(
    matrices: Any,
    view: str = "full",
    *,
    part: str = "upper",
    normalise: str | None = None,
) -> np.ndarray:
    """Apply a view to every matrix and stack the results into one cloud.

    Parameters
    ----------
    matrices : sequence of (r, c) ndarray
        For example one kernel per filter per model.  All must share a shape,
        so the stacked cloud has a consistent feature axis.
    view, part, normalise
        As for :func:`matrix_view`.

    Returns
    -------
    (n_points, n_features) ndarray

    Raises
    ------
    ValueError
        If the matrices do not all share a shape — stacking would otherwise
        produce a cloud whose feature axis means different things in different
        rows.
    """
    mats = list(matrices)
    if not mats:
        raise ValueError("stack_views received no matrices.")
    shapes = {np.asarray(m).shape for m in mats}
    if len(shapes) != 1:
        raise ValueError(
            f"All matrices must share a shape to stack into one cloud; got "
            f"{sorted(shapes)[:4]}{'…' if len(shapes) > 4 else ''}. Group them by "
            f"shape, or pick a view whose feature axis does not depend on the "
            f"differing dimension."
        )
    return np.vstack([matrix_view(m, view, part=part, normalise=normalise) for m in mats])
