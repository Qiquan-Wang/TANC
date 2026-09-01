"""mapper_sweep.py — Mapper over a parameter grid, with the cover made explicit.

Why this exists alongside :mod:`mapper_tool`
--------------------------------------------
:func:`~tanc.topo_tools.mapper_tool.run_mapper` builds one Mapper graph
through giotto-tda and hands back the graph.  That is the right tool when you
know your parameters.  It cannot answer the question this module exists for:
**is this graph telling me about my data, or about my cover?**

Answering that needs the cover cells themselves, which the giotto pipeline does
not expose.  A Mapper graph that reproduces the nerve of its own cover has
found nothing — every cover produces one, on any data at all.  So everything
here keeps the cover in hand and reports ``b1_excess = b1 - b1_nerve``
alongside the raw Betti numbers.

The overlap convention
----------------------
``overlap`` is the fraction of an interval shared with its neighbour::

        overlap = |I_k ∩ I_{k+1}| / |I_k|

matching KeplerMapper's ``perc_overlap`` and giotto-tda's ``overlap_frac``,
after Carrière, Michel & Oudot, *Statistical Analysis and Parameter Selection
for Mapper* (arXiv:1706.00204).  Interval width is therefore
``W = range / (n_intervals * (1 - overlap))``.

.. warning::
   A different convention — widening each interval by a *fraction of the
   spacing*, ``W = (range / n) * (1 + o)`` — appears in some hand-rolled Mapper
   code.  The two disagree: an ``o`` in the width-ratio convention equals
   ``o / (1 + o)`` here, so a nominal 0.67 there is 0.40 here.  Use
   :func:`convert_overlap` to translate old parameters.  Because ``W`` diverges
   as ``overlap`` approaches 1, this module accepts ``0 < overlap < 1``.

What is shared across a grid
----------------------------
Single-linkage at several distance thresholds is the same dendrogram cut in
several places, so :class:`CoverPlan` builds each cell's dendrogram once, on
first use, and every threshold after that is a cut.  Measured on cells of
2,000 points this is ~4.8x faster than recomputing per threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "convert_overlap",
    "max_intervals",
    "cover_intervals",
    "lens_ranges",
    "CoverPlan",
    "build_cover",
    "nerve_of",
    "DBSCANCells",
    "SingleLinkageCells",
    "FirstGapCells",
    "HDBSCANCells",
    "WardCells",
    "calibrate",
    "MapperGraph",
    "mapper_graph",
    "measure_graph",
    "MEASURE_GROUPS",
    "save_graph",
    "load_graph",
    "stored_nerve",
    "LENS_BUILDERS",
    "resolve_lens",
    "codensity",
    "resolve_preprocess",
    "validate_config",
    "MapperGrid",
]

_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Cover
# ─────────────────────────────────────────────────────────────────────────────

def convert_overlap(value: float, frm: str = "width_ratio", to: str = "standard") -> float:
    """Translate an overlap between the two conventions in circulation.

    Parameters
    ----------
    value : float
        The overlap to convert.
    frm, to : {"standard", "width_ratio"}
        ``"standard"`` is intersection over interval length (KeplerMapper,
        giotto-tda, this module).  ``"width_ratio"`` widens each interval by a
        fraction of the inter-centre spacing.

    Returns
    -------
    float

    Examples
    --------
    A hand-rolled cover's "67% overlap" is really 40% in the standard sense:

    >>> round(convert_overlap(0.67), 3)
    0.401

    And a genuine 67% needs a far larger width-ratio value:

    >>> round(convert_overlap(0.67, frm="standard", to="width_ratio"), 3)
    2.030
    """
    if frm == to:
        return float(value)
    if frm == "width_ratio" and to == "standard":
        return float(value) / (1.0 + float(value))
    if frm == "standard" and to == "width_ratio":
        if not 0.0 <= value < 1.0:
            raise ValueError("A standard overlap must satisfy 0 <= overlap < 1.")
        return float(value) / (1.0 - float(value))
    raise ValueError(
        f"Unknown conversion {frm!r} -> {to!r}; "
        "expected 'standard' or 'width_ratio' for both."
    )


def max_intervals(
    n_points: int,
    lens_dim: int = 2,
    overlap: float = 0.3,
    min_samples: int = 5,
    factor: int = 10,
) -> int:
    """Largest resolution whose cells still hold enough points to cluster.

    A cover cell that holds fewer points than the clusterer needs cannot
    produce a node, and a cover of such cells shatters the graph into
    singletons — which inflates ``b1`` spectacularly while meaning nothing.
    Expected occupancy is ``n_points / (n * (1 - overlap)) ** lens_dim``;
    requiring that to be at least ``factor * min_samples`` gives

    .. math:: n_{\\max} = \\frac{1}{1-g}
              \\left(\\frac{N}{c\\,m}\\right)^{1/d}

    Parameters
    ----------
    n_points : int
        Points in the cloud.
    lens_dim : int
        Lens dimension — the cover is a product over this many axes.
    overlap : float
        Standard-convention overlap.
    min_samples : int
        Points the clusterer needs to form a cluster.
    factor : int
        How many times ``min_samples`` a typical cell should hold.  The default
        of 10 targets ~50 points per cell.

    Returns
    -------
    int
        At least 1.

    Notes
    -----
    This assumes points spread evenly over the lens image.  Real lenses are
    peaked, so the *median* cell is emptier than this average — treat the
    result as an upper bound and check
    :attr:`CoverPlan.cell_size_median` on the cover you actually build.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must satisfy 0 <= overlap < 1; got {overlap}.")
    budget = n_points / float(factor * min_samples)
    if budget <= 1.0:
        return 1
    return max(1, int((budget ** (1.0 / lens_dim)) / (1.0 - overlap)))


def cover_intervals(
    v: np.ndarray,
    n_intervals: int,
    overlap: float,
    value_range: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    """Overlapping intervals covering the range of *v*, in the standard convention.

    Parameters
    ----------
    v : (N,) ndarray
        One lens coordinate.
    n_intervals : int
        Number of intervals.
    overlap : float
        Fraction of each interval shared with its neighbour, ``0 <= overlap < 1``.
    value_range : (float, float), optional
        Fix the interval boundaries to this range instead of deriving them from
        *v*.  Pass the **unfiltered** cloud's lens range when covering a subset,
        so that filtering the points does not also move the cover: otherwise a
        comparison across filter strengths varies two things at once and cannot
        attribute the difference to either.

    Returns
    -------
    list of (float, float)
        Closed intervals, ordered.  Their union contains the covered range.
    """
    if n_intervals < 1:
        raise ValueError(f"n_intervals must be >= 1; got {n_intervals}.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < 1 in the standard convention; "
            f"got {overlap}. Interval width diverges as overlap approaches 1. "
            f"If this value came from a width-ratio cover, convert it with "
            f"convert_overlap({overlap})."
        )
    if value_range is not None:
        if len(value_range) != 2 or np.ndim(value_range[0]) != 0:
            raise ValueError(
                f"value_range must be a single (min, max) pair for this one lens "
                f"axis; got {value_range!r}. If that came from lens_ranges(), it "
                f"holds one pair per axis — index it, or pass the whole thing to "
                f"build_cover's `value_ranges` instead."
            )
        lo, hi = float(value_range[0]), float(value_range[1])
    else:
        lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo:                       # constant lens: one interval holds everything
        return [(lo - 0.5, hi + 0.5)]
    spacing = (hi - lo) / n_intervals
    width = spacing / (1.0 - overlap)
    half = width / 2.0
    return [(lo + spacing * (i + 0.5) - half, lo + spacing * (i + 0.5) + half)
            for i in range(n_intervals)]


@dataclass
class CoverPlan:
    """A built cover, its nerve, and lazily-built per-cell dendrograms.

    Attributes
    ----------
    cells : list of ndarray
        Point indices in each non-empty cell.
    n_intervals, overlap, lens_dim : int, float, int
        The parameters that produced it.
    n_cells_total : int
        ``n_intervals ** lens_dim`` — including cells that came out empty.
    n_points : int
        Size of the cloud the cover was built on.
    """

    cells: list[np.ndarray]
    n_intervals: int
    overlap: float
    lens_dim: int
    n_cells_total: int
    n_points: int
    _nerve: dict[str, int] | None = field(default=None, repr=False)
    _dendrograms: dict[int, Any] = field(default_factory=dict, repr=False)

    # ── derived descriptors ──────────────────────────────────────────────────

    @property
    def n_nonempty(self) -> int:
        return len(self.cells)

    @property
    def empty_frac(self) -> float:
        return 1.0 - self.n_nonempty / max(self.n_cells_total, 1)

    @property
    def cell_sizes(self) -> np.ndarray:
        return np.array([len(c) for c in self.cells], dtype=np.int64) if self.cells \
            else np.zeros(0, dtype=np.int64)

    @property
    def cell_size_median(self) -> float:
        s = self.cell_sizes
        return float(np.median(s)) if len(s) else 0.0

    @property
    def cell_size_max(self) -> int:
        s = self.cell_sizes
        return int(s.max()) if len(s) else 0

    @property
    def mean_cells_per_point(self) -> float:
        """Average number of cells a point belongs to — the replication factor.

        Cover cells overlap, so a point is clustered once per cell it lands in
        and can contribute a node to each.  This is why a Mapper graph may hold
        more nodes than the cloud holds points.
        """
        return float(self.cell_sizes.sum()) / max(self.n_points, 1)

    def nerve(self) -> dict[str, int]:
        """Betti numbers of the nerve of this cover, computed once and cached.

        The nerve is what Mapper returns when the clustering inside each cell
        finds a single blob: one node per cell, edges wherever cells meet.  Any
        ``b1`` at or below the nerve's is a property of the cover, not the data.
        """
        if self._nerve is None:
            self._nerve = nerve_of(self.cells, self.n_points)
        return self._nerve

    def dendrogram(self, cell_index: int, X: np.ndarray, metric: str = "euclidean",
                   method: str = "single") -> Any:
        """Linkage dendrogram for one cell, built once per (cell, method) and reused.

        Sweeping several distance thresholds is the same dendrogram cut in
        several places; building it per threshold is the single largest
        avoidable cost in a Mapper sweep.  The cache is keyed by linkage method
        as well, so a grid mixing single and Ward linkage still builds each at
        most once per cell.

        Ward, centroid and median linkage are defined only for Euclidean
        distance -- they reason about cluster means -- so *metric* is ignored
        and forced to Euclidean for those.
        """
        if method in ("ward", "centroid", "median"):
            metric = "euclidean"
        key = (cell_index, method, metric)
        Z = self._dendrograms.get(key)
        if Z is None:
            from scipy.cluster.hierarchy import linkage
            from scipy.spatial.distance import pdist
            pts = X[self.cells[cell_index]]
            Z = linkage(pdist(pts, metric=metric), method=method)
            self._dendrograms[key] = Z
        return Z

    def release_dendrograms(self) -> None:
        """Drop cached dendrograms (they dominate memory for large cells)."""
        self._dendrograms.clear()


def build_cover(
    lens: np.ndarray,
    n_intervals: int,
    overlap: float,
    value_ranges: Sequence[tuple[float, float]] | None = None,
) -> CoverPlan:
    """Build the product cover of a lens image.

    Membership is computed once per lens axis and the cells formed by combining
    those columns, so a resolution-50 2-D cover costs two ``(N, 50)`` boolean
    masks rather than one dense ``(N, 2500)`` one.

    Parameters
    ----------
    lens : (N,) or (N, d) ndarray
        Lens values.  A 1-D array is treated as a single lens axis.
    n_intervals : int
        Intervals per lens axis.
    overlap : float
        Standard-convention overlap, ``0 <= overlap < 1``.
    value_ranges : sequence of (float, float), optional
        One ``(min, max)`` per lens axis, fixing the cover's boundaries.  Use
        :meth:`lens_ranges` on the full cloud and pass the result when covering
        a filtered subset, so the cover stays identical as the point set changes.

    Returns
    -------
    CoverPlan
    """
    L = np.asarray(lens)
    if L.ndim == 1:
        L = L[:, None]
    if L.ndim != 2:
        raise ValueError(f"lens must be 1-D or 2-D; got shape {L.shape}.")
    n_points, d = L.shape
    if value_ranges is not None and len(value_ranges) != d:
        raise ValueError(
            f"value_ranges has {len(value_ranges)} entries but the lens has {d} axes."
        )

    masks = []
    for j in range(d):
        col = L[:, j]
        vr = value_ranges[j] if value_ranges is not None else None
        mj = np.zeros((n_points, n_intervals), dtype=bool)
        for i, (a, b) in enumerate(cover_intervals(col, n_intervals, overlap, vr)):
            mj[:, i] = (col >= a) & (col <= b)
        masks.append(mj)

    cells: list[np.ndarray] = []
    for combo in product(*[range(n_intervals)] * d):
        m = masks[0][:, combo[0]]
        for j in range(1, d):
            m = m & masks[j][:, combo[j]]
        if m.any():
            cells.append(np.flatnonzero(m).astype(np.int32))

    return CoverPlan(
        cells=cells,
        n_intervals=n_intervals,
        overlap=float(overlap),
        lens_dim=d,
        n_cells_total=n_intervals ** d,
        n_points=n_points,
    )


def lens_ranges(lens: np.ndarray) -> list[tuple[float, float]]:
    """Per-axis ``(min, max)`` of a lens image, for pinning a cover.

    Compute this once on the full cloud and hand it to :func:`build_cover` for
    every filtered subset, so that comparisons across filter strengths change
    only the point set and never the cover.

    Examples
    --------
    >>> ranges = lens_ranges(L_full)                       # doctest: +SKIP
    >>> plan = build_cover(L_full[keep], 30, 0.4, ranges)  # doctest: +SKIP
    """
    L = np.asarray(lens)
    if L.ndim == 1:
        L = L[:, None]
    return [(float(L[:, j].min()), float(L[:, j].max())) for j in range(L.shape[1])]


def _incidence(groups: Sequence[np.ndarray], n_points: int):
    """Sparse ``(G, n_points)`` membership matrix: row *g* marks group *g*'s points."""
    from scipy.sparse import csr_matrix
    if len(groups) == 0:
        return csr_matrix((0, n_points), dtype=np.int32)
    lengths = np.fromiter((len(g) for g in groups), dtype=np.int64, count=len(groups))
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    indices = np.concatenate([np.asarray(g, dtype=np.int32) for g in groups])
    data = np.ones(len(indices), dtype=np.int32)
    return csr_matrix((data, indices, indptr), shape=(len(groups), n_points))


def _overlap_pairs(groups: Sequence[np.ndarray], n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Every ``(a < b)`` pair of groups sharing a point, from one sparse product.

    Intersecting all pairs with Python sets costs a set per group plus one
    intersection per pair; ``C @ C.T`` does the same work in a single sparse
    matrix multiply and materialises only the pairs that actually overlap.
    """
    if len(groups) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    C = _incidence(groups, n_points)
    A = (C @ C.T).tocoo()
    keep = A.row < A.col
    return A.row[keep], A.col[keep]


def nerve_of(cells: Sequence[np.ndarray], n_points: int) -> dict[str, int]:
    """Betti numbers of the nerve of a cover.

    Returns
    -------
    dict
        ``nerve_V``, ``nerve_E``, ``nerve_b0``, ``nerve_b1``.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    V = len(cells)
    if V == 0:
        return {"nerve_V": 0, "nerve_E": 0, "nerve_b0": 0, "nerve_b1": 0}
    ra, cb = _overlap_pairs(cells, n_points)
    E = int(len(ra))
    A = csr_matrix(
        (np.ones(2 * E, dtype=np.int8),
         (np.concatenate([ra, cb]), np.concatenate([cb, ra]))),
        shape=(V, V),
    )
    b0 = int(connected_components(A, directed=False, return_labels=False))
    return {"nerve_V": V, "nerve_E": E, "nerve_b0": b0, "nerve_b1": E - V + b0}


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell clusterers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DBSCANCells:
    """DBSCAN inside each cover cell.

    Parameters
    ----------
    eps : float or {"elbow"} or ("quantile", q)
        Neighbourhood radius.  ``"elbow"`` takes the knee of the sorted
        ``min_samples``-NN distance curve; ``("quantile", q)`` takes the *q*-th
        percentile of sampled pairwise distances.  Both are resolved once per
        cloud by :func:`calibrate` — the distance scale changes completely
        between representations, so a fixed number rarely transfers.
    min_samples : int
        DBSCAN's ``min_samples``.
    metric : str
        Any metric scikit-learn's DBSCAN accepts.

    Notes
    -----
    ``"elbow"`` is convenient but not reliable on curve-like or filamentary
    data: where points lie along a thin structure, the knee of the k-NN curve
    tends to land at the *noise* scale rather than the along-curve spacing, and
    the resulting radius shatters each structure into fragments.  On a noisy
    circle of 1,200 points with noise 0.03, it returns 0.034 — enough to break
    every arc — while any value from 0.15 upward recovers the loop.

    The damage is visible rather than silent: fragmentation shows up as a large
    ``n_components`` together with an inflated ``cpc_mean``.  Check those before
    trusting a graph built on an automatically chosen radius, and prefer sweeping
    an explicit ``("quantile", q)`` when the geometry is unknown.
    """

    eps: Any = "elbow"
    min_samples: int = 5
    metric: str = "euclidean"

    needs_dendrogram = False

    def labels(self, points: np.ndarray, *, plan=None, cell=None, X=None) -> np.ndarray:
        from sklearn.cluster import DBSCAN
        if not isinstance(self.eps, (int, float)):
            raise ValueError(
                f"eps is still {self.eps!r}; call calibrate(clusterer, X) before use."
            )
        if len(points) <= self.min_samples:
            # Too small for DBSCAN to find anything but noise; keep it as one
            # node rather than discarding the cell.
            return np.zeros(len(points), dtype=int)
        return DBSCAN(eps=float(self.eps), min_samples=self.min_samples,
                      metric=self.metric).fit_predict(points)


@dataclass
class SingleLinkageCells:
    """Single-linkage inside each cell, cut at a fixed distance.

    This is the clusterer of the original Mapper lineage.  Because a sweep over
    thresholds reuses one dendrogram per cell, put ``threshold`` on the
    innermost grid axis.

    Parameters
    ----------
    threshold : float or ("quantile", q)
        Distance at which to cut.  ``("quantile", q)`` resolves to the *q*-th
        percentile of sampled pairwise distances over the whole cloud.
    metric : str
        Distance metric for the dendrogram.
    max_cell : int
        Cells larger than this are skipped — single linkage is quadratic in
        memory, and a 20,000-point cell needs gigabytes for its distance matrix
        alone.  Skipped cells are reported, never silently dropped.
    """

    threshold: Any = ("quantile", 2.0)
    metric: str = "euclidean"
    max_cell: int = 8000

    needs_dendrogram = True

    def labels(self, points: np.ndarray, *, plan: CoverPlan, cell: int, X: np.ndarray) -> np.ndarray:
        from scipy.cluster.hierarchy import fcluster
        if not isinstance(self.threshold, (int, float)):
            raise ValueError(
                f"threshold is still {self.threshold!r}; call calibrate(clusterer, X) first."
            )
        if len(points) < 2:
            return np.zeros(len(points), dtype=int)
        Z = plan.dendrogram(cell, X, metric=self.metric)
        return fcluster(Z, t=float(self.threshold), criterion="distance") - 1


@dataclass
class FirstGapCells:
    """Single linkage cut at the first large gap in each cell's merge heights.

    This is the classical Mapper clusterer — the rule Singh, Memoli & Carlsson's
    lineage uses, and the one Gabrielsson & Carlsson (2019) applied via Ayasdi —
    and it takes **no threshold**.  Where
    :class:`SingleLinkageCells` imposes one distance on every cell,
    this adapts to each cell separately, which matters when the lens image has
    dense and sparse regions: a radius that resolves a crowded cell will merge
    everything in a sparse one.

    The rule: sort the dendrogram's merge heights, find the first gap between
    consecutive heights exceeding ``relative_gap`` times the full range of
    heights, and cut there.  If no gap qualifies, the cell is one cluster.

    Parameters
    ----------
    relative_gap : float
        Gap size as a fraction of the range of merge heights.  Larger values
        demand a more decisive separation and so yield fewer clusters.  The
        default matches giotto-tda's ``FirstSimpleGap``.
    linkage : {"single", "complete", "average", "ward"}
        Which dendrogram to cut.  The gap rule is a property of the merge
        heights, not of the linkage, so it applies to any of them -- this is
        what makes Ward automatic as well.  ``"ward"`` forces a Euclidean
        metric, since it reasons about cluster means.
    metric : str
        Distance metric for the dendrogram.
    max_cell : int
        Cells larger than this are skipped; single linkage is quadratic in
        memory.

    Notes
    -----
    Being threshold-free does not make it assumption-free: ``relative_gap`` is
    still a choice, and a cell whose points form a continuum has no honest gap
    to find.  Sweeping it against :class:`SingleLinkageCells` at fixed
    quantiles is the way to see whether the adaptivity is doing real work.
    """

    relative_gap: float = 0.3
    linkage: str = "single"
    metric: str = "euclidean"
    max_cell: int = 8000

    needs_dendrogram = True

    def labels(self, points: np.ndarray, *, plan: CoverPlan, cell: int, X: np.ndarray) -> np.ndarray:
        from scipy.cluster.hierarchy import fcluster
        if len(points) < 2:
            return np.zeros(len(points), dtype=int)
        Z = plan.dendrogram(cell, X, metric=self.metric, method=self.linkage)
        h = np.sort(Z[:, 2])
        span = float(h[-1] - h[0])
        if span <= 0:
            return np.zeros(len(points), dtype=int)
        gaps = np.diff(h)
        big = np.flatnonzero(gaps > self.relative_gap * span)
        if len(big) == 0:                    # no decisive separation: one cluster
            return np.zeros(len(points), dtype=int)
        cut = 0.5 * (h[big[0]] + h[big[0] + 1])
        return fcluster(Z, t=cut, criterion="distance") - 1


@dataclass
class HDBSCANCells:
    """HDBSCAN inside each cover cell -- density-based, with **no radius**.

    DBSCAN needs a single ``eps`` that must suit every density in the cell.
    HDBSCAN instead builds the whole hierarchy of density levels and extracts
    the most persistent clusters, so a cell containing both a tight group and a
    diffuse one can yield both.  That removes the most consequential parameter
    in the DBSCAN route: an ``eps`` slightly below the structure's scale
    shatters it into singletons, and on filamentary data the k-NN elbow
    heuristic tends to land at the noise scale rather than the structure scale.

    ``min_cluster_size`` remains a choice, but it is a count rather than a
    distance, so it does not have to be recalibrated when the representation
    changes scale -- which ``eps`` does.

    Parameters
    ----------
    min_cluster_size : int
        Smallest group HDBSCAN will call a cluster.
    metric : str
        Any metric scikit-learn's HDBSCAN accepts.
    """

    min_cluster_size: int = 5
    metric: str = "euclidean"

    needs_dendrogram = False

    def labels(self, points: np.ndarray, *, plan=None, cell=None, X=None) -> np.ndarray:
        from sklearn.cluster import HDBSCAN
        if len(points) <= self.min_cluster_size:
            return np.zeros(len(points), dtype=int)
        return HDBSCAN(min_cluster_size=self.min_cluster_size,
                       metric=self.metric).fit_predict(points)


@dataclass
class WardCells:
    """Ward-linkage inside each cell, cut at a fixed distance.

    Ward minimises within-cluster variance and is defined for Euclidean
    distance only — scikit-learn rejects any other metric, so this class takes
    no ``metric`` parameter rather than accepting one it would have to ignore.
    """

    threshold: Any = ("quantile", 5.0)
    max_cell: int = 8000

    needs_dendrogram = False
    metric = "euclidean"

    def labels(self, points: np.ndarray, *, plan=None, cell=None, X=None) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering
        if not isinstance(self.threshold, (int, float)):
            raise ValueError(
                f"threshold is still {self.threshold!r}; call calibrate(clusterer, X) first."
            )
        if len(points) < 2:
            return np.zeros(len(points), dtype=int)
        return AgglomerativeClustering(
            n_clusters=None, distance_threshold=float(self.threshold), linkage="ward"
        ).fit_predict(points)


def _elbow_eps(X: np.ndarray, min_samples: int, n_sample: int, seed: int) -> float:
    """Knee of the sorted k-NN distance curve, k = ``min_samples``."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    S = X if len(X) <= n_sample else X[rng.choice(len(X), n_sample, replace=False)]
    k = min(min_samples, max(1, len(S) - 1))
    d = np.sort(NearestNeighbors(n_neighbors=k + 1).fit(S).kneighbors(S)[0][:, k])
    x = np.linspace(0.0, 1.0, len(d))
    y = (d - d.min()) / (d.max() - d.min() + _EPS)
    return float(d[int(np.argmax((y[0] + (y[-1] - y[0]) * x) - y))])


def calibrate(clusterer, X: np.ndarray, *, seed: int = 0, n_sample: int = 1500):
    """Resolve a clusterer's data-dependent parameters against a point cloud.

    Distance scales differ completely between representations and
    preprocessings, so ``eps`` and linkage thresholds are specified as
    strategies and turned into numbers here — once per cloud, not once per
    configuration.

    Parameters
    ----------
    clusterer : DBSCANCells, SingleLinkageCells, WardCells or custom
        Returned unchanged if it has nothing to resolve.
    X : (N, D) ndarray
    seed : int
        Seed for the subsample used to estimate distance quantiles.
    n_sample : int
        Points sampled for the pairwise-distance quantile estimate.

    Returns
    -------
    A copy of *clusterer* with concrete numbers in place of strategies.
    """
    import copy
    from scipy.spatial.distance import pdist

    out = copy.copy(clusterer)
    rng = np.random.default_rng(seed)

    def _quantile(pct: float, metric: str) -> float:
        m = min(n_sample, len(X))
        S = X if len(X) <= m else X[rng.choice(len(X), m, replace=False)]
        return float(np.quantile(pdist(S, metric=metric), pct / 100.0))

    eps = getattr(out, "eps", None)
    if eps is not None and not isinstance(eps, (int, float)):
        if eps == "elbow":
            out.eps = _elbow_eps(X, getattr(out, "min_samples", 5), n_sample, seed)
        elif isinstance(eps, (tuple, list)) and len(eps) == 2 and eps[0] == "quantile":
            out.eps = _quantile(float(eps[1]), getattr(out, "metric", "euclidean"))
        else:
            raise ValueError(f"Unrecognised eps strategy {eps!r}.")

    thr = getattr(out, "threshold", None)
    if thr is not None and not isinstance(thr, (int, float)):
        if isinstance(thr, (tuple, list)) and len(thr) == 2 and thr[0] == "quantile":
            out.threshold = _quantile(float(thr[1]), getattr(out, "metric", "euclidean"))
        else:
            raise ValueError(f"Unrecognised threshold strategy {thr!r}.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The graph
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MapperGraph:
    """A Mapper graph that remembers the cover it came from.

    Attributes
    ----------
    nodes : list of ndarray
        Point indices belonging to each node.
    edges : ndarray
        ``(E, 2)`` node-index pairs, ``u < v``.
    clusters_per_cell : ndarray
        Clusters found in each cover cell.  A value of 1 everywhere means the
        graph *is* the nerve of its cover.
    skipped_cells : int
        Cells passed over because they exceeded the clusterer's ``max_cell``.
    """

    nodes: list[np.ndarray]
    edges: np.ndarray
    clusters_per_cell: np.ndarray
    n_points: int
    plan: CoverPlan | None = field(default=None, repr=False)
    lens: np.ndarray | None = field(default=None, repr=False)
    skipped_cells: int = 0

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return int(len(self.edges))

    @property
    def node_sizes(self) -> np.ndarray:
        return np.array([len(n) for n in self.nodes], dtype=np.int64) if self.nodes \
            else np.zeros(0, dtype=np.int64)

    def node_lens(self) -> np.ndarray:
        """Mean lens position of each node — the default colouring, and free."""
        if self.lens is None:
            raise ValueError("This graph was built without lens values.")
        L = self.lens if self.lens.ndim == 2 else self.lens[:, None]
        if not self.nodes:
            return np.zeros((0, L.shape[1]))
        return np.vstack([L[idx].mean(axis=0) for idx in self.nodes])

    def to_networkx(self):
        """A ``networkx.Graph`` with ``members``, ``size`` and ``mean_lens`` on nodes."""
        import networkx as nx
        g = nx.Graph()
        lens_vals = None
        if self.lens is not None:
            lens_vals = self.node_lens()
        for i, idx in enumerate(self.nodes):
            attrs = {"members": idx.tolist(), "size": int(len(idx))}
            if lens_vals is not None:
                attrs["mean_lens"] = float(np.mean(lens_vals[i]))
            g.add_node(i, **attrs)
        for u, v in self.edges:
            g.add_edge(int(u), int(v))
        return g


def mapper_graph(
    X: np.ndarray,
    plan: CoverPlan,
    clusterer,
    *,
    lens: np.ndarray | None = None,
    drop_noise: bool = True,
) -> MapperGraph:
    """Cluster within each cover cell and join nodes that share a point.

    Parameters
    ----------
    X : (N, D) ndarray
        The point cloud.
    plan : CoverPlan
        A cover built on the same points, by :func:`build_cover`.
    clusterer : DBSCANCells, SingleLinkageCells, WardCells or custom
        Must already be calibrated; see :func:`calibrate`.
    lens : ndarray, optional
        Lens values, stored on the result so nodes can be coloured by them.
    drop_noise : bool
        Whether DBSCAN's ``-1`` label is discarded rather than made a node.

    Returns
    -------
    MapperGraph
    """
    if plan.n_points != len(X):
        raise ValueError(
            f"Cover was built on {plan.n_points} points but X has {len(X)}. "
            f"Build the cover from the lens of this same cloud."
        )
    max_cell = getattr(clusterer, "max_cell", None)

    nodes: list[np.ndarray] = []
    per_cell = np.zeros(len(plan.cells), dtype=np.int32)
    skipped = 0

    for ci, idx in enumerate(plan.cells):
        if max_cell is not None and len(idx) > max_cell:
            skipped += 1
            continue
        labels = clusterer.labels(X[idx], plan=plan, cell=ci, X=X)
        made = 0
        for lab in np.unique(labels):
            if drop_noise and lab == -1:
                continue
            nodes.append(idx[labels == lab])
            made += 1
        per_cell[ci] = made

    ra, cb = _overlap_pairs(nodes, len(X))
    edges = np.column_stack([ra, cb]).astype(np.int64) if len(ra) else np.zeros((0, 2), np.int64)

    return MapperGraph(
        nodes=nodes, edges=edges, clusters_per_cell=per_cell,
        n_points=len(X), plan=plan, lens=lens, skipped_cells=skipped,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Measures
# ─────────────────────────────────────────────────────────────────────────────

#: Measure names grouped by what they describe. All groups except ``"expensive"``
#: are linear or near-linear in graph size and are computed by default;
#: ``"expensive"`` is quadratic or worse and must be requested deliberately.
MEASURE_GROUPS: dict[str, tuple[str, ...]] = {
    "size": ("n_nodes", "n_edges", "membership",
             "node_mean", "node_median", "node_max", "node_min"),
    "cover": ("n_cells", "n_nonempty", "empty_frac", "node_ratio",
              "mean_cells_per_point", "covered", "cell_size_median",
              "cell_size_max", "skipped_cells"),
    "lattice": ("cpc_mean", "cpc_frac_1", "cpc_max"),
    "topology": ("b0", "b1", "nerve_V", "nerve_E", "nerve_b0", "nerve_b1", "b1_excess"),
    "shape": ("mean_degree", "max_degree", "branch_points", "leaves",
              "isolated", "n_components", "largest_component"),
    "expensive": ("density", "mean_clustering", "diameter"),
}


def measure_graph(
    graph: MapperGraph,
    groups: Sequence[str] = ("size", "cover", "lattice", "topology", "shape"),
) -> dict[str, Any]:
    """Summarise a Mapper graph.

    The default groups are all linear or near-linear in graph size.  The
    ``"expensive"`` group adds ``density``, ``mean_clustering`` and
    ``diameter``; on a graph of a few thousand nodes those alone take seconds
    each, which across a large grid dominates the run.

    Parameters
    ----------
    graph : MapperGraph
    groups : sequence of str
        Names from :data:`MEASURE_GROUPS`.

    Returns
    -------
    dict
        Measures from the requested groups.  ``b1_excess`` is ``b1`` minus the
        nerve's ``b1``: positive means Mapper found structure the cover alone
        does not explain, and zero or below means it did not.
    """
    unknown = [g for g in groups if g not in MEASURE_GROUPS]
    if unknown:
        raise ValueError(
            f"Unknown measure group(s) {unknown}. Available: {sorted(MEASURE_GROUPS)}."
        )
    want = set(groups)
    out: dict[str, Any] = {}
    sizes = graph.node_sizes
    plan = graph.plan

    if "size" in want:
        out.update(
            n_nodes=graph.n_nodes,
            n_edges=graph.n_edges,
            membership=int(sizes.sum()),
            node_mean=float(sizes.mean()) if len(sizes) else 0.0,
            node_median=float(np.median(sizes)) if len(sizes) else 0.0,
            node_max=int(sizes.max()) if len(sizes) else 0,
            node_min=int(sizes.min()) if len(sizes) else 0,
        )

    if "cover" in want and plan is not None:
        covered = 0.0
        if graph.nodes:
            seen = np.zeros(graph.n_points, dtype=bool)
            for idx in graph.nodes:
                seen[idx] = True
            covered = float(seen.mean())
        out.update(
            n_cells=plan.n_cells_total,
            n_nonempty=plan.n_nonempty,
            empty_frac=round(plan.empty_frac, 4),
            node_ratio=round(graph.n_nodes / max(plan.n_nonempty, 1), 4),
            mean_cells_per_point=round(plan.mean_cells_per_point, 4),
            covered=round(covered, 4),
            cell_size_median=plan.cell_size_median,
            cell_size_max=plan.cell_size_max,
            skipped_cells=graph.skipped_cells,
        )

    if "lattice" in want:
        cpc = graph.clusters_per_cell
        out.update(
            cpc_mean=round(float(cpc.mean()), 4) if len(cpc) else 0.0,
            cpc_frac_1=round(float((cpc == 1).mean()), 4) if len(cpc) else 0.0,
            cpc_max=int(cpc.max()) if len(cpc) else 0,
        )

    need_components = bool({"topology", "shape"} & want)
    if need_components:
        b0, comp_sizes, degrees = _graph_structure(graph)

    if "topology" in want:
        b1 = graph.n_edges - graph.n_nodes + b0
        out.update(b0=b0, b1=b1)
        if plan is not None:
            nv = plan.nerve()
            out.update(nv)
            out["b1_excess"] = b1 - nv["nerve_b1"]

    if "shape" in want:
        out.update(
            mean_degree=round(float(degrees.mean()), 4) if len(degrees) else 0.0,
            max_degree=int(degrees.max()) if len(degrees) else 0,
            branch_points=int((degrees >= 3).sum()),
            leaves=int((degrees == 1).sum()),
            isolated=int((degrees == 0).sum()),
            n_components=b0,
            largest_component=int(max(comp_sizes)) if len(comp_sizes) else 0,
        )

    if "expensive" in want:
        out.update(_expensive_measures(graph))

    return out


def _graph_structure(graph: MapperGraph) -> tuple[int, np.ndarray, np.ndarray]:
    """Components and degrees, via sparse linear algebra rather than networkx."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = graph.n_nodes
    if n == 0:
        return 0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if graph.n_edges == 0:
        return n, np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    u, v = graph.edges[:, 0], graph.edges[:, 1]
    A = csr_matrix(
        (np.ones(2 * len(u), dtype=np.int8),
         (np.concatenate([u, v]), np.concatenate([v, u]))),
        shape=(n, n),
    )
    b0, labels = connected_components(A, directed=False, return_labels=True)
    comp_sizes = np.bincount(labels)
    degrees = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    return int(b0), comp_sizes, degrees


def _expensive_measures(graph: MapperGraph) -> dict[str, float]:
    """Density, mean clustering coefficient and diameter of the largest component."""
    import networkx as nx
    g = graph.to_networkx()
    if g.number_of_nodes() == 0:
        return {"density": 0.0, "mean_clustering": 0.0, "diameter": float("inf")}
    out = {
        "density": float(nx.density(g)),
        "mean_clustering": float(nx.average_clustering(g)),
    }
    comps = list(nx.connected_components(g))
    if comps:
        largest = g.subgraph(max(comps, key=len))
        out["diameter"] = float(nx.diameter(largest)) if largest.number_of_nodes() > 1 else 0.0
    else:
        out["diameter"] = float("inf")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────

def save_graph(path: Any, graph: MapperGraph) -> None:
    """Write a Mapper graph to a compressed ``.npz``, node membership included.

    Membership is stored as a flat index array plus offsets — the sparse form —
    so a graph of tens of thousands of nodes stays small.  Keeping the members
    is what lets a finished sweep be recoloured, or interrogated for which
    points landed together, without recomputing anything.
    """
    sizes = graph.node_sizes
    flat = np.concatenate(graph.nodes).astype(np.int32) if graph.nodes \
        else np.zeros(0, dtype=np.int32)
    payload = {
        "members_flat": flat,
        "members_offsets": np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        "edges": graph.edges.astype(np.int64),
        "clusters_per_cell": graph.clusters_per_cell.astype(np.int32),
        "n_points": np.int64(graph.n_points),
        "skipped_cells": np.int64(graph.skipped_cells),
    }
    if graph.lens is not None:
        payload["lens"] = np.asarray(graph.lens, dtype=np.float32)
    if graph.plan is not None:
        nv = graph.plan.nerve()
        payload["cover_params"] = np.array(
            [graph.plan.n_intervals, graph.plan.overlap, graph.plan.lens_dim,
             graph.plan.n_cells_total, graph.plan.n_nonempty],
            dtype=np.float64,
        )
        payload["nerve"] = np.array(
            [nv["nerve_V"], nv["nerve_E"], nv["nerve_b0"], nv["nerve_b1"]], dtype=np.int64
        )
    np.savez_compressed(path, **payload)


def load_graph(path: Any) -> MapperGraph:
    """Read a graph written by :func:`save_graph`.

    The cover's cells are not stored — they are recoverable from the lens and
    cover parameters, and storing them would roughly double the file — so the
    returned graph carries ``plan=None``.  Its nerve numbers survive in the
    file and are returned by :func:`stored_nerve`.
    """
    with np.load(path, allow_pickle=False) as z:
        offsets = z["members_offsets"]
        flat = z["members_flat"]
        nodes = [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
        return MapperGraph(
            nodes=nodes,
            edges=z["edges"],
            clusters_per_cell=z["clusters_per_cell"],
            n_points=int(z["n_points"]),
            plan=None,
            lens=z["lens"] if "lens" in z.files else None,
            skipped_cells=int(z["skipped_cells"]),
        )


def stored_nerve(path: Any) -> dict[str, int] | None:
    """Nerve numbers recorded alongside a saved graph, or ``None`` if absent."""
    with np.load(path, allow_pickle=False) as z:
        if "nerve" not in z.files:
            return None
        v, e, b0, b1 = (int(x) for x in z["nerve"])
        return {"nerve_V": v, "nerve_E": e, "nerve_b0": b0, "nerve_b1": b1}


# ─────────────────────────────────────────────────────────────────────────────
# Lenses
# ─────────────────────────────────────────────────────────────────────────────

def _lens_pca(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    return PCA(n_components=k, random_state=seed).fit_transform(X)


def _lens_tsne(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    return TSNE(n_components=k, random_state=seed, init="pca").fit_transform(X)


def _lens_umap(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The UMAP lens needs umap-learn. Install the 'examples' extra, which "
            "pins umap-learn==0.5.6 — later releases require scikit-learn>=1.6, "
            "which giotto-tda forbids."
        ) from exc
    return umap.UMAP(n_components=k, random_state=seed).fit_transform(X)


def _lens_l2(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    return np.linalg.norm(X, axis=1)[:, None]


def _lens_density(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    return codensity(X)[:, None]


#: Lens names and the number of output dimensions each produces.  A name ending
#: in a digit sets the dimension, so ``"pca2"`` is two-dimensional.  Any
#: callable ``f(X) -> (N, d)`` is also accepted and receives the whole matrix,
#: so global lenses (anything fitted across points) work as well as row-wise ones.
LENS_BUILDERS: dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {
    "pca": _lens_pca,
    "tsne": _lens_tsne,
    "umap": _lens_umap,
    "l2": _lens_l2,
    "density": _lens_density,
}


def resolve_lens(spec: Any, X: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Compute lens values for *X*.

    Parameters
    ----------
    spec : str or callable
        ``"pca1"``, ``"pca2"``, ``"tsne2"``, ``"umap2"``, ``"l2"``,
        ``"density"``, or a callable applied to the whole matrix.
    X : (N, D) ndarray
    seed : int
        Passed to any stochastic lens.  ``tsne`` and ``umap`` give different
        answers per seed, so this is recorded in the run manifest.

    Returns
    -------
    (N, d) ndarray
    """
    if callable(spec):
        L = np.asarray(spec(X))
        if L.ndim == 1:
            L = L[:, None]
        if L.ndim != 2 or len(L) != len(X):
            raise ValueError(
                f"A callable lens must return (N,) or (N, d) for N={len(X)}; "
                f"got shape {L.shape}."
            )
        return L.astype(np.float64)

    if not isinstance(spec, str):
        raise ValueError(f"Lens must be a name or a callable; got {type(spec).__name__}.")

    name, k = _split_lens_name(spec)
    if name not in LENS_BUILDERS:
        raise ValueError(
            f"Unknown lens {spec!r}. Known: {sorted(LENS_BUILDERS)} "
            f"(append a digit for the dimension, e.g. 'pca2'), or pass a callable."
        )
    return np.asarray(LENS_BUILDERS[name](X, k, seed), dtype=np.float64)


def _split_lens_name(spec: str) -> tuple[str, int]:
    """``"pca2"`` -> ``("pca", 2)``; a bare name defaults to one dimension.

    Registered names are matched whole before any digit is treated as a
    dimension suffix, so ``"l2"`` stays the L2-norm lens rather than being read
    as lens ``"l"`` in two dimensions.
    """
    s = spec.strip().lower()
    if s in LENS_BUILDERS:
        return s, 1
    digits = ""
    while s and s[-1].isdigit():
        digits = s[-1] + digits
        s = s[:-1]
    return s, int(digits) if digits else 1


def codensity(X: np.ndarray, k: int = 200, n_sub: int = 8000, seed: int = 0) -> np.ndarray:
    """Distance to the *k*-th nearest neighbour — small where the cloud is dense.

    This is the quantity behind the density filter used to ask whether a
    structure lives in a cloud's dense core or in its diffuse periphery.

    Parameters
    ----------
    X : (N, D) ndarray
    k : int
        Neighbour rank.
    n_sub : int
        Reference points sampled when ``N`` is large; distances are measured
        from every point to this sample, which keeps the cost linear in ``N``.
    seed : int

    Returns
    -------
    (N,) ndarray
    """
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    ref = X if len(X) <= n_sub else X[rng.choice(len(X), n_sub, replace=False)]
    kk = min(k, len(ref) - 1) if len(ref) > 1 else 1
    nn = NearestNeighbors(n_neighbors=kk).fit(ref)
    return nn.kneighbors(X)[0][:, -1]


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing and filtering
# ─────────────────────────────────────────────────────────────────────────────

def resolve_preprocess(spec: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply a preprocessing or filtering step.

    Preprocessing is **off by default** (``spec=None``): it changes the metric
    the clusterer sees, so it belongs on a swept axis rather than in the
    background.

    Parameters
    ----------
    spec : None, str, tuple, callable, or a list of these
        ``None`` passes the cloud through.  Names: ``"l2"`` (row-normalise),
        ``"mean_centre"``, ``"standardise"``.  Tuples describe filters:
        ``("density", k, p)`` keeps the densest fraction *p*;
        ``("norm", q, side)`` keeps points by ``‖x‖`` above (``"high"``) or
        below (``"low"``) the *q*-th percentile.  A list applies steps in order.
    X : (N, D) ndarray

    Returns
    -------
    (ndarray, ndarray or None)
        The transformed cloud, and the indices kept from the original when a
        filter dropped points (``None`` when every point survives).  The index
        array is what lets a filtered result be traced back to the full cloud.
    """
    if spec is None:
        return X, None
    if isinstance(spec, list):
        keep_total = None
        out = X
        for step in spec:
            out, keep = resolve_preprocess(step, out)
            if keep is not None:
                keep_total = keep if keep_total is None else keep_total[keep]
        return out, keep_total
    if callable(spec):
        return np.asarray(spec(X)), None

    if isinstance(spec, str):
        s = spec.strip().lower()
        if s == "l2":
            return X / (np.linalg.norm(X, axis=1, keepdims=True) + _EPS), None
        if s == "mean_centre":
            return X - X.mean(axis=0), None
        if s == "standardise":
            return (X - X.mean(axis=0)) / (X.std(axis=0) + _EPS), None
        raise ValueError(
            f"Unknown preprocessing {spec!r}. Names: 'l2', 'mean_centre', "
            f"'standardise'; filters are tuples such as ('density', 200, 0.3)."
        )

    if isinstance(spec, (tuple, list)) and spec:
        kind = str(spec[0]).lower()
        if kind == "density":
            k = int(spec[1]) if len(spec) > 1 else 200
            p = float(spec[2]) if len(spec) > 2 else 0.5
            rho = codensity(X, k=k)
            n_keep = max(1, int(round(p * len(X))))
            keep = np.argsort(rho)[:n_keep]          # smallest codensity = densest
            keep.sort()
            return X[keep], keep
        if kind == "norm":
            q = float(spec[1]) if len(spec) > 1 else 50.0
            side = str(spec[2]).lower() if len(spec) > 2 else "high"
            nrm = np.linalg.norm(X, axis=1)
            cut = np.percentile(nrm, q)
            keep = np.flatnonzero(nrm >= cut if side == "high" else nrm <= cut)
            return X[keep], keep
        raise ValueError(
            f"Unknown filter {spec[0]!r}. Known: 'density', 'norm'."
        )
    raise ValueError(f"Cannot interpret preprocessing spec {spec!r}.")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_config(
    cfg: dict[str, Any],
    *,
    n_points: int | None = None,
    n_features: int | None = None,
) -> str | None:
    """Reason this configuration cannot run, or ``None`` if it can.

    Checked before a sweep starts, so a contradictory combination costs
    milliseconds instead of surfacing partway through a long run.

    Rejects
    -------
    - Ward with a non-Euclidean metric — Ward minimises variance and is defined
      only for Euclidean distance.
    - A cosine metric on an L2-normalised cloud, which is Euclidean on that
      cloud: the configuration would duplicate another in the same grid.
    - PCA asking for more components than the data has features.
    - t-SNE beyond three components, which its Barnes-Hut solver cannot do.
    - An overlap outside ``[0, 1)``, or a resolution below 1.
    """
    clus = cfg.get("clusterer")
    metric = cfg.get("metric")
    prep = cfg.get("preprocess")
    lens = cfg.get("lens")

    if clus is not None:
        cm = getattr(clus, "metric", None)
        if isinstance(clus, WardCells) or type(clus).__name__ == "WardCells":
            if metric not in (None, "euclidean"):
                return (
                    f"Ward linkage is defined for Euclidean distance only; "
                    f"metric={metric!r} was requested."
                )
        if metric is not None and cm is not None and cm != metric:
            return (
                f"metric={metric!r} contradicts the clusterer's own metric={cm!r}. "
                f"Set it in one place."
            )

    eff_metric = metric or (getattr(clus, "metric", None) if clus is not None else None)
    if eff_metric == "cosine" and prep == "l2":
        return (
            "cosine distance on an L2-normalised cloud is a monotone function of "
            "Euclidean distance, so this duplicates the euclidean configuration."
        )

    if isinstance(lens, str):
        name, k = _split_lens_name(lens)
        if name == "pca" and n_features is not None and k > n_features:
            return f"lens {lens!r} needs {k} components but the cloud has {n_features} features."
        if name == "tsne" and k > 3:
            return f"lens {lens!r}: t-SNE supports at most 3 components."
        if name not in LENS_BUILDERS:
            return f"unknown lens {lens!r}; known: {sorted(LENS_BUILDERS)}."

    ovl = cfg.get("overlap")
    if ovl is not None and not 0.0 <= float(ovl) < 1.0:
        return (
            f"overlap={ovl} is outside [0, 1) in the standard convention. "
            f"If it came from a width-ratio cover, use convert_overlap({ovl})."
        )
    n_int = cfg.get("n_intervals")
    if n_int is not None and int(np.min(n_int)) < 1:
        return f"n_intervals={n_int} must be at least 1."
    return None


# ─────────────────────────────────────────────────────────────────────────────
# The grid
# ─────────────────────────────────────────────────────────────────────────────

class _CloudContext:
    """A preprocessed cloud plus its per-clusterer calibration cache.

    Calibration depends on the cloud but is requested by the innermost axis, so
    it cannot be a sweep stage.  Caching it here bounds the work at one
    calibration per (cloud, clusterer strategy) instead of one per configuration.
    """

    __slots__ = ("X", "keep", "seed", "_calib")

    def __init__(self, X: np.ndarray, keep: np.ndarray | None, seed: int):
        self.X = X
        self.keep = keep
        self.seed = seed
        self._calib: dict[str, Any] = {}

    def calibrated(self, clusterer):
        key = repr(clusterer)
        got = self._calib.get(key)
        if got is None:
            got = calibrate(clusterer, self.X, seed=self.seed)
            self._calib[key] = got
        return got


class _LensContext:
    __slots__ = ("cloud", "lens", "ranges")

    def __init__(self, cloud: _CloudContext, lens: np.ndarray):
        self.cloud = cloud
        self.lens = lens
        #: Extent of the *unfiltered* lens image.  Every cover below this point
        #: is pinned to it, so removing points never also moves the cover.
        self.ranges = lens_ranges(lens)


class _FilteredContext:
    """A subset of the cloud, still carrying the full cloud's lens extent.

    Point filters must be applied *after* the lens, not before.  A filter that
    correlates with the lens — codensity does, on any cloud whose density
    varies along it — shrinks the lens range, and a cover rebuilt on the subset
    shrinks with it.  The cover then holds roughly the same number of points per
    cell at every filter strength, and the comparison measures nothing.
    Keeping :attr:`ranges` from the unfiltered lens holds the cover still.
    """

    __slots__ = ("cloud", "lens", "ranges", "keep", "n_full")

    def __init__(self, lensctx: _LensContext, X: np.ndarray, lens: np.ndarray,
                 keep: np.ndarray | None):
        self.cloud = _CloudContext(X, keep, lensctx.cloud.seed)
        self.cloud._calib = lensctx.cloud._calib      # calibration is cloud-wide
        self.lens = lens
        self.ranges = lensctx.ranges                  # pinned to the FULL cloud
        self.keep = keep
        self.n_full = len(lensctx.lens)


class _CoverContext:
    __slots__ = ("lensctx", "plan")

    def __init__(self, lensctx, plan: CoverPlan):
        self.lensctx = lensctx
        self.plan = plan


@dataclass
class MapperGrid:
    """Mapper across a parameter grid, with the cover reported alongside the graph.

    Axes take a **scalar to pin** them or a **list to sweep** them; a tuple is a
    single composite value.  Declare them in the order given here — the sweep
    reuses clouds, lenses and covers down that order, so putting an expensive
    axis late makes it recompute needlessly.

    Parameters
    ----------
    clouds : ndarray or dict
        The point cloud, or a mapping from name to cloud.  A mapping adds a
        ``cloud`` axis, which is how several layers, views, or a trained and
        untrained pair are swept together.
    preprocess : spec or list of specs
        **Rescalings** -- ``"l2"``, ``"mean_centre"``, ``"standardise"``, or a
        callable -- which give every point new coordinates and remove none.
        Applied before the lens, because they change the distances the
        clusterer sees.  A spec that removes points is rejected here and
        belongs on *point_filter*.  Default ``None``.
    lens : str, callable, or list
        See :func:`resolve_lens`.
    point_filter : spec or list
        **Filters** -- ``("density", k, p)`` or ``("norm", q, side)`` -- which
        remove points and leave the survivors' coordinates untouched.  Applied *after* the lens, and the cover is
        pinned to the unfiltered lens range, so sweeping this axis changes only
        the point set.  Putting a filter on ``preprocess`` instead would shrink
        the lens image and move the cover with it, which confounds the filter
        with the cover and makes the comparison meaningless.
    n_intervals : int or list
    overlap : float or list
        Standard convention; see the module docstring.
    clusterer : clusterer or list
        Instances of :class:`DBSCANCells`, :class:`SingleLinkageCells`,
        :class:`WardCells`, or anything with a compatible ``labels`` method.
        Put this axis last: several single-linkage thresholds over one cover
        reuse each cell's dendrogram.
    metric : str or None
        Optional explicit metric, validated against the clusterer's own.
    measures : sequence of str
        Measure groups from :data:`MEASURE_GROUPS`.
    save_graphs : bool
        Write each graph to the store's artifacts, so a finished sweep can be
        recoloured and re-examined without recomputing.
    seed : int

    Examples
    --------
    >>> grid = MapperGrid(                                    # doctest: +SKIP
    ...     clouds={"conv1": K1, "conv2": K2},
    ...     lens=["pca2", "l2"],
    ...     n_intervals=[10, 20, 30],
    ...     overlap=[0.3, 0.5],
    ...     clusterer=[DBSCANCells(eps=("quantile", q)) for q in (1, 5, 25)],
    ... )
    >>> grid.validate()          # what would be rejected, and why   # doctest: +SKIP
    >>> store = grid.run("runs/kernels")                             # doctest: +SKIP
    """

    clouds: Any
    preprocess: Any = None
    lens: Any = "pca2"
    point_filter: Any = None
    n_intervals: Any = 10
    overlap: Any = 0.3
    clusterer: Any = None
    metric: Any = None
    measures: Sequence[str] = ("size", "cover", "lattice", "topology", "shape")
    save_graphs: bool = True
    seed: int = 0

    def __post_init__(self):
        if self.clusterer is None:
            self.clusterer = DBSCANCells()
        if not isinstance(self.clouds, dict):
            self.clouds = {"cloud": self.clouds}
        for name, X in self.clouds.items():
            if np.asarray(X).ndim != 2:
                raise ValueError(
                    f"Cloud {name!r} has shape {np.asarray(X).shape}; Mapper needs a "
                    f"2-D (n_points, n_features) matrix. Reshape it, and make sure "
                    f"the first axis is the one indexing points."
                )

    # ── axes ─────────────────────────────────────────────────────────────────

    def axes(self) -> dict[str, Any]:
        """The axis specification handed to the sweep engine, in reuse order."""
        return {
            "cloud": list(self.clouds) if len(self.clouds) > 1 else next(iter(self.clouds)),
            "preprocess": self.preprocess,
            "lens": self.lens,
            "point_filter": self.point_filter,
            "n_intervals": self.n_intervals,
            "overlap": self.overlap,
            "metric": self.metric,
            "clusterer": self.clusterer,
        }

    def _stages(self):
        from tanc.topo_tools._sweep_engine import Stage

        def build_cloud(prev, cloud):
            X = np.asarray(self.clouds[cloud], dtype=np.float64)
            return _CloudContext(X, None, self.seed)

        def apply_prep(prev, cloud, preprocess):
            X, keep = resolve_preprocess(preprocess, prev.X)
            if keep is not None:
                raise ValueError(
                    f"preprocess={preprocess!r} removed points. This axis is for "
                    f"rescalings such as 'l2', 'mean_centre' or 'standardise', which "
                    f"keep every point; it runs BEFORE the lens, so a filter here is "
                    f"recomputed on the subset and the cover shrinks with it -- the "
                    f"comparison then measures nothing. Put it on point_filter, which "
                    f"runs after the lens with the cover pinned."
                )
            return _CloudContext(np.asarray(X, dtype=np.float64), keep, self.seed)

        def make_lens(prev, cloud, preprocess, lens):
            return _LensContext(prev, resolve_lens(lens, prev.X, seed=self.seed))

        def apply_filter(prev, cloud, preprocess, lens, point_filter):
            # Applied after the lens, and the cover below keeps prev.ranges, so
            # varying this axis changes only the point set.
            if point_filter is None:
                return _FilteredContext(prev, prev.cloud.X, prev.lens, None)
            _, keep = resolve_preprocess(point_filter, prev.cloud.X)
            if keep is None:
                raise ValueError(
                    f"point_filter={point_filter!r} did not select a subset. This axis "
                    f"is for filters such as ('density', k, p) or ('norm', q, side); "
                    f"transforms like 'l2' belong on the preprocess axis, which runs "
                    f"before the lens."
                )
            return _FilteredContext(prev, prev.cloud.X[keep], prev.lens[keep], keep)

        def make_cover(prev, cloud, preprocess, lens, point_filter, n_intervals, overlap):
            plan = build_cover(prev.lens, int(n_intervals), float(overlap), prev.ranges)
            return _CoverContext(prev, plan)

        return [
            Stage("cloud", ("cloud",), build_cloud),
            Stage("preprocess", ("cloud", "preprocess"), apply_prep),
            Stage("lens", ("cloud", "preprocess", "lens"), make_lens),
            Stage("filter", ("cloud", "preprocess", "lens", "point_filter"), apply_filter),
            Stage("cover", ("cloud", "preprocess", "lens", "point_filter",
                            "n_intervals", "overlap"), make_cover),
        ]

    # ── validation ───────────────────────────────────────────────────────────

    def validate(self, verbose: bool = True) -> list[tuple[dict[str, Any], str]]:
        """Configurations that cannot run, each with its reason.

        Returns
        -------
        list of (config, reason)
            Empty when the whole grid is runnable.
        """
        from tanc.topo_tools._sweep_engine import expand_grid

        first = np.asarray(next(iter(self.clouds.values())))
        bad: list[tuple[dict[str, Any], str]] = []
        configs = expand_grid(self.axes())
        for cfg in configs:
            X = np.asarray(self.clouds[cfg["cloud"]])
            reason = validate_config(cfg, n_points=len(X), n_features=X.shape[1])
            if reason:
                bad.append((cfg, reason))
        if verbose:
            print(f"MapperGrid: {len(configs)} configurations, {len(bad)} rejected")
            # Group by reason: a single mistake usually rejects many
            # configurations, and listing them one by one buries the cause.
            by_reason: dict[str, int] = {}
            for _, reason in bad:
                by_reason[reason] = by_reason.get(reason, 0) + 1
            for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
                print(f"  {count:>4} x  {reason}")
            self._resolution_advice(first, bad)
        return bad

    def _resolution_advice(self, X: np.ndarray, rejected: Sequence = ()) -> None:
        """Warn where a resolution will out-run the points available to fill it."""
        from tanc.topo_tools._sweep_engine import _axis_values

        # Only lenses that will actually run should shape the advice; a
        # rejected 5-D lens must not make the 2-D ones look under-resourced.
        dead = {cfg.get("lens") for cfg, _ in rejected}
        lens_dims = []
        for spec in _axis_values(self.lens):
            if spec in dead:
                continue
            lens_dims.append(_split_lens_name(spec)[1] if isinstance(spec, str) else 2)
        if not lens_dims:
            return
        d = max(lens_dims)
        for ovl in _axis_values(self.overlap):
            cap = max_intervals(len(X), d, float(ovl))
            over = [n for n in _axis_values(self.n_intervals) if int(n) > cap]
            if over:
                print(
                    f"  note: at overlap {ovl} and a {d}-D lens, "
                    f"n_intervals above ~{cap} leaves typical cells too sparse to "
                    f"cluster — {over} may shatter. Check cell_size_median and "
                    f"cpc_mean in the results."
                )

    # ── execution ────────────────────────────────────────────────────────────

    def run(
        self,
        out: Any,
        *,
        resume: Any = None,
        progress: bool = True,
        on_error: str = "record",
    ):
        """Run the grid, recording every configuration.

        Parameters
        ----------
        out : str or Path
            Directory to create.  Never overwrites: an existing name gets a
            numeric suffix, and the path actually used is printed.
        resume : str or Path, optional
            Continue a previous run instead of starting one.  Completed
            configurations are skipped.
        progress : bool
        on_error : {"record", "raise"}

        Returns
        -------
        SweepStore
        """
        from tanc.topo_tools._sweep_engine import SweepStore, run_sweep

        store = SweepStore(resume, resume=True) if resume is not None else SweepStore(out)

        def evaluate(ctx: _CoverContext, **cfg):
            cloud = ctx.lensctx.cloud
            clus = cloud.calibrated(cfg["clusterer"])
            graph = mapper_graph(cloud.X, ctx.plan, clus, lens=ctx.lensctx.lens)
            row = measure_graph(graph, groups=self.measures)
            row["n_points"] = len(cloud.X)
            row["n_points_full"] = int(getattr(ctx.lensctx, "n_full", len(cloud.X)))
            row["kept_frac"] = round(len(cloud.X) / max(row["n_points_full"], 1), 4)
            if self.save_graphs:
                from tanc.topo_tools._sweep_engine import config_hash
                path = store.artifact_path(config_hash(cfg))
                save_graph(path, graph)
                row["graph_file"] = path.name
            return row

        return run_sweep(
            self.axes(),
            evaluate,
            store,
            stages=self._stages(),
            validate=lambda cfg: validate_config(
                cfg,
                n_points=len(self.clouds[cfg["cloud"]]),
                n_features=np.asarray(self.clouds[cfg["cloud"]]).shape[1],
            ),
            on_error=on_error,
            progress=progress,
            extra_manifest={"seed": self.seed, "measures": list(self.measures),
                            "clouds": {k: list(np.asarray(v).shape)
                                       for k, v in self.clouds.items()}},
        )
