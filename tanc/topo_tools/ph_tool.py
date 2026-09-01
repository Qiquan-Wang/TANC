"""ph_tool.py — persistent homology computation and statistics.

Applies VR / superlevel-set persistent homology to a ``GraphBundle`` and
returns a ``TopoResult``.

Backends
--------
* ``"ripser"`` (default) — fast exact PH via the ``ripser`` package.
* ``"gudhi"``            — GUDHI ``RipsComplex``.
* ``"giotto"``           — giotto-ph ``ripser_parallel`` (multithreaded ripser).
  NB: this is **giotto-ph**, not giotto-tda — giotto-tda 0.6.2 hard-pins
  ``scikit-learn==1.3.2`` and cannot run on a modern environment; giotto-ph has
  no such pin.  (giotto-tda is still used, separately, by the Mapper tool.)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from tanc.graph_builder._bundle import GraphBundle
from tanc.topo_tools._result import PersistenceResult, TopoResult


# ─────────────────────────────────────────────────────────────────────────────
# Watanabe directed clique complex (faithful PH of a feed-forward network)
# ─────────────────────────────────────────────────────────────────────────────
#
# ripser/GUDHI Rips compute *flag* complexes: a triangle enters at the max of
# its three edges.  Watanabe & Yamana's complex is a **directed clique complex
# with path-product relevance** — a different filtration that ripser cannot
# express, so it is built directly with a GUDHI ``SimplexTree`` here and routed
# through ``run_ph(..., input_complex="directed_clique")``.
#
#   * relevance R_ij = w_ij / Σ_k w_kj  (column-normalised; positive part for the
#     "Topological measurement" paper, |w| for the PHPM pruning paper),
#   * a direct edge is born at its relevance,
#   * an induced edge {i,k} (layers two apart) at the max 2-hop path product,
#   * a triangle {i,j,k} at the path product R[i,j]·R[j,k],
#
#   and every simplex's filtration *value* is the integer threshold index in
#   {1..64} of the paper's descending relevance schedule (not the relevance).
#   Path products are far smaller than either edge, so triangles enter much
#   later than a flag complex would — producing the paper's wide H1 belt.

def watanabe_schedule() -> np.ndarray:
    """The paper's 64-step **descending** relevance schedule.

    ``[c·10^e for e in 0..-6 for c in (1.0, 0.9, …, 0.2)] + [1e-7]`` — 9 values
    per decade over 7 decades plus a final ``1e-7`` ⇒ length 64, from
    ``R[0]=1.0`` down to ``R[63]=1e-7``.
    """
    return np.array(
        [c * 10.0 ** e for e in range(0, -7, -1)
         for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)]
        + [1.0e-7]
    )


_WATANABE_R = watanabe_schedule()
_WATANABE_R_ASC = _WATANABE_R[::-1].copy()
WATANABE_MIN_RELEVANCE = float(_WATANABE_R[-1])     # 1e-7


def fil_index(relevance: "np.ndarray | float") -> "np.ndarray | int":
    """Integer threshold level ``fil ∈ {1..64}`` for a relevance value.

    ``fil = #{thresholds strictly greater than relevance} + 1``; returns 65
    when ``relevance < WATANABE_MIN_RELEVANCE`` (simplex excluded).
    """
    n_gt = len(_WATANABE_R_ASC) - np.searchsorted(_WATANABE_R_ASC, relevance, side="right")
    return n_gt + 1


def relevance_matrices(
    weight_matrices: list[np.ndarray],
    mode: str = "positive",
    eps: float = 1e-12,
) -> list[np.ndarray]:
    """Column-normalise each weight matrix into relevances in ``[0, 1]``.

    ``mode="positive"`` uses ``max(w, 0)`` (measurement paper); ``"absolute"``
    uses ``|w|`` (PHPM).  Each target-neuron column sums to 1; all-zero columns
    map to zero relevance.
    """
    if mode == "positive":
        prep = lambda W: np.maximum(np.asarray(W, dtype=float), 0.0)
    elif mode == "absolute":
        prep = lambda W: np.abs(np.asarray(W, dtype=float))
    else:
        raise ValueError(f"mode must be 'positive' or 'absolute', got {mode!r}.")
    rels = []
    for W in weight_matrices:
        P = prep(W)
        col = P.sum(axis=0)
        safe = np.where(col == 0, 1.0, col)
        R = P / (safe + eps)
        R[:, col == 0] = 0.0
        rels.append(R)
    return rels


def _watanabe_layout(rel_matrices: list[np.ndarray]) -> tuple[list[int], list[int]]:
    sizes = [rel_matrices[0].shape[0]] + [r.shape[1] for r in rel_matrices]
    offsets = [sum(sizes[:l]) for l in range(len(sizes))]
    return sizes, offsets


def build_dnn_clique_complex(
    rel_matrices: list[np.ndarray],
    min_relevance: float = WATANABE_MIN_RELEVANCE,
):
    """Build the directed clique complex (≤ triangles) of an FCN.

    Returns ``(gudhi.SimplexTree, layer_sizes, offsets)``.  For each consecutive
    triple of layers ``(l, l+1, l+2)`` this matches PHPM exactly: direct edges,
    an induced edge born from the strongest 2-hop path, and one triangle per
    intermediate neuron born from that path's product.
    """
    try:
        import gudhi
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The Watanabe directed-clique-complex PH requires GUDHI.  "
            "Install with: pip install gudhi"
        ) from e

    sizes, offsets = _watanabe_layout(rel_matrices)

    def gid(layer, idx):
        return offsets[layer] + idx

    st = gudhi.SimplexTree()
    for v in range(sum(sizes)):
        st.insert([v], filtration=0.0)

    for l, R in enumerate(rel_matrices):
        for i, j in np.argwhere(R >= min_relevance):
            st.insert([gid(l, int(i)), gid(l + 1, int(j))],
                      filtration=float(fil_index(R[i, j])))

    for l in range(len(rel_matrices) - 1):
        ra, rb = rel_matrices[l], rel_matrices[l + 1]
        for i in range(ra.shape[0]):
            prod = ra[i, :][:, None] * rb                 # prod[j,k] = ra[i,j]·rb[j,k]
            for k in range(rb.shape[1]):
                col = prod[:, k]
                js = np.argwhere(col >= min_relevance).ravel()
                if js.size == 0:
                    continue
                gi, gk = gid(l, i), gid(l + 2, k)
                st.insert([gi, gk], filtration=float(fil_index(col[js].max())))
                for j in js:
                    st.insert([gi, gid(l + 1, int(j)), gk],
                              filtration=float(fil_index(col[j])))

    st.make_filtration_non_decreasing()
    return st, sizes, offsets


def compute_watanabe_ph(
    weight_matrices: list[np.ndarray],
    mode: str = "positive",
    max_dim: int = 1,
    min_relevance: float = WATANABE_MIN_RELEVANCE,
    stat_names: list[str] | None = None,
) -> TopoResult:
    """Directed-clique-complex persistent homology of an FCN's weights.

    ``ph_result.diagrams`` holds integer-axis ``(birth, death)`` diagrams;
    ``ph_result.metadata["watanabe"]`` carries the H1 representative simplices
    (``(birth, death, birth_edge, death_triangle)`` per class), the layer sizes
    and the relevance matrices — everything the PHPM pruner needs.

    This is the implementation behind ``run_ph(bundle,
    input_complex="directed_clique")``; call it directly when you already hold
    the FCN weight matrices.
    """
    rels = relevance_matrices(weight_matrices, mode=mode)
    st, sizes, _offsets = build_dnn_clique_complex(rels, min_relevance=min_relevance)
    st.persistence(homology_coeff_field=2, min_persistence=0.0)

    diagrams: dict[int, list[list[float]]] = {d: [] for d in range(max_dim + 1)}
    for dim, (b, d) in st.persistence():
        if dim <= max_dim and np.isfinite(d):
            diagrams[dim].append([float(b), float(d)])
    diagrams_np = {d: (np.array(v) if v else np.empty((0, 2)))
                   for d, v in diagrams.items()}

    h1_reps: list[tuple] = []
    for s_birth, s_death in st.persistence_pairs():
        if len(s_birth) == 2 and len(s_death) == 3:
            b, d = st.filtration(s_birth), st.filtration(s_death)
            if np.isfinite(d) and d > b:
                h1_reps.append((float(b), float(d),
                                tuple(int(v) for v in s_birth),
                                tuple(int(v) for v in s_death)))

    ph_result = PersistenceResult(
        diagrams=diagrams_np,
        metadata={
            "backend": "gudhi",
            "complex": "directed_clique_path_product",
            "n_simplices": int(st.num_simplices()),
            "watanabe": {
                "h1_representatives": h1_reps,
                "layer_sizes": sizes,
                "relevance_matrices": rels,
                "mode": mode,
            },
        },
    )
    stats = compute_statistics(ph_result, stat_names=stat_names)
    return TopoResult(
        tool="ph",
        ph_result=ph_result,
        statistics=stats,
        config={"complex": "directed_clique_path_product", "mode": mode,
                "max_dim": max_dim, "schedule_steps": len(_WATANABE_R)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_ph(
    bundle: GraphBundle,
    max_dim: int = 1,
    coeff: int = 2,
    backend: str = "ripser",
    epsilon: float | None = None,
    k: int | None = None,
    stat_names: list[str] | None = None,
    filtration_schedule: (
        None | int | list[float] | tuple[float, ...] | dict
    ) = None,
    schedule_units: str = "value",
    input_complex: str = "auto",
    relevance_mode: str = "positive",
    sparse: bool = False,
    **backend_kwargs,
) -> TopoResult:
    """Compute persistent homology from a ``GraphBundle``.

    Parameters
    ----------
    bundle : GraphBundle
        Input graph.  ``matrix_type`` controls filtration.
    max_dim : int
        Maximum homology dimension (default 1).
    coeff : int
        Field coefficient (default 2 = Z/2Z).
    backend : str
        ``"ripser"`` (default) | ``"gudhi"`` | ``"giotto"``.  ``"giotto"`` runs
        **giotto-ph** (``gph.ripser_parallel``, a parallel ripser), not
        giotto-tda — see the module docstring for why.
    epsilon : float or None
        Distance cut-off for ``"point_cloud"`` / ``"knn"`` input types.
    k : int or None
        Neighbours for kNN input type.
    stat_names : list[str] or None
        Statistics to compute.  Defaults to
        ``["total_persistence", "persistence_norm", "betti_curve",
        "persistence_entropy"]``.
    filtration_schedule : None, int, list[float], or dict
        Optional post-processing that snaps each (birth, death) onto a
        discrete schedule of filtration values.  Useful when matching a
        paper that defines its filtration as a fixed list of thresholds
        (e.g. Watanabe & Yamana 2021 use a 64-step log schedule).

        * ``None`` — continuous filtration, no snapping (default).
        * ``int n`` — ``n`` evenly spaced thresholds across the observed
          birth/death range.
        * ``list[float]`` — explicit threshold values.
        * ``dict`` — recipe expanded by ``build_filtration_schedule``;
          e.g. ``{"type": "log_decades", "max": 1.0, "n_decades": 7,
          "per_decade": 8}`` reproduces the Watanabe schedule.
    schedule_units : str
        Only consulted when ``filtration_schedule`` is given.

        * ``"value"`` (default) — replace each (birth, death) by the
          *value* of the nearest schedule point.  Axes stay in their
          original units.
        * ``"index"`` — replace each (birth, death) by the *index*
          (0-based) of the nearest schedule point.  Reproduces the
          paper-style integer-axis diagrams.

    input_complex : str
        ``"auto"`` (default) runs the standard VR / superlevel-set flag PH on
        the bundle's matrix.  ``"directed_clique"`` builds Watanabe's directed
        clique complex with path-product integer filtration instead — the
        bundle must carry the per-layer FCN weight matrices in
        ``metadata["per_layer_weights"]`` (the ``weight_graph`` builder stores
        them).  See :func:`compute_watanabe_ph`.
    relevance_mode : str
        Only for ``input_complex="directed_clique"``: ``"positive"`` (the
        measurement paper) or ``"absolute"`` (PHPM).
    sparse : bool
        Only for a similarity graph (``matrix_type="similarity"`` →
        superlevel filtration) with ``backend="ripser"``.  Build the
        filtration from a **sparse** distance matrix that stores only the
        graph's actual edges (non-zero similarities); absent pairs are treated
        as infinite distance and never connect.  For a layered weight graph
        (which has no intra-layer edges) this skips the dense flag complex ripser
        would otherwise build over all pairs — orders of magnitude faster with
        the same H1 births.  Absent edges no longer *close* loops, so H1 classes
        become essential (their infinite deaths are then resolved by the usual
        diagram cleaning); use it when the dense construction is too slow.

    Returns
    -------
    TopoResult
        ``tool="ph"``, ``ph_result``, ``statistics``, ``config``.
    """
    # Watanabe directed clique complex — a non-flag filtration that needs the
    # per-layer FCN weight matrices, not a distance matrix.
    if input_complex == "directed_clique":
        weights = bundle.metadata.get("per_layer_weights")
        if weights is None:
            raise ValueError(
                "input_complex='directed_clique' requires the bundle to carry "
                "metadata['per_layer_weights'] (use the weight_graph builder)."
            )
        return compute_watanabe_ph(
            weights, mode=relevance_mode, max_dim=max_dim, stat_names=stat_names,
        )

    # Resolve input_type and data from bundle
    matrix = bundle.matrix
    matrix_type = bundle.matrix_type

    if matrix_type == "distance":
        _validate_distance_matrix(matrix)
        input_type = "distance_matrix"
        data = matrix
    elif matrix_type == "similarity":
        # Pass similarity matrix directly; ripser treats it as distance
        # (superlevel-set via max - data is done by backend call)
        input_type = "superlevel"
        data = matrix
    elif matrix_type == "adjacency":
        dist = (1.0 - matrix).astype(float)
        np.fill_diagonal(dist, 0.0)
        input_type = "distance_matrix"
        data = dist
    else:
        raise ValueError(f"Unknown matrix_type '{matrix_type}'.")

    ph_result = compute_persistence(
        bundle=bundle,
        input_type=input_type,
        max_dim=max_dim,
        coeff=coeff,
        backend=backend,
        epsilon=epsilon,
        k=k,
        sparse=sparse,
        **backend_kwargs,
    )

    # Optional: snap diagrams onto a discrete filtration schedule.
    schedule_resolved: list[float] | None = None
    if filtration_schedule is not None:
        # For superlevel-set filtrations, ripser sees the data after a
        # ``max - sim`` transform, so its birth/death values live in
        # distance-space.  Snap onto a schedule that is naturally given
        # in similarity (e.g. relevance) units by first inverting the
        # diagrams back to similarity-space.
        if input_type == "superlevel":
            max_val = float(bundle.matrix.max())
            inverted: dict[int, np.ndarray] = {}
            for d, dgm in ph_result.ph_result.diagrams.items():
                if dgm.shape[0] > 0:
                    inverted[d] = max_val - dgm
                else:
                    inverted[d] = dgm.copy()
            ph_result.ph_result.diagrams = inverted

        schedule_resolved = build_filtration_schedule(
            filtration_schedule,
            diagrams=ph_result.ph_result.diagrams,
        )
        ph_result.ph_result.diagrams = snap_diagrams_to_schedule(
            ph_result.ph_result.diagrams,
            schedule_resolved,
            units=schedule_units,
        )
        ph_result.ph_result.metadata["filtration_schedule"] = list(schedule_resolved)
        ph_result.ph_result.metadata["schedule_units"] = schedule_units

    stats = compute_statistics(ph_result.ph_result, stat_names=stat_names)

    return TopoResult(
        tool="ph",
        ph_result=ph_result.ph_result,
        statistics=stats,
        config={
            "max_dim": max_dim,
            "coeff": coeff,
            "backend": backend,
            "input_type": input_type,
            "matrix_type": matrix_type,
            "epsilon": epsilon,
            "k": k,
            "filtration_schedule": schedule_resolved,
            "schedule_units": schedule_units if schedule_resolved is not None else None,
        },
        paper_reference=None,
    )


def compute_persistence(
    bundle: GraphBundle,
    input_type: str = "distance_matrix",
    max_dim: int = 1,
    coeff: int = 2,
    backend: str = "ripser",
    epsilon: float | None = None,
    k: int | None = None,
    sparse: bool = False,
    **backend_kwargs,
) -> TopoResult:
    """Low-level PH computation; dispatches to the requested backend.

    Parameters
    ----------
    bundle : GraphBundle
        ``bundle.matrix`` is the data array.
    input_type : str
        How to interpret ``bundle.matrix``:

        * ``"distance_matrix"`` — ``(N, N)`` symmetric distance matrix.
        * ``"point_cloud"``     — ``(N, D)`` point cloud; Euclidean VR.
        * ``"superlevel"``      — ``(N, N)`` similarity matrix; convert via
          ``max(data) - data`` before PH.
        * ``"knn"``             — ``(N, D)`` point cloud; build kNN graph.

    Returns
    -------
    TopoResult with ``ph_result`` populated.
    """
    data = bundle.matrix.astype(float)
    _validate_inputs(data, input_type, k)

    # Pre-process data according to input_type
    death_cap: float | None = None
    if sparse:
        if backend != "ripser":
            raise ValueError("sparse=True is only supported with backend='ripser'.")
        if input_type != "superlevel":
            raise NotImplementedError(
                "sparse=True currently supports only similarity graphs "
                "(matrix_type='similarity' → input_type='superlevel'); got "
                f"input_type={input_type!r}."
            )
        data_for_ph = _sparse_superlevel_distance(data)
        # Absent (intra-layer / non-consecutive) pairs are the edges that, in the
        # dense flag complex, close every loop at filtration value ``max(data)``.
        # Dropping them makes those H1 classes essential (death = inf); cap their
        # death at that same scale so the sparse diagram matches the dense one
        # (and stays a valid diagram — otherwise generic inf-cleaning would use
        # H0's smaller max death and push some deaths below their births).
        death_cap = float(data.max())
    else:
        data_for_ph = _preprocess(data, input_type, k)

    t0 = time.perf_counter()
    diagrams_raw = _dispatch(data_for_ph, input_type, max_dim, coeff, backend,
                             epsilon, **backend_kwargs)
    runtime = time.perf_counter() - t0

    if death_cap is not None:
        for dgm in diagrams_raw.values():
            if dgm.size:
                deaths = dgm[:, 1]
                dgm[:, 1] = np.where(np.isfinite(deaths), deaths, death_cap)

    # Clean infinite deaths
    diagrams = _clean_diagrams(diagrams_raw)

    ph_result = PersistenceResult(
        diagrams=diagrams,
        metadata={
            "runtime_seconds": runtime,
            "backend": backend,
            "input_type": input_type,
            "max_dim": max_dim,
            "coeff": coeff,
        },
    )
    return TopoResult(
        tool="ph",
        ph_result=ph_result,
        config={
            "input_type": input_type,
            "max_dim": max_dim,
            "coeff": coeff,
            "backend": backend,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Filtration schedule (post-processing)
# ─────────────────────────────────────────────────────────────────────────────

def build_filtration_schedule(
    spec: int | list[float] | tuple[float, ...] | dict,
    diagrams: dict[int, np.ndarray] | None = None,
) -> list[float]:
    """Resolve a ``filtration_schedule`` argument to a concrete list of values.

    Index direction follows the order of the returned list: index 0 is the
    *first* element.  ``snap_diagrams_to_schedule`` preserves this order, so
    callers choose whether the index axis runs from low-to-high or high-to-
    low simply by ordering the schedule.

    Accepted forms
    --------------
    * ``int n`` — ``n`` evenly spaced thresholds across the observed birth/
      death range in ``diagrams``.  Requires ``diagrams`` to be non-empty.
      Returned ascending.
    * ``list[float]`` / ``tuple[float]`` — used in the user's order.
    * ``dict`` — recipes:

      - ``{"type": "linear", "min": a, "max": b, "n": k}`` — ascending.
      - ``{"type": "log", "min": a, "max": b, "n": k}`` — geometric, ascending.
      - ``{"type": "log_decades", "max": M, "n_decades": d, "per_decade": p}``
        — Watanabe & Yamana (2021) schedule.  **Returned in descending
        order** so index 0 = the strictest (largest) threshold, matching
        the paper's plotting convention.  With ``max=1.0, n_decades=7,
        per_decade=8`` this yields exactly 64 values
        ``1.0, 0.9, ..., 0.2, 0.1, 0.09, ..., 0.02, 0.01, ..., 1e-6,
        9e-7, ..., 2e-7, 1e-7``.
    """
    if isinstance(spec, int):
        if diagrams is None:
            raise ValueError(
                "filtration_schedule=int requires diagrams to derive the range."
            )
        vals: list[float] = []
        for dgm in diagrams.values():
            if dgm.shape[0] > 0:
                vals.extend(dgm.ravel().tolist())
        if not vals:
            return [0.0]
        lo, hi = float(min(vals)), float(max(vals))
        if lo == hi:
            hi = lo + 1.0
        return np.linspace(lo, hi, spec).tolist()

    if isinstance(spec, (list, tuple)):
        return [float(x) for x in spec]

    if isinstance(spec, dict):
        kind = spec.get("type", "linear")
        if kind == "linear":
            return np.linspace(spec["min"], spec["max"], spec["n"]).tolist()
        if kind == "log":
            return np.geomspace(spec["min"], spec["max"], spec["n"]).tolist()
        if kind == "log_decades":
            M = float(spec.get("max", 1.0))
            n_dec = int(spec.get("n_decades", 7))
            per_dec = int(spec.get("per_decade", 8))
            schedule: list[float] = []
            for d in range(n_dec):
                base = M * (10.0 ** -d)
                next_base = base / 10.0
                # Decade head, plus `per_dec` interpolations strictly
                # between (base, next_base).  Next_base is the head of
                # the next decade so it is not added here.
                schedule.append(base)
                step = (base - next_base) / (per_dec + 1)
                for s in range(1, per_dec + 1):
                    schedule.append(base - s * step)
            schedule.append(M * (10.0 ** -n_dec))
            # Already descending by construction; dedupe while preserving order.
            seen: set[float] = set()
            return [x for x in schedule if not (x in seen or seen.add(x))]
        raise ValueError(f"Unknown filtration_schedule type '{kind}'.")

    raise TypeError(
        f"filtration_schedule must be None, int, list/tuple of floats, or dict; "
        f"got {type(spec).__name__}."
    )


def snap_diagrams_to_schedule(
    diagrams: dict[int, np.ndarray],
    schedule: list[float],
    units: str = "value",
) -> dict[int, np.ndarray]:
    """Snap each (birth, death) onto the nearest point of ``schedule``.

    Parameters
    ----------
    diagrams
        Dict ``{dim: (n, 2) ndarray}`` of birth/death pairs.
    schedule
        Filtration values in the user's chosen order.  Index 0 in the
        output corresponds to the first element of ``schedule``.  Pass a
        descending list to get paper-style indices for superlevel-set
        filtrations.
    units : str
        ``"value"`` -> coordinates are replaced by the matched schedule
        value.  ``"index"`` -> coordinates are replaced by the 0-based
        position of the matched schedule entry **in the order it was
        supplied**.
    """
    if units not in {"value", "index"}:
        raise ValueError(f"units must be 'value' or 'index', got '{units}'.")
    ts_orig = np.asarray(list(schedule), dtype=float)
    n = len(ts_orig)
    if n == 0:
        raise ValueError("schedule must be non-empty.")
    # Binary search on sorted values, then map back to the user's order.
    sort_perm = np.argsort(ts_orig)
    ts_sorted = ts_orig[sort_perm]

    out: dict[int, np.ndarray] = {}
    for d, dgm in diagrams.items():
        if dgm.shape[0] == 0:
            out[d] = dgm.copy()
            continue
        if n == 1:
            if units == "index":
                out[d] = np.zeros_like(dgm)
            else:
                out[d] = np.full_like(dgm, ts_orig[0])
            continue
        pos = np.clip(np.searchsorted(ts_sorted, dgm), 1, n - 1)
        left = ts_sorted[pos - 1]
        right = ts_sorted[pos]
        choose_left = (dgm - left) <= (right - dgm)
        sorted_idx = np.where(choose_left, pos - 1, pos)
        if units == "index":
            # Translate position in the sorted schedule back to position
            # in the user-supplied schedule.
            out[d] = sort_perm[sorted_idx].astype(float)
        else:
            out[d] = ts_sorted[sorted_idx]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_STATS = [
    "total_persistence",
    "persistence_norm",
    "betti_curve",
    "persistence_entropy",
]

_VALID_STATS = {
    "total_persistence",
    "persistence_norm",
    "betti_number",
    "betti_curve",
    "persistence_entropy",
    "asdsq",
    "convex_hull_area",
    "wasserstein_distance",
    "bottleneck_distance",
}


def compute_statistics(
    persistence_result: PersistenceResult,
    dims: list[int] | None = None,
    stat_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute topological summary statistics from persistence diagrams.

    Parameters
    ----------
    persistence_result : PersistenceResult
    dims : list[int] or None
        Dimensions to compute statistics for.  Defaults to all available dims.
    stat_names : list[str] or None
        Statistics to compute.  Defaults to
        ``["total_persistence", "persistence_norm", "betti_curve",
        "persistence_entropy"]``.

    Returns
    -------
    dict with keys ``"H{d}_{stat_name}"``.
    """
    if dims is None:
        dims = sorted(persistence_result.diagrams.keys())
    if stat_names is None:
        stat_names = _DEFAULT_STATS

    unknown = set(stat_names) - _VALID_STATS
    if unknown:
        raise ValueError(
            f"Unknown stat_names: {unknown}. Valid: {_VALID_STATS}."
        )

    result: dict[str, Any] = {}
    for d in dims:
        dgm = persistence_result.diagrams.get(d, np.empty((0, 2)))
        prefix = f"H{d}"
        for stat in stat_names:
            if stat == "total_persistence":
                result[f"{prefix}_total_persistence"] = total_persistence(dgm)
            elif stat == "persistence_norm":
                result[f"{prefix}_persistence_norm"] = persistence_norm(dgm)
            elif stat == "betti_curve":
                result[f"{prefix}_betti_curve"] = betti_curve(dgm)
            elif stat == "persistence_entropy":
                result[f"{prefix}_persistence_entropy"] = persistence_entropy(dgm)
            elif stat == "asdsq":
                # ASDSQ expands to 8 scalars per dimension (Ballester 2024).
                for name, val in asdsq_moments(dgm).items():
                    result[f"{prefix}_{name}"] = val
            elif stat == "convex_hull_area":
                result[f"{prefix}_convex_hull_area"] = convex_hull_area(dgm)
            elif stat == "betti_number":
                # Compute at a default epsilon (median)
                if dgm.shape[0] > 0:
                    eps = float(np.median(dgm[:, 0]))
                else:
                    eps = 0.0
                result[f"{prefix}_betti_number"] = betti_number(dgm, eps)
            # wasserstein_distance / bottleneck_distance require two diagrams;
            # skip in per-diagram statistics (use explicitly).

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Individual statistic functions
# ─────────────────────────────────────────────────────────────────────────────

def finite_lifetimes(dgm: np.ndarray) -> np.ndarray:
    """Lifetimes of the non-essential bars of a diagram.

    Essential classes have infinite death, so they have no finite lifetime to
    contribute to a sum, a mean or an entropy.  Excluding them is the
    convention used by the source papers -- Rieck et al.'s reference
    implementation passes ``unpairedData = 0.0``, giving the essential bar zero
    weight.  Count-based quantities (:func:`betti_number`, :func:`betti_curve`)
    do *not* use this: an essential class is alive at every scale and must be
    counted.
    """
    if dgm.shape[0] == 0:
        return np.zeros(0)
    life = dgm[:, 1] - dgm[:, 0]
    return life[np.isfinite(life)]


def total_persistence(dgm: np.ndarray, p: float = 1) -> float:
    """Sum of bar lifetimes^p: Σ (death_i - birth_i)^p.

    Essential (infinite-death) bars are excluded; see :func:`finite_lifetimes`.
    """
    life = finite_lifetimes(dgm)
    if life.size == 0:
        return 0.0
    return float(np.sum(life ** p))


def persistence_norm(dgm: np.ndarray, p: float = 2) -> float:
    """p-norm of bar lifetimes: (Σ (death_i - birth_i)^p)^(1/p).

    Neural Persistence metric from Rieck et al. (2019).
    Excludes infinite bars (the essential class) and any degenerate bars,
    consistent with the paper definition.
    """
    if dgm.shape[0] == 0:
        return 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    # Keep only finite positive lifetimes: excludes the essential bar (inf
    # lifetime), any NaN/inf from degenerate PH output, and zero-length bars.
    valid = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
    if valid.shape[0] == 0:
        return 0.0
    return float(np.sum(valid ** p) ** (1.0 / p))


def betti_number(dgm: np.ndarray, epsilon: float) -> int:
    """Count of bars where birth ≤ epsilon ≤ death."""
    if dgm.shape[0] == 0:
        return 0
    births = dgm[:, 0]
    deaths = dgm[:, 1]
    return int(np.sum((births <= epsilon) & (epsilon <= deaths)))


def betti_curve(dgm: np.ndarray, resolution: int = 100) -> np.ndarray:
    """Betti curve: betti_number evaluated on a linspace over the filtration."""
    if dgm.shape[0] == 0:
        return np.zeros(resolution)
    min_val = float(dgm[:, 0].min())
    finite_deaths = dgm[:, 1][np.isfinite(dgm[:, 1])]
    # The curve is sampled over the FINITE range; essential bars are still
    # counted at every epsilon by betti_number, which is what keeps b0 correct.
    max_val = float(finite_deaths.max()) if finite_deaths.size else min_val + 1.0
    if min_val == max_val:
        max_val = min_val + 1.0
    epsilons = np.linspace(min_val, max_val, resolution)
    return np.array([betti_number(dgm, eps) for eps in epsilons], dtype=float)


def convex_hull_area(dgm: np.ndarray) -> float:
    """Area of the convex hull of (birth, death) points in a diagram.

    Used by Watanabe & Yamana (2021) as a coarse measure of how spread
    out a network's H1 generators are.  Returns 0.0 when fewer than 3
    distinct points are present, or when all points are colinear.
    """
    if dgm.shape[0] < 3:
        return 0.0
    dgm = dgm[np.isfinite(dgm).all(axis=1)]     # a hull needs finite points
    if dgm.shape[0] < 3:
        return 0.0
    pts = np.unique(dgm, axis=0)
    if pts.shape[0] < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        return float(ConvexHull(pts).volume)  # in 2D, .volume is the area
    except Exception:
        return 0.0


_ASDSQ_NAMES = (
    "mean_birth", "std_birth", "mean_death", "std_death",
    "mean_birth_sq", "std_birth_sq", "mean_death_sq", "std_death_sq",
)


def asdsq_moments(dgm: np.ndarray) -> dict[str, float]:
    """The **ASDSQ** persistence summary of Ballester et al. (2024).

    Averages and standard deviations of the **births** and **deaths** of a
    diagram, concatenated with the same moments of their *element-wise squares*
    — i.e. ``mean/std`` of ``b``, ``d``, ``b²`` and ``d²``.  This was the best-
    performing vectorisation for predicting the generalization gap in the paper.

    The essential (infinite-death) bar and any non-finite points are dropped
    first.  Returns an ordered 8-entry dict; an empty diagram yields all zeros.
    Drop the ``*_sq`` keys to recover the simpler **ASD** summary.
    """
    if dgm.shape[0] == 0:
        return {n: 0.0 for n in _ASDSQ_NAMES}
    b, d = dgm[:, 0], dgm[:, 1]
    finite = np.isfinite(b) & np.isfinite(d)
    b, d = b[finite], d[finite]
    if b.shape[0] == 0:
        return {n: 0.0 for n in _ASDSQ_NAMES}
    vals = (
        float(b.mean()), float(b.std()), float(d.mean()), float(d.std()),
        float((b ** 2).mean()), float((b ** 2).std()),
        float((d ** 2).mean()), float((d ** 2).std()),
    )
    return dict(zip(_ASDSQ_NAMES, vals))


def persistence_entropy(dgm: np.ndarray) -> float:
    """- Σ p_i log(p_i) where p_i = lifetime_i / total_lifetime."""
    if dgm.shape[0] == 0:
        return 0.0
    lifetimes = finite_lifetimes(dgm)          # essential bars have no finite share
    if lifetimes.size == 0:
        return 0.0
    total = lifetimes.sum()
    if total == 0:
        return 0.0
    probs = lifetimes / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


# Ordered (key, human label) pairs for the full single-diagram statistic suite.
DIAGRAM_STAT_LABELS = {
    "n_bars":     "number of bars",
    "total":      "total persistence",
    "mean":       "mean persistence",
    "median":     "median persistence",
    "max":        "max persistence",
    "min":        "min persistence",
    "std":        "persistence std",
    "entropy":    "persistence entropy",
    "norm":       "persistence 2-norm",
    "gini":       "Gini (concentration)",
    "top1":       "top-1 dominance (max/total)",
    "skew":       "persistence skew",
    "birth_mean": "mean birth (feature scale)",
    "birth_max":  "max birth",
}


def diagram_statistics(dgm: np.ndarray) -> dict[str, float]:
    """A full suite of scalar summaries of one persistence diagram.

    Computed over finite, positive-lifetime bars (the essential class is
    expected to be already removed): bar count, total / mean / median / max /
    min / std of lifetimes, persistence entropy, persistence 2-norm, the Gini
    coefficient and top-1 dominance of the lifetime distribution (how
    concentrated the barcode is), lifetime skew, and mean / max **birth** (the
    filtration scale at which features appear).  Keys are those of
    :data:`DIAGRAM_STAT_LABELS`.

    Complements :func:`compute_statistics`, which returns a smaller,
    named-per-dimension set for a whole ``PersistenceResult``; this operates on
    a single ``(N, 2)`` diagram array and always returns every key (zeros for
    an empty diagram).
    """
    dgm = np.asarray(dgm, dtype=float).reshape(-1, 2)
    life = dgm[:, 1] - dgm[:, 0]
    keep = np.isfinite(life) & (life > 0)
    life, births = life[keep], dgm[:, 0][keep]
    if life.size == 0:
        return {k: 0.0 for k in DIAGRAM_STAT_LABELS}
    tp = float(life.sum())
    p = life / tp
    n = life.size
    sl = np.sort(life)
    gini = float(2 * np.sum(np.arange(1, n + 1) * sl) / (n * sl.sum()) - (n + 1) / n) if n > 1 else 0.0
    sd = float(life.std())
    skew = float(np.mean(((life - life.mean()) / sd) ** 3)) if sd > 0 else 0.0
    return {
        "n_bars":     float(n),
        "total":      tp,
        "mean":       float(life.mean()),
        "median":     float(np.median(life)),
        "max":        float(life.max()),
        "min":        float(life.min()),
        "std":        sd,
        "entropy":    float(-np.sum(p * np.log(p))),
        "norm":       float(np.sqrt(np.sum(life ** 2))),
        "gini":       gini,
        "top1":       float(life.max() / tp),
        "skew":       skew,
        "birth_mean": float(births.mean()),
        "birth_max":  float(births.max()),
    }


def _finite_pairs(dgm: np.ndarray) -> np.ndarray:
    """Drop essential bars so a diagram can enter a matching-based distance.

    Bottleneck and Wasserstein matchings are undefined between infinite points:
    an essential bar in one diagram has no finite counterpart to be matched to,
    and the distance would be infinite whenever the two diagrams disagree on how
    many essential classes they have.  The standard treatment is to compare the
    finite parts, so that is what happens here -- with the caveat that a
    difference in the *number* of essential classes is then invisible to these
    two distances.
    """
    if dgm is None or len(dgm) == 0:
        return np.zeros((0, 2))
    d = np.asarray(dgm, dtype=float)
    return d[np.isfinite(d).all(axis=1)]


def wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray, p: float = 2) -> float:
    """Wasserstein distance between two persistence diagrams (via persim).

    Essential bars are excluded; see :func:`_finite_pairs`.
    """
    try:
        import persim
        return float(persim.wasserstein(_finite_pairs(dgm1), _finite_pairs(dgm2),
                                        matching=False))
    except ImportError:
        raise ImportError(
            "persim is required for wasserstein_distance. "
            "Install with: pip install persim"
        )


def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    """Bottleneck distance between two persistence diagrams (via persim).

    Essential bars are excluded; see :func:`_finite_pairs`.
    """
    try:
        import persim
        return float(persim.bottleneck(_finite_pairs(dgm1), _finite_pairs(dgm2)))
    except ImportError:
        raise ImportError(
            "persim is required for bottleneck_distance. "
            "Install with: pip install persim"
        )


def compute_geodesic_betti(
    activations: np.ndarray | list[np.ndarray],
    k_range: tuple[int, int] = (2, 20),
    epsilon_resolution: int = 50,
    max_dim: int = 1,
    coeff: int = 2,
    backend: str = "ripser",
    max_iter: int = 10,
) -> dict:
    """Compute optimal Betti numbers by jointly searching k and epsilon.

    Implements the Naitzat et al. (2020) geodesic-distance approach with
    an efficient two-phase procedure:

    1. **Pre-compute phase** — compute PH once for every k in ``k_range``,
       storing the full persistence diagrams.
    2. **Search phase** — alternate between optimising k (at fixed epsilon)
       and optimising epsilon (at fixed k) until convergence.  Because all
       diagrams are pre-computed, each step is a cheap look-up.

    Stability is measured as the variance of Betti numbers in a local
    neighbourhood of the current (k, epsilon) point.  The algorithm
    converges when neither k nor epsilon changes between iterations.

    Parameters
    ----------
    activations : (N_samples, N_neurons) ndarray or list[ndarray]
        Activation matrix (or list of per-layer matrices, which are
        concatenated horizontally).
    k_range : (int, int)
        Range ``[k_min, k_max]`` to search.
    epsilon_resolution : int
        Number of epsilon values to sample across the filtration range.
    max_dim : int
        Maximum homology dimension to compute.
    coeff : int
        Field coefficient for PH (default 2 = Z/2Z).
    backend : str
        PH backend (``"ripser"`` | ``"gudhi"`` | ``"giotto"``).  ``"giotto"`` =
        giotto-ph (parallel ripser), not giotto-tda.
    max_iter : int
        Maximum alternating-optimisation iterations.

    Returns
    -------
    dict
        * ``"betti_numbers"`` — ``dict[int, int]``: Betti number per
          dimension at the optimal (k, epsilon).
        * ``"optimal_k"`` — ``int``: selected k.
        * ``"optimal_epsilon"`` — ``float``: selected epsilon.
        * ``"diagrams"`` — ``dict[int, dict[int, ndarray]]``:
          pre-computed persistence diagrams, keyed by
          ``diagrams[k][dim]``.
        * ``"betti_grid"`` — ``(len(k_values), epsilon_resolution,
          max_dim+1)`` ndarray: full Betti-number grid.
        * ``"k_values"`` — ``list[int]``: k values searched.
        * ``"epsilon_values"`` — ``(epsilon_resolution,)`` ndarray:
          epsilon values sampled.

    Notes
    -----
    This function is designed for the Naitzat et al. (2020) workflow
    where Betti numbers are tracked layer by layer.  Call it once per
    layer and compare the returned ``betti_numbers`` across layers to
    observe how topology simplifies through the network.

    Example
    -------
    ::

        from tanc.topo_tools import compute_geodesic_betti

        for name, A in zip(layer_names, activation_list):
            result = compute_geodesic_betti(A, k_range=(2, 15), max_dim=1)
            print(f"{name}: k={result['optimal_k']}, "
                  f"eps={result['optimal_epsilon']:.3f}, "
                  f"betti={result['betti_numbers']}")
    """
    from tanc.graph_builder.activation_graphs import _concat_activations
    from tanc.graph_builder.activation_graphs import _geodesic_distances

    activations = _concat_activations(activations)

    k_min, k_max = k_range
    k_max = min(k_max, activations.shape[0] - 2)
    if k_min > k_max:
        raise ValueError(
            f"k_range ({k_min}, {k_max}) is invalid for "
            f"{activations.shape[0]} samples."
        )
    k_values = list(range(k_min, k_max + 1))

    # ── Phase 1: pre-compute PH for every k ──
    all_diagrams: dict[int, dict[int, np.ndarray]] = {}
    all_dist_matrices: dict[int, np.ndarray] = {}

    for k in k_values:
        D = _geodesic_distances(activations, k)
        all_dist_matrices[k] = D
        bundle = GraphBundle(
            matrix=D,
            matrix_type="distance",
            node_features=None,
            node_labels=None,
            n_nodes=D.shape[0],
        )
        result = compute_persistence(
            bundle, max_dim=max_dim, coeff=coeff, backend=backend
        )
        all_diagrams[k] = result.ph_result.diagrams

    # ── Build a common epsilon grid across all k values ──
    eps_min = float("inf")
    eps_max = 0.0
    for k in k_values:
        D = all_dist_matrices[k]
        positive = D[D > 0]
        if len(positive) > 0:
            eps_min = min(eps_min, float(positive.min()))
            eps_max = max(eps_max, float(positive.max()))
    if eps_min >= eps_max:
        eps_min, eps_max = 0.0, 1.0
    epsilon_values = np.linspace(eps_min, eps_max, epsilon_resolution)

    # ── Build Betti-number grid: (n_k, n_eps, n_dims) ──
    n_k = len(k_values)
    n_dims = max_dim + 1
    betti_grid = np.zeros((n_k, epsilon_resolution, n_dims), dtype=int)
    for ki, k in enumerate(k_values):
        for di in range(n_dims):
            dgm = all_diagrams[k].get(di, np.empty((0, 2)))
            for ei, eps in enumerate(epsilon_values):
                betti_grid[ki, ei, di] = betti_number(dgm, eps)

    # ── Phase 2: alternating (k, epsilon) search ──
    # Stability score: variance of Betti-0 in a local neighbourhood.
    # Lower variance = more stable.
    k_half = max(1, n_k // 6)
    e_half = max(1, epsilon_resolution // 10)

    def _stability(ki: int, ei: int) -> float:
        """Variance of Betti-0 in a (2*k_half+1) x (2*e_half+1) window."""
        k_lo = max(0, ki - k_half)
        k_hi = min(n_k, ki + k_half + 1)
        e_lo = max(0, ei - e_half)
        e_hi = min(epsilon_resolution, ei + e_half + 1)
        patch = betti_grid[k_lo:k_hi, e_lo:e_hi, 0]  # Betti-0
        return float(np.var(patch))

    # Initialise at the centre of the grid
    best_ki = n_k // 2
    best_ei = epsilon_resolution // 2

    for _ in range(max_iter):
        prev_ki, prev_ei = best_ki, best_ei

        # Fix epsilon, optimise k
        scores_k = [_stability(ki, best_ei) for ki in range(n_k)]
        best_ki = int(np.argmin(scores_k))

        # Fix k, optimise epsilon
        scores_e = [_stability(best_ki, ei)
                     for ei in range(epsilon_resolution)]
        best_ei = int(np.argmin(scores_e))

        if best_ki == prev_ki and best_ei == prev_ei:
            break  # converged

    opt_k = k_values[best_ki]
    opt_eps = float(epsilon_values[best_ei])
    betti_numbers_out = {
        d: int(betti_grid[best_ki, best_ei, d]) for d in range(n_dims)
    }

    return {
        "betti_numbers": betti_numbers_out,
        "optimal_k": opt_k,
        "optimal_epsilon": opt_eps,
        "diagrams": all_diagrams,
        "betti_grid": betti_grid,
        "k_values": k_values,
        "epsilon_values": epsilon_values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_inputs(data: np.ndarray, input_type: str, k: int | None) -> None:
    valid_types = {"distance_matrix", "point_cloud", "superlevel", "knn"}
    if input_type not in valid_types:
        raise ValueError(
            f"Unknown input_type '{input_type}'. Valid: {valid_types}."
        )
    if input_type in ("distance_matrix", "superlevel"):
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError(
                f"input_type='{input_type}' requires a square (N, N) matrix, "
                f"got shape {data.shape}."
            )
    if input_type == "knn" and k is None:
        raise ValueError("input_type='knn' requires k to be specified.")


def _sparse_superlevel_distance(data: np.ndarray):
    """Sparse ``max - similarity`` distance matrix for a superlevel filtration.

    Stores only the graph's actual edges (non-zero, off-diagonal similarities);
    every absent pair is left unstored, which ripser reads as *infinite*
    distance (no edge, never connects).  This is the sparse counterpart of the
    dense ``max(data) - data`` transform in :func:`_preprocess`, and preserves
    the layered structure of a multipartite weight graph — the dense version
    instead fills every non-edge in at the final filtration value, forcing ripser
    to build the complete flag complex.

    A strongest edge (``similarity == max``) maps to distance 0; ripser reads a
    stored 0 as a length-0 edge, but a coo/csr round-trip can prune explicit
    zeros, so those are nudged to a tiny positive value to keep the edge.
    """
    from scipy import sparse as sp

    data = np.asarray(data, dtype=float)
    mx = float(data.max())
    coo = sp.coo_matrix(data)
    keep = (coo.row != coo.col) & (coo.data != 0.0)
    r, c, v = coo.row[keep], coo.col[keep], coo.data[keep]
    d = mx - v
    tiny = (abs(mx) if mx != 0.0 else 1.0) * 1e-12
    d = np.where(d <= 0.0, tiny, d)
    return sp.csr_matrix((d, (r, c)), shape=data.shape)


def _preprocess(data: np.ndarray, input_type: str, k: int | None) -> np.ndarray:
    """Convert data to the format expected by the backend."""
    if input_type == "distance_matrix":
        return data
    elif input_type == "point_cloud":
        return data
    elif input_type == "superlevel":
        max_val = data.max()
        result = max_val - data
        # Zero the diagonal: ripser treats d[i,i] as the vertex birth time, so
        # leaving max_val on the diagonal would push every vertex's birth out
        # to t=max_val, making all simplices appear simultaneously.
        np.fill_diagonal(result, 0.0)
        return result
    elif input_type == "knn":
        from sklearn.neighbors import kneighbors_graph
        G = kneighbors_graph(data, k, mode="distance")
        G_arr = G.toarray()
        G_sym = (G_arr + G_arr.T) / 2.0
        finite_mask = G_sym > 0
        if finite_mask.any():
            max_finite = G_sym[finite_mask].max()
            G_sym[G_sym == 0] = 2.0 * max_finite + 1.0
        np.fill_diagonal(G_sym, 0.0)
        return G_sym
    return data


def _dispatch(
    data: np.ndarray,
    input_type: str,
    max_dim: int,
    coeff: int,
    backend: str,
    epsilon: float | None,
    **backend_kwargs,
) -> dict[int, np.ndarray]:
    is_distance = input_type in ("distance_matrix", "superlevel", "knn")

    if backend == "ripser":
        return _run_ripser(data, max_dim, coeff, is_distance, epsilon, **backend_kwargs)
    elif backend == "gudhi":
        return _run_gudhi(data, max_dim, coeff, is_distance, epsilon, **backend_kwargs)
    elif backend == "giotto":
        return _run_giotto(data, max_dim, coeff, is_distance, epsilon, **backend_kwargs)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Valid: 'ripser', 'gudhi', 'giotto'."
        )


def _run_ripser(
    data: np.ndarray,
    max_dim: int,
    coeff: int,
    is_distance: bool,
    epsilon: float | None,
    **kwargs,
) -> dict[int, np.ndarray]:
    try:
        from ripser import ripser as _ripser
    except ImportError:
        raise ImportError("ripser is required: pip install ripser")

    call_kwargs: dict = {"maxdim": max_dim, "coeff": coeff, **kwargs}
    if is_distance:
        call_kwargs["distance_matrix"] = True
    if epsilon is not None:
        call_kwargs["thresh"] = epsilon

    result = _ripser(data, **call_kwargs)
    dgms = result["dgms"]
    return {d: np.array(dgms[d]) for d in range(len(dgms))}


def _run_gudhi(
    data: np.ndarray,
    max_dim: int,
    coeff: int,
    is_distance: bool,
    epsilon: float | None,
    **kwargs,
) -> dict[int, np.ndarray]:
    try:
        import gudhi
    except ImportError:
        raise ImportError("gudhi is required: pip install gudhi")

    if is_distance:
        rc = gudhi.RipsComplex(distance_matrix=data.tolist(), **kwargs)
    else:
        rc = gudhi.RipsComplex(points=data.tolist(), **kwargs)

    st = rc.create_simplex_tree(max_dimension=max_dim + 1)
    st.persistence(homology_coeff_field=coeff)
    pairs = st.persistence_pairs()

    diagrams: dict[int, list] = {d: [] for d in range(max_dim + 1)}
    for interval in st.persistence():
        d, (birth, death) = interval
        if d <= max_dim:
            diagrams[d].append([birth, death])

    return {d: np.array(v) if v else np.empty((0, 2)) for d, v in diagrams.items()}


def _run_giotto(
    data: np.ndarray,
    max_dim: int,
    coeff: int,
    is_distance: bool,
    epsilon: float | None,
    **kwargs,
) -> dict[int, np.ndarray]:
    """PH via **giotto-ph** (``gph.ripser_parallel``), a multithreaded C++ ripser.

    ``backend="giotto"`` uses giotto-ph, **not** giotto-tda.  giotto-tda 0.6.2
    (its last release) hard-pins ``scikit-learn==1.3.2`` and raises on modern
    scikit-learn (``check_array``'s ``force_all_finite`` was removed in sklearn
    ≥1.6), so it cannot coexist with a current environment.  giotto-ph has no
    such pin, runs on modern scikit-learn, is parallelised, and accepts the same
    dense/sparse distance-matrix input as the ``ripser`` backend.
    """
    try:
        from gph import ripser_parallel
    except ImportError:
        raise ImportError(
            "backend='giotto' now uses giotto-ph (parallel ripser); "
            "install it with: pip install giotto-ph"
        )

    call_kwargs: dict = {"maxdim": max_dim, "coeff": coeff, **kwargs}
    if is_distance:
        call_kwargs["metric"] = "precomputed"
    if epsilon is not None:
        call_kwargs["thresh"] = epsilon

    result = ripser_parallel(data, **call_kwargs)
    dgms = result["dgms"]
    return {d: np.array(dgms[d]) for d in range(len(dgms))}


def _validate_distance_matrix(M: np.ndarray, tol: float = 1e-8) -> None:
    """Refuse a matrix that is not a distance matrix.

    Ripser reads only one triangle, so an asymmetric matrix is silently
    half-ignored and a negative entry produces a filtration that runs backwards.
    Both give a diagram that looks ordinary and means nothing, so this raises
    rather than repairing: a matrix failing these checks reflects a mistake
    upstream, and guessing which triangle was intended would hide it.

    Raises
    ------
    ValueError
        If *M* is not square, not symmetric, has negative entries, a non-zero
        diagonal, or contains NaN.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(
            f"A distance matrix must be square; got shape {M.shape}. If this is "
            f"a point cloud, build a bundle with matrix_type='distance' first."
        )
    if np.isnan(M).any():
        raise ValueError(
            f"The distance matrix contains {int(np.isnan(M).sum())} NaN entries. "
            "NaN distances cannot be ordered, so the filtration is undefined."
        )
    if not np.allclose(M, M.T, atol=tol, equal_nan=False):
        worst = float(np.abs(M - M.T).max())
        raise ValueError(
            f"The distance matrix is not symmetric (largest |M - M.T| = {worst:.3g}). "
            "Persistent homology needs a symmetric matrix; ripser would silently "
            "use one triangle and ignore the other. Symmetrise it deliberately "
            "-- e.g. (M + M.T) / 2 -- rather than leaving the choice implicit."
        )
    if (M < -tol).any():
        raise ValueError(
            f"The distance matrix has {int((M < -tol).sum())} negative entries "
            f"(minimum {float(M.min()):.3g}). Distances must be non-negative; a "
            "negative entry makes the filtration run backwards. If these are "
            "similarities, pass matrix_type='similarity' instead."
        )
    diag = np.abs(np.diag(M))
    if (diag > tol).any():
        raise ValueError(
            f"The distance matrix has a non-zero diagonal (max {float(diag.max()):.3g}). "
            "A point must be at distance 0 from itself."
        )


def _clean_diagrams(
    diagrams_raw: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Repair degenerate entries, **keeping essential classes infinite**.

    A Vietoris-Rips diagram has one bar per essential class -- the connected
    component that never dies, and any higher cycle that never fills in -- and
    its death is genuinely infinite.  Replacing that infinity with a finite
    number makes the bar indistinguishable from an ordinary merge, which is
    wrong in two visible ways: ``betti_number`` then reports ``b0 = 0`` above the
    largest merge scale (impossible for a non-empty space), and lifetime sums
    silently gain a spurious long bar.

    Infinite deaths are therefore preserved.  Only genuinely malformed entries
    are repaired: non-finite *births*, and NaN deaths, which no valid diagram
    contains.  Consumers decide what to do with the infinities --
    :func:`betti_number` and :func:`betti_curve` count them (an essential class
    is alive at every scale), while the lifetime-based statistics exclude them,
    which matches the convention in the source papers.
    """
    cleaned: dict[int, np.ndarray] = {}
    for d, dgm in diagrams_raw.items():
        if dgm.shape[0] == 0:
            cleaned[d] = dgm.copy()
            continue
        dgm = dgm.copy()
        # Births are always finite in a valid VR diagram; repair defensively.
        births = dgm[:, 0]
        dgm[:, 0] = np.where(np.isfinite(births), births, 0.0)
        # Deaths: +inf is meaningful (essential) and is kept.  NaN is not, and
        # becomes +inf so the bar is treated as essential rather than dropped.
        deaths = dgm[:, 1]
        dgm[:, 1] = np.where(np.isnan(deaths), np.inf, deaths)
        cleaned[d] = dgm

    return cleaned
