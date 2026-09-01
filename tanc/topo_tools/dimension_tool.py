"""dimension_tool.py — intrinsic dimension estimation for activations and trajectories.

Source papers
-------------
* Ong et al. (2026) — Universal 2NN Estimator: global log-log 2NN.
* Ruppik et al. (2025) — Less is More: local 2NN on subsampled neighbourhoods.
* Birdal et al. (2021) — Intrinsic Dim & PH: VR filtration scaling.
* Dupuis et al. (2023) — Data-Dependent Fractal: loss-PH dimension.
* Andreeva et al. (2024) — Topological Gen. Bounds: magnitude dimension.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from tanc.graph_builder._bundle import GraphBundle
from tanc.topo_tools._result import TopoResult
from tanc.topo_tools.ph_tool import finite_lifetimes


# ─────────────────────────────────────────────────────────────────────────────
# 2NN utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_2nn_ratios(
    X: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """Compute the 2NN distance ratios μ_i = r2_i / r1_i for each point.

    Parameters
    ----------
    X : (N, N) ndarray
        Distance matrix for one layer.
    n_jobs : int

    Returns
    -------
    ratios : (N,) ndarray
        μ_i = r2_i / r1_i ≥ 1.

    Raises
    ------
    ValueError : N < 3.
    """
    N = X.shape[0]
    if N < 3:
        raise ValueError(f"N must be >= 3 for 2NN ratios, got N={N}.")

    sorted_dists = np.sort(X, axis=1)
    r1 = sorted_dists[:, 1]
    r2 = sorted_dists[:, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(r1 > 0, r2 / r1, 1.0)

    return ratios.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Global 2NN estimator (Ong et al., 2026)
# ─────────────────────────────────────────────────────────────────────────────

_EULER_GAMMA = 0.5772156649015329


def estimate_id_global(
    X: np.ndarray,
    n_jobs: int = 1,
) -> dict:
    """Estimate intrinsic dimensionality via the 2NN log-log estimator.

    Uses the second-to-first nearest-neighbour ratio ``μ = r₂/r₁``.  On a
    ``d``-dimensional manifold ``log μ`` is exponentially distributed with rate
    ``d``, so ``E[log log μ] = -γ - log d`` (γ = Euler-Mascheroni).  Inverting
    gives the **intrinsic dimension**

        ``id_estimate = exp(-γ - mean(log log μ))``.

    Returns dict with keys ``"id_estimate"`` (the dimension), ``"loglog_mean"``
    (the raw statistic), ``"loglog_ratios"``, ``"ratios"``, ``"n_valid"``,
    ``"method"``.
    """
    ratios = compute_2nn_ratios(X, n_jobs=n_jobs)
    valid_mask = ratios > 1.0
    valid_ratios = ratios[valid_mask]
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        loglog_ratios = np.array([])
        loglog_mean = 0.0
        id_estimate = 0.0
    else:
        loglog_ratios = np.log(np.log(valid_ratios))
        loglog_mean = float(np.mean(loglog_ratios))
        # Invert E[log log μ] = -γ - log d  →  d = exp(-γ - mean).
        id_estimate = float(np.exp(-_EULER_GAMMA - loglog_mean))

    return {
        "id_estimate": id_estimate,
        "loglog_mean": loglog_mean,
        "loglog_ratios": loglog_ratios,
        "ratios": ratios,
        "n_valid": n_valid,
        "method": "global_2nn",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Calibrated (k, j) estimator (Ong et al., 2026)
# ─────────────────────────────────────────────────────────────────────────────
#
# The asymptotic estimator above is exact only as N → ∞.  Ong et al. replace the
# fixed analytic constants with a *sample-size-calibrated* map: for each point
# they form  L_{k,j}(x) = -log log( R_k(x) / R_j(x) )  (R_m = m-th NN distance,
# k > j), average to  Lbar_{k,j}(X),  and estimate
#
#     d̂_{k,j}(X) = exp( α_{k,j}^{(n)} · Lbar_{k,j}(X) + β_{k,j}^{(n)} )
#
# where α, β are fitted (per sample size n and per (k, j)) against synthetic
# isotropic-Gaussian clouds of *known* dimension.

def _knn_L_statistic(X: np.ndarray, k: int, j: int) -> tuple[float, int]:
    """Return ``(mean_x L_{k,j}(x), n_valid)`` from a distance matrix ``X``.

    ``R_m`` is the ``m``-th nearest-neighbour distance (self excluded), so with
    a distance matrix the m-th neighbour is column ``m`` of the row-sorted
    matrix (column 0 is the zero self-distance).
    """
    sorted_d = np.sort(X, axis=1)
    Rj = sorted_d[:, j]
    Rk = sorted_d[:, k]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(Rj > 0, Rk / Rj, np.nan)
    valid = np.isfinite(ratio) & (ratio > 1.0)
    if not valid.any():
        return float("nan"), 0
    L = -np.log(np.log(ratio[valid]))
    return float(np.mean(L)), int(valid.sum())


_CALIB_CACHE: dict[tuple, tuple[float, float]] = {}


def _calibrate_alpha_beta(
    n: int, k: int, j: int,
    calib_dims: tuple[int, ...], n_repeats: int, seed: int,
) -> tuple[float, float]:
    """Fit ``(α, β)`` so ``log d ≈ α·Lbar + β`` on Gaussian clouds of known dim.

    Results are cached by ``(n, k, j, calib_dims, n_repeats, seed)`` because the
    calibration only depends on the sample size, not on the data.
    """
    key = (n, k, j, calib_dims, n_repeats, seed)
    if key in _CALIB_CACHE:
        return _CALIB_CACHE[key]

    from sklearn.metrics import pairwise_distances
    rng = np.random.default_rng(seed)
    Ls: list[float] = []
    logd: list[float] = []
    for d in calib_dims:
        for _ in range(n_repeats):
            pts = rng.standard_normal((n, d))
            Lbar, _ = _knn_L_statistic(pairwise_distances(pts), k, j)
            if np.isfinite(Lbar):
                Ls.append(Lbar); logd.append(np.log(d))

    if len(Ls) < 2:
        # Degenerate calibration — fall back to the asymptotic constants
        # (α = -1, β = -γ), i.e. d = exp(-γ - Lbar)·... — see estimate_id_global.
        ab = (-1.0, -_EULER_GAMMA)
    else:
        A = np.vstack([np.asarray(Ls), np.ones(len(Ls))]).T
        alpha, beta = np.linalg.lstsq(A, np.asarray(logd), rcond=None)[0]
        ab = (float(alpha), float(beta))
    _CALIB_CACHE[key] = ab
    return ab


def estimate_id_calibrated(
    X: np.ndarray,
    k: int = 2,
    j: int = 1,
    calib_dims: tuple[int, ...] | None = None,
    n_calib_repeats: int = 5,
    seed: int = 0,
    n_jobs: int = 1,          # accepted for a uniform estimator signature
) -> dict:
    """Calibrated (k, j) intrinsic-dimension estimator (Ong et al., 2026).

    Parameters
    ----------
    X : (N, N) ndarray
        Distance matrix for one layer.
    k, j : int
        Neighbour ranks with ``k > j`` (default 2, 1 recovers the 2NN ratio).
    calib_dims : tuple of int or None
        Known dimensions simulated to fit ``α, β``.  ``None`` → ``1..20``.
    n_calib_repeats : int
        Gaussian clouds per calibration dimension.

    Returns dict with ``"id_estimate"``, ``"loglog_mean"`` (Lbar), ``"alpha"``,
    ``"beta"``, ``"k"``, ``"j"``, ``"n_valid"``, ``"method"``.
    """
    if k <= j:
        raise ValueError(f"require k > j, got k={k}, j={j}.")
    N = X.shape[0]
    if N <= k:
        raise ValueError(f"need N > k for the {k}-th neighbour, got N={N}, k={k}.")
    if calib_dims is None:
        calib_dims = tuple(range(1, 21))

    Lbar, n_valid = _knn_L_statistic(X, k, j)
    alpha, beta = _calibrate_alpha_beta(N, k, j, calib_dims, n_calib_repeats, seed)
    id_estimate = float(np.exp(alpha * Lbar + beta)) if np.isfinite(Lbar) else 0.0

    return {
        "id_estimate": id_estimate,
        "loglog_mean": Lbar,
        "alpha": alpha,
        "beta": beta,
        "k": k,
        "j": j,
        "n_valid": n_valid,
        "method": "calibrated_2nn",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Local 2NN estimator (Ruppik et al., 2025)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_id_local(
    X: np.ndarray,
    n_subsample: int = 100,
    n_neighbours_local: int = 50,
    n_repeats: int = 10,
    seed: int = 0,
    n_jobs: int = 1,
) -> dict:
    """Estimate local intrinsic dimensionality via 2NN on subsampled
    local neighbourhoods (Ruppik et al., 2025).

    Returns dict with keys ``"id_mean"``, ``"id_std"``,
    ``"id_per_anchor"``, ``"method"``.
    """
    if n_neighbours_local < 3:
        raise ValueError(f"n_neighbours_local must be >= 3, got {n_neighbours_local}.")

    N = X.shape[0]
    rng = np.random.default_rng(seed)
    id_per_anchor: list[float] = []

    for _ in range(n_repeats):
        n_sample = min(n_subsample, N)
        anchors = rng.choice(N, size=n_sample, replace=False)
        for anchor in anchors:
            dists_to_anchor = X[anchor]
            n_nbrs = min(n_neighbours_local, N - 1)
            nbr_indices = np.argsort(dists_to_anchor)[1 : n_nbrs + 1]
            local_indices = np.concatenate([[anchor], nbr_indices])
            local_D = X[np.ix_(local_indices, local_indices)]
            if local_D.shape[0] < 3:
                continue
            local_result = estimate_id_global(local_D, n_jobs=n_jobs)
            id_per_anchor.append(local_result["id_estimate"])

    id_arr = np.array(id_per_anchor, dtype=float)
    return {
        "id_mean": float(np.mean(id_arr)) if len(id_arr) > 0 else 0.0,
        "id_std": float(np.std(id_arr)) if len(id_arr) > 0 else 0.0,
        "id_per_anchor": id_arr,
        "method": "local_2nn",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Layer-sweep utility
# ─────────────────────────────────────────────────────────────────────────────

def _to_distance_matrix(arr: np.ndarray) -> np.ndarray:
    """Return a square pairwise distance matrix.

    ``arr`` is treated as an already-computed distance matrix only when it is
    square **and** looks like one — symmetric with a (near-)zero diagonal.
    Otherwise it is treated as a raw point cloud ``(N_samples, N_features)`` and
    pairwise Euclidean distances are computed.

    The symmetry/zero-diagonal check matters because a point cloud can be square
    by coincidence (e.g. a layer with ``N_neurons == N_probe_samples``); guessing
    from the shape alone would then feed raw activations to the estimator as if
    they were distances and collapse the intrinsic-dimension estimate.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        looks_like_distance = (
            np.allclose(np.diagonal(arr), 0.0, atol=1e-8)
            and np.allclose(arr, arr.T, atol=1e-8)
        )
        if looks_like_distance:
            return arr
    from sklearn.metrics import pairwise_distances
    return pairwise_distances(arr, metric="euclidean")


def run_activation_id(
    activations,
    method: str = "global",
    layer_labels: list[str] | None = None,
    **method_kwargs,
) -> dict:
    """Estimate intrinsic dimension for each layer independently.

    Parameters
    ----------
    activations : list of (N, N) distance ndarrays **or** (N, D) activation
        arrays, or a single GraphBundle.  Raw activation arrays are
        automatically converted to pairwise Euclidean distance matrices.
    method : str
        ``"global"`` (asymptotic 2NN) | ``"local"`` (Ruppik) |
        ``"calibrated"`` (Ong et al. 2026: sample-size-calibrated (k, j)).
    layer_labels : list[str] or None

    Returns dict with keys ``"id_estimates"``, ``"layer_labels"``,
    ``"details"``, ``"method"``.
    """
    if isinstance(activations, GraphBundle):
        activations = [activations.matrix]

    n_layers = len(activations)
    if layer_labels is None:
        layer_labels = [f"Layer {i}" for i in range(n_layers)]

    _estimators = {
        "global": estimate_id_global,
        "local": estimate_id_local,
        "calibrated": estimate_id_calibrated,
    }
    if method not in _estimators:
        raise ValueError(
            f"Unknown method '{method}'. Valid: {sorted(_estimators)}."
        )
    estimator = _estimators[method]
    id_estimates: list[float] = []
    details: list[dict] = []
    for layer_data in activations:
        dist_mat = _to_distance_matrix(np.asarray(layer_data))
        res = estimator(dist_mat, **method_kwargs)
        id_estimates.append(res.get("id_estimate", res.get("id_mean", 0.0)))
        details.append(res)

    return {
        "id_estimates": id_estimates,
        "layer_labels": layer_labels,
        "details": details,
        "method": method,
    }


estimate_id_across_layers = run_activation_id


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory dimension wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_trajectory_dimension(
    trajectory: GraphBundle,
    method: str = "ph_dimension",
    loss_values: np.ndarray | None = None,
    **method_kwargs,
) -> TopoResult:
    """Estimate PH / magnitude dimension of a weight trajectory.

    Returns TopoResult(tool="dimension", dimension_result=...).
    """
    D = trajectory.matrix

    if method == "ph_dimension":
        dim_result = compute_ph_dimension(D, loss_values=loss_values, **method_kwargs)
    elif method in ("magnitude_dimension", "magnitude"):
        dim_result = compute_magnitude_dimension(D, **method_kwargs)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Valid: 'ph_dimension', 'magnitude_dimension'."
        )

    return TopoResult(
        tool="dimension",
        dimension_result=dim_result,
        config={"method": method},
    )


# ─────────────────────────────────────────────────────────────────────────────
# PH dimension (Birdal et al., 2021 / Dupuis et al., 2023)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ph_dimension(
    trajectory: np.ndarray,
    alpha: float = 1.0,
    subset_sizes: list[int] | None = None,
    loss_values: np.ndarray | None = None,
    seed: int = 0,
    backend: str = "ripser",
    distance_metric: str = "euclidean",
) -> dict:
    """Estimate the PH fractal dimension (Birdal et al. 2021 / Dupuis et al. 2023).

    Parameters
    ----------
    trajectory : (T, T) ndarray — pairwise distance matrix.
    alpha : float — lifetime sum exponent (default 1).
    subset_sizes : list[int] or None — default ``range(200, 1000, 50)`` clipped to
        T, matching Birdal et al.'s reference implementation.  Short trajectories
        fall back to ``[50, 100, 200, 500]``, then to a log-spaced schedule.
    loss_values : (T,) ndarray or None.
    seed, backend, distance_metric : see spec.

    Returns dict with keys ``"ph_dimension"``, ``"slope"``, ``"alpha"``,
    ``"e_alpha_values"``, ``"subset_sizes"``, ``"log_n"``,
    ``"log_e_alpha"``, ``"distance_metric"``, ``"method"``.
    """
    from tanc.topo_tools.ph_tool import compute_persistence

    T = trajectory.shape[0]
    if T < 10:
        raise ValueError(f"trajectory.shape[0]={T} must be >= 10.")

    if subset_sizes is None:
        # Birdal et al.'s reference implementation sweeps n from 200 to 1000 in
        # steps of 50 (`calculate_ph_dim(min_points=200, max_points=1000,
        # point_jump=50)`).  The floor matters: E_alpha(n) ~ n^((d-alpha)/d) is
        # asymptotic, and at small n a Vietoris-Rips diagram is dominated by
        # boundary effects, so those points sit off the line and drag the fitted
        # slope down.  Since d = alpha/(1-slope) is steep near slope -> 1, a small
        # downward bias in the slope becomes a large one in d.
        subset_sizes = [s for s in range(200, 1001, 50) if s <= T]
        if len(subset_sizes) < 2:
            # Too short for the reference grid — fall back to the coarse schedule,
            # which reaches lower n at the cost of some downward bias.
            subset_sizes = [s for s in [50, 100, 200, 500] if s <= T]
        if len(subset_sizes) < 2:
            # Trajectory too short for the default subset sizes — without this
            # guard the scaling fit has < 2 points and ph_dimension is silently
            # NaN.  Fall back to an adaptive log-spaced schedule and warn loudly.
            subset_sizes = sorted({
                int(round(s)) for s in np.geomspace(max(10, T // 8), T, num=5)
            })
            warnings.warn(
                f"PH-dimension: trajectory has only T={T} points, too few for "
                "the default subset sizes [50, 100, 200, 500].  Falling back to "
                f"adaptive sizes {subset_sizes}.  This estimate will be noisy — "
                "the method assumes a long trajectory (the paper uses thousands "
                "of SGD iterates); capture more snapshots for a faithful value.",
                stacklevel=2,
            )
    else:
        too_large = [s for s in subset_sizes if s > T]
        if too_large:
            raise ValueError(f"subset_sizes {too_large} exceed T={T}.")

    if distance_metric == "loss_difference" and loss_values is None:
        raise ValueError("distance_metric='loss_difference' requires loss_values.")

    rng = np.random.default_rng(seed)
    e_alpha_values: list[float] = []

    for n in subset_sizes:
        indices = rng.choice(T, size=n, replace=False)
        sub_D = trajectory[np.ix_(indices, indices)]

        if distance_metric == "loss_difference" and loss_values is not None:
            loss_sub = np.asarray(loss_values)[indices]
            sub_D = np.abs(loss_sub[:, None] - loss_sub[None, :])

        bundle = GraphBundle(
            matrix=sub_D, matrix_type="distance",
            node_features=None, node_labels=None, n_nodes=n,
        )
        result = compute_persistence(bundle, input_type="distance_matrix",
                                     max_dim=0, backend=backend)
        dgm = result.ph_result.diagrams.get(0, np.empty((0, 2)))
        if dgm.shape[0] == 0:
            e_alpha_values.append(0.0)
        else:
            # E_alpha(W) = sum over PH0 bars of |I(gamma)|^alpha (Birdal et al.
            # Def. / Dupuis et al. Eq. 25).  The essential class has infinite
            # death and therefore no finite lifetime to contribute; including it
            # would make every E_alpha infinite and the scaling fit undefined.
            lifetimes = finite_lifetimes(dgm)
            e_alpha_values.append(float(np.sum(lifetimes ** alpha)) if lifetimes.size else 0.0)

    valid = [(n, e) for n, e in zip(subset_sizes, e_alpha_values) if e > 0 and np.isfinite(e)]
    log_n_all = np.log(np.array(subset_sizes, dtype=float))
    log_e_all = np.array(
        [np.log(e) if e > 0 else float("nan") for e in e_alpha_values]
    )

    if len(valid) < 2:
        return {
            "ph_dimension": float("nan"), "slope": float("nan"), "alpha": alpha,
            "e_alpha_values": np.array(e_alpha_values), "subset_sizes": list(subset_sizes),
            "log_n": log_n_all, "log_e_alpha": log_e_all,
            "distance_metric": distance_metric, "method": "ph_dimension",
        }

    valid_n, valid_e = zip(*valid)
    b, _ = np.polyfit(np.log(np.array(valid_n, float)), np.log(np.array(valid_e, float)), 1)

    if b >= 1:
        # E_alpha(n) scales as n^((d - alpha)/d), so the log-log slope is
        # b = 1 - alpha/d and the dimension is d = alpha/(1 - b) — Algorithm 1,
        # line 10 of Birdal et al. (2021), and `return 1 / (1 - m)` in their
        # reference implementation, which fixes alpha = 1.  Any b < 1 gives a
        # positive d; only b >= 1 is undefined (it would need d <= 0), and that
        # means the lifetime-sum scaling did not hold.
        raise ValueError(
            f"Regression slope b={b:.4f} is >= 1, so the PH dimension "
            f"d = alpha/(1-b) is undefined (it would be negative). The "
            "lifetime-sum scaling failed — the trajectory is likely too short or "
            "degenerate (the method assumes a long trajectory; capture more "
            "snapshots), or alpha is too large."
        )

    return {
        "ph_dimension": float(alpha / (1.0 - b)), "slope": float(b), "alpha": alpha,
        "e_alpha_values": np.array(e_alpha_values), "subset_sizes": list(subset_sizes),
        "log_n": log_n_all, "log_e_alpha": log_e_all,
        "distance_metric": distance_metric, "method": "ph_dimension",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Magnitude dimension (Andreeva et al., 2024)
# ─────────────────────────────────────────────────────────────────────────────

def _magnitude(D: np.ndarray, t: float) -> float:
    """Magnitude |tX| of a finite metric space at scale ``t``.

    Solves ``Z w = 1`` for the similarity matrix ``Z = exp(-t D)`` and sums the
    weights.  ``lstsq`` rather than ``solve`` because ``Z`` becomes numerically
    singular at small ``t`` (every entry tends to 1), where the least-squares
    solution is still the right answer.
    """
    Z = np.exp(-t * D)
    w, *_ = np.linalg.lstsq(Z, np.ones(D.shape[0]), rcond=None)
    return float(w.sum())


def compute_magnitude_dimension(
    trajectory: np.ndarray,
    t_range: tuple[float, float] | None = None,
    n_scale: int = 30,
    seed: int = 0,
) -> dict:
    """Estimate the magnitude dimension (Andreeva et al., 2024).

    Magnitude ``|tX|`` interpolates between 1 as ``t -> 0`` and the cardinality
    ``|X|`` as ``t -> infinity``.  The dimension is the growth rate of
    ``log |tX|`` against ``log t`` in the **intermediate** regime, and it is
    read as the *maximum* of the instantaneous slope profile:

        ``dim = max_t  d log |tX| / d log t``

    Two details matter and were both wrong under a fixed range with one global
    regression.  First, the scaling window depends on the data -- it sits where
    ``t`` is comparable to the inverse of the typical distance -- so the scales
    are derived from the distance matrix rather than fixed.  Second, a single
    regression averages the rise and the saturation together and therefore
    always under-estimates; the reference implementation
    (``aidos-lab/magnipy``) takes finite differences and returns the profile
    maximum, which is what happens here.

    Parameters
    ----------
    trajectory : (T, T) ndarray
        A distance matrix.
    t_range : (float, float), optional
        Scale range.  ``None`` (default) derives it from the data, spanning
        roughly ``1/median_distance`` to well past the point of saturation.
        Pass an explicit range only to reproduce a specific published sweep.
    n_scale : int
        Number of scales sampled log-uniformly across the range.
    seed : int
        Accepted for a uniform estimator signature; the computation is
        deterministic.

    Returns
    -------
    dict
        ``"magnitude_dimension"`` (the profile maximum), ``"magnitude_values"``,
        ``"scale_values"``, ``"dimension_profile"`` (the instantaneous slopes),
        ``"t_at_max"``, ``"r_squared"``, ``"method"``.

    Raises
    ------
    ValueError
        If the matrix is not square, or too few scales yield a usable magnitude.
    """
    D = np.asarray(trajectory, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"trajectory must be a square distance matrix; got {D.shape}.")
    T = D.shape[0]

    if t_range is None:
        # Scale the sweep to the data: the growth window sits near the inverse
        # of the typical inter-point distance, and saturation follows a couple
        # of decades later.
        off = D[~np.eye(T, dtype=bool)]
        typ = float(np.median(off[off > 0])) if (off > 0).any() else 1.0
        t_lo, t_hi = 0.01 / typ, 100.0 / typ
    else:
        t_lo, t_hi = float(t_range[0]), float(t_range[1])

    t_values = np.logspace(np.log10(t_lo), np.log10(t_hi), n_scale)
    mag_values: list[float] = []
    valid_t: list[float] = []
    for t in t_values:
        try:
            m = _magnitude(D, t)
        except np.linalg.LinAlgError:
            continue
        # Magnitude lies in [1, T] for a positive-definite space; values outside
        # signal numerical breakdown at that scale rather than a real quantity.
        if np.isfinite(m) and 0.0 < m <= T * 1.05:
            mag_values.append(m)
            valid_t.append(float(t))

    if len(valid_t) < 5:
        raise ValueError(
            f"Only {len(valid_t)} usable scales out of {n_scale}. The distance "
            "matrix may be degenerate (duplicate points give a singular "
            "similarity matrix); pass an explicit t_range to target a narrower "
            "window."
        )

    log_t = np.log(np.asarray(valid_t))
    log_mag = np.log(np.asarray(mag_values))
    # Instantaneous slope by central differences on the log-log profile.
    profile = np.gradient(log_mag, log_t)
    i_max = int(np.nanargmax(profile))
    dimension = float(profile[i_max])

    # A local goodness-of-fit: how linear the profile is around its maximum.
    lo, hi = max(0, i_max - 2), min(len(log_t), i_max + 3)
    if hi - lo >= 3:
        c = np.polyfit(log_t[lo:hi], log_mag[lo:hi], 1)
        resid = log_mag[lo:hi] - np.polyval(c, log_t[lo:hi])
        ss_tot = float(np.sum((log_mag[lo:hi] - log_mag[lo:hi].mean()) ** 2))
        r_squared = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 1.0
    else:
        r_squared = float("nan")

    return {
        "magnitude_dimension": dimension,
        "magnitude_values": np.asarray(mag_values),
        "scale_values": np.asarray(valid_t),
        "dimension_profile": profile,
        "t_at_max": float(valid_t[i_max]),
        "r_squared": r_squared,
        "method": "magnitude_dimension",
    }

def plot_id_across_layers(layer_result: dict, ax=None, title: str | None = None):
    """Line plot of intrinsic dimension vs layer index (± std shading for local)."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    labels = layer_result["layer_labels"]
    ids = layer_result["id_estimates"]
    method = layer_result.get("method", "global")
    x = list(range(len(labels)))

    ax.plot(x, ids, marker="o", linewidth=2)
    if method == "local":
        details = layer_result.get("details", [])
        stds = np.array([d.get("id_std", 0.0) for d in details])
        ids_arr = np.array(ids)
        ax.fill_between(x, ids_arr - stds, ids_arr + stds, alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Intrinsic Dimension")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_2nn_ratio_distribution(
    ratios: np.ndarray, id_estimate: float | None = None,
    ax=None, title: str | None = None,
):
    """Histogram of μ = r2/r1 ratios."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.hist(ratios, bins=30, edgecolor="black", alpha=0.75)
    if id_estimate is not None:
        ax.axvline(id_estimate, color="red", linestyle="--",
                   label=f"ID = {id_estimate:.3f}")
        ax.legend()
    ax.set_xlabel("μ = r2 / r1")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_loglog_ratio_distribution(
    result: dict, ax=None, title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Histogram of log(log(μ)) for the global 2NN estimator.

    Raises ValueError if result['method'] != 'global_2nn'.
    """
    import matplotlib.pyplot as plt

    if result.get("method") != "global_2nn":
        raise ValueError(
            f"plot_loglog_ratio_distribution requires method='global_2nn', "
            f"got '{result.get('method')}'."
        )
    if figsize is None:
        figsize = (7, 4)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    loglog = result["loglog_ratios"]
    id_est = result["id_estimate"]
    n_valid = result["n_valid"]

    ax.hist(loglog, bins=30, edgecolor="black", alpha=0.75)
    ax.axvline(id_est, color="red", linestyle="--",
               label=f"ID = {id_est:.3f} (n={n_valid})")
    ax.legend()
    ax.set_xlabel("log(log(μ))")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_ph_scaling(ph_result: dict, ax=None, title: str | None = None):
    """Log-log scatter of α-lifetime sum vs subset size with regression."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    log_n = ph_result["log_n"]
    log_e = ph_result["log_e_alpha"]
    alpha = ph_result["alpha"]
    ph_dim = ph_result.get("ph_dimension", float("nan"))

    valid = np.isfinite(log_e)
    ax.scatter(log_n[valid], log_e[valid], label=f"α={alpha:.1f}")
    if valid.sum() >= 2:
        b, c = np.polyfit(log_n[valid], log_e[valid], 1)
        x_fit = np.linspace(log_n[valid].min(), log_n[valid].max(), 50)
        ax.plot(x_fit, b * x_fit + c, linestyle="--", alpha=0.8)

    ax.set_xlabel("log n")
    ax.set_ylabel("log S(n, α)")
    ax.text(0.05, 0.95, f"PH dim = {ph_dim:.3f}", transform=ax.transAxes, va="top")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_magnitude_scaling(mag_result: dict, ax=None, title: str | None = None):
    """Log-log scatter of Mag(t) vs t with fitted regression line."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    t_vals = mag_result["scale_values"]
    mag_vals = mag_result["magnitude_values"]
    mag_dim = mag_result.get("magnitude_dimension", float("nan"))
    r2 = mag_result.get("r_squared", float("nan"))

    log_t = np.log(t_vals)
    log_mag = np.log(np.abs(mag_vals))
    ax.scatter(log_t, log_mag, label="Mag(tX)")

    upper = len(t_vals) // 2
    if len(t_vals[upper:]) >= 2:
        b, c = np.polyfit(log_t[upper:], log_mag[upper:], 1)
        x_fit = np.linspace(log_t[upper:].min(), log_t[upper:].max(), 50)
        ax.plot(x_fit, b * x_fit + c, linestyle="--", label=f"slope={b:.3f}")

    ax.set_xlabel("log t")
    ax.set_ylabel("log Mag(tX)")
    ax.text(0.05, 0.95, f"Mag dim = {mag_dim:.3f}\nR² = {r2:.3f}",
            transform=ax.transAxes, va="top")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# WeightTrajectoryAnalyser / TrajectoryDimensionResult
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryDimensionResult:
    """Simple container for trajectory dimension results."""

    def __init__(self, result_dict: dict) -> None:
        self._dict = result_dict

    def __getattr__(self, name: str):
        try:
            return self._dict[name]
        except KeyError:
            raise AttributeError(f"No attribute '{name}'.")


class WeightTrajectoryAnalyser:
    """Convenience wrapper: applies trajectory dimension methods to a bundle.

    Builder-style class matching the rest of the toolkit's conventions:

    * :meth:`from_paper` — preconfigure for a published trajectory-dimension
      method (``birdal2021``, ``dupuis2023``, ``andreeva2024``).
    * :meth:`explain`  — describe the configured plan in plain English.
    * :meth:`fit`      — run whichever method was set by ``from_paper``
                         (or any method explicitly named via ``method=``).
    * :meth:`ph_dimension` / :meth:`magnitude_dimension` — call the
      respective estimator directly without going through ``from_paper``.

    Parameters
    ----------
    bundle : GraphBundle or None
        Pairwise-distance trajectory bundle.  May be left ``None`` at
        construction and supplied later via ``.fit(bundle=...)``.
    loss_values : ndarray or None
        Required for ``method="ph_loss"`` / ``"loss_difference"``.
    """

    def __init__(
        self,
        bundle: GraphBundle | None = None,
        loss_values: np.ndarray | None = None,
    ) -> None:
        self.bundle = bundle
        self.loss_values = loss_values
        # Populated by ``from_paper`` / ``configure``; consulted by ``fit``.
        self._default_method: str | None = None
        self._method_kwargs: dict = {}
        self.paper_reference: str | None = None

    # ── Alternative constructor ────────────────────────────────────────────

    @classmethod
    def from_paper(cls, name: str) -> "WeightTrajectoryAnalyser":
        """Preconfigure for a published trajectory-dimension preset.

        Recognised keys: ``"birdal2021"`` (PH dim, Euclidean trajectory),
        ``"dupuis2023"`` (PH dim, loss-difference trajectory),
        ``"andreeva2024"`` (magnitude dim).
        """
        from tanc.pipeline.paper_presets import PAPER_PRESETS
        if name not in PAPER_PRESETS:
            valid = sorted(PAPER_PRESETS.keys())
            raise ValueError(
                f"Unknown preset '{name}'. Valid:\n  " + "\n  ".join(valid)
            )
        preset = PAPER_PRESETS[name]
        if preset["tool"] != "dimension":
            raise ValueError(
                f"Preset '{name}' uses tool='{preset['tool']}', "
                "not 'dimension' — WeightTrajectoryAnalyser only handles "
                "dimension presets."
            )
        if preset["tool_kwargs"].get("estimator") != "trajectory_dimension":
            raise ValueError(
                f"Preset '{name}' uses an activation_id estimator; use "
                "TDAPipeline.from_paper instead."
            )
        method = preset["tool_kwargs"]["method"]
        # Map the paper-preset method names onto run_trajectory_dimension's
        # internal vocabulary.
        mapped = {
            "ph_euclidean":          "ph_dimension",
            "ph_loss":               "ph_dimension",
            "magnitude":             "magnitude_dimension",
            "magnitude_dimension":   "magnitude_dimension",
            "ph_dimension":          "ph_dimension",
        }.get(method, method)
        method_kwargs = {
            k: v for k, v in preset["tool_kwargs"].items()
            if k not in ("method", "estimator")
        }

        analyser = cls()
        analyser._default_method = mapped
        analyser._method_kwargs = method_kwargs
        analyser.paper_reference = preset.get("paper_reference")
        return analyser

    # ── Introspection ───────────────────────────────────────────────────────

    def configure(self, **kwargs) -> "WeightTrajectoryAnalyser":
        """Update fields in bulk after construction.  Returns self."""
        for k, v in kwargs.items():
            if k in {"bundle", "loss_values", "paper_reference"}:
                setattr(self, k, v)
            elif k == "method":
                self._default_method = v
            elif k == "method_kwargs":
                self._method_kwargs = dict(v or {})
            else:
                raise AttributeError(
                    f"WeightTrajectoryAnalyser has no configurable field '{k}'."
                )
        return self

    def explain(self, print_it: bool = True) -> str:
        """Plain-English summary of what :meth:`fit` will compute."""
        lines = ["WeightTrajectoryAnalyser"]
        if self.paper_reference:
            lines.append(f"  paper          : {self.paper_reference}")
        if self._default_method:
            lines.append(f"  default method : {self._default_method}")
        else:
            lines.append(
                "  default method : (not set — call .ph_dimension(), "
                ".magnitude_dimension(), or pass method= to .fit())"
            )
        for k, v in self._method_kwargs.items():
            lines.append(f"    {k} = {v!r}")
        if self.bundle is not None:
            lines.append(
                f"  bundle         : {self.bundle.n_nodes} nodes "
                f"({self.bundle.matrix_type})"
            )
        else:
            lines.append("  bundle         : (not set — pass to .fit(bundle=...))")
        if self.loss_values is not None:
            lines.append(
                f"  loss_values    : length {len(self.loss_values)}"
            )
        text = "\n".join(lines)
        if print_it:
            print(text)
        return text

    # ── Execution ───────────────────────────────────────────────────────────

    def fit(
        self,
        bundle: GraphBundle | None = None,
        loss_values: np.ndarray | None = None,
        method: str | None = None,
        **method_kwargs,
    ) -> TopoResult:
        """Run the configured (or explicitly named) trajectory-dimension method.

        Parameters
        ----------
        bundle : GraphBundle or None
            Defaults to ``self.bundle`` set at construction.
        loss_values : ndarray or None
            Defaults to ``self.loss_values``.
        method : str or None
            ``"ph_dimension"`` | ``"magnitude_dimension"``.  Defaults to
            whichever method was set by :meth:`from_paper` /
            :meth:`configure`.
        **method_kwargs
            Forwarded to ``run_trajectory_dimension``.  Merged with the
            kwargs stored by :meth:`from_paper`; explicit kwargs win.
        """
        b = bundle if bundle is not None else self.bundle
        if b is None:
            raise ValueError(
                "No bundle available. Pass one to .fit(bundle=...) or set it "
                "on the analyser at construction."
            )
        lv = loss_values if loss_values is not None else self.loss_values
        m = method or self._default_method
        if m is None:
            raise ValueError(
                "No method set. Either construct via .from_paper(...), call "
                ".configure(method='ph_dimension' | 'magnitude_dimension'), or "
                "pass method= explicitly to .fit()."
            )
        merged = {**self._method_kwargs, **method_kwargs}
        result = run_trajectory_dimension(
            b, method=m, loss_values=lv, **merged
        )
        if self.paper_reference:
            result.paper_reference = self.paper_reference
        return result

    def ph_dimension(self, **kwargs) -> TopoResult:
        if self.bundle is None:
            raise ValueError(
                "ph_dimension() requires self.bundle to be set."
            )
        return run_trajectory_dimension(
            self.bundle, method="ph_dimension",
            loss_values=self.loss_values, **kwargs,
        )

    def magnitude_dimension(self, **kwargs) -> TopoResult:
        if self.bundle is None:
            raise ValueError(
                "magnitude_dimension() requires self.bundle to be set."
            )
        return run_trajectory_dimension(
            self.bundle, method="magnitude_dimension", **kwargs,
        )
