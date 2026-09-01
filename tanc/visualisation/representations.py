"""representations.py — PH visualisation for TopoResults.

Provides persistence diagram, barcode, Betti-curve, and multi-panel
comparison plots for ``PersistenceResult`` objects produced by
``topo_tools.ph_tool``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from tanc.topo_tools._result import PersistenceResult


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

_DIM_COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def _dim_colour(d: int) -> str:
    return _DIM_COLOURS[d % len(_DIM_COLOURS)]


# ─────────────────────────────────────────────────────────────────────────────
# Persistence diagram
# ─────────────────────────────────────────────────────────────────────────────

def _infinity_line(dgm: np.ndarray) -> tuple[float, float, float]:
    """Finite plotting range for a diagram, and where to draw essential bars.

    Essential classes have infinite death and cannot be placed on a finite axis.
    The convention is to draw them on a dedicated line above every finite death,
    marked as infinity, so they stay visible and countable without distorting
    the scale.  Returns ``(lo, hi_finite, y_infinity)``.
    """
    d = np.asarray(dgm, dtype=float)
    fin = d[np.isfinite(d).all(axis=1)]
    if fin.size:
        lo = float(min(fin[:, 0].min(), fin[:, 1].min()))
        hi = float(max(fin[:, 0].max(), fin[:, 1].max()))
    else:                                    # every bar is essential
        births = d[:, 0][np.isfinite(d[:, 0])]
        lo = float(births.min()) if births.size else 0.0
        hi = lo + 1.0
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi, hi + (hi - lo) * 0.12


def plot_persistence_diagram(
    ph_result: PersistenceResult,
    dims: list[int] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    show_diagonal: bool = True,
    alpha: float = 0.7,
    point_size: int = 20,
    density: bool = False,
    bins: int = 60,
    cmap: str = "viridis",
) -> Figure:
    """Plot a persistence diagram (birth vs death scatter).

    Parameters
    ----------
    ph_result : PersistenceResult
    dims : list[int] or None
        Dimensions to plot.  Defaults to all available.
    ax : Axes or None
    figsize : (w, h) or None.  Default (6, 6).
    title : str or None
    show_diagonal : bool
        Draw the ``birth == death`` diagonal.
    alpha : float
        Point transparency (scatter mode).
    point_size : int
        Marker size (scatter mode).
    density : bool
        When ``True``, render the diagram as a scatter of *unique*
        (birth, death) pairs coloured by overlap multiplicity.  Matches
        the Dionysus 2 style used in Watanabe & Yamana (2021, Figs. 4-6):
        discrete dots, colour encodes how many generators share the
        same coordinate.  Only the highest dimension in ``dims`` is
        shown.  Defaults to scatter mode without colour encoding.
    bins : int
        Unused in density mode (kept for backward compatibility).
    cmap : str
        Colormap used to encode multiplicity in density mode.

    Returns
    -------
    matplotlib Figure
    """
    if figsize is None:
        figsize = (6, 6)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if dims is None:
        dims = sorted(ph_result.diagrams.keys())

    if density:
        # Render each unique (birth, death) as a scatter dot whose colour
        # encodes overlap multiplicity.  Closer to the paper's Figs. 4-6
        # than a 2-D histogram, especially when births/deaths are integer-
        # valued (as after snapping to a discrete filtration schedule).
        target_dim = dims[-1] if dims else 1
        dgm = ph_result.diagrams.get(target_dim, np.empty((0, 2)))
        if dgm.shape[0] == 0:
            ax.text(0.5, 0.5, f"empty H{target_dim}", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            unique_pts, inverse = np.unique(dgm, axis=0, return_inverse=True)
            counts = np.bincount(inverse)
            log_counts = np.log10(counts.astype(float) + 1.0)
            sc = ax.scatter(
                unique_pts[:, 0], unique_pts[:, 1],
                c=log_counts, s=point_size,
                cmap=cmap, alpha=alpha, edgecolors="none",
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("log10(1 + count)")
            lo, hi, y_inf = _infinity_line(dgm)
            n_ess = int(np.isinf(dgm[:, 1]).sum())
            if n_ess:
                # Essential bars sit on the infinity line rather than off-axis.
                ax.scatter(dgm[np.isinf(dgm[:, 1]), 0],
                           np.full(n_ess, y_inf),
                           marker="^", s=point_size, color="crimson",
                           zorder=5, label=f"essential ({n_ess})")
                ax.axhline(y_inf, color="crimson", linestyle=":", linewidth=0.8, alpha=0.6)
                ax.legend(loc="lower right", fontsize=8)
                hi = y_inf
            pad = (hi - lo) * 0.05 or 0.5
            if show_diagonal:
                # axline draws an infinite line that follows whatever
                # xlim/ylim the caller sets later, so the diagonal always
                # spans the visible box.
                ax.axline((0.0, 0.0), slope=1.0,
                          color="k", linewidth=0.6, alpha=0.4)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(title or f"Persistence Diagram (density, H{target_dim})")
        fig.tight_layout()
        return fig

    all_vals: list[float] = []
    ess: list[tuple[int, np.ndarray]] = []      # essential births, per dimension
    for d in dims:
        dgm = ph_result.diagrams.get(d, np.empty((0, 2)))
        if dgm.shape[0] == 0:
            continue
        vals = dgm.ravel()
        all_vals.extend(vals[np.isfinite(vals)].tolist())
        fin = dgm[np.isfinite(dgm[:, 1])]
        ax.scatter(
            fin[:, 0], fin[:, 1],
            s=point_size, alpha=alpha,
            color=_dim_colour(d), label=f"H{d}",
        )
        ess.append((d, dgm[np.isinf(dgm[:, 1]), 0]))

    if all_vals:
        min_v = min(all_vals)
        max_v = max(all_vals)
        n_ess = sum(len(b) for _, b in ess)
        if n_ess:
            y_inf = max_v + (max_v - min_v) * 0.12 or max_v + 0.1
            for d, births in ess:
                if len(births):
                    ax.scatter(births, np.full(len(births), y_inf), marker="^",
                               s=point_size, color=_dim_colour(d), zorder=5)
            ax.axhline(y_inf, color="crimson", linestyle=":", linewidth=0.8,
                       alpha=0.6, label=f"essential ({n_ess})")
            max_v = y_inf
        pad = (max_v - min_v) * 0.05 or 0.1
        lim = (min_v - pad, max_v + pad)
        if show_diagonal:
            ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.legend()
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Persistence Diagram")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Barcode
# ─────────────────────────────────────────────────────────────────────────────

def plot_barcode(
    ph_result: PersistenceResult,
    dims: list[int] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    linewidth: float = 1.5,
    max_bars: int | None = 40,
    sort_by: str = "lifetime",
) -> Figure:
    """Plot a persistence barcode (horizontal bars).

    Dense diagrams (e.g. H0 of a neuron graph has one bar per merge — often
    hundreds) make a raw barcode unreadably tall.  By default only the
    **longest ``max_bars`` bars** are shown and the figure height is capped.

    Parameters
    ----------
    ph_result : PersistenceResult
    dims : list[int] or None
    ax : Axes or None
    figsize : (w, h) or None.  Default ``(8, min(12, max(3, n_bars * 0.22)))``.
    title : str or None
    linewidth : float
    max_bars : int or None
        Keep only the ``max_bars`` longest-lifetime bars (``None`` = keep all).
    sort_by : str
        ``"lifetime"`` (longest at top — best for ranking) or ``"birth"``.

    Returns
    -------
    matplotlib Figure
    """
    if dims is None:
        dims = sorted(ph_result.diagrams.keys())

    # Collect all bars; clip the essential (infinite-death) bar to the data range.
    raw: list[tuple[float, float, int]] = []
    finite_max = 0.0
    for d in dims:
        dgm = ph_result.diagrams.get(d, np.empty((0, 2)))
        for row in dgm:
            b, de = float(row[0]), float(row[1])
            raw.append((b, de, d))
            if np.isfinite(de):
                finite_max = max(finite_max, de)
    finite_max = finite_max or 1.0
    cap = finite_max * 1.05
    bars = [(b, (cap if not np.isfinite(de) else de), de, d) for b, de, d in raw]

    n_total = len(bars)
    truncated = False
    if max_bars is not None and n_total > max_bars:
        # Keep the longest by (clipped) lifetime.
        bars.sort(key=lambda x: -(x[1] - x[0]))
        bars = bars[:max_bars]
        truncated = True

    if sort_by == "lifetime":
        bars.sort(key=lambda x: (x[1] - x[0]))          # longest at the top
    else:
        bars.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    n_bars = max(len(bars), 1)
    if figsize is None:
        figsize = (8, min(12.0, max(3.0, n_bars * 0.22)))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    legend_labels: set[str] = set()
    for i, (birth, death, death_raw, d) in enumerate(bars):
        label = f"H{d}" if f"H{d}" not in legend_labels else None
        if label:
            legend_labels.add(label)
        ax.plot([birth, death], [i, i], color=_dim_colour(d),
                linewidth=linewidth, label=label,
                solid_capstyle="butt")
        if not np.isfinite(death_raw):                  # mark the essential bar
            ax.plot(death, i, ">", color=_dim_colour(d), markersize=5)

    ax.set_yticks([])
    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Bars (longest first)" if sort_by == "lifetime" else "Bars")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    if title:
        ax.set_title(title)
    elif truncated:
        ax.set_title(f"Persistence Barcode — longest {n_bars} of {n_total} bars")
    else:
        ax.set_title("Persistence Barcode")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Betti curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_betti_curve(
    ph_result: PersistenceResult,
    dims: list[int] | None = None,
    resolution: int = 100,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot Betti curves: Betti number as a function of filtration value.

    Parameters
    ----------
    ph_result : PersistenceResult
    dims : list[int] or None
    resolution : int
        Number of filtration values to evaluate.
    ax : Axes or None
    figsize : (w, h) or None.  Default (8, 4).
    title : str or None

    Returns
    -------
    matplotlib Figure
    """
    from tanc.topo_tools.ph_tool import betti_curve as _betti_curve

    if figsize is None:
        figsize = (8, 4)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if dims is None:
        dims = sorted(ph_result.diagrams.keys())

    # Global filtration range
    all_vals: list[float] = []
    for d in dims:
        dgm = ph_result.diagrams.get(d, np.empty((0, 2)))
        if dgm.shape[0] > 0:
            all_vals.extend(dgm.ravel().tolist())

    if not all_vals:
        ax.set_xlabel("Filtration value")
        ax.set_ylabel("Betti number")
        if title:
            ax.set_title(title)
        return fig

    min_v = min(all_vals)
    max_v = max(all_vals)
    epsilons = np.linspace(min_v, max_v, resolution)

    for d in dims:
        dgm = ph_result.diagrams.get(d, np.empty((0, 2)))
        curve = _betti_curve(dgm, resolution=resolution)
        ax.step(epsilons, curve, color=_dim_colour(d), label=f"H{d}", where="post")

    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Betti number")
    ax.legend()
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Betti Curves")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Persistence landscape (Bubenik 2015)
# ─────────────────────────────────────────────────────────────────────────────

def _landscape_numpy(
    dgm: np.ndarray, k_max: int, resolution: int,
    t_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback landscape implementation when gudhi is unavailable."""
    if dgm.shape[0] == 0:
        if t_range is None:
            t_range = (0.0, 1.0)
        return np.zeros((k_max, resolution)), np.linspace(*t_range, resolution)

    births = dgm[:, 0]
    deaths = dgm[:, 1]
    if t_range is None:
        t_range = (float(births.min()), float(deaths.max()))
    t = np.linspace(*t_range, resolution)
    tents = np.maximum(
        0.0,
        np.minimum(t[None, :] - births[:, None], deaths[:, None] - t[None, :]),
    )
    sorted_tents = np.sort(tents, axis=0)[::-1, :]
    if sorted_tents.shape[0] < k_max:
        pad = np.zeros((k_max - sorted_tents.shape[0], resolution))
        sorted_tents = np.vstack([sorted_tents, pad])
    return sorted_tents[:k_max], t


def persistence_landscape(
    dgm: np.ndarray,
    k_max: int = 5,
    resolution: int = 200,
    t_range: tuple[float, float] | None = None,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the first ``k_max`` persistence landscape functions of one diagram.

    The k-th landscape lambda_k(t) is the k-th largest value of the tent
    function ``max(0, min(t-birth, death-t))`` over all bars at time ``t``.

    Backends
    --------
    ``backend="auto"`` (default)
        Use ``gudhi.representations.Landscape`` when gudhi is installed,
        else the numpy fallback.
    ``backend="gudhi"``
        Force gudhi; raises ImportError if unavailable.
    ``backend="numpy"``
        Force the numpy fallback (no optional deps).

    Returns
    -------
    landscapes : (k_max, resolution) ndarray
    t          : (resolution,) ndarray
    """
    if dgm.shape[0] == 0:
        if t_range is None:
            t_range = (0.0, 1.0)
        return np.zeros((k_max, resolution)), np.linspace(*t_range, resolution)

    if t_range is None:
        t_range = (float(dgm[:, 0].min()), float(dgm[:, 1].max()))

    if backend == "numpy":
        return _landscape_numpy(dgm, k_max, resolution, t_range)

    if backend in {"auto", "gudhi"}:
        try:
            from gudhi.representations.vector_methods import Landscape as _GLandscape
            L = _GLandscape(
                num_landscapes=k_max, resolution=resolution,
                sample_range=list(t_range),
            )
            # Gudhi returns flat (1, k_max * resolution); reshape to (k_max, res).
            flat = L.fit_transform([dgm.astype(float)])
            landscapes = np.asarray(flat[0]).reshape(k_max, resolution)
            t = np.linspace(*t_range, resolution)
            return landscapes, t
        except ImportError:
            if backend == "gudhi":
                raise ImportError(
                    "backend='gudhi' requested but gudhi is not installed. "
                    "Install with: pip install gudhi"
                )
            return _landscape_numpy(dgm, k_max, resolution, t_range)

    raise ValueError(f"Unknown backend '{backend}'. Valid: 'auto', 'gudhi', 'numpy'.")


def plot_persistence_landscape(
    ph_result: PersistenceResult,
    dim: int = 1,
    k_max: int = 5,
    resolution: int = 200,
    backend: str = "auto",
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Plot the first ``k_max`` persistence landscape functions.

    Bubenik (2015) — a stable, function-valued summary of a persistence
    diagram.  Lower-indexed landscapes capture the longest-lived features.

    Delegates the numeric computation to ``gudhi.representations.Landscape``
    when available; falls back to a self-contained numpy implementation
    otherwise.  Override with ``backend="gudhi"`` or ``backend="numpy"``.
    """
    dgm = ph_result.diagrams.get(dim, np.empty((0, 2)))
    landscapes, t = persistence_landscape(
        dgm, k_max=k_max, resolution=resolution, backend=backend,
    )

    if figsize is None:
        figsize = (7, 4)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.85, k_max))
    for k in range(k_max):
        if not np.any(landscapes[k] > 0):
            continue
        ax.plot(t, landscapes[k], color=colors[k], label=f"lambda_{k+1}")
    ax.set_xlabel("Filtration value")
    ax.set_ylabel("Landscape height")
    ax.legend(fontsize=8)
    ax.set_title(title or f"Persistence landscape (H{dim})")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Persistence image (Adams et al. 2017)
# ─────────────────────────────────────────────────────────────────────────────

def _image_numpy(
    dgm: np.ndarray, resolution: int, sigma: float | None,
    b_range: tuple[float, float] | None,
    p_range: tuple[float, float] | None,
    weight: str,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Fallback persistence-image implementation."""
    if dgm.shape[0] == 0:
        b_range = b_range or (0.0, 1.0)
        p_range = p_range or (0.0, 1.0)
        return np.zeros((resolution, resolution)), b_range, p_range
    bs = dgm[:, 0]
    ps = dgm[:, 1] - dgm[:, 0]
    if b_range is None:
        b_range = (float(bs.min()), float(bs.max()) or float(bs.min()) + 1.0)
    if p_range is None:
        p_range = (0.0, float(ps.max()) or 1.0)
    if sigma is None:
        sigma = float((p_range[1] - p_range[0]) / max(resolution, 1)) or 1e-3
    b_grid = np.linspace(*b_range, resolution)
    p_grid = np.linspace(*p_range, resolution)
    B, P = np.meshgrid(b_grid, p_grid, indexing="xy")
    weights = ps.copy() if weight == "linear" else np.ones_like(ps)
    img = np.zeros_like(B)
    for b, p, w in zip(bs, ps, weights):
        img += w * np.exp(-((B - b) ** 2 + (P - p) ** 2) / (2 * sigma ** 2))
    return img, b_range, p_range


def persistence_image(
    dgm: np.ndarray,
    resolution: int = 25,
    sigma: float | None = None,
    b_range: tuple[float, float] | None = None,
    p_range: tuple[float, float] | None = None,
    weight: str = "linear",
    backend: str = "auto",
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Compute the persistence image of one diagram.

    Each bar is mapped from ``(birth, death)`` to ``(birth, persistence)``
    and rendered as a Gaussian on a regular grid, weighted by ``weight``
    (``"linear"`` proportional to persistence, or ``"constant"``).

    Backends
    --------
    ``backend="auto"`` (default)
        Use ``persim.PersistenceImager`` if persim is installed; else
        ``gudhi.representations.PersistenceImage`` if gudhi is installed;
        else the numpy fallback.
    ``backend="persim"`` / ``backend="gudhi"`` / ``backend="numpy"``
        Force a specific backend (raises ImportError if unavailable).

    Returns
    -------
    image            : (resolution, resolution) ndarray, indexed
                       (persistence_idx, birth_idx).
    b_range, p_range : the (birth, persistence) ranges actually used.
    """
    if backend == "numpy":
        return _image_numpy(dgm, resolution, sigma, b_range, p_range, weight)

    if backend in {"auto", "persim"}:
        try:
            from persim import PersistenceImager
            # persim PersistenceImager works in (birth, persistence) space already.
            pimgr = PersistenceImager(
                pixel_size=None, kernel_params={"sigma": [[sigma or 0.1, 0.0],
                                                          [0.0, sigma or 0.1]]},
                weight=(lambda b, p: p) if weight == "linear" else None,
            )
            pimgr.fit([dgm.astype(float)], skew=True)
            # Override the auto-computed ranges if the user passed any.
            if b_range is not None:
                pimgr.birth_range = b_range
            if p_range is not None:
                pimgr.pers_range = p_range
            pimgr.resolution = (resolution, resolution)
            img = pimgr.transform([dgm.astype(float)], skew=True)[0]
            # persim returns (resolution, resolution) indexed (birth, persistence);
            # transpose so axis 0 is persistence (matches numpy fallback).
            img = np.asarray(img).T
            return img, tuple(pimgr.birth_range), tuple(pimgr.pers_range)
        except ImportError:
            if backend == "persim":
                raise ImportError(
                    "backend='persim' requested but persim is not installed. "
                    "Install with: pip install persim"
                )
            # fall through to try gudhi

    if backend in {"auto", "gudhi"}:
        try:
            from gudhi.representations.vector_methods import (
                PersistenceImage as _GPersistenceImage,
            )
            if dgm.shape[0] == 0:
                br = b_range or (0.0, 1.0)
                pr = p_range or (0.0, 1.0)
                return np.zeros((resolution, resolution)), br, pr
            bs = dgm[:, 0]
            ps = dgm[:, 1] - dgm[:, 0]
            if b_range is None:
                b_range = (float(bs.min()), float(bs.max()) or float(bs.min()) + 1.0)
            if p_range is None:
                p_range = (0.0, float(ps.max()) or 1.0)
            if sigma is None:
                sigma = float((p_range[1] - p_range[0]) / max(resolution, 1)) or 1e-3
            weight_fn = ((lambda x: x[1]) if weight == "linear"
                         else (lambda x: 1.0))
            pi = _GPersistenceImage(
                bandwidth=sigma, weight=weight_fn,
                resolution=[resolution, resolution],
                im_range=[b_range[0], b_range[1], p_range[0], p_range[1]],
            )
            flat = pi.fit_transform([dgm.astype(float)])
            img = np.asarray(flat[0]).reshape(resolution, resolution)
            return img, b_range, p_range
        except ImportError:
            if backend == "gudhi":
                raise ImportError(
                    "backend='gudhi' requested but gudhi is not installed. "
                    "Install with: pip install gudhi"
                )
            return _image_numpy(dgm, resolution, sigma, b_range, p_range, weight)

    raise ValueError(
        f"Unknown backend '{backend}'. Valid: 'auto', 'persim', 'gudhi', 'numpy'."
    )


def plot_persistence_image(
    ph_result: PersistenceResult,
    dim: int = 1,
    resolution: int = 25,
    sigma: float | None = None,
    weight: str = "linear",
    backend: str = "auto",
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    cmap: str = "inferno",
) -> Figure:
    """Plot the persistence image (birth vs persistence, Gaussian-smoothed).

    Adams et al. (2017) — discretised, vectorisable PH summary commonly fed
    to downstream classifiers.

    Delegates the numeric computation to ``persim.PersistenceImager`` when
    available, falling back to ``gudhi.representations.PersistenceImage``
    and then to a self-contained numpy implementation.  Override with
    ``backend="persim" | "gudhi" | "numpy"``.
    """
    dgm = ph_result.diagrams.get(dim, np.empty((0, 2)))
    img, b_range, p_range = persistence_image(
        dgm, resolution=resolution, sigma=sigma, weight=weight, backend=backend,
    )

    if figsize is None:
        figsize = (6, 5)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    extent = (b_range[0], b_range[1], p_range[0], p_range[1])
    im = ax.imshow(img, origin="lower", extent=extent, aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="intensity")
    ax.set_xlabel("Birth")
    ax.set_ylabel("Persistence (death - birth)")
    ax.set_title(title or f"Persistence image (H{dim})")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer Betti bar chart (Naitzat et al. 2020 style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_betti_layer_bars(
    results,
    dims: list[int] | None = None,
    layer_labels: list[str] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Grouped bar chart of Betti numbers (#bars) per dimension across layers.

    Useful for visualising the layer-by-layer "topological simplification"
    pattern reported in Naitzat et al. (2020): Betti numbers should
    decrease with depth as the network folds the input into class-pure
    components.

    Parameters
    ----------
    results : list[TopoResult] or dict[str, TopoResult]
        One ``TopoResult`` per layer.  When a dict is given, the keys are
        used as ``layer_labels``.
    dims : list[int] or None
        Homology dimensions to plot.  ``None`` = all dimensions present
        in the first result.
    layer_labels : list[str] or None
    ax, figsize, title : standard plot kwargs.
    """
    if isinstance(results, dict):
        layer_labels = list(results.keys())
        results = list(results.values())
    elif layer_labels is None:
        layer_labels = [f"Layer {i}" for i in range(len(results))]

    if not results:
        raise ValueError("results is empty.")
    if dims is None:
        # Use whichever dims are present in the first result.
        first_diagrams = getattr(results[0], "diagrams", None) or {}
        dims = sorted(first_diagrams.keys()) if first_diagrams else [0, 1]

    # Build (n_dims, n_layers) Betti count matrix.
    counts = np.zeros((len(dims), len(results)), dtype=float)
    for j, r in enumerate(results):
        dgms = getattr(r, "diagrams", None) or {}
        for i, d in enumerate(dims):
            counts[i, j] = float(dgms.get(d, np.empty((0, 2))).shape[0])

    if figsize is None:
        figsize = (max(5, 1.2 * len(results)), 4)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x = np.arange(len(results))
    width = 0.8 / max(len(dims), 1)
    for i, d in enumerate(dims):
        offset = (i - (len(dims) - 1) / 2) * width
        ax.bar(x + offset, counts[i], width,
               color=_dim_colour(d), label=f"H{d}")

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Betti number (# bars)")
    ax.legend()
    ax.set_title(title or "Betti numbers per layer")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Multi-panel comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_diagram_comparison(
    results: dict[str, PersistenceResult],
    kind: str = "diagram",
    dims: list[int] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot persistence diagrams / barcodes / Betti curves side-by-side.

    Parameters
    ----------
    results : dict[str, PersistenceResult]
        Keys become panel titles.
    kind : str
        ``"diagram"`` | ``"barcode"`` | ``"betti_curve"``
    dims : list[int] or None
    figsize : (w, h) or None.  Auto-scaled by number of panels.
    title : str or None
        Overall figure title.

    Returns
    -------
    matplotlib Figure
    """
    n = len(results)
    if n == 0:
        raise ValueError("results dict is empty.")

    if figsize is None:
        figsize = (6 * n, 5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    plot_fn = {
        "diagram": plot_persistence_diagram,
        "barcode": plot_barcode,
        "betti_curve": plot_betti_curve,
    }.get(kind)

    if plot_fn is None:
        raise ValueError(
            f"Unknown kind='{kind}'. Valid: 'diagram', 'barcode', 'betti_curve'."
        )

    for ax, (label, ph_result) in zip(axes, results.items()):
        plot_fn(ph_result, dims=dims, ax=ax, title=label)

    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig
