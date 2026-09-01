"""_sweep_engine.py — generic parameter-grid execution: expansion, staged caching, resume.

This module knows nothing about Mapper, persistent homology or dimension
estimation.  It answers one question: *given a set of parameter axes and a
function from one parameter combination to a row of numbers, how do I run all
the combinations without recomputing shared work, without losing results to a
crash, and without ever overwriting a previous run?*

The three pieces
----------------
:func:`expand_grid`
    Turns ``{"lens": ["pca2", "l2"], "overlap": 0.3}`` into a list of concrete
    configurations.  A **list** sweeps an axis; a **scalar pins** it.  A
    **tuple is always a single value**, so ``n_intervals=(30, 20)`` is one
    composite setting rather than two alternatives.

:class:`Stage`
    Declares that some expensive intermediate depends only on the *first few*
    axes.  Configurations are emitted in axis order, so a stage whose axes have
    not changed since the previous configuration is reused rather than
    recomputed.  This is what keeps a grid tractable: without it, a sweep over
    covers and clusterers recomputes the point cloud and the lens for every
    single combination.

:class:`SweepStore`
    The on-disk record.  Claims a fresh directory atomically (so two sweeps
    launched at once cannot collide, and an existing run is never overwritten),
    then appends one JSON line per finished configuration.  A crash costs the
    line in flight, not the file.

Ordering is the whole trick
---------------------------
Because :func:`expand_grid` varies the *last* axis fastest, configurations that
share a prefix are contiguous.  A stage therefore only needs to remember its
most recent value — not a dictionary of every value it has ever produced — so
peak memory stays flat no matter how large the grid is.

Notes
-----
Results are plain dictionaries of JSON-serialisable scalars.  Heavy artifacts
(graphs, point clouds) are the caller's responsibility; see
``SweepStore.artifact_path`` for where to put them so they stay keyed to the
configuration that produced them.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

__all__ = [
    "Sweep",
    "sweep",
    "expand_grid",
    "config_hash",
    "canonical_config",
    "display_config",
    "Stage",
    "StagedContext",
    "SweepStore",
    "run_sweep",
    "SweepError",
]

_STORE_FORMAT_VERSION = 1

# Packages whose versions materially change numerical results.  Recorded in the
# manifest so a run can be reproduced, or explained when it cannot be.
_TRACKED_PACKAGES = (
    "numpy", "scipy", "scikit-learn", "networkx",
    "giotto-tda", "gudhi", "ripser", "pandas",
)


class SweepError(RuntimeError):
    """Raised for malformed grids, stage declarations, or store state."""


# ─────────────────────────────────────────────────────────────────────────────
# Grid expansion
# ─────────────────────────────────────────────────────────────────────────────

class Sweep:
    """Explicit marker that an axis should be swept over *values*.

    A bare ``list`` is also treated as a sweep, which covers the common case.
    Use ``Sweep`` when the values are themselves lists and the intent would
    otherwise be ambiguous::

        Sweep([[10, 10], [20, 5]])   # two settings, each a per-dimension pair

    Parameters
    ----------
    values : sequence
        The alternatives for this axis.  Must be non-empty.
    """

    __slots__ = ("values",)

    def __init__(self, values: Sequence[Any]):
        vals = list(values)
        if not vals:
            raise SweepError("Sweep(...) needs at least one value; got an empty sequence.")
        self.values = vals

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Sweep({self.values!r})"


def sweep(*values: Any) -> Sweep:
    """Convenience constructor: ``sweep(1, 2, 3)`` is ``Sweep([1, 2, 3])``."""
    if len(values) == 1 and isinstance(values[0], (list, tuple)) and not isinstance(values[0], str):
        return Sweep(list(values[0]))
    return Sweep(list(values))


def _axis_values(value: Any) -> list[Any]:
    """The alternatives an axis specification stands for.

    ``Sweep`` and ``list`` mean "several"; everything else — including
    ``tuple``, which is how composite single values are written — means "one".
    """
    if isinstance(value, Sweep):
        return list(value.values)
    if isinstance(value, list):
        if not value:
            raise SweepError("An axis was given an empty list; use a scalar to pin it.")
        return list(value)
    return [value]


def expand_grid(axes: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand axis specifications into every concrete configuration.

    The **last** axis varies fastest, so configurations sharing a prefix of
    earlier axes are contiguous.  :func:`run_sweep` relies on this to reuse
    staged work, so declare axes from most expensive to least.

    Parameters
    ----------
    axes : dict
        Maps axis name to a scalar (pinned), a list or :class:`Sweep` (swept),
        or a tuple (a single composite value).

    Returns
    -------
    list of dict
        One dictionary per configuration, each carrying every axis name.

    Examples
    --------
    >>> expand_grid({"lens": ["pca2", "l2"], "overlap": 0.3})
    [{'lens': 'pca2', 'overlap': 0.3}, {'lens': 'l2', 'overlap': 0.3}]

    Order matters — put the expensive axis first so it is recomputed least:

    >>> cfgs = expand_grid({"layer": ["c1", "c2"], "overlap": [0.3, 0.5]})
    >>> [c["layer"] for c in cfgs]
    ['c1', 'c1', 'c2', 'c2']
    """
    if not axes:
        raise SweepError("expand_grid needs at least one axis.")
    names = list(axes)
    value_lists = [_axis_values(axes[n]) for n in names]
    return [dict(zip(names, combo)) for combo in product(*value_lists)]


def grid_size(axes: dict[str, Any]) -> int:
    """Number of configurations ``axes`` expands to, without building them."""
    n = 1
    for name in axes:
        n *= len(_axis_values(axes[name]))
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Configuration identity
# ─────────────────────────────────────────────────────────────────────────────

def canonical_config(cfg: dict[str, Any]) -> dict[str, str]:
    """Render a configuration as stable strings, for hashing and for the record.

    Values that are not JSON scalars are rendered with :func:`repr`, which is
    stable and informative for the objects that actually appear on grid axes
    (scikit-learn estimators repr as ``DBSCAN(eps=0.5)``).  Callables are
    rendered by qualified name, since their ``repr`` embeds a memory address
    that would change between processes and break resume.
    """
    out: dict[str, str] = {}
    for key in sorted(cfg):
        out[key] = _canonical_value(cfg[key])
    return out


def _canonical_value(v: Any) -> str:
    if v is None or isinstance(v, (str, bool, int)):
        return repr(v)
    if isinstance(v, float):
        return repr(float(v))
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_canonical_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_canonical_value(v[k])}" for k in sorted(v)) + "}"
    # A function or class: repr embeds an address, so use the qualified name.
    if callable(v) and hasattr(v, "__qualname__"):
        mod = getattr(v, "__module__", "?")
        return f"<{mod}.{v.__qualname__}>"
    return repr(v)


def display_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Render a configuration for *reading*, rather than for hashing.

    :func:`canonical_config` reprs every value so that a string and the integer
    that prints the same way cannot collide in a hash.  That is right for
    identity and wrong for a results table, where it would turn the cloud name
    ``conv1`` into ``"'conv1'"`` and make it useless as a lookup key.  Here,
    values JSON can represent natively are kept as they are; everything else
    falls back to the canonical string.
    """
    out: dict[str, Any] = {}
    for key in sorted(cfg):
        v = cfg[key]
        out[key] = v if v is None or isinstance(v, (str, bool, int, float)) \
            else _canonical_value(v)
    return out


def config_hash(cfg: dict[str, Any]) -> str:
    """A short, stable, content-addressed identifier for a configuration.

    The hash covers *every* axis, so two grids that differ in any parameter
    cannot collide in one store — the failure mode of keying results on a
    hand-picked subset of the axes.
    """
    blob = json.dumps(canonical_config(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Staged computation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stage:
    """An expensive intermediate that depends on only a prefix of the axes.

    Parameters
    ----------
    name : str
        Identifies the stage in progress output and timings.
    axes : tuple of str
        The axis names this stage's result depends on.  Must be a prefix of the
        grid's axis order: a stage cannot depend on a later axis than a stage
        declared after it.
    fn : callable
        ``fn(previous, **axis_values) -> value``.  *previous* is the preceding
        stage's value, or ``None`` for the first stage.  Only the axes named in
        ``axes`` are passed.

    Examples
    --------
    >>> Stage("cloud", ("layer", "view"), lambda prev, layer, view: build(layer, view))
    Stage(name='cloud', axes=('layer', 'view'), ...)
    """

    name: str
    axes: tuple[str, ...]
    fn: Callable[..., Any]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"Stage(name={self.name!r}, axes={self.axes!r}, ...)"


class StagedContext:
    """Runs stages for each configuration, reusing values whose axes are unchanged.

    Holds exactly one cached value per stage, which is sufficient because
    :func:`expand_grid` emits configurations in axis order.  Recomputation
    counts are tracked so a sweep can report how much work sharing actually
    saved.
    """

    def __init__(self, stages: Sequence[Stage], axis_order: Sequence[str]):
        _validate_stages(stages, axis_order)
        self.stages = list(stages)
        self._keys: list[Any] = [object()] * len(self.stages)
        self._values: list[Any] = [None] * len(self.stages)
        self.recomputes: dict[str, int] = {s.name: 0 for s in self.stages}
        self.seconds: dict[str, float] = {s.name: 0.0 for s in self.stages}

    def for_config(self, cfg: dict[str, Any]) -> Any:
        """Return the final stage's value for *cfg*, recomputing only what changed."""
        prev = None
        stale = False
        for i, st in enumerate(self.stages):
            key = tuple(cfg[a] for a in st.axes)
            # Once a stage recomputes, every stage after it must follow: its
            # input changed even if its own axis values did not.
            if stale or self._keys[i] != key:
                t0 = time.perf_counter()
                self._values[i] = st.fn(prev, **{a: cfg[a] for a in st.axes})
                self.seconds[st.name] += time.perf_counter() - t0
                self._keys[i] = key
                self.recomputes[st.name] += 1
                stale = True
            prev = self._values[i]
        return prev

    def release(self) -> None:
        """Drop cached values so large intermediates can be garbage-collected."""
        self._values = [None] * len(self.stages)
        self._keys = [object()] * len(self.stages)


def _validate_stages(stages: Sequence[Stage], axis_order: Sequence[str]) -> None:
    """Check every stage depends on a prefix of the axis order, and nests."""
    order = list(axis_order)
    seen: list[str] = []
    for st in stages:
        unknown = [a for a in st.axes if a not in order]
        if unknown:
            raise SweepError(
                f"Stage {st.name!r} depends on unknown axes {unknown}. "
                f"Known axes: {order}."
            )
        positions = [order.index(a) for a in st.axes]
        expected = list(range(len(st.axes)))
        if sorted(positions) != expected:
            raise SweepError(
                f"Stage {st.name!r} depends on {list(st.axes)}, which is not a prefix "
                f"of the axis order {order}. Reorder the grid so this stage's axes come "
                f"first, or widen the stage to include the axes in between — otherwise "
                f"its cached value would be reused when one of its real inputs changed."
            )
        if len(st.axes) < len(seen):
            raise SweepError(
                f"Stage {st.name!r} depends on fewer axes ({list(st.axes)}) than the "
                f"stage before it ({seen}). Declare stages from cheapest-to-vary "
                f"(fewest axes) to most."
            )
        seen = list(st.axes)


# ─────────────────────────────────────────────────────────────────────────────
# On-disk store
# ─────────────────────────────────────────────────────────────────────────────

class SweepStore:
    """An append-only run directory that is never silently overwritten.

    The directory is claimed with :func:`os.mkdir`, which fails rather than
    overwrites, so a fresh sweep launched against an existing name gets
    ``name-002``, ``name-003``, and concurrent launches cannot collide.

    Layout::

        <dir>/manifest.json     grid definition, versions, timings
        <dir>/configs.jsonl     hash -> full resolved configuration
        <dir>/results.jsonl     hash -> measures (append-only, one line each)
        <dir>/artifacts/        caller-owned heavy files, named by hash

    Parameters
    ----------
    path : str or Path
        Desired directory.  A numeric suffix is added if it already exists,
        unless *resume* is True.
    resume : bool
        Open an existing directory and continue it instead of claiming a new
        one.  Raises if the directory does not exist.
    """

    def __init__(self, path: Any, *, resume: bool = False):
        base = Path(path)
        if resume:
            if not base.is_dir():
                raise SweepError(
                    f"resume was requested but {base} does not exist. "
                    f"Omit resume to start a new run."
                )
            self.path = base
        else:
            self.path = _claim_directory(base)
        self.artifacts = self.path / "artifacts"
        self.artifacts.mkdir(exist_ok=True)
        self._results = self.path / "results.jsonl"
        self._configs = self.path / "configs.jsonl"
        self._done: set[str] | None = None

    # ── identity ─────────────────────────────────────────────────────────────

    def artifact_path(self, cfg_hash: str, suffix: str = ".npz") -> Path:
        """Where a caller should write the heavy artifact for one configuration."""
        return self.artifacts / f"{cfg_hash}{suffix}"

    def completed(self) -> set[str]:
        """Hashes of configurations already recorded, read once and cached."""
        if self._done is None:
            done: set[str] = set()
            if self._results.exists():
                with self._results.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            done.add(json.loads(line)["hash"])
                        except (ValueError, KeyError):
                            # A torn final line from an interrupted run: the
                            # configuration simply reruns.
                            continue
            self._done = done
        return self._done

    # ── writing ──────────────────────────────────────────────────────────────

    def write_manifest(self, **fields: Any) -> None:
        """Write (or overwrite) the run's manifest, stamping environment versions."""
        payload = {
            "format_version": _STORE_FORMAT_VERSION,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
            **fields,
        }
        (self.path / "manifest.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    def record_config(self, cfg_hash: str, cfg: dict[str, Any]) -> None:
        """Append the full resolved configuration for *cfg_hash*.

        Stored in display form, so a results table carries usable values —
        the hash it is keyed by already provides exact identity.
        """
        self._append(self._configs, {"hash": cfg_hash, "config": display_config(cfg)})

    def record_result(self, cfg_hash: str, row: dict[str, Any]) -> None:
        """Append one result row and mark the configuration complete."""
        self._append(self._results, {"hash": cfg_hash, **row})
        if self._done is not None:
            self._done.add(cfg_hash)

    @staticmethod
    def _append(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # ── reading ──────────────────────────────────────────────────────────────

    def rows(self) -> list[dict[str, Any]]:
        """Every recorded result, joined with its configuration."""
        configs: dict[str, dict[str, Any]] = {}
        if self._configs.exists():
            with self._configs.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        configs[rec["hash"]] = rec.get("config", {})
        out: list[dict[str, Any]] = []
        if self._results.exists():
            with self._results.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        out.append({**configs.get(rec.get("hash", ""), {}), **rec})
        return out

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"SweepStore({str(self.path)!r}, {len(self.completed())} recorded)"


def _claim_directory(base: Path) -> Path:
    """Create *base*, or ``base-002``, ``base-003``… — never reusing an existing one."""
    base.parent.mkdir(parents=True, exist_ok=True)
    candidate = base
    n = 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            n += 1
            candidate = base.with_name(f"{base.name}-{n:03d}")
            if n > 999:
                raise SweepError(
                    f"Could not claim a run directory near {base}: "
                    f"{base.name}-002 through -999 all exist."
                )


def _package_versions() -> dict[str, str]:
    """Installed versions of the packages that change numerical results."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    axes: dict[str, Any],
    evaluate: Callable[..., dict[str, Any]],
    store: SweepStore,
    *,
    stages: Sequence[Stage] = (),
    validate: Callable[[dict[str, Any]], str | None] | None = None,
    on_error: str = "record",
    progress: bool = True,
    progress_every: int = 25,
    extra_manifest: dict[str, Any] | None = None,
) -> SweepStore:
    """Run every configuration in *axes*, reusing staged work and recording as it goes.

    Invalid configurations are rejected **before** any of them runs, so a grid
    with a contradictory combination fails in milliseconds rather than partway
    through a long sweep.

    Parameters
    ----------
    axes : dict
        Axis specifications, as for :func:`expand_grid`.  Declare expensive
        axes first — staged reuse follows this order.
    evaluate : callable
        ``evaluate(context, **config) -> dict``.  *context* is the final
        stage's value (``None`` when no stages are declared).  Must return a
        dictionary of JSON-serialisable measures.
    store : SweepStore
        Where results are appended.  Already-recorded configurations are
        skipped, which is what makes a run resumable.
    stages : sequence of Stage
        Shared intermediates, declared from fewest axes to most.
    validate : callable, optional
        ``validate(config) -> None | str``.  Returning a string rejects that
        configuration with the string as the reason.  Run for every
        configuration up front.
    on_error : {"record", "raise"}
        Whether an exception inside *evaluate* is recorded and the sweep
        continues, or re-raised immediately.
    progress : bool
        Print a progress line periodically.
    progress_every : int
        Configurations between progress lines.
    extra_manifest : dict, optional
        Additional fields to store in the manifest.

    Returns
    -------
    SweepStore
        The same store, for chaining.

    Raises
    ------
    SweepError
        If the grid is malformed, the stages are not nested, or *on_error* is
        not one of the accepted values.
    """
    if on_error not in {"record", "raise"}:
        raise SweepError(f"on_error must be 'record' or 'raise'; got {on_error!r}.")

    configs = expand_grid(axes)
    axis_order = list(axes)
    ctx = StagedContext(stages, axis_order) if stages else None

    # Validate everything first: a contradictory axis combination should cost
    # milliseconds, not the first hour of a sweep.
    rejected: list[tuple[dict[str, Any], str]] = []
    runnable: list[dict[str, Any]] = []
    for cfg in configs:
        reason = validate(cfg) if validate is not None else None
        (rejected.append((cfg, reason)) if reason else runnable.append(cfg))

    store.write_manifest(
        axes=canonical_config({k: _axis_values(v) for k, v in axes.items()}),
        n_configs=len(configs),
        n_runnable=len(runnable),
        n_rejected=len(rejected),
        stages=[{"name": s.name, "axes": list(s.axes)} for s in stages],
        **(extra_manifest or {}),
    )
    for cfg, reason in rejected:
        h = config_hash(cfg)
        store.record_config(h, cfg)
        store.record_result(h, {"status": "rejected", "reason": reason})

    done = store.completed()
    todo = [c for c in runnable if config_hash(c) not in done]

    if progress:
        skipped = len(runnable) - len(todo)
        print(f"sweep → {store.path}")
        print(
            f"  {len(configs)} configs: {len(todo)} to run"
            + (f", {skipped} already recorded" if skipped else "")
            + (f", {len(rejected)} rejected" if rejected else "")
        )

    t0 = time.perf_counter()
    n_ok = n_err = 0
    for i, cfg in enumerate(todo, 1):
        h = config_hash(cfg)
        store.record_config(h, cfg)
        try:
            context = ctx.for_config(cfg) if ctx is not None else None
            row = evaluate(context, **cfg)
            if not isinstance(row, dict):
                raise SweepError(
                    f"evaluate must return a dict of measures; got {type(row).__name__}."
                )
            store.record_result(h, {"status": "ok", **row})
            n_ok += 1
        except SweepError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            if on_error == "raise":
                raise
            store.record_result(
                h,
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                    "traceback": traceback.format_exc(limit=3)[-1000:],
                },
            )
            n_err += 1
        if progress and (i % progress_every == 0 or i == len(todo)):
            rate = i / max(time.perf_counter() - t0, 1e-9)
            left = (len(todo) - i) / rate if rate else 0.0
            print(
                f"  [{i}/{len(todo)}] {rate:.2f} cfg/s, ~{left / 60:.1f} min left"
                + (f", {n_err} errors" if n_err else "")
            )

    if progress:
        elapsed = time.perf_counter() - t0
        print(f"  done in {elapsed:.0f}s — {n_ok} ok, {n_err} errors → {store.path}")
        if ctx is not None:
            shared = ", ".join(
                f"{name} {count}x ({ctx.seconds[name]:.1f}s)"
                for name, count in ctx.recomputes.items()
            )
            print(f"  staged work: {shared}  (vs {len(todo)}x each without staging)")
    if ctx is not None:
        ctx.release()
    return store
