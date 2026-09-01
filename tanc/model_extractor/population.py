"""population.py — training a seed population and reading it as a point cloud.

Many topological analyses of networks are population statements: *these*
structures appear across independently initialised networks, so they are
properties of the task and architecture rather than of one lucky run.  Making
that claim needs a population, and making it *reproducible* needs the seeds
recorded — including when the caller never chose any.

What varies between members
---------------------------
Three things could differ between two members of a population, and conflating
them makes the population uninterpretable::

    vary_init         = True    initial weights            (the point)
    vary_batch_order  = False   order batches are seen in
    vary_split        = False   which samples are held out

Only the first is on by default.  With the other two fixed, a difference
between members is attributable to initialisation alone, which is what a seed
population is normally asked about.  Turn them on deliberately, when the
question is about data order or split sensitivity instead.

Note that ``vary_batch_order=False`` does **not** mean the training loader is
unshuffled — shuffling stays on, because stochastic gradient descent needs it.
It means every member sees the *same* shuffled order.

Seeds
-----
Passing ``seeds=None`` draws them from a recorded master seed via
:class:`numpy.random.SeedSequence`, which gives streams that are both distinct
and statistically independent.  Distinctness alone is not enough: two
arbitrarily chosen integers can seed correlated streams.  Explicit seeds are
checked for duplicates and stored either way, so a run started without a seed
is still reproducible afterwards.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from tanc._compat import callable_property
from tanc.model_extractor._views import (
    matrix_view, stack_views, view_shape, conv_view, conv_view_shape, CONV_VIEWS)

__all__ = [
    "SeedPlan",
    "PopulationMember",
    "TrainedPopulation",
    "train_population",
]


# ─────────────────────────────────────────────────────────────────────────────
# Seeds
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SeedPlan:
    """Independent seed streams for a population, derived from one master seed.

    Parameters
    ----------
    master : int or None
        ``None`` draws a master from system entropy and records it, so an
        unseeded run remains reproducible after the fact.
    n : int
        Number of members.

    Attributes
    ----------
    seeds : list of int
        One per member, distinct and independent.

    Examples
    --------
    >>> plan = SeedPlan(master=1234, n=3)
    >>> len(set(plan.seeds)) == 3
    True
    >>> SeedPlan(master=1234, n=3).seeds == plan.seeds     # reproducible
    True
    """

    master: int | None = None
    n: int = 1
    seeds: list[int] = field(default_factory=list)

    def __post_init__(self):
        if self.master is None:
            self.master = int(np.random.SeedSequence().entropy % (2 ** 31 - 1))
        if not self.seeds:
            children = np.random.SeedSequence(self.master).spawn(self.n)
            self.seeds = [int(c.generate_state(1, dtype=np.uint32)[0] % (2 ** 31 - 1))
                          for c in children]
        if len(set(self.seeds)) != len(self.seeds):
            dupes = [s for s in set(self.seeds) if self.seeds.count(s) > 1]
            raise ValueError(
                f"Population seeds must be distinct; {dupes} appear more than once. "
                f"Duplicated seeds produce duplicate members, which inflates any "
                f"across-member agreement statistic."
            )

    @classmethod
    def from_seeds(cls, seeds: Sequence[int]) -> "SeedPlan":
        """Build a plan from explicit seeds, validating distinctness."""
        s = [int(x) for x in seeds]
        return cls(master=None, n=len(s), seeds=s)

    def derive(self, purpose: str, index: int) -> int:
        """A seed for one *purpose* within member *index*.

        Separate streams per purpose keep initialisation, batch order and
        splitting independent, so their effects can be told apart.
        """
        base = np.random.SeedSequence([self.seeds[index], abs(hash(purpose)) % (2 ** 31)])
        return int(base.generate_state(1, dtype=np.uint32)[0] % (2 ** 31 - 1))


def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy and whichever frameworks are importable."""
    import random
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# The population
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PopulationMember:
    """One trained network, its seed, and what was captured from it."""

    seed: int
    snapshot: Any                      # ModelSnapshot (final state)
    view: Any = None                   # TrainingView, when checkpointing was on
    accuracy: float | None = None
    loss: float | None = None
    trained: bool = True
    seconds: float = 0.0

    def weight(self, layer: str, source: str = "auto") -> np.ndarray:
        """A named layer's array from this member's final snapshot.

        Parameters
        ----------
        layer : str
        source : {"auto", "kernels", "weights", "activations"}
            Which captured aspect to read.  ``"auto"`` prefers the convolutional
            kernel when one exists for this layer — an ``(out, in, H, W)`` array
            whose filters are the natural unit of analysis — and falls back to
            the ``(fan_in, out)`` weight matrix otherwise.

        Raises
        ------
        KeyError
            With the layers that *are* available for that source, since the
            commonest cause is a ``layer_selection`` that never captured it.
        """
        stores = {
            "kernels": getattr(self.snapshot, "kernel_weights", {}) or {},
            "weights": getattr(self.snapshot, "weights", {}) or {},
            "activations": getattr(self.snapshot, "activations", {}) or {},
        }
        if source == "auto":
            for key in ("kernels", "weights"):
                if layer in stores[key]:
                    return np.asarray(stores[key][layer])
            available = sorted(set(stores["kernels"]) | set(stores["weights"]))
            raise KeyError(
                f"No layer {layer!r} in this population. Available: {available}. "
                f"If a convolutional layer is missing, it was probably not captured — "
                f"train_population needs layer_selection='linear_and_conv' (its default) "
                f"rather than None, which captures linear layers only."
            )
        if source not in stores:
            raise ValueError(
                f"source must be 'auto', 'kernels', 'weights' or 'activations'; got {source!r}."
            )
        if layer not in stores[source]:
            raise KeyError(
                f"No layer {layer!r} in {source!r}. Available: {sorted(stores[source])}."
            )
        return np.asarray(stores[source][layer])


@dataclass
class TrainedPopulation:
    """A set of independently initialised networks, ready to read as a cloud.

    Attributes
    ----------
    members : list of PopulationMember
    plan : SeedPlan
    path : Path or None
        Where the population was saved, if it was.
    """

    members: list[PopulationMember]
    plan: SeedPlan
    path: Path | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.members)

    @property
    def seeds(self) -> list[int]:
        return callable_property("seeds", [m.seed for m in self.members])

    def __call__(self):
        """Return this population unchanged, with a deprecation warning.

        ``trained`` and ``untrained`` are properties, so ``pop.trained()`` is a
        stray pair of parentheses rather than a different call.  Accepting it
        keeps the mistake cheap; the tolerance is removed after one release.
        """
        from tanc._compat import warn_called
        warn_called("trained / untrained")
        return self

    @property
    def trained(self) -> "TrainedPopulation":
        """Only the trained members — the complement of :attr:`untrained`."""
        return TrainedPopulation([m for m in self.members if m.trained], self.plan,
                                 self.path, self.config)

    @property
    def untrained(self) -> "TrainedPopulation":
        """Only the untrained control members, if any were captured."""
        return TrainedPopulation([m for m in self.members if not m.trained], self.plan,
                                 self.path, self.config)

    def layer_names(self, source: str = "auto") -> list[str]:
        """Layers captured across the population, for a given source.

        ``"auto"`` reports every layer reachable as either a convolutional
        kernel or a weight matrix.
        """
        if not self.members:
            return []
        snap = self.members[0].snapshot
        kern = set(getattr(snap, "kernel_weights", {}) or {})
        wts = set(getattr(snap, "weights", {}) or {})
        acts = set(getattr(snap, "activations", {}) or {})
        return sorted({"auto": kern | wts, "kernels": kern,
                       "weights": wts, "activations": acts}[source])

    @property
    def accuracies(self) -> np.ndarray:
        """Validation accuracy per member, ``nan`` where none was measured.

        A property: it is a cheap read of what was already recorded.  Calling it
        with parentheses still works for one release, with a warning.
        """
        return callable_property("accuracies", np.array(
            [m.accuracy if m.accuracy is not None else np.nan
             for m in self.members], dtype=float))

    # ── the bridge to Mapper ────────────────────────────────────────────────

    def cloud(
        self,
        layer: str,
        view: str = "full",
        *,
        part: str = "upper",
        normalise: str | None = None,
        per_filter: bool = True,
        source: str = "auto",
    ) -> np.ndarray:
        """Stack a layer across the population into one point cloud.

        Parameters
        ----------
        layer : str
            Layer name, as reported by :meth:`layer_names`.
        view : str
            A key of :data:`~tanc.model_extractor._views.VIEWS`.
        part, normalise
            Passed to :func:`~tanc.model_extractor._views.matrix_view`.
        per_filter : bool
            For a convolutional layer of shape ``(out, in, *k)``, whether each
            output filter becomes its own matrix (default) or the whole layer
            is flattened to one ``(out, -1)`` matrix.  Per-filter is the usual
            reading — it treats a learned kernel as the unit of analysis.
        source : {"auto", "kernels", "weights", "activations"}
            Which captured aspect to read; see :meth:`PopulationMember.weight`.

        Returns
        -------
        (n_points, n_features) ndarray

        Examples
        --------
        >>> X = pop.cloud("conv1", "gram_diag", normalise="rows")   # doctest: +SKIP
        >>> MapperGrid(clouds={"conv1": X}, lens="pca2").run("runs/c1")  # doctest: +SKIP
        """
        raw = [np.asarray(m.weight(layer, source)) for m in self.members]

        # A convolutional weight is 4-D and has no single "row": route it through
        # the conv views, where each name says which axes index points.  Doing
        # this before _as_matrices is what makes view="rows" mean the rows of the
        # kernel rather than the rows of an arbitrary flattening of it.
        if raw and raw[0].ndim > 2 and view in CONV_VIEWS:
            parts = [conv_view(A, view) for A in raw]
            widths = {p.shape[1] for p in parts}
            if len(widths) != 1:
                raise ValueError(
                    f"Members disagree on feature width for view={view!r}: "
                    f"{sorted(widths)}. All members must share a layer shape."
                )
            out = np.vstack(parts)
            return _normalise_cloud(out, normalise)

        mats: list[np.ndarray] = []
        for A in raw:
            mats.extend(_as_matrices(A, per_filter=per_filter))
        return stack_views(mats, view, part=part, normalise=normalise)

    def cloud_shape(self, layer: str, view: str = "full", *, part: str = "upper",
                    per_filter: bool = True, source: str = "auto") -> tuple[int, int]:
        """Shape :meth:`cloud` would return, without building it.

        Worth checking before a grid: the Gram views can be far wider than the
        matrix they come from.
        """
        if not self.members:
            return (0, 0)
        mats = _as_matrices(self.members[0].weight(layer, source), per_filter=per_filter)
        raw0 = np.asarray(self.members[0].weight(layer, source))
        if raw0.ndim > 2 and view in CONV_VIEWS:
            n_per_member, feats = conv_view_shape(raw0.shape, view)
            return (n_per_member * len(self.members), feats)
        n_per_matrix, feats = view_shape(mats[0].shape, view, part)
        return (n_per_matrix * len(mats) * len(self.members), feats)

    def member_index(self, layer: str, view: str = "full", *,
                     per_filter: bool = True, source: str = "auto") -> np.ndarray:
        """Which member each point of :meth:`cloud` came from.

        This is the label behind the cross-member colourings — whether a Mapper
        node mixes many networks or belongs to one.
        """
        counts = []
        for m in self.members:
            mats = _as_matrices(m.weight(layer, source), per_filter=per_filter)
            raw = np.asarray(m.weight(layer, source))
            if raw.ndim > 2 and view in CONV_VIEWS:
                counts.append(conv_view_shape(raw.shape, view)[0])
                continue
            n_per, _ = view_shape(mats[0].shape, view)
            counts.append(len(mats) * n_per)
        return np.repeat(np.arange(len(self.members)), counts)

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: Any) -> Path:
        """Write weights and provenance to *path*, never overwriting.

        Weights go to one compressed ``.npz`` per member — a plain array format
        that outlives the Python and library versions that produced it — with
        seeds, metrics and configuration in ``manifest.json`` beside them.
        """
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        target, n = base, 1
        while True:
            try:
                target.mkdir()
                break
            except FileExistsError:
                n += 1
                target = base.with_name(f"{base.name}-{n:03d}")

        # Weight matrices and convolutional kernels are stored in one file per
        # member, prefixed so the two can be told apart on load.  Both are kept:
        # a conv layer's (out, in, H, W) kernel is not recoverable from the
        # flattened (fan_in, out) matrix once the kernel axes are collapsed.
        for i, m in enumerate(self.members):
            arrays: dict[str, np.ndarray] = {}
            for k, v in (getattr(m.snapshot, "weights", {}) or {}).items():
                arrays[f"w/{k}"] = np.asarray(v)
            for k, v in (getattr(m.snapshot, "kernel_weights", {}) or {}).items():
                arrays[f"k/{k}"] = np.asarray(v)
            np.savez_compressed(target / f"member_{i:04d}.npz", **arrays)

        manifest = {
            "n_members": len(self.members),
            "master_seed": self.plan.master,
            "seeds": self.seeds,
            "trained": [bool(m.trained) for m in self.members],
            "accuracy": [m.accuracy for m in self.members],
            "loss": [m.loss for m in self.members],
            "seconds": [m.seconds for m in self.members],
            "layers": self.layer_names(),
            "config": self.config,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        self.path = target
        return target

    @classmethod
    def load(cls, path: Any) -> "TrainedPopulation":
        """Read a population written by :meth:`save`.

        The returned members carry weights and metadata but no live model — the
        analyses in this toolkit read weights, so a model is only needed if you
        intend to run further forward passes.
        """
        base = Path(path)
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        members: list[PopulationMember] = []
        for i in range(manifest["n_members"]):
            with np.load(base / f"member_{i:04d}.npz", allow_pickle=False) as z:
                weights = {k[2:]: z[k] for k in z.files if k.startswith("w/")}
                kernels = {k[2:]: z[k] for k in z.files if k.startswith("k/")}
                # Files written before the prefix was introduced hold weights only.
                if not weights and not kernels:
                    weights = {k: z[k] for k in z.files}
            members.append(PopulationMember(
                seed=manifest["seeds"][i],
                snapshot=_WeightsOnly(weights, kernels),
                accuracy=manifest["accuracy"][i],
                loss=manifest["loss"][i],
                trained=manifest["trained"][i],
                seconds=manifest.get("seconds", [0.0] * manifest["n_members"])[i],
            ))
        plan = SeedPlan(master=manifest.get("master_seed"),
                        n=manifest["n_members"], seeds=list(manifest["seeds"]))
        return cls(members, plan, path=base, config=manifest.get("config", {}))

    def __repr__(self) -> str:  # pragma: no cover - display only
        n_un = sum(1 for m in self.members if not m.trained)
        acc = self.accuracies
        acc_s = f", acc {np.nanmean(acc):.3f}±{np.nanstd(acc):.3f}" if np.isfinite(acc).any() else ""
        return (f"TrainedPopulation({len(self.members)} members"
                + (f", {n_un} untrained" if n_un else "") + acc_s + ")")


@dataclass
class _WeightsOnly:
    """Minimal stand-in for a ModelSnapshot when a population is loaded from disk.

    Carries the arrays the analyses read.  Activations are deliberately not
    stored — they are a function of the weights and the input data, so keeping
    them would multiply file size to preserve something recomputable.
    """

    weights: dict[str, np.ndarray] = field(default_factory=dict)
    kernel_weights: dict[str, np.ndarray] = field(default_factory=dict)
    activations: dict[str, np.ndarray] = field(default_factory=dict)


def _normalise_cloud(X: np.ndarray, how: str | None) -> np.ndarray:
    """Apply `matrix_view`'s normalise options to an already-built conv cloud."""
    if how is None:
        return X
    if how == "rows":
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    if how == "correlation":
        C = X - X.mean(axis=1, keepdims=True)
        return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    raise ValueError(
        f"normalise must be None, 'rows' or 'correlation'; got {how!r}.")


def _as_matrices(W: np.ndarray, *, per_filter: bool) -> list[np.ndarray]:
    """Read a weight array as a list of 2-D matrices.

    A ``(out, in)`` linear weight is one matrix.  A convolutional
    ``(out, in, *kernel)`` array becomes one matrix per output filter when
    *per_filter*, which treats a learned kernel as the unit of analysis, or a
    single ``(out, -1)`` matrix otherwise.
    """
    A = np.asarray(W)
    if A.ndim == 1:
        return [A[None, :]]
    if A.ndim == 2:
        return [A]
    if not per_filter:
        return [A.reshape(A.shape[0], -1)]
    return [A[i].reshape(A.shape[1], -1) if A[i].ndim > 1 else A[i][None, :]
            for i in range(A.shape[0])]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_population(
    model_fn: Callable[[], Any],
    train_data: Any,
    *,
    val_data: Any = None,
    extract_data: Any = None,
    n_models: int = 1,
    seeds: Sequence[int] | None = None,
    master_seed: int | None = None,
    epochs: int = 10,
    batch_size: int = 128,
    criterion: Any = None,
    optimizer_fn: Callable[[Any], Any] | None = None,
    compile_kwargs: dict | None = None,
    vary_init: bool = True,
    vary_batch_order: bool = False,
    vary_split: bool = False,
    checkpoint_every: Any = None,
    include_untrained: bool = False,
    aspects: list[str] | None = None,
    layer_selection: Any = "linear_and_conv",
    save_to: Any = None,
    device: str | None = None,
    verbose: bool = True,
) -> TrainedPopulation:
    """Train independently initialised networks and keep what they learned.

    Parameters
    ----------
    model_fn : callable
        Returns a **fresh, untrained** model each call.  Called once per member,
        after that member's seed is set, so initialisation follows the seed.
    train_data, val_data : array-like or loader
        Passed through to :func:`~tanc.model_extractor.extract_training`.
    extract_data : array-like, optional
        Fixed data for the snapshot forward pass.  Only needed when activations
        are captured; weights alone do not require it.
    n_models : int
        Members to train.  Ignored when *seeds* is given.
    seeds : sequence of int, optional
        Explicit seeds, validated for duplicates.  ``None`` derives independent
        seeds from *master_seed*.
    master_seed : int, optional
        ``None`` draws one from entropy and records it.
    epochs : int
        Training epochs per member.  ``0`` captures the initialised model
        without training it.
    criterion, optimizer_fn : callable
        PyTorch only.  ``optimizer_fn(model) -> optimizer`` is called per
        member, so each gets its own optimiser over its own parameters.
    compile_kwargs : dict, optional
        TensorFlow only; forwarded to ``model.compile``.
    vary_init, vary_batch_order, vary_split : bool
        Which sources of randomness differ between members.  See the module
        docstring — only initialisation varies by default.
    checkpoint_every : None, int, or {"epoch", "iteration"}
        ``None`` (default) keeps only the final state.  Otherwise a training
        trajectory is captured, at the cost of roughly
        ``n_parameters * 4 bytes`` per checkpoint per member, which grows
        quickly; prefer ``"epoch"`` before ``"iteration"``.
    include_untrained : bool
        Also capture each member at initialisation, as a matched control.
        Equivalent to a second population at ``epochs=0`` with the same seeds,
        and implemented that way.
    aspects, device, verbose
        Forwarded to extraction.
    layer_selection : str or list
        Forwarded to extraction, but defaulting to ``"linear_and_conv"`` rather
        than the ``None`` used elsewhere in the toolkit.  ``None`` captures
        **linear layers only**, which would silently leave a convolutional
        population with nothing to analyse.  Pass ``None`` explicitly for the
        linear-only behaviour.
    save_to : str or Path, optional
        Save immediately after training.  Recommended — training is the
        expensive step and a lost population cannot be recovered exactly
        without it.

    Returns
    -------
    TrainedPopulation

    Notes
    -----
    Seed-exact reproduction is reliable on CPU.  On GPU, and for TensorFlow in
    particular, nondeterministic kernels mean two runs with identical seeds can
    differ slightly; the seeds are still recorded, and the population remains
    statistically reproducible.
    """
    from tanc.model_extractor import extract_model, extract_training

    plan = (SeedPlan.from_seeds(seeds) if seeds is not None
            else SeedPlan(master=master_seed, n=n_models))
    members: list[PopulationMember] = []

    if verbose:
        print(f"train_population: {len(plan.seeds)} members, master seed {plan.master}")
        varying = [n for n, on in (("init", vary_init), ("batch order", vary_batch_order),
                                   ("split", vary_split)) if on]
        print(f"  varying: {', '.join(varying) if varying else 'nothing'}")

    for i, seed in enumerate(plan.seeds):
        t0 = time.perf_counter()
        _seed_everything(plan.derive("init", i) if vary_init else plan.seeds[0])

        if include_untrained or epochs == 0:
            model = model_fn()
            snap = extract_model(model=model, data=extract_data, aspects=aspects,
                                 layer_selection=layer_selection, clarify=False,
                                 device=device)
            members.append(PopulationMember(seed=seed, snapshot=snap, trained=False,
                                            seconds=time.perf_counter() - t0))
            if epochs == 0:
                continue

        model = model_fn()
        if vary_batch_order:
            _seed_everything(plan.derive("batch_order", i))
        elif vary_init:
            # Re-seed to a fixed stream so batch order matches across members
            # even though their initial weights differ.
            _seed_everything(plan.seeds[0] + 1)

        kwargs: dict[str, Any] = dict(
            model=model, extract_data=extract_data, train_data=train_data,
            val_data=val_data, batch_size=batch_size, epochs=epochs,
            aspects=aspects, layer_selection=layer_selection, clarify=False,
            device=device, verbose=False, target_accuracy=None,
        )
        if criterion is not None:
            kwargs["criterion"] = criterion
        if optimizer_fn is not None:
            kwargs["optimizer"] = optimizer_fn(model)
        if compile_kwargs is not None:
            kwargs["compile_kwargs"] = compile_kwargs
        if checkpoint_every is None:
            kwargs["snapshot_every"] = max(epochs, 1)      # final state only
            kwargs["snapshot_schedule"] = "epoch"
        elif isinstance(checkpoint_every, str):
            kwargs["snapshot_every"] = 1
            kwargs["snapshot_schedule"] = checkpoint_every
        else:
            kwargs["snapshot_every"] = int(checkpoint_every)
            kwargs["snapshot_schedule"] = "epoch"

        view = extract_training(**kwargs)
        final = view.final_snapshot          # a property, not a method
        members.append(PopulationMember(
            seed=seed, snapshot=final,
            view=view if checkpoint_every is not None else None,
            accuracy=getattr(final, "accuracy", None),
            loss=getattr(final, "loss", None),
            trained=True, seconds=time.perf_counter() - t0,
        ))
        if verbose:
            acc = getattr(final, "accuracy", None)
            acc_s = f", acc {acc:.3f}" if isinstance(acc, (int, float)) else ""
            print(f"  [{i + 1}/{len(plan.seeds)}] seed {seed}"
                  f"{acc_s}  ({time.perf_counter() - t0:.1f}s)")

    pop = TrainedPopulation(
        members, plan,
        config={
            "epochs": epochs, "batch_size": batch_size,
            "vary_init": vary_init, "vary_batch_order": vary_batch_order,
            "vary_split": vary_split, "checkpoint_every": checkpoint_every,
            "include_untrained": include_untrained,
        },
    )
    if save_to is not None:
        target = pop.save(save_to)
        if verbose:
            print(f"  saved -> {target}")
    return pop
