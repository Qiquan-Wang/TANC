"""extractors.py — class-based entry points for model and training extraction.

Two builder-style classes mirroring the ``TDAPipeline`` convention:

* :class:`ModelExtractor` — configure once, extract many snapshots with
  :meth:`ModelExtractor.extract`.
* :class:`TrainingExtractor` — configure training infra + extraction at
  construction (via :meth:`TrainingExtractor.for_pytorch` /
  :meth:`TrainingExtractor.for_tensorflow`), then call :meth:`run` to
  train and collect a :class:`TrainingView`.

The one-call functions ``extract_model`` and ``extract_training`` in
``model_extractor/__init__.py`` are thin wrappers around these classes
and remain fully backwards-compatible.
"""

from __future__ import annotations

from typing import Any

from tanc.model_extractor._snapshot import ModelSnapshot, TrainingView
from tanc.model_extractor._inspector import (
    detect_framework, inspect_model, resolve_layer_selection,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers — duplicated from __init__ to avoid a circular import.
# ─────────────────────────────────────────────────────────────────────────────

_VALID_ASPECTS: frozenset[str] = frozenset(
    {"weights", "activations", "classifications"}
)
_DEFAULT_ASPECTS: list[str] = ["weights", "activations", "classifications"]


def _validate_aspects(aspects: list[str] | None) -> list[str]:
    if aspects is None:
        return list(_DEFAULT_ASPECTS)
    bad = [a for a in aspects if a not in _VALID_ASPECTS]
    if bad:
        raise ValueError(
            f"Unknown aspect(s): {bad}. Valid: {sorted(_VALID_ASPECTS)}"
        )
    return list(aspects)


def _require_framework(model: Any) -> str:
    framework = detect_framework(model)
    if framework == "unknown":
        raise ValueError(
            f"Unsupported model type '{type(model).__name__}'. "
            "Supported: torch.nn.Module, tf.keras.Model."
        )
    return framework


# ─────────────────────────────────────────────────────────────────────────────
# Dataset → loader coercion — lets the user hand over a raw dataset and have
# the right loader built for the detected framework.
# ─────────────────────────────────────────────────────────────────────────────

def _as_torch_tensor(arr: Any, *, target: bool):
    """Convert *arr* to a torch tensor with a sensible dtype.

    Inputs become ``float32``; integer targets become ``int64`` (classification
    labels) while floating targets stay ``float32`` (regression).  Tensors are
    passed through unchanged so the caller keeps full control when needed.
    """
    import numpy as np
    import torch

    if isinstance(arr, torch.Tensor):
        return arr
    np_arr = np.asarray(arr)
    if target and np.issubdtype(np_arr.dtype, np.integer):
        return torch.as_tensor(np_arr, dtype=torch.long)
    return torch.as_tensor(np_arr, dtype=torch.float32)


def _build_pytorch_loader(data: Any, batch_size: int, shuffle: bool):
    """Coerce *data* into a torch ``DataLoader``.

    Accepts, in order:

    * ``None`` → ``None`` (no loader).
    * a ``DataLoader`` → returned unchanged.
    * a ``Dataset`` → wrapped in a ``DataLoader``.
    * an ``(X, y)`` tuple/list of arrays or tensors → wrapped in a
      ``TensorDataset`` then a ``DataLoader``.
    """
    if data is None:
        return None
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    if isinstance(data, DataLoader):
        return data
    if isinstance(data, Dataset):
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    if isinstance(data, (tuple, list)) and len(data) == 2:
        X, y = data
        dataset = TensorDataset(
            _as_torch_tensor(X, target=False),
            _as_torch_tensor(y, target=True),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    raise TypeError(
        "train_data / val_data for a PyTorch model must be a DataLoader, a "
        "torch Dataset, or an (X, y) tuple of arrays/tensors; got "
        f"{type(data).__name__}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ModelExtractor — single snapshot from a trained model
# ─────────────────────────────────────────────────────────────────────────────

class ModelExtractor:
    """Configure once, extract one or more :class:`ModelSnapshot` objects.

    The full model inspection (layer resolution, framework detection) is
    done at construction time so :meth:`explain` can show the plan before
    you spend cycles on a forward pass.

    Parameters
    ----------
    model
        A trained ``torch.nn.Module`` or ``tf.keras.Model``.
    aspects : list[str] or None
        Subset of ``["weights", "activations", "classifications"]``.
        Defaults to all three.
    layer_selection : str, list[str], list[int], or None
        Same semantics as :func:`tanc.model_extractor.extract_model`.
    clarify : bool
        Interactive layer-resolution prompt for ambiguous architectures.
    device : str or None
        PyTorch device string; ignored for TensorFlow.
    activation_pooling : str
        How to collapse each captured activation to ``(N_samples, features)``.
        ``"flatten"`` (default) reshapes ``(N, …) → (N, -1)`` — the historical
        behaviour, so MLP/CNN extraction is unchanged.  For transformer
        ``(batch, seq, hidden)`` activations, ``"last"`` / ``"first"`` /
        ``"mean"`` / ``"max"`` pool over the sequence axis (``"last"`` = the
        final-token summary of a causal decoder).
    soro_effective_weight : bool
        For Sum-of-Rank-One (SoRO) layers, also store the assembled effective
        weight ``W = U·diag(sigma)·Vᵀ`` in ``weights`` (default ``True``) so
        they feed the weight-based tools like FC layers; the trained factors are
        always kept in ``snapshot.soro_factors``.

    Examples
    --------
    >>> extractor = ModelExtractor(model, aspects=["weights", "activations"])
    >>> extractor.explain()
    >>> snap_train = extractor.extract(X_train)
    >>> snap_test  = extractor.extract(X_test)   # same model, different data
    """

    def __init__(
        self,
        model: Any,
        aspects: list[str] | None = None,
        layer_selection: str | list[str] | list[int] | None = None,
        clarify: bool = True,
        device: str | None = None,
        activation_pooling: str = "flatten",
        soro_effective_weight: bool = True,
    ) -> None:
        self.model = model
        self.framework = _require_framework(model)
        self.aspects = _validate_aspects(aspects)
        self._model_info = inspect_model(model)
        self.layer_names = resolve_layer_selection(
            self._model_info, self.aspects, layer_selection, clarify
        )
        self.device = device
        self.activation_pooling = activation_pooling
        self.soro_effective_weight = soro_effective_weight

    # ── Introspection ───────────────────────────────────────────────────────

    def configure(self, **kwargs) -> "ModelExtractor":
        """Update fields in bulk after construction.  Returns self for chaining."""
        for k, v in kwargs.items():
            if k == "aspects":
                self.aspects = _validate_aspects(v)
            elif k in {"layer_names", "device", "activation_pooling",
                       "soro_effective_weight"}:
                setattr(self, k, v)
            elif k == "layer_selection":
                self.layer_names = resolve_layer_selection(
                    self._model_info, self.aspects, v, clarify=False
                )
            else:
                raise AttributeError(
                    f"ModelExtractor has no configurable field '{k}'."
                )
        return self

    def explain(self, print_it: bool = True) -> str:
        """Plain-English summary of what :meth:`extract` will do."""
        lines = [
            f"ModelExtractor (framework={self.framework})",
            f"  aspects        : {self.aspects}",
            f"  layers ({len(self.layer_names)}): {self.layer_names}",
            f"  device         : {self.device or 'auto'}",
        ]
        text = "\n".join(lines)
        if print_it:
            print(text)
        return text

    # ── Execution ───────────────────────────────────────────────────────────

    def extract(self, data: Any) -> ModelSnapshot:
        """Run a forward pass on *data* and return a :class:`ModelSnapshot`."""
        if self.framework == "pytorch":
            from tanc.model_extractor._pytorch import extract_from_pytorch
            return extract_from_pytorch(
                model=self.model,
                data=data,
                aspects=self.aspects,
                layer_names=self.layer_names,
                device=self.device,
                activation_pooling=self.activation_pooling,
                soro_effective_weight=self.soro_effective_weight,
            )
        # tensorflow
        from tanc.model_extractor._tensorflow import extract_from_tf
        return extract_from_tf(
            model=self.model,
            data=data,
            aspects=self.aspects,
            layer_names=self.layer_names,
            activation_pooling=self.activation_pooling,
            soro_effective_weight=self.soro_effective_weight,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TrainingExtractor — train + capture snapshots
# ─────────────────────────────────────────────────────────────────────────────

class TrainingExtractor:
    """Train a model and capture snapshots at regular intervals.

    The framework is detected automatically from ``model``.  The simplest
    way to supply data is the cross-framework ``train_data`` / ``val_data``
    pair — hand over a raw dataset and the extractor builds the right loader
    for the detected framework:

    * **PyTorch** — ``(X, y)`` arrays/tensors are wrapped in a
      ``TensorDataset`` + ``DataLoader`` (``batch_size`` / ``shuffle``
      control batching); a ``Dataset`` is wrapped; a ``DataLoader`` is used
      as-is.  Still pass ``criterion`` and ``optimizer``.
    * **TensorFlow / Keras** — ``train_data`` / ``val_data`` are routed to
      ``model.fit`` (an ``(X, y)`` tuple or a ``tf.data.Dataset``); pass
      ``compile_kwargs`` if the model isn't compiled yet.

    A model that is neither PyTorch nor TensorFlow raises ``ValueError``
    (``Unsupported model type ...``) at construction.

    Backwards compatibility: the original PyTorch-specific ``train_loader`` /
    ``val_loader`` kwargs still work and take precedence over ``train_data`` /
    ``val_data`` when both are given.  The required kwargs for the detected
    framework are validated eagerly at construction time so you don't lose a
    long run to a missing argument.

    The :meth:`for_pytorch` / :meth:`for_tensorflow` factories remain
    available when you want an explicit, framework-specific signature.

    The :meth:`run` method does the training and returns a
    :class:`TrainingView`.  Each :meth:`run` call defaults to a *fresh*
    extraction — pass ``continue_training=True`` to append new snapshots
    to the previous run (the underlying model object continues to train
    in place because it is shared).

    Examples
    --------
    Auto-detected framework, raw dataset (preferred):

    >>> extractor = TrainingExtractor(
    ...     model        = model,                 # torch.nn.Module *or* tf.keras.Model
    ...     train_data   = (X_train, y_train),    # raw arrays — loader built for you
    ...     val_data     = (X_test,  y_test),     # held-out / test set
    ...     batch_size   = 128,
    ...     criterion    = nn.CrossEntropyLoss(), # torch only
    ...     optimizer    = torch.optim.Adam(model.parameters(), lr=1e-3),  # torch only
    ...     extract_data = X_test,
    ...     aspects      = ["weights", "activations"],
    ...     snapshot_every = 5,
    ... )
    >>> extractor.explain()
    >>> view = extractor.run(epochs=50, target_accuracy=0.95)
    >>> view = extractor.run(epochs=10, continue_training=True)

    Pre-built loaders still work (PyTorch, backwards compatible):

    >>> extractor = TrainingExtractor(model=model, train_loader=train_loader,
    ...     val_loader=val_loader, criterion=..., optimizer=..., extract_data=X_test)

    Explicit factory (equivalent — useful for clarity):

    >>> extractor = TrainingExtractor.for_pytorch(model=model, train_data=..., ...)
    """

    # Framework-exclusive kwarg names — used for framework-mismatch warnings.
    # ``train_data`` / ``val_data`` / ``batch_size`` / ``shuffle`` are shared
    # and therefore not listed here.
    _PT_ONLY_KWARGS = ("train_loader", "val_loader", "criterion", "optimizer", "scheduler")
    _TF_ONLY_KWARGS = ("compile_kwargs",)

    def __init__(
        self,
        model: Any,
        *,
        # Cross-framework data (preferred) — a raw dataset per split.
        train_data: Any = None,
        val_data: Any = None,
        batch_size: int = 128,
        shuffle: bool = True,
        # PyTorch-specific
        train_loader: Any = None,
        criterion: Any = None,
        optimizer: Any = None,
        val_loader: Any = None,
        scheduler: Any = None,
        loss_eval_data: Any = None,
        # TensorFlow-specific
        compile_kwargs: dict | None = None,
        # Shared
        extract_data: Any = None,
        aspects: list[str] | None = None,
        layer_selection: str | list[str] | list[int] | None = None,
        clarify: bool = True,
        snapshot_every: int = 1,
        snapshot_schedule: str = "epoch",
        device: str | None = None,
        # Internal — the factories set this to skip detection.
        framework: str | None = None,
    ) -> None:
        if framework is None:
            framework = _require_framework(model)

        if framework == "pytorch":
            # Explicit loaders win; otherwise build them from the raw dataset.
            if train_loader is None:
                train_loader = _build_pytorch_loader(
                    train_data, batch_size, shuffle=shuffle
                )
            if val_loader is None:
                val_loader = _build_pytorch_loader(
                    val_data, batch_size, shuffle=False
                )

            missing = []
            if train_loader is None: missing.append("train_data (or train_loader)")
            if criterion is None:    missing.append("criterion")
            if optimizer is None:    missing.append("optimizer")
            if missing:
                raise ValueError(
                    "TrainingExtractor detected a PyTorch model but the "
                    f"following required argument(s) were not provided: "
                    f"{missing}. Pass them as keyword arguments."
                )
            # Warn if TF-only kwargs were also supplied — likely user mistake.
            _args = locals()
            stray = [k for k in self._TF_ONLY_KWARGS if _args[k] is not None]
            if stray:
                import warnings as _w
                _w.warn(
                    f"Ignoring TensorFlow-only kwargs for a PyTorch model: "
                    f"{stray}.",
                    UserWarning, stacklevel=2,
                )
            self.framework_kwargs = dict(
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                val_loader=val_loader,
                scheduler=scheduler,
                loss_eval_data=loss_eval_data,
            )
        elif framework == "tensorflow":
            if train_data is None:
                raise ValueError(
                    "TrainingExtractor detected a TensorFlow model but "
                    "train_data was not provided. Pass train_data=..."
                )
            _args = locals()
            stray = [k for k in self._PT_ONLY_KWARGS if _args[k] is not None]
            if stray:
                import warnings as _w
                _w.warn(
                    f"Ignoring PyTorch-only kwargs for a TensorFlow model: "
                    f"{stray}.",
                    UserWarning, stacklevel=2,
                )
            self.framework_kwargs = dict(
                train_data=train_data,
                val_data=val_data,
                compile_kwargs=compile_kwargs,
                batch_size=batch_size,
            )
            device = None
        else:
            raise ValueError(f"Unsupported framework '{framework}'.")

        self.model = model
        self.framework = framework
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.extract_data = extract_data
        self.aspects = _validate_aspects(aspects)
        self.snapshot_every = int(snapshot_every)
        self.snapshot_schedule = snapshot_schedule
        self.device = device

        self._model_info = inspect_model(model)
        self.layer_names = resolve_layer_selection(
            self._model_info, self.aspects, layer_selection, clarify
        )
        self._last_view: TrainingView | None = None

    # ── Framework factories (explicit shortcuts) ───────────────────────────

    @classmethod
    def for_pytorch(
        cls,
        model: Any,
        extract_data: Any,
        train_data: Any = None,
        val_data: Any = None,
        batch_size: int = 128,
        shuffle: bool = True,
        train_loader: Any = None,
        criterion: Any = None,
        optimizer: Any = None,
        val_loader: Any = None,
        scheduler: Any = None,
        aspects: list[str] | None = None,
        layer_selection: str | list[str] | list[int] | None = None,
        clarify: bool = True,
        snapshot_every: int = 1,
        snapshot_schedule: str = "epoch",
        device: str | None = None,
    ) -> "TrainingExtractor":
        """Explicit factory for a PyTorch model.

        Equivalent to ``TrainingExtractor(model, train_data=..., ...)``
        with auto-detection, but asserts the model is actually a
        ``torch.nn.Module`` first.  Use when you want the framework
        choice to be self-documenting in code.  Supply either ``train_data``
        (a raw ``(X, y)`` / ``Dataset``, batched with ``batch_size`` /
        ``shuffle``) or a pre-built ``train_loader``.
        """
        if detect_framework(model) != "pytorch":
            raise ValueError(
                f"for_pytorch expects a torch.nn.Module, got {type(model).__name__}."
            )
        return cls(
            model=model,
            train_data=train_data,
            val_data=val_data,
            batch_size=batch_size,
            shuffle=shuffle,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            val_loader=val_loader,
            scheduler=scheduler,
            extract_data=extract_data,
            aspects=aspects,
            layer_selection=layer_selection,
            clarify=clarify,
            snapshot_every=snapshot_every,
            snapshot_schedule=snapshot_schedule,
            device=device,
            framework="pytorch",
        )

    @classmethod
    def for_tensorflow(
        cls,
        model: Any,
        train_data: Any,
        extract_data: Any,
        val_data: Any = None,
        batch_size: int = 128,
        compile_kwargs: dict | None = None,
        aspects: list[str] | None = None,
        layer_selection: str | list[str] | list[int] | None = None,
        clarify: bool = True,
        snapshot_every: int = 1,
        snapshot_schedule: str = "epoch",
    ) -> "TrainingExtractor":
        """Explicit factory for a TensorFlow / Keras model.

        Equivalent to ``TrainingExtractor(model, train_data=..., ...)``
        with auto-detection.
        """
        if detect_framework(model) != "tensorflow":
            raise ValueError(
                f"for_tensorflow expects a tf.keras.Model, got {type(model).__name__}."
            )
        return cls(
            model=model,
            train_data=train_data,
            val_data=val_data,
            batch_size=batch_size,
            compile_kwargs=compile_kwargs,
            extract_data=extract_data,
            aspects=aspects,
            layer_selection=layer_selection,
            clarify=clarify,
            snapshot_every=snapshot_every,
            snapshot_schedule=snapshot_schedule,
            framework="tensorflow",
        )

    # ── Introspection ───────────────────────────────────────────────────────

    @property
    def view(self) -> TrainingView | None:
        """Most recent :class:`TrainingView` produced by :meth:`run`."""
        return self._last_view

    def configure(self, **kwargs) -> "TrainingExtractor":
        """Update fields in bulk after construction.  Returns self for chaining."""
        for k, v in kwargs.items():
            if k == "aspects":
                self.aspects = _validate_aspects(v)
            elif k == "layer_selection":
                self.layer_names = resolve_layer_selection(
                    self._model_info, self.aspects, v, clarify=False
                )
            elif k in {
                "extract_data", "snapshot_every", "snapshot_schedule",
                "device", "layer_names",
            }:
                setattr(self, k, v)
            elif k in self.framework_kwargs:
                self.framework_kwargs[k] = v
            else:
                raise AttributeError(
                    f"TrainingExtractor has no configurable field '{k}'."
                )
        return self

    def reset(self) -> "TrainingExtractor":
        """Drop the cached :attr:`view` (does not reset the model itself)."""
        self._last_view = None
        return self

    def explain(self, print_it: bool = True) -> str:
        """Plain-English summary of what :meth:`run` will do."""
        lines = [f"TrainingExtractor (framework={self.framework})"]
        lines.append(f"  aspects          : {self.aspects}")
        lines.append(
            f"  layers ({len(self.layer_names)}): {self.layer_names}"
        )
        lines.append(
            f"  snapshots        : every {self.snapshot_every} "
            f"{self.snapshot_schedule}(s)"
        )
        lines.append(f"  batch_size       : {self.batch_size}")
        lines.append(f"  device           : {self.device or 'auto'}")
        for k, v in self.framework_kwargs.items():
            if v is None:
                lines.append(f"  {k:14s}   : (not set)")
            else:
                lines.append(f"  {k:14s}   : {type(v).__name__}")
        if self._last_view is not None:
            lines.append(
                f"  previous view    : {len(self._last_view.snapshots)} snapshots"
            )
        text = "\n".join(lines)
        if print_it:
            print(text)
        return text

    # ── Execution ───────────────────────────────────────────────────────────

    def run(
        self,
        epochs: int = 100,
        target_accuracy: float | None = 0.98,
        verbose: bool = True,
        continue_training: bool = False,
    ) -> TrainingView:
        """Train the model and capture snapshots.

        Parameters
        ----------
        epochs : int
            Maximum number of training epochs (default 100).
        target_accuracy : float or None
            Early-stop threshold.  ``None`` runs the full ``epochs``.
        verbose : bool
            Print training progress and snapshot info.
        continue_training : bool
            ``False`` (default) — fresh run: the cached :attr:`view` is
            dropped before training starts and the new view replaces it.

            ``True`` — append the new snapshots to the previous
            :attr:`view`.  The underlying model object is reused, so its
            training state carries over automatically.

        Returns
        -------
        TrainingView
            The captured snapshots — either the new run's view or, when
            ``continue_training=True``, the cumulative view.
        """
        if not continue_training:
            self._last_view = None

        if self.framework == "pytorch":
            from tanc.model_extractor._pytorch import train_and_extract_pytorch
            new_view = train_and_extract_pytorch(
                model=self.model,
                train_loader=self.framework_kwargs["train_loader"],
                criterion=self.framework_kwargs["criterion"],
                optimizer=self.framework_kwargs["optimizer"],
                aspects=self.aspects,
                layer_names=self.layer_names,
                extract_data=self.extract_data,
                epochs=epochs,
                target_accuracy=target_accuracy,
                snapshot_every=self.snapshot_every,
                snapshot_schedule=self.snapshot_schedule,
                val_loader=self.framework_kwargs.get("val_loader"),
                scheduler=self.framework_kwargs.get("scheduler"),
                loss_eval_data=self.framework_kwargs.get("loss_eval_data"),
                device=self.device,
                verbose=verbose,
            )
        else:
            from tanc.model_extractor._tensorflow import train_and_extract_tf
            new_view = train_and_extract_tf(
                model=self.model,
                train_data=self.framework_kwargs["train_data"],
                aspects=self.aspects,
                layer_names=self.layer_names,
                extract_data=self.extract_data,
                epochs=epochs,
                target_accuracy=target_accuracy,
                snapshot_every=self.snapshot_every,
                val_data=self.framework_kwargs.get("val_data"),
                compile_kwargs=self.framework_kwargs.get("compile_kwargs"),
                batch_size=self.framework_kwargs.get("batch_size"),
                verbose=verbose,
            )

        if continue_training and self._last_view is not None:
            self._last_view.snapshots.extend(new_view.snapshots)
            self._last_view.total_epochs += new_view.total_epochs
        else:
            self._last_view = new_view
        return self._last_view