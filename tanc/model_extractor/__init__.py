"""model_extractor — extract weights, activations, and classifications from
neural network models for downstream TDA analysis.

This module sits **upstream** of the rest of the toolkit: it takes any
supported neural-network model, runs a smart inspection of its architecture,
and produces :class:`ModelSnapshot` / :class:`TrainingView` objects whose
arrays feed directly into :mod:`tanc.graph_builder`.

Supported frameworks
--------------------
* **PyTorch** — ``torch.nn.Module``
* **TensorFlow / Keras** — ``tf.keras.Model``

Two extraction modes
--------------------
**Final view** — extract from an already-trained model:

.. code-block:: python

    from tanc.model_extractor import extract_model

    snapshot = extract_model(
        model  = my_trained_mlp,
        data   = X_test,
        aspects = ["weights", "activations", "classifications"],
    )

    # Feed directly into graph_builder:
    from tanc.graph_builder import build_weight_graph, build_activation_graph

    bundle_W = build_weight_graph(snapshot.weight_matrices())
    bundle_A = build_activation_graph(snapshot.all_activation_matrices()[0])

**Training view** — hook into the training loop and collect snapshots:

.. code-block:: python

    from tanc.model_extractor import extract_training

    # PyTorch — hand over the raw dataset; the loader is built for you
    view = extract_training(
        model        = model,
        train_data   = (X_train, y_train),
        val_data     = (X_test,  y_test),
        batch_size   = 128,
        criterion    = nn.CrossEntropyLoss(),
        optimizer    = torch.optim.Adam(model.parameters(), lr=1e-3),
        extract_data = X_test,
        aspects      = ["weights", "activations"],
        epochs       = 50,
        snapshot_every = 5,
        target_accuracy = 0.95,
    )

    traj   = view.weight_trajectory()          # list[list[ndarray]]
    bundle = build_weight_trajectory(traj)     # → GraphBundle

    # TensorFlow / Keras
    view = extract_training(
        model        = keras_model,
        train_data   = (X_train, y_train),
        compile_kwargs = dict(
            optimizer = "adam",
            loss      = "sparse_categorical_crossentropy",
            metrics   = ["accuracy"],
        ),
        extract_data   = X_test,
        aspects        = ["weights", "activations"],
        epochs         = 50,
        snapshot_every = 5,
    )

Layer selection
---------------
By default, :func:`extract_model` and :func:`extract_training` inspect the
model architecture and choose layers automatically.  For models with **both**
linear and convolutional layers an interactive prompt is shown (suppressed by
``clarify=False``).  You can also specify layers explicitly:

.. code-block:: python

    extract_model(model, X, layer_selection="all_linear")
    extract_model(model, X, layer_selection="all_conv")
    extract_model(model, X, layer_selection=["fc1", "fc2"])
    extract_model(model, X, layer_selection=[0, 2, 4])

Supplying a trained model
-------------------------
If you already have a trained model, call :func:`extract_model` — no training
needed.  If you want to inspect the architecture first, call :func:`inspect`:

.. code-block:: python

    from tanc.model_extractor import inspect
    info = inspect(my_model)   # prints summary table, returns ModelInfo
"""

from __future__ import annotations

import warnings
from typing import Any

from tanc.model_extractor._snapshot import ModelSnapshot, TrainingView
from tanc.model_extractor._inspector import (
    detect_framework,
    inspect_model,
    resolve_layer_selection,
    ModelInfo,
    LayerInfo,
)
from tanc.model_extractor.extractors import (
    ModelExtractor,
    TrainingExtractor,
)

# ── Valid aspect names ─────────────────────────────────────────────────────────

_VALID_ASPECTS: frozenset[str] = frozenset({"weights", "activations", "classifications"})
_DEFAULT_ASPECTS: list[str] = ["weights", "activations", "classifications"]


def _validate_aspects(aspects: list[str] | None) -> list[str]:
    if aspects is None:
        return list(_DEFAULT_ASPECTS)
    bad = [a for a in aspects if a not in _VALID_ASPECTS]
    if bad:
        raise ValueError(
            f"Unknown aspect(s): {bad}. "
            f"Valid values: {sorted(_VALID_ASPECTS)}"
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
# inspect — architecture summary
# ─────────────────────────────────────────────────────────────────────────────

def inspect(model: Any) -> ModelInfo:
    """Inspect a model's architecture and print a formatted summary table.

    Useful for discovering layer names and indices before calling
    :func:`extract_model` or :func:`extract_training` with an explicit
    ``layer_selection``.

    Parameters
    ----------
    model
        Any supported model (``torch.nn.Module`` or ``tf.keras.Model``).

    Returns
    -------
    ModelInfo
        Contains layer names, types, shapes, and parameter counts.
        Prints the summary table as a side effect.

    Examples
    --------
    >>> info = inspect(my_model)
    >>> linear_names = [l.name for l in info.linear_layers]
    """
    info = inspect_model(model)
    print(info.summary())
    return info


# ─────────────────────────────────────────────────────────────────────────────
# extract_model — final view
# ─────────────────────────────────────────────────────────────────────────────

def extract_model(
    model: Any,
    data: Any,
    aspects: list[str] | None = None,
    layer_selection: str | list[str] | list[int] | None = None,
    clarify: bool = True,
    device: str | None = None,
    activation_pooling: str = "flatten",
    soro_effective_weight: bool = True,
) -> ModelSnapshot:
    """Extract model data from a pre-trained model (*final view*).

    Inspects the model, resolves which layers to hook (prompting interactively
    if the model has mixed layer types and ``clarify=True``), then runs a
    single forward pass to collect the requested aspects.

    Parameters
    ----------
    model
        A trained model.  Supported types:

        * ``torch.nn.Module``
        * ``tf.keras.Model``

    data : array-like
        Representative input data for activation / classification extraction.
        Should cover the range of inputs you care about (e.g. a test set).
        Not required to compute weights — but a forward pass is always run
        if ``"activations"`` or ``"classifications"`` is in ``aspects``.

    aspects : list[str] or None
        Which aspects to extract.  Any subset of::

            ["weights", "activations", "classifications"]

        ``None`` → all three.

    layer_selection : str, list[str], list[int], or None
        Which layers to hook for weights and activations.

        * ``None`` — auto-detect.  Pure MLP → all linear layers.
          Pure CNN → all conv layers.  Mixed → interactive prompt
          (if ``clarify=True``), else linear layers with a warning.
        * ``"all_linear"`` — all Linear / Dense layers.
        * ``"all_conv"``   — all convolutional layers.
        * ``"linear_and_conv"`` — both.
        * ``"all"``        — every parameterized layer.
        * ``list[str]``    — explicit layer names, e.g. ``["fc1", "fc2"]``.
        * ``list[int]``    — explicit layer indices, e.g. ``[0, 2, 4]``.

    clarify : bool
        When ``True`` (default) and ``layer_selection=None`` produces an
        ambiguous result (mixed layer types), the user is prompted
        interactively via ``input()``.  Set to ``False`` for non-interactive
        usage or to suppress the prompt.

    device : str or None
        PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
        ``None`` → auto-detect.  Ignored for TensorFlow models.

    activation_pooling : str
        How to collapse each captured activation to ``(N_samples, features)``.
        ``"flatten"`` (default) reshapes ``(N, …) → (N, -1)``.  For transformer
        ``(batch, seq, hidden)`` activations, ``"last"`` / ``"first"`` /
        ``"mean"`` / ``"max"`` pool over the sequence axis — e.g. ``"last"``
        takes the final-token summary of a causal decoder.

    soro_effective_weight : bool
        For Sum-of-Rank-One (SoRO) layers, also store the assembled effective
        weight in ``weights`` (default ``True``) so they feed weight-based tools
        like FC layers; the trained factors are always kept in
        ``snapshot.soro_factors``.

    Returns
    -------
    ModelSnapshot
        All requested aspects stored as numpy arrays.  Key accessors::

            snapshot.weight_matrices()          # list[ndarray] → build_weight_graph
            snapshot.all_activation_matrices()  # list[ndarray] → build_activation_graph
            snapshot.classifications            # ndarray (N, classes)
            snapshot.predicted_labels           # ndarray (N,)

    Examples
    --------
    Weights only (MLP):

    >>> snapshot = extract_model(mlp, X_test, aspects=["weights"])
    >>> bundle   = build_weight_graph(snapshot.weight_matrices())

    All aspects, specific layers (CNN):

    >>> snapshot = extract_model(
    ...     cnn, X_test,
    ...     aspects        = ["activations", "classifications"],
    ...     layer_selection = "all_conv",
    ... )
    >>> acts = snapshot.all_activation_matrices()

    Specify layers by name:

    >>> snapshot = extract_model(model, X_test, layer_selection=["fc1", "fc2"])
    """
    # Thin wrapper around the class form.  The class is what the rest of
    # the package and the docs lead with; this function stays as the
    # one-call convenience shortcut.
    extractor = ModelExtractor(
        model=model,
        aspects=aspects,
        layer_selection=layer_selection,
        clarify=clarify,
        device=device,
        activation_pooling=activation_pooling,
        soro_effective_weight=soro_effective_weight,
    )
    return extractor.extract(data)


# ─────────────────────────────────────────────────────────────────────────────
# extract_training — training view
# ─────────────────────────────────────────────────────────────────────────────

def extract_training(
    model: Any,
    extract_data: Any,
    aspects: list[str] | None = None,
    layer_selection: str | list[str] | list[int] | None = None,
    clarify: bool = True,
    # ── Cross-framework data (preferred) ──────────────────────────────────────
    train_data: Any = None,
    val_data: Any = None,
    batch_size: int = 128,
    shuffle: bool = True,
    # ── PyTorch-specific ──────────────────────────────────────────────────────
    train_loader: Any = None,
    criterion: Any = None,
    optimizer: Any = None,
    val_loader: Any = None,
    scheduler: Any = None,
    loss_eval_data: Any = None,
    # ── TensorFlow-specific ───────────────────────────────────────────────────
    compile_kwargs: dict | None = None,
    # ── Shared training params ────────────────────────────────────────────────
    epochs: int = 100,
    target_accuracy: float | None = 0.98,
    snapshot_every: int = 1,
    snapshot_schedule: str = "epoch",
    device: str | None = None,
    verbose: bool = True,
) -> TrainingView:
    """Train a model and extract :class:`ModelSnapshot` objects at regular
    intervals (*training view*).

    Parameters
    ----------
    model
        An untrained (or partially trained) model.
    extract_data : array-like
        Fixed representative data used for *every* snapshot's forward pass.
        Must be in the same format as the model's expected input.

    aspects : list[str] or None
        Aspects to capture at each snapshot.  ``None`` → all three.

    layer_selection : str, list, or None
        Which layers to hook.  Same options as :func:`extract_model`.

    clarify : bool
        Interactive layer prompt for mixed-type models.

    Cross-framework data (preferred)
    --------------------------------
    train_data, val_data
        A raw dataset per split — the extractor builds the right loader for
        the detected framework.  Accepts an ``(X, y)`` tuple of arrays/tensors
        (PyTorch wraps it in a ``TensorDataset`` + ``DataLoader``; TensorFlow
        routes it to ``model.fit``), a ``Dataset`` / ``tf.data.Dataset``, or a
        ready ``DataLoader``.  ``val_data`` is your held-out / test set.
    batch_size : int
        Batch size used when building a loader from raw ``(X, y)`` / ``Dataset``
        data (default 128).  Ignored when you pass a ready loader / dataset.
    shuffle : bool
        Shuffle the **training** loader built from raw data (default ``True``).
        The validation loader is never shuffled.

    PyTorch parameters
    ------------------
    train_loader : DataLoader, optional
        Pre-built loader yielding ``(inputs, labels)`` batches.  Use instead of
        ``train_data`` when you want full control; takes precedence if both are
        given.
    criterion : callable
        Loss function, e.g. ``nn.CrossEntropyLoss()``.  **Required** for PyTorch.
    optimizer : Optimizer
        E.g. ``torch.optim.Adam(model.parameters(), lr=1e-3)``.  **Required**.
    val_loader : DataLoader or None
        Pre-built validation loader (alternative to ``val_data``).
    scheduler : torch LR scheduler or None
        Optional learning-rate scheduler (e.g.
        ``torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)``),
        stepped once at the end of each epoch.

    TensorFlow / Keras parameters
    -----------------------------
    compile_kwargs : dict or None
        If the model is not yet compiled, pass
        ``{"optimizer": ..., "loss": ..., "metrics": [...]}`` here.

    Shared training parameters
    --------------------------
    epochs : int
        Maximum number of training epochs (default 100).
    target_accuracy : float or None
        Stop early when validation accuracy meets this threshold.
        Default 0.98.  ``None`` → run for the full ``epochs``.
    snapshot_every : int
        Capture a snapshot every N epochs (default 1 = every epoch).
    snapshot_schedule : str
        ``"epoch"`` (default) | ``"iteration"`` (PyTorch only).
        With ``"iteration"``, ``snapshot_every`` counts mini-batch steps.
    device : str or None
        PyTorch device.  ``None`` → auto-detect.
    verbose : bool
        Print training progress and snapshot info.

    Returns
    -------
    TrainingView
        Contains an ordered list of :class:`ModelSnapshot` objects.

        Key accessors::

            view.weight_trajectory()            # list[list[ndarray]]
            view.activation_trajectory("fc1")   # list[ndarray]
            view.losses()                        # list[float]
            view.accuracies()                    # list[float]
            view.final_snapshot                  # ModelSnapshot

    Examples
    --------
    PyTorch — raw dataset, capture every 5 epochs, stop at 95 % accuracy:

    .. code-block:: python

        view = extract_training(
            model         = model,
            train_data    = (X_train, y_train),    # no loader needed
            val_data      = (X_test,  y_test),
            batch_size    = 128,
            criterion     = nn.CrossEntropyLoss(),
            optimizer     = torch.optim.Adam(model.parameters(), lr=1e-3),
            extract_data  = X_test,
            aspects       = ["weights", "activations"],
            epochs        = 100,
            snapshot_every = 5,
            target_accuracy = 0.95,
        )

        traj   = view.weight_trajectory()
        bundle = build_weight_trajectory(traj, distance="euclidean")

    TensorFlow — every 10 epochs:

    .. code-block:: python

        view = extract_training(
            model        = keras_model,
            train_data   = (X_train, y_train),
            compile_kwargs = dict(
                optimizer = "adam",
                loss      = "sparse_categorical_crossentropy",
                metrics   = ["accuracy"],
            ),
            extract_data   = X_test,
            epochs         = 100,
            snapshot_every = 10,
        )
    """
    # Thin wrapper around TrainingExtractor.  The class auto-detects the
    # framework from ``model`` and validates the right set of required
    # kwargs internally — no branching needed here.
    extractor = TrainingExtractor(
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
        loss_eval_data=loss_eval_data,
        compile_kwargs=compile_kwargs,
        extract_data=extract_data,
        aspects=aspects,
        layer_selection=layer_selection,
        clarify=clarify,
        snapshot_every=snapshot_every,
        snapshot_schedule=snapshot_schedule,
        device=device,
    )
    return extractor.run(
        epochs=epochs,
        target_accuracy=target_accuracy,
        verbose=verbose,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Class-based entry points (preferred)
    "ModelExtractor",
    "TrainingExtractor",
    # One-call function shortcuts
    "extract_model",
    "extract_training",
    "inspect",
    # Container types (re-exported for convenience)
    "ModelSnapshot",
    "TrainingView",
    "ModelInfo",
    "LayerInfo",
]
