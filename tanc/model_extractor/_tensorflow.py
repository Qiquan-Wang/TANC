"""_tensorflow.py — TensorFlow / Keras-specific extraction and training.

Two main entry points:
    extract_from_tf          — final-view extraction from a trained Keras model.
    train_and_extract_tf     — training-view extraction using a Keras Callback.

Weight convention
-----------------
All weight matrices are returned in **(N_in, N_out)** form:
  * Dense      : Keras stores ``(in, out)`` → returned as-is.
  * Conv1D     : Keras stores ``(kW, in_ch, out_ch)``
                 → reshaped to ``(kW * in_ch, out_ch)``.
  * Conv2D     : Keras stores ``(kH, kW, in_ch, out_ch)``
                 → reshaped to ``(kH * kW * in_ch, out_ch)``.
This matches what build_weight_graph expects.

Activation convention
---------------------
Activations are collected using a sub-model that outputs the selected
intermediate layers alongside the main model output.  Spatial outputs
(Conv1D, Conv2D, etc.) are flattened to **(N_samples, N_neurons)**.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from tanc.model_extractor._snapshot import ModelSnapshot, TrainingView
from tanc.model_extractor._inspector import inspect_tensorflow
from tanc.model_extractor._activations import (
    ACTIVATION_POOLINGS, pool_activation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Weight extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_weights(
    model, layer_names: list[str], soro_effective_weight: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict]]:
    """Extract and reorient kernel weight matrices from a Keras model.

    Parameters
    ----------
    soro_effective_weight : bool
        For SoRO layers, also store the assembled effective weight
        ``W = U · diag(sigma) · Vᵀ`` in ``weights`` (as ``(N_in, N_out)``)
        so the layer feeds the weight-based tools like any FC layer.

    Returns
    -------
    weights : dict[str, ndarray]
        ``{layer_name: (N_in, N_out)}`` — 2-D matrices for
        :func:`~tanc.graph_builder.build_weight_graph`.
    kernel_weights : dict[str, ndarray]
        ``{layer_name: (out_ch, in_ch, kH, kW)}`` — raw conv filter
        tensors in PyTorch layout for
        :func:`~tanc.graph_builder.build_kernel_graph`.
        Keras Conv1D kernels ``(kW, in_ch, out_ch)`` are transposed
        and promoted to 4-D ``(out_ch, in_ch, 1, kW)``.
        Keras Conv2D kernels ``(kH, kW, in_ch, out_ch)`` are
        transposed to ``(out_ch, in_ch, kH, kW)``.
        Only populated for convolutional layers.
    soro_factors : dict[str, dict[str, ndarray]]
        ``{layer_name: {"U", "sigma", "V"}}`` for each SoRO layer.
    """
    from tanc.model_extractor._inspector import (
        is_soro_module, soro_factor_tensors,
    )

    layer_map = {l.name: l for l in model.layers}
    weights: dict[str, np.ndarray] = {}
    kernel_weights: dict[str, np.ndarray] = {}
    soro_factors: dict[str, dict[str, np.ndarray]] = {}

    for name in layer_names:
        if name not in layer_map:
            warnings.warn(f"Layer '{name}' not found in model; skipping.", UserWarning)
            continue

        layer = layer_map[name]

        # SoRO layer: capture trained factors (+ optional assembled weight).
        if is_soro_module(layer):
            facs = soro_factor_tensors(layer)
            if facs is None:
                warnings.warn(
                    f"SoRO layer '{name}' exposes no factors; skipping weights.",
                    UserWarning,
                )
                continue
            U = np.asarray(facs[0], dtype=float)               # (out, r)
            sigma = np.asarray(facs[1], dtype=float).ravel()   # (r,)
            V = np.asarray(facs[2], dtype=float)               # (in, r)
            soro_factors[name] = {"U": U, "sigma": sigma, "V": V}
            if soro_effective_weight:
                weights[name] = (V * sigma) @ U.T              # (in, out)
            continue

        keras_weights = layer.get_weights()
        if not keras_weights:
            continue

        # First weight is conventionally the kernel
        w = np.asarray(keras_weights[0], dtype=float)

        if w.ndim > 2:
            # Store raw conv weights for kernel analysis.
            # Keras layout → PyTorch layout, with Conv1D promoted to 4-D.
            if w.ndim == 3:
                # Conv1D: (kW, in_ch, out_ch) → (out_ch, in_ch, 1, kW)
                raw = np.transpose(w, (2, 1, 0))[:, :, np.newaxis, :]
            else:
                # Conv2D: (kH, kW, in_ch, out_ch) → (out_ch, in_ch, kH, kW)
                raw = np.transpose(w, (3, 2, 0, 1))
            kernel_weights[name] = raw

            # Conv: (kH, kW, in_ch, out_ch) → (kH*kW*in_ch, out_ch)
            out_ch = w.shape[-1]
            w = w.reshape(-1, out_ch)
        # Dense: (in, out) → already correct
        # (no transpose needed)

        weights[name] = w

    return weights, kernel_weights, soro_factors


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass with intermediate activations
# ─────────────────────────────────────────────────────────────────────────────

def _run_forward(
    model,
    data,
    layer_names: list[str],
    aspects: list[str],
    pooling: str = "flatten",
) -> tuple[dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
    """Run a forward pass, collecting intermediate activations + final output.

    Uses a tf.keras.Model with multiple outputs to capture activations without
    modifying the original model.  Falls back gracefully if the model's output
    tensors are not accessible (e.g. subclassed models without ``call`` tracing).

    Returns
    -------
    activations : dict[str, (N, neurons) ndarray]
    classifications : (N, classes) ndarray or None
    predicted_labels : (N,) ndarray or None
    """
    import tensorflow as tf

    x = np.asarray(data, dtype=np.float32)

    needs_acts = "activations" in aspects and bool(layer_names)
    needs_cls  = "classifications" in aspects

    activations: dict[str, np.ndarray] = {}
    classifications = None
    predicted_labels = None

    # ── Capture intermediate-layer activations ─────────────────────────────────
    if needs_acts:
        want = set(layer_names)
        captured = False
        # A Sequential is a simple layer chain; forward-pass through it and grab the
        # named intermediates directly.  This is Keras-3-safe: there,
        # ``layer.output`` / ``model.input`` are undefined until the model is traced,
        # so ``tf.keras.Model(inputs=model.input, outputs=[layer.output, ...])`` raises
        # "the layer <sequential> has never been called".
        if isinstance(model, tf.keras.Sequential):
            h = tf.convert_to_tensor(x)
            for layer in model.layers:
                h = layer(h, training=False)
                if layer.name in want:
                    act_np = h.numpy() if hasattr(h, "numpy") else np.asarray(h)
                    activations[layer.name] = pool_activation(act_np, pooling).astype(float)
            main_output = h
            captured = True
            for name in want - set(activations):
                warnings.warn(
                    f"Layer '{name}' not found for activation capture; skipping.",
                    UserWarning,
                )
        else:
            # Functional / subclassed model: build a sub-model from layer outputs.
            try:
                layer_map = {l.name: l for l in model.layers}
                intermediate_outputs, valid_names = [], []
                for name in layer_names:
                    if name in layer_map:
                        intermediate_outputs.append(layer_map[name].output)
                        valid_names.append(name)
                    else:
                        warnings.warn(
                            f"Layer '{name}' not found for activation capture; skipping.",
                            UserWarning,
                        )
                if intermediate_outputs:
                    sub_model = tf.keras.Model(
                        inputs=model.input,
                        outputs=intermediate_outputs + [model.output],
                    )
                    results = sub_model(x, training=False)
                    main_output = results[-1]
                    for name, act in zip(valid_names, results[:-1]):
                        act_np = act.numpy() if hasattr(act, "numpy") else np.asarray(act)
                        activations[name] = pool_activation(act_np, pooling).astype(float)
                    captured = True
                else:
                    main_output = model(x, training=False)
                    captured = True
            except Exception as exc:
                warnings.warn(
                    f"Could not build intermediate activation model: {exc}. "
                    "Capturing main output only.",
                    UserWarning,
                )
        if not captured:
            main_output = model(x, training=False)
    else:
        main_output = model(x, training=False)

    # ── Classifications ────────────────────────────────────────────────────────
    if needs_cls:
        logits_np = (
            main_output.numpy()
            if hasattr(main_output, "numpy")
            else np.asarray(main_output)
        )
        classifications = logits_np.astype(float)
        if logits_np.ndim == 2 and logits_np.shape[1] > 1:
            predicted_labels = np.argmax(logits_np, axis=1)
        else:
            flat = logits_np.ravel()
            predicted_labels = (flat > 0.5).astype(np.int64)

    return activations, classifications, predicted_labels


# ─────────────────────────────────────────────────────────────────────────────
# Final-view extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_tf(
    model,
    data,
    aspects: list[str],
    layer_names: list[str],
    epoch: int | None = None,
    iteration: int | None = None,
    loss: float | None = None,
    accuracy: float | None = None,
    activation_pooling: str = "flatten",
    soro_effective_weight: bool = True,
) -> ModelSnapshot:
    """Extract a :class:`~tanc.model_extractor.ModelSnapshot` from a
    Keras model.

    Parameters
    ----------
    model : tf.keras.Model
    data : array-like
        Input data for the forward pass.
    aspects : list[str]
        Subset of ``["weights", "activations", "classifications"]``.
    layer_names : list[str]
        Layer names to extract weights/activations from.
    epoch, iteration, loss, accuracy
        Provenance fields — populated automatically by the training extractor.
    activation_pooling : str
        How to collapse a captured activation to ``(N_samples, features)``;
        see :func:`tanc.model_extractor._activations.pool_activation`.
        ``"flatten"`` (default) matches the historical behaviour; the token-pool
        modes target transformer ``(batch, seq, hidden)`` activations.
    soro_effective_weight : bool
        For SoRO layers, also store the assembled effective weight in
        ``weights`` (default) so they feed weight-based tools like FC layers;
        the trained factors are always kept in ``snapshot.soro_factors``.
    """
    if activation_pooling not in ACTIVATION_POOLINGS:
        raise ValueError(
            f"activation_pooling={activation_pooling!r} invalid; "
            f"choose one of {ACTIVATION_POOLINGS}."
        )

    weights: dict[str, np.ndarray] = {}
    kernel_weights: dict[str, np.ndarray] = {}
    soro_factors: dict[str, dict[str, np.ndarray]] = {}
    if "weights" in aspects and layer_names:
        weights, kernel_weights, soro_factors = _extract_weights(
            model, layer_names, soro_effective_weight=soro_effective_weight
        )

    activations, classifications, predicted_labels = _run_forward(
        model, data, layer_names, aspects, activation_pooling
    )

    model_info = inspect_tensorflow(model)

    return ModelSnapshot(
        weights=weights,
        activations=activations,
        kernel_weights=kernel_weights,
        soro_factors=soro_factors,
        classifications=classifications,
        predicted_labels=predicted_labels,
        inputs=np.asarray(data, dtype=float),
        epoch=epoch,
        iteration=iteration,
        loss=loss,
        accuracy=accuracy,
        framework="tensorflow",
        metadata={
            "model_class": type(model).__name__,
            "captured_layers": layer_names,
            "aspects": aspects,
            "total_params": model_info.total_params,
            "activation_pooling": activation_pooling,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training-view extraction
# ─────────────────────────────────────────────────────────────────────────────

def train_and_extract_tf(
    model,
    train_data,
    aspects: list[str],
    layer_names: list[str],
    extract_data,
    epochs: int = 100,
    target_accuracy: float | None = 0.98,
    snapshot_every: int = 1,
    val_data=None,
    compile_kwargs: dict | None = None,
    batch_size: int | None = None,
    verbose: bool = True,
) -> TrainingView:
    """Train a Keras model and capture :class:`~tanc.model_extractor.ModelSnapshot`
    objects at the end of each selected epoch.

    Parameters
    ----------
    model : tf.keras.Model
    train_data : tf.data.Dataset, (x, y) tuple, or generator
        Training data.
    aspects : list[str]
    layer_names : list[str]
    extract_data : array-like
        Fixed data used for every snapshot forward pass.
    epochs : int
        Maximum training epochs.
    target_accuracy : float or None
        Stop early when the metric tracked by the snapshot callback meets
        this threshold.  The callback checks ``"val_accuracy"``,
        ``"val_acc"``, ``"accuracy"``, ``"acc"`` in that order.
        ``None`` → no early stopping.
    snapshot_every : int
        Capture a snapshot every N epochs (1 = every epoch).
    val_data
        Validation data passed to ``model.fit()``.
    compile_kwargs : dict or None
        If the model is not already compiled, call ``model.compile(**compile_kwargs)``
        before training.  Example::

            compile_kwargs=dict(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
    verbose : bool
        Print snapshot progress (Keras fit verbosity is separate).

    Returns
    -------
    TrainingView
    """
    import tensorflow as tf

    # Compile if needed
    try:
        is_compiled = model._is_compiled
    except AttributeError:
        is_compiled = False

    if compile_kwargs is not None and not is_compiled:
        model.compile(**compile_kwargs)

    snapshots: list[ModelSnapshot] = []

    class _SnapshotCallback(tf.keras.callbacks.Callback):
        """Keras callback that captures a ModelSnapshot at epoch end."""

        def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
            logs = logs or {}
            # Only snapshot at the requested interval
            if (epoch + 1) % snapshot_every != 0:
                return

            loss_val = logs.get("loss")
            # Accept several common accuracy key names
            acc_val = (
                logs.get("val_accuracy")
                or logs.get("val_acc")
                or logs.get("accuracy")
                or logs.get("acc")
            )

            snap = extract_from_tf(
                model=self.model,
                data=extract_data,
                aspects=aspects,
                layer_names=layer_names,
                epoch=epoch + 1,
                loss=float(loss_val) if loss_val is not None else None,
                accuracy=float(acc_val) if acc_val is not None else None,
            )
            snapshots.append(snap)

            if verbose:
                loss_s = f"{loss_val:.4f}" if loss_val is not None else "—"
                acc_s  = f"{acc_val:.4f}"  if acc_val  is not None else "—"
                print(f"  [snapshot] epoch={epoch + 1}  loss={loss_s}  acc={acc_s}")

            # Early stopping via target_accuracy
            if (target_accuracy is not None
                    and acc_val is not None
                    and acc_val >= target_accuracy):
                if verbose:
                    print(f"  Target accuracy {target_accuracy:.2%} reached — stopping.")
                self.model.stop_training = True

    callback = _SnapshotCallback()

    # Keras 3's model.fit treats a positional tuple as a multi-input X.
    # Unpack (x, y) tuples so they're routed to fit(x=..., y=...).
    if isinstance(train_data, tuple) and len(train_data) == 2:
        fit_kwargs = dict(x=train_data[0], y=train_data[1])
    else:
        fit_kwargs = dict(x=train_data)

    # batch_size is ignored by Keras when x is a tf.data.Dataset / generator
    # (those carry their own batching), so only forward it for array inputs.
    if batch_size is not None and "x" in fit_kwargs and "y" in fit_kwargs:
        fit_kwargs["batch_size"] = batch_size

    history = model.fit(
        **fit_kwargs,
        epochs=epochs,
        validation_data=val_data,
        callbacks=[callback],
        verbose=1 if verbose else 0,
    )

    actual_epochs = len(history.history.get("loss", []))

    # Ensure a final snapshot exists (in case last epoch was not a snapshot epoch)
    if not snapshots or snapshots[-1].epoch != actual_epochs:
        h_losses = history.history.get("loss", [None])
        h_accs   = (
            history.history.get("val_accuracy")
            or history.history.get("val_acc")
            or history.history.get("accuracy")
            or history.history.get("acc")
            or [None]
        )
        final_snap = extract_from_tf(
            model=model,
            data=extract_data,
            aspects=aspects,
            layer_names=layer_names,
            epoch=actual_epochs,
            loss=float(h_losses[-1]) if h_losses[-1] is not None else None,
            accuracy=float(h_accs[-1]) if h_accs[-1] is not None else None,
        )
        snapshots.append(final_snap)

    return TrainingView(
        snapshots=snapshots,
        schedule="epoch",
        interval=snapshot_every,
        total_epochs=actual_epochs,
        metadata={
            "framework": "tensorflow",
            "model_class": type(model).__name__,
            "target_accuracy": target_accuracy,
            "epochs_requested": epochs,
        },
    )
