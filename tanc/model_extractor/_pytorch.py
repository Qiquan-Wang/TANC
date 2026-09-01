"""_pytorch.py — PyTorch-specific weight/activation extraction and training loop.

Two main entry points:
    extract_from_pytorch  — final-view extraction from a trained model.
    train_and_extract_pytorch — training-view extraction with hook-based snapshots.

Weight convention
-----------------
All weight matrices are returned in **(N_in, N_out)** form:
  * nn.Linear  : stored as (out, in) → transposed to (in, out).
  * nn.Conv1d  : stored as (out_ch, in_ch, kW) → reshaped to
                 (in_ch * kW, out_ch).
  * nn.Conv2d  : stored as (out_ch, in_ch, kH, kW) → reshaped to
                 (in_ch * kH * kW, out_ch).
  * nn.Conv*   : any higher-dimensional conv follows the same pattern.
This matches what build_weight_graph expects.

Activation convention
---------------------
Activations are captured via forward hooks and flattened to **(N_samples, N_neurons)**:
  * Linear output : already 2-D.
  * Conv1d output : (N, C, L) → (N, C*L).
  * Conv2d output : (N, C, H, W) → (N, C*H*W).
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np

from tanc.model_extractor._snapshot import ModelSnapshot, TrainingView
from tanc.model_extractor._inspector import inspect_pytorch
from tanc.model_extractor._activations import (
    ACTIVATION_POOLINGS, pool_activation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tensor utilities
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy(tensor) -> np.ndarray:
    """Detach a PyTorch tensor and convert to a numpy array."""
    import torch
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _unwrap_tensor(value):
    """Recursively find the first Tensor inside nested tuples/lists."""
    import torch
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _unwrap_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _safe_to_numpy(data) -> np.ndarray | None:
    """Convert input data to numpy, returning None on failure."""
    try:
        import torch
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Weight extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_weights(
    model, layer_names: list[str], soro_effective_weight: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict]]:
    """Extract and reorient weight matrices from a nn.Module.

    Parameters
    ----------
    soro_effective_weight : bool
        For SoRO layers, also store the assembled effective weight
        ``W = U · diag(sigma) · Vᵀ`` in ``weights`` (as ``(N_in, N_out)``) so
        the layer feeds the weight-based tools like any FC layer.  When
        ``False``, only the raw factors are kept (recover ``W`` on demand via
        :meth:`ModelSnapshot.effective_weight`).

    Returns
    -------
    weights : dict[str, ndarray]
        ``{layer_name: (N_in, N_out)}`` — 2-D matrices for
        :func:`~tanc.graph_builder.build_weight_graph`.
    kernel_weights : dict[str, ndarray]
        ``{layer_name: (out_ch, in_ch, kH, kW)}`` — raw conv filter
        tensors for :func:`~tanc.graph_builder.build_kernel_graph`.
        Conv1d weights are promoted to 4-D ``(out_ch, in_ch, 1, kW)``
        so all conv layers share a uniform shape convention.
        Only populated for convolutional layers.
    soro_factors : dict[str, dict[str, ndarray]]
        ``{layer_name: {"U", "sigma", "V"}}`` for each SoRO layer.
    """
    import torch.nn as nn
    from tanc.model_extractor._inspector import (
        is_soro_module, soro_factor_tensors,
    )

    global _CONV_TYPES
    _CONV_TYPES = tuple(
        c for c in (getattr(nn, f"Conv{d}d", None) for d in (1, 2, 3)) if c is not None
    ) + tuple(
        c for c in (getattr(nn, f"ConvTranspose{d}d", None) for d in (1, 2, 3))
        if c is not None
    )

    modules = dict(model.named_modules())
    weights: dict[str, np.ndarray] = {}
    kernel_weights: dict[str, np.ndarray] = {}
    soro_factors: dict[str, dict[str, np.ndarray]] = {}
    attention_weights: dict[str, dict[str, np.ndarray]] = {}

    for name in layer_names:
        if name not in modules:
            warnings.warn(f"Layer '{name}' not found in model; skipping.", UserWarning)
            continue

        module = modules[name]

        # SoRO layer: capture the trained factors (U, sigma, V), and optionally
        # the assembled effective weight so existing weight tools see it as FC.
        if is_soro_module(module):
            facs = soro_factor_tensors(module)
            if facs is None:
                warnings.warn(
                    f"SoRO layer '{name}' exposes no factors; skipping weights.",
                    UserWarning,
                )
                continue
            U = _to_numpy(facs[0]).astype(float)          # (out, r)
            sigma = _to_numpy(facs[1]).astype(float).ravel()   # (r,)
            V = _to_numpy(facs[2]).astype(float)          # (in, r)
            soro_factors[name] = {"U": U, "sigma": sigma, "V": V}
            if soro_effective_weight:
                weights[name] = (V * sigma) @ U.T         # (in, out)
            continue

        # Attention: the four projections live in two parameters, with Q, K and V
        # stacked into ``in_proj_weight`` as (3*embed, embed).  Split them so each
        # projection is an ordinary (N_in, N_out) matrix that every weight-based
        # tool can consume, and record the grouping in ``attention_weights``.
        if _is_mha(module):
            blocks = _attention_blocks(module)
            if not blocks:
                warnings.warn(
                    f"Attention layer '{name}' exposes no in_proj_weight or "
                    f"out_proj; skipping weights.", UserWarning,
                )
                continue
            attention_weights[name] = blocks
            for part, mat in blocks.items():
                weights[f"{name}.{part}_proj"] = mat
            continue

        w_tensor = getattr(module, "weight", None)
        if w_tensor is None:
            # A parameterised layer the extractor has no rule for — recurrent
            # layers keep their weights in weight_ih_l0 / weight_hh_l0, for
            # instance.  Say so rather than dropping it silently: the caller
            # asked for this layer and would otherwise get a result that quietly
            # omits it.
            if any(p.requires_grad for p in module.parameters(recurse=False)):
                warnings.warn(
                    f"Layer '{name}' ({type(module).__name__}) holds parameters but "
                    f"exposes no '.weight', so no weight matrix was extracted for "
                    f"it. Its parameters are "
                    f"{sorted(n for n, _ in module.named_parameters(recurse=False))}. "
                    f"Recurrent layers are not yet supported.",
                    UserWarning,
                )
            continue

        w = _to_numpy(w_tensor)

        if w.ndim < 2:
            # A 1-D parameter is a per-feature scale or gain — LayerNorm and the
            # BatchNorms are the common cases.  It is not a linear map between two
            # spaces, so it has no rows, columns or graph structure, and every
            # construction downstream expects a 2-D matrix.  Skip it rather than
            # emit something nothing can consume.  Its *activations* are still
            # captured normally, and those are meaningful.
            warnings.warn(
                f"Layer '{name}' ({type(module).__name__}) has a 1-D weight of shape "
                f"{w.shape} — a per-feature scale, not a weight matrix — so it was "
                f"skipped for weight analysis. Its activations are unaffected.",
                UserWarning,
            )
            continue

        if w.ndim > 2 and not isinstance(module, _CONV_TYPES):
            # Only conv layers have their >2-D weights reinterpreted as filters.
            # Anything else with a >2-D parameter would otherwise be silently
            # misread as a convolution kernel.
            warnings.warn(
                f"Layer '{name}' ({type(module).__name__}) has a {w.ndim}-D weight of "
                f"shape {w.shape} but is not a recognised convolution, so it was not "
                f"treated as a filter bank. It has been flattened to 2-D instead.",
                UserWarning,
            )
            w = w.reshape(w.shape[0], -1).T
            weights[name] = w.astype(float)
            continue

        if w.ndim > 2:
            # Store raw conv weights for kernel analysis,
            # promoting Conv1d (out_ch, in_ch, kW) → (out_ch, in_ch, 1, kW)
            raw = w
            if raw.ndim == 3:
                raw = raw[:, :, np.newaxis, :]
            kernel_weights[name] = raw.astype(float)

            # Conv: (out_ch, in_ch, *ksize) → (out_ch, in_ch * kH * kW)
            w = w.reshape(w.shape[0], -1)
            # → (in_ch * kH * kW, out_ch) i.e. (N_in, N_out)
            w = w.T
        elif isinstance(module, nn.Linear):
            # Linear: (out, in) → (in, out)
            w = w.T

        weights[name] = w.astype(float)

    return weights, kernel_weights, soro_factors, attention_weights


# Populated on first use by _extract_weights, which imports torch lazily.
_CONV_TYPES: tuple = ()


def _is_mha(module) -> bool:
    """True for ``nn.MultiheadAttention`` (torch imported lazily by the caller)."""
    import torch.nn as nn
    cls = getattr(nn, "MultiheadAttention", None)
    return cls is not None and isinstance(module, cls)


def _attention_blocks(module) -> dict[str, np.ndarray]:
    """Split an attention layer into its four projection matrices.

    PyTorch stores the query, key and value projections stacked together in
    ``in_proj_weight`` with shape ``(3 * embed_dim, embed_dim)``, and the output
    projection separately in ``out_proj.weight``.  Each is returned transposed to
    the toolkit's ``(N_in, N_out)`` convention, so a projection matrix behaves
    exactly like any fully-connected weight.

    When ``_qkv_same_embed_dim`` is False the three projections have their own
    parameters (``q_proj_weight`` and friends) and those are used instead.

    Returns
    -------
    dict[str, ndarray]
        Keys ``"q"``, ``"k"``, ``"v"``, ``"o"`` — omitting any that is absent.
    """
    out: dict[str, np.ndarray] = {}

    packed = getattr(module, "in_proj_weight", None)
    if packed is not None:
        W = _to_numpy(packed)                     # (3*embed, embed)
        if W.shape[0] % 3 == 0:
            e = W.shape[0] // 3
            for i, part in enumerate(("q", "k", "v")):
                out[part] = W[i * e:(i + 1) * e].T.astype(float)
        else:                                      # unexpected packing: keep whole
            out["qkv"] = W.T.astype(float)
    else:                                          # separate q/k/v parameters
        for part in ("q", "k", "v"):
            p = getattr(module, f"{part}_proj_weight", None)
            if p is not None:
                out[part] = _to_numpy(p).T.astype(float)

    o_proj = getattr(module, "out_proj", None)
    o_w = getattr(o_proj, "weight", None) if o_proj is not None else None
    if o_w is not None:
        out["o"] = _to_numpy(o_w).T.astype(float)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Activation capture via forward hooks
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _activation_hooks(model, layer_names: list[str], pooling: str = "flatten"):
    """Context manager that installs forward hooks and accumulates outputs.

    Yields a dict ``{layer_name: list_of_batch_arrays}``.
    After the ``with`` block, the lists contain one array per forward call.
    ``pooling`` is forwarded to :func:`pool_activation` (see it for the
    transformer token-pooling semantics).
    """
    modules = dict(model.named_modules())
    captured: dict[str, list[np.ndarray]] = {n: [] for n in layer_names}
    handles = []

    for name in layer_names:
        if name not in modules:
            warnings.warn(
                f"Layer '{name}' not found for activation hook; skipping.",
                UserWarning,
            )
            continue

        def _make_hook(layer_name: str):
            def hook(module, inp, output):
                tensor = _unwrap_tensor(output)
                if tensor is None:
                    return
                act = pool_activation(_to_numpy(tensor), pooling)
                captured[layer_name].append(act)
            return hook

        handle = modules[name].register_forward_hook(_make_hook(name))
        handles.append(handle)

    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass
# ─────────────────────────────────────────────────────────────────────────────

def _run_forward(
    model,
    data,
    layer_names: list[str],
    aspects: list[str],
    device: str,
    pooling: str = "flatten",
) -> tuple[dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
    """Run a forward pass and collect activations + final output.

    Returns
    -------
    activations : dict[str, (N, neurons) ndarray]
    classifications : (N, classes) ndarray or None
    predicted_labels : (N,) ndarray or None
    """
    import torch

    model.eval()

    needs_acts = "activations" in aspects and bool(layer_names)
    needs_cls  = "classifications" in aspects

    # Nothing here needs a forward pass — a weights-only extraction, for instance.
    # Return before touching ``data``, which is legitimately None in that case.
    if not (needs_acts or needs_cls):
        return {}, None, None
    if data is None:
        raise ValueError(
            "A forward pass is needed for aspects "
            f"{[a for a in aspects if a in ('activations', 'classifications')]}, "
            "but no data was supplied. Pass extract_data=<a fixed batch>, or drop "
            "those aspects to analyse weights alone."
        )

    # Prepare input tensor.  Integer inputs are left as integers: a model whose
    # first layer is an nn.Embedding takes token indices, and casting those to
    # float makes the embedding lookup fail outright.  Everything else is float32.
    if not isinstance(data, torch.Tensor):
        arr = np.asarray(data)
        x = torch.as_tensor(
            arr, dtype=torch.long if np.issubdtype(arr.dtype, np.integer)
            else torch.float32
        )
    elif data.dtype in (torch.long, torch.int, torch.int16, torch.int8, torch.bool):
        x = data
    else:
        x = data.float()
    x = x.to(device)

    activations: dict[str, np.ndarray] = {}
    classifications = None
    predicted_labels = None

    with torch.no_grad():
        if needs_acts:
            with _activation_hooks(model, layer_names, pooling) as captured:
                output = model(x)
            for name, batches in captured.items():
                if batches:
                    activations[name] = np.concatenate(batches, axis=0)
        else:
            output = model(x)

        if needs_cls:
            logits = _to_numpy(output)
            classifications = logits
            if logits.ndim == 2 and logits.shape[1] > 1:
                predicted_labels = np.argmax(logits, axis=1)
            else:
                flat = logits.ravel()
                predicted_labels = (flat > 0.5).astype(np.int64)

    return activations, classifications, predicted_labels


# ─────────────────────────────────────────────────────────────────────────────
# Final-view extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_pytorch(
    model,
    data,
    aspects: list[str],
    layer_names: list[str],
    device: str | None = None,
    epoch: int | None = None,
    iteration: int | None = None,
    loss: float | None = None,
    accuracy: float | None = None,
    activation_pooling: str = "flatten",
    soro_effective_weight: bool = True,
) -> ModelSnapshot:
    """Extract a :class:`~tanc.model_extractor.ModelSnapshot` from a
    PyTorch model.

    Parameters
    ----------
    model : nn.Module
        The model to extract from (may be in any training state).
    data : array-like or torch.Tensor
        Input data for the forward pass (activations + classifications).
        Shape must match the model's expected input.
    aspects : list[str]
        Subset of ``["weights", "activations", "classifications"]``.
    layer_names : list[str]
        Layers to extract weights/activations from.
    device : str or None
        PyTorch device.  ``None`` → auto-detect (CUDA if available, else CPU).
    epoch, iteration, loss, accuracy
        Provenance fields — populated automatically by the training extractor.
    activation_pooling : str
        How to collapse a captured activation to ``(N_samples, features)``.
        ``"flatten"`` (default) reshapes ``(N, …) → (N, -1)``.  For
        transformer-style ``(batch, seq, hidden)`` activations, the token-pool
        modes ``"last"`` / ``"first"`` / ``"mean"`` / ``"max"`` reduce over the
        sequence axis (e.g. ``"last"`` = final-token summary of a causal
        decoder).  See :func:`_pool_activation`.
    soro_effective_weight : bool
        For SoRO layers, also store the assembled effective weight in
        ``weights`` (default) so they feed weight-based tools like FC layers;
        the raw factors are always kept in ``snapshot.soro_factors``.
    """
    import torch

    if activation_pooling not in ACTIVATION_POOLINGS:
        raise ValueError(
            f"activation_pooling={activation_pooling!r} invalid; "
            f"choose one of {ACTIVATION_POOLINGS}."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    weights: dict[str, np.ndarray] = {}
    kernel_weights: dict[str, np.ndarray] = {}
    soro_factors: dict[str, dict[str, np.ndarray]] = {}
    attention_weights: dict[str, dict[str, np.ndarray]] = {}
    if "weights" in aspects and layer_names:
        weights, kernel_weights, soro_factors, attention_weights = _extract_weights(
            model, layer_names, soro_effective_weight=soro_effective_weight
        )

    activations, classifications, predicted_labels = _run_forward(
        model, data, layer_names, aspects, device, activation_pooling
    )

    model_info = inspect_pytorch(model)

    return ModelSnapshot(
        weights=weights,
        activations=activations,
        kernel_weights=kernel_weights,
        soro_factors=soro_factors,
        attention_weights=attention_weights,
        classifications=classifications,
        predicted_labels=predicted_labels,
        inputs=_safe_to_numpy(data),
        epoch=epoch,
        iteration=iteration,
        loss=loss,
        accuracy=accuracy,
        framework="pytorch",
        metadata={
            "model_class": type(model).__name__,
            "captured_layers": layer_names,
            "aspects": aspects,
            "total_params": model_info.total_params,
            "device": device,
            "activation_pooling": activation_pooling,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training-view extraction
# ─────────────────────────────────────────────────────────────────────────────

def train_and_extract_pytorch(
    model,
    train_loader,
    criterion,
    optimizer,
    aspects: list[str],
    layer_names: list[str],
    extract_data,
    epochs: int = 100,
    target_accuracy: float | None = 0.98,
    snapshot_every: int = 1,
    snapshot_schedule: str = "epoch",
    val_loader=None,
    scheduler=None,
    loss_eval_data=None,
    device: str | None = None,
    verbose: bool = True,
) -> TrainingView:
    """Train a PyTorch model and capture :class:`~tanc.model_extractor.ModelSnapshot`
    objects at regular intervals.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
        Yields ``(inputs, labels)`` batches.
    criterion : callable
        Loss function, e.g. ``nn.CrossEntropyLoss()``.
    optimizer : torch.optim.Optimizer
    aspects : list[str]
        Which aspects to capture per snapshot.
    layer_names : list[str]
        Layers to hook for weights / activations.
    extract_data : array-like or Tensor
        Fixed representative data used for every snapshot.  Not shuffled
        between snapshots, so results are comparable across epochs.
    epochs : int
        Maximum training epochs (default 100).
    target_accuracy : float or None
        Early-stop threshold applied to validation accuracy.
        ``None`` → always train for the full ``epochs``.
        Default 0.98.
    snapshot_every : int
        Capture a snapshot every N epochs (``schedule='epoch'``) or
        every N iterations (``schedule='iteration'``).
    snapshot_schedule : str
        ``"epoch"`` | ``"iteration"``
    val_loader : DataLoader or None
        Validation loader for accuracy computation.  If ``None``, accuracy
        fields in snapshots will be ``None``.
    device : str or None
        ``None`` → auto-detect.
    verbose : bool
        Print training progress.

    Returns
    -------
    TrainingView
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    snapshots: list[ModelSnapshot] = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _accuracy(loader) -> float | None:
        if loader is None:
            return None
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                out = model(inputs)
                if out.shape[-1] > 1:
                    preds = out.argmax(dim=-1)
                else:
                    preds = (out.squeeze() > 0.5).long()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

    def _per_sample_losses():
        """Per-sample loss on a fixed eval set (for the Dupuis pseudo-metric)."""
        if loss_eval_data is None:
            return None
        Xe, ye = loss_eval_data
        Xe = Xe.to(device) if torch.is_tensor(Xe) else torch.as_tensor(Xe).to(device)
        ye = ye.to(device) if torch.is_tensor(ye) else torch.as_tensor(ye).to(device)
        model.eval()
        prev = getattr(criterion, "reduction", None)
        try:
            if prev is not None:
                criterion.reduction = "none"
            with torch.no_grad():
                out = model(Xe)
                losses = criterion(out, ye)
                if losses.ndim == 0:            # criterion ignored reduction
                    losses = torch.nn.functional.cross_entropy(out, ye, reduction="none")
                elif losses.ndim > 1:           # e.g. per-element MSE → per-sample
                    losses = losses.reshape(losses.shape[0], -1).mean(dim=1)
            return losses.detach().cpu().numpy()
        finally:
            if prev is not None:
                criterion.reduction = prev

    def _snapshot(epoch_num: int, iter_num: int, loss_val: float | None,
                  train_acc: float | None = None) -> ModelSnapshot:
        acc = _accuracy(val_loader)
        snap = extract_from_pytorch(
            model=model,
            data=extract_data,
            aspects=aspects,
            layer_names=layer_names,
            device=device,
            epoch=epoch_num,
            iteration=iter_num,
            loss=loss_val,
            accuracy=acc,
        )
        snap.train_accuracy = train_acc   # running training accuracy for this epoch
        snap.per_sample_losses = _per_sample_losses()
        if verbose:
            loss_s = f"{loss_val:.4f}" if loss_val is not None else "—"
            acc_s  = f"{acc:.4f}"      if acc  is not None else "—"
            tag = f"iter={iter_num}" if snapshot_schedule == "iteration" else f"epoch={epoch_num}"
            print(f"  [snapshot] {tag}  loss={loss_s}  acc={acc_s}")
        return snap

    # ── Training loop ─────────────────────────────────────────────────────────

    global_iter = 0
    stop_early = False
    last_loss = None
    last_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        train_correct = train_total = 0

        for batch in train_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            global_iter += 1

            # Running training accuracy from the predictions we just made (free).
            with torch.no_grad():
                preds = out.argmax(dim=-1) if out.shape[-1] > 1 else (out.squeeze() > 0.5).long()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
            train_acc = train_correct / train_total if train_total else None

            if snapshot_schedule == "iteration" and global_iter % snapshot_every == 0:
                iter_loss = running_loss / n_batches
                snap = _snapshot(epoch, global_iter, iter_loss, train_acc)
                snapshots.append(snap)
                if (target_accuracy is not None
                        and snap.accuracy is not None
                        and snap.accuracy >= target_accuracy):
                    if verbose:
                        print(f"  Target accuracy {target_accuracy:.2%} reached — stopping.")
                    stop_early = True
                    break

        last_loss = running_loss / max(n_batches, 1)
        last_epoch = epoch

        # Step a per-epoch LR scheduler (e.g. CosineAnnealingLR / StepLR), if given.
        if scheduler is not None:
            scheduler.step()

        if verbose and snapshot_schedule == "epoch":
            print(f"Epoch {epoch}/{epochs}  loss={last_loss:.4f}")

        if snapshot_schedule == "epoch" and epoch % snapshot_every == 0:
            epoch_train_acc = train_correct / train_total if train_total else None
            snap = _snapshot(epoch, global_iter, last_loss, epoch_train_acc)
            snapshots.append(snap)
            if (target_accuracy is not None
                    and snap.accuracy is not None
                    and snap.accuracy >= target_accuracy):
                if verbose:
                    print(f"  Target accuracy {target_accuracy:.2%} reached — stopping.")
                stop_early = True

        if stop_early:
            break

    # Always ensure a final snapshot is present
    if not snapshots or snapshots[-1].epoch != last_epoch:
        snapshots.append(_snapshot(last_epoch, global_iter, last_loss))

    return TrainingView(
        snapshots=snapshots,
        schedule=snapshot_schedule,
        interval=snapshot_every,
        total_epochs=last_epoch,
        metadata={
            "framework": "pytorch",
            "model_class": type(model).__name__,
            "target_accuracy": target_accuracy,
            "device": device,
            "epochs_requested": epochs,
        },
    )
