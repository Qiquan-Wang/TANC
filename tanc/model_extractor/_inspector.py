"""_inspector.py — framework detection, model architecture inspection,
and layer-selection resolution.

This module is framework-agnostic: it defines the shared LayerInfo /
ModelInfo data structures and provides inspect_pytorch / inspect_tensorflow
implementations, plus the resolve_layer_selection logic that includes
an interactive clarification prompt when the model has mixed layer types.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# SoRO (Sum of Rank One) layers
#
# There is no canonical PyTorch SoRO class, so a module is recognised by a
# small duck-typed *protocol*.  A layer counts as SoRO if it exposes its
# trained SVD-style factors, W = U · diag(sigma) · Vᵀ, via **either**:
#   * a ``soro_factors(self) -> (U, sigma, V)`` method (preferred, explicit), or
#   * attributes ``U``, ``V`` and one of ``sigma`` / ``singular_values`` / ``s``,
# optionally with an ``effective_weight(self)`` method returning the assembled
# matrix.  Shape convention: ``U`` is ``(out, r)``, ``sigma`` is ``(r,)``,
# ``V`` is ``(in, r)``.
# ─────────────────────────────────────────────────────────────────────────────

_SORO_SIGMA_ATTRS = ("sigma", "singular_values", "s")


def soro_factor_tensors(module) -> tuple | None:
    """Return the SoRO ``(U, sigma, V)`` factors of *module*, else ``None``.

    Returns the factors as-is (framework tensors) — no numpy/torch conversion,
    so this stays framework-neutral and import-free.
    """
    getter = getattr(module, "soro_factors", None)
    if callable(getter):
        res = getter()
        if res is not None and len(res) == 3:
            return tuple(res)
    if hasattr(module, "U") and hasattr(module, "V"):
        for s_attr in _SORO_SIGMA_ATTRS:
            sig = getattr(module, s_attr, None)
            if sig is not None:
                return (module.U, sig, module.V)
    return None


def soro_effective_tensor(module):
    """Return ``module.effective_weight()`` if the module provides it, else ``None``."""
    getter = getattr(module, "effective_weight", None)
    return getter() if callable(getter) else None


def is_soro_module(module) -> bool:
    """True if *module* exposes the SoRO factor / effective-weight protocol."""
    return (soro_factor_tensors(module) is not None
            or callable(getattr(module, "effective_weight", None)))


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerInfo:
    """Metadata about a single layer / module."""

    name: str
    """Dotted name as returned by ``named_modules()`` or the Keras layer name."""

    layer_type: str
    """Semantic category: ``"linear"`` | ``"conv"`` | ``"norm"`` |
    ``"embedding"`` | ``"attention"`` | ``"recurrent"`` | ``"soro"`` |
    ``"other"``."""

    class_name: str
    """Python class name, e.g. ``"Linear"``, ``"Conv1d"``, ``"Conv2d"``."""

    in_features: int | None
    """Number of input features / channels (``None`` if not inferrable)."""

    out_features: int | None
    """Number of output features / channels."""

    param_count: int
    """Total trainable parameter count for this layer."""

    weight_shape: tuple | None
    """Shape of the primary weight tensor (before any reshaping)."""

    has_weights: bool
    """True if the layer has at least one trainable weight tensor."""

    index: int
    """Position in the model's layer list (0-based)."""


@dataclass
class ModelInfo:
    """Architecture summary returned by :func:`inspect_model`."""

    framework: str
    """``"pytorch"`` | ``"tensorflow"``"""

    model_class: str
    """Python class name of the top-level model."""

    layers: list[LayerInfo]
    """All leaf layers / modules (containers excluded)."""

    total_params: int
    """Total trainable parameter count for the whole model."""

    # ── Filtered views ────────────────────────────────────────────────────────

    @property
    def linear_layers(self) -> list[LayerInfo]:
        """Layers of type ``"linear"`` (Linear, Dense, Bilinear, …)."""
        return [l for l in self.layers if l.layer_type == "linear"]

    @property
    def conv_layers(self) -> list[LayerInfo]:
        """Layers of type ``"conv"`` (Conv1d/2d/3d, separable, transposed, …)."""
        return [l for l in self.layers if l.layer_type == "conv"]

    @property
    def soro_layers(self) -> list[LayerInfo]:
        """Layers of type ``"soro"`` (Sum-of-Rank-One factored FC layers)."""
        return [l for l in self.layers if l.layer_type == "soro"]

    @property
    def parameterized_layers(self) -> list[LayerInfo]:
        """All layers that have at least one weight tensor."""
        return [l for l in self.layers if l.has_weights]

    @property
    def has_mixed_types(self) -> bool:
        """True if the model has both linear and conv layers."""
        types = {
            l.layer_type for l in self.parameterized_layers
            if l.layer_type in ("linear", "conv")
        }
        return len(types) > 1

    # ── Human-readable summary ────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a formatted table of all parameterized layers."""
        header = (
            f"Model : {self.model_class}  [{self.framework}]\n"
            f"Params: {self.total_params:,}\n\n"
            f"{'Idx':<5} {'Name':<35} {'Type':<16} {'Class':<22} {'Weight shape'}"
        )
        sep = "-" * 100
        rows = []
        for l in self.layers:
            if not l.has_weights:
                continue
            ws = str(l.weight_shape) if l.weight_shape else "—"
            rows.append(
                f"{l.index:<5} {l.name:<35} {l.layer_type:<16} "
                f"{l.class_name:<22} {ws}"
            )
        return "\n".join([header, sep] + rows)


# ─────────────────────────────────────────────────────────────────────────────
# Framework detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_framework(model: Any) -> str:
    """Identify which deep-learning framework a model belongs to.

    Returns
    -------
    ``"pytorch"`` | ``"tensorflow"`` | ``"unknown"``
    """
    module_path = type(model).__module__ or ""

    if "torch" in module_path:
        return "pytorch"
    if "tensorflow" in module_path or "keras" in module_path:
        return "tensorflow"

    # Fallback: isinstance checks (handles subclasses defined in user code)
    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            return "pytorch"
    except ImportError:
        pass

    try:
        import tensorflow as tf
        if isinstance(model, tf.Module):
            return "tensorflow"
    except ImportError:
        pass

    try:
        import keras
        if isinstance(model, keras.Model):
            return "tensorflow"
    except ImportError:
        pass

    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_pytorch(model) -> ModelInfo:
    """Inspect a ``torch.nn.Module`` and return a :class:`ModelInfo`."""
    try:
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "PyTorch is required to inspect PyTorch models. "
            "Install with: pip install torch"
        )

    # Build type → category mapping dynamically so it works across torch versions
    type_map: dict[type, str] = {}

    for cls_name, category in [
        ("Linear", "linear"),
        ("Bilinear", "linear"),
        ("LazyLinear", "linear"),
        ("Conv1d", "conv"), ("Conv2d", "conv"), ("Conv3d", "conv"),
        ("ConvTranspose1d", "conv"), ("ConvTranspose2d", "conv"),
        ("ConvTranspose3d", "conv"),
        ("LazyConv1d", "conv"), ("LazyConv2d", "conv"), ("LazyConv3d", "conv"),
        ("BatchNorm1d", "norm"), ("BatchNorm2d", "norm"), ("BatchNorm3d", "norm"),
        ("LayerNorm", "norm"), ("GroupNorm", "norm"),
        ("InstanceNorm1d", "norm"), ("InstanceNorm2d", "norm"),
        ("InstanceNorm3d", "norm"),
        ("Embedding", "embedding"), ("EmbeddingBag", "embedding"),
        ("MultiheadAttention", "attention"),
        ("RNN", "recurrent"), ("LSTM", "recurrent"), ("GRU", "recurrent"),
    ]:
        cls = getattr(nn, cls_name, None)
        if cls is not None:
            type_map[cls] = category

    layers: list[LayerInfo] = []
    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    soro_prefixes: list[str] = []
    unit_prefixes: list[str] = []
    mha_cls = getattr(nn, "MultiheadAttention", ())
    for idx, (name, module) in enumerate(model.named_modules()):
        if name == "":
            continue  # skip the root module
        # Skip anything nested inside a SoRO layer — it is captured as one unit.
        if any(name.startswith(p) for p in soro_prefixes):
            continue
        # Same for attention: MultiheadAttention owns an ``out_proj`` child, so
        # without this it would be dropped by the container guard below while its
        # child reappeared as an anonymous Linear.  Capture it as one layer and
        # suppress the child.
        if any(name.startswith(p) for p in unit_prefixes):
            continue
        if mha_cls and isinstance(module, mha_cls):
            unit_prefixes.append(name + ".")
            embed = getattr(module, "embed_dim", None)
            layers.append(LayerInfo(
                name=name,
                layer_type="attention",
                class_name=type(module).__name__,
                in_features=embed,
                out_features=embed,
                param_count=sum(p.numel() for p in module.parameters()
                                if p.requires_grad),
                weight_shape=(tuple(module.in_proj_weight.shape)
                              if getattr(module, "in_proj_weight", None) is not None
                              else None),
                has_weights=True,
                index=idx,
            ))
            continue

        # SoRO layers are treated as leaves even though they may hold child
        # modules (e.g. sub-Linears for U / V): capture the whole factored layer.
        if is_soro_module(module):
            soro_prefixes.append(name + ".")
            facs = soro_factor_tensors(module)
            eff = soro_effective_tensor(module)
            out_feat = in_feat = None
            weight_shape = None
            if facs is not None:                     # U (out, r), V (in, r)
                out_feat = int(facs[0].shape[0])
                in_feat = int(facs[2].shape[0])
                weight_shape = (out_feat, in_feat)
            elif eff is not None:                    # effective W (out, in)
                weight_shape = tuple(int(s) for s in eff.shape)
                out_feat, in_feat = weight_shape[0], weight_shape[1]
            layers.append(LayerInfo(
                name=name,
                layer_type="soro",
                class_name=type(module).__name__,
                in_features=in_feat,
                out_features=out_feat,
                param_count=sum(p.numel() for p in module.parameters()
                                if p.requires_grad),
                weight_shape=weight_shape,
                has_weights=True,
                index=idx,
            ))
            continue

        # Skip container modules (Sequential, ModuleList, etc.)
        if list(module.children()):
            continue

        layer_type = "other"
        for cls, cat in type_map.items():
            if isinstance(module, cls):
                layer_type = cat
                break

        weight_attr = getattr(module, "weight", None)
        has_weights = weight_attr is not None
        weight_shape = tuple(weight_attr.shape) if has_weights else None
        param_count = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )

        in_feat = getattr(module, "in_features", None)
        out_feat = getattr(module, "out_features", None)
        # Infer from weight shape when not directly available
        if weight_shape is not None and len(weight_shape) >= 2:
            if out_feat is None:
                out_feat = weight_shape[0]
            if in_feat is None:
                in_feat = weight_shape[1]

        layers.append(LayerInfo(
            name=name,
            layer_type=layer_type,
            class_name=type(module).__name__,
            in_features=in_feat,
            out_features=out_feat,
            param_count=param_count,
            weight_shape=weight_shape,
            has_weights=has_weights,
            index=idx,
        ))

    return ModelInfo(
        framework="pytorch",
        model_class=type(model).__name__,
        layers=layers,
        total_params=total_params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TensorFlow / Keras inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_tensorflow(model) -> ModelInfo:
    """Inspect a ``tf.keras.Model`` and return a :class:`ModelInfo`."""
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        raise ImportError(
            "TensorFlow is required to inspect Keras models. "
            "Install with: pip install tensorflow"
        )

    TF_TYPE_MAP: dict[str, str] = {
        "Dense": "linear",
        "Conv1D": "conv", "Conv2D": "conv", "Conv3D": "conv",
        "Conv1DTranspose": "conv", "Conv2DTranspose": "conv", "Conv3DTranspose": "conv",
        "SeparableConv1D": "conv", "SeparableConv2D": "conv",
        "DepthwiseConv1D": "conv", "DepthwiseConv2D": "conv",
        "BatchNormalization": "norm", "LayerNormalization": "norm",
        "GroupNormalization": "norm",
        "Embedding": "embedding",
        "MultiHeadAttention": "attention",
        "LSTM": "recurrent", "GRU": "recurrent", "SimpleRNN": "recurrent",
        "Bidirectional": "recurrent",
    }

    layers: list[LayerInfo] = []

    try:
        total_params = model.count_params()
    except Exception:
        total_params = sum(
            w.numpy().size for w in model.trainable_weights
        )

    for idx, layer in enumerate(model.layers):
        class_name = type(layer).__name__

        # SoRO layers (duck-typed) are captured as one factored FC unit.
        if is_soro_module(layer):
            facs = soro_factor_tensors(layer)
            eff = soro_effective_tensor(layer)
            out_feat = in_feat = None
            weight_shape = None
            if facs is not None:                     # U (out, r), V (in, r)
                out_feat = int(facs[0].shape[0])
                in_feat = int(facs[2].shape[0])
                weight_shape = (out_feat, in_feat)
            elif eff is not None:                    # effective W (out, in)
                weight_shape = tuple(int(s) for s in eff.shape)
                out_feat, in_feat = weight_shape[0], weight_shape[1]
            layers.append(LayerInfo(
                name=layer.name,
                layer_type="soro",
                class_name=class_name,
                in_features=in_feat,
                out_features=out_feat,
                param_count=sum(int(w.size) for w in layer.get_weights()),
                weight_shape=weight_shape,
                has_weights=True,
                index=idx,
            ))
            continue

        layer_type = TF_TYPE_MAP.get(class_name, "other")

        keras_weights = layer.get_weights()
        has_weights = len(keras_weights) > 0
        primary_weight = keras_weights[0] if has_weights else None
        weight_shape = tuple(primary_weight.shape) if primary_weight is not None else None
        param_count = sum(w.size for w in keras_weights)

        in_feat = None
        out_feat = None
        if hasattr(layer, "units"):
            out_feat = layer.units
        if weight_shape is not None and len(weight_shape) >= 2:
            if in_feat is None:
                in_feat = weight_shape[0]
            if out_feat is None:
                out_feat = weight_shape[-1]

        layers.append(LayerInfo(
            name=layer.name,
            layer_type=layer_type,
            class_name=class_name,
            in_features=in_feat,
            out_features=out_feat,
            param_count=param_count,
            weight_shape=weight_shape,
            has_weights=has_weights,
            index=idx,
        ))

    return ModelInfo(
        framework="tensorflow",
        model_class=type(model).__name__,
        layers=layers,
        total_params=total_params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unified inspection entry point
# ─────────────────────────────────────────────────────────────────────────────

def inspect_model(model: Any) -> ModelInfo:
    """Inspect any supported model and return a :class:`ModelInfo`.

    Dispatches to the framework-specific implementation based on
    :func:`detect_framework`.
    """
    framework = detect_framework(model)
    if framework == "pytorch":
        return inspect_pytorch(model)
    elif framework == "tensorflow":
        return inspect_tensorflow(model)
    raise ValueError(
        f"Cannot inspect model of type '{type(model).__name__}'. "
        "Supported frameworks: PyTorch (torch.nn.Module), "
        "TensorFlow/Keras (tf.keras.Model)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Layer selection resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_layer_selection(
    model_info: ModelInfo,
    aspects: list[str],
    layer_selection: str | list[str] | list[int] | None,
    clarify: bool = True,
) -> list[str]:
    """Resolve ``layer_selection`` into a concrete list of layer names.

    Parameters
    ----------
    model_info : ModelInfo
        Architecture summary from :func:`inspect_model`.
    aspects : list[str]
        The extraction aspects requested.  Only relevant when
        ``aspects`` contains ``"weights"`` or ``"activations"`` — if
        neither is requested, returns an empty list immediately.
    layer_selection : str, list, or None
        Controls which layers are selected:

        * ``None`` — auto-detect.  For pure MLP: all linear layers.
          For pure CNN: all conv layers.  For mixed (CNN + classifier
          head): prompt interactively if ``clarify=True``, else default
          to linear with a warning.
        * ``"all_linear"`` — all Linear / Dense layers.
        * ``"all_conv"``   — all convolutional layers.
        * ``"linear_and_conv"`` — both linear and conv layers.
        * ``"all"``        — every parameterized layer.
        * ``list[str]``    — explicit layer names.
        * ``list[int]``    — explicit layer indices.

    clarify : bool
        When ``True`` and auto-detection is ambiguous, use ``input()``
        to ask the user which layers to hook.  When ``False``, fall back
        silently with a warning.

    Returns
    -------
    list[str]
        Ordered list of layer names to extract.
    """
    needs_layers = any(a in aspects for a in ("weights", "activations"))
    if not needs_layers:
        return []

    parameterized = model_info.parameterized_layers
    if not parameterized:
        warnings.warn(
            "No parameterized layers found in model — nothing to extract.",
            UserWarning,
            stacklevel=3,
        )
        return []

    # ── Explicit list of names ────────────────────────────────────────────────
    if (isinstance(layer_selection, list)
            and layer_selection
            and isinstance(layer_selection[0], str)):
        available = {l.name for l in model_info.layers}
        bad = [n for n in layer_selection if n not in available]
        if bad:
            raise ValueError(
                f"Unknown layer name(s): {bad}\n\n"
                f"{model_info.summary()}"
            )
        return list(layer_selection)

    # ── Explicit list of indices ──────────────────────────────────────────────
    if (isinstance(layer_selection, list)
            and layer_selection
            and isinstance(layer_selection[0], int)):
        by_index = {l.index: l for l in model_info.layers}
        bad = [i for i in layer_selection if i not in by_index]
        if bad:
            raise ValueError(
                f"Unknown layer indices: {bad}. "
                f"Valid indices: {sorted(by_index.keys())}"
            )
        return [by_index[i].name for i in layer_selection]

    # ── String presets ────────────────────────────────────────────────────────
    preset_map = {
        "all_linear":       lambda: [l.name for l in model_info.linear_layers],
        "all_conv":         lambda: [l.name for l in model_info.conv_layers],
        "soro":             lambda: [l.name for l in model_info.soro_layers],
        "all_attention":    lambda: [l.name for l in parameterized
                                     if l.layer_type == "attention"],
        "all_embedding":    lambda: [l.name for l in parameterized
                                     if l.layer_type == "embedding"],
        "linear_and_conv":  lambda: [l.name for l in parameterized
                                     if l.layer_type in ("linear", "conv")],
        "all":              lambda: [l.name for l in parameterized],
    }
    if layer_selection in preset_map:
        names = preset_map[layer_selection]()
    elif layer_selection is None:
        names = _auto_select(model_info, clarify)
    else:
        raise ValueError(
            f"Unknown layer_selection='{layer_selection}'. "
            "Valid string presets: 'all_linear', 'all_conv', 'soro', "
            "'all_attention', 'all_embedding', 'linear_and_conv', 'all'. "
            "Or pass a list of layer names (str) or indices (int)."
        )

    if not names:
        warnings.warn(
            "Layer selection resolved to an empty list — nothing will be extracted.",
            UserWarning,
            stacklevel=3,
        )
    return names


# ─────────────────────────────────────────────────────────────────────────────
# Auto-selection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _auto_select(model_info: ModelInfo, clarify: bool) -> list[str]:
    """Choose layers automatically when layer_selection=None."""
    # SoRO layers are fully-connected in spirit, so group them with linear.
    fc_layers = model_info.linear_layers + model_info.soro_layers
    has_fc = bool(fc_layers)
    has_conv = bool(model_info.conv_layers)

    if has_fc and not has_conv:
        return [l.name for l in fc_layers]

    if has_conv and not has_fc:
        return [l.name for l in model_info.conv_layers]

    if has_conv and has_fc:
        if clarify:
            return _prompt_user(model_info)
        else:
            warnings.warn(
                f"Model '{model_info.model_class}' has both fully-connected "
                "(linear / SoRO) and conv layers. Defaulting to the "
                "fully-connected layers. Pass layer_selection='all_linear', "
                "'all_conv', 'soro', 'linear_and_conv', 'all', or a list of "
                "names/indices to suppress this warning.",
                UserWarning,
                stacklevel=4,
            )
            return [l.name for l in fc_layers]

    # Only norm, embedding, etc. — return all parameterized
    return [l.name for l in model_info.parameterized_layers]


def _prompt_user(model_info: ModelInfo) -> list[str]:
    """Interactive prompt: ask the user which layers to hook."""
    print()
    print("=" * 72)
    print("  Layer Selection Required")
    print("=" * 72)
    print(
        f"\n  Your model '{model_info.model_class}' contains both "
        "linear and convolutional layers.\n"
    )
    print(model_info.summary())
    print()
    print("  Options:")
    print("    1  all_linear       — fully-connected layers only")
    print("    2  all_conv         — convolutional layers only")
    print("    3  linear_and_conv  — both linear and conv layers")
    print("    4  all              — all parameterized layers")
    print("    5  custom           — enter layer names or indices")
    print()

    by_index = {l.index: l for l in model_info.layers}
    available_names = {l.name for l in model_info.layers}

    while True:
        try:
            raw = input("  Choice [1-5 or layer names/indices, comma-separated]: ").strip()
        except EOFError:
            warnings.warn(
                "Non-interactive environment: defaulting to linear layers.",
                UserWarning,
                stacklevel=5,
            )
            return [l.name for l in model_info.linear_layers]

        preset_shortcuts = {
            "1": "all_linear",
            "2": "all_conv",
            "3": "linear_and_conv",
            "4": "all",
            "all_linear": "all_linear",
            "all_conv": "all_conv",
            "linear_and_conv": "linear_and_conv",
            "all": "all",
        }
        if raw.lower() in preset_shortcuts:
            key = preset_shortcuts[raw.lower()]
            mapping = {
                "all_linear":      lambda: [l.name for l in model_info.linear_layers],
                "all_conv":        lambda: [l.name for l in model_info.conv_layers],
                "linear_and_conv": lambda: [l.name for l in model_info.parameterized_layers
                                            if l.layer_type in ("linear", "conv")],
                "all":             lambda: [l.name for l in model_info.parameterized_layers],
            }
            return mapping[key]()

        if raw == "5" or raw.lower() == "custom":
            try:
                raw2 = input("  Enter layer names or indices (comma-separated): ").strip()
            except EOFError:
                return [l.name for l in model_info.linear_layers]
            raw = raw2

        # Try to parse as indices or names
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if not parts:
            print("  Please enter a valid choice.")
            continue

        # All integers → indices
        try:
            indices = [int(p) for p in parts]
            bad = [i for i in indices if i not in by_index]
            if bad:
                print(f"  Unknown indices: {bad}. Try again.")
                continue
            return [by_index[i].name for i in indices]
        except ValueError:
            pass

        # Names
        bad = [n for n in parts if n not in available_names]
        if bad:
            print(f"  Unknown layer names: {bad}. Try again.")
            continue
        return parts
