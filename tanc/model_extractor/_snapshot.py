"""_snapshot.py — container objects for extracted model data.

ModelSnapshot  — a single point-in-time capture (weights, activations,
                 classifications).  Produced by both the final-view and
                 training-view extractors.

TrainingView   — an ordered collection of ModelSnapshots captured at
                 regular intervals during a training run.  Provides
                 trajectory accessors that feed directly into the
                 graph_builder functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np

from tanc._serialization import SaveLoadMixin


# ─────────────────────────────────────────────────────────────────────────────
# ModelSnapshot
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelSnapshot(SaveLoadMixin):
    """Extracted model data at a single point in time.

    All arrays are numpy ndarrays so they flow directly into the
    graph_builder and topo_tools modules with no extra conversion.

    Attributes
    ----------
    weights : dict[str, ndarray]
        ``{layer_name: weight_matrix}`` where each matrix is 2-D and
        follows the ``(N_in, N_out)`` convention expected by
        :func:`~tanc.graph_builder.build_weight_graph`.

        * **Linear / Dense layers** — stored as ``(in_features, out_features)``.
        * **Conv1d / Conv1D layers** — filters reshaped to
          ``(in_channels * kW, out_channels)``.
        * **Conv2d / Conv2D layers** — filters reshaped to
          ``(in_channels * kH * kW, out_channels)``.

    activations : dict[str, ndarray]
        ``{layer_name: activation_matrix}`` where each matrix is
        ``(N_samples, N_neurons)``, ready for
        :func:`~tanc.graph_builder.build_activation_graph`.

        Conv spatial outputs are flattened: ``(N, C, H, W) → (N, C*H*W)``.

    kernel_weights : dict[str, ndarray]
        ``{layer_name: raw_conv_tensor}`` where each tensor is 4-D
        ``(out_ch, in_ch, kH, kW)``, ready for
        :func:`~tanc.graph_builder.build_kernel_graph`.

        Conv1d weights are promoted from ``(out_ch, in_ch, kW)`` to
        ``(out_ch, in_ch, 1, kW)`` so all conv layers share a uniform
        4-D shape.  Only populated for convolutional layers.

    soro_factors : dict[str, dict[str, ndarray]]
        ``{layer_name: {"U": (out, r), "sigma": (r,), "V": (in, r)}}`` — the
        trained factors of each Sum-of-Rank-One (SoRO) layer.  The assembled
        effective weight ``W = U · diag(sigma) · Vᵀ`` is also stored in
        ``weights`` (as ``(in, out)``) unless disabled, so SoRO layers feed the
        weight-based tools like any FC layer.  See :meth:`effective_weight`.

    attention_weights : dict[str, dict[str, ndarray]]
        ``{layer_name: {"q": …, "k": …, "v": …, "o": …}}`` — the four projection
        matrices of each attention layer, each ``(N_in, N_out)``.  PyTorch stores
        query, key and value stacked together in ``in_proj_weight``; they are
        split here so each behaves like an ordinary fully-connected weight.  The
        same matrices also appear individually in ``weights`` under the names
        ``"<layer>.q_proj"``, ``"<layer>.k_proj"`` and so on, so every
        weight-based tool reaches them without knowing about attention.

    classifications : ndarray or None
        Raw model output — ``(N_samples, N_classes)`` for multi-class or
        ``(N_samples,)`` for binary.
    predicted_labels : ndarray or None
        ``(N_samples,)`` integer class predictions (argmax of logits).
    inputs : ndarray or None
        The input data used for the forward pass.
    epoch : int or None
        Training epoch at capture time.  ``None`` for final-view extractions.
    iteration : int or None
        Global training iteration at capture time.
    loss : float or None
        Training loss at capture time.
    accuracy : float or None
        Validation accuracy at capture time (if a val loader was supplied).
    framework : str
        ``"pytorch"`` | ``"tensorflow"`` | ``"unknown"``
    metadata : dict
        Provenance: model class name, captured layer names, aspect list,
        total parameter count, etc.
    """

    weights: dict[str, np.ndarray] = field(default_factory=dict)
    activations: dict[str, np.ndarray] = field(default_factory=dict)
    kernel_weights: dict[str, np.ndarray] = field(default_factory=dict)
    soro_factors: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    attention_weights: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    classifications: np.ndarray | None = None
    predicted_labels: np.ndarray | None = None
    inputs: np.ndarray | None = None
    epoch: int | None = None
    iteration: int | None = None
    loss: float | None = None
    accuracy: float | None = None            # validation accuracy (if val loader given)
    train_accuracy: float | None = None      # training-set accuracy at capture time
    per_sample_losses: np.ndarray | None = None  # (n_eval,) per-sample loss (Dupuis metric)
    framework: str = "unknown"
    metadata: dict = field(default_factory=dict)

    # ── Accessors for downstream graph_builder functions ─────────────────────

    def weight_matrices(self, layers: list[str] | None = None) -> list[np.ndarray]:
        """Return a list of weight matrices for use with ``build_weight_graph``.

        Parameters
        ----------
        layers : list[str] or None
            Layer names to include, in order.
            ``None`` → all captured weight layers in insertion order.

        Returns
        -------
        list of ``(N_in, N_out)`` ndarrays

        Examples
        --------
        >>> W_list = snapshot.weight_matrices()
        >>> bundle  = build_weight_graph(W_list)
        """
        names = layers if layers is not None else list(self.weights.keys())
        missing = [n for n in names if n not in self.weights]
        if missing:
            raise KeyError(
                f"Layer(s) not found in snapshot weights: {missing}. "
                f"Available: {list(self.weights.keys())}"
            )
        return [self.weights[n] for n in names]

    def kernel_weight(self, layer_name: str) -> np.ndarray:
        """Return the raw 4-D conv weight tensor for use with ``build_kernel_graph``.

        Conv1d weights are promoted to ``(out_ch, in_ch, 1, kW)`` so all
        conv layers share a uniform 4-D shape.

        Parameters
        ----------
        layer_name : str

        Returns
        -------
        ndarray, shape ``(out_ch, in_ch, kH, kW)``

        Examples
        --------
        >>> w = snapshot.kernel_weight("conv1")
        >>> bundle = build_kernel_graph(w, distance="vne")
        """
        if layer_name not in self.kernel_weights:
            raise KeyError(
                f"Layer '{layer_name}' not found in snapshot kernel_weights. "
                f"Available: {list(self.kernel_weights.keys())}"
            )
        return self.kernel_weights[layer_name]

    def soro_factor(self, layer_name: str) -> dict[str, np.ndarray]:
        """Return the SoRO factors ``{"U", "sigma", "V"}`` for one layer.

        ``U`` is ``(out, r)``, ``sigma`` is ``(r,)``, ``V`` is ``(in, r)`` —
        the trained rank-one components, in their ordinal (not magnitude) order.
        """
        if layer_name not in self.soro_factors:
            raise KeyError(
                f"Layer '{layer_name}' not found in snapshot soro_factors. "
                f"Available: {list(self.soro_factors.keys())}"
            )
        return self.soro_factors[layer_name]

    def effective_weight(self, layer_name: str) -> np.ndarray:
        """Return the effective weight of a layer as ``(N_in, N_out)``.

        For a SoRO layer this is ``W = (V · diag(sigma) · Uᵀ)`` — i.e. the
        assembled ``U · diag(sigma) · Vᵀ`` transposed into the ``(in, out)``
        convention used by :meth:`weight_matrices`.  Returns the value already
        cached in ``weights`` when present, else computes it from the factors so
        it is available even if effective-weight storage was disabled.
        """
        if layer_name in self.weights:
            return self.weights[layer_name]
        if layer_name in self.soro_factors:
            f = self.soro_factors[layer_name]
            U, sigma, V = f["U"], np.asarray(f["sigma"]), f["V"]
            return (V * sigma) @ U.T            # (in, r)·(r) then (r, out) → (in, out)
        raise KeyError(
            f"Layer '{layer_name}' has neither a stored weight nor SoRO factors."
        )

    def activation_matrix(self, layer_name: str) -> np.ndarray:
        """Return the ``(N_samples, N_neurons)`` activation array for one layer.

        Parameters
        ----------
        layer_name : str

        Returns
        -------
        ndarray, shape ``(N_samples, N_neurons)``
        """
        if layer_name not in self.activations:
            raise KeyError(
                f"Layer '{layer_name}' not found in snapshot activations. "
                f"Available: {list(self.activations.keys())}"
            )
        return self.activations[layer_name]

    def all_activation_matrices(self) -> list[np.ndarray]:
        """Return all activation matrices in capture order.

        Returns
        -------
        list of ``(N_samples, N_neurons)`` ndarrays — one per captured layer.

        Examples
        --------
        >>> acts = snapshot.all_activation_matrices()
        >>> results = [build_activation_graph(a) for a in acts]
        """
        return list(self.activations.values())

    def coupled_weight_activations(
        self, layers: list[str] | None = None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return ``[(weight, pre_synaptic_activation)]`` pairs.

        Each weight matrix ``W_l`` of shape ``(N_in, N_out)`` is paired with
        the **pre-synaptic** activation — the activation of the layer that
        *feeds into* ``W_l``, shape ``(N_samples, N_in)``.

        For the first weight layer, the pre-synaptic activation is
        ``self.inputs`` (the raw input data).  For subsequent layers it is the
        (post-synaptic) activation of the previous layer.

        This matches the Gebhart et al. (2019) convention:
        ``φ(u, v) = |w_{u→v} · h_u|`` where ``h_u`` is the activation of the
        source neuron.

        Useful for ``build_weight_graph`` with
        ``edge_weight='weighted_activation'`` or
        ``edge_weight='correlation'``.

        Parameters
        ----------
        layers : list[str] or None
            ``None`` → intersection of weight and activation layer names,
            in weight-insertion order.

        Returns
        -------
        list of ``(weight_matrix, pre_synaptic_activation)`` tuples
        """
        if layers is None:
            layers = [n for n in self.weights if n in self.activations]
        missing_w = [n for n in layers if n not in self.weights]
        missing_a = [n for n in layers if n not in self.activations]
        if missing_w:
            raise KeyError(f"Missing weight layers: {missing_w}")
        if missing_a:
            raise KeyError(f"Missing activation layers: {missing_a}")

        # Build ordered list of all activation layer names for index lookup
        all_act_names = list(self.activations.keys())

        result = []
        for n in layers:
            idx = all_act_names.index(n)
            if idx == 0:
                # First layer: pre-synaptic activation is the raw input.
                if self.inputs is None:
                    raise ValueError(
                        f"Pre-synaptic activation for first layer '{n}' "
                        "requires self.inputs, but inputs were not captured."
                    )
                # Flatten image/conv inputs (N, C, H, W) → (N, C*H*W) so they can
                # pair with the first weight matrix (N_in, N_out); a raw 4-D input
                # tensor would otherwise broadcast-fail in the coupled builder.
                pre_act = np.asarray(self.inputs)
                if pre_act.ndim > 2:
                    pre_act = pre_act.reshape(pre_act.shape[0], -1)
            else:
                # Pre-synaptic = output of the previous layer
                pre_act = self.activations[all_act_names[idx - 1]]
                weight = self.weights[n]
                if pre_act.ndim == 2 and weight.ndim == 2:
                    if pre_act.shape[1] != weight.shape[0]:
                        if (
                            weight.shape[0] > 0
                            and pre_act.shape[1] % weight.shape[0] == 0
                        ):
                            warnings.warn(
                                (
                                    f"Inferring last-timestep pre-synaptic activation "
                                    f"for layer '{n}' because activation shape "
                                    f"{pre_act.shape} does not match weight input dim "
                                    f"{weight.shape[0]}."
                                ),
                                UserWarning,
                            )
                            pre_act = pre_act[:, -weight.shape[0]:]
                        else:
                            raise ValueError(
                                (
                                    f"Cannot pair weight layer '{n}' with pre-synaptic "
                                    f"activation of shape {pre_act.shape}. Expected "
                                    f"activation width {weight.shape[0]} to match weight "
                                    "input dimension."
                                )
                            )
            result.append((self.weights[n], pre_act))
        return result

    def layer_names(self, aspect: str = "weights") -> list[str]:
        """Return captured layer names for a given aspect.

        Parameters
        ----------
        aspect : str
            ``"weights"`` | ``"activations"``
        """
        if aspect == "weights":
            return list(self.weights.keys())
        elif aspect == "activations":
            return list(self.activations.keys())
        raise ValueError(f"Unknown aspect '{aspect}'. Valid: 'weights', 'activations'.")

    def __repr__(self) -> str:
        loc = f"epoch={self.epoch}" if self.epoch is not None else "final"
        w = len(self.weights)
        a = len(self.activations)
        c = "yes" if self.classifications is not None else "no"
        return (
            f"ModelSnapshot(framework={self.framework!r}, {loc}, "
            f"weight_layers={w}, activation_layers={a}, "
            f"classifications={c})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TrainingView
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingView(SaveLoadMixin):
    """Ordered collection of ModelSnapshots captured during training.

    Attributes
    ----------
    snapshots : list[ModelSnapshot]
        Snapshots ordered chronologically (earliest first).
    schedule : str
        ``"epoch"`` | ``"iteration"``
    interval : int
        Capture interval — every N epochs or N iterations.
    total_epochs : int
        Total number of epochs actually completed.
    metadata : dict
        Training provenance (framework, target accuracy, device, …).
    """

    snapshots: list[ModelSnapshot]
    schedule: str = "epoch"
    interval: int = 1
    total_epochs: int = 0
    metadata: dict = field(default_factory=dict)

    # ── Single-snapshot accessors ─────────────────────────────────────────────

    @property
    def final_snapshot(self) -> ModelSnapshot:
        """The last captured snapshot."""
        if not self.snapshots:
            raise IndexError("TrainingView contains no snapshots.")
        return self.snapshots[-1]

    def snapshot_at(self, epoch: int) -> ModelSnapshot:
        """Return the snapshot whose epoch is closest to ``epoch``.

        Parameters
        ----------
        epoch : int
        """
        if not self.snapshots:
            raise IndexError("TrainingView contains no snapshots.")
        return min(self.snapshots, key=lambda s: abs((s.epoch or 0) - epoch))

    # ── Trajectory accessors (feed directly into graph_builder) ──────────────

    def weight_trajectory(
        self, layers: list[str] | None = None
    ) -> list[list[np.ndarray]]:
        """Return the weight trajectory as a ``list[list[ndarray]]``.

        Each element corresponds to one snapshot and is a list of
        ``(N_in, N_out)`` weight matrices — one per selected layer.
        This is the format accepted by
        :func:`~tanc.graph_builder.build_weight_trajectory`.

        Parameters
        ----------
        layers : list[str] or None
            Layer names to include.  ``None`` → all captured weight layers.

        Returns
        -------
        list[list[ndarray]], length = ``len(self.snapshots)``

        Examples
        --------
        >>> traj = view.weight_trajectory()
        >>> bundle = build_weight_trajectory(traj, distance="euclidean")
        """
        return [s.weight_matrices(layers) for s in self.snapshots]

    def activation_trajectory(self, layer_name: str) -> list[np.ndarray]:
        """Return activation matrices for one layer across all snapshots.

        Parameters
        ----------
        layer_name : str

        Returns
        -------
        list of ``(N_samples, N_neurons)`` ndarrays, one per snapshot
        that captured this layer.
        """
        return [
            s.activation_matrix(layer_name)
            for s in self.snapshots
            if layer_name in s.activations
        ]

    def per_snapshot_weight_matrices(
        self, layers: list[str] | None = None
    ) -> list[list[np.ndarray]]:
        """Alias for :meth:`weight_trajectory` — explicit name for clarity."""
        return self.weight_trajectory(layers)

    def loss_trajectory(self) -> np.ndarray:
        """Return the loss at each snapshot as a 1-D numpy array.

        Unlike :meth:`losses`, this preserves alignment with
        :meth:`weight_trajectory` by using ``np.nan`` for snapshots
        where the loss was not recorded, rather than dropping them.

        This is the format expected by
        ``build_weight_trajectory(loss_values=...)``.

        Returns
        -------
        ndarray, shape ``(n_snapshots,)``
        """
        return np.array(
            [s.loss if s.loss is not None else np.nan for s in self.snapshots],
            dtype=float,
        )

    def accuracy_trajectory(self) -> np.ndarray:
        """Return the accuracy at each snapshot as a 1-D numpy array.

        Unlike :meth:`accuracies`, this preserves alignment with
        :meth:`weight_trajectory` by using ``np.nan`` for snapshots
        where accuracy was not recorded.

        Returns
        -------
        ndarray, shape ``(n_snapshots,)``
        """
        return np.array(
            [s.accuracy if s.accuracy is not None else np.nan
             for s in self.snapshots],
            dtype=float,
        )

    def train_accuracy_trajectory(self) -> np.ndarray:
        """Training-set accuracy at each snapshot (``np.nan`` where not recorded).

        Recorded automatically by the PyTorch training extractor from the
        predictions made during each epoch's training pass.

        Returns
        -------
        ndarray, shape ``(n_snapshots,)``
        """
        return np.array(
            [s.train_accuracy if s.train_accuracy is not None else np.nan
             for s in self.snapshots],
            dtype=float,
        )

    def per_sample_loss_trajectory(self) -> np.ndarray | None:
        """Return the ``(T, n_eval)`` per-sample loss matrix, or ``None``.

        Populated when the training extractor was given ``loss_eval_data``.
        This is the input to the Dupuis et al. (2023) per-sample loss
        pseudo-metric (``build_weight_trajectory(distance="loss_difference")``).
        """
        if not self.snapshots or self.snapshots[0].per_sample_losses is None:
            return None
        if any(s.per_sample_losses is None for s in self.snapshots):
            return None
        return np.stack([np.asarray(s.per_sample_losses) for s in self.snapshots])

    # ── Scalar series ─────────────────────────────────────────────────────────

    def epochs(self) -> list[int]:
        """Return list of snapshot epoch indices."""
        return [s.epoch for s in self.snapshots if s.epoch is not None]

    def iterations(self) -> list[int]:
        """Return list of snapshot iteration indices."""
        return [s.iteration for s in self.snapshots if s.iteration is not None]

    def losses(self) -> list[float]:
        """Return training loss at each snapshot."""
        return [s.loss for s in self.snapshots if s.loss is not None]

    def accuracies(self) -> list[float]:
        """Return validation accuracy at each snapshot."""
        return [s.accuracy for s in self.snapshots if s.accuracy is not None]

    def __len__(self) -> int:
        return len(self.snapshots)

    def __repr__(self) -> str:
        return (
            f"TrainingView(snapshots={len(self.snapshots)}, "
            f"schedule='{self.schedule}', interval={self.interval}, "
            f"total_epochs={self.total_epochs})"
        )
