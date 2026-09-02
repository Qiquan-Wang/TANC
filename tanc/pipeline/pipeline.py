"""pipeline.py — TDAPipeline orchestrator.

Wires Module 1 (graph_builder) → Module 2 (topo_tools) → Visualisation
into a single callable object with ``.from_paper()`` presets.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tanc.pipeline.compatibility import check_compatibility
from tanc.pipeline.paper_presets import PAPER_PRESETS
from tanc.pipeline.model_adapter import (
    extract_loss_values,
    is_model_data,
    snapshot_to_builder_data,
)
from tanc.topo_tools._result import TopoResult, TopoResultSet


# Builder name → function mapping
_BUILDERS = {
    "weight_graph":          "tanc.graph_builder.weight_graphs.build_weight_graph",
    "point_cloud_graph":     "tanc.graph_builder.point_cloud_graphs.build_point_cloud_graph",
    "activation_graph":      "tanc.graph_builder.activation_graphs.build_activation_graph",
    "kernel_graph":          "tanc.graph_builder.kernel_graphs.build_kernel_graph",
    "labelled_complex_graph":"tanc.graph_builder.boundary_graphs.build_labelled_complex_graph",
    "polyhedral_graph":      "tanc.graph_builder.boundary_graphs.build_polyhedral_graph",
    "weight_trajectory":     "tanc.graph_builder.weight_trajectory.build_weight_trajectory",
}

# Tool name → function mapping
_TOOLS = {
    "ph":        "tanc.topo_tools.ph_tool.run_ph",
    "mapper":    "tanc.topo_tools.mapper_tool.run_mapper",
    "dimension": None,  # handled separately due to sub-method dispatch
}


def _import_fn(dotted_path: str):
    """Dynamically import a function by dotted path."""
    module_path, fn_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


def _indexed_path(path: str, index: int) -> str:
    """Insert ``_{index}`` before the file extension of *path*.

    ``"diagram.png", 0`` → ``"diagram_0.png"``.  Used when ``reproduce`` saves
    one figure per per-layer result.
    """
    from pathlib import Path
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{index}{p.suffix}"))


# Minimum aspects required for each named representation.
_REPRESENTATION_ASPECTS: dict[str, list[str]] = {
    "weights":           ["weights"],
    "activations":       ["activations"],
    "coupled":           ["weights", "activations"],
    "inputs_labels":     ["activations", "classifications"],
    "weight_trajectory": ["weights"],
}


def _infer_aspects(representation) -> list[str]:
    """Return the minimum aspects needed for *representation*.

    Falls back to all three when representation is None or a callable
    (we cannot know what the callable will pull off the snapshot).
    """
    if representation is None or callable(representation):
        return ["weights", "activations", "classifications"]
    return _REPRESENTATION_ASPECTS.get(representation, ["activations"])


def _apply_representation(obj: Any, representation) -> Any:
    """Pull a specific array (or tuple/list of arrays) off *obj*.

    Parameters
    ----------
    obj
        A ``ModelSnapshot`` or ``TrainingView``.
    representation
        One of the string keywords, a layer name, or a callable.
    """
    from tanc.model_extractor._snapshot import TrainingView

    if callable(representation):
        return representation(obj)

    is_view = isinstance(obj, TrainingView)
    snap = obj.final_snapshot if is_view else obj

    if representation == "weights":
        return snap.weight_matrices()
    if representation == "activations":
        return snap.all_activation_matrices()
    if representation == "coupled":
        return snap.coupled_weight_activations()
    if representation == "inputs_labels":
        if snap.inputs is None or snap.predicted_labels is None:
            raise ValueError(
                "representation='inputs_labels' requires both snap.inputs and "
                "snap.predicted_labels — ensure aspects includes 'activations' "
                "and 'classifications'."
            )
        return (snap.inputs, snap.predicted_labels)
    if representation == "weight_trajectory":
        if not is_view:
            raise ValueError(
                "representation='weight_trajectory' requires a TrainingView "
                "— use fit_training(), not fit_model()."
            )
        return obj.weight_trajectory()
    # Treat as a layer name → single activation matrix.
    try:
        return snap.activation_matrix(representation)
    except Exception as exc:
        raise ValueError(
            f"representation={representation!r} is not a recognised keyword "
            f"and could not be resolved as a layer name: {exc}"
        ) from exc


class TDAPipeline:
    """Composable two-module topological analysis pipeline.

    Chains a graph builder (Module 1) with a topological tool (Module 2),
    validates their compatibility, and returns a ``TopoResult``.

    Parameters
    ----------
    builder : str, callable, or None
        Module 1 builder.  Valid strings: ``"weight_graph"``,
        ``"activation_graph"``, ``"labelled_complex_graph"``,
        ``"polyhedral_graph"``, ``"weight_trajectory"``.
        If None, Module 1 is skipped (data passed directly to Module 2).
    builder_kwargs : dict or None
    tool : str
        Module 2 tool.  Valid strings: ``"ph"``, ``"mapper"``,
        ``"dimension"``.
    tool_kwargs : dict or None
    visualisation : str or None
        Default kind for ``.plot()``.
    paper_reference : str or None
        Set automatically by ``.from_paper()``.

    Examples
    --------
    >>> pipe = TDAPipeline.from_paper("watanabe2021")
    >>> result = pipe.fit(weight_matrices)
    >>> result.plot("diagram")

    >>> pipe = TDAPipeline.from_paper("naitzat2020")
    >>> results = pipe.fit(per_layer_activations)  # list[TopoResult]
    """

    def __init__(
        self,
        builder: str | callable | None = None,
        builder_kwargs: dict | None = None,
        tool: str = "ph",
        tool_kwargs: dict | None = None,
        visualisation: str | None = None,
        paper_reference: str | None = None,
    ) -> None:
        self.builder = builder
        self.builder_kwargs = builder_kwargs or {}
        self.tool = tool
        self.tool_kwargs = tool_kwargs or {}
        self.visualisation = visualisation
        self.paper_reference = paper_reference
        self._validated = False

    # ── Validation ─────────────────────────────────────────────────────────

    def validate(self) -> "TDAPipeline":
        """Run compatibility checks and inject defaults.

        Called automatically by ``.fit()``.  May be called manually to
        catch configuration errors early.  Returns self for chaining.
        """
        if isinstance(self.builder, str) or self.builder is None:
            self.builder_kwargs, self.tool_kwargs = check_compatibility(
                builder_name=self.builder or "none",
                builder_kwargs=self.builder_kwargs,
                tool_name=self.tool,
                tool_kwargs=self.tool_kwargs,
            )
        self._validated = True
        return self

    # ── Fit ────────────────────────────────────────────────────────────────

    def fit(self, data: Any, representation=None) -> "TopoResult | list[TopoResult]":
        """Run the pipeline end-to-end.

        Parameters
        ----------
        data
            Input data.  Type depends on ``builder``:

            * ``"weight_graph"``:
              ``np.ndarray | list[ndarray] | list[tuple[ndarray, ndarray]]``
            * ``"activation_graph"``:
              ``np.ndarray`` (single layer) or ``list[ndarray]`` (per-layer).
              When ``list`` and ``tool="ph"``, returns ``list[TopoResult]``.
            * ``"labelled_complex_graph"``:
              ``tuple[ndarray, ndarray]``  (points, labels)
            * ``"polyhedral_graph"``:
              ``np.ndarray``  activation or binary patterns
            * ``builder=None`` (dimension tools):
              ``list[ndarray]`` for activation_id,
              ``ndarray`` for trajectory_dimension.

            You may also pass a ``ModelSnapshot`` / ``TrainingView`` directly —
            the pipeline pulls whatever the chosen builder/tool needs.
        representation : str, callable, or None
            Only used when *data* is a ``ModelSnapshot`` / ``TrainingView``.
            Says *what to pull off it* before building, so you don't have to
            extract arrays by hand: ``"weights"``, ``"activations"``,
            ``"coupled"``, ``"inputs_labels"``, a layer-name string, or a
            callable ``snap -> array(s)`` (e.g. ``lambda s:
            s.weight_matrices(["fc1", "fc2"])`` to study one sub-stack).
            ``None`` lets the builder decide.

        Returns
        -------
        TopoResult or list[TopoResult]
        """
        if not self._validated:
            self.validate()

        # "Just pass the snapshot and say what you want" — resolve the chosen
        # representation to arrays before the normal model-data translation.
        if representation is not None and is_model_data(data):
            data = _apply_representation(data, representation)

        # If the user hands us a ModelSnapshot / TrainingView, translate it
        # into whatever array(s) the chosen builder/tool actually consumes.
        if is_model_data(data):
            # For dupuis2023-style ph_loss: extract loss values from the
            # TrainingView before translation strips them away.
            if (self.builder is None
                    and self.tool_kwargs.get("estimator") == "trajectory_dimension"
                    and self.tool_kwargs.get("method") == "ph_loss"
                    and "loss_values" not in self.tool_kwargs):
                loss_vals = extract_loss_values(data)
            else:
                loss_vals = None

            translated = snapshot_to_builder_data(
                data,
                builder=self.builder if isinstance(self.builder, str) else None,
                builder_kwargs=self.builder_kwargs,
                tool=self.tool,
                tool_kwargs=self.tool_kwargs,
            )
            if self.builder is None:
                return self._run_dimension(translated, _loss_values=loss_vals)
            return self.fit(translated)

        # ── Module 1: build GraphBundle ─────────────────────────────────
        if self.builder is None:
            # Dimension tools; bypass graph builder
            return self._run_dimension(data)

        # Per-layer list handling for activation_graph + ph.
        # Exception: distance="correlation" (Ballester 2024) builds ONE
        # whole-network functional graph, so a per-layer list is concatenated
        # into a single neuron set rather than fitted layer-by-layer.
        if (self.builder == "activation_graph"
                and isinstance(data, list)
                and self.tool == "ph"
                and self.builder_kwargs.get("distance") != "correlation"):
            # A TopoResultSet is a list, but forwards attribute access to the
            # single element when there is one, so `.statistics` is safe to
            # reach for without knowing whether a preset is per-layer.
            return TopoResultSet(self._single_fit(d) for d in data)

        return self._single_fit(data)

    def _single_fit(self, data: Any) -> TopoResult:
        """Run builder + tool on a single data item."""
        bundle = self._run_builder(data)
        return self._run_tool(bundle)

    def _run_builder(self, data):
        """Dispatch to the appropriate Module 1 builder."""
        if callable(self.builder):
            return self.builder(data, **self.builder_kwargs)

        if self.builder not in _BUILDERS:
            raise ValueError(
                f"Unknown builder '{self.builder}'.  "
                f"Valid strings: {list(_BUILDERS.keys())}."
            )

        builder_fn = _import_fn(_BUILDERS[self.builder])

        # labelled_complex_graph expects (points, labels) separately
        if self.builder == "labelled_complex_graph":
            if isinstance(data, tuple) and len(data) == 2:
                points, labels = data
                return builder_fn(points, labels, **self.builder_kwargs)
            raise ValueError(
                "builder='labelled_complex_graph' expects "
                "data=(points, labels) tuple."
            )

        # naitzat2020: auto-select k if None
        if (self.builder == "activation_graph"
                and self.builder_kwargs.get("distance") == "geodesic"
                and self.builder_kwargs.get("k") is None):
            from tanc.graph_builder.activation_graphs import find_optimal_k
            k = find_optimal_k(data)
            kwargs = dict(self.builder_kwargs)
            kwargs["k"] = k
            return builder_fn(data, **kwargs)

        return builder_fn(data, **self.builder_kwargs)

    def _run_tool(self, bundle) -> TopoResult:
        """Dispatch to the appropriate Module 2 tool."""
        if self.tool == "ph":
            run_ph = _import_fn("tanc.topo_tools.ph_tool.run_ph")
            result = run_ph(bundle, **self.tool_kwargs)
        elif self.tool == "mapper":
            run_mapper = _import_fn("tanc.topo_tools.mapper_tool.run_mapper")
            result = run_mapper(bundle, **self.tool_kwargs)
        elif self.tool == "dimension":
            from tanc.topo_tools.dimension_tool import run_trajectory_dimension
            result = run_trajectory_dimension(bundle, **self.tool_kwargs)
        else:
            raise ValueError(
                f"Unknown tool '{self.tool}'.  Valid: 'ph', 'mapper', 'dimension'."
            )
        result.paper_reference = self.paper_reference
        result.default_plot_kind = self.visualisation
        return result

    def _run_dimension(self, data, _loss_values=None) -> TopoResult:
        """Handle builder=None dimension tools directly."""
        method = self.tool_kwargs.get("method", "global")
        estimator = self.tool_kwargs.get("estimator", "activation_id")

        if estimator == "activation_id":
            from tanc.topo_tools.dimension_tool import run_activation_id
            extra_kwargs = {k: v for k, v in self.tool_kwargs.items()
                           if k not in ("method", "estimator")}
            layer_result = run_activation_id(
                data, method=method, **extra_kwargs
            )
            result = TopoResult(
                tool="dimension",
                dimension_result=layer_result,
                config=self.tool_kwargs,
                paper_reference=self.paper_reference,
            )
        elif estimator == "trajectory_dimension":
            from tanc.graph_builder.weight_trajectory import build_weight_trajectory
            from tanc.topo_tools.dimension_tool import run_trajectory_dimension

            # Build a GraphBundle from the raw trajectory
            traj_method = method
            # loss_values: from explicit tool_kwargs, or passed via _loss_values
            # (the latter is set by fit() when data was a TrainingView)
            loss_values = self.tool_kwargs.get("loss_values", _loss_values)
            if traj_method in ("ph_euclidean", "ph_loss"):
                distance = "loss_difference" if traj_method == "ph_loss" else "euclidean"
                bundle = build_weight_trajectory(data, distance=distance,
                                                 loss_values=loss_values)
                extra = {k: v for k, v in self.tool_kwargs.items()
                         if k not in ("method", "estimator", "loss_values")}
                result = run_trajectory_dimension(
                    bundle, method="ph_dimension",
                    loss_values=loss_values, **extra
                )
            elif traj_method == "magnitude":
                bundle = build_weight_trajectory(data, distance="euclidean")
                extra = {k: v for k, v in self.tool_kwargs.items()
                         if k not in ("method", "estimator")}
                result = run_trajectory_dimension(
                    bundle, method="magnitude_dimension", **extra
                )
            else:
                raise ValueError(f"Unknown method '{traj_method}'.")
            result.paper_reference = self.paper_reference
        else:
            raise ValueError(
                f"Unknown estimator '{estimator}'. "
                "Valid: 'activation_id', 'trajectory_dimension'."
            )

        result.default_plot_kind = self.visualisation
        return result

    # ── Class method: from_paper ───────────────────────────────────────────

    @classmethod
    def from_paper(cls, paper: str, **kwargs) -> "TDAPipeline":
        """Construct a TDAPipeline preconfigured to reproduce a published method.

        Parameters
        ----------
        paper : str
            Paper key.  Valid values::

                "watanabe2021", "rieck2019", "gebhart2019", "lacombe2021",
                "naitzat2020", "karuppiah2025", "ballester2024",
                "ramamurthy2019", "liu2023",
                "rathore2021", "zhou2023", "gabrielsson2019", "gabella2021",
                "ruppik2025", "ong2026",
                "birdal2021", "dupuis2023", "andreeva2024".

        **kwargs
            Override any preset parameter.  Dotted keys target nested dicts::

                builder_kwargs__edge_weight="absolute"
                tool_kwargs__max_dim=2

        Returns
        -------
        TDAPipeline

        Raises
        ------
        ValueError : unknown paper key.

        Examples
        --------
        >>> pipe = TDAPipeline.from_paper("watanabe2021")
        >>> result = pipe.fit(weight_matrices)
        >>> result.plot("diagram")

        >>> pipe = TDAPipeline.from_paper("naitzat2020")
        >>> results = pipe.fit(per_layer_activations)  # list[TopoResult]

        >>> pipe = TDAPipeline.from_paper("ramamurthy2019")
        >>> result = pipe.fit((input_points, predicted_labels))
        """
        if paper not in PAPER_PRESETS:
            valid = sorted(PAPER_PRESETS.keys())
            raise ValueError(
                f"Unknown paper '{paper}'. Valid keys:\n  " + "\n  ".join(valid)
            )

        preset = PAPER_PRESETS[paper].copy()
        builder_kwargs = dict(preset["builder_kwargs"])
        tool_kwargs = dict(preset["tool_kwargs"])

        for k, v in kwargs.items():
            if k.startswith("builder_kwargs__"):
                builder_kwargs[k.removeprefix("builder_kwargs__")] = v
            elif k.startswith("tool_kwargs__"):
                tool_kwargs[k.removeprefix("tool_kwargs__")] = v
            else:
                preset[k] = v

        return cls(
            builder=preset["builder"],
            builder_kwargs=builder_kwargs,
            tool=preset["tool"],
            tool_kwargs=tool_kwargs,
            visualisation=preset.get("visualisation"),
            paper_reference=preset["paper_reference"],
        )

    # ── Discoverability / introspection ────────────────────────────────────

    @classmethod
    def list_presets(cls, print_it: bool = True) -> list[str]:
        """List every available ``from_paper(...)`` preset.

        Grouped by tool when printed.  Returns the list of names regardless.
        """
        from tanc.pipeline.paper_presets import list_presets as _list
        return _list(print_it=print_it)

    @classmethod
    def describe_preset(cls, paper: str, print_it: bool = True) -> str:
        """Pretty-print one preset's pipeline, inputs, and expected outputs."""
        from tanc.pipeline.paper_presets import describe_preset
        return describe_preset(paper, print_it=print_it)

    def explain(self, print_it: bool = True) -> str:
        """Describe in plain English what this pipeline will do when fit.

        Walks the configured builder + tool + tool_kwargs and renders the
        full plan as a human-readable block.
        """
        lines: list[str] = []
        if self.paper_reference:
            lines.append(f"TDAPipeline — {self.paper_reference}")
        else:
            lines.append("TDAPipeline")

        # Builder stage
        if self.builder is None:
            lines.append("  Stage 1: (skipped — direct dimension estimation)")
        elif callable(self.builder):
            lines.append(f"  Stage 1: builder = custom callable "
                         f"{getattr(self.builder, '__name__', repr(self.builder))}")
        else:
            lines.append(f"  Stage 1: builder = {self.builder}")
        if self.builder_kwargs:
            for k, v in self.builder_kwargs.items():
                lines.append(f"             {k} = {v!r}")

        # Tool stage
        lines.append(f"  Stage 2: tool    = {self.tool}")
        if self.tool_kwargs:
            for k, v in self.tool_kwargs.items():
                lines.append(f"             {k} = {v!r}")

        # Output
        if self.visualisation:
            lines.append(f"  Default plot   : .plot({self.visualisation!r})")

        text = "\n".join(lines)
        if print_it:
            print(text)
        return text

    # ── Convenience methods ────────────────────────────────────────────────

    def plot(self, result: TopoResult, kind: str | None = None):
        """Convenience wrapper: ``result.plot(kind or self.visualisation)``."""
        k = kind or self.visualisation or "diagram"
        return result.plot(k)

    def reproduce(self, data: Any, kind: str | None = None,
                  save: str | None = None, representation=None, **plot_kwargs):
        """Fit the pipeline and render its default plot in one call.

        Designed for paper-preset reproduction: configure with
        ``from_paper(...)``, hand in the appropriately-shaped data, and
        get back the matplotlib figure plus the underlying ``TopoResult``.

        Parameters
        ----------
        data
            Forwarded to ``self.fit(data)``.
        kind : str or None
            Plot kind override; defaults to ``self.visualisation``.
        save : str or None
            When given, write the figure to this path (extension picks the
            format).  If the fit yields several results (one figure per layer),
            the layer index is inserted before the extension, e.g.
            ``diagram.png`` → ``diagram_0.png``, ``diagram_1.png`` ….
        **plot_kwargs
            Forwarded to ``result.plot``.

        Returns
        -------
        (result, fig) : (TopoResult | list[TopoResult], Figure | list[Figure])
        """
        result = self.fit(data, representation=representation)
        k = kind or self.visualisation

        if isinstance(result, list):
            figs = []
            for i, r in enumerate(result):
                fig_save = _indexed_path(save, i) if save is not None else None
                figs.append(r.plot(k, save=fig_save, **plot_kwargs))
            return result, figs
        return result, result.plot(k, save=save, **plot_kwargs)

    def compare(
        self,
        data_dict: dict[str, Any],
        kind: str | None = None,
        layer: int | None = None,
    ):
        """Run ``.fit()`` on each entry in data_dict and plot side-by-side.

        Parameters
        ----------
        data_dict : dict[str, data]
            Keys become panel labels.  Values are passed to ``.fit()``.
        kind : str or None
            Visualisation kind; falls back to ``self.visualisation``.
        layer : int or None
            Which layer to compare when the analysis is per-layer (a
            :class:`TopoResultSet` of more than one result).  Required in that
            case, since each label must reduce to a single panel.

        Returns
        -------
        matplotlib Figure
        """
        from tanc.visualisation.representations import plot_diagram_comparison
        from tanc.topo_tools._result import PersistenceResult, TopoResultSet

        results = {label: self.fit(data) for label, data in data_dict.items()}
        k = kind or self.visualisation or "diagram"

        # Per-layer analyses return one result per layer, so reduce each entry
        # to the single result its panel will show.
        reduced: dict[str, Any] = {}
        for label, res in results.items():
            if not isinstance(res, TopoResultSet):
                reduced[label] = res
            elif len(res) == 1:
                reduced[label] = res[0]
            elif layer is None:
                raise ValueError(
                    f"'{label}' produced {len(res)} per-layer results, so there is no "
                    f"single panel to draw for it. Pass layer=<index> to compare one "
                    f"layer (0-{len(res) - 1})."
                )
            else:
                reduced[label] = res[layer]
        results = reduced

        if k in {"diagram", "barcode", "betti_curve"}:
            # Extract PersistenceResult objects for comparison plot
            ph_results: dict[str, PersistenceResult] = {}
            for label, res in results.items():
                if hasattr(res, "ph_result") and res.ph_result is not None:
                    ph_results[label] = res.ph_result
            if ph_results:
                return plot_diagram_comparison(ph_results, kind=k)

        # Fallback: side-by-side individual plots
        import matplotlib.pyplot as plt
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, (label, res) in zip(axes, results.items()):
            res.plot(k, ax=ax, title=label)
        fig.tight_layout()
        return fig

    # ── Model-first entry points ───────────────────────────────────────────────

    def fit_model(
        self,
        model,
        data,
        aspects: list[str] | None = None,
        layer_selection=None,
        representation=None,
        device: str | None = None,
        activation_pooling: str = "flatten",
        soro_effective_weight: bool = True,
    ) -> "TopoResult | list[TopoResult]":
        """Extract from a trained model and run the pipeline end-to-end.

        Parameters
        ----------
        model
            Trained PyTorch or TensorFlow model.
        data
            Input data for the extraction forward pass (numpy array or
            torch Tensor).
        aspects : list[str] or None
            Which aspects to extract: any subset of
            ``["weights", "activations", "classifications"]``.
            Inferred automatically from *representation* when None.
        layer_selection : None, str, list[str], or list[int]
            Which layers to capture.  None = auto-select based on builder.
        representation : str, callable, or None
            Controls what is pulled off the snapshot before being handed
            to the builder:

            * ``None`` — let the builder decide (same as ``fit(snapshot)``).
            * ``"weights"`` — ``snap.weight_matrices()``
            * ``"activations"`` — ``snap.all_activation_matrices()``
            * ``"coupled"`` — ``snap.coupled_weight_activations()``
            * ``"inputs_labels"`` — ``(snap.inputs, snap.predicted_labels)``
            * a layer name string — ``snap.activation_matrix(layer_name)``
            * a callable — called with the full ``ModelSnapshot``; return
              whatever the builder expects.
        device : str or None
            PyTorch device string; ignored for TensorFlow models.
        activation_pooling : str
            How to collapse each captured activation to ``(N_samples,
            features)``.  ``"flatten"`` (default) preserves the historical
            behaviour; for transformer ``(batch, seq, hidden)`` activations,
            ``"last"`` / ``"mean"`` / ``"first"`` / ``"max"`` pool over the
            sequence axis (e.g. ``"last"`` = final-token summary).
        soro_effective_weight : bool
            For Sum-of-Rank-One (SoRO) layers, also store the assembled
            effective weight so they feed weight-based tools like FC layers
            (default ``True``); the trained factors are always kept in
            ``snapshot.soro_factors``.

        Returns
        -------
        TopoResult or list[TopoResult]

        Examples
        --------
        >>> pipe = TDAPipeline.from_paper("watanabe2021")
        >>> result = pipe.fit_model(model, X_test)

        >>> # last-token activations of a transformer's blocks
        >>> pipe = TDAPipeline.from_paper("ong2026")
        >>> result = pipe.fit_model(llm, prompts, representation="activations",
        ...                         layer_selection=["blocks.0", "blocks.1"],
        ...                         activation_pooling="last")

        >>> pipe = TDAPipeline.from_paper("naitzat2020")
        >>> results = pipe.fit_model(model, X_test, representation="activations")

        >>> pipe = TDAPipeline(builder="activation_graph", tool="ph")
        >>> result = pipe.fit_model(model, X_test, representation="fc3")
        """
        from tanc.model_extractor import extract_model

        inferred_aspects = aspects if aspects is not None else _infer_aspects(representation)

        snap = extract_model(
            model=model,
            data=data,
            aspects=inferred_aspects,
            layer_selection=layer_selection,
            clarify=False,
            device=device,
            activation_pooling=activation_pooling,
            soro_effective_weight=soro_effective_weight,
        )

        if representation is None:
            return self.fit(snap)
        return self.fit(_apply_representation(snap, representation))

    def fit_training(
        self,
        model,
        extract_data,
        epochs: int = 100,
        target_accuracy: float | None = None,
        snapshot_every: int = 1,
        snapshot_schedule: str = "epoch",
        aspects: list[str] | None = None,
        layer_selection=None,
        representation=None,
        verbose: bool = True,
        **train_kwargs,
    ) -> "TopoResult | list[TopoResult]":
        """Train a model, capture snapshots, and run the pipeline end-to-end.

        Parameters
        ----------
        model
            PyTorch or TensorFlow model (need not be pre-trained).
        extract_data
            Data used for the extraction forward pass at each snapshot.
        epochs : int
            Number of training epochs.
        target_accuracy : float or None
            Stop early when validation accuracy reaches this threshold.
        snapshot_every : int
            Capture a snapshot every N epochs (or iterations).
        snapshot_schedule : str
            ``"epoch"`` or ``"iteration"``.
        aspects : list[str] or None
            Which aspects to capture at each snapshot.  Inferred from
            *representation* when None.
        layer_selection : None, str, list[str], or list[int]
            Which layers to capture.
        representation : str, callable, or None
            Controls what is passed to the builder.  Same options as
            ``fit_model``, plus:

            * ``"weight_trajectory"`` — ``training_view.weight_trajectory()``
              (full per-layer weight matrices across all snapshots).
            * ``None`` — passes the full ``TrainingView`` to ``fit()``;
              trajectory-aware presets (e.g. ``ong2026``, ``ruppik2025``)
              will capture the trajectory automatically; snapshot-based
              presets will use the final snapshot.
            * a callable — called with the full ``TrainingView``.
        verbose : bool
            Print training progress.
        **train_kwargs
            Framework-specific arguments forwarded to ``extract_training()``:

            *Cross-framework data (preferred)*: ``train_data`` and ``val_data``
            as a raw dataset per split (``(X, y)`` arrays, a ``Dataset``, or a
            ready loader) plus ``batch_size`` — the right loader is built for
            the detected framework.

            *PyTorch*: ``criterion``, ``optimizer`` (pre-built) **or**
            ``optimizer_class`` + ``optimizer_kwargs`` (built automatically
            against model.parameters()), ``device``; pre-built ``train_loader``
            / ``val_loader`` still accepted.

            *TensorFlow*: ``compile_kwargs``.

        Returns
        -------
        TopoResult or list[TopoResult]

        Examples
        --------
        >>> pipe = TDAPipeline.from_paper("ong2026")
        >>> result = pipe.fit_training(
        ...     model, X_test, epochs=200,
        ...     train_data=(X_tr, y_tr), val_data=(X_te, y_te),
        ...     criterion=crit, optimizer=opt,
        ... )

        >>> pipe = TDAPipeline.from_paper("watanabe2021")
        >>> result = pipe.fit_training(
        ...     model, X_test, epochs=200,
        ...     representation="weights",
        ...     train_loader=loader, criterion=crit, optimizer=opt,
        ... )
        """
        from tanc.model_extractor import extract_training

        inferred_aspects = aspects if aspects is not None else _infer_aspects(representation)

        # Allow optimizer_class + optimizer_kwargs instead of a pre-built optimizer,
        # which is necessary when the model is created fresh inside the caller.
        if "optimizer_class" in train_kwargs:
            opt_cls = train_kwargs.pop("optimizer_class")
            opt_kwargs = train_kwargs.pop("optimizer_kwargs", {})
            train_kwargs["optimizer"] = opt_cls(model.parameters(), **opt_kwargs)

        training_view = extract_training(
            model=model,
            extract_data=extract_data,
            aspects=inferred_aspects,
            layer_selection=layer_selection,
            epochs=epochs,
            target_accuracy=target_accuracy,
            snapshot_every=snapshot_every,
            snapshot_schedule=snapshot_schedule,
            verbose=verbose,
            **train_kwargs,
        )

        if representation is None:
            return self.fit(training_view)
        return self.fit(_apply_representation(training_view, representation))

    def over_training(
        self,
        view,
        measures,
        layers=None,
        representation=None,
        on_error: str = "nan",
    ):
        """Track scalar topological measures across a training trajectory.

        Runs this configured pipeline on every snapshot of *view* (per layer,
        when *layers* is given) and returns a ``TrajectorySeries`` — the
        numbers plus a ``.plot()``.  This is the reusable form of the
        hand-rolled "fit each snapshot and pull one statistic" loop, and it
        works with any pipeline, including ``TDAPipeline.from_paper(...)``.

        Parameters
        ----------
        view : TrainingView
        measures : str | callable | sequence of them
            What to read off each snapshot's ``TopoResult``: a statistics key
            (e.g. ``"H1_total_persistence"``), the literal ``"dimension"``, or a
            callable ``TopoResult -> float``.
        layers : sequence of str or None
            When given, the pipeline runs on each layer independently → one line
            per layer.  ``None`` → a single combined series.
        representation : str | callable or None
            How each snapshot becomes pipeline input.  With *layers* set:
            ``"weights"`` (default), ``"activations"``, or a ``(snapshot, layer)``
            callable.  With ``layers=None``: any ``fit`` representation.
        on_error : str
            ``"nan"`` (default) or ``"raise"``.

        Returns
        -------
        TrajectorySeries

        Examples
        --------
        >>> series = TDAPipeline.from_paper("watanabe2021").over_training(
        ...     view, measures=["H1_total_persistence"], layers=["fc1", "fc2"])
        >>> series.plot()   # one line per layer
        """
        from tanc.visualisation.trajectory_plots import pipeline_trajectory

        return pipeline_trajectory(
            view, self, measures,
            layers=layers, representation=representation, on_error=on_error,
        )

    def fit_models(
        self,
        models: list,
        data,
        representation=None,
        layer_idx: int | None = None,
        aspects: list[str] | None = None,
        layer_selection=None,
        device: str | None = None,
    ) -> "TopoResult | list[TopoResult]":
        """Extract a representation from each model, stack into one point cloud, and run.

        Runs the same extraction as ``fit_model`` on every model in *models*,
        then vertically stacks the resulting arrays before calling ``fit()``.

        This is the entry point for multi-model analyses such as
        Gabrielsson & Carlsson (2019), where conv kernels from several trained
        models are pooled into one point cloud before Mapper.

        Parameters
        ----------
        models : list
            Trained PyTorch or TensorFlow models.  All must share the same
            architecture (so that stacked arrays have compatible shapes).
        data
            Input data for the extraction forward pass.
        representation : str, callable, or None
            Same options as ``fit_model``.  The result for each model must
            be a 2-D array ``(n_points, n_features)`` so that stacking is
            unambiguous.  When the representation returns a *list* of arrays
            (e.g. ``"weights"`` → one matrix per layer), ``layer_idx`` must
            be given to select a single layer.
        layer_idx : int or None
            When *representation* yields a list, pick this layer index.
            Required if the extracted value is a list; ignored otherwise.
        aspects, layer_selection, device
            Forwarded to ``extract_model`` — same semantics as ``fit_model``.

        Returns
        -------
        TopoResult or list[TopoResult]
            The result of running ``fit()`` on the stacked point cloud.

        Examples
        --------
        Stack layer-1 kernels from three models (gabrielsson2019):

        >>> pipe = TDAPipeline.from_paper("gabrielsson2019")
        >>> result = pipe.fit_models(
        ...     [model_a, model_b, model_c],
        ...     X_train,
        ...     representation="weights",
        ...     layer_idx=0,          # which layer's weight matrix to stack
        ... )

        Custom callable — flatten conv kernels to vectors before stacking:

        >>> result = pipe.fit_models(
        ...     models,
        ...     X_train,
        ...     representation=lambda snap: (
        ...         snap.weight_matrices()[0].reshape(snap.weight_matrices()[0].shape[0], -1)
        ...     ),
        ... )
        """
        from tanc.model_extractor import extract_model

        inferred_aspects = aspects if aspects is not None else _infer_aspects(representation)

        arrays = []
        for model in models:
            snap = extract_model(
                model=model,
                data=data,
                aspects=inferred_aspects,
                layer_selection=layer_selection,
                clarify=False,
                device=device,
            )
            arr = (
                _apply_representation(snap, representation)
                if representation is not None
                else snap.weight_matrices()        # sensible default for multi-model
            )

            # If the representation returned a list (e.g. "weights"), pick a layer.
            if isinstance(arr, list):
                if layer_idx is None:
                    raise ValueError(
                        "fit_models: the representation returned a list of arrays "
                        "(one per layer).  Set layer_idx to select a single layer "
                        "for stacking, or use a callable that returns a single 2-D array."
                    )
                arr = arr[layer_idx]

            arr = np.atleast_2d(arr)
            arrays.append(arr)

        stacked = np.vstack(arrays)
        return self.fit(stacked)

    def fit_each(
        self,
        models: list,
        data,
        representation=None,
        aspects: list[str] | None = None,
        layer_selection=None,
        device: str | None = None,
    ) -> "list[TopoResult]":
        """Run the pipeline *separately* on each model — one result per model.

        Unlike :meth:`fit_models` (which stacks every model into a single point
        cloud), ``fit_each`` extracts and runs the pipeline independently for
        every model and returns the list of results in order.  This is the
        population entry point for per-model summaries — e.g. Ballester et al.
        (2024), where each network yields one ASDSQ persistence-summary vector
        and those vectors are later regressed onto the generalization gap.

        Parameters
        ----------
        models : list
            Trained PyTorch or TensorFlow models.  They need *not* share an
            architecture — each is processed on its own.
        data
            Input data for the extraction forward pass, shared across models.
        representation : str, callable, or None
            Same options as :meth:`fit_model`.  ``None`` lets the builder
            consume the snapshot directly; ``"activations"`` hands the builder
            the concatenated per-layer activation matrices (the natural input
            for the ``ballester2024`` functional-graph preset).
        aspects, layer_selection, device
            Forwarded to extraction — same semantics as :meth:`fit_model`.

        Returns
        -------
        list[TopoResult]
            One result per model, each carrying its own ``statistics`` (e.g.
            the ASDSQ keys for the ballester preset).

        Examples
        --------
        >>> pipe = TDAPipeline.from_paper("ballester2024")
        >>> results = pipe.fit_each(models, X_eval,
        ...                         representation="activations",
        ...                         layer_selection="linear_and_conv")
        >>> S = np.array([[r.statistics[k] for k in keys] for r in results])
        """
        from tanc.model_extractor import extract_model

        inferred_aspects = aspects if aspects is not None else _infer_aspects(representation)

        results: list[TopoResult] = []
        for model in models:
            snap = extract_model(
                model=model,
                data=data,
                aspects=inferred_aspects,
                layer_selection=layer_selection,
                clarify=False,
                device=device,
            )
            obj = snap if representation is None else _apply_representation(snap, representation)
            results.append(self.fit(obj))
        return results

    def __repr__(self) -> str:
        return (
            f"TDAPipeline("
            f"builder={self.builder!r}, "
            f"tool={self.tool!r}, "
            f"paper_reference={self.paper_reference!r})"
        )
