"""tanc.tour() / tanc.help() — printed onboarding for new users.

These helpers print a top-down overview of the package without forcing the
user to read READMEs.  They are intentionally lightweight — for the real
docs see ``GETTING_STARTED.md`` at the repo root or the rendered Sphinx
build under ``docs/_build/html``.
"""

from __future__ import annotations


_OVERVIEW = """\
TANC — Topological Data Analysis for Neural Networks
============================================================

The package has four modules wired in series:

    Neural network model
         |
         v
    [model_extractor]   -->  ModelSnapshot / TrainingView
         |
         v
    [graph_builder]     -->  GraphBundle  (distance / similarity / adjacency)
         |
         v
    [topo_tools]        -->  TopoResult   (PH / Mapper / dimension)
         |
         v
    [visualisation]     -->  matplotlib Figures

You can enter the pipeline at whichever stage you have data for — hand
``TDAPipeline.fit()`` a model, a snapshot, weight matrices, activation
arrays, or a precomputed distance matrix.
"""

_QUICKSTART = """\
Quick start
-----------

    >>> import tanc as tda
    >>> from tanc import TDAPipeline, ModelExtractor, TrainingExtractor
    >>>
    >>> # 1. Reproduce a published method end-to-end:
    >>> pipe = TDAPipeline.from_paper("watanabe2021")
    >>> pipe.explain()                         # what will this pipeline do?
    >>> result, fig = pipe.reproduce(weight_matrices)
    >>>
    >>> # 2. Inspect what came out:
    >>> result.describe()                      # populated fields + available plots
    >>> result.plots_available()               # ['diagram', 'barcode', 'betti_curve']
    >>> H1 = result.diagram(dim=1)             # short for result.ph_result.diagrams[1]
    >>>
    >>> # 3. Browse the catalogue:
    >>> TDAPipeline.list_presets()             # 17 papers, grouped by tool
    >>> TDAPipeline.describe_preset("rieck2019")
    >>>
    >>> # 4. Extract from a trained model (builder pattern, like TDAPipeline):
    >>> snap = ModelExtractor(model, aspects=['weights']).extract(X_test)
    >>>
    >>> # 5. Train and capture snapshots — framework auto-detected from `model`:
    >>> ext = TrainingExtractor(
    ...     model=model, train_data=(X_train, y_train), val_data=(X_test, y_test),
    ...     criterion=..., optimizer=...,        # torch only; loader built for you
    ...     extract_data=X_test, snapshot_every=5)
    >>> ext.explain()
    >>> view = ext.run(epochs=50, target_accuracy=0.95)
"""


def _summarise_module(module_name: str) -> str:
    import importlib
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        return f"  (failed to import {module_name}: {e})"
    names = getattr(mod, "__all__", None) or [
        n for n in dir(mod) if not n.startswith("_")
    ]
    return ", ".join(sorted(names))


def tour(print_it: bool = True) -> str:
    """Print a tour of the package: modules, presets, and plot catalogue.

    Parameters
    ----------
    print_it : bool
        When ``True`` (default) the tour is printed.  In all cases the
        text is returned.
    """
    from tanc.pipeline.paper_presets import PAPER_PRESETS

    lines: list[str] = []
    lines.append(_OVERVIEW)
    lines.append(_QUICKSTART)

    # Paper presets
    lines.append("Paper presets")
    lines.append("-------------")
    by_tool: dict[str, list[str]] = {}
    for name, preset in PAPER_PRESETS.items():
        by_tool.setdefault(preset["tool"], []).append(name)
    for tool in ("ph", "mapper", "dimension"):
        if tool not in by_tool:
            continue
        names = sorted(by_tool[tool])
        lines.append(f"  [{tool}]  ({len(names)} presets)")
        for n in names:
            ref = PAPER_PRESETS[n].get("paper_reference", "")
            lines.append(f"    {n:18s}  {ref}")
        lines.append("")

    # Visualisation catalogue
    lines.append("Visualisation catalogue (tanc.visualisation)")
    lines.append("--------------------------------------------------")
    lines.append("  Persistence representations:")
    lines.append("    plot_persistence_diagram, plot_barcode, plot_betti_curve,")
    lines.append("    plot_diagram_comparison,")
    lines.append("    plot_persistence_landscape, plot_persistence_image,")
    lines.append("    plot_betti_layer_bars   (Naitzat-style per-layer bars)")
    lines.append("  Training-dynamics (driven by a TrainingView):")
    lines.append("    plot_training_curves, plot_weight_norm_trajectory,")
    lines.append("    plot_activation_stats_trajectory")
    lines.append("  Topological trajectories over epochs:")
    lines.append("    plot_id_over_training, plot_id_trajectory_all_layers,")
    lines.append("    plot_ph_dimension_over_training,")
    lines.append("    plot_magnitude_dimension_over_training,")
    lines.append("    plot_ph_statistic_trajectory, plot_betti_trajectory,")
    lines.append("    plot_diagram_distance_trajectory,")
    lines.append("    plot_diagram_distance_matrix, plot_ph_statistic_pairplot,")
    lines.append("    plot_diagram_evolution")
    lines.append("  Dashboards / animations / pipeline diagram:")
    lines.append("    plot_training_summary, make_training_animation,")
    lines.append("    plot_pipeline_diagram")
    lines.append("  Per-snapshot dimension diagnostics:")
    lines.append("    plot_id_across_layers, plot_ph_scaling, plot_magnitude_scaling,")
    lines.append("    plot_2nn_ratio_distribution, plot_loglog_ratio_distribution,")
    lines.append("    plot_id_with_uncertainty, plot_id_qq")
    lines.append("  Graphs / embeddings / TU scores:")
    lines.append("    plot_graph_matrix, plot_node_embedding,")
    lines.append("    plot_tu_score_distribution, plot_tu_roc,")
    lines.append("    plot_pathways_on_network   (Gebhart-style pathway overlay)")
    lines.append("  Mapper companions (topo_tools):")
    lines.append("    node_exemplars              (TopoAct-style per-node samples)")
    lines.append("    mapper_gw_distance          (Zhou-style GW distance — needs POT)")
    lines.append("")

    lines.append("Where to go next")
    lines.append("----------------")
    lines.append("  * tanc.help('<thing>')        — focused help on a sub-topic")
    lines.append("  * GETTING_STARTED.md                — full walkthrough")
    lines.append("  * docs/_build/html/index.html       — rendered API reference")
    lines.append("                                        (run `make html` in docs/)")
    lines.append("  * paper_reproduce/                  — one notebook per preset")

    text = "\n".join(lines)
    if print_it:
        print(text)
    return text


_TOPIC_HANDLERS: dict[str, str] = {
    "presets":     "Paper presets",
    "papers":      "Paper presets",
    "pipeline":    "Pipeline",
    "extract":     "Model extraction",
    "extraction":  "Model extraction",
    "graph":       "Graph builders",
    "builders":    "Graph builders",
    "ph":          "Persistent homology",
    "persistence": "Persistent homology",
    "mapper":      "Mapper",
    "dimension":   "Dimension estimation",
    "id":          "Dimension estimation",
    "viz":         "Visualisation",
    "visualisation":"Visualisation",
    "plots":       "Visualisation",
    "training":    "Training trajectories",
    "trajectory":  "Training trajectories",
    "uncertainty": "Topological uncertainty",
    "tu":          "Topological uncertainty",
    "save":        "Saving & loading",
    "load":        "Saving & loading",
    "io":          "Saving & loading",
}


def help(topic: str | None = None) -> str:
    """Focused help for one sub-topic of the package.

    Parameters
    ----------
    topic : str or None
        Topic keyword.  Run ``help()`` with no argument to list valid
        topics.  Recognised: ``"presets"``, ``"pipeline"``, ``"extract"``,
        ``"graph"``, ``"ph"``, ``"mapper"``, ``"dimension"``, ``"viz"``,
        ``"training"``, ``"uncertainty"``.

    Notes
    -----
    Shadows ``builtins.help``; access the builtin via ``builtins.help`` if
    needed inside a session where you've done ``from tanc import help``.
    """
    if topic is None:
        print("Available topics: " + ", ".join(sorted(set(_TOPIC_HANDLERS.values()))))
        print("Usage: tanc.help('presets')")
        return ""

    key = topic.lower().strip()
    section = _TOPIC_HANDLERS.get(key)
    if section is None:
        print(f"Unknown topic '{topic}'. Valid: "
              + ", ".join(sorted(_TOPIC_HANDLERS.keys())))
        return ""

    if section == "Paper presets":
        from tanc.pipeline.paper_presets import list_presets
        list_presets(print_it=True)
        print("Use TDAPipeline.describe_preset('<name>') for details on one preset.")
        return ""

    if section == "Pipeline":
        text = (
            "TDAPipeline — single entry point for end-to-end runs.\n\n"
            "  from tanc import TDAPipeline\n"
            "  pipe = TDAPipeline.from_paper('rieck2019')\n"
            "  pipe.explain()                  # what the pipeline will do\n"
            "  result = pipe.fit(weight_matrices)\n"
            "  result, fig = pipe.reproduce(weight_matrices)  # fit + plot in one call\n"
            "  pipe.compare({'A': data_a, 'B': data_b})       # side-by-side panels"
        )
        print(text); return text

    if section == "Model extraction":
        text = (
            "Model extraction — go from raw model + data to typed snapshots.\n"
            "Both classes auto-detect the framework from `model`.\n\n"
            "Class form (preferred):\n"
            "  from tanc.model_extractor import ModelExtractor, TrainingExtractor\n"
            "  snap = ModelExtractor(model, aspects=['weights']).extract(X)\n"
            "  ext  = TrainingExtractor(\n"
            "             model=model,                          # torch or tf\n"
            "             train_data=(X_tr, y_tr),              # raw dataset, both frameworks\n"
            "             val_data=(X_te, y_te), batch_size=128,  # loader built for you\n"
            "             criterion=..., optimizer=...,         # torch only\n"
            "             # or compile_kwargs=...,              # tf only\n"
            "             extract_data=X_test, snapshot_every=5)\n"
            "  view = ext.run(epochs=50, target_accuracy=0.95)\n"
            "  view = ext.run(epochs=10, continue_training=True)   # resume training\n\n"
            "  # train_data/val_data accept (X, y) arrays, a Dataset, or a ready loader.\n"
            "  # train_loader/val_loader still work (PyTorch) for full control.\n\n"
            "Explicit factories (equivalent, self-documenting):\n"
            "  TrainingExtractor.for_pytorch(model, train_data=..., ...)\n"
            "  TrainingExtractor.for_tensorflow(model, train_data=..., ...)\n\n"
            "One-call shortcuts (equivalent, no continuation):\n"
            "  from tanc.model_extractor import extract_model, extract_training\n"
            "  snap = extract_model(model, X)\n"
            "  view = extract_training(model, train_data=..., ...)\n\n"
            "Snapshots/views feed straight into TDAPipeline.fit()."
        )
        print(text); return text

    if section == "Graph builders":
        text = (
            "Graph builders — turn arrays into GraphBundle objects.\n\n"
            "  build_weight_graph, build_activation_graph, build_kernel_graph,\n"
            "  build_labelled_complex_graph, build_polyhedral_graph,\n"
            "  build_weight_trajectory\n\n"
            "Each returns a GraphBundle with matrix_type in\n"
            "{'distance', 'similarity', 'adjacency'}."
        )
        print(text); return text

    if section == "Persistent homology":
        text = (
            "Persistent homology — run_ph(bundle) returns a TopoResult.\n\n"
            "  from tanc.topo_tools import run_ph\n"
            "  result = run_ph(bundle, max_dim=1)\n"
            "  H1 = result.diagram(dim=1)               # convenience accessor\n"
            "  result.describe()                        # available plots + stats"
        )
        print(text); return text

    if section == "Mapper":
        text = (
            "Mapper — run_mapper(bundle) returns a TopoResult.\n\n"
            "  from tanc.topo_tools import run_mapper\n"
            "  result = run_mapper(bundle, filter_fn='l2_norm',\n"
            "                      n_intervals=40, overlap_frac=0.3)\n"
            "  result.mapper.graph                      # networkx Graph\n"
            "  result.plot('graph')"
        )
        print(text); return text

    if section == "Dimension estimation":
        text = (
            "Dimension estimation — three flavours.\n\n"
            "  * Activation ID (per-layer):  run_activation_id(activations,\n"
            "      method='calibrated' (Ong 2026) | 'global' (asymptotic 2NN) | 'local')\n"
            "  * Trajectory PH dimension:    compute_ph_dimension(traj_distance_matrix)\n"
            "  * Magnitude dimension:        compute_magnitude_dimension(traj_distance_matrix)\n\n"
            "All return dicts; the TDAPipeline wraps them in a TopoResult\n"
            "with .dimension as the scalar accessor.\n\n"
            "Visual diagnostics:\n"
            "  plot_id_across_layers(result)                 # the basic line plot\n"
            "  plot_id_with_uncertainty(activations, n=20)   # mean +/- 1 SD bands\n"
            "  plot_id_qq(result, dist='norm')               # tail-behaviour QQ\n"
            "  plot_2nn_ratio_distribution / plot_loglog_ratio_distribution"
        )
        print(text); return text

    if section == "Visualisation":
        from tanc import visualisation
        names = sorted(n for n in getattr(visualisation, "__all__", []))
        text = (
            "tanc.visualisation — every plot function.\n\n  "
            + "\n  ".join(names)
            + "\n\nMost users access these via result.plot(kind) on a TopoResult."
            + "\n  result.plot('diagram', save='fig.pdf')   # write straight to disk"
            + "\n  See tda.help('save') to persist results, not just figures."
        )
        print(text); return text

    if section == "Saving & loading":
        text = (
            "Saving & loading — stop at any stage, resume without recomputing.\n\n"
            "Every result container has .save(path) and a .load(path) classmethod\n"
            "(a pickled '.tda' file; the suffix is added if you omit it):\n\n"
            "  view.save('trajectory.tda');   TrainingView.load('trajectory.tda')\n"
            "  snap.save('snap.tda');         ModelSnapshot.load('snap.tda')\n"
            "  bundle.save('graph.tda');      GraphBundle.load('graph.tda')\n"
            "  result.save('ph.tda');         TopoResult.load('ph.tda')\n\n"
            "Figures save via the plot path, not .save():\n"
            "  result.plot('diagram', save='fig.pdf')      # extension picks format\n"
            "  pipe.reproduce(W, save='run.png')           # per-layer -> run_0.png, ...\n\n"
            ".load() is type-checked (TopoResult.load on a graph file raises).\n"
            "WARNING: .tda files are pickles — load only ones you trust, and treat\n"
            "them as cache (not guaranteed stable across major dependency upgrades)."
        )
        print(text); return text

    if section == "Training trajectories":
        text = (
            "Training-trajectory plots — driven by a TrainingView.\n\n"
            "  plot_training_curves(view)                       # loss & accuracy\n"
            "  plot_weight_norm_trajectory(view)                # ||W_l|| per layer\n"
            "  plot_activation_stats_trajectory(view, layer)    # mean/std/sparsity\n"
            "  plot_id_over_training(view, layer)               # ID over epochs\n"
            "  plot_id_trajectory_all_layers(view)              # ID for all layers\n"
            "  plot_ph_dimension_over_training(view)            # sliding-window PH dim\n"
            "  plot_magnitude_dimension_over_training(view)     # sliding-window mag dim\n"
            "  plot_ph_statistic_trajectory(view, stat='...')   # any PH stat\n"
            "  plot_betti_trajectory(view, dim=1)               # Betti_d vs epoch\n"
            "  plot_diagram_distance_trajectory(view)           # dist to reference dgm\n"
            "  plot_diagram_distance_matrix(view)               # pairwise heatmap\n"
            "  plot_ph_statistic_pairplot(view, stats=...)      # scatter matrix\n"
            "  plot_diagram_evolution(view, n_panels=6)         # diagram small-multiples\n\n"
            "Dashboards & animations:\n"
            "  plot_training_summary(view)                      # 2x3 panel summary\n"
            "  make_training_animation(view, out='train.gif')   # animated PH\n"
            "  plot_pipeline_diagram(pipe)                      # flowchart of a pipeline"
        )
        print(text); return text

    if section == "Topological uncertainty":
        text = (
            "Topological uncertainty (Lacombe et al. 2021).\n\n"
            "  from tanc.topo_tools import TopologicalUncertainty\n"
            "  tu = TopologicalUncertainty(weight_matrices, n_classes=10)\n"
            "  tu.fit(train_activations_per_layer, train_predictions)\n"
            "  scores = tu.score(test_activations_per_layer, test_predictions)\n\n"
            "Visualise with:\n"
            "  plot_tu_score_distribution(scores, y_true, y_pred)"
        )
        print(text); return text

    return ""