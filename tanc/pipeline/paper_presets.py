"""paper_presets.py — single source of truth for all .from_paper() configurations.

Every paper preset is a plain dict consumed by both
``TDAPipeline.from_paper()`` and individual module ``.from_paper()`` methods.
"""

from __future__ import annotations

PAPER_PRESETS: dict[str, dict] = {

    # ── Architecture papers ──────────────────────────────────────────────────
    "watanabe2021": {
        "description": "Watanabe & Yamana (2020) — Topological Measurement / Pruning of Deep Neural Networks Using Persistent Homology",
        "builder":         "weight_graph",
        "builder_kwargs":  {
            "edge_weight":     "relevance",
            "graph_scope":     "full",
            "induced_paths":   False,   # the clique complex builds path edges itself
            "node_feature_fn": None,
        },
        "tool":            "ph",
        "tool_kwargs":     {
            "max_dim":        1,
            "input_complex":  "directed_clique",
            "relevance_mode": "positive",
        },
        "visualisation":   "diagram",
        "paper_reference": "Watanabe & Yamana (2020)",
        "input_format":    "list[weight_matrix]",
        "notes": (
            "Faithful **directed clique complex** of the FCN (GUDHI), NOT a "
            "flag complex on a similarity matrix.  Relevance R_ij = w_ij / Σ_k "
            "w_kj (column-normalised; positive part here, |w| for PHPM via "
            "tool_kwargs__relevance_mode='absolute').  Direct edges, induced "
            "edges (max 2-hop path product) and triangles (path product) each "
            "take the integer threshold-index (1..64) of the paper's descending "
            "relevance schedule as their filtration value.  The H1 diagram "
            "reproduces the paper's wide 'belt' above the diagonal, and "
            "ph_result.metadata['watanabe'] carries the representative simplices "
            "used by tanc.applications.pruning (PHPM)."
        ),
    },

    "rieck2019": {
        "description": "Rieck et al. (2019) — Neural Persistence: A Complexity Measure for Deep Neural Networks Using Algebraic Topology",
        "builder":         "weight_graph",
        "builder_kwargs":  {
            "edge_weight":     "normalized",
            "graph_scope":     "bipartite",
            "induced_paths":   False,
            "node_feature_fn": "laplacian_eigenvectors",
        },
        "tool":            "ph",
        "tool_kwargs":     {"max_dim": 0, "backend": "ripser"},
        "visualisation":   "diagram",
        "paper_reference": "Rieck et al. (2019)",
        "input_format":    "list[weight_matrix]",
        "notes": (
            "Bipartite graph, one layer at a time.  H0 persistence norm "
            "(p-norm of bar lifetimes) is the Neural Persistence metric.  "
            "edge_weight='normalized' rescales **each layer's** |w| to [0,1] "
            "(per-layer max), matching the paper's per-layer normalisation — "
            "this is what makes Neural Persistence rise monotonically during "
            "training (their Fig. 4).  Use builder_kwargs__edge_weight="
            "'global_normalized' for a single network-wide constant instead."
        ),
    },

    "gebhart2019": {
        "description": "Gebhart et al. (2019) — Characterizing the Shape of Activation Space in Deep Neural Networks",
        "builder":         "weight_graph",
        "builder_kwargs":  {
            "edge_weight":     "weighted_activation",
            "graph_scope":     "multipartite",
            "induced_paths":   False,
            "node_feature_fn": "laplacian_eigenvectors",
        },
        "tool":            "ph",
        "tool_kwargs":     {"max_dim": 0, "backend": "ripser"},
        "visualisation":   "diagram",
        "paper_reference": "Gebhart et al. (2019)",
        "input_format":    "list[tuple[weight_matrix, activation_matrix]]",
        "notes": (
            "Activation-weighted graph.  Long-lifetime H0 generators "
            "correspond to prominent signal pathways."
        ),
    },

    "lacombe2021": {
        "description": "Lacombe et al. (2021) — Topological Uncertainty: Monitoring Trained Neural Networks Through Persistence of Activation Graphs",
        "builder":         "weight_graph",
        "builder_kwargs":  {
            "edge_weight":     "weighted_activation",
            "graph_scope":     "bipartite",
            "induced_paths":   False,
            "layer_subset":    None,      # user sets this
            "node_feature_fn": "laplacian_eigenvectors",
        },
        "tool":            "ph",
        "tool_kwargs":     {"max_dim": 0, "backend": "ripser"},
        "visualisation":   "diagram",
        "paper_reference": "Lacombe et al. (2021)",
        "input_format":    "list[tuple[weight_matrix, activation_matrix]]",
        "notes": (
            "One bipartite activation graph is built per layer independently "
            "(graph_scope='bipartite'), yielding one persistence diagram per "
            "layer.  Pass layer_subset via builder_kwargs override to restrict "
            "to specific layers.  The user selects which layer's diagram to "
            "interpret for uncertainty monitoring."
        ),
    },

    # ── Activation papers ────────────────────────────────────────────────────
    "naitzat2020": {
        "description": "Naitzat et al. (2020) — Topology of Deep Neural Networks",
        "builder":         "activation_graph",
        "builder_kwargs":  {"distance": "geodesic", "k": None},
        "tool":            "ph",
        "tool_kwargs":     {"max_dim": 1, "backend": "ripser"},
        "visualisation":   "diagram",
        "paper_reference": "Naitzat et al. (2020)",
        "input_format":    "list[activation_matrix]",
        "notes": (
            "Graph-geodesic distance (δ_k) on the k-NN graph: Euclidean "
            "distances are used only to form k nearest neighbours; thereafter "
            "every edge has unit length regardless of Euclidean length. "
            "This normalises densities across layers and allows PH to be "
            "compared at the same scale.  k is selected automatically via "
            "betti_stability criterion.  Run per layer; PH tracks Betti "
            "number decrease layer by layer."
        ),
    },

    "karuppiah2025": {
        "description": "Karuppiah et al. (2025) — Topological Data Analysis (TDA) as a Framework for Understanding Deep Learning Behavior",
        "builder":         "activation_graph",
        "builder_kwargs":  {"distance": "euclidean"},
        "tool":            "ph",
        "tool_kwargs":     {"max_dim": 1, "backend": "ripser"},
        "visualisation":   "diagram",
        "paper_reference": "Karuppiah et al. (2025)",
        "input_format":    "list[activation_matrix]",
        "notes":           "Direct VR on Euclidean activation distances, per layer.",
    },

    "ballester2024": {
        "description": "Ballester et al. (2024) — Predicting the Generalization Gap in Neural Networks Using Topological Data Analysis",
        "builder":         "activation_graph",
        "builder_kwargs":  {
            "distance":       "correlation",
            "drop_constant":  True,
            "node_sampling":  "importance",
            "max_neurons":    300,
        },
        "tool":            "ph",
        "tool_kwargs":     {"max_dim": 1, "backend": "ripser", "stat_names": ["asdsq"]},
        "visualisation":   "diagram",
        "paper_reference": "Ballester et al. (2024)",
        "input_format":    "single_activation_matrix",
        "notes": (
            "Correlation distance on neurons (columns).  Graph is "
            "(N_neurons × N_neurons).  Constant neurons are dropped and the "
            "node set is importance-sampled to max_neurons (Eqs. 4.2-4.3); "
            "raise builder_kwargs__max_neurons toward the paper's 3000 for "
            "full scale.  result.statistics carries the ASDSQ summary "
            "(H{0,1}_{mean,std}_{birth,death}[_sq])."
        ),
    },

    # ── Boundary papers ──────────────────────────────────────────────────────
    "ramamurthy2019": {
        "description": "Ramamurthy, Varshney & Mody (2019) — Topological Data Analysis of Decision Boundaries with Application to Model Selection",
        "builder":         "labelled_complex_graph",
        "builder_kwargs":  {"distance_metric": "euclidean"},
        "tool":            "ph",
        "tool_kwargs":     {"max_dim": 1, "backend": "ripser"},
        "visualisation":   "diagram",
        "paper_reference": "Ramamurthy et al. (2019)",
        "input_format":    "points_and_labels",
        "notes": (
            "Labelled Vietoris-Rips complex: vertices S = one class "
            "(source_class), reference W = the rest.  Same-class S vertices are "
            "kept only when within gamma of W (near the decision boundary; gamma "
            "defaults to the median nearest-W distance), and VR PH is run on the "
            "distances among those eligible vertices.  builder_kwargs: "
            "source_class, gamma / gamma_quantile, scale='global'|'local' "
            "(locally-scaled labelled VR via kNN radii)."
        ),
    },

    "liu2023": {
        "description": "Liu, Cole, Peterson & Kirby (2023) — ReLU Neural Networks, Polyhedral Decompositions, and Persistent Homology",
        "builder":         "polyhedral_graph",
        "builder_kwargs":  {"input_type": "auto"},
        "tool":            "ph",
        "tool_kwargs":     {"max_dim": 1, "backend": "ripser"},
        "visualisation":   "diagram",
        "paper_reference": "Liu et al. (2023)",
        "input_format":    "activation_patterns",
        "notes": (
            "Hamming distance on unique binary activation patterns.  "
            "Each node = one polyhedral region of the input space partition."
        ),
    },

    # ── Mapper papers ─────────────────────────────────────────────────────────
    "rathore2021": {
        "description": "Rathore et al. (2021) — TopoAct: Visually Exploring the Shape of Activations in Deep Learning",
        "builder":         "activation_graph",
        "builder_kwargs":  {"distance": "euclidean"},
        "tool":            "mapper",
        "tool_kwargs":     {
            "filter_fn":    "l2_norm",
            "n_intervals":  70,
            "overlap_frac": 0.3,
        },
        "visualisation":   "graph",
        "paper_reference": "Rathore et al. (2021)",
        "input_format":    "list[activation_matrix]",
        "notes": (
            "L2-norm of activation vectors as filter (not PCA).  "
            "For Mapper, the activation matrix is used directly as the "
            "point cloud X (node_features).  n_intervals=70 and overlap=30% "
            "match the primary experimental parameters; the paper also tests "
            "20% and 50% overlap."
        ),
    },

    "zhou2023": {
        "description": "Zhou et al. (2023) — Comparing Mapper Graphs of Artificial Neuron Activations",
        "builder":         "activation_graph",
        "builder_kwargs":  {"distance": "euclidean"},
        "tool":            "mapper",
        "tool_kwargs":     {
            "filter_fn":    "l2_norm",
            "n_intervals":  40,
            "overlap_frac": 0.2,
        },
        "visualisation":   "graph",
        "paper_reference": "Zhou et al. (2023)",
        "input_format":    "list[activation_matrix]",
        "notes": (
            "L2-norm filter (reflects how strongly the network reacts to an "
            "input).  Primary parameters: n=40 intervals, p=20% overlap, "
            "DBSCAN clustering.  The preset builds one Mapper; the paper's main "
            "contribution — a Gromov-Wasserstein distance between Mapper "
            "hypergraphs (nodes as sample-membership hyperedges, cluster-size "
            "masses) for comparing activation topology across layers — is "
            "available as tanc.topo_tools.mapper_hypergraph_gw_distance("
            "result_a, result_b).  (The plain-graph mapper_gw_distance is only a "
            "partial proxy.)"
        ),
    },

    "gabrielsson2019": {
        "description": "Gabrielsson & Carlsson (2019) — Exposition and Interpretation of the Topology of Neural Networks",
        "builder":         "activation_graph",
        "builder_kwargs":  {"distance": "vne"},
        "tool":            "mapper",
        "tool_kwargs":     {
            "filter_fn":    "pca",
            "n_components": 2,
            "n_intervals":  30,
            "overlap_frac": 0.667,
        },
        "visualisation":   "graph",
        "paper_reference": "Gabrielsson & Carlsson (2019)",
        "input_format":    "kernel_matrix",
        "notes": (
            "Input: flattened conv kernels from multiple models.  "
            "data shape = (n_models × n_kernels_per_layer, kernel_dim).  "
            "Mapper uses Variance Normalised Euclidean (VNE) metric.  "
            "Cover parameters: Mapper(30, 3) → n_intervals=30, gain g=3 "
            "→ overlap_frac = 1 - 1/g ≈ 0.667.  "
            "Mapper finds structural clusters across model families.  "
            "Note: this analysis requires training multiple model instances "
            "and stacking their kernels into a single point cloud before "
            "running Mapper."
        ),
    },

    "gabella2021": {
        "description": "Gabella (2021) — Topology of Learning in Artificial Neural Networks",
        "builder":         "activation_graph",
        "builder_kwargs":  {"distance": "euclidean"},
        "tool":            "mapper",
        "tool_kwargs":     {
            "filter_fn":    "l2_norm",
            "n_intervals":  10,
            "overlap_frac": 0.3,
        },
        "visualisation":   "graph",
        "paper_reference": "Gabella (2021)",
        "input_format":    "weight_trajectory",
        "notes": (
            "data shape = (n_steps × N_l, N_{l-1}).  "
            "Row i*N_l + j = incoming weights of neuron j at step i.  "
            "Primary filter: L2-norm (branches grow from origin → tree "
            "topology).  For the secondary surface analysis, override with "
            "filter_fn='pca', n_components=3 to expose the 2D learning "
            "surface as a densely connected grid."
        ),
    },

    # ── Dimension estimation papers ──────────────────────────────────────────
    "ruppik2025": {
        "description": "Ruppik et al. (2025) — Less is More: Local Intrinsic Dimensions of Contextual Language Models",
        "builder":         None,
        "builder_kwargs":  {},
        "tool":            "dimension",
        "tool_kwargs":     {"method": "local", "estimator": "activation_id"},
        "visualisation":   "id_layers",
        "paper_reference": "Ruppik et al. (2025)",
        "input_format":    "list[activation_matrix]",
        "notes":           "Local 2NN on subsampled neighbourhoods per layer.",
    },

    "ong2026": {
        "description": "Ong et al. (2026) — A Universal Nearest-Neighbor Estimator for Intrinsic Dimensionality",
        "builder":         None,
        "builder_kwargs":  {},
        "tool":            "dimension",
        "tool_kwargs":     {"method": "calibrated", "estimator": "activation_id"},
        "visualisation":   "id_layers",
        "paper_reference": "Ong et al. (2026)",
        "input_format":    "list[activation_matrix]",
        "notes":           ("Calibrated (k, j) nearest-neighbour estimator: per-point "
                            "L_{k,j} = -log log(R_k/R_j) averaged, then the "
                            "sample-size-calibrated map d = exp(alpha*Lbar + beta), with "
                            "(alpha, beta) fitted against Gaussian clouds of known "
                            "dimension.  Defaults k=2, j=1.  Pass method='global' for the "
                            "uncalibrated asymptotic 2NN estimator."),
    },

    "birdal2021": {
        "description": "Birdal et al. (2021) — Intrinsic Dimension, Persistent Homology and Generalization in Neural Networks",
        "builder":         None,
        "builder_kwargs":  {},
        "tool":            "dimension",
        "tool_kwargs":     {
            "method":    "ph_euclidean",
            "estimator": "trajectory_dimension",
            "alpha":     1.0,
        },
        "visualisation":   "ph_scaling",
        "paper_reference": "Birdal et al. (2021)",
        "input_format":    "weight_trajectory",
        "notes":           "VR filtration on weight trajectory; α=1 lifetime scaling.",
    },

    "dupuis2023": {
        "description": "Dupuis et al. (2023) — Generalization Bounds using Data-Dependent Fractal Dimensions",
        "builder":         None,
        "builder_kwargs":  {},
        "tool":            "dimension",
        "tool_kwargs":     {
            "method":    "ph_loss",
            "estimator": "trajectory_dimension",
            "alpha":     1.0,
        },
        "visualisation":   "ph_scaling",
        "paper_reference": "Dupuis et al. (2023)",
        "input_format":    "weight_trajectory_with_loss",
        "notes":           ("Data-dependent pseudo-metric on the weight trajectory: "
                            "rho(w_i, w_j) = mean_s |ell(w_i, z_s) - ell(w_j, z_s)| "
                            "(mean over samples of the absolute per-sample loss "
                            "difference), then PH-dimension of that distance matrix.  "
                            "Give the TrainingExtractor loss_eval_data=(X, y) so per-"
                            "sample losses are captured; without it the metric degrades "
                            "to |mean-loss_i - mean-loss_j| (a warning is emitted)."),
    },

    "andreeva2024": {
        "description": "Andreeva et al. (2024) — Metric Space Magnitude and Generalisation in Neural Networks",
        "builder":         None,
        "builder_kwargs":  {},
        "tool":            "dimension",
        "tool_kwargs":     {
            "method":    "magnitude",
            "estimator": "trajectory_dimension",
        },
        "visualisation":   "magnitude_scaling",
        "paper_reference": "Andreeva et al. (2024)",
        "input_format":    "weight_trajectory",
        "notes": (
            "Magnitude function Mag(tW) computed on a sliding window of "
            "training trajectory points (W = {w_i}).  Magnitude dimension = "
            "slope of log-log plot of Mag(tW) vs t, estimating the fractal "
            "dimension of the optimisation trajectory.  Lower dimension "
            "correlates with better generalisation."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-tool defaults for expected_plots / key_statistics.
# Individual presets may override either field; otherwise these defaults apply.
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_DEFAULTS = {
    "ph": {
        "expected_plots": ["diagram", "barcode", "betti_curve"],
        "key_statistics": [
            "H0_total_persistence", "H0_persistence_norm", "H0_persistence_entropy",
            "H1_total_persistence", "H1_persistence_norm", "H1_persistence_entropy",
        ],
    },
    "mapper": {
        "expected_plots": ["graph"],
        "key_statistics": ["n_nodes", "n_edges", "density", "n_connected_components"],
    },
    "dimension": {
        "expected_plots": ["id_layers", "ph_scaling", "magnitude_scaling"],
        "key_statistics": ["id_estimate", "ph_dimension", "magnitude_dimension"],
    },
}

# Per-preset overrides where the defaults don't match what the paper highlights.
_PRESET_OVERRIDES: dict[str, dict] = {
    "watanabe2021": {
        "expected_plots":   ["diagram"],
        "key_statistics":   ["H1_total_persistence", "H1_persistence_entropy"],
    },
    "rieck2019": {
        "expected_plots":   ["diagram", "barcode"],
        "key_statistics":   ["H0_persistence_norm"],  # = Neural Persistence
    },
    "lacombe2021": {
        "expected_plots":   ["diagram", "barcode"],
        "key_statistics":   ["H0_total_persistence", "H0_persistence_norm"],
    },
    "naitzat2020": {
        "expected_plots":   ["diagram", "betti_curve"],
        "key_statistics":   ["H0_total_persistence", "H1_total_persistence"],
    },
    "rathore2021":   {"expected_plots": ["graph"]},
    "zhou2023":      {"expected_plots": ["graph"]},
    "gabrielsson2019": {"expected_plots": ["graph"]},
    "gabella2021":   {"expected_plots": ["graph"]},
    "ruppik2025":    {"expected_plots": ["id_layers"], "key_statistics": ["id_estimate"]},
    "ong2026":       {"expected_plots": ["id_layers"], "key_statistics": ["id_estimate"]},
    "birdal2021":    {"expected_plots": ["ph_scaling"], "key_statistics": ["ph_dimension"]},
    "dupuis2023":    {"expected_plots": ["ph_scaling"], "key_statistics": ["ph_dimension"]},
    "andreeva2024":  {"expected_plots": ["magnitude_scaling"], "key_statistics": ["magnitude_dimension"]},
}

# Apply defaults + overrides exactly once at module import.
for _name, _preset in PAPER_PRESETS.items():
    _defaults = _TOOL_DEFAULTS.get(_preset["tool"], {})
    _preset.setdefault("expected_plots", list(_defaults.get("expected_plots", [])))
    _preset.setdefault("key_statistics", list(_defaults.get("key_statistics", [])))
    for _k, _v in _PRESET_OVERRIDES.get(_name, {}).items():
        _preset[_k] = _v


def describe_preset(name: str, print_it: bool = True) -> str:
    """Return (and by default print) a human-readable summary of one preset.

    Parameters
    ----------
    name : str
        Preset key, e.g. ``"watanabe2021"``.
    print_it : bool
        When ``True`` (default) the summary is also printed.
    """
    if name not in PAPER_PRESETS:
        valid = sorted(PAPER_PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{name}'. Valid:\n  " + "\n  ".join(valid)
        )

    p = PAPER_PRESETS[name]
    lines = [
        f"{name} — {p.get('paper_reference', '')}",
        f"  description : {p['description']}",
        f"  pipeline    : builder={p['builder']!r}  →  tool={p['tool']!r}",
        f"  input_format: {p.get('input_format', '?')}",
        f"  default plot: {p.get('visualisation', '(none)')}",
        f"  expected    : {', '.join(p.get('expected_plots', []) or ['(none)'])}",
        f"  key stats   : {', '.join(p.get('key_statistics', []) or ['(none)'])}",
    ]
    if p.get("notes"):
        # Wrap notes to ~80 chars
        import textwrap
        wrapped = textwrap.fill(p["notes"], width=78,
                                initial_indent="  notes       : ",
                                subsequent_indent="                ")
        lines.append(wrapped)
    text = "\n".join(lines)
    if print_it:
        print(text)
    return text


def list_presets(print_it: bool = True) -> list[str]:
    """List all preset names and one-line descriptions.

    Returns the list of names (regardless of ``print_it``).
    """
    names = sorted(PAPER_PRESETS.keys())
    if print_it:
        # Group by tool for readability.
        by_tool: dict[str, list[str]] = {}
        for n in names:
            t = PAPER_PRESETS[n]["tool"]
            by_tool.setdefault(t, []).append(n)
        for tool in ("ph", "mapper", "dimension"):
            if tool not in by_tool:
                continue
            print(f"[{tool}]")
            for n in by_tool[tool]:
                ref = PAPER_PRESETS[n].get("paper_reference", "")
                print(f"  {n:18s}  {ref}")
            print()
    return names
