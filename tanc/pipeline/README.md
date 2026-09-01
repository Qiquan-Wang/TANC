# pipeline

Orchestrator module that wires `graph_builder` (Module 1, the **graph construction**) to `topo_tools` (Module 2, the **method**) into a single callable pipeline. The point is to **mix and match**: choose a space to study, a `builder`, a `tool`, and an output (see the four axes in `docs/composing.rst`). Reproducing a published method via `TDAPipeline.from_paper()` is a **shortcut** that pre-fills those choices — 18 presets ship, but they are convenience coordinates, not the core feature.

## Data flow

```
Raw data
   │
   ▼
TDAPipeline
   ├── Module 1: graph_builder → GraphBundle
   │         ↓
   ├── Module 2: topo_tools   → TopoResult
   │         ↓
   └── Visualisation           → Figure
```

## Quick start

```python
from tanc import TDAPipeline

# Compose your own: pick a construction (builder) + a method (tool), fit a space.
pipe = TDAPipeline(builder="activation_graph", tool="ph", tool_kwargs={"max_dim": 1})
result = pipe.fit(activation_matrix)
result.plot("diagram")

# Activation-based per-layer: returns one TopoResult per layer
pipe = TDAPipeline(builder="activation_graph", tool="ph")
results = pipe.fit(per_layer_activations)  # list[TopoResult]

# Compare models side-by-side
pipe.compare({"ModelA": data_a, "ModelB": data_b}, kind="diagram")

# Shortcut: a preset pre-fills builder/tool/kwargs for a published method.
pipe = TDAPipeline.from_paper("watanabe2021")
result = pipe.fit(weight_matrices)
result.plot("diagram")
```

## TDAPipeline

### Construction

```python
# From scratch
pipe = TDAPipeline(
    builder="weight_graph",
    builder_kwargs={"edge_weight": "normalized", "graph_scope": "full"},
    tool="ph",
    tool_kwargs={"max_dim": 1, "backend": "ripser"},
)

# From a paper preset
pipe = TDAPipeline.from_paper("rieck2019")

# Override preset parameters
pipe = TDAPipeline.from_paper(
    "watanabe2021",
    tool_kwargs__max_dim=2,
    builder_kwargs__edge_weight="absolute",
)
```

### Parameters

| Parameter | Description |
|---|---|
| `builder` | `"weight_graph"`, `"activation_graph"`, `"labelled_complex_graph"`, `"polyhedral_graph"`, `"weight_trajectory"`, `None`. If `None`, Module 1 is skipped and the pipeline runs a dimension-based tool directly. |
| `builder_kwargs` | Dict passed to the builder function |
| `tool` | `"ph"`, `"mapper"`, `"dimension"` |
| `tool_kwargs` | Dict passed to the tool function |
| `visualisation` | Default plot kind for `.plot()` |

### Methods

| Method | Description |
|---|---|
| `.fit(data)` | Run the full pipeline on raw arrays, lists, tuples, or model snapshots/training views. Returns `TopoResult` or `list[TopoResult]` (e.g. `activation_graph` + `ph` on a list returns one result per layer). |
| `.fit_model(model, data, ...)` | Extract weights/activations/classifications from a trained model and run the pipeline end-to-end. `activation_pooling="last"/"mean"/…` pools transformer `(batch, seq, hidden)` activations over the sequence axis (see `model_extractor`). |
| `.fit_training(...)` | Train a model, capture snapshots, and run the pipeline on the extracted training view. |
| `.fit_models(models, data, ...)` | Extract the same representation from multiple models, **stack** them into one point cloud, and run the pipeline once. |
| `.fit_each(models, data, ...)` | Run the pipeline **separately** on each model and return `list[TopoResult]` — one per model. Population entry point for per-model summaries (e.g. one ASDSQ vector per net in Ballester 2024). |
| `.over_training(view, measures, layers=None, ...)` | Track scalar measure(s) across a training trajectory: runs this pipeline per snapshot (per layer when `layers=[...]`) and returns a `TrajectorySeries` (numbers + `.plot()`). The reusable form of a hand-rolled per-snapshot `.fit()` loop. |
| `.validate()` | Run compatibility checks early (called automatically by `.fit()`) |
| `.plot(result, kind)` | Convenience wrapper for `result.plot(kind)` |
| `.compare(data_dict, kind)` | Fit on each entry and plot side-by-side |

### Per-model summaries over a population (`fit_each`)

`fit_each` runs the whole preset independently for every model and returns one
`TopoResult` per net — each carrying its own `.statistics`. This is how the
Ballester (2024) reproduction turns a population of trained networks into a feature
matrix (one ASDSQ vector per model) for the downstream gap regression:

```python
pipe = TDAPipeline.from_paper("ballester2024", builder_kwargs__max_neurons=300)
results = pipe.fit_each(models, X_eval,
                        representation="activations",
                        layer_selection="linear_and_conv")
S = np.array([[r.statistics[k] for k in asdsq_keys] for r in results])  # (n_models, 16)
```

## Paper presets

`TDAPipeline.from_paper(key)` configures the pipeline to reproduce a published method. Available presets:

### Architecture (weight) papers

| Key | Paper | Builder | Tool | Input |
|---|---|---|---|---|
| `watanabe2021` | Watanabe & Yamana (2020) | `weight_graph` (relevance) | `ph` (**directed clique complex**, path-product integer filtration → H1 belt; representative cycles for PHPM pruning) | `list[weight_matrix]` (the FCN head) |
| `rieck2019` | Rieck et al. (2019) — Neural Persistence | `weight_graph` (bipartite) | `ph` (H0) | `list[weight_matrix]` |
| `gebhart2019` | Gebhart et al. (2019) | `weight_graph` (weighted activation) | `ph` (H0) | `list[(weight, activation)]` |
| `lacombe2021` | Lacombe et al. (2021) — Topological Uncertainty | `weight_graph` (weighted activation + layer subset) | `ph` (H0) | `list[(weight, activation)]` |

### Activation papers

| Key | Paper | Builder | Tool | Input |
|---|---|---|---|---|
| `naitzat2020` | Naitzat et al. (2020) | `activation_graph` (geodesic, auto-k) | `ph` (H0+H1) | `list[activation_matrix]` |
| `karuppiah2025` | Karuppiah et al. (2025) | `activation_graph` (euclidean) | `ph` (H0+H1) | `list[activation_matrix]` |
| `ballester2024` | Ballester et al. (2024) | `activation_graph` (correlation, drop-constant + importance neuron sampling) | `ph` (H0+H1, **ASDSQ** statistics) | `activation_matrix` |

### Boundary papers

| Key | Paper | Builder | Tool | Input |
|---|---|---|---|---|
| `ramamurthy2019` | Ramamurthy et al. (2019) | `labelled_complex_graph` | `ph` (H0+H1) | `(points, labels)` |
| `liu2023` | Liu et al. (2023) | `polyhedral_graph` | `ph` (H0+H1) | `activation_patterns` |

### Mapper papers

| Key | Paper | Builder | Tool | Input |
|---|---|---|---|---|
| `rathore2021` | Rathore et al. (2021) — TopoAct | `activation_graph` | `mapper` (L2-norm, 70 intervals) | `list[activation_matrix]` |
| `zhou2023` | Zhou et al. (2023) — Comparing Mapper Graphs | `activation_graph` | `mapper` (L2-norm, 40 intervals) | `list[activation_matrix]` |
| `gabrielsson2019` | Gabrielsson & Carlsson (2019) | `activation_graph` | `mapper` (PCA 2D, VNE metric) | `kernel_matrix` |
| `gabella2021` | Gabella (2021) | `activation_graph` | `mapper` (L2-norm primary) | `weight_trajectory` |

### Dimension estimation papers

| Key | Paper | Method | Input |
|---|---|---|---|
| `ruppik2025` | Ruppik et al. (2025) | Local 2NN | `list[activation_matrix]` |
| `ong2026` | Ong et al. (2026) | Calibrated (k, j) 2NN (`method="global"` for the uncalibrated asymptotic estimator) | `list[activation_matrix]` |
| `birdal2021` | Birdal et al. (2021) | PH fractal dimension | `weight_trajectory` |
| `dupuis2023` | Dupuis et al. (2023) | Loss-PH dimension | `weight_trajectory + loss` |
| `andreeva2024` | Andreeva et al. (2024) | Magnitude dimension | `weight_trajectory` |

## Compatibility checking

`compatibility.py` validates builder + tool combinations before running:

1. **Mapper needs node features** — if using `tool="mapper"` with `builder="weight_graph"`, the pipeline automatically sets `builder_kwargs["node_feature_fn"] = "laplacian_eigenvectors"` and emits a warning unless node features are provided explicitly.
2. **Correlation + PH** — if using `tool="ph"` with `builder="activation_graph"` and `distance="correlation"`, the pipeline warns that PH is computed on neuron-neuron distances, not sample-sample distances.
3. **Induced paths require full scope** — `builder="weight_graph"` with `induced_paths=True` must also use `graph_scope="full"`; otherwise a `ValueError` is raised.
4. **Mapper requires giotto-tda** — `tool="mapper"` raises `ImportError` if the `giotto-tda` dependency is not installed, with a pip install hint.

## File structure

```
pipeline/
├── __init__.py          # Re-exports TDAPipeline and PAPER_PRESETS
├── pipeline.py          # TDAPipeline class
├── paper_presets.py     # PAPER_PRESETS dict (all 18 presets)
└── compatibility.py     # Cross-module validation rules
```
