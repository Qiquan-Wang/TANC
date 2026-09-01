# tanc

Topological Data Analysis tools for neural networks.

This package provides a composable pipeline for extracting model data, building graphs, computing topological summaries, and visualising results.

## Package overview

### Core modules

- `tanc.pipeline`
  - `TDAPipeline` — orchestrates graph building, topological analysis, and plotting.
  - `TDAPipeline.from_paper(key)` — preconfigured presets for 18 published methods.
  - `TDAPipeline.over_training(view, measures, layers=...)` — track scalar measure(s) across a training trajectory, per layer, as a `TrajectorySeries`.
  - `check_compatibility()` — validates builder/tool combinations and injects required defaults.
  - `PAPER_PRESETS` — single source of truth for preset configurations.

- `tanc.model_extractor`
  - `inspect(model)` — inspect model architecture and discover layer names.
  - `extract_model(model, data, ...)` — extract weights, activations, and classifications from a trained model into a `ModelSnapshot`. `activation_pooling="last"/"mean"/…` pools transformer `(batch, seq, hidden)` activations over the sequence axis.
  - `extract_training(model, extract_data, ...)` — train a model while capturing snapshots into a `TrainingView`.
  - `ModelSnapshot` / `TrainingView` — containers for extracted model state and trajectory data.

- `tanc.graph_builder`
  - `build_weight_graph()` — weighted similarity graphs from weight matrices or coupled weight+activation pairs.
  - `build_point_cloud_graph()` — a single weight matrix as a point cloud of its rows or columns (`orientation="rows"|"cols"`); the complement of the neuron graph.
  - `build_activation_graph()` — activation-space distance graphs with Euclidean, geodesic, correlation, or VNE metrics.
  - `build_labelled_complex_graph()` — label-masked distance graphs for boundary topology.
  - `build_polyhedral_graph()` — Hamming-distance graphs over binary activation patterns.
  - `build_kernel_graph()` — filter-distance graphs for convolutional kernels.
  - `build_weight_trajectory()` — trajectory graphs for training dynamics and loss-based distances.
  - `compute_node_features()` — synthetic node embeddings for Mapper and graph analysis.
  - `GraphBundle` — shared container output from all graph builders.

- `tanc.topo_tools`
  - `run_ph()` — compute persistent homology and summary statistics.
  - `run_mapper()` — build Mapper graphs from node features via giotto-tda.
  - `apply_filter()` — compute Mapper lens values.
  - `mapper_pipeline()` — run the low-level Mapper pipeline and return graph outputs.
  - `plot_mapper_graph()` — draw Mapper graphs.
  - `run_activation_id()` / `estimate_id_across_layers()` — activation intrinsic dimensionality estimation.
  - `run_trajectory_dimension()` — dimension estimation on weight trajectories (`ph_dimension`, `magnitude_dimension`).
  - `WeightTrajectoryAnalyser` — helper for PH/magnitude trajectory analyses.
  - `TopoResult`, `PersistenceResult` — unified output containers for all tools.

- `tanc.visualisation`
  - `plot_persistence_diagram()` — plot persistence diagrams.
  - `plot_barcode()` — plot barcodes.
  - `plot_betti_curve()` — plot Betti curves.
  - `plot_diagram_comparison()` — side-by-side comparison plots.
  - `pipeline_trajectory()` / `TrajectorySeries` — run any pipeline over a `TrainingView` and track scalar measure(s) per epoch, optionally per layer.
  - `make_figure()`, `annotate_ph_stats()`, `format_paper_reference()` — plotting utilities.

## Workflow

1. **Extract model data**
   - Use `extract_model()` for a trained model.
   - Use `extract_training()` for snapshot-based training analysis.

2. **Build a graph**
   - Use a graph builder from `tanc.graph_builder` to convert weights, activations, or labels into a `GraphBundle`.

3. **Run topology tools**
   - Use `run_ph()`, `run_mapper()`, or `run_activation_id()` / `run_trajectory_dimension()`.

4. **Visualise results**
   - Plot directly from `TopoResult` or use `tanc.visualisation` helpers.

## Quick start

```python
from tanc import TDAPipeline

pipe = TDAPipeline.from_paper("watanabe2021")
result = pipe.fit(weight_matrices)
result.plot("diagram")
```

### Full pipeline from a trained model

```python
from tanc.model_extractor import extract_model
from tanc import TDAPipeline

snapshot = extract_model(my_model, X_test)
pipe = TDAPipeline.from_paper("naitzat2020")
results = pipe.fit(snapshot.all_activation_matrices())
```

## Top-level exports

The package re-exports several core types at `tanc`:

- `TDAPipeline`
- `PAPER_PRESETS`
- `GraphBundle`
- `TopoResult`
- `PersistenceResult`
- `ModelSnapshot`
- `TrainingView`
- `ModelInfo`
- `LayerInfo`

## Compatibility rules

The pipeline enforces compatibility before execution:

- `mapper` on `weight_graph` auto-injects node features if missing.
- `activation_graph` + `ph` + `distance="correlation"` warns about neuron-neuron distance semantics.
- `weight_graph` with `induced_paths=True` requires `graph_scope="full"`.
- `mapper` requires `giotto-tda`.

## File structure

```
tanc/
├── __init__.py
├── pipeline/
├── graph_builder/
├── model_extractor/
├── topo_tools/
└── visualisation/
```
