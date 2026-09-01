# visualisation

Plotting utilities for TANC. Covers persistence diagrams, barcodes,
Betti curves, training-dynamics curves, topological summaries over
training, intrinsic-dimension trajectories, graph/embedding views, and
topological-uncertainty score distributions.

## Data flow

```
TopoResult
   │
   ├── result.plot("diagram")     → persistence diagram figure
   ├── result.plot("barcode")     → barcode figure
   ├── result.plot("betti_curve") → Betti curve figure
   │
   └── (Mapper / dimension plots are handled in their respective topo_tools files)

TrainingView
   │
   ├── plot_training_curves(view)                  → loss & accuracy
   ├── plot_weight_norm_trajectory(view)           → ‖W_l‖ over epochs
   ├── plot_activation_stats_trajectory(view, l)   → activation mean/std/…
   ├── plot_id_over_training(view, l)              → intrinsic dim over epochs
   ├── plot_id_trajectory_all_layers(view)         → ID across all layers
   ├── plot_ph_dimension_over_training(view)       → sliding-window PH dim
   ├── plot_magnitude_dimension_over_training(view)→ sliding-window mag dim
   ├── plot_ph_statistic_trajectory(view, stat)    → PH stat over epochs
   ├── plot_betti_trajectory(view, dim)            → Betti_d over epochs
   ├── plot_diagram_distance_trajectory(view)      → distance to reference dgm
   ├── pipeline_trajectory(view, pipe, measures)   → any pipeline's scalar(s), per layer
   └── plot_diagram_evolution(view)                → diagram small-multiples

GraphBundle
   │
   ├── plot_graph_matrix(bundle)      → heatmap of distance/sim/adjacency
   └── plot_node_embedding(bundle)    → 2-D MDS / t-SNE coloured by label

TU scores
   │
   └── plot_tu_score_distribution(scores, y_true, y_pred)
```

Most users interact with the persistence plots via `result.plot(kind)` on
a `TopoResult`. Everything else takes a `TrainingView`, `GraphBundle`, or
raw score array directly.

## Persistence representations

### Persistence diagram

Birth vs death scatter plot. Points above the diagonal represent
topological features; distance from the diagonal represents feature
lifetime (significance).

```python
from tanc.visualisation import plot_persistence_diagram

fig = plot_persistence_diagram(
    ph_result,
    dims=[0, 1],
    show_diagonal=True,
    alpha=0.7,
    point_size=20,
    title="My Diagram",
)
```

#### Density (overlap) mode

When a diagram contains many overlapping generators — e.g. the H1 clouds
in Watanabe & Yamana (2021, Figs. 4–6) — a plain scatter wastes ink on
coincident points. Pass `density=True` to render *unique* (birth, death)
pairs as discrete dots, coloured by how many generators share that
coordinate (`log10(1 + count)`).

```python
fig = plot_persistence_diagram(
    ph_result,
    dims=[1],
    density=True,
    point_size=25,
    alpha=0.9,
    cmap="viridis",
)
```

### Barcode

Horizontal bars from birth to death, coloured by homology dimension. Dense
diagrams (e.g. `H0` of a neuron graph has one bar per merge — often hundreds)
are unreadable as a raw barcode, so by default only the **longest `max_bars`
bars** are shown (sorted by lifetime), the figure height is capped, and the
essential (infinite-death) bar is clipped to the data range and marked `>`.

```python
from tanc.visualisation import plot_barcode

fig = plot_barcode(ph_result, dims=[0, 1])               # longest 40 bars (default)
fig = plot_barcode(ph_result, max_bars=None)             # show every bar
fig = plot_barcode(ph_result, max_bars=20, sort_by="birth")
```

### Betti curve

Betti number as a step function of filtration value.

```python
from tanc.visualisation import plot_betti_curve

fig = plot_betti_curve(ph_result, dims=[0, 1], resolution=100)
```

### Side-by-side comparison

Plot the same kind of visualisation for multiple results in a single row
of panels.

```python
from tanc.visualisation import plot_diagram_comparison

fig = plot_diagram_comparison(
    {"Model A": ph_result_a, "Model B": ph_result_b},
    kind="diagram",
    dims=[0, 1],
)
```

## Training-dynamics plots

All driven by a `TrainingView` returned by
`tanc.model_extractor.extract_training`.

### Loss & accuracy

```python
from tanc.visualisation import plot_training_curves

fig = plot_training_curves(view, twin_axes=True)
# twin_axes=False stacks loss and accuracy as separate subplots.
```

### Weight-norm trajectory

Per-layer matrix norm of `W_l` across epochs. `norm` is one of
`"fro" | "l1" | "l2" | "nuc"`.

```python
from tanc.visualisation import plot_weight_norm_trajectory

fig = plot_weight_norm_trajectory(view, layers=["fc1", "fc2"], norm="fro")
```

### Activation statistics over training

Summary statistics of one layer's activations vs epoch. `stats` is any
subset of `("mean", "std", "sparsity", "max")`.

```python
from tanc.visualisation import plot_activation_stats_trajectory

fig = plot_activation_stats_trajectory(view, layer="fc1",
                                       stats=("mean", "std", "sparsity"))
```

## Intrinsic-dimension trajectories

Both functions run an ID estimator (`"global"` 2NN of Ong et al. 2026, or
`"local"` 2NN of Ruppik et al. 2025) on the chosen layer's activations at
every snapshot and plot ID vs epoch.

### Single layer (overlay multiple methods)

```python
from tanc.visualisation import plot_id_over_training

fig = plot_id_over_training(view, layer="fc1",
                            methods=("global", "local"))
```

### All layers

`layout="overlay"` puts every layer on one axes; `layout="grid"`
small-multiples one subplot per layer.

```python
from tanc.visualisation import plot_id_trajectory_all_layers

fig = plot_id_trajectory_all_layers(view, method="global",
                                    layout="overlay")
fig = plot_id_trajectory_all_layers(view, method="global",
                                    layout="grid", ncols=3)
```

## Trajectory dimensions (PH dim, magnitude dim)

Both estimators operate on the *weight trajectory* — a distance matrix
between training snapshots. The plot helpers slide a window of `window`
consecutive snapshots, build a trajectory distance matrix, run the
estimator, and plot dimension vs window-centre epoch.

```python
from tanc.visualisation import (
    plot_ph_dimension_over_training,
    plot_magnitude_dimension_over_training,
)

fig = plot_ph_dimension_over_training(view, window=20, stride=1,
                                      distance="euclidean", alpha=1.0)

fig = plot_magnitude_dimension_over_training(view, window=20,
                                             t_range=(0.1, 10.0),
                                             n_scale=30)
```

Per-snapshot scaling diagnostics (log-log fits) are re-exported from
`topo_tools.dimension_tool`:

```python
from tanc.visualisation import (
    plot_ph_scaling, plot_magnitude_scaling,
    plot_2nn_ratio_distribution, plot_loglog_ratio_distribution,
    plot_id_across_layers,
)
```

## PH summaries over training

Each function rebuilds a `GraphBundle` from every snapshot (default: a
weight graph over all layers; override with `layer=...` or a custom
`builder` callable) and tracks a scalar summary vs epoch.

```python
from tanc.visualisation import (
    plot_ph_statistic_trajectory,
    plot_betti_trajectory,
    plot_diagram_distance_trajectory,
)

# total_persistence / persistence_norm / persistence_entropy / convex_hull_area
fig = plot_ph_statistic_trajectory(view, stat="total_persistence", dim=0)

# Betti_d at a fixed filtration value
fig = plot_betti_trajectory(view, dim=1, epsilon=0.5)

# Wasserstein/bottleneck distance to the final (or initial) diagram
fig = plot_diagram_distance_trajectory(view, ref="final",
                                       metric="wasserstein", dim=1)
```

### Custom graph builder

Per-snapshot functions take a `builder: ModelSnapshot → GraphBundle`
callable so you can swap the default weight graph for an activation
graph, kernel graph, etc.

```python
from tanc.graph_builder import build_activation_graph

def act_bundle(snap):
    return build_activation_graph(snap.activation_matrix("fc2"))

fig = plot_ph_statistic_trajectory(view, stat="persistence_entropy",
                                   dim=0, builder=act_bundle)
```

## Pipeline-driven trajectories (any pipeline, per layer)

The functions above bake in their own graph builder. To instead track **whatever a configured `TDAPipeline` produces** — including a `from_paper(...)` preset — across training, use `pipeline_trajectory` (or the `TDAPipeline.over_training` convenience). It runs the pipeline once per snapshot, pulls one-or-more scalar **measures** off each `TopoResult`, and returns a `TrajectorySeries` (the numbers plus a `.plot()`). Pass `layers=[...]` to get **one line per layer** rather than concatenating layers of different shapes.

```python
from tanc import TDAPipeline

# Track H1 total persistence of two head layers, per layer, over training.
series = TDAPipeline.from_paper("watanabe2021").over_training(
    view,
    measures=["H0_total_persistence", "H1_total_persistence"],
    layers=["fc1", "fc2"],
    representation="weights",
)
fig = series.plot()                 # 1 panel per measure, 1 line per layer
df  = series.to_frame()             # tidy (epoch, measure, series, value) DataFrame
```

A **measure** is a statistics key (`"H1_total_persistence"`), the literal `"dimension"` (the `TopoResult.dimension` scalar — works for activation-ID / PH-dim / magnitude), or a callable `TopoResult -> float`. Equivalent standalone form:

```python
from tanc.visualisation import pipeline_trajectory

series = pipeline_trajectory(view, pipe, measures=["dimension"],
                             layers=["fc1", "fc2"], representation="activations")
```

`series.plot(layout="grid" | "overlay")` — `grid` gives one subplot per measure (default for multiple), `overlay` puts them on one axis. Single measure → one axis, one line per layer.

## Diagram evolution (small-multiples)

Compute PH at evenly-spaced epoch checkpoints and plot each as its own
panel.

```python
from tanc.visualisation import plot_diagram_evolution

fig = plot_diagram_evolution(view, n_panels=6, kind="diagram",
                             dims=[0, 1], ncols=3)
fig = plot_diagram_evolution(view, n_panels=4, kind="barcode")
```

## Graph and embedding views

For inspecting the `GraphBundle` that feeds `topo_tools` directly.

```python
from tanc.visualisation import plot_graph_matrix, plot_node_embedding

fig = plot_graph_matrix(bundle, reorder_by_label=True, log=False)
fig = plot_node_embedding(bundle, method="mds")    # or "tsne"
```

`plot_graph_matrix` handles `matrix_type ∈ {"distance", "similarity",
"adjacency"}` and picks a sensible default colormap per type.
`plot_node_embedding` colours points by `bundle.node_labels` when present.

## Topological-uncertainty score distribution

```python
from tanc.visualisation import plot_tu_score_distribution

# Plain histogram
fig = plot_tu_score_distribution(scores)

# Split correct vs misclassified when labels are available
fig = plot_tu_score_distribution(scores, y_true=y_true, y_pred=y_pred)
```

## Utility functions

### `make_figure`

```python
from tanc.visualisation import make_figure

fig, ax = make_figure(ax=None, figsize=(8, 6), default_figsize=(6, 5))
```

### `annotate_ph_stats`

```python
from tanc.visualisation import annotate_ph_stats

annotate_ph_stats(ax, statistics, dim=0)
# Adds: total_persistence, persistence_norm, persistence_entropy
```

### `format_paper_reference`

```python
from tanc.visualisation import format_paper_reference

label = format_paper_reference("Watanabe & Yamana (2021)")
# "Method: Watanabe & Yamana (2021)"
```

## Colour scheme

Each homology dimension gets a consistent colour across all PH plot types:

| Dimension | Colour |
|---|---|
| H0 | tab:blue |
| H1 | tab:orange |
| H2 | tab:green |
| H3 | tab:red |
| H4 | tab:purple |

## Shared parameters

All plot functions accept these common parameters:

| Parameter | Type | Description |
|---|---|---|
| `ax` | `matplotlib.Axes` or `None` | Reuse an existing axes. `None` = create a new figure |
| `figsize` | `(width, height)` or `None` | Figure size in inches. `None` = sensible default |
| `title` | `str` or `None` | Plot title |

Persistence plots additionally accept `ph_result: PersistenceResult` and
`dims: list[int] | None`.

## Stable PH summaries (landscape & image)

Two function-valued / pixel-valued summaries of a persistence diagram —
both stable under bottleneck perturbation and routinely used as
vectorised inputs to downstream classifiers.

```python
from tanc.visualisation import (
    plot_persistence_landscape,  # Bubenik (2015)
    plot_persistence_image,      # Adams et al. (2017)
    persistence_landscape,       # raw numeric form
    persistence_image,           # raw numeric form
)

fig = plot_persistence_landscape(ph_result, dim=1, k_max=5)
fig = plot_persistence_image(ph_result, dim=1, resolution=25,
                             sigma=None, weight="linear")
```

Backends. Both functions delegate the numeric work to a battle-tested
optional library when one is installed:

| Function | Default backend search | Fallback |
|---|---|---|
| `persistence_landscape` | `gudhi.representations.Landscape` | numpy |
| `persistence_image`     | `persim.PersistenceImager` → `gudhi.representations.PersistenceImage` | numpy |

Pass `backend="gudhi" | "persim" | "numpy"` to force a specific choice.
The numpy fallback has no extra deps but runs single-threaded — for
moderately-sized diagrams (a few hundred bars) it's still milliseconds.

## Cross-snapshot views

These take a `TrainingView` and condense it differently:

```python
from tanc.visualisation import (
    plot_diagram_distance_matrix,
    plot_ph_statistic_pairplot,
)

# Pairwise Wasserstein/bottleneck heatmap across all snapshots
fig = plot_diagram_distance_matrix(view, metric="wasserstein", dim=1)

# Scatter-matrix of multiple PH statistics, points coloured by epoch
fig = plot_ph_statistic_pairplot(
    view,
    stats=("total_persistence", "persistence_norm", "persistence_entropy"),
    dim=0,
)
```

## Dashboards and animations

```python
from tanc.visualisation import (
    plot_training_summary,
    make_training_animation,
)

# 2x3 panel: loss/acc, weight-norm, ID-over-training, H0 total persistence,
# Betti_1 trajectory, Wasserstein distance to final diagram.  Any failing
# panel falls back to a placeholder without aborting the figure.
fig = plot_training_summary(view)

# Animated PH evolution.  kind = "diagram" | "barcode" | "landscape" | "image".
# Save to GIF (pillow) or MP4 (requires ffmpeg).
anim = make_training_animation(view, kind="diagram", out="train.gif", fps=4)
```

## Paper-companion plots (Gebhart, Lacombe, Naitzat, Liu)

```python
from tanc.visualisation import (
    plot_pathways_on_network,   # Gebhart et al. 2019 — signal-pathway overlay
    h0_signal_pathways,         # Gebhart et al. 2019 — H0 representative cycles
    plot_tu_roc,                # Lacombe et al. 2021 — TU misclassification ROC
    plot_betti_layer_bars,      # Naitzat et al. 2020 — Betti-per-layer bars
    plot_polyhedral_regions,    # Liu et al. 2023 — ReLU linear-region map
)

# Gebhart pathways. mode="h0" highlights the network's actual H0 representative
# signal pathways (the maximum-spanning-tree / single-linkage clusters of the
# |w.h| activation graph), coloured by cluster; mode="magnitude" is the old
# top-|w| proxy.  Pass the activation-weighted edges for "h0".
fig = plot_pathways_on_network(phi_matrices, mode="h0", n_pathways=6)
fig = plot_pathways_on_network(weight_matrices, mode="magnitude", top_frac=0.05)

# The underlying extractor, if you want the cluster assignment / cut edges:
paths = h0_signal_pathways(phi_matrices, n_pathways=6)   # cluster_of, tree_edges, death_edges

# Liu polyhedral decomposition of a 2-D input space: each distinct binary ReLU
# pattern is one linear region; optionally overlay the decision boundary.
fig = plot_polyhedral_regions(xx, yy, grid_patterns, points=X, point_labels=y,
                              decision=pred)

# Misclassification ROC curve from TU scores (sklearn metrics under the hood)
fig = plot_tu_roc(tu_scores, y_true=y_true, y_pred=y_pred)

# Grouped bar chart of per-layer Betti numbers (decreasing-with-depth pattern)
fig = plot_betti_layer_bars(layer_results, dims=[0, 1])
```

## ID uncertainty bands and tail-behaviour diagnostics

```python
from tanc.visualisation import plot_id_with_uncertainty, plot_id_qq
from tanc.topo_tools import estimate_id_global

# Per-layer ID with mean +/- 1 SD from bootstrap subsampling
fig = plot_id_with_uncertainty(activations, method="global",
                               n_bootstraps=20, subsample_frac=0.8)

# QQ plot of log(log mu) values — heavy upper tail inflates the global ID.
res = estimate_id_global(distance_matrix)
fig = plot_id_qq(res, dist="norm")        # or pass the dict / a distance matrix
```

## Pipeline diagram

```python
from tanc import TDAPipeline
from tanc.visualisation import plot_pipeline_diagram

pipe = TDAPipeline.from_paper("rieck2019")
fig = plot_pipeline_diagram(pipe)         # boxes for builder / tool / default plot
```

## File structure

```
visualisation/
├── __init__.py              # Re-exports every public plot function
├── representations.py       # Persistence diagram, barcode, Betti curve, landscape, image, comparison
├── training_plots.py        # Loss/accuracy, weight-norm, activation-stats
├── trajectory_plots.py      # ID/PH-dim/mag-dim/PH-stat/Betti/dgm-distance over epochs,
│                            #   pairwise diagram distance matrix, PH-stat pairplot,
│                            #   plus pipeline_trajectory / TrajectorySeries (any pipeline, per layer)
├── evolution_plots.py       # Diagram small-multiples across training
├── summary_plots.py         # plot_training_summary, make_training_animation
├── graph_plots.py           # GraphBundle heatmap, node embedding, TU score histogram,
│                            #   ID-with-uncertainty (bootstrap bands), ID QQ plot
├── pipeline_diagram.py      # plot_pipeline_diagram (flowchart of a TDAPipeline)
└── visualisation_utils.py   # make_figure, annotate_ph_stats, format_paper_reference
```

## Sweep plots

Drawing one Mapper graph answers "what does this configuration give?".  A sweep
asks a different question — does the structure survive a *range* of parameters,
or is it an accident of one setting? — and that needs the graphs side by side.

```
sweep_plots
   ├── plot_mapper_sweep_graph(graph)   → one graph, full size, with its scale bar
   ├── plot_graph_panel(graphs)         → many graphs as a contact sheet
   ├── plot_stability_heatmap(rows)     → a measure across two swept axes
   ├── plot_cover_degeneracy(rows)      → nerve collapse vs shattering
   ├── plot_node_size_distribution(rows)→ node sizes across the sweep
   ├── plot_filter_sweep(rows)          → a measure against point_filter strength
   └── plot_population_summary(pop)     → the trained population itself
```

`plot_graph_panel` takes a sequence of graphs — load them with
`result.graph(row)`, which reads from the store rather than recomputing:

```python
from tanc.visualisation import plot_graph_panel

rows = result.rows()
plot_graph_panel(
    [result.graph(r) for r in rows],
    titles=[f"n={r['n_intervals']} ovl={r['overlap']}" for r in rows],
    suptitle="conv1, trained",
)
```

Per-tile colourbars are suppressed: at contact-sheet scale they take more room
than the graph does, and relative node size is already legible from the marker
areas.  Use `plot_mapper_sweep_graph` when you want one graph at full size with
its scale.
