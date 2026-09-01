# topo_tools

Module 2 of the pipeline — the **method** axis (see `docs/composing.rst`). Consumes `GraphBundle` objects from `graph_builder` and produces `TopoResult` objects containing topological summaries. Three interchangeable tools are available: persistent homology (PH), Mapper, and intrinsic / fractal / magnitude dimension estimation — pick whichever lens you want on the space you built. `TopoResult` also persists with `.save()` / `TopoResult.load()` so an expensive computation (e.g. PH) is computed once and reloaded.

## Data flow

```
GraphBundle
     │
     ├──→ run_ph()                 → TopoResult (ph_result, statistics)
     ├──→ run_mapper()             → TopoResult (mapper_graph, node_members, ...)
     └──→ run_trajectory_dimension() → TopoResult (dimension_result)
            or
         run_activation_id()       → dict (id_estimates per layer)
```

## Persistent homology — ph_tool

Computes VR or superlevel-set persistent homology from a GraphBundle.

### Backends

Three interchangeable PH backends:

| Backend | Package | Install |
|---|---|---|
| `"ripser"` (default) | ripser | `pip install ripser` |
| `"gudhi"` | GUDHI | `pip install gudhi` |
| `"giotto"` | giotto-ph (parallel ripser) — **not** giotto-tda¹ | `pip install giotto-ph` |

¹ The `"giotto"` PH backend uses **giotto-ph** (`gph.ripser_parallel`), a multithreaded ripser. giotto-tda 0.6.2 hard-pins `scikit-learn==1.3.2` and fails on modern scikit-learn, so it is not usable here. (giotto-tda is still required, separately, by the **Mapper** tool.)

### Usage

```python
from tanc.topo_tools import run_ph, compute_persistence

# High-level: builds PH + computes statistics
result = run_ph(bundle, max_dim=1, backend="ripser")
result.ph_result.diagrams    # {0: ndarray, 1: ndarray}
result.statistics            # {"H0_total_persistence": ..., "H1_persistence_entropy": ...}

# Low-level: PH only, no statistics
result = compute_persistence(bundle, max_dim=1)
```

### Directed clique complex (Watanabe & Yamana)

ripser/GUDHI Rips compute **flag** complexes — a triangle enters at the *max* of
its three edges. Watanabe & Yamana's construction is different: a **directed
clique complex with path-product relevance**, where an induced edge and a
triangle are born at the *product* of relevances along a path (each ≤ 1, so they
enter much later), and every simplex's filtration value is the **integer
threshold index (1–64)** of the paper's descending relevance schedule. This
produces the paper's wide H1 "belt" above the diagonal, which a flag complex
cannot. Route a feed-forward network's weight matrices through it via:

```python
from tanc.topo_tools import compute_watanabe_ph

# weight_matrices = consecutive (N_in, N_out) FCN weight matrices
result = compute_watanabe_ph(weight_matrices, mode="positive")  # or "absolute" (PHPM)
result.diagram(1)                                  # integer-axis H1 belt
result.ph_result.metadata["watanabe"]["h1_representatives"]  # birth edge + death triangle per loop
```

It is also wired into the pipeline preset — `TDAPipeline.from_paper("watanabe2021")`
runs `run_ph(bundle, input_complex="directed_clique", relevance_mode="positive")`,
reading the per-layer weights the `weight_graph` builder stashes in
`metadata["per_layer_weights"]`. The representative cycles feed
`tanc.applications.pruning` (PHPM).

### Discrete filtration schedules

By default `run_ph` computes a continuous superlevel-set / VR filtration and reports births and deaths in the original filtration units (similarity or distance). Some papers, however, define their PH on a fixed list of thresholds and report births/deaths as integer indices into that list (e.g. Watanabe & Yamana 2021 use a 64-step log schedule). Pass `filtration_schedule=` to snap each (birth, death) onto that schedule **after** the continuous computation — no extra PH runs are needed.

```python
from tanc.topo_tools import (
    run_ph,
    build_filtration_schedule,
    snap_diagrams_to_schedule,
)

# (1) Use the helper directly on already-computed diagrams
result = run_ph(bundle, max_dim=1)
schedule = build_filtration_schedule(
    {"type": "log_decades", "max": 1.0, "n_decades": 7, "per_decade": 8}
)
snapped = snap_diagrams_to_schedule(
    result.ph_result.diagrams, schedule, units="index"
)

# (2) Or have run_ph do it in one call
result = run_ph(
    bundle, max_dim=1,
    filtration_schedule={"type": "log_decades", "max": 1.0,
                         "n_decades": 7, "per_decade": 8},
    schedule_units="index",   # "index" = paper-style integer axes
                              # "value" = keep filtration units
)
```

Accepted forms for `filtration_schedule`:

| Form | Meaning |
|---|---|
| `None` (default) | No snapping — continuous filtration, original units. |
| `int n` | `n` linearly spaced thresholds across the observed birth/death range. |
| `list[float]` / `tuple[float]` | Explicit threshold values (sorted ascending). |
| `{"type": "linear", "min": a, "max": b, "n": k}` | Uniform schedule. |
| `{"type": "log", "min": a, "max": b, "n": k}` | Geometric schedule. |
| `{"type": "log_decades", "max": M, "n_decades": d, "per_decade": p}` | Watanabe-style decade schedule. `max=1.0, n_decades=7, per_decade=8` → 64 thresholds 1.0, 0.9, …, 0.2, 0.1, 0.09, …, 1e-7. |

`schedule_units`: `"value"` replaces (birth, death) with the matched schedule values; `"index"` replaces them with the 0-based position in the schedule.

**How it works.** PH is run once with the backend's native continuous filtration. The resulting diagrams are then projected onto the nearest schedule point. This is mathematically equivalent to computing PH separately at each threshold (when no two distinct birth/death values fall between consecutive schedule points) but is ~`len(schedule)`× cheaper.

### Statistics

Individual statistic functions can be called directly on a persistence diagram:

```python
from tanc.topo_tools import (
    total_persistence,      # sum of lifetimes^p
    persistence_norm,       # p-norm of lifetimes (Neural Persistence metric)
    betti_number,           # count of bars alive at epsilon
    betti_curve,            # Betti number over a filtration linspace
    persistence_entropy,    # Shannon entropy of lifetime distribution
    asdsq_moments,          # mean/std of births, deaths & their squares (Ballester 2024)
    convex_hull_area,       # area of the (birth, death) convex hull
    wasserstein_distance,   # between two diagrams (requires persim)
    bottleneck_distance,    # between two diagrams (requires persim)
)

dgm = result.ph_result.diagrams[0]
total_persistence(dgm, p=1)
persistence_norm(dgm, p=2)
betti_number(dgm, epsilon=0.5)
betti_curve(dgm, resolution=100)
persistence_entropy(dgm)
convex_hull_area(dgm)   # Watanabe & Yamana (2021) Table 4 metric
asdsq_moments(dgm)      # {"mean_birth": ..., "std_death_sq": ...} — Ballester (2024)
```

**ASDSQ** (`asdsq_moments`) is the best-performing persistence summary in Ballester
et al. (2024) for predicting the generalization gap: the average and standard
deviation of the diagram's births and deaths, concatenated with the same moments of
their *element-wise squares* (8 scalars per diagram). The essential (infinite-death)
bar is dropped; drop the `*_sq` keys to recover the simpler **ASD** summary.

Or compute a batch via `compute_statistics`:

```python
from tanc.topo_tools import compute_statistics

stats = compute_statistics(
    result.ph_result,
    stat_names=["total_persistence", "persistence_norm", "persistence_entropy"],
)
# {"H0_total_persistence": 3.14, "H0_persistence_norm": 1.73, ...}

# ASDSQ expands to 8 keys per dimension: H{d}_{mean,std}_{birth,death}[_sq]
stats = compute_statistics(result.ph_result, stat_names=["asdsq"])
# {"H0_mean_birth": ..., "H1_std_death_sq": ...}
```

Valid `stat_names`: `total_persistence`, `persistence_norm`, `betti_number`,
`betti_curve`, `persistence_entropy`, `asdsq`, `convex_hull_area`.

## Topological Uncertainty — uncertainty

Per-sample monitoring score from Lacombe et al. (2021), with the underlying primitives exposed as standalone functions.

### Functional primitives

```python
from tanc.topo_tools import (
    bipartite_mst_diagram,    # sorted MST edge weights of |W * x|, sparse-backed
    frechet_mean_diagram,     # elementwise mean of equal-length sorted diagrams
    diagram_l2_distance,      # L2 distance between sorted equal-length diagrams
)

# Persistence diagram of one (layer, sample) pair (Lacombe et al. 2021 Section 2.2)
d = bipartite_mst_diagram(W, x)        # (N_in + N_out - 1,) sorted descending

# Frechet mean of a list of diagrams (their average)
mean = frechet_mean_diagram([d1, d2, d3, ...])

# Paper's metric Dist
dist = diagram_l2_distance(d, mean)
```

The MST is computed via `scipy.sparse.csgraph.minimum_spanning_tree` on a sparse CSR adjacency, so large layers (e.g. `9216 × 128` ≈ 1.2 M edges) are tractable without instantiating the ~700 MB dense matrix.

### `TopologicalUncertainty` class

Streaming `partial_fit` / `score` monitor that accumulates per-(layer, predicted-class) Fréchet-mean diagrams from training activations, then scores new inputs by their average per-layer L2 distance to the mean of their predicted class.

```python
from tanc.topo_tools import TopologicalUncertainty

tu = TopologicalUncertainty(
    weight_matrices=[W_fc1, W_fc2],   # one per layer; W_l shape (N_in_l, N_out_l)
    n_classes=10,
)

# Streaming fit on training data (no full diagram storage required)
for X_batch in train_batches:
    activations_per_layer = extract_pre_activations(X_batch)   # list of (B, N_in_l)
    predictions           = network_predictions(X_batch)        # (B,)
    tu.partial_fit(activations_per_layer, predictions)

# Inspect what was fit
tu.mean_diagrams      # dict (layer_idx, class_idx) -> Frechet-mean diagram
tu.class_counts       # array of per-class training sample counts

# Score new inputs -- higher = more anomalous
scores = tu.score(test_activations_per_layer, test_predictions)
```

The fit is streaming-equivalent to a single fit on the concatenated data (verified in unit tests). Memory cost: one sorted vector per `(layer, class)` pair — bounded regardless of training set size.

### How TU is used in practice

`scores` is a real number, not a class label. Two typical usage patterns:

1. **OOD detection** — flag inputs whose TU exceeds the q-th quantile of the training-set TU distribution. The paper's Table 1 uses TU and `1 - confidence` head-to-head as binary OOD detectors and reports AUC + FPR@95%TPR.
2. **Distribution shift monitoring** — track the TU distribution of incoming batches; a sudden shift in the 1-D distribution warns of input drift even when softmax confidence stays near 1 (paper Section 4.3 / Fig. 5).

## Mapper — mapper_tool

Builds a Mapper graph from node features using giotto-tda. Requires `pip install giotto-tda`.

### Usage

```python
from tanc.topo_tools import run_mapper

result = run_mapper(
    bundle,
    filter_fn="pca",           # "pca" | "l2_norm" | "eccentricity" | "entropy" | callable
    n_components=2,
    n_intervals=10,
    overlap_frac=0.3,
    compute_ph=True,            # optionally compute PH of the Mapper graph
)

result.mapper_graph            # networkx.Graph
result.mapper_node_members     # {node_id: [point_indices]}
result.mapper_filter_values    # (N, k) ndarray
result.mapper_graph_stats      # {"n_nodes": ..., "density": ..., ...}
```

### Mapper pipeline and plotting helpers

```python
from tanc.topo_tools import mapper_pipeline, plot_mapper_graph

bundle, graph, node_members, filter_values = mapper_pipeline(
    X,
    filter_fn="l2_norm",
    n_intervals=20,
    overlap_frac=0.3,
)
fig = plot_mapper_graph(graph, node_members)
```

### Graph analysis functions

```python
from tanc.topo_tools import (
    mapper_graph_stats,        # n_nodes, n_edges, density, diameter, etc.
    cycle_score,               # first Betti number of the graph
    class_separation_score,    # fraction of cross-class edges
    mapper_graph_adjacency,    # (N, N) weighted adjacency matrix
    compute_mapper_ph,         # PH of the Mapper graph itself
)
```

### Filter functions

```python
from tanc.topo_tools import apply_filter

values = apply_filter(X, "pca", n_components=2)         # PCA projection
values = apply_filter(X, "l2_norm")                    # L2 norm lens
values = apply_filter(X, "eccentricity")               # distance from mean
values = apply_filter(X, "entropy")                    # row-wise entropy
values = apply_filter(X, my_custom_filter_fn)            # any callable
```

## Dimension estimation — dimension_tool

Estimates intrinsic dimensionality of activation spaces or weight trajectories.

### Activation intrinsic dimension (2NN)

```python
from tanc.topo_tools import (
    run_activation_id,
    estimate_id_across_layers,
    estimate_id_global,
    estimate_id_calibrated,
    estimate_id_local,
)

# Per-layer ID estimation
layer_result = run_activation_id(
    activations,         # list of (N, N) distance matrices or (N, D) activation arrays
    method="calibrated", # "calibrated" (Ong 2026) | "global" (asymptotic 2NN) | "local" (Ruppik)
)
layer_result["id_estimates"]   # [1.2, 3.4, 5.6, ...]
layer_result["layer_labels"]   # ["Layer 0", "Layer 1", ...]

# Three estimators:
#   global     — asymptotic 2NN:  d = exp(-γ - mean(log log μ)),  μ = r₂/r₁.
#   calibrated — Ong et al. (2026): per-point L_{k,j} = -log log(R_k/R_j), averaged,
#                then the sample-size-calibrated map  d = exp(α·Lbar + β), with (α, β)
#                fitted against Gaussian clouds of known dimension.  Accepts k, j
#                (default 2, 1).  Corrects the finite-sample bias of `global`.
result = estimate_id_calibrated(distance_matrix, k=2, j=1)
result = estimate_id_global(distance_matrix)      # {"id_estimate": ..., "loglog_mean": ...}
result = estimate_id_local(distance_matrix, n_subsample=100, n_neighbours_local=50)
```

### Trajectory dimension

```python
from tanc.topo_tools import (
    run_trajectory_dimension,
    compute_ph_dimension,
    compute_magnitude_dimension,
    WeightTrajectoryAnalyser,
)

# Via run wrapper
result = run_trajectory_dimension(bundle, method="ph_dimension", loss_values=losses)
result.dimension_result["ph_dimension"]

result = run_trajectory_dimension(bundle, method="magnitude_dimension")
result.dimension_result["magnitude_dimension"]

## Mapper graph comparison and exemplars

Companion helpers to `run_mapper` matching Zhou et al. (2023) and TopoAct
(Rathore et al. 2021) use cases.

```python
from tanc.topo_tools import (
    mapper_hypergraph_gw_distance, mapper_gw_distance, node_exemplars,
)

# Zhou et al. (2023): Gromov-Wasserstein distance between two Mappers as
# HYPERGRAPHS — nodes are sample-membership hyperedges, cost is the Jaccard
# distance between member sets, and node masses are the cluster sizes.  This is
# the paper's construction; pass the TopoResults directly.  Requires POT.
d = mapper_hypergraph_gw_distance(result_a, result_b)

# Plain-graph GW on the Mapper adjacency with uniform node masses.  It ignores
# node-to-sample membership, so it is a coarser comparison than the hypergraph
# form above — cheaper, and enough when only graph shape is of interest.
d_graph = mapper_gw_distance(result_a.mapper.graph, result_b.mapper.graph)

# Per-node exemplar indices for downstream visualisation
# (the toolkit is deliberately not image-aware; render exemplars yourself).
exemplars = node_exemplars(result, data=X, k=5, rank_by="centroid")
# exemplars is dict[node_id, ndarray of data indices].
```

## Trajectory-dimension class

# Via convenience class (builder style — matches TDAPipeline)
analyser = WeightTrajectoryAnalyser(bundle, loss_values=losses)
ph_result  = analyser.ph_dimension(alpha=1.0)
mag_result = analyser.magnitude_dimension()

# Or preconfigure for a published method:
analyser = WeightTrajectoryAnalyser.from_paper("birdal2021")
analyser.explain()                      # what will be computed
result = analyser.fit(bundle)            # runs whichever method the preset selected
# Supported presets: 'birdal2021', 'dupuis2023', 'andreeva2024'
```

### Dimension visualisation

```python
from tanc.topo_tools import (
    plot_id_across_layers,           # ID vs layer index line plot
    plot_2nn_ratio_distribution,     # histogram of mu = r2/r1
    plot_loglog_ratio_distribution,  # histogram of log(log(mu))
    plot_ph_scaling,                 # log-log scatter of lifetime sum vs subset size
    plot_magnitude_scaling,          # log-log scatter of Mag(t) vs t
)
```

## Output containers

### PersistenceResult

```python
ph_result.diagrams     # {dim: (n_bars, 2) ndarray with [birth, death] columns}
ph_result.metadata     # {"runtime_seconds": ..., "backend": ..., ...}
```

### TopoResult

Unified container returned by all tools:

```python
result.tool                  # "ph" | "mapper" | "dimension"
result.ph_result             # PersistenceResult or None
result.statistics            # dict or None
result.mapper_graph          # networkx.Graph or None
result.mapper_node_members   # dict or None
result.mapper_graph_stats    # dict or None
result.dimension_result      # dict or None
result.config                # full parameter record
result.paper_reference       # citation string (from presets)

# Plotting (dispatches based on tool)
result.plot("diagram")       # PH: persistence diagram
result.plot("barcode")       # PH: barcode
result.plot("betti_curve")   # PH: Betti curve
result.plot("graph")         # Mapper: network visualisation
result.plot("id_layers")     # Dimension: ID across layers
result.plot("ph_scaling")    # Dimension: PH scaling plot
result.plot("magnitude_scaling")  # Dimension: magnitude scaling plot
```

## File structure

```
topo_tools/
├── __init__.py          # Re-exports all public functions and classes
├── _result.py           # TopoResult, PersistenceResult dataclasses
├── ph_tool.py           # run_ph, compute_persistence, all statistic functions
├── mapper_tool.py       # run_mapper, apply_filter, graph stats, plotting
└── dimension_tool.py    # 2NN estimators, PH/magnitude dimension, plotting
```
