# graph_builder

The **construction** axis (see `docs/composing.rst`): converts raw neural-network data — the *space* you chose to study (weight matrices, activation matrices, class labels, binary activation patterns) — into `GraphBundle` objects, the standardised container that flows into all downstream `topo_tools` analyses. Each builder is one way to turn a space into shape; you mix and match it with the method (`tool`) downstream.

## Data flow

```
weights / activations / labels / patterns
         │
         ▼
   graph_builder function
         │
         ▼
     GraphBundle
    ┌──────────────────────────────┐
    │ matrix      (N, N) ndarray   │
    │ matrix_type "distance" /     │
    │             "similarity" /   │
    │             "adjacency"      │
    │ node_features (N, d) or None │
    │ node_labels   (N,)  or None  │
    │ n_nodes       int            │
    │ metadata      dict           │
    └──────────────────────────────┘
         │
         ▼
     topo_tools (ph / mapper / dimension)
```

## Builders

### build_weight_graph

Builds a weighted **similarity** adjacency matrix from weight matrices and/or activation matrices.

```python
from tanc.graph_builder import build_weight_graph

# Single weight matrix
bundle = build_weight_graph(W)

# Multiple layers
bundle = build_weight_graph([W1, W2, W3], edge_weight="normalized", graph_scope="full")

# Coupled (weight + activation) per layer
bundle = build_weight_graph(
    [(W1, A1), (W2, A2)],
    edge_weight="weighted_activation",
    graph_scope="multipartite",
)
```

**Parameters:**

| Parameter | Options | Description |
|---|---|---|
| `edge_weight` | `"absolute"`, `"normalized"`, `"relevance"`, `"correlation"`, `"weighted_activation"` | How to compute edge weights from raw data |
| `graph_scope` | `"bipartite"`, `"multipartite"`, `"full"` | How layers are assembled into one matrix |
| `induced_paths` | `True` / `False` | Add max-product induced-path connections (requires `graph_scope="full"`) |
| `layer_subset` | `list[int]` or `None` | Select specific layers by index |
| `node_feature_fn` | `"laplacian_eigenvectors"`, `"degree_features"`, callable, `None` | Per-node embedding method |

**Edge weight modes:**
- `"absolute"` — raw absolute weights `|w_ij|`.
- `"normalized"` — global max normalisation: `|w_ij| / max(|W|)`.
- `"relevance"` — per-neuron column-normalised positive weights (Watanabe & Yamana 2021, Eq. 1): `R_ij = max(w_ij, 0) / Σ_k max(w_kj, 0)`. Neurons with no positive incoming weights get zero relevance.
- `"correlation"` — absolute Pearson correlation between neuron activations. Requires coupled `(weight, activation)` input.
- `"weighted_activation"` — activation-weighted edges (Gebhart et al. 2019): `|w_ij · h_i|` where `h_i` is the mean pre-synaptic activation. Requires coupled input.

**Source papers:** Watanabe & Yamana (2021), Rieck et al. (2019), Gebhart et al. (2019), Lacombe et al. (2021).

### build_point_cloud_graph

Reads a **single** weight matrix as a **point cloud** — each row (or column) is one
point — and returns the pairwise-**distance** bundle. The complement of
`build_weight_graph`'s neuron graph: the coordinate-free counterpart of separating a
layer's left and right singular subspaces.

```python
from tanc.graph_builder import build_point_cloud_graph

# W is (N_in, N_out)
rows = build_point_cloud_graph(W, orientation="rows")               # N_in input features as points in R^{N_out}
cols = build_point_cloud_graph(W, orientation="cols", metric="cosine")  # N_out output neurons as points in R^{N_in}
```

**Parameters:**

| Parameter | Options | Description |
|---|---|---|
| `orientation` | `"rows"`, `"cols"` | `"rows"` → input features as points in output space; `"cols"` → output neurons (receptive fields) as points in input space |
| `metric` | `"euclidean"`, `"cosine"`, `"correlation"`, `"cityblock"` | Distance between points |
| `attach_points` | `True` / `False` | Store point coordinates as `node_features` (Mapper-ready) |

Pipeline use — record one layer so a single matrix reaches the builder:

```python
pipe = TDAPipeline(builder="point_cloud_graph", builder_kwargs={"orientation": "cols"}, tool="ph")
result = pipe.fit_model(model, X, representation="weights", layer_selection=["fc1"])
```

### build_activation_graph

Builds a **distance** matrix from an activation point cloud.

```python
from tanc.graph_builder import build_activation_graph, find_optimal_k

# Single layer
bundle = build_activation_graph(activations, distance="euclidean")

# All layers at once (list of per-layer matrices, concatenated horizontally)
bundle = build_activation_graph(snapshot.all_activation_matrices(), distance="euclidean")

# Geodesic distances (kNN graph → shortest paths)
k = find_optimal_k(activations)
bundle = build_activation_graph(activations, distance="geodesic", k=k)

# Correlation distances between neurons based on activations
bundle = build_activation_graph(activations, distance="correlation")

# Ballester (2024) functional graph: drop constant neurons, then
# importance-sample the neuron node set down to a budget (Eqs. 4.2-4.3)
bundle = build_activation_graph(
    snapshot.all_activation_matrices(),   # whole network's units, concatenated
    distance="correlation",
    drop_constant=True,
    node_sampling="importance",           # or "uniform"
    max_neurons=3000,
)
bundle.metadata["selected_neurons"]       # indices of the kept neurons
```

**Parameters:**

| Parameter | Options | Description |
|---|---|---|
| `activations` | `(N, D)` ndarray or `list[ndarray]` | Single layer or list of per-layer matrices (concatenated horizontally) |
| `distance` | `"euclidean"`, `"geodesic"`, `"correlation"` | Distance metric |
| `k` | `int` or `None` | Neighbours for geodesic. Use `find_optimal_k()` for automatic selection |
| `drop_constant` | `True` / `False` | Drop zero-variance (constant-activation) neurons. Correlation is undefined for them, so this is the safe default for `distance="correlation"` |
| `node_sampling` | `"importance"`, `"uniform"`, `None` | Sub-sample neuron nodes before building the graph. `"importance"` = Ballester et al. (2024) Eqs. 4.2-4.3 (weight ∝ how often a neuron is the argmax-\|activation\|, with a floor so every neuron is reachable) |
| `max_neurons` | `int` or `None` | Neuron budget for `node_sampling` (ignored when the network already has ≤ this many neurons) |
| `sampling_seed` | `int` | Seed for `node_sampling` (deterministic per call) |

`drop_constant` and `node_sampling` operate on neuron-nodes and are only valid for `distance="correlation"`.

**Source papers:** Naitzat et al. (2020), Karuppiah et al. (2025), Ballester et al. (2024).

### build_labelled_complex_graph

Builds the **labelled Vietoris–Rips complex**: the vertices are one class `S`
(`source_class`), the opposite class is the reference set `W`, and a same-class
vertex is kept only when it lies **within `gamma` of `W`** (near the decision
boundary). VR persistent homology on the returned distances then realises the
labelled complex — same-class simplices filtered by proximity to the opposite
class, *not* a cross-class distance graph.

```python
from tanc.graph_builder import build_labelled_complex_graph

# vertices = class 0 points near the boundary (default gamma = median dist to W)
bundle = build_labelled_complex_graph(points, labels, source_class=0)

# keep all of S / a wider band, or use the locally-scaled variant
bundle = build_labelled_complex_graph(points, labels, gamma_quantile=1.0)
bundle = build_labelled_complex_graph(points, labels, scale="local", k_local=5)
```

**Source paper:** Ramamurthy, Varshney & Mody (2019).  *(The labelled Čech
variant needs a ball-intersection/α-complex and GUDHI; this builds labelled VR.)*

### build_polyhedral_graph

Builds a Hamming **distance** matrix over unique ReLU activation patterns. Each node represents one polyhedral region of the input space partition.

```python
from tanc.graph_builder import build_polyhedral_graph

bundle = build_polyhedral_graph(activations, input_type="auto")
```

**Source paper:** Liu, Cole, Peterson & Kirby (2023).

### build_kernel_graph

Builds a **distance** matrix from convolutional spatial filters (Carlsson & Gabrielsson 2020). Each spatial filter `(kH, kW)` is one point in a `kH*kW`-dimensional space.

```python
from tanc.graph_builder import build_kernel_graph

# Single model: raw PyTorch weight tensor (out_ch, in_ch, kH, kW)
# → reshape to (out_ch * in_ch, kH, kW) to get individual spatial filters
w = model.conv1.weight.detach().numpy()          # (out_ch, in_ch, 3, 3)
filters = w.reshape(-1, *w.shape[2:])            # (out_ch*in_ch, 3, 3)
bundle = build_kernel_graph(filters, distance="vne")

# Pool spatial filters across 100 trained model instances
ws = [m.conv1.weight.detach().numpy().reshape(-1, 3, 3) for m in models]
bundle = build_kernel_graph(ws, distance="vne",
                            density_filter=True, density_quantile=0.2)

# Already-flattened (N_filters, D) input also accepted
bundle = build_kernel_graph(filters_2d, distance="euclidean")
```

**Parameters:**

| Parameter | Options | Description |
|---|---|---|
| `weights` | `(N, kH, kW)` or `(N, D)` ndarray, or `list[ndarray]` | Spatial filters from one or more model instances. Lists are vstacked |
| `distance` | `"euclidean"`, `"vne"` | Distance metric. VNE normalises each dimension by its variance |
| `density_filter` | `True` / `False` | Remove low-density points before computing distances |
| `density_k` | `int` | Neighbours for density estimate (only when `density_filter=True`) |
| `density_quantile` | `float` in `(0, 1)` | Fraction of lowest-density points to remove |
| `node_feature_fn` | `"laplacian_eigenvectors"`, `"degree_features"`, callable, `None` | Per-node embedding method. `None` uses raw filter weights |

**Source paper:** Carlsson & Gabrielsson (2020).

### build_weight_trajectory

Builds a **distance** matrix where each node is a training step and edges encode movement in parameter space (or loss space).

```python
from tanc.graph_builder import build_weight_trajectory

# (T, P) array: T timesteps, P flattened parameters
bundle = build_weight_trajectory(trajectory, distance="euclidean")

# Loss-difference metric (Dupuis et al.): pass a (T, n_samples) PER-SAMPLE loss
# matrix for the pseudo-metric rho = mean_s |ell_i,s - ell_j,s|.
bundle = build_weight_trajectory(trajectory, distance="loss_difference",
                                 loss_values=per_sample_losses)   # (T, n_samples)
```

**Parameters:**

| Parameter | Options | Description |
|---|---|---|
| `distance` | `"euclidean"`, `"cosine"`, `"loss_difference"` | Inter-step distance metric |
| `loss_values` | `(T, n_samples)` or `(T,)` ndarray | For `"loss_difference"`. A 2-D **per-sample** matrix → the Dupuis pseudo-metric `mean_s\|ell_i,s − ell_j,s\|`; a 1-D scalar-loss vector → `\|mean-loss_i − mean-loss_j\|` (warns). Capture per-sample losses with `TrainingExtractor(loss_eval_data=(X, y))`. |

**Source papers:** Birdal et al. (2021), Andreeva et al. (2024), Dupuis et al. (2023).

## GraphBundle

The shared contract object. All builders return one and all topo_tools consume one.

```python
from tanc.graph_builder import GraphBundle

bundle.matrix          # (N, N) ndarray
bundle.matrix_type     # "distance" | "similarity" | "adjacency"
bundle.node_features   # (N, d) ndarray or None
bundle.node_labels     # (N,) ndarray or None
bundle.n_nodes         # int
bundle.metadata        # dict with builder provenance
```

**Matrix type semantics:**
- `"distance"` — entries are distances (0 on diagonal). Passed to ripser as a distance matrix.
- `"similarity"` — entries are similarities (max on diagonal). Converted via superlevel-set filtration before PH.
- `"adjacency"` — unweighted 0/1 graph. Converted via `1 - A` before PH.

## Node features

When the builder's input does not naturally produce per-node features (e.g. weight-only graphs), `compute_node_features` generates synthetic embeddings:

```python
from tanc.graph_builder import compute_node_features

features = compute_node_features(matrix, "similarity", "laplacian_eigenvectors", n_components=8)
features = compute_node_features(matrix, "distance", "degree_features", n_components=4)
features = compute_node_features(matrix, "similarity", my_custom_fn)
```

Built-in methods: `"laplacian_eigenvectors"` (spectral embedding) and `"degree_features"` (degree, clustering, betweenness, closeness centrality).

## File structure

```
graph_builder/
├── __init__.py             # Re-exports all public functions
├── _bundle.py              # GraphBundle dataclass
├── weight_graphs.py        # build_weight_graph
├── activation_graphs.py    # build_activation_graph, find_optimal_k
├── kernel_graphs.py        # build_kernel_graph
├── boundary_graphs.py      # build_labelled_complex_graph, build_polyhedral_graph
├── weight_trajectory.py    # build_weight_trajectory
└── node_features.py        # compute_node_features
```
