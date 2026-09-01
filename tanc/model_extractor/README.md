# model_extractor

Upstream extraction module for TANC. Takes a neural network model (PyTorch or TensorFlow/Keras), inspects its architecture, and extracts weights, activations, and classifications into containers that feed directly into the `graph_builder` module.

## Two equivalent API styles

The module exposes the same functionality through a class-based interface (preferred — same shape as `TDAPipeline`) and one-call function shortcuts:

| Class form (preferred) | Function shortcut |
|---|---|
| `ModelExtractor(model, ...).extract(data)` | `extract_model(model, data, ...)` |
| `TrainingExtractor(model, ...).run(...)` | `extract_training(model, ...)` |

Both class constructors auto-detect the framework from the model object — pass a `torch.nn.Module` and you get the PyTorch backend, pass a `tf.keras.Model` and you get the TensorFlow backend. The required kwargs for the detected framework are validated eagerly at construction time so a missing argument never costs you 30 min of training.

If you prefer an explicit, self-documenting choice, the factories `TrainingExtractor.for_pytorch(...)` / `.for_tensorflow(...)` are still available and behave identically.

The class form is what tutorials, examples, and `tanc.tour()` lead with. The functions are unchanged and remain fully backwards-compatible.

## Data flow

```
your model + data
       │
       ▼
  ModelExtractor(...).extract(data)     or   extract_model(...)
  TrainingExtractor(...).run()          or   extract_training(...)
       │
       ▼
  ModelSnapshot / TrainingView
       │
       ├── snapshot.weight_matrices()         → build_weight_graph()
       ├── snapshot.all_activation_matrices()  → build_activation_graph()
       ├── snapshot.coupled_weight_activations() → build_weight_graph(edge_weight="weighted_activation")
       └── view.weight_trajectory()           → build_weight_trajectory()
```

## Supported frameworks

- **PyTorch** — `torch.nn.Module`
- **TensorFlow / Keras** — `tf.keras.Model`

The framework is auto-detected from the model object. No configuration needed.

## Two extraction modes

### Final view — already-trained model

Use `ModelExtractor` (or the `extract_model` shortcut) when you have a model that's already been trained and you want to extract its current state.

```python
from tanc.model_extractor import ModelExtractor

# Configure once
extractor = ModelExtractor(
    model           = my_trained_model,
    aspects         = ["weights", "activations", "classifications"],
    layer_selection = "all_linear",
)
extractor.explain()                       # what will be extracted
snap_test  = extractor.extract(X_test)    # extract from any input batch
snap_train = extractor.extract(X_train)   # same configuration, different data

# Feed into graph_builder
from tanc.graph_builder import build_weight_graph, build_activation_graph

bundle_W = build_weight_graph(snap_test.weight_matrices())
bundle_A = build_activation_graph(snap_test.all_activation_matrices()[0])
```

One-call shortcut (equivalent to the above for a single batch):

```python
from tanc.model_extractor import extract_model

snapshot = extract_model(
    model   = my_trained_model,
    data    = X_test,
    aspects = ["weights", "activations", "classifications"],
)
```

**What gets extracted:**

| Aspect | Shape | Description |
|---|---|---|
| `weights` | `(N_in, N_out)` per layer | Weight matrices, transposed/reshaped to consistent orientation |
| `activations` | `(N_samples, N_neurons)` per layer | Layer outputs from a forward pass on your data |
| `classifications` | `(N_samples, N_classes)` | Raw model output (logits/probabilities) |

#### Activation pooling (transformers / LLMs)

By default a captured activation is **flattened** to `(N_samples, features)` (`(N, C, H, W) → (N, C·H·W)`), which is what MLP/CNN analysis wants. Transformer blocks emit `(batch, seq_len, hidden)`, where flattening conflates token positions with features. Pass `activation_pooling=` to reduce over the **sequence axis** instead, so each sequence becomes one point in `hidden`-space:

```python
# last-token summary of a causal decoder's blocks
snap = extract_model(llm, prompts,
                     aspects=["activations"],
                     layer_selection=["blocks.0", "blocks.1"],
                     activation_pooling="last")
snap.activation_matrix("blocks.0").shape   # (N_prompts, hidden)
```

| `activation_pooling` | Effect on `(batch, seq, hidden)` |
|---|---|
| `"flatten"` (default) | `(N, seq·hidden)` — unchanged historical behaviour |
| `"last"` | final token — a summary of the whole sequence for a causal decoder |
| `"first"` | first token (≈ BERT `[CLS]`) |
| `"mean"` / `"max"` | pool across all tokens |

Token pooling assumes the `(batch, seq, hidden)` layout and applies only to 3-D activations (2-D and conv feature maps always flatten). Available on `ModelExtractor`, `extract_model`, and `TDAPipeline.fit_model`, for both PyTorch and TensorFlow. The choice is recorded in `snapshot.metadata["activation_pooling"]`.

#### Sum-of-Rank-One (SoRO) layers

A SoRO (Sum-of-Rank-One) layer replaces a fully-connected layer with its trained SVD-style factors `W = U · diag(σ) · Vᵀ`. There is no canonical layer class (in PyTorch **or** Keras), so the extractor recognises a module/layer as SoRO by a small **duck-typed protocol** — implement **either**:

- a `soro_factors(self) -> (U, sigma, V)` method (preferred, explicit), or
- attributes `U`, `V`, and one of `sigma` / `singular_values` / `s`,

optionally with an `effective_weight(self)` method. Shape convention: `U` is `(out, r)`, `sigma` is `(r,)`, `V` is `(in, r)`. Supported for **both PyTorch and TensorFlow/Keras** — the same protocol, no registration needed beyond implementing it on your class.

```python
class SoRO(nn.Module):
    def __init__(self, in_f, out_f, r):
        super().__init__()
        self.U = nn.Parameter(torch.randn(out_f, r) * 0.1)   # (out, r)
        self.sigma = nn.Parameter(torch.rand(r))             # (r,)
        self.V = nn.Parameter(torch.randn(in_f, r) * 0.1)    # (in, r)
    def forward(self, x): return ((x @ self.V) * self.sigma) @ self.U.t()
    def soro_factors(self): return self.U, self.sigma, self.V
```

Such a layer is:

- **Inspected** as `layer_type="soro"` and treated as a single leaf even when built from sub-`nn.Linear`s (its internals aren't listed separately).
- **Selectable** by name or the `layer_selection="soro"` preset (and auto-selected as fully-connected).
- **Weight-captured** as both the trained factors (in `snapshot.soro_factors[name] = {"U","sigma","V"}`) **and** the assembled effective weight `W` (stored in `snapshot.weights[name]` as `(in, out)`), so a SoRO layer feeds the weight-graph / weight-trajectory tools exactly like an FC layer. Pass `soro_effective_weight=False` to keep **only** the factors (recover `W` on demand via `snapshot.effective_weight(name)`).
- **Activation-captured** like any layer — a forward hook on the SoRO module records its output.

```python
snap = extract_model(net, X, aspects=["weights", "activations"], layer_selection="all")
snap.soro_factor("soro")        # {"U": (out,r), "sigma": (r,), "V": (in,r)}  — the trained factors
snap.effective_weight("soro")   # (in, out) assembled W (cached in .weights, or computed from factors)
snap.weight_matrices()          # includes the SoRO effective W → build_weight_graph
```

Captured across training too: `fit_training(..., layer_selection="soro")` records the factors at every snapshot, so `view.snapshots[i].soro_factor(name)` tracks U/Σ/V epoch by epoch.

### Training view — run training with snapshot hooks

Use `TrainingExtractor` (or the `extract_training` shortcut) to train a model from scratch while capturing snapshots at regular intervals.

The framework is auto-detected from the model object — you only need to pass the kwargs that belong to that framework. The simplest way to supply data is the cross-framework `train_data` / `val_data` pair: hand over a raw dataset and the extractor builds the right loader for the detected framework. A model that is neither PyTorch nor TensorFlow raises `ValueError` (`Unsupported model type ...`) immediately.

`train_data` / `val_data` accept an `(X, y)` tuple of arrays/tensors, a `Dataset` / `tf.data.Dataset`, or a ready `DataLoader`. For PyTorch, `(X, y)` is wrapped in a `TensorDataset` + `DataLoader` using `batch_size` (default 128) and `shuffle` (training only). `val_data` is your held-out / test set.

**PyTorch (auto-detected) — raw dataset, no loader needed:**

```python
from tanc.model_extractor import TrainingExtractor

# The framework is detected from ``model``. Required PyTorch infra
# (a dataset, criterion, optimizer) is validated *here*, not 30 mins
# into training.
extractor = TrainingExtractor(
    model        = model,
    train_data   = (X_train, y_train),   # raw arrays — loader built for you
    val_data     = (X_test,  y_test),    # held-out / test set
    batch_size   = 128,
    criterion    = nn.CrossEntropyLoss(),
    optimizer    = torch.optim.Adam(model.parameters(), lr=1e-3),
    extract_data = X_test,
    aspects      = ["weights", "activations"],
    snapshot_every = 5,
)
extractor.explain()                                 # describe the plan
view = extractor.run(epochs=50, target_accuracy=0.95)

# Continue training from where the previous .run() stopped.
# The model object is reused, so its trained weights carry over.
view = extractor.run(epochs=20, continue_training=True)

# Weight trajectory for TDA
traj   = view.weight_trajectory()
bundle = build_weight_trajectory(traj, distance="euclidean")

# Scalar series
view.losses()      # [0.83, 0.41, 0.19, ...]
view.accuracies()  # [0.62, 0.84, 0.93, ...]
```

Already have loaders? They still work — pass `train_loader` / `val_loader` instead of `train_data` / `val_data` (they take precedence if both are given):

```python
extractor = TrainingExtractor(
    model=model, train_loader=train_loader, val_loader=val_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    extract_data=X_test,
)
```

**TensorFlow / Keras (auto-detected):**

```python
extractor = TrainingExtractor(
    model        = keras_model,
    train_data   = (X_train, y_train),
    val_data     = (X_val, y_val),
    compile_kwargs = dict(
        optimizer = "adam",
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    ),
    extract_data = X_test,
    aspects      = ["weights", "activations"],
    snapshot_every = 10,
)
view = extractor.run(epochs=50)
```

**Explicit factories** — equivalent, useful when you want the framework choice to be self-documenting:

```python
extractor = TrainingExtractor.for_pytorch(model=model, train_data=(X_train, y_train), ...)
extractor = TrainingExtractor.for_tensorflow(model=keras_model, train_data=..., ...)
```

**One-call shortcut (equivalent, no continuation, mostly for quick scripts):**

```python
from tanc.model_extractor import extract_training
view = extract_training(model=model, train_data=(X_train, y_train), criterion=...,
                        optimizer=..., extract_data=X_test,
                        epochs=50, snapshot_every=5)
```

**Training parameters:**

| Parameter | Set at... | Default | Description |
|---|---|---|---|
| `epochs` | `.run()` | 100 | Maximum training epochs |
| `target_accuracy` | `.run()` | 0.98 | Stop early when validation accuracy reaches this. `None` to disable |
| `verbose` | `.run()` | `True` | Print training progress |
| `continue_training` | `.run()` | `False` | Append snapshots to the previous `.run()` instead of resetting |
| `snapshot_every` | construction | 1 | Capture a snapshot every N epochs (or iterations) |
| `snapshot_schedule` | construction | `"epoch"` | `"epoch"` or `"iteration"` (PyTorch only) |
| `train_data` / `val_data` | construction | `None` | Raw dataset per split — `(X, y)`, a `Dataset`, or a loader. Cross-framework |
| `batch_size` | construction | 128 | Batch size when building a loader from raw `(X, y)` / `Dataset` data |
| `shuffle` | construction | `True` | Shuffle the training loader built from raw data (val never shuffled) |
| `train_loader` / `val_loader` | construction | `None` | Pre-built PyTorch loaders (alternative to `train_data` / `val_data`) |
| `scheduler` | construction | `None` | Optional PyTorch LR scheduler (e.g. `CosineAnnealingLR`); stepped once per epoch |
| `loss_eval_data` | construction | `None` | Optional `(X, y)`; when given, per-epoch **per-sample** losses are captured on it (for the Dupuis per-sample metric) |

Each captured `TrainingView` exposes per-epoch series: `accuracy_trajectory()` (validation), `train_accuracy_trajectory()` (training — recorded automatically), `loss_trajectory()`, and `per_sample_loss_trajectory()` (a `(T, n_eval)` matrix, populated only when `loss_eval_data` was given; feeds `build_weight_trajectory(distance="loss_difference")`).

The **learning rate, optimizer, and loss** are *not* separate `TrainingExtractor` arguments — they live in the objects you pass:

- **PyTorch:** the learning rate is whatever you set on the optimizer, e.g. `optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)`; the loss is `criterion=`.
- **TensorFlow / Keras:** both go in `compile_kwargs`, e.g. `compile_kwargs={"optimizer": tf.keras.optimizers.Adam(1e-3), "loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]}`.

This keeps the extractor framework-agnostic — it never second-guesses your training recipe. `batch_size` *is* an extractor argument because it only matters when the extractor builds a loader for you from raw `(X, y)` data.

The split between *structural* parameters (set at construction, rarely change) and *schedule* parameters (set per `.run()` call) mirrors `TDAPipeline`. Use `extractor.configure(snapshot_every=10, ...)` to update structural fields after construction.

### Saving a snapshot or trajectory

Training is the expensive step, so both containers persist with `.save(path)` / `.load(path)` (a pickled `.tda` file) — capture once, re-analyse forever:

```python
view = extractor.run(epochs=50)
view.save("trajectory.tda")

from tanc.model_extractor import TrainingView
view = TrainingView.load("trajectory.tda")     # ModelSnapshot.load(...) likewise
```

The same `.save` / `.load` pair is available on `GraphBundle` and `TopoResult`, and `result.plot(kind, save="fig.pdf")` writes a figure directly. See **Saving & loading results** in `GETTING_STARTED.md`. (`.tda` files are pickles — load only ones you trust.)

## Layer selection

The module inspects your model's architecture to determine which layers to hook for weights and activations.

### Automatic selection

| Model type | Auto-selected layers |
|---|---|
| Pure MLP (only Linear/Dense) | All linear layers |
| Pure CNN (only Conv) | All conv layers |
| Mixed (Linear + Conv) | Interactive prompt (see below) |

### Interactive prompt for mixed models

When a model has both linear and conv layers and `layer_selection=None`, the module prints a summary and asks:

```
========================================================================
  Layer Selection Required
========================================================================

  Your model 'ResNet18' contains both linear and convolutional layers.

  Model : ResNet18  [pytorch]
  Params: 11,689,512

  Idx   Name                                Type             Class                  Weight shape
  ----------------------------------------------------------------------------------------------------
  2     conv1                                conv             Conv2d                 (64, 3, 7, 7)
  5     layer1.0.conv1                       conv             Conv2d                 (64, 64, 3, 3)
  ...
  62    fc                                   linear           Linear                 (1000, 512)

  Options:
    1  all_linear       — fully-connected layers only
    2  all_conv         — convolutional layers only
    3  linear_and_conv  — both linear and conv layers
    4  all              — all parameterized layers
    5  custom           — enter layer names or indices

  (also available non-interactively: all_attention, all_embedding, soro)

  Choice [1-5 or layer names/indices, comma-separated]:
```

Set `clarify=False` to skip the prompt (defaults to linear layers with a warning).

### Explicit selection

```python
# String presets
extract_model(model, X, layer_selection="all_linear")
extract_model(model, X, layer_selection="all_conv")
extract_model(model, X, layer_selection="soro")            # Sum-of-Rank-One layers
extract_model(model, X, layer_selection="all_attention")   # nn.MultiheadAttention
extract_model(model, X, layer_selection="all_embedding")   # nn.Embedding
extract_model(model, X, layer_selection="linear_and_conv")
extract_model(model, X, layer_selection="all")

# By layer name
extract_model(model, X, layer_selection=["fc1", "fc2", "fc3"])

# By layer index
extract_model(model, X, layer_selection=[0, 2, 4])
```

## Inspecting a model

Use `inspect` to see the architecture before extraction:

```python
from tanc.model_extractor import inspect

info = inspect(my_model)
# Prints a summary table and returns a ModelInfo object

# Programmatic access
info.linear_layers      # list of LayerInfo
info.conv_layers        # list of LayerInfo
info.has_mixed_types    # bool
```

## Container types

### ModelSnapshot

A single point-in-time capture. Key attributes and methods:

```python
snapshot.weights              # dict[str, ndarray] — layer_name → (N_in, N_out)
snapshot.activations          # dict[str, ndarray] — layer_name → (N_samples, N_neurons)
snapshot.classifications      # ndarray (N_samples, N_classes)
snapshot.predicted_labels     # ndarray (N_samples,)
snapshot.inputs               # ndarray — the input data used
snapshot.epoch                # int or None
snapshot.loss                 # float or None
snapshot.accuracy             # float or None
snapshot.framework            # "pytorch" or "tensorflow"

snapshot.weight_matrices()                # list[ndarray]
snapshot.all_activation_matrices()        # list[ndarray]
snapshot.activation_matrix("fc1")         # ndarray for one layer
snapshot.coupled_weight_activations()     # [(weight, activation)] pairs
snapshot.soro_factors                     # dict[str, {"U","sigma","V"}] — SoRO factors
snapshot.soro_factor("soro")              # {"U","sigma","V"} for one SoRO layer
snapshot.effective_weight("soro")         # (in, out) assembled/derived weight
snapshot.layer_names("weights")           # list[str]
```

### TrainingView

An ordered collection of ModelSnapshots from a training run:

```python
view.snapshots                # list[ModelSnapshot]
view.final_snapshot           # the last snapshot
view.snapshot_at(epoch=25)    # closest snapshot to epoch 25
view.total_epochs             # int

view.weight_trajectory()                  # list[list[ndarray]]
view.activation_trajectory("fc1")         # list[ndarray]
view.losses()                             # list[float]
view.accuracies()                         # list[float]
view.epochs()                             # list[int]
```

## Weight conventions

All weight matrices are normalised to `(N_in, N_out)`:

| Layer type | Raw shape | Returned shape |
|---|---|---|
| PyTorch `nn.Linear` | `(out, in)` | `(in, out)` — transposed |
| PyTorch `nn.Conv2d` | `(out_ch, in_ch, kH, kW)` | `(in_ch*kH*kW, out_ch)` — reshaped + transposed |
| Keras `Dense` | `(in, out)` | `(in, out)` — unchanged |
| Keras `Conv2D` | `(kH, kW, in_ch, out_ch)` | `(kH*kW*in_ch, out_ch)` — reshaped |

## File structure

```
model_extractor/
├── __init__.py        # Public API: extract_model, extract_training, inspect
├── _snapshot.py       # ModelSnapshot, TrainingView dataclasses
├── _inspector.py      # Framework detection, model inspection, layer selection
├── _activations.py    # Framework-neutral activation pooling (flatten/last/mean/…)
├── _pytorch.py        # PyTorch: forward hooks, weight extraction, training loop
└── _tensorflow.py     # TF/Keras: sub-model activations, Callback-based training
```


## Which layer types are extracted

**Weights** are read from these layer kinds:

| layer | what is captured | shape |
| --- | --- | --- |
| `nn.Linear` | the weight matrix, transposed | `(N_in, N_out)` |
| `nn.Conv1d/2d/3d` | the filter tensor, plus a flattened matrix | `(out, in, h, w)` |
| `nn.Embedding` | the table, untransposed — **rows are token vectors** | `(vocab, dim)` |
| `nn.MultiheadAttention` | Q, K, V and O split apart, each `(N_in, N_out)` | see below |
| SoRO layers | the factors `U`, `sigma`, `V`, plus the effective weight | |

**Activations** are captured for any hookable module, including transformer
blocks, with pooling for `(batch, seq, hidden)` outputs.

### Attention

PyTorch stores query, key and value stacked together in `in_proj_weight` with
shape `(3 * embed, embed)`, and the output projection separately. The extractor
splits them, so each projection behaves like an ordinary fully-connected weight:

```python
snap = extract_model(model, X, layer_selection="all_attention")
snap.weights["attn.q_proj"]        # (embed, embed) — likewise k_proj, v_proj, o_proj
snap.attention_weights["attn"]     # {"q": ..., "k": ..., "v": ..., "o": ...}
```

Because they land in `snapshot.weights` under derived names, every existing
builder, topological method and view reaches them with no extra work.

### What is *not* extracted, and what happens

| case | behaviour |
| --- | --- |
| recurrent (`nn.LSTM/GRU/RNN`) | **warns** and skips — their weights live in `weight_ih_l0` / `weight_hh_l0`, and the gates are stacked, so extracting them as one matrix would analyse a meaningless concatenation |
| norm layers (`LayerNorm`, `BatchNorm`) | **warns** and skips for weights — a 1-D per-feature scale is not a linear map and has no rows, columns or graph structure. Their *activations* are captured normally |
| an unknown module with a >2-D weight | **warns**, flattens to 2-D, and does **not** pretend it is a convolution filter bank |
| attention *maps* (`softmax(QK^T/sqrt(d))`) | not yet captured — they are input-dependent and per-head, a third kind of object alongside weights and activations |

Each of these used to pass silently. They now say what they declined to do.
