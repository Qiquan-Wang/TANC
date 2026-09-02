<img src="https://raw.githubusercontent.com/Qiquan-Wang/TANC/main/assets/logo.png" alt="TANC logo — a tank whose treads trace a figure-eight" align="left" width="200" />

# TANC

**T**opological **A**nalysis of **N**eural networks through **C**omposition — a
composable extraction → graph → topology → visualisation pipeline for studying
trained networks with persistent homology, Mapper, and intrinsic dimension.

<br clear="left" />

Licensed under **AGPL-3.0-or-later** (see
[LICENSE](https://github.com/Qiquan-Wang/TANC/blob/main/LICENSE)); the required
dependency giotto-tda is AGPLv3, so a combined work cannot be redistributed
under permissive terms.

---

## Getting started

A first-pass tour for new users. For deeper module-by-module references see
the per-module `README.md` files.

## Install

```bash
pip install -e .
```

## The mental model — four choices

The core idea is **mix and match**. A trained network has several *spaces* you can
study. You pick one, pick how to turn it into a graph, pick a topological method, and
pick what to read off. Every analysis is a point along four axes:

```
   1. SPACE              2. CONSTRUCTION       3. METHOD        4. OUTPUT
   weights         ──>   weight_graph     ──>  PH         ──>   diagram / barcode
   weights (rows/cols)   point_cloud_graph     Mapper           Mapper graph
   activations           activation_graph      dimension        scaling plot
   inputs + labels       labelled_complex                       statistics / number
   weight trajectory     polyhedral / traj
```

Mechanically these flow through four modules in series — `model_extractor` produces a
`ModelSnapshot` / `TrainingView` (**axis 1**), `graph_builder` produces a `GraphBundle`
(**axis 2**), `topo_tools` produces a `TopoResult` (**axis 3**), and `visualisation`
renders it (**axis 4**):

```
   model → model_extractor → graph_builder → topo_tools → visualisation
```

You can enter at whichever stage you have data for. Hand `TDAPipeline.fit()` a model,
a snapshot, a list of weight matrices, a list of activation arrays, or even a
precomputed distance matrix — the pipeline figures out where you joined.

Reproducing a published method is **not** a separate feature — it is just a
*shortcut* that pre-fills these four choices (`TDAPipeline.from_paper("...")`). The
`TUTORIAL.ipynb` notebook trains one model and composes several analyses on it; the
worked examples below show the same idea in miniature.

## Prefer a point-and-click interface? Use the Visual Builder

If you'd rather not write code by hand, the toolkit ships with a **Visual
Builder** — a browser front-end that lets you make all four choices above by
clicking, then assembles and runs the analysis for you. Launch it from the repo
root (with your environment active):

```bash
python web/server.py      # then open http://localhost:8000
# or:  ./web/serve        # same thing; auto-finds the project's venv
```

In the browser you can, without writing any code:

- **Build a network** by clicking layers (Linear, Conv2d, SoRO, Transformer, …) —
  shapes are inferred automatically.
- **Pick a dataset** (MNIST, FashionMNIST, KMNIST, CIFAR-10/100 — downloaded with
  pure numpy, no torchvision) and set training options (epochs, optimizer, LR).
- **Choose what to record** (weights and/or activations, from which layers) and
  an **analysis**: one of the 18 published **paper presets**, a **custom
  pipeline** (graph builder + PH / Mapper / dimension), or an **over-training**
  study that recomputes a topological summary each epoch.
- **Generate the Python** live — then **Copy** it, **Download** the `.py`, or hit
  **Run ▷** to execute it in place and watch the log stream with a live epoch
  progress bar, with the result figures appearing at the end.

Toggle **PyTorch** or **TensorFlow** at the top and the generated code switches
accordingly. The `Run ▷` button executes code locally, so keep the server on your
own machine — see [`web/README.md`](https://github.com/Qiquan-Wang/TANC/blob/main/web/README.md)
for the full feature tour and
kernel-selection notes.

> Just want the code and not the server? Open `web/index.html` directly in a
> browser, build your experiment, and copy/download the script to run yourself.

## 60-second start

```python
import tanc as tda
from tanc import TDAPipeline

tda.tour()                                 # printed overview of the package

# Compose your own: pick a construction (builder) and a method (tool).
pipe = TDAPipeline(builder="activation_graph", tool="ph", tool_kwargs={"max_dim": 1})
pipe.explain()                             # what will the pipeline do?
result = pipe.fit(activation_matrices)     # fit a space → TopoResult
result.describe()                          # populated fields + available plots
result.plot("diagram")                     # read off an output

# Shortcut: a preset just pre-fills builder/tool for a published method.
pipe = TDAPipeline.from_paper("rieck2019")
result, fig = pipe.reproduce(weight_matrices)   # fit + render in one call
```

See **`composing`** (in the docs) for the full menu of spaces, builders, tools, and
outputs, and which combinations are valid.

## Starting from just a trained model

The examples below assume you already have arrays (weight matrices, activations,
...). If all you have is a **trained model**, you don't need to extract anything
by hand — `pipe.fit_model(model, X)` runs extraction *and* the pipeline in one
call. The framework (PyTorch / TensorFlow) is auto-detected from the model
object.

```python
import torch
import torch.nn as nn
from tanc import TDAPipeline

# ── 1. Your model. Any trained torch.nn.Module or tf.keras.Model works;
#       here is a tiny fully-connected net so the snippet is self-contained.
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_classes)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = MLP(in_dim=784, n_classes=10)
# ... train `model` as usual ...
model.eval()

X = torch.randn(300, 784)                  # a representative batch of inputs

# ── 2. Model in, TopoResult out — no manual extraction step.
pipe = TDAPipeline.from_paper("watanabe2021")
pipe.explain()                             # what the pipeline will do
result = pipe.fit_model(model, X)          # extract weights + run PH in one call
result.describe()
fig = result.plot("diagram")
```

`fit_model` pulls whatever the preset needs off the model. For presets that work
on **activations** (Mapper, intrinsic dimension, ...) ask for that representation:

```python
# Intrinsic dimension per layer, straight from the model.
id_result = TDAPipeline.from_paper("ong2026").fit_model(
    model, X, representation="activations")
print("mean ID across layers:", id_result.dimension)
```

For a transformer, add `activation_pooling="last"` (or `"mean"`) so each
sequence's `(seq, hidden)` block collapses to one `hidden`-space point (the
final-token summary of a causal decoder) instead of being flattened:

```python
id_result = TDAPipeline.from_paper("ong2026").fit_model(
    llm, prompts, representation="activations",
    layer_selection=["blocks.0", "blocks.1"], activation_pooling="last")
```

Prefer the explicit two-step form when you want to inspect or reuse the
extracted snapshot (e.g. feed it to several pipelines):

```python
from tanc.model_extractor import ModelExtractor

snap = ModelExtractor(model, aspects=["weights", "activations"]).extract(X)
result_ph = TDAPipeline.from_paper("watanabe2021").fit(snap)
result_id = TDAPipeline.from_paper("ong2026").fit(snap)
```

To analyse a model **while it trains** (capturing snapshots across epochs), use
`TrainingExtractor` / `pipe.fit_training(...)` — see *Inspecting and comparing*
below and `tda.help("extract")`.

## Three worked examples

### A. Persistent homology of a trained model — Rieck et al. (2019)

```python
from tanc import TDAPipeline

# Either pass a list of weight matrices directly...
pipe = TDAPipeline.from_paper("rieck2019")
result = pipe.fit(weight_matrices)         # list of (N_in, N_out) ndarrays

# ...or hand in a ModelSnapshot extracted from a trained network.
from tanc.model_extractor import ModelExtractor
extractor = ModelExtractor(my_torch_model, aspects=["weights"])
snap = extractor.extract(X_test)
result = pipe.fit(snap)

# Look at what came out
result.describe()
#   TopoResult(tool='ph')
#     paper       : Rieck et al. (2019)
#     data        : ph_result (H0: 12 bars)
#     statistics  : H0_total_persistence, H0_persistence_norm, ...
#     plots       :
#       .plot('diagram') — Birth-vs-death scatter ...
#       .plot('barcode') — Horizontal bars from birth to death...
#       .plot('betti_curve') — Step function of Betti_d ...

H0 = result.diagram(dim=0)                 # shortcut for result.ph_result.diagrams[0]
fig = result.plot()                        # uses the preset's default kind
fig = result.plot("barcode")               # or pick explicitly
```

### B. Mapper on activation space — Rathore et al. (2021)

```python
pipe = TDAPipeline.from_paper("rathore2021")
result = pipe.fit(per_layer_activations)   # list[(N_samples, N_neurons)]

result.mapper.graph                        # networkx Graph
result.mapper.stats                        # n_nodes, density, ...

fig = result.plot("graph")
```

### C. Intrinsic dimension of activations — Ong et al. (2026)

```python
pipe = TDAPipeline.from_paper("ong2026")
result = pipe.fit(per_layer_activations)

result.dimension                           # mean ID across layers
result.dimension_result["id_estimates"]    # per-layer values
fig = result.plot("id_layers")
```

## Sweeping Mapper — is this a discovery or just my cover?

Running Mapper once gives you a graph. It does not tell you whether that graph
describes your data or reflects the cover you happened to choose — and **every**
cover produces a graph, on any data at all, including noise.

`MapperStudy` trains a seed population, sweeps the parameters that are usually
guessed, and reports the **nerve of the cover** alongside every graph, so the two
can be compared:

```python
from tanc import MapperStudy          # also: MapperGrid, train_population
from tanc.topo_tools import DBSCANCells

study = MapperStudy(
    model_fn     = make_net,               # returns a fresh untrained model
    train_data   = (X, y),
    n_models     = 20, epochs = 30,
    criterion    = nn.CrossEntropyLoss(),
    optimizer_fn = lambda m: torch.optim.Adam(m.parameters()),

    layer        = ["conv1", "conv2"],     # swept
    view         = ["rows", "kernel"],     # swept — see below
    lens         = ["pca2", "l2"],
    n_intervals  = [10, 20, 30],
    overlap      = [0.3, 0.5],
    clusterer    = [DBSCANCells(eps=("quantile", q)) for q in (1, 5, 25)],
)

study.validate()                    # checked BEFORE anything expensive runs
result = study.run("runs/study01")

result.plateaus()[0]["spans"]       # parameter regions with stable topology
result.errors                       # a property, not a method
result.leading(measure="b1_excess", min_node_median=20)

from tanc.visualisation import plot_graph_panel
rows = result.rows()
plot_graph_panel([result.graph(r) for r in rows], suptitle="the whole sweep")
```

A **view** decides what a single point *is*, and the answer can differ between
them. For a 2-D weight the views are `rows`, `cols`, `full`, the norm and sum
profiles, and the Gram readings. For a convolutional `(out, in, h, w)` weight
each view instead names which axes index points — `rows` and `cols` are the rows
and columns *of the kernel*, alongside `kernel`, `out_channel`, `in_channel` and
`tap`. See `docs/parameters.rst` for the full table.

`plot_graph_panel` draws the whole sweep as one contact sheet, which is how you
tell a structure that survives a range of parameters from one that appears at a
single lucky setting.

**`b1_excess`** is the graph's first Betti number *minus its cover's*. The same
noisy circle, same clusterer, only the lens dimension changed:

| lens | `b1` | nerve `b1` | excess | reading |
|------|-----:|-----------:|-------:|---------|
| 1-D  |    1 |          0 |  **1** | a real loop |
| 2-D  |   37 |         37 | **≈0** | the cover's own loops |

Read `b1` alone and the second row looks like a spectacular finding.

`validate()` catches two things before the first model is built: a training
configuration that would not capture what the grid asks for (the expensive
mistake — three hours of training, then discovering the layer was never
recorded), and grid configurations that are internally contradictory.

⚠️ **Overlap convention.** `overlap` is the fraction of an interval shared with
its neighbour, matching KeplerMapper and giotto-tda. Some hand-rolled Mapper code
instead widens intervals by a fraction of the *spacing*, where a nominal `0.67`
is only `0.40` here. Use `convert_overlap()` when porting parameters.

Runs never overwrite: an existing directory name gets `-002`, and the path used
is printed up front. Node membership is saved with each graph, so a finished
sweep can be recoloured and re-examined without recomputing anything.

Full guide: [`docs/sweep_overview.rst`](https://github.com/Qiquan-Wang/TANC/blob/main/docs/sweep_overview.rst).

## Inspecting and comparing

```python
# What can I plot from this result?
result.plots_available()
# ['diagram', 'barcode', 'betti_curve']

# Side-by-side comparison panels
fig = pipe.compare({"Model A": data_a, "Model B": data_b})

# Multi-snapshot training run — hand over the raw dataset, no loaders needed
from tanc.model_extractor import extract_training
view = extract_training(
    model, extract_data=X_test,
    train_data=(X_train, y_train), val_data=(X_test, y_test),
    batch_size=128,                                   # adjustable
    criterion=criterion, optimizer=optimizer,         # lr lives in the optimizer
    epochs=20, snapshot_every=2,
)

from tanc.visualisation import (
    plot_training_curves, plot_id_trajectory_all_layers,
    plot_ph_dimension_over_training, plot_diagram_evolution,
)
plot_training_curves(view)
plot_id_trajectory_all_layers(view, method="global", layout="grid")
plot_ph_dimension_over_training(view, window=20)
plot_diagram_evolution(view, n_panels=6)

# Track any preset's scalar output across training — one line per layer —
# without a hand-rolled per-snapshot loop:
series = TDAPipeline.from_paper("watanabe2021").over_training(
    view, measures=["H1_total_persistence"], layers=["fc1", "fc2"])
series.plot()                                     # TrajectorySeries: numbers + .plot()
```

## Class-style vs function-style entry points

The toolkit favours a consistent **builder + run** pattern so every
stage feels the same:

| Stage | Class form (preferred) | One-call shortcut |
|---|---|---|
| Extract from a trained model | `ModelExtractor(model, ...).extract(data)` | `extract_model(model, data, ...)` |
| Train and capture snapshots | `TrainingExtractor(model, ...).run(epochs=...)` | `extract_training(model, ..., epochs=...)` |
| Run a full TDA pipeline | `TDAPipeline.from_paper(...).fit(data)` | (none — pipelines are always class-based) |
| Trajectory-dimension analysis | `WeightTrajectoryAnalyser.from_paper(...).fit(bundle)` | `run_trajectory_dimension(bundle, method=...)` |

`ModelExtractor` and `TrainingExtractor` both **auto-detect the framework
from the model** — pass a `torch.nn.Module` and the PyTorch backend
runs, pass a `tf.keras.Model` and the TensorFlow backend runs. You only
need to supply the kwargs that match.

Every class form has the same three methods so you can move between
them without re-learning: **`.from_paper(...)`** (where applicable)
preconfigures a preset, **`.explain()`** prints the plan in plain
English, and **`.run()` / `.extract()` / `.fit()`** does the work and
returns the typed result.

```python
# Train and continue training in two separate calls — only possible with
# the class form because state lives on the extractor.  Framework is
# detected from `model` automatically.
extractor = TrainingExtractor(
    model=model,
    train_data=(X_train, y_train), val_data=(X_test, y_test),  # raw dataset
    batch_size=128,                                             # adjustable
    criterion=criterion, optimizer=optimizer,
    extract_data=X_test, snapshot_every=5,
)
view = extractor.run(epochs=50, target_accuracy=0.9)
view = extractor.run(epochs=10, continue_training=True)

# If you want the framework choice spelled out in code, use the
# equivalent explicit factories instead:
extractor = TrainingExtractor.for_pytorch(model=model, train_data=(X, y), ...)
extractor = TrainingExtractor.for_tensorflow(model=keras_model, train_data=..., ...)
```

### What's adjustable in a training run

| Knob | Where you set it | Notes |
|---|---|---|
| **batch_size** | `TrainingExtractor(batch_size=…)` | Used when a loader is built from raw `(X, y)` / `Dataset`. Ignored if you pass a ready `DataLoader`. |
| **learning rate** | inside the **optimizer** you pass | e.g. `torch.optim.Adam(model.parameters(), lr=1e-3)` (PyTorch) or `compile_kwargs={"optimizer": ...}` (TF). |
| **LR schedule** | `TrainingExtractor(scheduler=…)` | Any PyTorch scheduler, e.g. `CosineAnnealingLR(opt, T_max=epochs)`; stepped once per epoch. |
| **optimizer / loss** | `optimizer=` / `criterion=` (PyTorch), `compile_kwargs=` (TF) | Any `torch.optim` / `nn` loss, or Keras-compiled equivalents. |
| **shuffle** | `TrainingExtractor(shuffle=…)` | Shuffles the training loader built from raw data (validation never shuffled). |
| **epochs / target_accuracy** | `.run(epochs=…, target_accuracy=…)` | Per-run; `target_accuracy=None` runs the full `epochs`. |
| **snapshot_every / snapshot_schedule** | construction | How often a snapshot is captured (`"epoch"` or `"iteration"`). |
| **aspects / layer_selection / device** | construction | What to capture, from which layers, on which device. |

## One-call dashboards

```python
from tanc.visualisation import (
    plot_training_summary, make_training_animation, plot_pipeline_diagram,
)

# 2x3 panel of loss/acc, weight-norm, ID, H0 total persistence, Betti_1,
# and Wasserstein distance to the final diagram. Failed panels degrade
# gracefully so the figure still renders.
plot_training_summary(view)

# Diagram / barcode / landscape / image animation across snapshots
make_training_animation(view, kind="diagram", out="train.gif")

# Flowchart of a configured pipeline (great for paper-example READMEs)
plot_pipeline_diagram(TDAPipeline.from_paper("rieck2019"))
```

## Stable PH summaries and diagnostics

```python
from tanc.visualisation import (
    plot_persistence_landscape, plot_persistence_image,
    plot_diagram_distance_matrix, plot_ph_statistic_pairplot,
    plot_id_with_uncertainty, plot_id_qq,
)

plot_persistence_landscape(result.ph_result, dim=1, k_max=5)
plot_persistence_image(result.ph_result, dim=1, resolution=25)

plot_diagram_distance_matrix(view, metric="wasserstein", dim=1)
plot_ph_statistic_pairplot(view, dim=0,
    stats=("total_persistence", "persistence_norm", "persistence_entropy"))

# Mean +/- 1 SD across bootstrap subsamples, per layer
plot_id_with_uncertainty(activations, method="global", n_bootstraps=20)

# QQ plot of log(log mu) values to inspect tail behaviour
plot_id_qq(distance_matrix, dist="norm")
```

## Saving & loading results

Every result container persists with a `.save(path)` method and a matching
`.load(path)` classmethod, so you can stop at any stage and pick up later
without recomputing. Files use a single pickled `.tda` envelope (the suffix is
added automatically if you omit it).

```python
# Stage 1 — the extracted trajectory (skip retraining next time)
view.save("runs/trajectory.tda")
from tanc.model_extractor import TrainingView
view = TrainingView.load("runs/trajectory.tda")     # ModelSnapshot.load(...) too

# Stage 2 — the built graph
bundle.save("runs/graph.tda")
from tanc.graph_builder import GraphBundle
bundle = GraphBundle.load("runs/graph.tda")

# Stage 3 — the computed topology (skip recomputing expensive PH)
result = pipe.fit(W)
result.save("runs/watanabe_ph.tda")
from tanc.topo_tools import TopoResult
result = TopoResult.load("runs/watanabe_ph.tda")    # diagrams, Mapper graph,
                                                    # dimension results, stats

# Stage 4 — the figure, straight to disk (format from the extension)
result.plot("diagram", save="figs/diagram.pdf")
pipe.reproduce(W, save="figs/run.png")              # per-layer runs → run_0.png, run_1.png, …
```

`load` is type-checked: `TopoResult.load("graph.tda")` raises rather than
silently handing back a `GraphBundle`.

> **Note.** `.tda` files are pickles — load only files you created or trust, and
> treat them as cache, not a long-term archive (numpy / networkx pickles are not
> guaranteed stable across major dependency upgrades).

## Common patterns

```python
# I have a single layer's activations. What's its intrinsic dimension?
from tanc.topo_tools import run_activation_id
res = run_activation_id([activations], method="global")
res["id_estimates"][0]

# I have a sequence of training snapshots. What's the PH fractal dim
# of the weight trajectory?
from tanc.graph_builder import build_weight_trajectory
from tanc.topo_tools import compute_ph_dimension
bundle = build_weight_trajectory(view.weight_trajectory(), distance="euclidean")
res = compute_ph_dimension(bundle.matrix, alpha=1.0)
res["ph_dimension"]

# I want the same plot for every paper-preset PH method, side by side.
import matplotlib.pyplot as plt
for name in ["rieck2019", "gebhart2019", "naitzat2020"]:
    res, fig = TDAPipeline.from_paper(name).reproduce(snapshot)
    fig.suptitle(name)
```

## Where to look next

- `tanc.tour()` / `tanc.help('<topic>')` — interactive overview
- `TDAPipeline.list_presets()` / `TDAPipeline.describe_preset(name)`
- `tanc/<module>/README.md` — module-level reference
- `paper_reproduce/` — one notebook per paper preset
