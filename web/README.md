# TANC — Visual Builder

A graphical, no-code front-end for [`tanc`](https://github.com/Qiquan-Wang/TANC/tree/main/tanc). Assemble a
neural network, pick a dataset and a topological/geometric analysis, and it
**generates a runnable Python script** that uses the toolkit to train the
network, record weights/activations, and run the analysis.

![flow](https://img.shields.io/badge/build-model%20%E2%86%92%20data%20%E2%86%92%20record%20%E2%86%92%20analyse%20%E2%86%92%20code-4f46e5)

## What you can do

0. **Framework** — toggle **PyTorch** or **TensorFlow** at the top of the Model
   card; the generated code, model, and training switch accordingly (Keras for
   TF). *Transformer* layers are PyTorch-only.
1. **Model** — click layers from the palette to build a network:
   Linear, Conv2d, ReLU, MaxPool2d, Flatten, BatchNorm, Dropout, Embedding,
   **SoRO** (Sum-of-Rank-One factored FC), and **Transformer** encoder blocks.
   Rename, reorder (drag or ↑/↓), and edit each layer's hyper-parameters.
   **Shapes are inferred automatically** — each layer shows its `in → out`
   shape, and input-side sizes are filled for you: a Conv2d's `in_channels`
   from the previous layer (or the dataset), a Linear's `in_features` from the
   flattened feature count, same-conv `padding` from the kernel, BatchNorm
   `num_features`, and the **final layer's `out_features` = number of classes**.
   Auto fields are read-only (tagged *auto*); you set the rest. `nn.Flatten()`
   is inserted wherever an image tensor meets a Linear/SoRO.
2. **Data** — choose a built-in dataset (MNIST, FashionMNIST, KMNIST, CIFAR-10,
   CIFAR-100). These are downloaded and parsed with **pure numpy — no
   torchvision required** (MNIST-family `idx` files, CIFAR pickles), cached
   under `./data`. Or point at a custom name/URL (a loader stub is emitted).
3. **Training** — epochs, optimizer, learning rate, device.
4. **Record** — which layers to capture (a preset group like `all_linear` /
   `soro`, or specific layers) and what: **weights** and/or **activations**
   (with transformer token-pooling: last / mean / first / max).
5. **Analysis** — a **paper preset** (18 published methods via
   `TDAPipeline.from_paper`), a **custom pipeline** (graph builder + tool —
   PH / Mapper / dimension — with parameters, representation and plot; builders
   include the weight graph, weight rows/cols point clouds, activation graphs,
   conv-kernel graphs, decision-boundary and ReLU-region complexes), or
   **over training** (recompute a topological summary each epoch). Over-training
   tracks: diagram distance to the initial / final / **previous** epoch, PH
   statistics, Betti numbers, the epoch×epoch diagram-distance heatmap, a
   PH-statistic pairplot, intrinsic dimension of one layer (from activations
   **or weights**, rows or columns — so individual neurons' response profiles
   can be the points) or of all layers, and the sliding-window PH / magnitude
   dimension of the weight trajectory. Every PH-based track has a
   **construction selector** — the default multipartite weight graph (with
   edge-weight choice), a point cloud of one weight matrix's rows/columns, or
   an activation cloud (samples or neurons as points). For Mapper analyses,
   tick **Interactive 3-D graph** to get a rotatable/zoomable plotly graph
   instead of a static image.

   **Multiple instances** (Training → *Model instances*): train N copies with
   different seeds and pool them — conv-kernel Mapper across models
   (`gabrielsson2019`), or the **weight trajectories of N runs from one shared
   init** coloured by run (`gabella2021`).
6. **Generate** — the Python updates live. **Copy** it, **Download** `.py`, or
   **Run ▷** it in place (needs the runner below).

## Use it

### First-time setup 

Nothing here is tied to a specific machine — you just need a Python environment
with the dependencies. The one-time `setup` script creates one from
`docs/requirements.txt`:

```bash
# macOS / Linux
./web/setup          # creates <repo>/.venv with everything, then:
./web/serve          # launch → http://localhost:8000
```
```bat
REM Windows
web\setup.bat
web\serve.bat
```

Already have an environment (venv or conda) with the deps? Skip `setup` — just
**activate it and run `./web/serve`** (or `python web/server.py`). 

### Just generate code (no server needed)

Open `web/index.html` in a browser. Build your experiment, copy or download the
script, and run it yourself:

```bash
pip install torch          # (PyTorch) — or: pip install tensorflow   (TensorFlow)
                           # + the tanc package on your PYTHONPATH
python tda_experiment.py
```

The generated script is self-contained: it defines the model (including a
`SoRO` class when used), downloads/parses the data with numpy (no torchvision),
trains, and runs the pipeline, saving figures as `tda_result_*.png`. It targets
whichever framework you selected — plain PyTorch, or Keras/TensorFlow.

### Generate *and run* from the browser

```bash
./web/serve                         # launcher: uses the project's .venv automatically
# or:  python web/server.py         # (uses whatever `python` you run it with)
# then open http://localhost:8000
```

`./web/serve` finds the project's virtual-env (`.venv` near the repo, or an
active `$VIRTUAL_ENV`/`$CONDA_PREFIX`) so torch / tensorflow / giotto-tda /
tanc are all present. If you run `server.py` directly with an incomplete
interpreter, the runner **still** falls back to the best detected environment
(one with giotto-tda) rather than failing on Mapper.

The server serves the UI and adds a working **Run ▷** button that executes the
generated code (in a temp dir, with the repo on `PYTHONPATH`) and **streams the
log live** — stdout lines appear as they are printed, an **epoch progress bar**
tracks training (it understands the generated loops', Keras's and
`TrainingExtractor`'s epoch lines), and the figures arrive at the end.
Stdlib-only — no extra install.

#### Choosing the Python environment ("kernel")

The code needs `torch` **or** `tensorflow` (per your framework choice) plus
`tanc`, which may live in a
specific venv/conda env. The **Run with kernel** dropdown (top of the code
panel) lists detected interpreters — local `.venv`/`venv`/`env` folders near
the repo, `$VIRTUAL_ENV` / `$CONDA_PREFIX`, conda/mamba envs under
`~/miniconda3|anaconda3|miniforge3|mambaforge/envs/*`, and `pyenv` versions —
each **actually imports** the packages (so a broken install shows as missing)
and is badged `✓` when it has the framework you selected (torch/tensorflow) plus
`tanc`. **Run ▷** executes with the selected one; press **⟳** to rescan.

If your env isn't auto-detected, the simplest fix is to **launch the server
from it** — it's always listed as *(current)*:

```bash
conda activate tanc        # or: source /path/to/.venv/bin/activate
python web/server.py
```

> ⚠️ **Local use only.** `Run ▷` executes arbitrary Python on your machine. Do
> not expose this server to a network. Set `TDA_RUN_TIMEOUT` (seconds) and
> `PORT` via environment variables if needed.

## How the generated code maps to the toolkit

| UI choice | Generated call |
|---|---|
| Layer stack | `nn.Sequential(OrderedDict([...]))` / `keras.Sequential([...])` with **named** layers |
| SoRO layer | a `SoRO(nn.Module)` / `SoRODense(keras.layers.Layer)` implementing `soro_factors()` (the toolkit's protocol) |
| Dataset | numpy loader (idx / CIFAR pickle) → `X_tr/y_tr/X_te/y_te` (NCHW tensors for torch, NHWC arrays for TF) |
| Record (snapshot analysis) | `pipe.fit_model(model, X_te, aspects=…, layer_selection=…, representation=…, activation_pooling=…)` |
| Record (trajectory analysis) | `TrainingExtractor(...).run(epochs)` → `view`; `pipe.fit(view)` |
| Paper preset | `TDAPipeline.from_paper("<key>")` |
| Custom pipeline | `TDAPipeline(builder=…, builder_kwargs=…, tool=…, tool_kwargs=…)` |
| Builder `point_cloud_graph` | a single weight matrix as a cloud of its rows or columns — `builder_kwargs={"orientation": "rows"｜"cols", "metric": …}`; record one layer |
| Plot | `result.plot("<kind>", save="tda_result_i.png")` |

**Snapshot vs trajectory** is chosen automatically: dimension estimators over
the *weight trajectory* (Birdal, Dupuis, Andreeva, Gabella) and the
`trajectory_dimension` estimator train while capturing per-epoch snapshots;
everything else trains once and analyses the final model.

## Files

```
web/
├── index.html   # the app markup
├── styles.css   # theme-aware styling (light/dark)
├── app.js       # UI logic + the Python code generator
├── server.py    # optional stdlib runner (serve + execute)
└── README.md
```

## Notes & limits

- Shapes are **not** auto-validated — you set each layer's dimensions (the input
  shape / flattened size for the chosen dataset is shown as a hint). An
  `nn.Flatten()` is auto-inserted before a leading Linear/SoRO on image data.
- A few exotic presets need specific inputs (e.g. Ramamurthy → inputs+labels,
  Liu → ReLU activation patterns, Gabrielsson → conv kernels); the UI flags
  these with a note and emits best-effort code.
- Both **PyTorch and TensorFlow** are generated. Note two TF limits: the
  builder skips *Transformer* layers (no single-layer Keras equivalent), and
  the toolkit's TF **activation** capture can be limited for some Sequential
  models — weight-based analyses are the most reliable in TF.
