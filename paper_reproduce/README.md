# Paper reproductions

Minimum-line reproductions of every `TDAPipeline.from_paper(...)` preset. Each notebook compresses the relevant analysis to `TDAPipeline.from_paper("<preset>").reproduce(data)` plus a paper-specific follow-up plot. Every notebook now ships with a **dataset & model** block at the top showing the data, models, headline result, and reference GitHub repo (when one exists) so you know exactly what to swap in.

**10 of the 18 notebooks now run on real data** — they load MNIST or CIFAR-10/100 from `data/`, train a tiny model briefly, and feed the *real* extracted weights / activations into the `TDAPipeline` preset. The remaining 8 are either paper-synthetic (toy 2-D/3-D manifolds) or waiting on datasets we haven't downloaded yet (LLM / ImageNet).

Headline paper numbers will *not* match — the real-data notebooks use deliberately tiny models (SmallMLP / SmallCNN, ~50k params) trained briefly on CPU for a couple of minutes. The toolkit's pipeline is the same one the paper used; the *qualitative* behaviour (e.g. Neural Persistence growing with depth, Betti decreasing across layers, magnitude-dimension trend) should appear. Swap in a bigger model and longer training for closer match.

**See [DATASETS.md](DATASETS.md) for the full list of datasets and pre-trained checkpoints needed across all 18 notebooks**, with size estimates and download recommendations.

## Catalogue

🟢 = uses real data + briefly-trained model, 🟡 = paper-synthetic data (toy manifolds), ⚪ = waiting on a dataset we haven't fetched.

| Notebook | Mode | Paper | Tool | Reference code |
|---|---|---|---|---|
| [watanabe2021](watanabe2021.ipynb) | 🟢 CIFAR-10 + CNN_FCN (belt diagram + PHPM pruning) | Watanabe & Yamana (2020) | PH | [satoru-watanabe-aw/DNNtopology](https://github.com/satoru-watanabe-aw/DNNtopology) |
| [rieck2019](rieck2019.ipynb) | 🟢 MNIST + SmallMLP | Rieck et al. (2019) — Neural Persistence | PH | [BorgwardtLab/Neural-Persistence](https://github.com/BorgwardtLab/Neural-Persistence) |
| [gebhart2019](gebhart2019.ipynb) | 🟢 MNIST + SmallMLP | Gebhart et al. (2019) | PH | — |
| [lacombe2021](lacombe2021.ipynb) | 🟢 MNIST + SmallMLP | Lacombe et al. (2021) — Topological Uncertainty | PH+TU | — |
| [naitzat2020](naitzat2020.ipynb) | 🟡 synthetic toy manifolds | Naitzat et al. (2020) | PH | — |
| [karuppiah2025](karuppiah2025.ipynb) | 🟡 synthetic | Karuppiah et al. (2025) | PH | — |
| [ballester2024](ballester2024.ipynb) | 🟢 CIFAR-10 + VGG-like population (gap regression) | Ballester et al. (2024) | PH | [rballeba/PredictingGeneralizationGapUsingPersistentHomology](https://github.com/rballeba/PredictingGeneralizationGapUsingPersistentHomology) |
| [ramamurthy2019](ramamurthy2019.ipynb) | 🟡 synthetic | Ramamurthy et al. (2019) | PH | — |
| [liu2023](liu2023.ipynb) | 🟡 synthetic | Liu et al. (2023) | PH | [cglrtrgy/GoL_Toolbox](https://github.com/cglrtrgy/GoL_Toolbox) |
| [rathore2021](rathore2021.ipynb) | ⚪ no torchvision | Rathore et al. (2021) — TopoAct | Mapper | [tdavislab/TopoAct](https://github.com/tdavislab/TopoAct) |
| [zhou2023](zhou2023.ipynb) | 🟢 CIFAR-10 + SmallCNN | Zhou et al. (2023) | Mapper | [tdavislab/mapper-compare](https://github.com/tdavislab/mapper-compare) |
| [gabrielsson2019](gabrielsson2019.ipynb) | ⚪ no torchvision | Gabrielsson & Carlsson (2019) | Mapper | — |
| [gabella2021](gabella2021.ipynb) | 🟢 MNIST + SmallMLP | Gabella (2021) | Mapper | [maximevictor/topo-learning](https://github.com/maximevictor/topo-learning) |
| [ruppik2025](ruppik2025.ipynb) | ⚪ no LLM datasets yet | Ruppik et al. (2025) | Dimension | [aidos-lab/Topo_LLM_public](https://github.com/aidos-lab/Topo_LLM_public) |
| [ong2026](ong2026.ipynb) | 🟡 synthetic (no NN) | Ong et al. (2026) | Dimension | — |
| [birdal2021](birdal2021.ipynb) | 🟢 CIFAR-10 + SmallCNN | Birdal et al. (2021) | Dimension | [tolgabirdal/PHDimGeneralization](https://github.com/tolgabirdal/PHDimGeneralization) |
| [dupuis2023](dupuis2023.ipynb) | 🟢 CIFAR-10 + SmallCNN | Dupuis et al. (2023) | Dimension | [benjiDupuis/data_dependent_dimensions](https://github.com/benjiDupuis/data_dependent_dimensions) |
| [andreeva2024](andreeva2024.ipynb) | 🟢 CIFAR-10 + SmallCNN | Andreeva et al. (2024) | Dimension | [aidos-lab/magnipy](https://github.com/aidos-lab/magnipy) |

## Real-data flow (10 notebooks)

All 10 🟢 notebooks share the same template — three setup cells then `TDAPipeline.from_paper(...).reproduce(data)`:

```python
from paper_reproduce._torch_setup import load_cifar10, SmallCNN, train_briefly
X_tr, y_tr = load_cifar10("train", n_samples=1500)
X_te, y_te = load_cifar10("test",  n_samples=400)
model = SmallCNN(in_channels=3, num_classes=10, width=8)
view  = train_briefly(model, X_tr, y_tr, X_te, y_te, epochs=2)

# Pipeline input drawn from real extracted weights / activations
snap = view.final_snapshot
data = snap.weight_matrices()    # or snap.coupled_weight_activations() / view.weight_trajectory()

from tanc import TDAPipeline
pipe = TDAPipeline.from_paper("watanabe2021")
result, fig = pipe.reproduce(data)
```

[`_torch_setup.py`](_torch_setup.py) holds `load_mnist`, `load_fashion_mnist`, `load_cifar10`, `load_cifar100`, `SmallMLP`, `SmallCNN`, `train_briefly`, `extract_snapshot` — single source of truth for the model/training boilerplate.

Per-notebook timing on CPU: ~3–5 s for MNIST + SmallMLP, ~30–90 s for CIFAR-10 + SmallCNN, depending on `n_samples` and `epochs`. The defaults in each notebook are tuned to run end-to-end in a few minutes.

The reproduction pattern across all PH and dimension presets is the same:

```python
pipe = TDAPipeline.from_paper("<preset>")
pipe.explain()                       # what will run
result, fig = pipe.reproduce(data)   # fit + render
```

For Mapper papers the second line returns one figure per layer when given a list of activation arrays. The `lacombe2021` notebook uses `TopologicalUncertainty` directly because it's a fit/score class, not a pipeline.

## Model-first vs. explicit extraction

**Guiding principle: if you use a paper preset, the pipeline extracts the data for you — you don't pull weights/activations by hand.** The model-first entry points do the extraction:

```python
pipe.fit_model(model, X, representation="activations", layer_selection=...)  # extract + run
pipe.fit_each(models, X, ...)        # one result per model (e.g. ballester2024 ASDSQ)
pipe.reproduce(view)                 # a TrainingView / ModelSnapshot is auto-translated
```

So a notebook should only call `extract_snapshot(...)` / `snap.weight_matrices()` / `snap.all_activation_matrices()` when the analysis needs a **non-standard transform the adapter can't express**. Every notebook that still extracts manually says *why* in-cell. The audit:

| Category | Notebooks |
|---|---|
| **Model-first, no manual extraction** | `andreeva2024`, `ballester2024`, `birdal2021`, `dupuis2023`, `karuppiah2025`, `rathore2021`, `ruppik2025` |
| **No model to extract** (generated manifolds / classifiers / weight-trajectory cloud) | `ong2026`, `ramamurthy2019`, `gabella2021` |
| **Core analysis model-first; manual only for a *downstream* step** | `gebhart2019` (φ=`|w·h|` for the pathway plot), `rieck2019` (per-epoch NP from the training view), `watanabe2021` (weights for PHPM pruning) |
| **Manual by necessity — special prep no preset can express** | `liu2023` (binarise hidden ReLU patterns, drop the output layer), `gabrielsson2019` (harvest + contrast-normalise conv kernels), `naitzat2020` (prepend the input layer + standardise each layer), `zhou2023` (one Mapper graph *per layer* — feeding the list at once would concatenate them) |

Rule of thumb: reach for `fit_model` / `fit_each` first; only extract by hand for the last category, and leave a one-line note explaining the transform.

## Toolkit functions these notebooks rely on

Some notebooks need functions beyond the core pipeline; each lives in the toolkit
and is exercised by the notebook listed here:

| Notebook | New function | Where it lives |
|---|---|---|
| `gebhart2019` | `plot_pathways_on_network` | `tanc.visualisation` |
| `lacombe2021` | `plot_tu_roc` | `tanc.visualisation` |
| `naitzat2020` | `plot_betti_layer_bars` | `tanc.visualisation` |
| `rathore2021`, `gabrielsson2019` | `node_exemplars` | `tanc.topo_tools` |
| `zhou2023` | `mapper_gw_distance` (needs `pip install pot`) | `tanc.topo_tools` |

## How the notebooks are generated

[`_generate.py`](_generate.py) is the source of truth — edit cell specs there and re-run:

```bash
py paper_reproduce/_generate.py
```

This emits all 18 `.ipynb` files in seconds.

## What the pipeline cannot reproduce (transparency)

* **Exact paper numbers** still require the original datasets and trained models. The notebooks now point you at the right repos (when one exists) and name the datasets, so swapping in real data is a matter of filling the synthetic-data cell.
* **Papers without a public reference repo** (`watanabe2021`, `gebhart2019`, `lacombe2021`, `naitzat2020`, `karuppiah2025`, `ramamurthy2019`, `gabrielsson2019`, `ong2026`) — the dataset and headline result are still described, but you'll need to source models / data from the descriptions in the paper.
* **Multi-run generalization-correlation experiments** (`birdal2021`, `dupuis2023`, `andreeva2024`) need many models trained at varying scales; each notebook reproduces the per-model path and points to the reference repo for the multi-model sweep. `ballester2024` goes further: it trains a small VGG-like *population* on the laptop GPU and runs the full ASDSQ → generalization-gap regression end-to-end (scale it up via the notebook's tunables for paper-level numbers).
* **Image / kernel exemplars** are returned as data-indices by `node_exemplars`; rendering the actual images / kernels is left to the user since the toolkit isn't image-aware.

Run any notebook with `jupyter notebook paper_reproduce/<name>.ipynb` or open in VS Code. They expect the same optional deps as the rest of the toolkit (`ripser` for PH, `networkx` for Mapper, `persim` for Wasserstein/bottleneck, `pot` for `mapper_gw_distance`).