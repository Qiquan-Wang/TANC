# Datasets needed to reproduce the 18 notebooks

This file aggregates every dataset and pre-trained checkpoint referenced by the `paper_reproduce/` notebooks, classifies it by download size, and tells you exactly which notebook needs which file. Use it to decide what to bring into `data/` and what to leave out.

Sizes are uncompressed unless noted. Suggested locations assume a `data/` folder at the repo root — but `_torch_setup.py` resolves it robustly: it uses `$TANC_DATA` if set, otherwise the first `data/` folder (containing a known dataset) found at the repo root **or any ancestor directory**, so `data/` sitting *next to* the repo also works.

## Decision matrix — at a glance

| # | Dataset / model zoo | Approx size | Notebooks that need it | Recommendation |
|---|---|---|---|---|
| 1 | **Synthetic generators** (toy 2D/3D manifolds, modular arithmetic) | 0 (code only) | naitzat2020, ramamurthy2019, liu2023, ong2026, ruppik2025-grokking | No download — generated in-notebook |
| 2 | **MNIST** ✅ present in `data/MNIST/` | 12 MB | rieck2019, gebhart2019, gabella2021, gabrielsson2019, lacombe2021 | **Full download** |
| 3 | **Fashion-MNIST** ✅ present in `data/FashionMNIST/` | 30 MB | rieck2019 | **Full download** |
| 4 | **CIFAR-10** ✅ present in `data/cifar-10-batches-py/` | 170 MB | rieck2019, watanabe2021, gabrielsson2019, rathore2021, zhou2023, birdal2021, dupuis2023, andreeva2024, ballester2024, lacombe2021 | **Full download** (most-shared dataset) |
| 5 | **CIFAR-100** ✅ present in `data/cifar-100-python/` | 170 MB | dupuis2023 (supplement), andreeva2024 | **Full download** |
| 6 | **SVHN** | 2.5 GB | gabrielsson2019 | **Partial** — Format-2 cropped digits (~1.4 GB) suffices |
| 7 | **GoEmotions** | 100 MB | ruppik2025 | **Full download** |
| 8 | **MultiWOZ 2.2** | 400 MB | ruppik2025 | **Full download** |
| 9 | **UCI tabular sample** (~5 small datasets) | < 50 MB total | ramamurthy2019 | **Full download** |
| 10 | **PGDL competition models** | 1 - 8 GB depending on track | ballester2024 | **Partial** — Task 1 (CIFAR-10) public models only (~1 GB) |
| 11 | **ImageNet 2012** | ~150 GB (train) | gabrielsson2019, rathore2021 | **Skip raw images** — use pre-trained InceptionV1 + activation cache only |
| 12 | **Pre-trained model checkpoints** (InceptionV1, ResNet-18, AlexNet, VGG, BERT-base, RoBERTa-base) | ~6 GB combined | rathore2021, birdal2021, dupuis2023, andreeva2024, ruppik2025 | **Lazy download** — torchvision / huggingface fetch on demand |

Total if you take every "**Full download**" plus the partial recommendations: **~7 GB**. Without ImageNet + PGDL full set: well under 5 GB.

---

## 1. Synthetic generators (no download)

Generated inside the notebooks. Listed here for completeness.

| Where | What | How it's generated |
|---|---|---|
| `naitzat2020` | Entangled tori, linked rings, annuli with known Betti numbers | Custom samplers — see Naitzat et al. (2020) Sec. 4 |
| `ramamurthy2019` | XOR, swiss-roll, concentric rings | `sklearn.datasets.make_moons`, `make_swiss_roll`, custom |
| `liu2023` | Annulus, two-moons | `sklearn.datasets.make_circles`, `make_moons` |
| `ong2026` | Swiss roll, sphere, torus | Manifold samplers (paper-supplied scripts) |
| `ruppik2025` (grokking task) | Modular arithmetic pairs | One-line numpy generator |

---

## 2. Small fully-downloadable datasets (< 2 GB)

### MNIST  — 12 MB
- **Used by:** `rieck2019`, `gebhart2019`, `gabella2021`, `gabrielsson2019`, `lacombe2021`
- **Source:** [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) or `torchvision.datasets.MNIST(root=..., download=True)`
- **Notes:** Standard 60k/10k train/test split. Used as the smallest sanity check by almost every PH paper.

### Fashion-MNIST — 30 MB
- **Used by:** `rieck2019`
- **Source:** [github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) or `torchvision.datasets.FashionMNIST`
- **Notes:** Drop-in replacement for MNIST; same shapes, harder.

### CIFAR-10 — 170 MB ✅ already present (`data/cifar-10-batches-py/`)
- **Used by:** `rieck2019`, `watanabe2021`, `gabrielsson2019`, `rathore2021`, `zhou2023`, `birdal2021`, `dupuis2023`, `andreeva2024`, `ballester2024`, `lacombe2021`
- **Source:** [www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html) or `torchvision.datasets.CIFAR10(root=..., download=True)`
- **Re-download:** `py data/_download_cifar.py` (idempotent)
- **Notes:** The single most-shared dataset across the 18 notebooks.

### CIFAR-100 — 170 MB ✅ already present (`data/cifar-100-python/`)
- **Used by:** `dupuis2023` (supplement), `andreeva2024`
- **Source:** Same as CIFAR-10 page; `torchvision.datasets.CIFAR100`
- **Re-download:** `py data/_download_cifar.py` (idempotent)

### GoEmotions — 100 MB
- **Used by:** `ruppik2025` (emotion-recognition LID experiment)
- **Source:** [github.com/google-research/google-research/tree/master/goemotions](https://github.com/google-research/google-research/tree/master/goemotions) or HuggingFace `go_emotions`
- **Notes:** 58k Reddit comments × 27 emotion labels. Plain TSV / Parquet.

### MultiWOZ 2.2 — 400 MB
- **Used by:** `ruppik2025` (dialogue-state-tracking LID experiment)
- **Source:** [github.com/budzianowski/multiwoz](https://github.com/budzianowski/multiwoz)
- **Notes:** v2.2 release recommended over 2.1; processed JSON dialogues.

### UCI tabular sample — < 50 MB total
- **Used by:** `ramamurthy2019`
- **Source:** [archive.ics.uci.edu/datasets](https://archive.ics.uci.edu/datasets) — pick a handful (e.g. Iris, Wine, Pima Diabetes, Breast Cancer Wisconsin). Or `sklearn.datasets.fetch_openml` for individual fetches.
- **Notes:** The paper's model-selection use case works on any of them; no specific list required.

---

## 3. Datasets > 2 GB — partial recommended

### SVHN — 2.5 GB (Format-2 cropped) / 13 GB (Format-1 full street images)
- **Used by:** `gabrielsson2019`
- **Source:** [ufldl.stanford.edu/housenumbers](http://ufldl.stanford.edu/housenumbers/) or `torchvision.datasets.SVHN`
- **Partial recommendation:** Format-2 only (the `train_32x32.mat` + `test_32x32.mat`) — ~1.4 GB. Plenty for the conv-kernel analysis Gabrielsson does.

### PGDL competition models — 1 - 8 GB
- **Used by:** `ballester2024`
- **Source:** [competitions.codalab.org/competitions/25301](https://competitions.codalab.org/competitions/25301) (NeurIPS 2020 PGDL) or the reference repo [rballeba/PredictingGeneralizationGapUsingPersistentHomology](https://github.com/rballeba/PredictingGeneralizationGapUsingPersistentHomology)
- **Partial recommendation:** Task 1 public models (CIFAR-10, ~1 GB). The full benchmark has 8 tracks; one is enough for a methodology reproduction.

### ImageNet 2012 — ~150 GB train / 6.3 GB val
- **Used by:** `gabrielsson2019`, `rathore2021`
- **Source:** [image-net.org/challenges/LSVRC/2012](https://image-net.org/challenges/LSVRC/2012) (registration required)
- **Partial recommendation: skip raw images entirely.** Both papers care about activations / kernels of pretrained networks. Use the pre-trained model directly (`torchvision.models.googlenet(pretrained=True)`, etc.) and a couple hundred validation images (val set is 6.3 GB but a 1k random subset is < 200 MB).

---

## 4. Pre-trained model checkpoints

These are usually fetched lazily by the model library; they don't live in `data/`. Listed here so you know what you'll end up downloading.

| Model | Approx size | Notebooks | How |
|---|---|---|---|
| InceptionV1 / GoogLeNet (ImageNet) | 50 MB | `rathore2021` | `torchvision.models.googlenet(weights="DEFAULT")` |
| ResNet-18 (ImageNet) | 45 MB | `birdal2021`, `dupuis2023`, `andreeva2024` | `torchvision.models.resnet18(weights="DEFAULT")` |
| AlexNet (ImageNet) | 230 MB | `birdal2021`, `dupuis2023` | `torchvision.models.alexnet(weights="DEFAULT")` |
| VGG-16 (ImageNet) | 530 MB | `birdal2021`, `dupuis2023`, `andreeva2024` | `torchvision.models.vgg16(weights="DEFAULT")` |
| BERT-base-uncased | 440 MB | `ruppik2025` | `transformers.AutoModel.from_pretrained("bert-base-uncased")` |
| RoBERTa-base | 500 MB | `ruppik2025` | `transformers.AutoModel.from_pretrained("roberta-base")` |
| PGDL CNN zoo (small CNNs, Task 1) | ~1 GB | `ballester2024` | Bundled with the PGDL competition data |
| Custom small MLPs / CNNs | < 5 MB each | watanabe2021, rieck2019, gebhart2019, lacombe2021, naitzat2020, karuppiah2025, ramamurthy2019, liu2023, gabrielsson2019, gabella2021 | Train from scratch in the notebook (`TrainingExtractor`) — no checkpoint needed |

---

## Suggested `data/` layout

```
data/
├── README.md                # links back to this file
├── mnist/                   # torchvision-style cache; first ~12 MB
├── fashion_mnist/           # 30 MB
├── cifar10/                 # 170 MB
├── cifar100/                # 170 MB
├── svhn_format2/            # 1.4 GB partial
├── goemotions/              # 100 MB
├── multiwoz_2_2/            # 400 MB
├── uci/                     # < 50 MB
├── pgdl_task1/              # ~1 GB partial
├── imagenet_val_subset/     # ~200 MB if you keep 1k val images
└── pretrained/              # lazy cache; torchvision/transformers populate this
```

Total disk if you take everything in this layout: **~3.5 GB raw data + ~2 GB pretrained model cache = ~5.5 GB**.

If you skip the partials (SVHN, PGDL, ImageNet subset): **under 1 GB**.

---

## Per-notebook cross-reference

| Notebook | Needs |
|---|---|
| `watanabe2021` | CIFAR-10 + custom MLP (train in-notebook) |
| `rieck2019` | MNIST, Fashion-MNIST, CIFAR-10 + custom MLPs / CNNs |
| `gebhart2019` | MNIST + custom MLP |
| `lacombe2021` | MNIST or CIFAR-10 + a trained classifier |
| `naitzat2020` | Synthetic only |
| `karuppiah2025` | Any small classifier (review paper, illustrative use) |
| `ballester2024` | PGDL Task 1 models + CIFAR-10 (partial recommended) |
| `ramamurthy2019` | Synthetic + a few UCI sets |
| `liu2023` | Synthetic only |
| `rathore2021` | InceptionV1 (pre-trained) + ImageNet val subset |
| `zhou2023` | CIFAR-10 + a small CNN |
| `gabrielsson2019` | MNIST, CIFAR-10, SVHN-Format2, ImageNet val subset + many CNNs |
| `gabella2021` | MNIST + small MLP weight trajectories |
| `ruppik2025` | MultiWOZ 2.2, GoEmotions, modular arithmetic + BERT / RoBERTa |
| `ong2026` | Synthetic only |
| `birdal2021` | CIFAR-10 + ResNet-18 / AlexNet / VGG-16 |
| `dupuis2023` | CIFAR-10 (+ CIFAR-100 for supplement) + ResNet-18 / AlexNet / VGG-16 |
| `andreeva2024` | CIFAR-10, CIFAR-100 + ResNet-18 / VGG-16 |

---

## What to download first (my recommendation)

1. **CIFAR-10** (170 MB) — covers 10 of the 18 notebooks.
2. **MNIST** (12 MB) — covers another 5.
3. **Pre-trained ResNet-18 + VGG-16** (~600 MB, lazy via torchvision) — covers the three trajectory-dim papers.
4. Everything else on demand once the first three are in place.

If you do those three, you've unlocked ~14 / 18 notebooks. The remaining four (`ruppik2025`, `rathore2021`, `gabrielsson2019`, `ballester2024`) want substantially more data and you might prefer to do them last.