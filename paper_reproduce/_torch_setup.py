"""_torch_setup.py — shared dataset loaders + tiny models for paper notebooks.

All loaders read from a local ``data/`` cache (set up by
``data/_download_cifar.py`` and the existing MNIST / FashionMNIST folders).
Its location is resolved automatically: ``$TANC_DATA`` if set, else the
first ``data/`` folder (containing a known dataset) found at the repo root or
any ancestor directory — so it works whether ``data/`` sits inside the repo or
alongside it.  See ``_resolve_data_dir``.

Models are intentionally small so each notebook trains in a couple of
minutes on CPU.  The point is to exercise the TDA pipeline on *real*
extracted weights / activations — not to match paper headline numbers,
which require GPU + full training.

Usage in a notebook:

    from paper_reproduce._torch_setup import (
        load_mnist, load_cifar10, SmallMLP, SmallCNN, train_briefly,
    )

    X_tr, y_tr = load_mnist("train", n_samples=2000)
    X_te, y_te = load_mnist("test",  n_samples=400)
    model = SmallMLP(input_dim=784, hidden_dims=[64, 32], output_dim=10)
    view  = train_briefly(model, X_tr, y_tr, X_te, y_te, epochs=2)
    snap  = view.final_snapshot
"""

from __future__ import annotations

import gzip
import os
import pathlib
import pickle
import sys
from typing import Any

import numpy as np


# Repo root = parent of this file's directory.
_HERE = pathlib.Path(__file__).resolve().parent

# Subfolders that mark a directory as a real dataset root (any one is enough).
_DATA_MARKERS = ("MNIST", "FashionMNIST", "cifar-10-batches-py", "cifar-100-python")


def _looks_like_data_dir(path: pathlib.Path) -> bool:
    """True if *path* exists and holds at least one known dataset subfolder."""
    return path.is_dir() and any((path / m).exists() for m in _DATA_MARKERS)


def _resolve_data_dir() -> pathlib.Path:
    """Locate the ``data/`` folder holding the datasets, robustly.

    Search order (first hit wins):
      1. ``$TANC_DATA`` — explicit override.
      2. ``<repo>/data`` — the documented default (parent of this file's dir).
      3. ``<ancestor>/data`` for each ancestor above the repo — covers checkouts
         where ``data/`` sits next to, rather than inside, the repo.
      4. ``<cwd>/data`` — running from somewhere with its own data folder.

    Candidates 2–4 must actually contain a known dataset subfolder to be picked;
    if nothing qualifies we fall back to the documented default so error
    messages still point at the expected ``<repo>/data`` location.
    """
    env = os.environ.get("TANC_DATA")
    if env:
        return pathlib.Path(env).expanduser()

    default = _HERE.parent / "data"
    candidates = [default]
    candidates += [parent / "data" for parent in _HERE.parents]
    candidates.append(pathlib.Path.cwd() / "data")

    for cand in candidates:
        if _looks_like_data_dir(cand):
            return cand
    return default


_DATA = _resolve_data_dir()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders — all return (X, y) numpy arrays.
# ─────────────────────────────────────────────────────────────────────────────

def _read_idx(path: pathlib.Path) -> np.ndarray:
    """Read an IDX file (MNIST / Fashion-MNIST raw format).  Handles .gz."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        buf = f.read()
    # IDX header: magic (4 bytes), then ndim 32-bit ints for shape.
    magic = int.from_bytes(buf[:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(buf[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    arr = np.frombuffer(buf, dtype=np.uint8, offset=4 + 4 * ndim)
    return arr.reshape(dims)


def _load_idx_pair(folder: pathlib.Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (images, labels) from an IDX-style folder for the given split."""
    prefix = "train" if split == "train" else "t10k"
    candidates_images = [
        folder / f"{prefix}-images-idx3-ubyte",
        folder / f"{prefix}-images-idx3-ubyte.gz",
    ]
    candidates_labels = [
        folder / f"{prefix}-labels-idx1-ubyte",
        folder / f"{prefix}-labels-idx1-ubyte.gz",
    ]
    img_path = next(p for p in candidates_images if p.exists())
    lbl_path = next(p for p in candidates_labels if p.exists())
    X = _read_idx(img_path).astype(np.float32) / 255.0
    y = _read_idx(lbl_path).astype(np.int64)
    return X, y


def load_mnist(
    split: str = "train",
    n_samples: int | None = None,
    flatten: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST from ``data/MNIST/raw/``.

    Parameters
    ----------
    split : "train" | "test"
    n_samples : int or None
        Truncate to the first ``n_samples`` (after shuffling for reproducibility).
    flatten : bool
        ``True`` -> (N, 784); ``False`` -> (N, 1, 28, 28).
    """
    X, y = _load_idx_pair(_DATA / "MNIST" / "raw", split)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]
    if flatten:
        X = X.reshape(len(X), -1)
    else:
        X = X[:, None, :, :]   # (N, 1, 28, 28)
    return X, y


def load_fashion_mnist(
    split: str = "train",
    n_samples: int | None = None,
    flatten: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Fashion-MNIST from ``data/FashionMNIST/raw/`` (same format as MNIST)."""
    X, y = _load_idx_pair(_DATA / "FashionMNIST" / "raw", split)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]
    if flatten:
        X = X.reshape(len(X), -1)
    else:
        X = X[:, None, :, :]
    return X, y


def load_cifar10(
    split: str = "train",
    n_samples: int | None = None,
    flatten: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 from ``data/cifar-10-batches-py/``.

    Returns
    -------
    X : (N, 3, 32, 32) float32 in [0, 1] (or (N, 3072) if ``flatten=True``).
    y : (N,) int64.
    """
    folder = _DATA / "cifar-10-batches-py"
    if split == "train":
        files = [folder / f"data_batch_{i}" for i in range(1, 6)]
    else:
        files = [folder / "test_batch"]
    Xs, ys = [], []
    for f in files:
        with f.open("rb") as fh:
            batch = pickle.load(fh, encoding="bytes")
        Xs.append(batch[b"data"])
        ys.append(np.asarray(batch[b"labels"], dtype=np.int64))
    X = np.concatenate(Xs).astype(np.float32) / 255.0
    y = np.concatenate(ys)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]
    if not flatten:
        X = X.reshape(len(X), 3, 32, 32)
    return X, y


def load_cifar100(
    split: str = "train",
    n_samples: int | None = None,
    flatten: bool = False,
    coarse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-100 from ``data/cifar-100-python/``."""
    folder = _DATA / "cifar-100-python"
    f = folder / ("train" if split == "train" else "test")
    with f.open("rb") as fh:
        batch = pickle.load(fh, encoding="bytes")
    X = np.asarray(batch[b"data"], dtype=np.float32) / 255.0
    key = b"coarse_labels" if coarse else b"fine_labels"
    y = np.asarray(batch[key], dtype=np.int64)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]
    if not flatten:
        X = X.reshape(len(X), 3, 32, 32)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Small models — torch is required for everything below.
# ─────────────────────────────────────────────────────────────────────────────

def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for the model helpers.  "
            "Install with: pip install torch"
        ) from e


def SmallMLP(input_dim: int, hidden_dims: list[int], output_dim: int,
             dropout: float = 0.0, batchnorm: bool = False):
    """A tiny ReLU MLP suitable for MNIST-scale problems.

    Returns a ``torch.nn.Sequential`` with named ``fc{i}`` modules so the
    layer names show up in ``ModelExtractor.layer_names``.  Optional
    ``dropout`` / ``batchnorm`` on the hidden layers (Rieck et al. 2019 study
    how these regularisers change Neural Persistence).
    """
    _require_torch()
    import torch.nn as nn
    layers: list[tuple[str, nn.Module]] = []
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    for i in range(len(dims) - 1):
        layers.append((f"fc{i+1}", nn.Linear(dims[i], dims[i+1])))
        if i < len(dims) - 2:
            if batchnorm:
                layers.append((f"bn{i+1}", nn.BatchNorm1d(dims[i+1])))
            layers.append((f"relu{i+1}", nn.ReLU()))
            if dropout > 0:
                layers.append((f"drop{i+1}", nn.Dropout(dropout)))
    from collections import OrderedDict
    return nn.Sequential(OrderedDict(layers))


def SmallCNN(in_channels: int = 3, num_classes: int = 10, width: int = 16):
    """A tiny ConvNet for CIFAR-scale (3x32x32) inputs.

    Two conv stages + one FC layer — small enough to train in ~1 minute
    on CPU per epoch with ``n_samples=2000``.
    """
    _require_torch()
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(width * 2 * 8 * 8, num_classes),
    )


def VGGLike(
    in_channels: int = 3,
    num_classes: int = 10,
    n_blocks: int = 2,
    convs_per_block: int = 1,
    base_width: int = 16,
    hidden: int = 128,
    n_dense: int = 1,
    dropout: float = 0.0,
    batchnorm: bool = True,
):
    """A small VGG-style ConvNet matching the architecture family of
    Ballester et al. (2024), Task 1 (PGDL Task 1 = VGG-like on CIFAR-10).

    The paper varies, per network: number of convolutional layers (2 or 6),
    convolutional blocks per layer (1 or 3), filters in the last conv layer
    (256 or 512), dense layers (1 or 2), and dropout (0 or 0.5).  This builder
    exposes the same knobs at a reduced scale so a *population* of nets with a
    spread of generalization gaps trains quickly on a laptop GPU (MPS/CUDA).

    Each conv stage is ``convs_per_block`` × (Conv3x3 → ReLU) followed by a
    2×2 MaxPool, with the channel width doubling per stage.  A global average
    pool feeds a dense head of ``n_dense`` hidden layers of size ``hidden``
    (ReLU, optional Dropout) and a linear classifier.

    Named modules (``conv{stage}_{j}``, ``fc{i}``) surface as layer names in
    the extractor so activations can be pulled per layer.
    """
    _require_torch()
    import torch.nn as nn
    from collections import OrderedDict

    layers: list[tuple[str, nn.Module]] = []
    c_in = in_channels
    w = base_width
    for stage in range(1, n_blocks + 1):
        for j in range(1, convs_per_block + 1):
            layers.append((f"conv{stage}_{j}", nn.Conv2d(c_in, w, 3, padding=1)))
            if batchnorm:
                layers.append((f"bn{stage}_{j}", nn.BatchNorm2d(w)))
            layers.append((f"relu{stage}_{j}", nn.ReLU()))
            c_in = w
        layers.append((f"pool{stage}", nn.MaxPool2d(2)))
        w *= 2
    layers.append(("gap", nn.AdaptiveAvgPool2d(1)))
    layers.append(("flatten", nn.Flatten()))

    d_in = c_in
    for i in range(1, n_dense + 1):
        layers.append((f"fc{i}", nn.Linear(d_in, hidden)))
        layers.append((f"fc{i}_relu", nn.ReLU()))
        if dropout > 0:
            layers.append((f"fc{i}_drop", nn.Dropout(dropout)))
        d_in = hidden
    layers.append(("classifier", nn.Linear(d_in, num_classes)))
    return nn.Sequential(OrderedDict(layers))


def CNN_FCN(
    in_channels: int = 3,
    num_classes: int = 10,
    width: int = 16,
    feat: int = 128,
    hidden: int = 64,
):
    """A small CNN feature extractor + a multi-layer FCN head.

    Matches the architecture family of Watanabe & Yamana (2020): a conv stack
    produces a ``feat``-dim feature vector (global average pool), then a fully
    connected head ``feat → hidden → num_classes`` classifies it.  The paper
    applies persistent-homology pruning to exactly this FCN head, so the head
    has **two consecutive Dense layers** — enough for the directed clique
    complex to form triangles (unlike a single-Dense ``SmallCNN``).

    The extractor's default weight selection (linear layers) yields the two
    head weight matrices ``[(feat, hidden), (hidden, num_classes)]`` — the FCN.
    """
    _require_torch()
    import torch.nn as nn
    from collections import OrderedDict
    return nn.Sequential(OrderedDict([
        ("conv1", nn.Conv2d(in_channels, width, 3, padding=1)),
        ("relu1", nn.ReLU()),
        ("pool1", nn.MaxPool2d(2)),
        ("conv2", nn.Conv2d(width, feat, 3, padding=1)),
        ("relu2", nn.ReLU()),
        ("gap", nn.AdaptiveAvgPool2d(1)),
        ("flatten", nn.Flatten()),
        ("fc1", nn.Linear(feat, hidden)),     # FCN head, layer 1
        ("fc1_relu", nn.ReLU()),
        ("fc2", nn.Linear(hidden, num_classes)),  # FCN head, layer 2
    ]))


def CNN_FCN2(
    in_channels: int = 3,
    num_classes: int = 10,
    conv_channels: tuple[int, ...] = (32, 64, 128),
    hidden: int = 64,
    head_pool: int = 1,
    batchnorm: bool = True,
):
    """Deeper CNN feature extractor + a 2-layer FCN head (CNN_FCN, scaled up).

    The conv stack is one ``Conv3x3 → (BatchNorm) → ReLU → MaxPool2d(2)`` block
    per entry in ``conv_channels``; each block halves the spatial size
    (``32 → 16 → 8 → 4 → …``), so at most 5 blocks before a 32×32 input is
    pooled to 1×1.  The last conv map is average-pooled to ``head_pool ×
    head_pool`` and flattened into the FCN head ``Linear → ReLU → Linear``.

    The FCN head is the two consecutive Dense layers the Watanabe directed
    clique complex is built on; ``fc1``'s input is the last conv channel count
    times ``head_pool²`` (so ``head_pool=1`` keeps the head small / PH cheap;
    ``head_pool=4`` keeps spatial detail for accuracy at a larger PH cost).

    Examples
    --------
    >>> model = CNN_FCN2(conv_channels=(32, 64, 128), hidden=64)   # 3 conv stages
    >>> model = CNN_FCN2(conv_channels=(64, 128, 256, 256), hidden=128)  # deeper/wider
    """
    _require_torch()
    import torch.nn as nn
    from collections import OrderedDict

    layers: list[tuple[str, nn.Module]] = []
    c_in = in_channels
    for s, c_out in enumerate(conv_channels, start=1):
        layers.append((f"conv{s}", nn.Conv2d(c_in, c_out, 3, padding=1)))
        if batchnorm:
            layers.append((f"bn{s}", nn.BatchNorm2d(c_out)))
        layers.append((f"relu{s}", nn.ReLU()))
        layers.append((f"pool{s}", nn.MaxPool2d(2)))   # halves H×W each stage
        c_in = c_out

    feat_dim = c_in * head_pool * head_pool
    layers += [
        ("gap", nn.AdaptiveAvgPool2d(head_pool)),
        ("flatten", nn.Flatten()),
        ("fc1", nn.Linear(feat_dim, hidden)),          # FCN head, layer 1
        ("fc1_relu", nn.ReLU()),
        ("fc2", nn.Linear(hidden, num_classes)),       # FCN head, layer 2
    ]
    return nn.Sequential(OrderedDict(layers))


# ─────────────────────────────────────────────────────────────────────────────
# Device + evaluation helpers.
# ─────────────────────────────────────────────────────────────────────────────

def pick_device(prefer: str | None = None) -> str:
    """Return the best available torch device string.

    Order of preference: explicit ``prefer`` → CUDA → Apple MPS (Metal) → CPU.
    """
    _require_torch()
    import torch
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def accuracy(model: Any, X: np.ndarray, y: np.ndarray,
             device: str | None = None, batch_size: int = 256) -> float:
    """Top-1 accuracy of ``model`` on ``(X, y)`` evaluated on ``device``."""
    _require_torch()
    import torch
    device = device or pick_device()
    model = model.to(device).eval()
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size].to(device)
            logits = model(xb)
            pred = logits.argmax(1).cpu()
            correct += (pred == y_t[i:i + batch_size]).sum().item()
    return correct / len(X_t)


# ─────────────────────────────────────────────────────────────────────────────
# Training — uses our own TrainingExtractor to get a TrainingView back.
# ─────────────────────────────────────────────────────────────────────────────

def _to_torch_dataset(X: np.ndarray, y: np.ndarray):
    """Wrap (X, y) numpy arrays in a torch TensorDataset."""
    import torch
    from torch.utils.data import TensorDataset
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    return TensorDataset(X_t, y_t)


def train_briefly(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 1e-3,
    aspects: list[str] | None = None,
    snapshot_every: int = 1,
    layer_selection: str | list[str] | None = None,
    target_accuracy: float | None = None,
    weight_decay: float = 0.0,
    device: str | None = None,
    verbose: bool = True,
):
    """Train ``model`` briefly and return a :class:`TrainingView`.

    Routes through the toolkit's own :class:`TrainingExtractor` so the
    returned view feeds directly into ``TDAPipeline.fit(view)``.

    ``device`` is a torch device string (``"cuda"``/``"mps"``/``"cpu"``);
    ``None`` auto-detects via :func:`pick_device`.
    """
    _require_torch()
    import torch
    from torch.utils.data import DataLoader

    from tanc.model_extractor import TrainingExtractor

    device = device or pick_device()

    train_loader = DataLoader(
        _to_torch_dataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(
            _to_torch_dataset(X_val, y_val),
            batch_size=batch_size, shuffle=False,
        )

    extractor = TrainingExtractor(
        model=model,
        train_loader=train_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay),
        val_loader=val_loader,
        extract_data=X_val if X_val is not None else X_train[:128],
        aspects=aspects or ["weights", "activations"],
        layer_selection=layer_selection,
        snapshot_every=snapshot_every,
        device=device,
        clarify=False,
    )
    if verbose:
        extractor.explain()
    return extractor.run(
        epochs=epochs, target_accuracy=target_accuracy, verbose=verbose,
    )


def extract_snapshot(
    model: Any,
    X: np.ndarray,
    aspects: list[str] | None = None,
    layer_selection: str | list[str] | None = None,
):
    """Run one forward pass and return a :class:`ModelSnapshot`.

    Thin shortcut around :class:`ModelExtractor.extract`.
    """
    _require_torch()
    from tanc.model_extractor import ModelExtractor

    return ModelExtractor(
        model=model,
        aspects=aspects or ["weights", "activations"],
        layer_selection=layer_selection,
        clarify=False,
    ).extract(X)