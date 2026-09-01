"""_generate.py — emit one .ipynb per paper preset.

Each notebook is a small, self-contained reproduction that exercises the
``TDAPipeline.from_paper(...).reproduce(...)`` path the paper uses.  Toy
data so notebooks run end-to-end in seconds; swap in real data by
editing the synthetic-data cell.

Run with:
    py paper_reproduce/_generate.py
"""

from __future__ import annotations

import json
import pathlib
import sys


HERE = pathlib.Path(__file__).resolve().parent

# Optional CLI filter: `python _generate.py gabella2021 watanabe2021` regenerates
# only those notebooks (leaves the rest — and any hand-edits — untouched).
ONLY = set(sys.argv[1:])

IMPORTS = """\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # make `tanc` importable
import numpy as np
import matplotlib.pyplot as plt
from tanc import TDAPipeline
np.random.seed(0)
"""

# Real-data notebooks (10) train a tiny model briefly on the dataset cached
# in ``data/``.  The helper module lives next to this file.
REAL_IMPORTS = """\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # make `paper_reproduce` importable
import numpy as np
import matplotlib.pyplot as plt
import torch
from tanc import TDAPipeline
from paper_reproduce._torch_setup import (
    load_mnist, load_fashion_mnist, load_cifar10, load_cifar100,
    SmallMLP, SmallCNN, CNN_FCN, CNN_FCN2, VGGLike, train_briefly, extract_snapshot, accuracy, pick_device,
)
np.random.seed(0); torch.manual_seed(0)
DEVICE = pick_device()                      # 'cuda' → 'mps' (Apple GPU) → 'cpu'
"""


def nb(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


# These notebooks have bespoke, hand-authored content (full headline
# reproductions) that this skeleton generator would overwrite — skip them.
BESPOKE = {"ballester2024", "andreeva2024"}


def write(name: str, cells: list[tuple[str, str]]) -> None:
    if ONLY and name not in ONLY:
        return
    if name in BESPOKE:
        print(f"  skipped {name}.ipynb (bespoke — not regenerated)")
        return
    path = HERE / f"{name}.ipynb"
    with path.open("w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1)
    print(f"  wrote {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Paper-info block — appears at the top of every notebook.
# ─────────────────────────────────────────────────────────────────────────────

def paper_info(
    *,
    arxiv: str | None = None,
    venue: str | None = None,
    datasets: str = "",
    models: str = "",
    headline: str = "",
    github: str | None = None,
    notes_url: str | None = None,
) -> str:
    """Render the standard 'Dataset & model' markdown block."""
    lines = ["## Dataset & model in the paper\n\n"]
    if arxiv:
        lines.append(f"- **arXiv:** [{arxiv}]({arxiv})\n")
    if venue:
        lines.append(f"- **Venue:** {venue}\n")
    if datasets:
        lines.append(f"- **Datasets used:** {datasets}\n")
    if models:
        lines.append(f"- **Models used:** {models}\n")
    if headline:
        lines.append(f"- **Headline result to aim for:** {headline}\n")
    if github:
        lines.append(f"- **Reference code:** [{github}]({github})\n")
    else:
        lines.append("- **Reference code:** no public repo found in web search "
                     "— check paper supplementary materials.\n")
    if notes_url:
        lines.append(f"- **Other notes:** {notes_url}\n")
    lines.append(
        "\nTo reproduce paper numbers, replace the synthetic-data cell below "
        "with the real model snapshot / activations / trajectory.  Everything "
        "else in the notebook stays the same.\n"
    )
    return "".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Generic skeleton
# ─────────────────────────────────────────────────────────────────────────────

def make_real(
    preset: str, title: str, summary: str, info_md: str,
    setup_code: str, data_code: str | None = None,
    plot_cell: str | None = None,
    extra_cells: list[tuple[str, str]] | None = None,
    notes: str = "",
    methodology: str = "",
    reproduce_input: str = "data",
) -> list[tuple[str, str]]:
    """Skeleton for notebooks that train a real (tiny) model on a real dataset.

    ``methodology`` is a rich 'What the paper does' markdown block inserted
    after the title.  ``reproduce_input`` is the variable handed to
    ``reproduce(...)``; pass ``"view"`` / ``"snap"`` for the model-first path
    (the pipeline's adapter extracts what the preset needs) and set
    ``data_code=None`` to drop the explicit-extraction cell entirely.
    """
    cells: list[tuple[str, str]] = []
    cells.append(("markdown", f"# {preset} — {title}\n\n{summary}\n"))
    if methodology:
        cells.append(("markdown", methodology))
    cells.append(("markdown", info_md))
    cells.append(("markdown",
                  "**This notebook now runs on real data.**  A tiny model is "
                  "trained briefly on the local dataset cache under `data/`.  "
                  "Headline numbers won't match the paper (that needs full "
                  "training on GPU); the *qualitative* behaviour the paper "
                  "reports should still appear.\n"))
    cells.append(("code", REAL_IMPORTS))
    cells.append(("markdown", "## 1. Load dataset + train a small model briefly\n"))
    cells.append(("code", setup_code))
    n = 2
    if data_code is not None:
        cells.append(("markdown",
                      f"## {n}. Pull the right input out of the extracted snapshot\n"))
        cells.append(("code", data_code))
        n += 1
    if reproduce_input != "data":
        cells.append(("markdown",
                      f"## {n}. Reproduce via `TDAPipeline.from_paper` — model-first\n\n"
                      "We hand the trained model object straight to the pipeline: "
                      "`from_paper` already knows what this paper needs, and the "
                      "pipeline's model-adapter extracts it for us — no manual "
                      "feature pulling required.\n"))
    else:
        cells.append(("markdown",
                      f"## {n}. Reproduce via `TDAPipeline.from_paper`\n"))
    n += 1
    cells.append(("code",
                  f'pipe = TDAPipeline.from_paper("{preset}")\n'
                  'pipe.explain()\n'))
    cells.append(("code",
                  f"result, fig = pipe.reproduce({reproduce_input})\nresult.describe()\n"))
    if plot_cell:
        cells.append(("markdown", f"## {n}. Paper-specific follow-up plot\n"))
        cells.append(("code", plot_cell))
    if extra_cells:
        for c in extra_cells:
            cells.append(c)
    cells.append(("markdown", "## Notes & gaps\n\n" + notes))
    return cells


def make(preset: str, title: str, summary: str, info_md: str,
         data_cell: str, plot_cell: str | None = None,
         extra_cells: list[tuple[str, str]] | None = None,
         notes: str = "", methodology: str = "") -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    cells.append(("markdown", f"# {preset} — {title}\n\n{summary}\n"))
    if methodology:
        cells.append(("markdown", methodology))
    cells.append(("markdown", info_md))
    cells.append(("code", IMPORTS))
    cells.append(("markdown", "## 1. Synthetic data (swap in your real data here)\n"))
    cells.append(("code", data_cell))
    cells.append(("markdown",
                  "## 2. Reproduce in two lines via `TDAPipeline.from_paper`\n"))
    cells.append(("code",
                  f'pipe = TDAPipeline.from_paper("{preset}")\n'
                  'pipe.explain()\n'))
    cells.append(("code", "result, fig = pipe.reproduce(data)\nresult.describe()\n"))
    if plot_cell:
        cells.append(("markdown", "## 3. Paper-specific follow-up plot\n"))
        cells.append(("code", plot_cell))
    if extra_cells:
        for c in extra_cells:
            cells.append(c)
    cells.append(("markdown", "## Notes & gaps\n\n" + notes))
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# 1. watanabe2021
# ─────────────────────────────────────────────────────────────────────────────

write("watanabe2021", make_real(
    preset="watanabe2021",
    title="Watanabe & Yamana (2020) — Topological Measurement of DNNs Using PH",
    summary=(
        "Builds the FCN's **directed clique complex** with path-product "
        "relevance and integer-index filtration (GUDHI), then runs H1 PH.  "
        "Signature result: a wide 'belt' of H1 generators above the diagonal."
    ),
    info_md=paper_info(
        arxiv="https://arxiv.org/abs/2106.03016",
        venue="ISAIM 2020 / AIKE 2020 (Pruning)",
        datasets="CIFAR-10.",
        models="CNN feature extractor + a fully-connected head (the paper uses "
               "FCN 300→100→10).  Here: a small CNN_FCN whose linear head is "
               "the FCN the complex is built on.",
        headline="The H1 persistence diagram of the FCN's directed clique "
                 "complex shows a characteristic belt of loop generators well "
                 "above the diagonal (paper Fig. 2); the same complex's "
                 "representative cycles drive PHPM pruning.",
        github="https://github.com/satoru-watanabe-aw/DNNtopology",
    ),
    setup_code=(
        "# CNN with a 2-layer FCN head on CIFAR-10.  The clique complex needs\n"
        "# >=2 consecutive Dense layers, so we use CNN_FCN (not SmallCNN).\n"
        "X_tr, y_tr = load_cifar10('train', n_samples=4000)\n"
        "X_te, y_te = load_cifar10('test',  n_samples=1000)\n"
        "\n"
        "model = CNN_FCN(in_channels=3, num_classes=10, width=32, feat=128, hidden=64)\n"
        "# aspects=['weights'] -> the extractor captures the linear (FCN) weights.\n"
        "view  = train_briefly(model, X_tr, y_tr, X_te, y_te, epochs=12,\n"
        "                      aspects=['weights'], device=DEVICE, verbose=False)\n"
        "print('final test acc:', view.final_snapshot.accuracy)\n"
        "print('FCN head weight matrices:',\n"
        "      [W.shape for W in view.final_snapshot.weight_matrices()])\n"
    ),
    methodology=(
        "## What the paper does\n\n"
        "Reads the trained **fully-connected head** as a weighted directed graph "
        "and studies its loops with persistent homology.  Each weight is turned "
        "into a *relevance* `R_ij = w_ij / Σ_k w_kj` (column-normalised per "
        "target neuron), and the network is turned into a **directed clique "
        "complex**:\n\n"
        "* a **direct edge** (consecutive layers) is born at its relevance,\n"
        "* an **induced edge** between layers two apart is born at the strongest "
        "two-hop **path product** `maxⱼ R[i,j]·R[j,k]`,\n"
        "* a **triangle** `{i,j,k}` is born at the path product `R[i,j]·R[j,k]`,\n\n"
        "and every simplex's filtration *value* is the **integer threshold "
        "index** (1–64) of the paper's descending relevance schedule.\n\n"
        "**Methodology:** FCN weights → relevance → directed clique complex "
        "(path-product, integer filtration) → `H1` PH.\n\n"
        "**Expected result:** because a path product of two relevances (each "
        "≤ 1) is far smaller than either edge, triangles enter the filtration "
        "much later than the loops they fill — so `H1` generators form a wide "
        "**belt above the diagonal** (paper Fig. 2), *not* points hugging it.\n\n"
        "> This is a genuine **fix**: the old preset ran a *flag*-complex PH on "
        "a similarity matrix (triangle = max of its edges), which squashed the "
        "diagram onto the diagonal.  The toolkit now builds the directed clique "
        "complex (`run_ph(..., input_complex='directed_clique')`), matching the "
        "reference implementation bit-for-bit.\n"
    ),
    reproduce_input="view",
    plot_cell=(
        "# H1 diagram: the belt of loop generators above the diagonal.\n"
        'fig = result.plot("diagram", dims=[1])\n'
        "d1 = result.diagram(1)\n"
        "print(f'H1 generators: {len(d1)}  |  mean lifetime (belt width): '\n"
        "      f'{(d1[:,1]-d1[:,0]).mean():.1f}')\n"
    ),
    extra_cells=[
        ("markdown",
         "## 5. PHPM pruning — prune the FCN by its strongest H1 loops\n\n"
         "Watanabe & Yamana's **PHPM** pruning keeps the weight-edges carrying "
         "the strongest H1 loops (smallest `birth + death`) and zeroes the rest, "
         "comparing against the **global magnitude pruning (GMP)** baseline.  "
         "PHPM uses the *absolute*-relevance complex, so we rebuild the PH with "
         "`relevance_mode='absolute'` and read its representative cycles from "
         "`tanc.applications.pruning`.\n"),
        ("code",
         "import copy, torch\n"
         "from tanc.applications import (\n"
         "    phpm_pruning_mask, gmp_pruning_mask, prune_diagnostics)\n"
         "\n"
         "# PHPM uses |w| relevance; rebuild the directed-clique PH in that mode.\n"
         "ph = TDAPipeline.from_paper('watanabe2021',\n"
         "                            tool_kwargs__relevance_mode='absolute').fit(view)\n"
         "W = view.final_snapshot.weight_matrices()      # [(feat,hidden),(hidden,10)]\n"
         "\n"
         "def eval_with_masks(masks):\n"
         "    '''Zero the FCN Dense weights per mask (N_in,N_out -> torch (out,in)) and score.'''\n"
         "    m = copy.deepcopy(model).to('cpu').eval()\n"
         "    lins = [l for l in m.modules() if isinstance(l, torch.nn.Linear)]\n"
         "    with torch.no_grad():\n"
         "        for layer, mask in zip(lins, masks):\n"
         "            layer.weight.mul_(torch.as_tensor(mask.T, dtype=layer.weight.dtype))\n"
         "        pred = m(torch.as_tensor(X_te, dtype=torch.float32)).argmax(1).numpy()\n"
         "    return float((pred == y_te).mean())\n"
         "\n"
         "ratios = [0.0, 0.6, 0.7, 0.8, 0.9]\n"
         "phpm_acc, gmp_acc = [], []\n"
         "for r in ratios:\n"
         "    if r == 0.0:\n"
         "        a = eval_with_masks([np.ones_like(w, bool) for w in W])\n"
         "        phpm_acc.append(a); gmp_acc.append(a); continue\n"
         "    phpm_acc.append(eval_with_masks(phpm_pruning_mask(W, ph, r)))\n"
         "    gmp_acc.append(eval_with_masks(gmp_pruning_mask(W, r, per_layer=True)))\n"
         "    print(f'ratio {r}:  PHPM={phpm_acc[-1]:.3f}  GMP={gmp_acc[-1]:.3f}')\n"),
        ("code",
         "plt.figure(figsize=(6, 4))\n"
         "plt.plot(ratios, phpm_acc, 'o-', label='PHPM (topological)')\n"
         "plt.plot(ratios, gmp_acc, 's--', label='GMP (magnitude)')\n"
         "plt.xlabel('pruning ratio'); plt.ylabel('test accuracy')\n"
         "plt.title('PHPM vs GMP pruning of the FCN head (cf. Watanabe Table I)')\n"
         "plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()\n"),
    ],
    notes=(
        "* **Reproduced faithfully:** the directed clique complex with "
        "path-product integer filtration — identical H1 to the reference "
        "`DNNtopology` / `phpm.py` implementation.\n"
        "* `result.ph_result.metadata['watanabe']` carries the H1 representative "
        "simplices; the companion **PHPM pruning** notebook uses them to prune "
        "the FCN by its strongest loops.\n"
        "* The paper's FCN is 300→100→10; scale `feat`/`hidden` up for a denser "
        "belt closer to their Fig. 2.\n"
    ),
))


# ─────────────────────────────────────────────────────────────────────────────
# 2. rieck2019
# ─────────────────────────────────────────────────────────────────────────────

write("rieck2019", make_real(
    preset="rieck2019",
    title="Rieck et al. (2019) — Neural Persistence",
    summary=(
        "Bipartite weight graph, one layer at a time.  Neural Persistence = "
        "H0 persistence norm (p=2) — the scalar tracked in the paper."
    ),
    info_md=paper_info(
        arxiv="https://arxiv.org/abs/1812.09764",
        venue="ICLR 2019",
        datasets="MNIST, Fashion-MNIST, CIFAR-10.",
        models="MLPs and small CNNs; the paper sweeps depth, width, "
               "dropout, and batch normalisation.",
        headline="Neural Persistence (H0 persistence norm) tracks model "
                 "complexity and correlates with dropout / batch "
                 "normalisation gains (Sec. 4.1, Fig. 4).",
        github="https://github.com/BorgwardtLab/Neural-Persistence",
    ),
    setup_code=(
        "# MLP on MNIST, capturing the weights at EVERY epoch so we can track\n"
        "# Neural Persistence across training (the paper's early-stopping signal).\n"
        "X_tr, y_tr = load_mnist('train', n_samples=3000)\n"
        "X_te, y_te = load_mnist('test',  n_samples=600)\n"
        "\n"
        "model = SmallMLP(input_dim=784, hidden_dims=[100, 100], output_dim=10)\n"
        "view  = train_briefly(model, X_tr, y_tr, X_te, y_te, epochs=15,\n"
        "                      aspects=['weights'], snapshot_every=1,\n"
        "                      device=DEVICE, verbose=False)\n"
        "snap  = view.final_snapshot\n"
        "print(f'{len(view)} epoch snapshots; final val acc = {snap.accuracy}')\n"
    ),
    methodology=(
        "## What the paper does\n\n"
        "**Neural Persistence** measures the structural complexity of a trained "
        "layer from its weights alone. Each layer is viewed as a weighted "
        "**bipartite** graph (input units ↔ output units); a superlevel-set "
        "filtration on the (normalised) absolute weights gives an `H0` persistence "
        "diagram, and the paper's scalar is its **`p=2` persistence norm** — the "
        "Neural Persistence of that layer.\n\n"
        "**Methodology:** per-layer bipartite weight graph (each layer's |w| "
        "rescaled to [0,1]) → `H0` PH → `persistence_norm` (p=2).\n\n"
        "**Expected result (verified below):** Neural Persistence **rises "
        "monotonically as the network learns** and plateaus when validation "
        "accuracy does — so it works as an **early-stopping criterion without a "
        "validation set** (their Fig. 4).\n"
    ),
    reproduce_input="view",
    plot_cell=(
        '# Neural Persistence = H0 persistence norm in result.statistics.\n'
        'np_value = result.statistics["H0_persistence_norm"]\n'
        'print(f"Neural Persistence of the trained net = {np_value:.4f}")\n'
        '\n'
        '# Barcode of H0 generators (paper Fig. 4 style).\n'
        'fig = result.plot("barcode", dims=[0])\n'
    ),
    extra_cells=[
        ("markdown",
         "## 4. Verify the paper's finding — NP rises during training\n\n"
         "Rieck et al.'s central result: Neural Persistence **increases as the "
         "network learns** and tracks validation accuracy — an early-stopping "
         "signal computed from the weights alone.  We recompute NP from the "
         "weights captured at every epoch.\n"),
        ("code",
         "np_pipe = TDAPipeline.from_paper('rieck2019')\n"
         "np_curve  = [np_pipe.fit(s.weight_matrices()).statistics['H0_persistence_norm']\n"
         "             for s in view.snapshots]\n"
         "acc_curve = [s.accuracy for s in view.snapshots]\n"
         "epochs = range(1, len(np_curve) + 1)\n"
         "\n"
         "fig, ax1 = plt.subplots(figsize=(7, 4))\n"
         "ax1.plot(epochs, np_curve, 'o-', color='tab:blue')\n"
         "ax1.set_xlabel('epoch'); ax1.set_ylabel('Neural Persistence', color='tab:blue')\n"
         "ax2 = ax1.twinx()\n"
         "ax2.plot(epochs, acc_curve, 's--', color='tab:red')\n"
         "ax2.set_ylabel('val accuracy', color='tab:red')\n"
         "ax1.set_title('Neural Persistence rises with training (Rieck Fig. 4)')\n"
         "fig.tight_layout(); plt.show()\n"
         "print(f'NP: {np_curve[0]:.2f} (epoch 1) -> {np_curve[-1]:.2f} "
         "(epoch {len(np_curve)})  |  +{np_curve[-1]-np_curve[0]:.1f} over training')\n"),
    ],
    notes=(
        "* **Verified** Rieck et al.'s central finding on a real MNIST MLP: "
        "Neural Persistence rises monotonically across training and plateaus "
        "with validation accuracy — the early-stopping criterion.\n"
        "* This needs the paper's **per-layer** weight normalisation (the preset "
        "default; `edge_weight='global_normalized'` for one network-wide "
        "constant doesn't show the clean rise).\n"
        "* The paper's dropout/batch-norm comparison (Fig. 4) needs the full "
        "per-layer *max-persistence* normalisation that makes diagrams "
        "comparable across architectures — not implemented here, so that "
        "secondary result isn't reproduced.\n"
    ),
))


# ─────────────────────────────────────────────────────────────────────────────
# 3. gebhart2019 — uses plot_pathways_on_network (new!)
# ─────────────────────────────────────────────────────────────────────────────

write("gebhart2019", make_real(
    preset="gebhart2019",
    title="Gebhart et al. (2019) — Characterizing Activation Space",
    summary=(
        "Activation-weighted graph: edge weight = |W_ij * h_i| where h_i "
        "is the pre-synaptic activation.  Long H0 bars correspond to "
        "prominent signal pathways."
    ),
    info_md=paper_info(
        arxiv="https://arxiv.org/abs/1901.09496",
        venue="ICMLA 2019",
        datasets="MNIST.",
        models="Small MLPs trained for digit classification.",
        headline="H0 persistence intervals of activation graphs identify "
                 "task-relevant substructures.",
        github=None,
    ),
    setup_code=(
        "# Tiny MLP on MNIST + an extracted snapshot on a small input batch.\n"
        "X_tr, y_tr = load_mnist('train', n_samples=2000)\n"
        "X_te, y_te = load_mnist('test',  n_samples=400)\n"
        "\n"
        "model = SmallMLP(input_dim=784, hidden_dims=[64, 32], output_dim=10)\n"
        "view  = train_briefly(model, X_tr, y_tr, X_te, y_te, epochs=2)\n"
        "snap  = extract_snapshot(model, X_te[:64],\n"
        "                        aspects=['weights', 'activations'])\n"
        "snap.inputs = X_te[:64]   # so coupled_weight_activations can fill the first layer\n"
        "print(f'snapshot ready: {len(snap.weights)} weight layers')\n"
    ),
    methodology=(
        "## What the paper does\n\n"
        "Where Neural Persistence uses weights alone, this paper makes the graph "
        "**input-dependent**: each edge is weighted by `|W_ij · h_i|`, the product "
        "of the weight and the *pre-synaptic activation* `h_i` for a given input. "
        "Persistent homology of that activation-weighted graph yields `H0` "
        "generators whose long-lifetime bars correspond to the network's most "
        "prominent, *distinct* **signal pathways** for that input.\n\n"
        "**Methodology:** coupled (weight, pre-activation) per layer → "
        "`|w·h|` activation graph → `H0` PH → long bars = salient pathways.\n\n"
        "**The pathways are real H0 representatives.** For a graph, `H0` "
        "persistence *is* single-linkage clustering: the merge tree is the "
        "**maximum spanning forest**, and the longest-lifetime `H0` bars are its "
        "weakest tree edges.  Cutting the top few splits the network into its "
        "most persistent clusters — the distinct pathways.  The plot below "
        "(`mode='h0'`) colours each pathway and dashes the `H0` *death* edges "
        "that separate them — unlike a top-|w| magnitude heuristic, which is "
        "*not* topological.\n\n"
        "**Expected result:** the `H0` barcode isolates a handful of long bars "
        "(task-relevant pathways) above a band of short, noise-level bars; those "
        "pathways are the coloured clusters on the network graph.\n"
    ),
    reproduce_input="snap",
    plot_cell=(
        "# H0 of the |w.h| activation graph.  The diagram + the LONGEST bars:\n"
        'fig = result.plot("diagram", dims=[0])\n'
        'fig = result.plot("barcode", dims=[0], max_bars=30)   # capped, longest first\n'
    ),
    extra_cells=[
        ("markdown",
         "## 4. H0 signal pathways — coloured by persistent cluster (faithful)\n\n"
         "We build the **activation-weighted** edges `|w·h|` (the graph the H0 "
         "above is computed on) and highlight the **maximum-spanning-tree "
         "(merge-tree) edges** — the genuine H0 representatives — coloured by "
         "which of the top `n_pathways` persistent clusters they belong to.  "
         "Dashed black edges are the H0 *deaths* that separate the pathways.  "
         "The 784-pixel input layer is dropped for readability, so pathways are "
         "shown through the hidden neurons (`L0`=hidden-1, `L1`=hidden-2, "
         "`L2`=outputs).\n"),
        ("code",
         "from tanc.visualisation import plot_pathways_on_network\n"
         "\n"
         "# |w . h| activation-weighted edges, per layer (mean over the batch).\n"
         "coupled = snap.coupled_weight_activations()      # [(W, pre_act), ...]\n"
         "phi = [np.abs(W * pre.mean(axis=0)[:, None]) for W, pre in coupled]\n"
         "print('activation-weighted layers:', [p.shape for p in phi])\n"
         "\n"
         "# Faithful H0 pathways over the hidden neurons (drop the pixel layer).\n"
         "fig = plot_pathways_on_network(phi[1:], mode='h0', n_pathways=6)\n"),
    ],
    notes=(
        "* Reproduced: `|w·h|` activation-graph H0 PH on real MNIST activations, "
        "with the pathways drawn as the **actual H0 representative clusters** "
        "(maximum spanning forest), not a magnitude heuristic.\n"
        "* `plot_pathways_on_network(..., mode='magnitude')` keeps the old "
        "top-|w| proxy; `mode='h0'` is the topological version.\n"
        "* `result.plot('barcode', max_bars=N)` caps a dense H0 barcode to the "
        "N longest bars (the essential bar is marked `>`).\n"
    ),
))


# ─────────────────────────────────────────────────────────────────────────────
# 4. lacombe2021 — uses plot_tu_roc (new!)
# ─────────────────────────────────────────────────────────────────────────────

write("lacombe2021", [
    ("markdown",
     "# lacombe2021 — Lacombe et al. (2021) Topological Uncertainty\n\n"
     "Per-sample uncertainty score: distance to the Fréchet-mean MST "
     "diagram of the predicted class.\n"),
    ("markdown",
     "## What the paper does\n\n"
     "**Topological Uncertainty (TU)** turns a forward pass into an anomaly "
     "score *without* a test set. For each layer the paper builds the bipartite "
     "graph `|W · x|` of a sample and reads off its **minimum spanning tree** as "
     "a 0-dimensional persistence diagram. During fit it accumulates, per layer "
     "and per *predicted* class, the **Fréchet mean** (average) of those "
     "diagrams over the training set. A new input is scored by the average "
     "`L2` distance between its per-layer diagrams and the stored means for its "
     "predicted class.\n\n"
     "**Methodology:** per-(layer, sample) MST diagram → per-(layer, class) "
     "Fréchet-mean diagram → TU = mean `L2` distance to the predicted class's "
     "mean.\n\n"
     "**Expected result:** misclassified / out-of-distribution inputs score "
     "*higher* than correct in-distribution inputs, so TU works as an OOD "
     "detector and a distribution-shift monitor even when softmax confidence "
     "stays high (paper Fig. 5, Table 1).\n"),
    ("markdown", paper_info(
        arxiv="https://arxiv.org/abs/2105.04404",
        venue="IJCAI 2021",
        datasets="Synthetic + MNIST, CIFAR-10, molecular graphs.",
        models="Standard MLPs / GNNs trained at the relevant task.",
        headline="TU = mean over layers of L2 distance to the Frechet-mean "
                 "MST diagram of the predicted class; misclassified inputs "
                 "score higher than correct inputs (paper Fig. 5).",
        github=None,
    )),
    ("markdown",
     "**This notebook now runs on real MNIST data.**  A tiny MLP is "
     "trained briefly and pre-synaptic activations are extracted via the "
     "toolkit's `ModelExtractor`.\n"),
    ("code", REAL_IMPORTS),
    ("markdown", "## 1. Train a small MLP on MNIST\n"),
    ("code",
     "X_tr, y_tr = load_mnist('train', n_samples=2000)\n"
     "X_te, y_te = load_mnist('test',  n_samples=300)\n"
     "n_classes = 10\n"
     "model = SmallMLP(input_dim=784, hidden_dims=[64, 32], output_dim=n_classes)\n"
     "view  = train_briefly(model, X_tr, y_tr, X_te, y_te, epochs=2)\n"
     "print(f'final acc = {view.final_snapshot.accuracy:.4f}')\n"),
    ("markdown",
     "## 2. Extract real pre-synaptic activations for train + test\n"),
    ("code",
     "# Snapshot on train batch + test batch.  We pull pre-synaptic\n"
     "# activations per layer via snap.coupled_weight_activations().\n"
     "snap_tr = extract_snapshot(model, X_tr[:300], aspects=['weights', 'activations'])\n"
     "snap_te = extract_snapshot(model, X_te,       aspects=['weights', 'activations'])\n"
     "snap_tr.inputs = X_tr[:300]\n"
     "snap_te.inputs = X_te\n"
     "\n"
     "train_pre_acts = [pre for _, pre in snap_tr.coupled_weight_activations()]\n"
     "test_pre_acts  = [pre for _, pre in snap_te.coupled_weight_activations()]\n"
     "with torch.no_grad():\n"
     "    train_preds = model(torch.from_numpy(X_tr[:300])).argmax(1).numpy()\n"
     "    test_preds  = model(torch.from_numpy(X_te)).argmax(1).numpy()\n"
     "weights = snap_tr.weight_matrices()\n"
     "print(f'  pre-syn shapes: {[a.shape for a in train_pre_acts]}')\n"),
    ("markdown", "## 3. Fit Frechet means and score test inputs\n"),
    ("code",
     "from tanc.topo_tools import TopologicalUncertainty\n"
     "tu = TopologicalUncertainty(weight_matrices=weights, n_classes=n_classes)\n"
     "tu.fit(train_pre_acts, train_preds)\n"
     "scores = tu.score(test_pre_acts, test_preds)\n"
     "print('TU score stats:',\n"
     "      f'mean={scores.mean():.3f}, std={scores.std():.3f}')\n"),
    ("markdown",
     "## 4. Score distribution + misclassification ROC\n"),
    ("code",
     "from tanc.visualisation import "
     "plot_tu_score_distribution, plot_tu_roc\n"
     "plot_tu_score_distribution(scores, y_true=y_te, y_pred=test_preds)\n"
     "if (y_te != test_preds).sum() > 0 and (y_te == test_preds).sum() > 0:\n"
     "    plot_tu_roc(scores, y_true=y_te, y_pred=test_preds)\n"
     "else:\n"
     "    print('All-correct or all-wrong on this small test slice; ROC undefined.')\n"),
    ("markdown",
     "## Notes & gaps\n\n"
     "* Reproduced: per-sample TU on real MNIST activations from a "
     "briefly-trained MLP, plus histogram and misclassification ROC.\n"
     "* The paper's CIFAR-10 / GNN experiments follow the same flow — "
     "swap the loader and the model and the rest stays the same.\n"),
])


# ─────────────────────────────────────────────────────────────────────────────
# 5. naitzat2020 — uses plot_betti_layer_bars (new!)
# ─────────────────────────────────────────────────────────────────────────────

write("naitzat2020", [
    ("markdown",
     "# naitzat2020 — Naitzat et al. (2020) Topology of DNNs\n\n"
     "Geodesic kNN PH per layer.  Betti numbers *decrease* with depth as "
     "the network simplifies input topology.\n"),
    ("markdown",
     "## What the paper does\n\n"
     "The paper asks what a ReLU network *does to the shape of the data*. It "
     "feeds a topologically non-trivial dataset (e.g. entangled tori, linked "
     "rings) through the network and, at each layer, builds a **kNN graph** of "
     "the activations and computes its **geodesic** (shortest-path) distance "
     "matrix, then runs Vietoris–Rips persistent homology to read off the Betti "
     "numbers `β0` (components) and `β1` (loops).\n\n"
     "**Methodology:** per-layer activations → kNN geodesic distance → VR PH "
     "(`H0`,`H1`) → Betti numbers per layer.\n\n"
     "**Expected result:** `β0` and `β1` *decrease monotonically with depth* — "
     "a well-trained network progressively simplifies the data's topology, "
     "collapsing it toward linearly separable blobs by the final hidden layer "
     "(their Fig. 11 bar chart, reproduced below).\n"),
    ("markdown", paper_info(
        arxiv="https://arxiv.org/abs/2004.06093",
        venue="JMLR 2020",
        datasets="Synthetic 2-D / 3-D toy manifolds (entangled tori, "
                 "linked rings, annuli) designed to have known Betti numbers.",
        models="Small fully-connected ReLU/tanh networks of varying depth.",
        headline="Betti_0 and Betti_1 decrease monotonically with layer "
                 "index, hitting their trained-class baseline by the final "
                 "hidden layer.  Their Fig. 11 is the bar chart we now have.",
        github=None,
    )),
    ("markdown",
     "**This notebook runs on real data.**  A ReLU net is trained on a 2-D "
     "dataset with *known* topology (two concentric circles, β₁=2) and we run "
     "the per-layer geodesic PH on its activations.\n"),
    ("code", REAL_IMPORTS),
    ("markdown",
     "## 1. A topological dataset — two concentric circles (β₁ = 2)\n\n"
     "Two rings (inner vs outer class), 900 points.  The two loops give "
     "`β₁ = 2`; classifying them forces the network to *untangle* the "
     "topology.\n"),
    ("code",
     "rng = np.random.default_rng(0)\n"
     "n = 900\n"
     "t = rng.uniform(0, 2*np.pi, n)\n"
     "r = np.where(np.arange(n) < n//2, 0.5, 1.5) + 0.03*rng.standard_normal(n)\n"
     "X = np.c_[r*np.cos(t), r*np.sin(t)].astype(np.float32)\n"
     "y = (np.arange(n) >= n//2).astype(np.int64)\n"
     "plt.figure(figsize=(4, 4)); plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=6)\n"
     "plt.gca().set_aspect('equal'); plt.title('two concentric circles (β₁=2)'); plt.show()\n"),
    ("markdown",
     "## 2. Train a *wide* ReLU net to 100% accuracy\n\n"
     "Faithful reproductions need a network that actually untangles the data — "
     "a wide MLP trained to (near-)zero error on the laptop GPU.\n"),
    ("code",
     "model = SmallMLP(input_dim=2, hidden_dims=[40, 40, 40, 40], output_dim=2)\n"
     "train_briefly(model, X, y, X, y, epochs=300,\n"
     "              aspects=['activations'], device=DEVICE, verbose=False)\n"
     "print('train accuracy:', accuracy(model, X, y, device=DEVICE))\n"
     "snap = extract_snapshot(model.to('cpu').eval(), X,\n"
     "                        aspects=['activations'], layer_selection='all_linear')\n"
     "layers = [X] + snap.all_activation_matrices()[:-1]   # input + hidden layers\n"),
    ("markdown",
     "## 3. Per-layer persistent homology → Betti decreases with depth\n\n"
     "We standardise each layer and run **Vietoris–Rips** PH, counting only the "
     "**persistent** `H1` loops (lifetime > ½ · max) — the robust Betti number "
     "the paper tracks.  (The paper's exact metric is graph-geodesic; VR on the "
     "standardised cloud gives the same simplification cleanly at this scale.)\n\n"
     "*(Explicit extraction on purpose: this analysis needs the **input** layer "
     "prepended and each layer **standardised** — prep the model-first adapter "
     "can't express, unlike the activation/weight presets elsewhere.)*\n"),
    ("code",
     "# Standardise each layer so scale doesn't dominate the filtration.\n"
     "layers_n = [(A - A.mean(0)) / (A.std(0) + 1e-9) for A in layers]\n"
     "pipe = TDAPipeline.from_paper('naitzat2020', builder_kwargs__distance='euclidean')\n"
     "results = pipe.fit(layers_n)         # list[TopoResult], one per layer\n"
     "\n"
     "def persistent_b1(r, frac=0.5):\n"
     "    d = r.diagram(1)\n"
     "    if not len(d):\n"
     "        return 0\n"
     "    lt = d[:, 1] - d[:, 0]\n"
     "    return int((lt > frac * lt.max()).sum())\n"
     "\n"
     "b1 = [persistent_b1(r) for r in results]\n"
     "for i, (r, b) in enumerate(zip(results, b1)):\n"
     "    print(f'  layer {i}: persistent β1 = {b}   (raw H1 bars = {len(r.diagram(1))})')\n"),
    ("code",
     "plt.figure(figsize=(6, 4))\n"
     "plt.plot(range(len(b1)), b1, 'o-')\n"
     "plt.xticks(range(len(b1)), ['input'] + [f'h{i}' for i in range(1, len(b1))])\n"
     "plt.ylabel('persistent β1 (loops)'); plt.xlabel('layer')\n"
     "plt.title('Topology simplifies with depth (two circles)'); plt.grid(alpha=0.3)\n"
     "plt.tight_layout(); plt.show()\n"
     "\n"
     "from tanc.visualisation import plot_diagram_comparison\n"
     "plot_diagram_comparison({f'L{i}': r.ph_result for i, r in enumerate(results)},\n"
     "                        kind='diagram', dims=[1])\n"),
    ("markdown",
     "## Notes & gaps\n\n"
     "* Reproduced the paper's signature result on **real activations** of a "
     "wide net trained to 100%: the input's **two loops** (β₁=2) are "
     "progressively simplified, ending at **one** deep in the net — Betti "
     "decreases with depth (their Fig. 11).\n"
     "* The paper's exact filtration is graph-geodesic; we use standardised "
     "Vietoris–Rips, which gives the same simplification without the small-scale "
     "kNN-geodesic noise.  Counting *persistent* bars is what makes the trend "
     "robust.\n"),
])


# ─────────────────────────────────────────────────────────────────────────────
# 6. karuppiah2025
# ─────────────────────────────────────────────────────────────────────────────

write("karuppiah2025", [
    ("markdown",
     "# karuppiah2025 — Karuppiah et al. (2025) TDA for Understanding DL\n\n"
     "Direct VR persistent homology on Euclidean activation distances, "
     "**per layer**, of a trained CNN.  Diagram statistics track layer depth.\n"),
    ("markdown",
     "## What the paper does\n\n"
     "A survey/methodology article framing TDA as a lens on deep learning.  The "
     "concrete recipe: take each layer's activation matrix as a point cloud "
     "(one point per input sample), compute **Euclidean** distances between the "
     "activation vectors, and run Vietoris–Rips persistent homology on that "
     "distance matrix — one diagram per layer.\n\n"
     "**Methodology:** per-layer activations → Euclidean distance → VR PH "
     "(`H0`,`H1`) → persistence-diagram statistics per layer.\n\n"
     "**Expected result:** the diagram summaries (number of `H1` loops, "
     "persistence entropy, total persistence) **vary systematically with "
     "depth** — a trained network progressively simplifies the topology of its "
     "representations, so later layers have fewer, shorter-lived generators.\n"),
    ("markdown", paper_info(
        venue="IEEE ICTBIG 2025",
        datasets="Standard image classification benchmarks (the paper is "
                 "broadly survey-style).",
        models="Generic CNNs / MLPs as illustrative examples.",
        headline="Persistence-diagram statistics correlate with layer "
                 "depth and learned representation quality.",
        github=None,
    )),
    ("markdown",
     "**This notebook runs on real data.**  A small CNN is trained briefly on "
     "the local CIFAR-10 cache and its per-layer activations are analysed.\n"),
    ("code", REAL_IMPORTS),
    ("markdown", "## 1. Train a small CNN on CIFAR-10\n"),
    ("code",
     "X_tr, y_tr = load_cifar10('train', n_samples=2000)\n"
     "X_te, y_te = load_cifar10('test',  n_samples=400)\n"
     "model = CNN_FCN(in_channels=3, num_classes=10, width=16, feat=64, hidden=32)\n"
     "train_briefly(model, X_tr, y_tr, X_te, y_te, epochs=4,\n"
     "              aspects=['activations'], device=DEVICE, verbose=False)\n"
     "print('trained.')\n"),
    ("markdown",
     "## 2. Per-layer VR persistent homology — model-first\n\n"
     "`fit_model(..., representation='activations')` extracts each layer's "
     "activation matrix on a 64-sample batch and runs VR PH per layer, "
     "returning one `TopoResult` per layer.\n"),
    ("code",
     'pipe = TDAPipeline.from_paper("karuppiah2025")\n'
     "results = pipe.fit_model(model, X_te[:64],\n"
     "                         representation='activations',\n"
     "                         layer_selection='linear_and_conv')\n"
     "for i, r in enumerate(results):\n"
     "    print(f'  layer {i}: H0={len(r.diagram(0))} bars, H1={len(r.diagram(1))} loops')\n"),
    ("markdown",
     "## 3. Diagram statistics vs layer depth\n\n"
     "The paper's headline: topological summaries change monotonically with "
     "depth as the representation is simplified.\n"),
    ("code",
     "layers = list(range(len(results)))\n"
     "n_h1     = [len(r.diagram(1)) for r in results]\n"
     "entropy  = [r.statistics.get('H1_persistence_entropy', 0.0) for r in results]\n"
     "totpers  = [r.statistics.get('H1_total_persistence', 0.0) for r in results]\n"
     "\n"
     "fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))\n"
     "for ax, y, lab in zip(axes, [n_h1, entropy, totpers],\n"
     "                      ['# H1 loops', 'H1 persistence entropy', 'H1 total persistence']):\n"
     "    ax.plot(layers, y, 'o-'); ax.set_xlabel('layer depth'); ax.set_ylabel(lab)\n"
     "    ax.set_xticks(layers); ax.grid(alpha=0.3)\n"
     "fig.suptitle('Per-layer H1 summaries vs depth (CNN_FCN on CIFAR-10)')\n"
     "fig.tight_layout(); plt.show()\n"),
    ("markdown", "## 4. Per-layer persistence diagrams\n"),
    ("code",
     "from tanc.visualisation import plot_diagram_comparison\n"
     "plot_diagram_comparison({f'L{i}': r.ph_result for i, r in enumerate(results)},\n"
     "                        kind='diagram', dims=[0, 1])\n"),
    ("markdown",
     "## Notes & gaps\n\n"
     "* Reproduced on **real per-layer activations** (model-first via "
     "`fit_model`), not synthetic points.\n"
     "* The paper is methodology-heavy; the depth trend (fewer / shorter-lived "
     "H1 generators deeper in the net) is the qualitative result to look for "
     "— echoing Naitzat et al. (2020).  Train longer / wider for a cleaner "
     "monotonic trend.\n"
     "* VR PH cost grows with the **sample count** (points in the complex); "
     "64 samples keeps H1 fast.\n"),
])


# ─────────────────────────────────────────────────────────────────────────────
# 7. ballester2024
# ─────────────────────────────────────────────────────────────────────────────

write("ballester2024", make_real(
    preset="ballester2024",
    title="Ballester et al. (2024) — Predicting Generalization via TDA",
    summary=(
        "Correlation-distance VR PH over concatenated activations.  PH "
        "summary statistics regress on the generalization gap."
    ),
    info_md=paper_info(
        arxiv="https://arxiv.org/abs/2203.12330",
        venue="Neurocomputing 2024",
        datasets="PGDL competition datasets — CIFAR-10 + many "
                 "pre-trained CNN configurations.  Here we use the local "
                 "CIFAR-10 cache and one briefly-trained CNN.",
        models="PGDL benchmark CNNs.  Here: a tiny SmallCNN — same "
               "methodology, fewer parameters.",
        headline="Linear models on persistence-diagram summaries predict "
                 "the generalization gap with rank correlation comparable "
                 "to the PGDL winners.",
        github="https://github.com/rballeba/PredictingGeneralizationGapUsingPersistentHomology",
    ),
    setup_code=(
        "X_tr, y_tr = load_cifar10('train', n_samples=1500)\n"
        "X_te, y_te = load_cifar10('test',  n_samples=400)\n"
        "\n"
        "model = SmallCNN(in_channels=3, num_classes=10, width=8)\n"
        "view  = train_briefly(model, X_tr, y_tr, X_te, y_te, epochs=2)\n"
        "snap  = extract_snapshot(model, X_te[:200],\n"
        "                        aspects=['activations'])\n"
        "print(f'final acc = {view.final_snapshot.accuracy}')\n"
    ),
    data_code=(
        "# Pipeline expects concatenated activations across the network.\n"
        "acts = list(snap.activations.values())\n"
        "data = np.concatenate([a.reshape(a.shape[0], -1) for a in acts], axis=1)\n"
        "print(f'concatenated activation matrix: {data.shape}')\n"
    ),
    plot_cell=(
        "from tanc.visualisation import plot_persistence_image\n"
        'plot_persistence_image(result.ph_result, dim=1, resolution=25)\n'
        '\n'
        '# PH summary statistics — these are the regressor inputs.\n'
        'for k, v in result.statistics.items():\n'
        '    print(f"  {k:40s} = {v:.4f}")\n'
    ),
    notes=(
        "* Reproduced: correlation-distance VR PH on real CIFAR-10 "
        "activations + persistence-image vectorisation.\n"
        "* The full regression onto generalization gap needs a *set* of "
        "trained models; this notebook produces the per-model PH inputs.\n"
    ),
))


# ─────────────────────────────────────────────────────────────────────────────
# 8. ramamurthy2019
# ─────────────────────────────────────────────────────────────────────────────

write("ramamurthy2019", [
    ("markdown",
     "# ramamurthy2019 — Ramamurthy et al. (2019) TDA of Decision Boundaries\n\n"
     "Compare two **trained classifiers** by the topology of their decision "
     "boundaries — the paper's model-selection use of the labelled complex.\n"),
    ("markdown",
     "## What the paper does\n\n"
     "TDA applied to **decision boundaries** for model selection.  Sample the "
     "input space, label each point by a classifier's **prediction**, and build "
     "a **labelled Vietoris–Rips** complex in which same-(predicted-)class pairs "
     "are masked to a sentinel distance — so the filtration only grows edges "
     "*between* the predicted classes.  The resulting persistent homology "
     "characterises the **decision-boundary** topology, not the data clusters.\n\n"
     "**Methodology:** classifier predictions on a grid → same-class-masked VR "
     "complex → PH of the cross-class boundary.\n\n"
     "**Expected result:** classifiers whose boundaries have different topology "
     "give different diagrams, so the diagrams *select the right model* without "
     "a held-out metric.\n"),
    ("markdown", paper_info(
        arxiv="https://arxiv.org/abs/1805.09949",
        venue="ICML 2019",
        datasets="Synthetic 2-D problems (XOR, concentric rings) + UCI tabular.",
        models="Logistic regression, RBF-SVM, MLPs — model selection between "
               "them.  Here: logistic regression vs an MLP on concentric circles.",
        headline="The labelled-complex PH diagram differentiates models that "
                 "fit decision boundaries of different topology, guiding model "
                 "selection.",
        github=None,
    )),
    ("markdown",
     "**This notebook trains real classifiers.**  Concentric circles — a linear "
     "model *cannot* separate them (wrong, straight boundary); an MLP learns the "
     "correct *circular* boundary.\n"),
    ("code", REAL_IMPORTS),
    ("markdown", "## 1. Concentric circles + two classifiers\n"),
    ("code",
     "from sklearn.linear_model import LogisticRegression\n"
     "rng = np.random.default_rng(0)\n"
     "n = 160; t = rng.uniform(0, 2*np.pi, n)\n"
     "r = np.where(np.arange(n) < n//2, 0.5, 1.5) + 0.03*rng.standard_normal(n)\n"
     "X = np.c_[r*np.cos(t), r*np.sin(t)].astype(np.float32)\n"
     "y = (np.arange(n) >= n//2).astype(np.int64)\n"
     "\n"
     "linear = LogisticRegression().fit(X, y)\n"
     "mlp = SmallMLP(2, [32, 16], 2)\n"
     "train_briefly(mlp, X, y, X, y, epochs=150, aspects=['activations'], device=DEVICE, verbose=False)\n"
     "mlp.cpu().eval()\n"
     "print('linear acc =', (linear.predict(X) == y).mean(),\n"
     "      ' MLP acc =', accuracy(mlp, X, y, device='cpu'))\n"),
    ("markdown",
     "## 2. Label a grid by each model's prediction → the decision boundary\n"),
    ("code",
     "g = 22; xs = np.linspace(-2, 2, g); xx, yy = np.meshgrid(xs, xs)\n"
     "grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)\n"
     "pred_lin = linear.predict(grid)\n"
     "with torch.no_grad():\n"
     "    pred_mlp = mlp(torch.from_numpy(grid)).argmax(1).numpy()\n"
     "\n"
     "fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))\n"
     "for ax, (name, p) in zip(axes, [('linear', pred_lin), ('MLP', pred_mlp)]):\n"
     "    ax.contourf(xx, yy, p.reshape(g, g), levels=1, cmap='coolwarm', alpha=0.4)\n"
     "    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=8, edgecolors='k', linewidths=0.2)\n"
     "    ax.set_title(f'{name} decision regions'); ax.set_aspect('equal')\n"
     "plt.tight_layout(); plt.show()\n"),
    ("markdown",
     "## 3. Boundary topology selects the model\n\n"
     "The labelled-complex `H1` of each model's prediction grid: the wrong "
     "(straight) boundary and the correct (circular) boundary give "
     "**measurably different** diagrams.\n"),
    ("code",
     'pipe = TDAPipeline.from_paper("ramamurthy2019")\n'
     "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
     "for ax, (name, p) in zip(axes, [('linear (straight)', pred_lin), ('MLP (circular)', pred_mlp)]):\n"
     "    d1 = pipe.fit((grid, p)).diagram(1)\n"
     "    ax.hist(d1[:, 0], bins=30, color='tab:purple', alpha=0.8)\n"
     "    ax.set_title(f'{name}\\nH1 boundary diagram (n={len(d1)})')\n"
     "    ax.set_xlabel('birth = boundary scale')\n"
     "    print(f'  {name:18s}: H1 bars = {len(d1):6d}  mean birth = {d1[:, 0].mean():.2f}')\n"
     "plt.tight_layout(); plt.show()\n"),
    ("markdown",
     "## Notes & gaps\n\n"
     "* Reproduced the paper's **model-selection** use on **real trained "
     "classifiers**: the labelled-complex `H1` of the linear model's straight "
     "boundary differs clearly from the MLP's correct circular boundary "
     "(different bar counts and birth scales).\n"
     "* The raw H1 *count* is large (every cross-class pair seeds structure in "
     "the sentinel-masked complex), so compare the **diagrams** (counts / births "
     "/ Betti curves), not a single loop count — exactly the diagram-distance "
     "comparison the paper uses to rank models.\n"),
])


# ─────────────────────────────────────────────────────────────────────────────
# 9. liu2023
# ─────────────────────────────────────────────────────────────────────────────

write("liu2023", [
    ("markdown",
     "# liu2023 — Liu et al. (2023) ReLU NNs, Polyhedra & PH\n\n"
     "Binary ReLU activation patterns partition input space into polyhedral "
     "regions; PH on the Hamming distance between regions recovers the "
     "**loops of the data manifold**.\n"),
    ("markdown",
     "## What the paper does\n\n"
     "A ReLU network partitions input space into **polyhedral regions** — one "
     "per distinct binary activation pattern (which units are on/off).  Each "
     "input lands in one region; treating the unique patterns as points with a "
     "**Hamming distance** and running persistent homology reads off the "
     "topology of how those regions tile the data.\n\n"
     "**Methodology:** trained ReLU net → per-input binary activation pattern → "
     "Hamming distance over unique regions → VR PH (`H0`,`H1`).\n\n"
     "**Expected result:** the number of persistent `H1` generators recovers "
     "the number of **loops** in the underlying data manifold (their Sec. 5) — "
     "feed data on a circle (one loop) and a single dominant `H1` generator "
     "should appear, well above the noise.\n"),
    ("markdown", paper_info(
        arxiv="https://arxiv.org/abs/2306.17418",
        venue="TAG-ML 2023 (PMLR 221)",
        datasets="Small synthetic 2-D classification problems with known "
                 "homology (annulus, two-moons).",
        models="Small ReLU MLPs trained to classification.",
        headline="PH H1 generators on the polyhedral decomposition graph "
                 "correctly count the number of loops in the underlying data "
                 "manifold (their Sec. 5 experiments).",
        github="https://github.com/cglrtrgy/GoL_Toolbox",
    )),
    ("markdown",
     "**This notebook runs on real data.**  A ReLU MLP is trained on a 2-D "
     "dataset with *known* topology (a circle, β1 = 1) and its polyhedral "
     "decomposition is analysed.\n"),
    ("code", REAL_IMPORTS),
    ("markdown",
     "## 1. A dataset with known topology — a circle (β₁ = 1)\n\n"
     "Points sampled on a noisy circle form a single 1-D loop.  The label is a "
     "half-plane split (`sin θ > 0`) so the decision boundary crosses the loop "
     "and the network must carve regions around it.\n"),
    ("code",
     "rng = np.random.default_rng(0)\n"
     "N = 600\n"
     "theta = rng.uniform(0, 2 * np.pi, N)\n"
     "r = 1.0 + 0.04 * rng.standard_normal(N)\n"
     "X = np.column_stack([r * np.cos(theta), r * np.sin(theta)]).astype(np.float32)\n"
     "y = (np.sin(theta) > 0).astype(np.int64)\n"
     "\n"
     "plt.figure(figsize=(4, 4))\n"
     "plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=10)\n"
     "plt.gca().set_aspect('equal'); plt.title('Circle data (1 loop), coloured by label')\n"
     "plt.show()\n"),
    ("markdown",
     "## 2. Train a ReLU MLP and read its activation patterns\n\n"
     "For each input we take the sign pattern of the hidden ReLU units "
     "(on/off) — its polyhedral region.  We extract the hidden layers' "
     "activations explicitly and binarise them, **excluding the output layer** "
     "(the method is defined on the hidden ReLU units).\n"),
    ("code",
     "model = SmallMLP(input_dim=2, hidden_dims=[64, 64], output_dim=2)\n"
     "train_briefly(model, X, y, X, y, epochs=60,\n"
     "              aspects=['activations'], device=DEVICE, verbose=False)\n"
     "\n"
     "snap = extract_snapshot(model.to('cpu').eval(), X,\n"
     "                        aspects=['activations'], layer_selection='all_linear')\n"
     "acts = snap.all_activation_matrices()           # [fc1, fc2, output]\n"
     "patterns = np.concatenate([(a > 0).astype(int) for a in acts[:-1]], axis=1)\n"
     "print(f'pattern matrix {patterns.shape} -> '\n"
     "      f'{len(np.unique(patterns, axis=0))} unique polyhedral regions')\n"),
    ("markdown",
     "## 3. Visualise the polyhedral decomposition in input space\n\n"
     "Because the input is 2-D we can *see* the regions: evaluate the ReLU "
     "patterns on a dense grid — each distinct pattern is one linear "
     "(polyhedral) region, and the boundaries are the network's piecewise-"
     "linear cuts.  The data loop (points) threads through these polygons — "
     "exactly the structure `H1` measures.\n"),
    ("code",
     "from tanc.visualisation import plot_polyhedral_regions\n"
     "\n"
     "g = 300\n"
     "xs = np.linspace(-1.5, 1.5, g)\n"
     "xx, yy = np.meshgrid(xs, xs)\n"
     "grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)\n"
     "snap_g = extract_snapshot(model, grid, aspects=['activations'],\n"
     "                          layer_selection='all_linear')\n"
     "grid_patterns = np.concatenate(\n"
     "    [(a > 0).astype(int) for a in snap_g.all_activation_matrices()[:-1]], axis=1)\n"
     "fig = plot_polyhedral_regions(xx, yy, grid_patterns, points=X, point_labels=y)\n"
     "plt.show()\n"),
    ("markdown",
     "## 4. Persistent homology of the polyhedral decomposition\n"),
    ("code",
     'pipe = TDAPipeline.from_paper("liu2023")\n'
     "result, fig = pipe.reproduce(patterns)\n"
     "result.describe()\n"),
    ("markdown",
     "## 5. The data's loop is recovered by H1\n\n"
     "A single `H1` generator should stand out far above the rest — that is "
     "the circle's one loop, found purely from the network's region structure.\n"),
    ("code",
     "d1 = result.diagram(1)\n"
     "life = np.sort(d1[:, 1] - d1[:, 0])[::-1]\n"
     "print('top H1 lifetimes:', np.round(life[:6], 3))\n"
     "print(f'dominant generator = {life[0]:.3f}, '\n"
     "      f'{life[0] / life[1]:.1f}x the next -> the 1 loop of the circle')\n"
     "fig = result.plot('diagram', dims=[0, 1])\n"),
    ("markdown",
     "## 6. How the polyhedral regions relate to the task\n\n"
     "Within a single region every hidden ReLU is fixed on/off, so the network "
     "is an **affine** map there — the classifier is **piecewise-linear**.  Its "
     "**decision boundary** (bold black below) is therefore a polyline: a "
     "straight segment inside each region it crosses, bending only at region "
     "boundaries.  So the net *spends* regions where the task needs a curvy "
     "boundary — few large regions in easy areas, many small ones hugging a hard "
     "boundary.  The decomposition is the network's piecewise-linear scaffolding "
     "for the task; section 4's PH then measures the *topology* of how those "
     "regions tile the data.\n"),
    ("code",
     "from sklearn.datasets import make_moons\n"
     "\n"
     "def make_2d(kind, n=600, seed=0):\n"
     "    '''Four 2-D tasks with different boundary shapes.'''\n"
     "    rng = np.random.default_rng(seed)\n"
     "    if kind == 'circle':\n"
     "        th = rng.uniform(0, 2*np.pi, n); r = 1 + 0.04*rng.standard_normal(n)\n"
     "        X = np.c_[r*np.cos(th), r*np.sin(th)]; y = (np.sin(th) > 0).astype(int)\n"
     "    elif kind == 'moons':\n"
     "        X, y = make_moons(n, noise=0.08, random_state=seed); X = (X-X.mean(0))/X.std(0)\n"
     "    elif kind == 'xor':\n"
     "        X = rng.uniform(-1.4, 1.4, (n, 2)); y = ((X[:,0] > 0) ^ (X[:,1] > 0)).astype(int)\n"
     "    elif kind == 'spiral':\n"
     "        k = n//2; t = np.sqrt(rng.uniform(0.04, 1, k))*2.8*np.pi\n"
     "        A = np.c_[t*np.cos(t), t*np.sin(t)]; B = np.c_[t*np.cos(t+np.pi), t*np.sin(t+np.pi)]\n"
     "        X = np.vstack([A, B])/9; y = np.r_[np.zeros(k), np.ones(k)].astype(int)\n"
     "    return X.astype(np.float32), y.astype(np.int64)\n"
     "\n"
     "def decomposition(model, X, g=220):\n"
     "    '''Grid ReLU patterns + predicted class over the input window.'''\n"
     "    model = model.to('cpu').eval()\n"
     "    lim = float(np.abs(X).max() * 1.15); xs = np.linspace(-lim, lim, g)\n"
     "    xx, yy = np.meshgrid(xs, xs)\n"
     "    grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)\n"
     "    s = extract_snapshot(model, grid, aspects=['activations'], layer_selection='all_linear')\n"
     "    pat = np.concatenate([(a > 0).astype(int) for a in s.all_activation_matrices()[:-1]], axis=1)\n"
     "    with torch.no_grad():\n"
     "        pred = model(torch.from_numpy(grid)).argmax(1).numpy()\n"
     "    return xx, yy, pat, pred\n"),
    ("markdown",
     "## 7. A gallery of 2-D tasks — decomposition + persistence diagram\n\n"
     "Each row: **left** — the polyhedral regions, their boundaries, the bold "
     "**decision boundary** and the data; **right** — the **persistence "
     "diagram** of that network's activation patterns.  Watch how regions crowd "
     "where the boundary curves, and note that only the **circle** shows a "
     "prominent off-diagonal `H1` point (its loop) — the contractible "
     "moons/xor/spiral keep their `H1` near the diagonal (noise).\n"),
    ("code",
     'pipe = TDAPipeline.from_paper("liu2023")\n'
     "fig, axes = plt.subplots(4, 2, figsize=(10, 18))\n"
     "for (ax_dec, ax_dgm), kind in zip(axes, ['circle', 'moons', 'xor', 'spiral']):\n"
     "    torch.manual_seed(0)\n"
     "    Xk, yk = make_2d(kind)\n"
     "    mk = SmallMLP(2, [64, 64], int(yk.max()) + 1)\n"
     "    train_briefly(mk, Xk, yk, Xk, yk, epochs=60,\n"
     "                  aspects=['activations'], device=DEVICE, verbose=False)\n"
     "    # left: polyhedral decomposition + decision boundary\n"
     "    xx, yy, pat, pred = decomposition(mk, Xk)\n"
     "    plot_polyhedral_regions(xx, yy, pat, points=Xk, point_labels=yk, decision=pred,\n"
     "                            ax=ax_dec, title=f'{kind} — {len(np.unique(pat, axis=0))} regions')\n"
     "    # right: persistence diagram of the data-point activation patterns\n"
     "    sd = extract_snapshot(mk.to('cpu').eval(), Xk,\n"
     "                          aspects=['activations'], layer_selection='all_linear')\n"
     "    dpat = np.concatenate([(a > 0).astype(int) for a in sd.all_activation_matrices()[:-1]], axis=1)\n"
     "    res = pipe.fit(dpat)\n"
     "    d1 = res.diagram(1)\n"
     "    top = float(np.sort(d1[:, 1] - d1[:, 0])[::-1][0]) if len(d1) else 0.0\n"
     "    res.plot('diagram', dims=[0, 1], ax=ax_dgm, title=f'{kind} — PH (top H1={top:.2f})')\n"
     "fig.tight_layout(); plt.show()\n"),
    ("markdown",
     "## Notes & gaps\n\n"
     "* Reproduced on a **real ReLU network** over data with known topology: a "
     "1-loop circle yields one dominant `H1` generator above the noise floor.\n"
     "* **Regions ↔ task:** the classifier is affine within each region, so the "
     "decision boundary is piecewise-linear and regions concentrate where it "
     "must curve (gallery).  Clean `H1` loop-counting (section 5) only applies "
     "when the *data manifold itself* has loops — moons/xor/spiral are "
     "contractible, so their `H1` is noise despite rich region structure.\n"
     "* Loop-counting is sensitive to the data/task/architecture (the network "
     "must form regions that wrap the loop — a *linearly separable* task won't).  "
     "Two well-separated circles, for instance, are split linearly and show no "
     "loops; the paper's Sec. 5 uses carefully designed manifolds.\n"
     "* Region-count evolution over training: capture a `view` and use "
     "`plot_ph_statistic_trajectory(view, stat='total_persistence')`.\n"),
])


# ─────────────────────────────────────────────────────────────────────────────
# 10. rathore2021 — uses node_exemplars (new!)
# ─────────────────────────────────────────────────────────────────────────────

write("rathore2021", [
    ("markdown",
     "# rathore2021 — Rathore et al. (2021) TopoAct\n\n"
     "Mapper on a trained CNN's **activation vectors** (L2-norm lens) — the "
     "graph exposes concept clusters in activation space.\n"),
    ("markdown",
     "## What the paper does\n\n"
     "**TopoAct** explores the *shape* of a layer's activation space with "
     "**Mapper**.  Each input's activation vector is a point; the lens is its "
     "**`L2` norm**; overlapping lens bins are clustered and linked into a "
     "Mapper graph whose nodes are activation clusters and edges are overlaps.\n\n"
     "**Methodology:** activation vectors → `L2`-norm lens → Mapper graph "
     "(nodes = clusters, edges = overlap).\n\n"
     "**Expected result:** the graph exposes interpretable **concept clusters** "
     "(at ImageNet scale, dogs/vehicles/… with branching subhierarchy).  At our "
     "small scale, Mapper nodes should still group inputs of the **same class** "
     "— colour the graph by label to see it.\n"),
    ("markdown", paper_info(
        arxiv="https://arxiv.org/abs/1912.06332",
        venue="Computer Graphics Forum 2021 (EuroVis)",
        datasets="ImageNet, CIFAR-10.",
        models="InceptionV1 (GoogLeNet).  Here: a small CNN on CIFAR-10.",
        headline="The Mapper graph of mixed4c activations reveals high-level "
                 "concept clusters with branching topology matching the "
                 "ImageNet subhierarchy.",
        github="https://github.com/tdavislab/TopoAct",
        notes_url="Live demo: https://tdavislab.github.io/TopoAct/",
    )),
    ("markdown",
     "**This notebook runs on real activations.**  A small CNN is trained on "
     "the local CIFAR-10 cache and its penultimate activations are Mapper-ed.\n"),
    ("code", REAL_IMPORTS),
    ("markdown", "## 1. Train a CNN, take a layer's activation vectors\n"),
    ("code",
     "X_tr, y_tr = load_cifar10('train', n_samples=5000)\n"
     "X_te, y_te = load_cifar10('test',  n_samples=500)\n"
     "# A plain (no-BatchNorm) CNN: the L2-norm lens needs activation *magnitude*\n"
     "# to vary across inputs, which BatchNorm would flatten away.\n"
     "model = CNN_FCN(in_channels=3, num_classes=10, width=24, feat=96, hidden=48)\n"
     "train_briefly(model, X_tr, y_tr, X_te, y_te, epochs=40,\n"
     "              aspects=['activations'], device=DEVICE, verbose=False)\n"
     "print('test accuracy:', accuracy(model, X_te, y_te, device=DEVICE))\n"
     "n = 400; labels = y_te[:n]\n"),
    ("markdown",
     "## 2. Mapper of activation space, coloured by class — model-first\n\n"
     "`fit_model` extracts the penultimate (`fc1`) layer's activations and runs "
     "Mapper — no manual extraction.  A coarser cover than the paper's "
     "`Mapper(70, 0.3)` keeps the small-scale graph readable.\n"),
    ("code",
     'pipe = TDAPipeline.from_paper("rathore2021",\n'
     "                            tool_kwargs__n_intervals=30,\n"
     "                            tool_kwargs__overlap_frac=0.3)\n"
     "result = pipe.fit_model(model, X_te[:n],\n"
     "                        representation='fc1', layer_selection='all_linear')\n"
     "fig = result.plot('graph', color_by=labels,\n"
     "                  title='TopoAct Mapper of fc1 activations (coloured by class)')\n"
     "print('Mapper graph:', result.mapper.stats)\n"),
    ("markdown",
     "## 3. Per-node exemplars — do nodes group one class?\n\n"
     "`node_exemplars` returns the representative input indices per node; we "
     "look at their class labels to check each cluster is class-coherent.\n"),
    ("code",
     "from tanc.topo_tools import node_exemplars\n"
     "members = result.mapper.node_members\n"
     "for nid, idxs in list(members.items())[:8]:\n"
     "    cls = labels[np.asarray(idxs)]\n"
     "    vals, cnts = np.unique(cls, return_counts=True)\n"
     "    purity = cnts.max() / cnts.sum()\n"
     "    print(f'  node {nid:2d}: {len(idxs):3d} inputs, dominant class {vals[cnts.argmax()]} '\n"
     "          f'(purity {purity:.2f})')\n"),
    ("markdown",
     "## Notes & gaps\n\n"
     "* Reproduced on **real CNN activations**: the L2-norm-lens Mapper graph, "
     "coloured by class, with per-node class purity.\n"
     "* **Proper model not reproducible here.** TopoAct's clean concept "
     "hierarchy comes from **InceptionV1 on ImageNet** — a large *pretrained* "
     "net we can't load (no `torchvision`/ImageNet on this machine).  A "
     "from-scratch CIFAR CNN only yields *partially* class-coherent nodes "
     "(dominant-class purity ≈ 4× the 0.1 random level).  A bigger/deeper net "
     "doesn't help: BatchNorm flattens the activation magnitudes the L2-norm "
     "lens relies on, so the plain CNN above is the better stand-in.\n"
     "* `node_exemplars(...)` returns indices to render image thumbnails in your "
     "own (image-aware) pipeline.\n"),
])


# ─────────────────────────────────────────────────────────────────────────────
# 11. zhou2023 — uses mapper_gw_distance (new!)
# ─────────────────────────────────────────────────────────────────────────────

write("zhou2023", [
    ("markdown",
     "# zhou2023 — Zhou et al. (2023) Comparing Mapper Graphs Across Layers\n\n"
     "One **L2-norm Mapper graph per layer**, compared with the "
     "**Gromov–Wasserstein** distance — the paper's central tool.\n"),
    ("markdown",
     "## What the paper does\n\n"
     "Builds a **Mapper** graph of the activations at *each* layer (the same "
     "`L2`-norm lens as TopoAct) and **compares** the per-layer graphs with a "
     "**Gromov–Wasserstein (GW)** distance — treating each Mapper graph as a "
     "metric-measure space (shortest-path distances + uniform node weights).  "
     "GW gives one number for how much the activation topology differs between "
     "two layers.\n\n"
     "**Methodology:** per-layer activations → per-layer Mapper graph → "
     "pairwise GW distance.\n\n"
     "**Expected result:** GW quantifies how the activation topology reshapes "
     "across the network, broadly shrinking toward the class-separated late "
     "layers.\n"),
    ("markdown", paper_info(
        arxiv=None,
        venue="TopoInVis 2023 (co-located with IEEE VIS)",
        datasets="CIFAR-10 activations.",
        models="Standard CNN classifiers; activations across layers.",
        headline="GW distance between layer-wise Mapper graphs quantifies how "
                 "activation topology changes with depth.",
        github="https://github.com/tdavislab/mapper-compare",
    )),
    ("markdown",
     "**This notebook runs on real activations.**  A small CNN is trained on "
     "the local CIFAR-10 cache; we Mapper *each* layer and compare with GW.\n"),
    ("code", REAL_IMPORTS),
    ("markdown", "## 1. Train a CNN, take per-layer activations\n"),
    ("code",
     "X_tr, y_tr = load_cifar10('train', n_samples=2000)\n"
     "X_te, y_te = load_cifar10('test',  n_samples=400)\n"
     "model = CNN_FCN(in_channels=3, num_classes=10, width=16, feat=64, hidden=32)\n"
     "train_briefly(model, X_tr, y_tr, X_te, y_te, epochs=6,\n"
     "              aspects=['activations'], device=DEVICE, verbose=False)\n"
     "snap = extract_snapshot(model.to('cpu').eval(), X_te[:300],\n"
     "                        aspects=['activations'], layer_selection='linear_and_conv')\n"
     "layers = [a.reshape(a.shape[0], -1) for a in snap.all_activation_matrices()]\n"
     "print('per-layer activation shapes:', [a.shape for a in layers])\n"),
    ("markdown",
     "## 2. One Mapper graph per layer\n\n"
     "We fit the preset to **each layer separately** — feeding the whole list "
     "at once would concatenate the layers into a single graph, leaving nothing "
     "to compare.\n"),
    ("code",
     'pipe = TDAPipeline.from_paper("zhou2023",\n'
     "                            tool_kwargs__n_intervals=14,\n"
     "                            tool_kwargs__overlap_frac=0.3)\n"
     "results = [pipe.fit(layer) for layer in layers]   # one Mapper graph per layer\n"
     "import networkx as nx\n"
     "fig, axes = plt.subplots(1, len(results), figsize=(4*len(results), 4))\n"
     "for ax, (i, r) in zip(np.atleast_1d(axes), enumerate(results)):\n"
     "    r.plot('graph', title=f'layer {i} ({r.mapper.graph.number_of_nodes()} nodes)', ax=ax)\n"
     "plt.tight_layout(); plt.show()\n"),
    ("markdown",
     "## 3. Gromov–Wasserstein distances between the layers' Mapper graphs\n\n"
     "The paper's tool: a GW distance between every pair of per-layer Mapper "
     "graphs (requires POT, `pip install pot`).\n"),
    ("code",
     "from tanc.topo_tools import mapper_gw_distance\n"
     "L = len(results)\n"
     "GW = np.zeros((L, L))\n"
     "for i in range(L):\n"
     "    for j in range(i + 1, L):\n"
     "        GW[i, j] = GW[j, i] = mapper_gw_distance(results[i].mapper.graph,\n"
     "                                                 results[j].mapper.graph)\n"
     "print('consecutive-layer GW:',\n"
     "      [f'L{i}->L{i+1}: {GW[i, i+1]:.1f}' for i in range(L - 1)])\n"
     "\n"
     "plt.figure(figsize=(5, 4.2))\n"
     "plt.imshow(GW, cmap='viridis')\n"
     "plt.colorbar(label='Gromov-Wasserstein distance')\n"
     "plt.xticks(range(L), [f'L{i}' for i in range(L)])\n"
     "plt.yticks(range(L), [f'L{i}' for i in range(L)])\n"
     "plt.title(\"GW distance between layers' Mapper graphs\")\n"
     "plt.tight_layout(); plt.show()\n"),
    ("markdown",
     "## Notes & gaps\n\n"
     "* Reproduced the paper's central tool on real CIFAR-10 activations: one "
     "Mapper graph per layer, compared with the **Gromov–Wasserstein** distance "
     "(`mapper_gw_distance`, via POT) — the full pairwise GW matrix is shown "
     "above.\n"
     "* The exact 'monotonic decrease with depth' is scale-dependent (small "
     "per-layer graphs are noisy and sometimes disconnected); train wider / use "
     "more samples and the paper's `Mapper(40, 0.2)` cover to sharpen it.\n"),
])


# ─────────────────────────────────────────────────────────────────────────────
# 12. gabrielsson2019 — uses node_exemplars too
# ─────────────────────────────────────────────────────────────────────────────

write("gabrielsson2019", [
    ("markdown",
     "# gabrielsson2019 — Gabrielsson & Carlsson (2019) Topology of NNs\n\n"
     "Mapper on the **real 3×3 conv kernels** harvested from trained CNNs.  "
     "PCA lens + VNE metric expose the edge / on-off / Gabor-like filter "
     "clusters.\n"),
    ("markdown",
     "## What the paper does\n\n"
     "Pools the **3×3 convolutional kernels** from many trained CNNs into one "
     "point cloud (each kernel is a 9-vector), preprocesses them (mean-centre "
     "to drop the DC term, keep the high-contrast ones, project to the unit "
     "sphere), and studies the cloud's shape with **Mapper** — a **PCA** lens "
     "and a variance-normalised Euclidean (VNE) metric.\n\n"
     "**Methodology:** trained conv kernels → mean-centre + contrast-select + "
     "normalise → VNE metric + PCA lens → Mapper graph of kernel space.\n\n"
     "**Expected result:** learned kernels concentrate on a **Klein-bottle-"
     "like** surface — primary/secondary circles of oriented edge- and "
     "blob-detectors recur across architectures (their Fig. 4).\n\n"
     "> **Scale caveat.** The clean Klein bottle needs *many well-trained* "
     "CNNs (the paper uses >1000).  Here we pool the kernels of a handful of "
     "briefly-trained nets, so the Mapper graph won't be a textbook circle — "
     "but the point is that the methodology runs on **real learned filters**, "
     "and the per-node exemplars below show they are genuinely structured "
     "oriented filters, not noise.\n"),
    ("markdown", paper_info(
        arxiv="https://arxiv.org/abs/1810.03234",
        venue="ICMLA 2019",
        datasets="MNIST, CIFAR-10, SVHN, ImageNet — over a thousand CNNs "
                 "trained per dataset to study filter-shape clusters.",
        models="Variety of CNN architectures (LeNet-style, VGG variants, "
               "and pretrained ImageNet networks).",
        headline="3x3 conv kernels cluster on a Klein-bottle-like surface "
                 "regardless of the network architecture (their Fig. 4).",
        github=None,
    )),
    ("markdown",
     "**This notebook runs on real data.**  A few small VGG-like CNNs are "
     "trained briefly on the local CIFAR-10 cache and their 3×3 conv kernels "
     "are harvested.\n"),
    ("code", REAL_IMPORTS),
    ("markdown", "## 1. Train a few CNNs and harvest their 3×3 conv kernels\n"),
    ("code",
     "N_MODELS = 4\n"
     "X_tr, y_tr = load_cifar10('train', n_samples=2000)\n"
     "X_te, y_te = load_cifar10('test',  n_samples=400)\n"
     "\n"
     "def harvest_kernels(model):\n"
     "    '''All 3x3 conv kernels of a model, as (n_kernels, 9).'''\n"
     "    snap = extract_snapshot(model.to('cpu').eval(), X_te[:32],\n"
     "                            aspects=['weights'], layer_selection='all_conv')\n"
     "    mats = [np.asarray(w, float).reshape(-1, 9)         # (out*in, 9)\n"
     "            for w in snap.kernel_weights.values() if w.shape[-2:] == (3, 3)]\n"
     "    return np.concatenate(mats, axis=0)\n"
     "\n"
     "kernels = []\n"
     "for seed in range(N_MODELS):\n"
     "    torch.manual_seed(seed)\n"
     "    m = VGGLike(n_blocks=2, convs_per_block=2, base_width=32)\n"
     "    train_briefly(m, X_tr, y_tr, X_te, y_te, epochs=4, device=DEVICE, verbose=False)\n"
     "    kernels.append(harvest_kernels(m))\n"
     "kernels = np.concatenate(kernels, axis=0)\n"
     "print(f'harvested {kernels.shape[0]} real 3x3 kernels from {N_MODELS} CNNs')\n"),
    ("markdown",
     "## 2. Preprocess — mean-centre, keep high-contrast, unit-normalise\n\n"
     "The same recipe Carlsson uses for natural-image patches: remove each "
     "kernel's mean (DC term), keep the top-contrast kernels (the rest are "
     "near-zero noise), and project onto the unit sphere so *shape*, not "
     "magnitude, drives the topology.\n"),
    ("code",
     "K = kernels - kernels.mean(axis=1, keepdims=True)        # drop DC term\n"
     "norm = np.linalg.norm(K, axis=1)\n"
     "K = K[norm >= np.quantile(norm, 0.70)]                   # top 30% contrast\n"
     "K = K / np.linalg.norm(K, axis=1, keepdims=True)         # onto the unit sphere\n"
     "\n"
     "rng = np.random.default_rng(0)\n"
     "if len(K) > 1500:                                        # keep Mapper tractable\n"
     "    K = K[rng.choice(len(K), 1500, replace=False)]\n"
     "data = K\n"
     "print(f'kernel point cloud for Mapper: {data.shape}')\n"),
    ("markdown",
     "## 3. Mapper on the kernel cloud\n\n"
     "A slightly coarser cover than the paper's `Mapper(30, 3)` keeps the "
     "small-scale graph readable.\n"),
    ("code",
     'pipe = TDAPipeline.from_paper("gabrielsson2019",\n'
     "                            tool_kwargs__n_intervals=12,\n"
     "                            tool_kwargs__overlap_frac=0.4)\n"
     "pipe.explain()\n"),
    ("code",
     "result, fig = pipe.reproduce(data)\n"
     "result.describe()\n"),
    ("markdown", "## 4. The Mapper graph of kernel space\n"),
    ("code",
     "import networkx as nx\n"
     "fig = result.plot('graph', title='Mapper of real 3x3 conv kernels')\n"
     "g = result.mapper.graph\n"
     "print(f'nodes={g.number_of_nodes()}  edges={g.number_of_edges()}  '\n"
     "      f'components={nx.number_connected_components(g)}')\n"),
    ("markdown",
     "## 5. Exemplar kernels per node — are they structured filters?\n\n"
     "Each Mapper node groups similar kernels; rendering one exemplar per node "
     "as a 3×3 image is the real test that the methodology found *structure* — "
     "you should see oriented edge / on-off / centre-surround filters, not "
     "random speckle.\n"),
    ("code",
     "from tanc.topo_tools import node_exemplars\n"
     "ex = node_exemplars(result, data=data, k=1, rank_by='centroid')\n"
     "nodes = list(ex)[:24]\n"
     "fig, axes = plt.subplots(4, 6, figsize=(9, 6))\n"
     "for ax, nid in zip(axes.ravel(), nodes):\n"
     "    ker = data[ex[nid][0]].reshape(3, 3)\n"
     "    ax.imshow(ker, cmap='RdBu', vmin=-abs(ker).max(), vmax=abs(ker).max())\n"
     "    ax.set_title(f'node {nid}', fontsize=7); ax.axis('off')\n"
     "for ax in axes.ravel()[len(nodes):]:\n"
     "    ax.axis('off')\n"
     "fig.suptitle('One exemplar 3x3 kernel per Mapper node (RdBu)')\n"
     "fig.tight_layout(); plt.show()\n"),
    ("markdown",
     "## Notes & gaps\n\n"
     "* Reproduced on **real learned kernels** (not synthetic): harvest → "
     "contrast-normalise → VNE + PCA Mapper, plus exemplar-kernel rendering.\n"
     "* The textbook Klein bottle needs *many well-trained* CNNs; pool more "
     "models (`N_MODELS`), train longer, and use the paper's cover "
     "(`n_intervals=30, overlap_frac=0.667`) to push toward it.\n"
     "* `node_exemplars(...)` returns indices into `data`, so you can render "
     "exemplars in any modality (here, 3×3 kernel images).\n"),
])


# ─────────────────────────────────────────────────────────────────────────────
# 13. gabella2021
# ─────────────────────────────────────────────────────────────────────────────

write("gabella2021", [
    ("markdown",
     "# gabella2021 — Gabella (2021) Topology of Learning\n\n"
     "Mapper on weight trajectories: filter = L2 norm of the parameter "
     "vector.  Several runs from one initialisation give a **tree-like** "
     "learning topology.\n"),
    ("markdown",
     "## What the paper does\n\n"
     "Views *training itself* as a curve through weight space and visualises "
     "it with **Mapper**. The parameter vectors `{W_i}` snapshotted along "
     "training are the point cloud; the lens is the **`L2` norm** of the "
     "parameter vector (a proxy for 'how far from initialisation'); Mapper "
     "then summarises the optimisation path as a graph.\n\n"
     "**Methodology:** weight trajectories `{W_i}` → `L2`-norm lens → Mapper "
     "graph of the learning path.\n\n"
     "**Expected result:** the Mapper graph is **tree-like**, branching out "
     "from the initialisation origin as training explores weight space "
     "(their Fig. 5).\n\n"
     "> **Why several runs.** A *single* SGD run is a 1-D path and its `L2` "
     "norm grows roughly monotonically, so Mapper returns one **curve**, not a "
     "tree.  Branches appear when several trajectories **share an origin** — so "
     "we train a few models from the *same initialisation* with different "
     "batch orders and Mapper them together.  (Explicit extraction on purpose: "
     "Mapper needs the trajectories as one `(T, D)` point cloud, which the "
     "activations-oriented model-first adapter would not build.)\n"),
    ("markdown", paper_info(
        arxiv="https://arxiv.org/abs/1902.08160",
        venue="EPFL technical report / preprint",
        datasets="MNIST.",
        models="Small fully-connected ReLU networks trained from scratch.",
        headline="L2-norm-filter Mapper of the weight trajectory shows tree "
                 "topology branching from the initialisation origin (Fig. 5).",
        github="https://github.com/maximevictor/topo-learning",
    )),
    ("markdown",
     "**This notebook runs on real data.**  A few tiny MLPs are trained "
     "briefly on the local MNIST cache; the qualitative tree the paper "
     "reports should appear.\n"),
    ("code", REAL_IMPORTS),
    ("markdown", "## 1. Train several runs from one initialisation\n"),
    ("code",
     "import copy\n"
     "from tanc.model_extractor import TrainingExtractor\n"
     "from torch.utils.data import DataLoader, TensorDataset\n"
     "\n"
     "N_RUNS = 6                       # branches that share a common root\n"
     "X_tr, y_tr = load_mnist('train', n_samples=2000)\n"
     "X_te, y_te = load_mnist('test',  n_samples=400)\n"
     "\n"
     "train_loader = DataLoader(\n"
     "    TensorDataset(torch.as_tensor(X_tr, dtype=torch.float32),\n"
     "                  torch.as_tensor(y_tr, dtype=torch.long)),\n"
     "    batch_size=64, shuffle=True)\n"
     "val_loader = DataLoader(\n"
     "    TensorDataset(torch.as_tensor(X_te, dtype=torch.float32),\n"
     "                  torch.as_tensor(y_te, dtype=torch.long)),\n"
     "    batch_size=64)\n"
     "\n"
     "# One fixed initialisation shared by every run.\n"
     "base = SmallMLP(input_dim=784, hidden_dims=[32, 16], output_dim=10)\n"
     "\n"
     "def run_trajectory(seed):\n"
     "    '''Train a copy of `base` (same init, different batch order) and\n"
     "    return its weight trajectory flattened to (T, D).'''\n"
     "    model = copy.deepcopy(base)\n"
     "    torch.manual_seed(seed)\n"
     "    ext = TrainingExtractor(\n"
     "        model=model, train_loader=train_loader,\n"
     "        criterion=torch.nn.CrossEntropyLoss(),\n"
     "        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),\n"
     "        val_loader=val_loader, extract_data=X_te[:64],\n"
     "        aspects=['weights'], snapshot_every=2,\n"
     "        snapshot_schedule='iteration', device=DEVICE, clarify=False,\n"
     "    )\n"
     "    view = ext.run(epochs=3, target_accuracy=None, verbose=False)\n"
     "    return np.stack([np.concatenate([w.ravel() for w in snap_w])\n"
     "                     for snap_w in view.weight_trajectory()])\n"
     "\n"
     "trajectories = [run_trajectory(s) for s in range(N_RUNS)]\n"
     "print(f'{N_RUNS} runs x {len(trajectories[0])} snapshots each')\n"),
    ("markdown", "## 2. Concatenate the trajectories into one point cloud\n"),
    ("code",
     "# Each run is one branch; they share the common initialisation as root.\n"
     "data   = np.concatenate(trajectories, axis=0)            # (sum T, D)\n"
     "run_id = np.concatenate([np.full(len(t), i)\n"
     "                         for i, t in enumerate(trajectories)])\n"
     "print(f'weight-trajectory point cloud: {data.shape}')\n"),
    ("markdown",
     "## 3. Reproduce via `TDAPipeline.from_paper`\n\n"
     "A slightly finer Mapper cover (`n_intervals=15`, `overlap_frac=0.4`) "
     "resolves the branch points where the runs diverge.\n"),
    ("code",
     'pipe = TDAPipeline.from_paper("gabella2021",\n'
     "                            tool_kwargs__n_intervals=15,\n"
     "                            tool_kwargs__overlap_frac=0.4)\n"
     "pipe.explain()\n"),
    ("code",
     "result, fig = pipe.reproduce(data)\n"
     "result.describe()\n"),
    ("markdown",
     "## 4. The learning tree — coloured by run\n\n"
     "Nodes are coloured by which run their snapshots come from; the shared "
     "early-training nodes form the root and the runs fan out into branches.\n"),
    ("code",
     "import networkx as nx\n"
     "fig = result.plot('graph', color_by=run_id, layout='kamada_kawai',\n"
     "                  title=f'{N_RUNS} runs from one init (coloured by run)')\n"
     "g = result.mapper.graph\n"
     "branch = sum(1 for _, d in g.degree() if d > 2)\n"
     "print(f'nodes={g.number_of_nodes()}  edges={g.number_of_edges()}  '\n"
     "      f'branch nodes (deg>2)={branch}  '\n"
     "      f'components={nx.number_connected_components(g)}')\n"),
    ("markdown",
     "## Notes & gaps\n\n"
     "* Reproduced: the tree-like learning topology from several real weight "
     "trajectories sharing one initialisation.  A *single* run instead gives "
     "one curve (set `N_RUNS=1` to see it).\n"
     "* More branches: raise `N_RUNS`, or nudge `tool_kwargs__n_intervals` / "
     "`overlap_frac`.  Colour by training step instead with "
     "`color_by=np.arange(len(data))`.\n"
     "* Secondary 2-D PCA *surface* analysis: re-run with "
     "`from_paper('gabella2021', tool_kwargs__filter_fn='pca', "
     "tool_kwargs__n_components=3)`.\n"),
])


# ─────────────────────────────────────────────────────────────────────────────
# 14. ruppik2025
# ─────────────────────────────────────────────────────────────────────────────

write("ruppik2025", [
    ("markdown",
     "# ruppik2025 — Ruppik et al. (2025) Less is More (local ID)\n\n"
     "Local 2NN intrinsic dimension, **per layer**, of a trained network — a "
     "stand-in for the paper's language-model layers.\n"),
    ("markdown",
     "## What the paper does\n\n"
     "Estimates the **local intrinsic dimension (LID)** of a model's hidden "
     "representations with a **two-nearest-neighbour (2NN)** estimator applied "
     "*locally* (per neighbourhood), then tracks how LID changes across layers "
     "and training.\n\n"
     "**Methodology:** per-layer representations → local 2NN ID estimate → "
     "mean LID per layer.\n\n"
     "**Expected result:** in the paper, mean LID is an early-warning signal "
     "for **overfitting** and **grokking** (Figs. 2, 7).  Here, on a CNN, the "
     "robust qualitative effect is that LID **shrinks with depth** — early "
     "layers hold high-dimensional representations, later layers compress "
     "toward the low-dimensional class structure.\n"),
    ("markdown", paper_info(
        arxiv="https://arxiv.org/abs/2506.01034",
        venue="NeurIPS 2025",
        datasets="MultiWOZ, GoEmotions, modular-arithmetic grokking.",
        models="BERT, RoBERTa, small transformers.  Here: a small CNN on "
               "CIFAR-10 as a portable stand-in.",
        headline="Mean local intrinsic dimension predicts overfitting and "
                 "grokking phase transitions (Figs. 2 and 7).",
        github="https://github.com/aidos-lab/Topo_LLM_public",
        notes_url="Companion repo: https://github.com/aidos-lab/grokking-via-lid",
    )),
    ("markdown",
     "**This notebook runs on real activations.**  A small CNN is trained "
     "briefly on the local CIFAR-10 cache; we estimate local ID at each layer.\n"),
    ("code", REAL_IMPORTS),
    ("markdown", "## 1. Train a small CNN\n"),
    ("code",
     "X_tr, y_tr = load_cifar10('train', n_samples=2000)\n"
     "X_te, y_te = load_cifar10('test',  n_samples=400)\n"
     "model = CNN_FCN(in_channels=3, num_classes=10, width=16, feat=64, hidden=32)\n"
     "train_briefly(model, X_tr, y_tr, X_te, y_te, epochs=5,\n"
     "              aspects=['activations'], device=DEVICE, verbose=False)\n"
     "print('trained.')\n"),
    ("markdown",
     "## 2. Local intrinsic dimension per layer — model-first\n\n"
     "The preset does the work: `fit_model` extracts each layer's activations "
     "and runs the local-ID estimator per layer — no manual extraction.\n"),
    ("code",
     'pipe = TDAPipeline.from_paper("ruppik2025")\n'
     "result = pipe.fit_model(model, X_te[:200],\n"
     "                        representation='activations',\n"
     "                        layer_selection='linear_and_conv')\n"
     "lid = result.dimension_result['id_estimates']\n"
     "for i, d in enumerate(lid):\n"
     "    print(f'  layer {i}: local ID = {d:.2f}')\n"),
    ("markdown", "## 3. Local ID vs depth\n"),
    ("code",
     "plt.figure(figsize=(6, 4))\n"
     "plt.plot(range(len(lid)), lid, 'o-')\n"
     "plt.xlabel('layer depth (conv → fc)'); plt.ylabel('local intrinsic dimension')\n"
     "plt.xticks(range(len(lid))); plt.grid(alpha=0.3)\n"
     "plt.title('Local ID shrinks with depth (CNN on CIFAR-10)')\n"
     "plt.tight_layout(); plt.show()\n"),
    ("markdown",
     "## Notes & gaps\n\n"
     "* Reproduced on **real CNN activations**: local 2NN ID, computed per "
     "layer, falls from a high-dimensional early representation toward the "
     "low-dimensional class structure deeper in the net.\n"
     "* The paper's overfitting/grokking signals are LLM-specific; "
     "`Topo_LLM_public` has the full extraction.  The toolkit's estimator "
     "(now returning a calibrated dimension) handles any per-layer "
     "activations.\n"),
])


# ─────────────────────────────────────────────────────────────────────────────
# 15. ong2026
# ─────────────────────────────────────────────────────────────────────────────

write("ong2026", make(
    preset="ong2026",
    title="Ong et al. (2026) — Universal 2NN Estimator",
    summary=(
        "Global mean(log(log mu)) over all 2NN ratios.  Universal "
        "scale-free invariant."
    ),
    info_md=paper_info(
        arxiv="https://arxiv.org/abs/2603.10493",
        venue="Preprint (Queen Mary / Oxford collaboration)",
        datasets="Synthetic point clouds on Swiss roll, sphere, torus, "
                 "and other test manifolds of known dimension.",
        models="No NN involved — the paper is about the estimator itself; "
               "test data is sampled directly from manifolds.",
        headline="The mean(log(log mu)) estimator converges to the true "
                 "intrinsic dimension on every test distribution without "
                 "tuning (their Fig. 4 benchmark table).",
        github=None,
    ),
    methodology=(
        "## What the paper does\n\n"
        "Proposes a **parameter-free** intrinsic-dimension estimator. For every "
        "point it forms the ratio `mu = r2 / r1` of its two nearest-neighbour "
        "distances; under a uniform-density assumption `mu` follows a known "
        "distribution, and the global estimate is simply `mean(log(log mu))` "
        "over all points — a scale-free invariant with no hyperparameters to "
        "tune.\n\n"
        "**Methodology:** 2NN distance ratios `mu` → `mean(log(log mu))` → "
        "intrinsic dimension.\n\n"
        "**Expected result:** the estimator recovers the true dimension across "
        "many test manifolds (Swiss roll, sphere, torus, …) without tuning "
        "(their Fig. 4 benchmark). No neural network is involved — it's about "
        "the estimator itself.\n"
    ),
    data_cell=(
        "# Point clouds sampled from manifolds of KNOWN intrinsic dimension,\n"
        "# isometrically embedded in a higher-dimensional ambient space.\n"
        "rng = np.random.default_rng(0)\n"
        "\n"
        "def embed(P, D):\n"
        "    '''Rotate a d-D point set into D-D (random orthonormal embedding).'''\n"
        "    Q = np.linalg.qr(rng.standard_normal((D, P.shape[1])))[0]\n"
        "    return (P @ Q.T).astype(np.float32)\n"
        "\n"
        "manifolds = {\n"
        "    'line (d=1)':  (embed(rng.uniform(0, 1, (800, 1)), 10), 1),\n"
        "    'plane (d=2)': (embed(rng.uniform(0, 1, (800, 2)), 10), 2),\n"
        "    'cube (d=3)':  (embed(rng.uniform(0, 1, (800, 3)), 10), 3),\n"
        "    'cube (d=5)':  (embed(rng.uniform(0, 1, (800, 5)), 10), 5),\n"
        "}\n"
        "# Swiss roll: a 2-D sheet curled into 3-D.\n"
        "t = 1.5*np.pi*(1 + 2*rng.uniform(0, 1, 800)); h = 21*rng.uniform(0, 1, 800)\n"
        "manifolds['swiss roll (d=2)'] = (np.c_[t*np.cos(t), h, t*np.sin(t)].astype(np.float32), 2)\n"
        "\n"
        "data       = [v[0] for v in manifolds.values()]\n"
        "true_dims  = [v[1] for v in manifolds.values()]\n"
        "print('manifolds:', list(manifolds))\n"
    ),
    plot_cell=(
        "# The estimator returns one intrinsic dimension per cloud.\n"
        "est = result.dimension_result['id_estimates']\n"
        "for name, e, td in zip(manifolds, est, true_dims):\n"
        "    print(f'  {name:18s} estimated d = {e:.2f}  (true {td})')\n"
        "\n"
        "names = list(manifolds); x = np.arange(len(names))\n"
        "plt.figure(figsize=(8, 4))\n"
        "plt.bar(x - 0.2, est, 0.4, label='estimated')\n"
        "plt.bar(x + 0.2, true_dims, 0.4, label='true')\n"
        "plt.xticks(x, names, rotation=20, ha='right'); plt.ylabel('intrinsic dimension')\n"
        "plt.legend(); plt.title('Ong 2NN estimator recovers the intrinsic dimension')\n"
        "plt.tight_layout(); plt.show()\n"
    ),
    notes=(
        "* Reproduced on **generated manifolds of known dimension**: the "
        "parameter-free 2NN estimator recovers d≈1, 2, 3 and the swiss roll's "
        "d≈2 with no tuning.\n"
        "* Higher dimensions are mildly under-estimated with finite samples "
        "(a known 2NN property) — raise the sample count to tighten them.\n"
    ),
))


# ─────────────────────────────────────────────────────────────────────────────
# 16. birdal2021
# ─────────────────────────────────────────────────────────────────────────────

write("birdal2021", make_real(
    preset="birdal2021",
    title="Birdal et al. (2021) — Intrinsic Dim, PH & Generalization",
    summary=(
        "VR PH on a weight trajectory.  Slope of log E_alpha vs log n is "
        "the PH dim and correlates with generalisation."
    ),
    info_md=paper_info(
        arxiv="https://arxiv.org/abs/2111.13171",
        venue="NeurIPS 2021",
        datasets="CIFAR-10.",
        models="ResNet-18, AlexNet, VGG.  Here: SmallCNN — same trajectory "
               "methodology, smaller scale.",
        headline="PH-dimension of the weight trajectory correlates with the "
                 "generalisation error (Pearson rho ~ 0.7).",
        github="https://github.com/tolgabirdal/PHDimGeneralization",
    ),
    methodology=(
        "## What the paper does\n\n"
        "Measures the **fractal (persistent-homology) dimension** of the SGD "
        "**weight trajectory** and links it to generalisation. The iterates "
        "`{W_i}` from (the late phase of) training form a point cloud; for "
        "subsets of growing size `n` the paper sums the `H0` bar lifetimes "
        "`E_α(n) = Σ (death − birth)^α`, and the slope `b` of `log E_α` vs "
        "`log n` gives the PH dimension `dim = b / (1 − b)`.\n\n"
        "**Methodology:** weight trajectory `{W_i}` → Euclidean distances → "
        "`H0` PH on growing subsets → slope of `log E_α` vs `log n` → PH "
        "dimension.\n\n"
        "**Expected result:** the PH dimension correlates with the "
        "generalisation error across many trained models (Pearson `ρ ≈ 0.7`) — "
        "lower trajectory dimension ↔ better generalisation.\n\n"
        "> **Needs a long trajectory.** The scaling fit needs subsets of size "
        "up to a few hundred, so we snapshot **every iteration over ~30 epochs** "
        "(`T ≈ 700` points). A short trajectory (`T < 50`) leaves the default "
        "subset sizes empty — the toolkit now falls back to adaptive sizes and "
        "warns, but a long trajectory is what makes the estimate meaningful.\n"
    ),
    setup_code=(
        "from tanc.model_extractor import TrainingExtractor\n"
        "from torch.utils.data import DataLoader, TensorDataset\n"
        "\n"
        "X_tr, y_tr = load_cifar10('train', n_samples=1500)\n"
        "X_te, y_te = load_cifar10('test',  n_samples=400)\n"
        "model = SmallCNN(in_channels=3, num_classes=10, width=8)\n"
        "\n"
        "train_loader = DataLoader(\n"
        "    TensorDataset(torch.as_tensor(X_tr, dtype=torch.float32),\n"
        "                  torch.as_tensor(y_tr, dtype=torch.long)),\n"
        "    batch_size=64, shuffle=True)\n"
        "val_loader = DataLoader(\n"
        "    TensorDataset(torch.as_tensor(X_te, dtype=torch.float32),\n"
        "                  torch.as_tensor(y_te, dtype=torch.long)),\n"
        "    batch_size=64)\n"
        "\n"
        "# Snapshot EVERY iteration over many epochs so the trajectory is long\n"
        "# enough for the PH-dimension scaling fit (subset sizes up to 500).\n"
        "ext = TrainingExtractor(\n"
        "    model=model,\n"
        "    train_loader=train_loader,\n"
        "    criterion=torch.nn.CrossEntropyLoss(),\n"
        "    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),\n"
        "    val_loader=val_loader, extract_data=X_te[:64],\n"
        "    aspects=['weights'],\n"
        "    snapshot_every=1, snapshot_schedule='iteration',\n"
        "    device=DEVICE, clarify=False,\n"
        ")\n"
        "view = ext.run(epochs=30, target_accuracy=None, verbose=False)\n"
        "print(f'trajectory length: {len(view)} snapshots')\n"
    ),
    reproduce_input="view",
    plot_cell=(
        'fig = result.plot("ph_scaling")\n'
        "print('PH dimension:', result.dimension)\n"
    ),
    notes=(
        "* Reproduced on a real Euclidean weight trajectory (`T ≈ 700` points), "
        "fed model-first — the pipeline adapter pulls the trajectory off the "
        "`TrainingView`.\n"
        "* The generalisation-correlation (`ρ ≈ 0.7`) needs *many* trained "
        "models, each contributing one PH dimension; see the reference repo for "
        "the full sweep.\n"
    ),
))


# ─────────────────────────────────────────────────────────────────────────────
# 17. dupuis2023
# ─────────────────────────────────────────────────────────────────────────────

write("dupuis2023", make_real(
    preset="dupuis2023",
    title="Dupuis et al. (2023) — Data-Dependent Fractal Dimensions",
    summary=(
        "Same scaling law as Birdal but with a loss-difference distance "
        "metric — more sensitive to flat minima."
    ),
    info_md=paper_info(
        arxiv="https://arxiv.org/abs/2302.02766",
        venue="ICML 2023",
        datasets="CIFAR-10 (+ CIFAR-100 in the supplement).",
        models="ResNet / AlexNet / VGG.  Here: SmallCNN — same loss-"
               "difference methodology, smaller scale.",
        headline="Loss-difference PH dim gives tighter generalisation "
                 "bounds than the Euclidean PH dim, especially in the "
                 "flat-minimum regime.",
        github="https://github.com/benjiDupuis/data_dependent_dimensions",
    ),
    methodology=(
        "## What the paper does\n\n"
        "Refines Birdal's idea with a **data-dependent** metric. Instead of "
        "Euclidean distance between weight iterates, the trajectory points are "
        "compared by their **loss difference** `|L(W_i) − L(W_j)|`, and the same "
        "`H0`-lifetime scaling law gives a PH dimension. This makes the "
        "dimension sensitive to the *flatness* of the minimum, not just the "
        "geometry of the path.\n\n"
        "**Methodology:** weight trajectory + per-step losses → loss-difference "
        "distance → `H0` PH scaling → data-dependent PH dimension.\n\n"
        "**Expected result:** the loss-difference PH dimension yields **tighter** "
        "generalisation bounds than the Euclidean version, especially in the "
        "flat-minimum regime.\n\n"
        "> Like Birdal, this needs a **long** trajectory for the scaling fit, so "
        "we snapshot every iteration over ~30 epochs.\n"
    ),
    setup_code=(
        "from tanc.model_extractor import TrainingExtractor\n"
        "from torch.utils.data import DataLoader, TensorDataset\n"
        "\n"
        "X_tr, y_tr = load_cifar10('train', n_samples=1500)\n"
        "X_te, y_te = load_cifar10('test',  n_samples=400)\n"
        "model = SmallCNN(in_channels=3, num_classes=10, width=8)\n"
        "\n"
        "train_loader = DataLoader(\n"
        "    TensorDataset(torch.as_tensor(X_tr, dtype=torch.float32),\n"
        "                  torch.as_tensor(y_tr, dtype=torch.long)),\n"
        "    batch_size=64, shuffle=True)\n"
        "val_loader = DataLoader(\n"
        "    TensorDataset(torch.as_tensor(X_te, dtype=torch.float32),\n"
        "                  torch.as_tensor(y_te, dtype=torch.long)),\n"
        "    batch_size=64)\n"
        "\n"
        "# Long per-iteration trajectory for the loss-difference scaling fit.\n"
        "ext = TrainingExtractor(\n"
        "    model=model,\n"
        "    train_loader=train_loader,\n"
        "    criterion=torch.nn.CrossEntropyLoss(),\n"
        "    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),\n"
        "    val_loader=val_loader, extract_data=X_te[:64],\n"
        "    aspects=['weights'],\n"
        "    snapshot_every=1, snapshot_schedule='iteration',\n"
        "    device=DEVICE, clarify=False,\n"
        ")\n"
        "view = ext.run(epochs=30, target_accuracy=None, verbose=False)\n"
        "print(f'trajectory length: {len(view)} snapshots; '\n"
        "      f'loss[0]={view.losses()[0]:.4f} -> loss[-1]={view.losses()[-1]:.4f}')\n"
    ),
    reproduce_input="view",
    plot_cell=(
        'fig = result.plot("ph_scaling")\n'
        "print('PH dimension (loss-difference):', result.dimension)\n"
    ),
    notes=(
        "* Reproduced on a real loss-difference trajectory, fed **model-first**: "
        "passing the `TrainingView` lets the pipeline extract both the weight "
        "trajectory and the per-step losses for the `ph_loss` metric with no "
        "extra wiring.\n"
    ),
))


# ─────────────────────────────────────────────────────────────────────────────
# 18. andreeva2024
# ─────────────────────────────────────────────────────────────────────────────

write("andreeva2024", make_real(
    preset="andreeva2024",
    title="Andreeva et al. (2024) — Magnitude & Generalisation",
    summary=(
        "Magnitude Mag(tW) on the weight trajectory; magnitude dimension "
        "= slope of log Mag vs log t."
    ),
    info_md=paper_info(
        arxiv="https://arxiv.org/abs/2305.05611",
        venue="TAG-ML 2023 (PMLR 221)",
        datasets="CIFAR-10 and CIFAR-100.",
        models="ResNet-18 / AlexNet / VGG.  Here: SmallCNN on CIFAR-10.",
        headline="Magnitude dimension correlates with generalisation: "
                 "lower magnitude dim => better generalisation (Fig. 3).",
        github="https://github.com/aidos-lab/magnipy",
        notes_url="Companion library `magnipy` provides standalone "
                  "magnitude computations.",
    ),
    setup_code=(
        "from tanc.model_extractor import TrainingExtractor\n"
        "from torch.utils.data import DataLoader, TensorDataset\n"
        "\n"
        "X_tr, y_tr = load_cifar10('train', n_samples=1500)\n"
        "X_te, y_te = load_cifar10('test',  n_samples=400)\n"
        "model = SmallCNN(in_channels=3, num_classes=10, width=8)\n"
        "\n"
        "train_loader = DataLoader(\n"
        "    TensorDataset(torch.as_tensor(X_tr, dtype=torch.float32),\n"
        "                  torch.as_tensor(y_tr, dtype=torch.long)),\n"
        "    batch_size=64, shuffle=True)\n"
        "val_loader = DataLoader(\n"
        "    TensorDataset(torch.as_tensor(X_te, dtype=torch.float32),\n"
        "                  torch.as_tensor(y_te, dtype=torch.long)),\n"
        "    batch_size=64)\n"
        "\n"
        "ext = TrainingExtractor(\n"
        "    model=model,\n"
        "    train_loader=train_loader,\n"
        "    criterion=torch.nn.CrossEntropyLoss(),\n"
        "    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),\n"
        "    val_loader=val_loader, extract_data=X_te[:64],\n"
        "    aspects=['weights'],\n"
        "    snapshot_every=2, snapshot_schedule='iteration', clarify=False,\n"
        ")\n"
        "view = ext.run(epochs=2, target_accuracy=None, verbose=False)\n"
        "print(f'trajectory length: {len(view)} snapshots')\n"
    ),
    data_code=(
        "data = view.weight_trajectory()\n"
    ),
    plot_cell=(
        'fig = result.plot("magnitude_scaling")\n'
        "print('Magnitude dimension:', result.dimension)\n"
    ),
    notes=(
        "* Reproduced on a real weight trajectory.\n"
        "* For the per-epoch view, combine "
        "`plot_magnitude_dimension_over_training(view)` with a sliding "
        "window.\n"
    ),
))


print("\nDone." + (f" Regenerated only: {', '.join(sorted(ONLY))}." if ONLY
                    else " Generated all notebooks.") + f"\n  in {HERE}")