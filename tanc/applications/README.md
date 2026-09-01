# applications

Task-level applications built on top of the core `graph_builder` → `topo_tools`
pipeline. These go one step beyond producing a topological summary — they *use*
it for a downstream task.

## Persistent-homology network pruning — `pruning`

Reproduces Watanabe & Yamana, *Deep Neural Network Pruning Using Persistent
Homology* (PHPM, AIKE 2020). PHPM prunes a fully-connected sub-network by
**keeping the weight-edges that carry its strongest H1 loops** and zeroing the
rest, using the representative cycles of the directed clique complex
(`tanc.topo_tools.compute_watanabe_ph`).

```python
import numpy as np
from tanc import TDAPipeline
from tanc.applications import (
    phpm_pruning_mask, gmp_pruning_mask, apply_masks, prune_diagnostics)

# weight_matrices = the FCN head, a list of consecutive (N_in, N_out) matrices
ph = TDAPipeline.from_paper("watanabe2021",
                            tool_kwargs__relevance_mode="absolute").fit(weight_matrices)

phpm = phpm_pruning_mask(weight_matrices, ph, pruning_ratio=0.9)   # keep-masks (True=keep)
gmp  = gmp_pruning_mask(weight_matrices, pruning_ratio=0.9, per_layer=True)  # baseline

pruned = apply_masks(weight_matrices, phpm)        # zero the un-kept weights
prune_diagnostics(phpm)                            # kept/total, dead target neurons
```

| Function | Description |
|---|---|
| `phpm_pruning_mask(weights, ph_result, ratio)` | PHPM keep-masks: walk H1 classes by ascending `birth+death`, keep every real edge touched by each class's birth edge + death triangle until the target is met (magnitude fallback tops up). |
| `gmp_pruning_mask(weights, ratio, per_layer=False)` | Global magnitude pruning baseline. `per_layer=True` removes the same fraction within each matrix (robust variant). |
| `simplex_to_real_edges(simplex, rel_matrices, layer_sizes)` | Map a complex simplex (direct/induced edge or triangle) back to its real weight edges. |
| `apply_masks(weights, masks)` | Copies of the weights with un-kept entries zeroed. |
| `prune_diagnostics(masks)` | kept/total per layer, achieved ratio, dead target neurons. |
| `fcn_accuracy(forward, weights, masks, X, y)` | Framework-agnostic top-1 accuracy with masking applied (`forward(X, weights) -> logits`). |

The H1 diagram and PHPM masks are **bit-for-bit identical** to the reference
[`DNNtopology`](https://github.com/satoru-watanabe-aw/DNNtopology) implementation.
See `paper_reproduce/watanabe2021.ipynb` for the end-to-end demo (belt diagram +
PHPM-vs-GMP pruning curve).
