"""TopoResult and PersistenceResult — unified output containers for Module 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from tanc._serialization import SaveLoadMixin


@dataclass
class PersistenceResult:
    """Raw output of a persistence homology computation.

    Parameters
    ----------
    diagrams : dict[int, ndarray]
        Maps homology dimension ``d`` to an ``(n_bars, 2)`` array whose
        columns are ``[birth, death]``.  Infinite deaths have been
        replaced with the maximum finite death value.
    metadata : dict
        Provenance: ``runtime_seconds``, ``backend``, ``input_type``, etc.
    """

    diagrams: dict[int, np.ndarray]
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Mapper accessor
# ─────────────────────────────────────────────────────────────────────────────

class MapperView(NamedTuple):
    """Grouped accessor for the four mapper-related fields on a TopoResult.

    Lets users write ``result.mapper.graph`` instead of remembering each
    individual ``result.mapper_*`` attribute name.
    """

    graph: Any | None
    node_members: dict[int, list[int]] | None
    filter_values: np.ndarray | None
    stats: dict | None


# ─────────────────────────────────────────────────────────────────────────────
# Plot kinds registry — single source of truth for routing
# ─────────────────────────────────────────────────────────────────────────────

# Maps (tool, kind) → (required_attr, plotter_dotted_path, one_line_doc).
# - required_attr is the attribute on TopoResult that must be populated.
# - plotter_dotted_path returns (module, function_name); resolved lazily.
_PLOT_KINDS: dict[str, dict[str, dict[str, str]]] = {
    "ph": {
        "diagram": {
            "requires": "ph_result",
            "fn": "tanc.visualisation.representations.plot_persistence_diagram",
            "doc": "Birth-vs-death scatter of persistence pairs.",
        },
        "barcode": {
            "requires": "ph_result",
            "fn": "tanc.visualisation.representations.plot_barcode",
            "doc": "Horizontal bars from birth to death, sorted by birth.",
        },
        "betti_curve": {
            "requires": "ph_result",
            "fn": "tanc.visualisation.representations.plot_betti_curve",
            "doc": "Step function of Betti_d across the filtration range.",
        },
    },
    "mapper": {
        "graph": {
            "requires": "mapper_graph",
            "fn": "tanc.topo_tools.mapper_tool.plot_mapper_graph",
            "doc": "Node-link drawing of the Mapper graph.",
        },
        "ph_diagram": {
            "requires": "ph_result",
            "fn": "tanc.visualisation.representations.plot_persistence_diagram",
            "doc": "PH of the Mapper graph (requires compute_ph=True).",
        },
    },
    "dimension": {
        "ph_scaling": {
            "requires": "dimension_result",
            "fn": "tanc.topo_tools.dimension_tool.plot_ph_scaling",
            "doc": "Log-log lifetime-sum vs subset size with regression.",
        },
        "magnitude_scaling": {
            "requires": "dimension_result",
            "fn": "tanc.topo_tools.dimension_tool.plot_magnitude_scaling",
            "doc": "Log-log Mag(tX) vs t with fitted slope.",
        },
        "id_layers": {
            "requires": "dimension_result",
            "fn": "tanc.topo_tools.dimension_tool.plot_id_across_layers",
            "doc": "Intrinsic dimension per layer (line plot).",
        },
    },
}


def _resolve_plot_fn(dotted: str):
    import importlib
    mod_path, fn_name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod_path), fn_name)


def _save_figure(fig: Any, path: str) -> str:
    """Write *fig* (a matplotlib Figure or Axes) to *path*.

    Plotters in this toolkit return a ``Figure``, but we accept an ``Axes``
    too and resolve its parent figure, so ``save=`` is robust either way.
    """
    target = fig
    if not hasattr(target, "savefig"):
        target = getattr(target, "figure", None)
    if target is None or not hasattr(target, "savefig"):
        raise TypeError(
            "save= was given but the plot did not return a matplotlib Figure "
            f"(got {type(fig).__name__})."
        )
    from pathlib import Path
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    target.savefig(p, bbox_inches="tight")
    return str(p)


@dataclass
class TopoResult(SaveLoadMixin):
    """Unified output container for any topological summary.

    Parameters
    ----------
    tool : str
        Which Module 2 tool produced this result.
        ``"ph"`` | ``"mapper"`` | ``"dimension"`` | ``"combined"``
    ph_result : PersistenceResult or None
        Populated for ``tool="ph"`` and optionally for ``tool="mapper"``
        when ``compute_ph=True``.
    statistics : dict or None
        Topological statistics keyed ``"H{d}_{stat_name}"``.
    mapper_graph : networkx.Graph or None
        Populated for ``tool="mapper"``.
    mapper_node_members : dict[int, list[int]] or None
        Maps mapper-node index to list of original data-point indices.
    mapper_filter_values : (N, k) ndarray or None
        Filter values for each data point.
    mapper_graph_stats : dict or None
        Graph-level statistics (n_nodes, density, etc.).
    dimension_result : dict or None
        Raw dict returned by the chosen dimension estimator.
    config : dict
        Full record of all parameters used.
    paper_reference : str or None
        Citation string when created via a paper preset.
    """

    tool: str
    ph_result: PersistenceResult | None = None
    statistics: dict[str, Any] | None = None
    mapper_graph: Any | None = None
    mapper_node_members: dict[int, list[int]] | None = None
    mapper_filter_values: np.ndarray | None = None
    mapper_graph_stats: dict | None = None
    dimension_result: dict | None = None
    config: dict = field(default_factory=dict)
    paper_reference: str | None = None
    default_plot_kind: str | None = None

    # ------------------------------------------------------------------ #
    # Convenience accessors                                                #
    # ------------------------------------------------------------------ #

    @property
    def diagrams(self) -> dict[int, np.ndarray] | None:
        """Shortcut for ``self.ph_result.diagrams`` (or None)."""
        return self.ph_result.diagrams if self.ph_result is not None else None

    def diagram(self, dim: int = 0) -> np.ndarray:
        """Return the persistence diagram for one homology dimension.

        Convenience for ``self.ph_result.diagrams[dim]``.  Raises a clear
        error if no PH result is present or ``dim`` was not computed.
        """
        if self.diagrams is None:
            raise ValueError(
                "No persistence diagrams on this TopoResult "
                f"(tool='{self.tool}', ph_result is None)."
            )
        if dim not in self.diagrams:
            raise KeyError(
                f"dim={dim} not computed. Available: {sorted(self.diagrams.keys())}."
            )
        return self.diagrams[dim]

    @property
    def mapper(self) -> MapperView:
        """Grouped accessor for the four mapper-related fields."""
        return MapperView(
            graph=self.mapper_graph,
            node_members=self.mapper_node_members,
            filter_values=self.mapper_filter_values,
            stats=self.mapper_graph_stats,
        )

    @property
    def dimension(self) -> float | None:
        """Scalar dimension estimate, picked from ``dimension_result`` by method.

        Returns the most-natural single number for the estimator used:
        ``id_estimate`` (mean over layers) for activation ID, ``ph_dimension``
        for the Birdal/Dupuis estimator, ``magnitude_dimension`` for Andreeva.
        Returns ``None`` when ``dimension_result`` is empty.
        """
        d = self.dimension_result
        if not d:
            return None
        method = d.get("method", "")
        if method in {"global_2nn", "local_2nn", "global", "local",
                      "calibrated", "calibrated_2nn"}:
            ids = d.get("id_estimates")
            if ids is not None:
                arr = np.asarray(ids, dtype=float)
                arr = arr[np.isfinite(arr)]
                return float(arr.mean()) if arr.size else None
            return float(d.get("id_estimate", float("nan")))
        if method == "ph_dimension":
            return float(d.get("ph_dimension", float("nan")))
        if method == "magnitude_dimension":
            return float(d.get("magnitude_dimension", float("nan")))
        # Fallback: first scalar-looking value
        for key in ("id_estimate", "ph_dimension", "magnitude_dimension"):
            if key in d:
                return float(d[key])
        return None

    # ------------------------------------------------------------------ #
    # Discoverability                                                      #
    # ------------------------------------------------------------------ #

    def _kind_table(self) -> dict[str, dict[str, str]]:
        """Return the kind→spec table relevant to ``self.tool``."""
        if self.tool == "combined":
            merged: dict[str, dict[str, str]] = {}
            for sub in _PLOT_KINDS.values():
                merged.update(sub)
            return merged
        return _PLOT_KINDS.get(self.tool, {})

    def plots_available(self) -> list[str]:
        """List the ``kind`` strings ``self.plot(kind)`` will accept *right now*.

        Filters by which underlying fields are populated *and*, for the
        dimension tool, by which plot makes sense for the estimator used
        (``id_layers`` for activation-ID, ``ph_scaling`` for the PH
        estimator, ``magnitude_scaling`` for magnitude).
        """
        out: list[str] = []
        for kind, spec in self._kind_table().items():
            req = spec["requires"]
            if getattr(self, req, None) is None:
                continue
            if req == "dimension_result":
                method = (self.dimension_result or {}).get("method", "")
                if kind == "id_layers" and method not in {
                    "global_2nn", "local_2nn", "global", "local",
                    "calibrated", "calibrated_2nn",
                }:
                    continue
                if kind == "ph_scaling" and method != "ph_dimension":
                    continue
                if kind == "magnitude_scaling" and method != "magnitude_dimension":
                    continue
            out.append(kind)
        return out

    def describe(self, print_it: bool = True) -> str:
        """Return (and by default print) a human-readable summary.

        Covers: tool, paper reference, which sub-results are populated,
        available statistics, available plot kinds, and config.
        """
        lines: list[str] = []
        lines.append(f"TopoResult(tool={self.tool!r})")
        if self.paper_reference:
            lines.append(f"  paper       : {self.paper_reference}")

        # Populated sub-results
        present: list[str] = []
        if self.ph_result is not None:
            dims = sorted(self.ph_result.diagrams.keys())
            counts = ", ".join(
                f"H{d}: {self.ph_result.diagrams[d].shape[0]} bars" for d in dims
            )
            present.append(f"ph_result ({counts})")
        if self.mapper_graph is not None:
            n_nodes = (self.mapper_graph_stats or {}).get("n_nodes", "?")
            present.append(f"mapper_graph ({n_nodes} nodes)")
        if self.dimension_result is not None:
            method = self.dimension_result.get("method", "?")
            dim = self.dimension
            present.append(
                f"dimension_result (method={method}, "
                f"dim={dim:.3f})" if dim is not None
                else f"dimension_result (method={method})"
            )
        if present:
            lines.append("  data        : " + "; ".join(present))
        else:
            lines.append("  data        : (empty)")

        if self.statistics:
            keys = ", ".join(list(self.statistics.keys())[:6])
            extra = "" if len(self.statistics) <= 6 else f" (+{len(self.statistics)-6} more)"
            lines.append(f"  statistics  : {keys}{extra}")

        plots = self.plots_available()
        if plots:
            kt = self._kind_table()
            lines.append("  plots       :")
            for k in plots:
                doc = kt[k]["doc"]
                lines.append(f"    .plot({k!r}) — {doc}")
        else:
            lines.append("  plots       : (none — populate ph_result / mapper_graph / dimension_result first)")

        if self.config:
            cfg_keys = ", ".join(list(self.config.keys())[:8])
            extra = "" if len(self.config) <= 8 else f" (+{len(self.config)-8} more)"
            lines.append(f"  config keys : {cfg_keys}{extra}")

        text = "\n".join(lines)
        if print_it:
            print(text)
        return text

    # ------------------------------------------------------------------ #
    # Plotting                                                             #
    # ------------------------------------------------------------------ #

    def plot(self, kind: str | None = None, *, save: str | None = None, **kwargs):
        """Route to the appropriate visualisation.

        Parameters
        ----------
        kind : str or None
            Which plot to render.  ``None`` → fall back to
            ``self.default_plot_kind`` (set by ``TDAPipeline.from_paper``)
            or the first entry from :meth:`plots_available`.

            For ``tool="ph"``:
                ``"diagram"`` | ``"barcode"`` | ``"betti_curve"``
            For ``tool="mapper"``:
                ``"graph"`` | ``"ph_diagram"`` (requires ph_result)
            For ``tool="dimension"``:
                ``"ph_scaling"`` | ``"magnitude_scaling"`` | ``"id_layers"``
            For ``tool="combined"``:
                Any of the above; routed to whichever sub-result applies.
        save : str or None
            When given, write the rendered figure to this path (format inferred
            from the extension, e.g. ``.png`` / ``.pdf`` / ``.svg``) before
            returning it.

        Returns
        -------
        matplotlib.figure.Figure
        """
        available = self.plots_available()
        if kind is None:
            kind = self.default_plot_kind
            if kind is None or kind not in available:
                if not available:
                    raise ValueError(
                        f"No plots available for this TopoResult "
                        f"(tool='{self.tool}', no sub-results populated)."
                    )
                kind = available[0]

        table = self._kind_table()
        if kind not in table:
            raise ValueError(
                f"kind='{kind}' is not valid for tool='{self.tool}'. "
                f"Known kinds for this tool: {sorted(table.keys())}."
            )
        if kind not in available:
            req = table[kind]["requires"]
            raise ValueError(
                f"kind='{kind}' requires '{req}' to be populated on this "
                f"TopoResult, but it is None. Available right now: {available}."
            )

        spec = table[kind]
        plot_fn = _resolve_plot_fn(spec["fn"])

        # Each plotter has its own first-positional-argument convention.
        if spec["requires"] == "ph_result":
            fig = plot_fn(self.ph_result, **kwargs)
        elif spec["requires"] == "mapper_graph":
            fig = plot_fn(
                self.mapper_graph,
                self.mapper_node_members,
                self.mapper_filter_values,
                **kwargs,
            )
        elif spec["requires"] == "dimension_result":
            fig = plot_fn(self.dimension_result, **kwargs)
        else:
            raise RuntimeError(f"Unrecognised requires='{spec['requires']}'.")

        if save is not None:
            _save_figure(fig, save)
        return fig


class TopoResultSet(list):
    """A list of :class:`TopoResult`, one per layer, that also acts like one.

    Some analyses are per-layer (Naitzat et al., Karuppiah et al.) and return one
    result for each; most return a single result for the whole network.  Callers
    then had to know which, because ``result.statistics`` worked for one shape
    and raised ``AttributeError`` for the other.

    This is a real ``list`` -- indexing, iteration and ``len`` behave exactly as
    before -- but attribute access falls through to the single element when there
    is exactly one, and otherwise raises an error that says which layer to pick.
    So ``.statistics`` is safe to reach for without branching on the preset.
    """

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if len(self) == 1:
            return getattr(self[0], name)
        raise AttributeError(
            f"This analysis produced {len(self)} per-layer results, so there is "
            f"no single {name!r} to read. Index a layer -- result[0].{name} -- or "
            f"iterate: [r.{name} for r in result]."
        )

    def __repr__(self) -> str:
        return f"TopoResultSet({len(self)} per-layer results)"
