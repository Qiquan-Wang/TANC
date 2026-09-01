"""mapper_study.py — one call from an untrained architecture to a swept Mapper analysis.

A study joins the three expensive stages that are otherwise wired by hand:
train a seed population, read each member as a point cloud, and sweep Mapper
over the cover and clustering parameters.  Everything it does is available
piecewise — :func:`~tanc.model_extractor.population.train_population`,
:class:`~tanc.topo_tools.mapper_sweep.MapperGrid`, and the plotting
functions can all be used directly.  This is the assembled version for when you
want the whole thing.

Checking before training, not after
-----------------------------------
The costly ordering mistake in this kind of work is training a population for
hours and only then discovering that the layer you meant to analyse was never
captured, or that the epoch axis you swept needed checkpoints that were turned
off.  :meth:`MapperStudy.requirements` derives what the grid will need from the
grid itself, and :meth:`MapperStudy.validate` checks that against the training
configuration **before** the first model is built.

Layers and views are grid axes
------------------------------
Which layer and which reading of it carries the structure is usually the open
question, so both are swept rather than fixed.  Each ``(layer, view)`` pair
becomes its own named cloud, and the sweep engine reuses that cloud across
every lens, cover and clusterer below it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tanc._compat import callable_property

__all__ = ["MapperStudy", "MapperStudyResult"]


def _as_list(v: Any) -> list:
    """Axis values: a list sweeps, anything else pins (a tuple is one value)."""
    return list(v) if isinstance(v, list) else [v]


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MapperStudyResult:
    """Everything a finished study produced.

    Attributes
    ----------
    store : SweepStore
        The run directory, with per-configuration results and saved graphs.
    population : TrainedPopulation or None
        The networks analysed, when the study trained them.
    clouds : dict
        The named ``(layer, view)`` point clouds that were swept.
    labels : dict
        Per-cloud point labels — which member each point came from — for the
        cross-member colourings.
    """

    store: Any
    population: Any = None
    clouds: dict[str, np.ndarray] = field(default_factory=dict)
    labels: dict[str, np.ndarray] = field(default_factory=dict)

    # ── the table ───────────────────────────────────────────────────────────

    def rows(self, status: str | None = "ok") -> list[dict]:
        """Result rows, by default only those that ran successfully."""
        rs = self.store.rows()
        return [r for r in rs if status is None or r.get("status") == status]

    def table(self):
        """Results as a :class:`pandas.DataFrame`, if pandas is available."""
        import pandas as pd
        return pd.DataFrame(self.store.rows())

    @property
    def rejected(self) -> list[dict]:
        """Configurations refused before running, each with its reason."""
        return callable_property("rejected", self.rows(status="rejected"))

    @property
    def errors(self) -> list[dict]:
        """Configurations that raised, each with its exception."""
        return callable_property("errors", self.rows(status="error"))

    # ── selection ───────────────────────────────────────────────────────────

    def leading(
        self,
        measure: str = "b1_excess",
        *,
        min_node_median: float = 5.0,
        max_node_ratio: float | None = None,
        n: int = 10,
    ) -> list[dict]:
        """Best-scoring configurations that are also legible.

        Ranking on a topological measure alone selects for degenerate graphs:
        a shattered cover produces an enormous ``b1`` that is just
        ``E - V + b0`` over a dust of singleton nodes.  The legibility filter
        is therefore applied first, not offered as an afterthought.

        Parameters
        ----------
        measure : str
            Ranked descending.
        min_node_median : float
            Reject graphs whose median node holds fewer points than this.  The
            *median* matters rather than the mean, because a handful of huge
            nodes can carry a mean while most nodes are singletons.
        max_node_ratio : float, optional
            Reject graphs with more nodes per cell than this — the opposite
            failure, where the clustering has fragmented every cell.
        n : int
            How many to return.
        """
        out = []
        for r in self.rows():
            if measure not in r:
                continue
            if float(r.get("node_median", 0)) < min_node_median:
                continue
            if max_node_ratio is not None and float(r.get("node_ratio", 0)) > max_node_ratio:
                continue
            out.append(r)
        out.sort(key=lambda r: float(r[measure]), reverse=True)
        return out[:n]

    def plateaus(
        self,
        measures: Sequence[str] = ("b0", "b1", "b1_excess"),
        *,
        min_size: int = 3,
        round_to: int = 0,
    ) -> list[dict]:
        """Groups of configurations whose topology is indistinguishable.

        A structure that survives a *range* of parameters is evidence; one that
        appears at a single setting is a parameter accident.  Configurations are
        grouped by their rounded measure tuple, and groups reported largest
        first.

        Returns
        -------
        list of dict
            Each with ``signature``, ``n_configs``, and the parameter values
            spanned.
        """
        groups: dict[tuple, list[dict]] = {}
        for r in self.rows():
            if not all(m in r for m in measures):
                continue
            key = tuple(round(float(r[m]), round_to) for m in measures)
            groups.setdefault(key, []).append(r)

        out = []
        for key, members in groups.items():
            if len(members) < min_size:
                continue
            spans: dict[str, list] = {}
            for axis in ("cloud", "lens", "n_intervals", "overlap", "clusterer"):
                vals = sorted({str(m.get(axis)) for m in members if axis in m})
                if vals:
                    spans[axis] = vals
            out.append({
                "signature": dict(zip(measures, key)),
                "n_configs": len(members),
                "spans": spans,
                "configs": members,
            })
        out.sort(key=lambda g: -g["n_configs"])
        return out

    # ── graphs ──────────────────────────────────────────────────────────────

    def graph(self, row_or_hash: Any):
        """Load the saved Mapper graph for a result row or configuration hash."""
        from tanc.topo_tools.mapper_sweep import load_graph
        h = row_or_hash["hash"] if isinstance(row_or_hash, dict) else row_or_hash
        path = self.store.artifact_path(h)
        if not path.exists():
            raise FileNotFoundError(
                f"No saved graph for {h}. Was the study run with save_graphs=False?"
            )
        return load_graph(path)

    def cloud_of(self, row: dict) -> np.ndarray:
        """The point cloud a result row was computed from."""
        return self.clouds[row["cloud"]]

    def labels_of(self, row: dict) -> np.ndarray:
        """Per-point member indices for a row's cloud, for colouring."""
        return self.labels[row["cloud"]]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"MapperStudyResult({len(self.rows())} ok, "
                f"{len(self.rejected)} rejected, {len(self.errors)} errors "
                f"-> {self.store.path})")


# ─────────────────────────────────────────────────────────────────────────────
# Study
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MapperStudy:
    """Train a population, read it as clouds, and sweep Mapper over them.

    Supply **either** ``model_fn`` and ``train_data`` to train a population, or
    ``population`` / ``clouds`` to analyse something you already have.

    Parameters
    ----------
    model_fn : callable, optional
        Returns a fresh untrained model.  Triggers training.
    train_data, val_data, extract_data : optional
        Passed to :func:`~tanc.model_extractor.population.train_population`.
    population : TrainedPopulation, optional
        Analyse an existing population instead of training one.
    clouds : dict, optional
        Analyse point clouds directly, bypassing models entirely.
    n_models, seeds, master_seed, epochs, batch_size : optional
        Population size, seeding and training length; see ``train_population``.
    criterion, optimizer_fn, checkpoint_every, include_untrained : optional
        Training parameters; see ``train_population``.
    layer_selection, aspects, device : optional
        Extraction parameters; see ``train_population``.
    layer : str or list
        Layer(s) to analyse.  A list sweeps them.
    view : str or list
        Reading(s) of each layer — see
        :data:`~tanc.model_extractor._views.VIEWS`.  A list sweeps them.
    part, normalise, source, per_filter
        How each layer is turned into matrices; see
        :meth:`~tanc.model_extractor.population.TrainedPopulation.cloud`.
    preprocess : spec or list
        Transforms, applied before the lens.
    point_filter : spec or list
        Filters selecting a subset of points, applied *after* the lens with the
        cover pinned to the unfiltered lens range -- so sweeping this axis
        changes only the point set, not the cover.
    lens, n_intervals, overlap, metric, clusterer, measures, save_graphs
        Grid axes and options; see
        :class:`~tanc.topo_tools.mapper_sweep.MapperGrid`.
    verbose : bool

    Examples
    --------
    >>> study = MapperStudy(                                       # doctest: +SKIP
    ...     model_fn=make_net, train_data=(X, y), n_models=20, epochs=30,
    ...     criterion=nn.CrossEntropyLoss(),
    ...     optimizer_fn=lambda m: torch.optim.Adam(m.parameters()),
    ...     layer=["0", "3"], view=["full", "gram_diag"],
    ...     n_intervals=[10, 20, 30], overlap=[0.3, 0.5],
    ... )
    >>> study.validate()                 # before anything expensive runs
    >>> result = study.run("runs/study01")
    >>> result.plateaus()[0]["spans"]
    """

    # ── sources ──
    model_fn: Any = None
    train_data: Any = None
    val_data: Any = None
    extract_data: Any = None
    population: Any = None
    clouds: dict[str, np.ndarray] | None = None

    # ── training ──
    n_models: int = 1
    seeds: Sequence[int] | None = None
    master_seed: int | None = None
    epochs: int = 10
    batch_size: int = 128
    criterion: Any = None
    optimizer_fn: Any = None
    checkpoint_every: Any = None
    include_untrained: bool = False
    layer_selection: Any = "linear_and_conv"
    aspects: list[str] | None = None
    device: str | None = None

    # ── representation ──
    layer: Any = None
    view: Any = "full"
    part: str = "upper"
    normalise: str | None = None
    source: str = "auto"
    per_filter: bool = True

    # ── grid ──
    preprocess: Any = None
    point_filter: Any = None
    lens: Any = "pca2"
    n_intervals: Any = 10
    overlap: Any = 0.3
    metric: Any = None
    clusterer: Any = None
    measures: Sequence[str] = ("size", "cover", "lattice", "topology", "shape")
    save_graphs: bool = True
    seed: int = 0
    verbose: bool = True

    #: Set by validate(); run() re-validates quietly once this is true.
    _validated: bool = field(default=False, repr=False)

    def __post_init__(self):
        n_sources = sum(x is not None for x in (self.model_fn, self.population, self.clouds))
        if n_sources == 0:
            raise ValueError(
                "MapperStudy needs something to analyse: pass model_fn (to train a "
                "population), population (to analyse an existing one), or clouds "
                "(to analyse point clouds directly)."
            )
        if n_sources > 1:
            raise ValueError(
                "MapperStudy was given more than one source (model_fn / population / "
                "clouds). Pass exactly one so it is unambiguous which is analysed."
            )
        if self.model_fn is not None and self.train_data is None:
            raise ValueError("model_fn was given but train_data was not.")

    # ── requirements ────────────────────────────────────────────────────────

    def requirements(self) -> dict[str, Any]:
        """What the grid needs the training stage to have captured.

        Derived from the grid, so it stays correct as axes change.

        Returns
        -------
        dict
            ``layers``, ``needs_activations``, ``needs_checkpoints``.
        """
        return {
            "layers": [l for l in _as_list(self.layer) if l is not None],
            "needs_activations": self.source == "activations",
            "needs_checkpoints": False,          # reserved for an epoch axis
        }

    def validate(self, verbose: bool | None = None) -> list[tuple[dict, str]]:
        """Check the study can run, before anything expensive happens.

        Two kinds of problem are caught: a training configuration that would
        not capture what the grid asks for, and grid configurations that are
        internally contradictory.

        Returns
        -------
        list of (config, reason)
            Grid-level rejections.  Requirement failures raise instead, because
            they invalidate the whole study rather than individual points in it.

        Raises
        ------
        ValueError
            If the training configuration cannot supply what the grid needs.
        """
        verbose = self.verbose if verbose is None else verbose
        self._validated = True
        req = self.requirements()

        if req["needs_activations"] and self.extract_data is None and self.clouds is None:
            raise ValueError(
                "source='activations' needs extract_data — a fixed batch to run "
                "through each model — but none was given. Activations cannot be "
                "recovered afterwards without it."
            )
        if req["needs_checkpoints"] and self.checkpoint_every is None:
            raise ValueError(
                "The grid sweeps over training time, which needs checkpoints, but "
                "checkpoint_every is None. Set it to 'epoch' before training — a "
                "trajectory cannot be reconstructed from a final snapshot."
            )
        if self.population is not None and req["layers"]:
            have = set(self.population.layer_names(self.source if self.source != "auto" else "auto"))
            missing = [l for l in req["layers"] if l not in have]
            if missing:
                raise ValueError(
                    f"The grid asks for layer(s) {missing}, which this population does "
                    f"not carry. Available: {sorted(have)}. A population captures only "
                    f"the layers selected when it was trained."
                )
        if self.clouds is None and self.model_fn is not None and not req["layers"]:
            raise ValueError(
                "No layer was given, so there is nothing to read from the trained "
                "models. Set layer= to a layer name, or a list to sweep several."
            )

        if verbose:
            print(f"MapperStudy requirements: layers={req['layers']}, "
                  f"activations={req['needs_activations']}, "
                  f"checkpoints={req['needs_checkpoints']}  — training config is compatible")

        # Grid-level validation needs the clouds, which may not exist yet.  When
        # they do not, validate a stand-in of the right width so the axis checks
        # that do not depend on the data still run early.
        grid = self._grid(self._clouds_or_placeholder())
        return grid.validate(verbose=verbose)

    # ── execution ───────────────────────────────────────────────────────────

    def run(self, out: Any, *, resume: Any = None) -> MapperStudyResult:
        """Train if needed, build the clouds, and sweep.

        Parameters
        ----------
        out : str or Path
            Run directory.  Never overwritten — an existing name gets a numeric
            suffix and the path actually used is printed.
        resume : str or Path, optional
            Continue a previous run, skipping completed configurations.

        Returns
        -------
        MapperStudyResult
        """
        # Always re-validate — the configuration may have changed since an
        # explicit call — but stay quiet the second time so calling validate()
        # yourself does not double the output.
        self.validate(verbose=self.verbose and not getattr(self, "_validated", False))

        pop = self.population
        if self.clouds is not None:
            clouds = dict(self.clouds)
            labels = {k: np.zeros(len(v), dtype=int) for k, v in clouds.items()}
        else:
            if pop is None:
                from tanc.model_extractor.population import train_population
                # Ask the extractor for exactly what the grid needs.  Left as None
                # this defaults to every aspect, including activations — which
                # forces a forward pass, and so demands extract_data even for a
                # study that only ever looks at weights.
                aspects = self.aspects
                if aspects is None:
                    aspects = ["weights"]
                    if self.requirements()["needs_activations"]:
                        aspects.append("activations")
                pop = train_population(
                    self.model_fn, self.train_data, val_data=self.val_data,
                    extract_data=self.extract_data, n_models=self.n_models,
                    seeds=self.seeds, master_seed=self.master_seed, epochs=self.epochs,
                    batch_size=self.batch_size,
                    criterion=self.criterion, optimizer_fn=self.optimizer_fn,
                    checkpoint_every=self.checkpoint_every,
                    include_untrained=self.include_untrained,
                    aspects=aspects, layer_selection=self.layer_selection,
                    device=self.device, verbose=self.verbose,
                )
            clouds, labels = self._build_clouds(pop)

        if self.verbose:
            print(f"MapperStudy: {len(clouds)} cloud(s)")
            for name, X in clouds.items():
                print(f"  {name:<28} {X.shape}")

        grid = self._grid(clouds)
        store = grid.run(out, resume=resume, progress=self.verbose)
        result = MapperStudyResult(store=store, population=pop,
                                   clouds=clouds, labels=labels)
        self._write_study_manifest(store, clouds, pop)
        return result

    # ── internals ───────────────────────────────────────────────────────────

    def _build_clouds(self, pop) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """One named cloud per (layer, view) pair, with per-point member labels.

        When the population carries an untrained control, trained and untrained
        members become **separate** clouds rather than one pooled cloud.  Pooling
        them would measure the topology of the union, which is neither group, and
        would leave nothing to compare the trained networks against — the whole
        point of capturing a control.
        """
        groups: list[tuple[str, Any]] = [("", pop)]
        if any(not m.trained for m in pop.members):
            trained, untrained = pop.trained, pop.untrained
            if trained.members and untrained.members:
                groups = [(" [trained]", trained), (" [untrained]", untrained)]

        clouds: dict[str, np.ndarray] = {}
        labels: dict[str, np.ndarray] = {}
        for layer in _as_list(self.layer):
            for view in _as_list(self.view):
                for suffix, sub in groups:
                    name = f"{layer}:{view}{suffix}"
                    clouds[name] = sub.cloud(
                        layer, view, part=self.part, normalise=self.normalise,
                        per_filter=self.per_filter, source=self.source,
                    )
                    labels[name] = sub.member_index(
                        layer, view, per_filter=self.per_filter, source=self.source,
                    )
        return clouds, labels

    def _clouds_or_placeholder(self) -> dict[str, np.ndarray]:
        """Real clouds when available, otherwise a stand-in for early validation."""
        if self.clouds is not None:
            return dict(self.clouds)
        if self.population is not None:
            return self._build_clouds(self.population)[0]
        names = [f"{l}:{v}" for l in _as_list(self.layer) for v in _as_list(self.view)]
        # Width is unknown before training; use something wide enough that the
        # lens-component checks are not tripped by the placeholder itself.
        return {n: np.zeros((2, 64)) for n in names}

    def _grid(self, clouds: dict[str, np.ndarray]):
        from tanc.topo_tools.mapper_sweep import MapperGrid
        return MapperGrid(
            clouds=clouds, preprocess=self.preprocess,
            point_filter=self.point_filter, lens=self.lens,
            n_intervals=self.n_intervals, overlap=self.overlap,
            metric=self.metric, clusterer=self.clusterer,
            measures=self.measures, save_graphs=self.save_graphs, seed=self.seed,
        )

    def _write_study_manifest(self, store, clouds, pop) -> None:
        """Record the study-level configuration beside the sweep's own manifest."""
        payload = {
            "layers": _as_list(self.layer),
            "views": _as_list(self.view),
            "part": self.part,
            "normalise": self.normalise,
            "source": self.source,
            "per_filter": self.per_filter,
            "clouds": {k: list(v.shape) for k, v in clouds.items()},
            "population": None if pop is None else {
                "n_members": len(pop),
                "master_seed": pop.plan.master,
                "seeds": pop.seeds,
                "config": pop.config,
            },
        }
        (Path(store.path) / "study.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
