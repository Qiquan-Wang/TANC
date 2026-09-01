Mapper sweeps
=============

Running Mapper once gives you a graph. It does not tell you whether that graph
describes your data or merely reflects the cover you chose — and **every** cover
produces a graph, on any data at all, including noise.

This part of the toolkit exists to answer that question. It sweeps Mapper across
the parameters that are usually guessed, reports the **nerve of the cover**
alongside every graph so the two can be compared, and flags the two ways a
Mapper graph goes wrong.

.. contents::
   :local:
   :depth: 2


The one number to read first
----------------------------

``b1_excess`` is the first Betti number of the Mapper graph minus that of the
nerve of its own cover:

.. math:: b_1^{\text{excess}} = b_1(\text{Mapper}) - b_1(\text{nerve})

A cover of overlapping cells has loops of its own. Mapper inherits them whether
or not the data has any structure, so ``b1`` on its own cannot distinguish a
discovery from an artefact of the cover. The same point cloud makes this vivid —
a noisy circle, identical clusterer, only the lens dimension changed:

.. list-table::
   :header-rows: 1
   :widths: 12 10 14 12 40

   * - lens
     - ``b1``
     - nerve ``b1``
     - excess
     - reading
   * - 1-D
     - 1
     - 0
     - **1**
     - a real loop
   * - 2-D
     - 37
     - 37
     - **≈ 0**
     - the cover's own loops

Read ``b1`` alone and the second row looks like a spectacular finding.


The overlap convention
----------------------

``overlap`` is the fraction of an interval shared with its neighbour:

.. math:: \text{overlap} = \frac{|I_k \cap I_{k+1}|}{|I_k|}

matching KeplerMapper's ``perc_overlap`` and giotto-tda's ``overlap_frac``, after
Carrière, Michel & Oudot, *Statistical Analysis and Parameter Selection for
Mapper* (`arXiv:1706.00204 <https://arxiv.org/abs/1706.00204>`_). Interval width
is therefore ``range / (n_intervals * (1 - overlap))``, and ``overlap`` must
satisfy ``0 <= overlap < 1`` because the width diverges as it approaches 1.

.. warning::
   A different convention appears in some hand-rolled Mapper code: widening each
   interval by a fraction of the *spacing*, ``width = (range / n) * (1 + o)``.
   The two disagree. An ``o`` there equals ``o / (1 + o)`` here, so a nominal
   **0.67 in the width-ratio convention is only 0.40 in the standard one**.

   :func:`~tanc.topo_tools.mapper_sweep.convert_overlap` translates
   between them::

       >>> convert_overlap(0.67)                                  # -> 0.401
       >>> convert_overlap(0.67, frm="standard", to="width_ratio") # -> 2.030

   If you are porting parameters from existing code, check which convention it
   used before comparing results.


How many intervals can the data support?
----------------------------------------

A cover cell holding fewer points than the clusterer needs cannot produce a
node, and a cover of such cells shatters the graph into singletons — which
inflates ``b1`` spectacularly while meaning nothing.
:func:`~tanc.topo_tools.mapper_sweep.max_intervals` gives the ceiling:

.. math:: n_{\max} = \frac{1}{1-g}\left(\frac{N}{c\,m}\right)^{1/d}

for ``N`` points, a ``d``-dimensional lens, overlap ``g``, ``min_samples`` ``m``,
and ``c`` cells' worth of headroom (default 10, targeting ~50 points per cell).

This assumes points spread evenly across the lens image. Real lenses are peaked,
so the *median* cell is emptier than this average — treat the number as an upper
bound and check ``cell_size_median`` on the cover you actually build.
:meth:`~tanc.topo_tools.mapper_sweep.MapperGrid.validate` prints a warning
when a requested resolution exceeds it.


The parameter axes
------------------

Every axis takes a **scalar to pin** it or a **list to sweep** it. A tuple is a
single composite value, so ``n_intervals=(30, 20)`` is one per-dimension setting
rather than two alternatives.

Declare them in this order. The sweep reuses expensive work down the order, so an
axis placed late is recomputed needlessly.

.. list-table::
   :header-rows: 1
   :widths: 18 52 22

   * - axis
     - values
     - reused below it
   * - ``cloud``
     - named point clouds (``layer:view`` pairs, from a study)
     - the cloud
   * - ``preprocess``
     - ``None``, ``"l2"``, ``"mean_centre"``, ``"standardise"``,
       ``("density", k, p)``, ``("norm", q, side)``, callable
     - the cloud
   * - ``lens``
     - ``"pca1"``, ``"pca2"``, ``"l2"``, ``"tsne2"``, ``"umap2"``,
       ``"density"``, or a callable on the whole matrix
     - the lens
   * - ``n_intervals``
     - any integer up to ``max_intervals``
     - the cover
   * - ``overlap``
     - ``0 <= overlap < 1``
     - the cover
   * - ``metric``
     - any scikit-learn metric, or ``None``
     - —
   * - ``clusterer``
     - ``DBSCANCells``, ``SingleLinkageCells``, ``WardCells``
     - —

Put ``clusterer`` last. Several single-linkage thresholds over one cover reuse
each cell's dendrogram, which is roughly a 4.8x saving on that arm.

Preprocessing defaults to ``None``. It changes the metric the clusterer sees, so
it belongs on a swept axis rather than running in the background.


Choosing epsilon
----------------

For DBSCAN, ``eps`` must be **above** the largest gap between consecutive points
along a structure and **below** the separation between distinct structures
sharing a cover cell. That window is usually wide — on a noisy circle a loop was
recovered across ``eps`` from 0.05 to 1.30, a 26-fold range.

.. warning::
   ``eps="elbow"`` is convenient but unreliable on curve-like or filamentary
   data. The knee of the k-NN distance curve tends to land at the *noise* scale
   rather than the along-curve spacing. On a circle of 1,200 points with noise
   0.03 it returns 0.034 — enough to break every arc — while anything from 0.15
   up recovers the loop cleanly.

   The damage is visible rather than silent: it shows as a large
   ``n_components`` together with an inflated ``cpc_mean``. Check both before
   trusting a graph built on an automatically chosen radius.

The reliable approach is not to pick a value but to sweep
``("quantile", q)`` over a log-spaced range and **look for the plateau**. A loop
that holds across a range of ``eps`` is evidence; one that appears at a single
value is not.

The two failure modes are not symmetric. Too large collapses toward the nerve,
where ``b1_excess`` goes to zero and you can *see* there is nothing. Too small
shatters the graph and produces a large ``b1`` that looks like a discovery.
**Err large.**


Reading the diagnostics
-----------------------

Three measures catch the two opposite failures, which is why a single criterion
cannot serve:

``node_ratio``
    Nodes per non-empty cover cell. At 1, the graph *is* the nerve.

``cpc_frac_1``
    Fraction of cells yielding exactly one cluster. At 1, likewise — a lattice.

``node_median``
    Median node size. Falling to 1 means the opposite failure: the graph has
    shattered, and its large ``b1`` is ``E - V + b0`` over a dust of singletons.

Use the *median* node size, never the mean. A handful of huge nodes can carry a
mean of 27 while the median sits at 1.

:func:`~tanc.visualisation.plot_cover_degeneracy` draws all three against
resolution;
:func:`~tanc.visualisation.plot_stability_heatmap` shows a measure across
the cover parameters, where a real feature appears as a *region* rather than an
isolated cell.


Colouring nodes
---------------

Any per-point quantity can colour a node. The obvious choice — how many source
models a node contains — is badly confounded with node size: for a population of
100 networks, a 400-point node holds ~98 of them by chance.

:func:`~tanc.visualisation.node_colour` therefore offers
``normalise="vs_expected"``, dividing by the count expected under independent
assignment, :math:`M(1 - (1 - 1/M)^n)`. Measured on a 100-member population,
raw diversity correlates with node size at **r = 0.98**; the normalised version
at **r = 0.04**.

A ratio near 1 is the **pass**: a node cannot hold more sources than it has
members, so exceeding the null is impossible, and falling well below it means a
few sources contributed most of the node.

Recolouring is free. Node membership is saved with each graph, so a finished
sweep can be re-examined without recomputing any Mapper.


Worked example
--------------

.. code-block:: python

   from tanc.pipeline import MapperStudy
   from tanc.topo_tools import DBSCANCells

   study = MapperStudy(
       model_fn    = make_net,             # returns a fresh untrained model
       train_data  = (X, y),
       n_models    = 20,
       epochs      = 30,
       criterion   = nn.CrossEntropyLoss(),
       optimizer_fn= lambda m: torch.optim.Adam(m.parameters()),

       layer       = ["conv1", "conv2"],   # swept
       view        = ["rows", "kernel"],   # swept — conv views; see parameters
       lens        = ["pca2", "l2"],
       n_intervals = [10, 20, 30],
       overlap     = [0.3, 0.5],
       clusterer   = [DBSCANCells(eps=("quantile", q)) for q in (1, 5, 25)],
   )

   study.validate()                # before anything expensive happens
   result = study.run("runs/study01")

   result.plateaus()[0]["spans"]   # parameter regions with stable topology
   result.leading(measure="b1_excess", min_node_median=20)

``validate()`` checks two things before the first model is built: that the
training configuration will capture what the grid asks for, and that no grid
configuration is internally contradictory. Discovering after three hours of
training that the layer you wanted was never recorded is the expensive mistake
this prevents.


Reproducibility
---------------

Populations are seeded through
:class:`~tanc.model_extractor.population.SeedPlan`, which derives distinct,
statistically independent streams from one master seed via
``numpy.random.SeedSequence``. Distinctness alone is not enough — two arbitrary
integers can seed correlated streams.

Passing no seed still records one, so an unseeded run remains reproducible
afterwards. Three sources of randomness are controlled separately::

    vary_init        = True    initial weights            (the point)
    vary_batch_order = False   order batches are seen in
    vary_split       = False   which samples are held out

Only initialisation varies by default, so a difference between members is
attributable to it alone.

.. note::
   Seed-exact reproduction is reliable on CPU. On GPU, and for TensorFlow in
   particular, nondeterministic kernels mean identical seeds can give slightly
   different weights. The seeds are still recorded and the population remains
   statistically reproducible.


Where results go
----------------

A run claims its directory atomically and **never overwrites**: an existing name
gets ``-002``, ``-003``, and the path actually used is printed at the start, not
the end, so a crashed run can still be found.

.. code-block:: text

   runs/study01/
     manifest.json     grid definition, seeds, library versions
     study.json        layers, views, population provenance
     configs.jsonl     config hash -> resolved parameters
     results.jsonl     config hash -> measures (append-only)
     artifacts/        one .npz per graph: members, edges, lens, nerve

Results are appended one line at a time and flushed, so a crash costs the line in
flight rather than the file. Re-running the same grid into the same directory
does nothing unless you pass ``resume=``, which is deliberate: resuming is
something you ask for by name.
