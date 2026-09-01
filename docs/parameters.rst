Parameter reference
===================

A consolidated cheat-sheet of the **valid values** for every ``builder_kwargs`` and
``tool_kwargs`` option, so you don't have to read each function's docstring to find out
what you can put where. (At runtime the same information is available from
:meth:`TDAPipeline.describe_preset`, :meth:`TDAPipeline.explain`, and
``tda.help("graph")`` / ``tda.help("ph")`` / ``tda.help("mapper")`` /
``tda.help("dimension")``.)

The authoritative source is always the function docstrings on the
:doc:`api/graph_builder` and :doc:`api/topo_tools` pages — this page summarises the
categorical choices.

Builders (``builder=`` + ``builder_kwargs=``)
---------------------------------------------

``"weight_graph"`` — weight space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - kwarg
     - valid values
     - notes
   * - ``edge_weight``
     - ``"normalized"`` (default), ``"absolute"``, ``"global_normalized"``,
       ``"relevance"`` · *coupled input also:* ``"correlation"``,
       ``"weighted_activation"``
     - how a weight matrix becomes edge weights; ``"relevance"`` is the
       Watanabe column-normalised relevance
   * - ``graph_scope``
     - ``"multipartite"`` (default), ``"bipartite"``, ``"full"``
     - layer-to-layer wiring; ``"full"`` is one graph over all layers
   * - ``induced_paths``
     - ``True`` | ``False`` (default)
     - requires ``graph_scope="full"``
   * - ``node_feature_fn``
     - ``"laplacian_eigenvectors"`` (default), ``"degree_features"``, ``None``,
       or a callable
     - node features (needed by Mapper)
   * - ``n_node_features``
     - ``int`` (default ``8``)
     - feature dimensionality
   * - ``layer_subset``
     - ``list[int]`` | ``None``
     - restrict to specific layers

``"point_cloud_graph"`` — weight rows / columns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reads a single weight matrix as a **point cloud** rather than a neuron graph —
each row (or column) is one point — and returns the pairwise-distance bundle.
The complementary view to ``weight_graph``; the coordinate-free counterpart of
separating a layer's left and right singular subspaces. Record **one** layer so a
single matrix reaches the builder (e.g. ``layer_selection=["fc1"]``).

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - kwarg
     - valid values
     - notes
   * - ``orientation``
     - ``"rows"`` | ``"cols"`` (default ``"rows"``)
     - for a ``(N_in, N_out)`` weight: ``"rows"`` → the ``N_in`` input features as
       points in output space; ``"cols"`` → the ``N_out`` output neurons
       (receptive fields) as points in input space
   * - ``metric``
     - ``"euclidean"`` (default), ``"cosine"``, ``"correlation"``, ``"cityblock"``
     - distance between points (``scipy.spatial.distance.cdist``)
   * - ``attach_points``
     - ``True`` (default) | ``False``
     - store point coordinates as ``node_features`` (Mapper-ready)

``"activation_graph"`` — activation space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - kwarg
     - valid values
     - notes
   * - ``distance``
     - ``"euclidean"`` (default), ``"geodesic"``, ``"correlation"``, ``"vne"``
     - metric between activation vectors; ``"geodesic"`` builds a kNN manifold
       distance (Naitzat)
   * - ``k``
     - ``int`` | ``None``
     - neighbours for ``"geodesic"``; auto-selected when ``None``
   * - ``drop_constant``
     - ``True`` | ``False`` (default)
     - drop always-off neurons
   * - ``node_sampling`` / ``max_neurons``
     - ``str`` / ``int`` | ``None``
     - subsample neurons for speed
   * - ``node_feature_fn`` / ``n_node_features``
     - as for ``weight_graph``
     - features for Mapper

``"weight_trajectory"`` — trajectory space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - kwarg
     - valid values
     - notes
   * - ``distance``
     - ``"euclidean"`` (default), ``"cosine"``, ``"loss_difference"``
     - metric between snapshots.  For ``"loss_difference"`` pass ``loss_values``:
       a ``(T, n_samples)`` **per-sample** loss matrix gives the Dupuis et al.
       pseudo-metric ``mean_s|ell_i,s - ell_j,s|``; a ``(T,)`` scalar-loss vector
       falls back to ``|mean-loss_i - mean-loss_j|`` (warns)

``"labelled_complex_graph"`` — input + label space (Ramamurthy et al.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Builds the **labelled VR complex**: vertices = one class ``S``, filtered to those
near the opposite class ``W``.

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - kwarg
     - valid values
     - notes
   * - ``distance_metric``
     - ``"euclidean"`` (default), ``"cosine"``, ``"correlation"``
     - metric among vertices / to ``W``
   * - ``source_class``
     - ``int`` | ``None`` (default)
     - the class used as vertices ``S``; ``None`` → first label
   * - ``gamma`` / ``gamma_quantile``
     - ``float`` | ``None`` / ``float`` (default ``0.5``)
     - keep ``S`` vertices within ``gamma`` of ``W``; ``gamma=None`` sets it from
       the quantile of nearest-``W`` distances (``1.0`` keeps all of ``S``)
   * - ``scale``
     - ``"global"`` (default), ``"local"``
     - ``"local"`` = locally-scaled labelled VR (÷ kNN radii, ``k_local``)

``"polyhedral_graph"`` (Liu et al.) takes ``input_type`` (``"auto"`` default).
See :doc:`api/graph_builder` for full signatures.

Tools (``tool=`` + ``tool_kwargs=``)
------------------------------------

``"ph"`` — persistent homology
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - kwarg
     - valid values
     - notes
   * - ``max_dim``
     - ``int`` (default ``1``) — typically ``0``, ``1``, ``2``
     - highest homology dimension
   * - ``coeff``
     - prime ``int`` (default ``2``)
     - field coefficient (``2`` = ℤ/2ℤ)
   * - ``backend``
     - ``"ripser"`` (default), ``"gudhi"``, ``"giotto"``
     - PH engine
   * - ``input_complex``
     - ``"auto"`` (default), ``"directed_clique"``
     - ``"directed_clique"`` is the Watanabe FCN complex
   * - ``relevance_mode``
     - ``"positive"`` (default), ``"absolute"``
     - only for ``input_complex="directed_clique"``
   * - ``schedule_units``
     - ``"value"`` (default), ``"index"``
     - units for ``filtration_schedule``
   * - ``epsilon`` / ``k``
     - ``float`` / ``int`` | ``None``
     - distance/neighbour thresholds

``"mapper"`` — Mapper
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - kwarg
     - valid values
     - notes
   * - ``filter_fn``
     - ``"pca"`` (default), ``"l2_norm"``, ``"eccentricity"``, ``"entropy"``,
       or a callable
     - the lens; ``"pca"`` spreads classes apart
   * - ``n_components``
     - ``int`` (default ``2``)
     - lens dimensionality
   * - ``n_intervals``
     - ``int`` (default ``10``)
     - cover resolution (more = finer graph)
   * - ``overlap_frac``
     - ``float`` in ``(0, 1)`` (default ``0.3``)
     - cover overlap (more = more edges)
   * - ``compute_ph``
     - ``True`` | ``False`` (default)
     - also run PH on the Mapper graph

To **compare** two Mapper outputs (Zhou et al.), use
``tanc.topo_tools.mapper_hypergraph_gw_distance(result_a, result_b)`` — a
Gromov-Wasserstein distance over the Mappers as hypergraphs (nodes = sample
memberships, cluster-size masses).  The plain-graph ``mapper_gw_distance`` is
only a partial proxy.

``"dimension"`` — intrinsic / fractal / magnitude dimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - kwarg
     - valid values
     - notes
   * - ``estimator``
     - ``"activation_id"``, ``"trajectory_dimension"``
     - per-layer intrinsic dim, or whole-trajectory dim
   * - ``method`` *(activation_id)*
     - ``"global"``, ``"local"``, ``"calibrated"``
     - ``"global"`` asymptotic 2NN; ``"local"`` (Ruppik); ``"calibrated"`` (Ong,
       the ong2026 default) sample-size-calibrated ``(k, j)`` — accepts
       ``k``/``j`` (default 2, 1)
   * - ``method`` *(trajectory_dimension)*
     - ``"magnitude"``, ``"ph_euclidean"``, ``"ph_loss"``
     - magnitude dimension, or PH (fractal) dimension over Euclidean / loss
       distance (``"ph_loss"`` uses the Dupuis per-sample metric when per-sample
       losses were captured via ``TrainingExtractor(loss_eval_data=(X, y))``)
       distance

Representation (``representation=``)
------------------------------------

When you pass a ``ModelSnapshot`` / ``TrainingView`` to ``fit`` / ``reproduce``,
``representation=`` selects *which space* to pull off it:

* ``"weights"`` — all weight matrices
* ``"activations"`` — all per-layer activation matrices
* ``"coupled"`` — paired (weight, activation) per layer
* ``"inputs_labels"`` — ``(snap.inputs, snap.predicted_labels)``
* a **layer-name string** (e.g. ``"fc1"``) — that one layer's activations
* a **callable** ``snap -> array(s)`` — e.g. a sub-stack of weights:
  ``lambda s: s.weight_matrices(["fc1", "fc2"])``

Activation pooling (``activation_pooling=``)
--------------------------------------------

At extraction time (``ModelExtractor``, ``extract_model``, ``TDAPipeline.fit_model``)
this controls how a captured activation collapses to ``(N_samples, features)``:

* ``"flatten"`` (default) — ``(N, …) -> (N, -1)``; the historical behaviour, so
  MLP/CNN extraction is unchanged.
* ``"last"`` / ``"first"`` / ``"mean"`` / ``"max"`` — **token pooling** for
  transformer ``(batch, seq, hidden)`` activations, reducing over the *sequence*
  axis (``"last"`` = final-token summary of a causal decoder; ``"first"`` ≈ BERT
  ``[CLS]``).  Applies to 3-D activations only; 2-D and conv feature maps always
  flatten.  Works for both PyTorch and TensorFlow.

Layer selection (``layer_selection=``)
--------------------------------------

Which layers the extractor hooks for weights / activations.  String presets:
``"all_linear"`` · ``"all_conv"`` · ``"soro"`` (Sum-of-Rank-One layers) ·
``"linear_and_conv"`` · ``"all"``; or a list of layer names / indices; or
``None`` to auto-detect (MLP → linear, CNN → conv, mixed → prompt).

SoRO layers (``soro_effective_weight=``)
----------------------------------------

A **SoRO** (Sum-of-Rank-One) layer is a factored FC layer ``W = U · diag(sigma) ·
Vᵀ``.  It is recognised by a duck-typed protocol — a module exposing
``soro_factors() -> (U, sigma, V)`` (or ``U`` / ``V`` + ``sigma`` /
``singular_values`` / ``s`` attributes), optionally ``effective_weight()`` — so no
registration is needed beyond implementing that on your layer class (PyTorch and
Keras).  Extraction always keeps the trained factors in ``snapshot.soro_factors``;
``soro_effective_weight=True`` (default) *also* stores the assembled ``W`` in
``snapshot.weights`` (as ``(in, out)``) so the layer feeds the weight-based tools
like any FC layer.  Set it ``False`` to keep only the factors (recover ``W`` via
``snapshot.effective_weight(name)``).

Mapper sweeps (``MapperGrid`` / ``MapperStudy``)
------------------------------------------------

Every axis takes a **scalar to pin** it or a **list to sweep** it.  A ``tuple`` is
one composite value, so ``n_intervals=(30, 20)`` is a single per-dimension setting
rather than two alternatives; wrap genuinely list-valued alternatives in
``Sweep([...])``.  See :doc:`sweep_overview` for what the measures mean.

Declare axes in the order below — shared work is reused down the order, so an
axis placed late is recomputed needlessly.

``layer`` (``MapperStudy``)
    Layer name(s), as reported by ``population.layer_names()``.  Each
    ``(layer, view)`` pair becomes its own named cloud.

``view`` (``MapperStudy``)
    How a layer is read as points.  **The available views depend on the rank of
    the weight**, because a 4-D convolutional tensor has no single "row".

    *For a 2-D weight* — a linear layer, an embedding table, an attention
    projection, or an activation matrix:
    ``"full"``, ``"rows"``, ``"cols"``, ``"row_norm"``, ``"col_norm"``,
    ``"row_sum"``, ``"col_sum"``, ``"gram_rows"``, ``"gram_cols"``,
    ``"gram_diag"``.  With ``part="upper"`` (default) the Gram views keep the
    upper triangle, which loses nothing and roughly halves the dimensionality.
    ``normalise="rows"`` L2-normalises each row first, so ``"gram_diag"``
    becomes the mean cosine at each lag.

    *For a convolutional weight* ``(out, in, h, w)``, each view names **which
    axes index points**; the rest become that point's coordinates.  For
    ``Conv2d(1, 16, kernel_size=(5, 5))``:

    .. list-table::
       :header-rows: 1
       :widths: 16 22 14 48

       * - view
         - point axes
         - cloud
         - meaning
       * - ``"rows"``
         - ``out, in, h``
         - ``(80, 5)``
         - one **kernel row** per point
       * - ``"cols"``
         - ``out, in, w``
         - ``(80, 5)``
         - one **kernel column** per point
       * - ``"kernel"``
         - ``out, in``
         - ``(16, 25)``
         - one whole kernel per point
       * - ``"out_channel"``
         - ``out``
         - ``(16, 25)``
         - one output channel per point (spans all input channels)
       * - ``"in_channel"``
         - ``in``
         - ``(1, 400)``
         - one input channel per point
       * - ``"tap"``
         - ``h, w``
         - ``(25, 16)``
         - one spatial position per point

    ``"rows"`` and ``"cols"`` mean the rows and columns *of the kernel itself*,
    which is how a kernel is normally drawn and discussed.  Conv1d weights are
    promoted to ``(out, in, 1, w)``, so ``"cols"`` there would give points of
    dimension 1 — that is **refused** with a message naming the alternatives,
    rather than silently producing a cloud with no geometry.

    ``"full"`` is deliberately *not* a conv view: its meaning on a conv layer
    already depends on ``per_filter``, and routing it would silently change
    existing results.  Use ``"out_channel"`` for the per-filter reading.

``preprocess``
    ``None`` (default — preprocessing changes the metric the clusterer sees, so
    it belongs on a swept axis), ``"l2"``, ``"mean_centre"``, ``"standardise"``,
    ``("density", k, p)`` to keep the densest fraction *p*, ``("norm", q, side)``
    to keep points by norm, a callable, or a list applied in order.

``lens``
    ``"pca1"``, ``"pca2"``, ``"pcaN"``, ``"l2"``, ``"tsne1"``, ``"tsne2"``,
    ``"umap2"``, ``"density"``, or a callable applied to the **whole matrix** (so
    lenses fitted across points, not just row-wise, are supported).  ``tsne`` and
    ``umap`` are stochastic — their seed is recorded in the manifest.

``n_intervals``
    Any integer ≥ 1, up to ``max_intervals(n_points, lens_dim, overlap)``.
    Beyond that, typical cells hold too few points to cluster and the graph
    shatters; ``validate()`` warns.

``overlap``
    ``0 <= overlap < 1``, in the **standard** convention
    (intersection ÷ interval length, as in KeplerMapper and giotto-tda).  Width
    diverges as it approaches 1.  Converting from a width-ratio cover: use
    :func:`~tanc.topo_tools.mapper_sweep.convert_overlap`.

``metric``
    Any scikit-learn metric, or ``None`` to take the clusterer's own.  Ward
    accepts ``"euclidean"`` only.  A cosine metric on an L2-normalised cloud is
    rejected as a duplicate of the euclidean configuration.

``clusterer``
    ``DBSCANCells(eps=…, min_samples=…, metric=…)``,
    ``SingleLinkageCells(threshold=…, metric=…, max_cell=…)``, or
    ``WardCells(threshold=…)``.  ``eps`` accepts a float, ``"elbow"``, or
    ``("quantile", q)``; thresholds accept a float or ``("quantile", q)``.  Put
    this axis **last** — several single-linkage thresholds over one cover reuse
    each cell's dendrogram.

``measures``
    Groups from ``MEASURE_GROUPS``: ``"size"``, ``"cover"``, ``"lattice"``,
    ``"topology"``, ``"shape"`` (all default), plus ``"expensive"``
    (``density``, ``mean_clustering``, ``diameter``) which is opt-in because it
    costs seconds per graph.


Properties and methods
----------------------

The toolkit follows one rule for accessors:

* a **property** when it is a cheap read of something already computed —
  ``result.errors``, ``result.rejected``, ``population.seeds``,
  ``population.accuracies``, ``population.trained``, ``population.untrained``;
* a **method** when it takes arguments or does real work —
  ``result.rows(status=...)``, ``result.leading(...)``, ``result.graph(row)``,
  ``population.cloud(layer, view)``, ``population.save(path)``.

The rule is easy to state and impossible to guess from a name, and getting it
wrong in the property-called-as-method direction used to fail unhelpfully::

    >>> population.trained()
    TypeError: 'TrainedPopulation' object is not callable

These accessors now tolerate the stray parentheses for one release, returning the
same value with a ``DeprecationWarning``.  Ordinary list and array behaviour is
unaffected.  The tolerance will be removed, after which the call fails loudly —
which is the right end state, but not a good way to discover the rule.
