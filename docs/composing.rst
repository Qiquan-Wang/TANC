Composing analyses — the core idea
==================================

A neural network is not one object to analyse but **several spaces** you can study,
and TANC is built around *choosing* one and pointing a topological method at it.
Every analysis is a point along four axes:

.. code-block:: text

   ┌─ 1. SPACE ────────┐   ┌─ 2. CONSTRUCTION ─┐   ┌─ 3. METHOD ──┐   ┌─ 4. OUTPUT ──────┐
   │ weights           │   │ weight_graph      │   │ PH           │   │ diagram/barcode  │
   │ activations       │ → │ activation_graph  │ → │ Mapper       │ → │ Mapper graph     │
   │ inputs + labels   │   │ labelled_complex  │   │ dimension    │   │ scaling plot     │
   │ weight trajectory │   │ polyhedral / traj │   │              │   │ statistics       │
   └───────────────────┘   └───────────────────┘   └──────────────┘   └──────────────────┘

The whole toolkit is the freedom to **mix and match** these. Reproducing a published
method (:doc:`paper_reproductions`) is just a *shortcut* that pre-fills the four
choices — never the only path.

The four axes
-------------

**1. Space** — *which part of the network.* Produced by the extractor
(:doc:`api/model_extractor`) from a trained model:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Space
     - How you get it
     - Shape
   * - weight space
     - ``snap.weight_matrices([...])``
     - list of ``(in, out)`` matrices
   * - activation space
     - ``snap.all_activation_matrices()``
     - list of ``(N, neurons)`` matrices
   * - input + label space
     - ``(snap.inputs, snap.predicted_labels)``
     - point cloud + integer labels
   * - weight-trajectory space
     - ``view`` / ``view.weight_trajectory()``
     - ordered snapshots over training

**2. Construction** (``builder=``) — *which graph to build.*
``"weight_graph"`` · ``"point_cloud_graph"`` · ``"activation_graph"`` ·
``"kernel_graph"`` · ``"labelled_complex_graph"`` · ``"polyhedral_graph"`` ·
``"weight_trajectory"`` · or ``None`` to feed the method directly (e.g. intrinsic
dimension needs no graph). ``"point_cloud_graph"`` reads one weight matrix as a
cloud of its rows or columns — the complement of ``"weight_graph"``'s neuron graph.

**3. Method** (``tool=``) — *which topological tool.*
``"ph"`` (persistent homology) · ``"mapper"`` · ``"dimension"`` (intrinsic /
fractal / magnitude dimension).

**4. Output** — *what you read off.* ``result.plot(kind)`` with, by tool,
``diagram`` | ``barcode`` | ``betti_curve`` (PH), ``graph`` | ``ph_diagram``
(Mapper), ``id_layers`` | ``ph_scaling`` | ``magnitude_scaling`` (dimension); plus
``result.statistics``, ``result.describe()``, and ``result.save()``.

For the **valid values of every** ``builder_kwargs`` / ``tool_kwargs`` option
(``edge_weight``, ``distance``, ``filter_fn``, ``method``, …) see the
:doc:`parameters` cheat-sheet.

Putting it together
-------------------

You don't have to extract arrays by hand. Pass a ``ModelSnapshot`` /
``TrainingView`` straight to ``fit`` (or ``reproduce``) and *say what you want* with
``representation=`` — ``"weights"``, ``"activations"``, a layer name like ``"fc1"``, or
a callable for a sub-stack (``lambda s: s.weight_matrices(["fc1", "fc2"])``).  The
``fit_model`` / ``fit_training`` entry points do the same starting from a live model.
For a transformer, ``fit_model(..., activation_pooling="last")`` collapses each
``(batch, seq, hidden)`` block over the sequence axis (final-token summary) so every
sequence is one point in ``hidden``-space instead of being flattened.

To turn any lens into a **time series**, ``pipe.over_training(view, measures=[...],
layers=[...])`` runs the pipeline on every snapshot and returns a ``TrajectorySeries``
(the numbers plus a ``.plot()``) — one line per layer.

The call is always the same shape — pick coordinates, ``fit`` a space, ``plot`` an
output:

.. code-block:: python

   from tanc import TDAPipeline

   # weight space → weight_graph → PH → diagram
   pipe = TDAPipeline(builder="weight_graph", tool="ph", tool_kwargs={"max_dim": 1})
   result = pipe.fit(snap.weight_matrices(["fc1", "fc2"]))
   result.plot("diagram")

   # activation space → activation_graph → Mapper → graph
   pipe = TDAPipeline(builder="activation_graph",
                      builder_kwargs={"distance": "euclidean"}, tool="mapper")
   pipe.fit(snap.all_activation_matrices()[-2]).plot("graph")

   # activation space → (no graph) → intrinsic dimension per layer
   pipe = TDAPipeline(builder=None, tool="dimension",
                      tool_kwargs={"estimator": "activation_id", "method": "global"})
   pipe.fit(snap.all_activation_matrices()).plot("id_layers")

   # weight-trajectory space → (no graph) → magnitude dimension
   pipe = TDAPipeline(builder=None, tool="dimension",
                      tool_kwargs={"estimator": "trajectory_dimension",
                                   "method": "magnitude"})
   pipe.fit(view).plot("magnitude_scaling")

Which combinations make sense
-----------------------------

The axes are **not fully independent** — each space has a natural construction.
``TDAPipeline.validate()`` (run automatically by ``fit``) calls
:func:`tanc.check_compatibility`, which injects sensible defaults and raises a
clear error on a nonsensical pairing rather than producing a misleading picture. The
combinations that mean something:

.. list-table::
   :header-rows: 1
   :widths: 24 26 18 32

   * - Space
     - ``builder``
     - ``tool``
     - Typical output
   * - weight space
     - ``weight_graph``
     - ``ph``
     - diagram / barcode / Betti curve
   * - activation space
     - ``activation_graph``
     - ``ph`` / ``mapper``
     - diagram / Mapper graph
   * - activation space
     - ``None``
     - ``dimension``
     - ``id_layers`` (intrinsic dim per layer)
   * - input + label space
     - ``labelled_complex_graph``
     - ``ph``
     - diagram (decision-boundary topology)
   * - activation patterns
     - ``polyhedral_graph``
     - ``ph``
     - diagram (linear-region structure)
   * - weight trajectory
     - ``weight_trajectory`` / ``None``
     - ``dimension``
     - ``ph_scaling`` / ``magnitude_scaling``

A fifth axis: time
------------------

Because :class:`~tanc.model_extractor.TrainingView` captures the network at
many points during training, *any* of the above becomes a time series — watch a
diagram, an intrinsic dimension, or a Mapper graph evolve as the network learns. See
the training-dynamics plots in :doc:`visualisation_overview`.

The shortcut
------------

When a published method happens to fix a useful set of coordinates,
``TDAPipeline.from_paper("<name>")`` hands you that pipeline pre-filled — identical to
what you would build by hand:

.. code-block:: python

   pipe = TDAPipeline.from_paper("watanabe2021")   # weight_graph → ph, directed clique
   pipe.explain()                                  # prints the four choices it fixed
   result, fig = pipe.reproduce(weight_matrices)

See :doc:`paper_reproductions` for the full recipe gallery, and the ``TUTORIAL.ipynb``
notebook in the repository for an end-to-end, train-one-model-and-compose walkthrough.
