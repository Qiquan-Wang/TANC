TANC documentation
==================

**T**\ opological **A**\ nalysis of **N**\ eural networks through **C**\ omposition.

A **composable pipeline**: choose the *space* of the network you want to study (its
weights, its activations, the trajectory its optimiser traces), the *graph
construction* to apply, the topological *method* to run (persistent homology, Mapper,
or intrinsic dimension), and the *output* you read off (a diagram, a Mapper graph, a
scaling plot, a number).

Mix and match those four choices to build whatever lens you need. Reproducing a
published method is just a **shortcut** — ``TDAPipeline.from_paper("...")`` pre-fills
the four choices for you.

First result
------------

.. code-block:: bash

   pip install -e .

.. code-block:: python

   from tanc import TDAPipeline

   pipe   = TDAPipeline.from_paper("watanabe2021")
   result = pipe.fit(weight_matrices)
   result.plot("diagram")

Hand it a raw model instead of matrices and the extractor works out the framework
for you:

.. code-block:: python

   from tanc.model_extractor import extract_model

   snapshot = extract_model(my_trained_model, X_test)
   result   = pipe.fit(snapshot.weight_matrices())

Where to start
--------------

**New here?** Read :doc:`composing` first — it is the core idea, the four axes
(space × construction × method × output) and how to combine them, with the
compatibility matrix. Then :doc:`getting_started` for install and worked examples.

**Want to see it run?** :doc:`tutorial` is the hands-on tour: train one CNN on
CIFAR-10 (PyTorch *and* TensorFlow) and mix-and-match every analysis on it end to
end.

**Prefer point-and-click?** :doc:`visual_builder` documents the Visual Builder —
a no-code front-end that assembles a network, dataset and analysis, then generates
the runnable Python script for you.

**Reproducing a paper?** :doc:`paper_reproductions` is the preset gallery — one
rendered notebook per ``from_paper`` preset.

**Looking up a value?** :doc:`parameters` is the cheat-sheet of valid values for
every ``builder_kwargs`` / ``tool_kwargs`` option (``edge_weight``, ``distance``,
``filter_fn``, ``method``, …).

**Sweeping Mapper?** :doc:`sweep_overview` covers running Mapper across a parameter
grid: train a seed population, sweep the cover and clustering, and read
``b1_excess`` — the graph's first Betti number *minus its cover's* — so a discovery
can be told apart from an artefact of the cover. Includes the overlap-convention
warning and how to choose ``eps``.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   composing
   parameters
   tutorial
   visual_builder
   pipeline_overview
   sweep_overview
   visualisation_overview

.. toctree::
   :maxdepth: 1
   :caption: Preset recipes (shortcuts)

   paper_reproductions

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/pipeline
   api/model_extractor
   api/graph_builder
   api/topo_tools
   api/sweep
   api/applications
   api/visualisation
   api/persistence

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
