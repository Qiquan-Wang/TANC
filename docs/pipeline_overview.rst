Pipeline overview
=================

.. currentmodule:: tanc

The single entry point is :class:`TDAPipeline`.  It wires a **graph construction**
(``builder=``) to a **topological method** (``tool=``) — the middle two of the four
axes in :doc:`composing`.  Construct one of two ways:

* **Direct construction** (the general case) with ``builder=``, ``tool=``,
  ``builder_kwargs=``, ``tool_kwargs=`` — you choose each stage::

      pipe = TDAPipeline(builder="activation_graph", tool="mapper",
                         builder_kwargs={"distance": "euclidean"})

* :meth:`TDAPipeline.from_paper("<name>") <TDAPipeline.from_paper>` — a **shortcut**
  that pre-fills those same arguments for a published method (18 presets at time of
  writing; run :meth:`TDAPipeline.list_presets` for the catalogue).  See
  :doc:`composing` for the full menu of builders, tools, and the spaces that feed
  them.

Methods
-------

``fit_model`` / ``fit_training`` / ``fit_models`` / ``fit_each`` are the
*model-first* entry points: hand them a trained model (or several) and the
pipeline extracts the weights/activations it needs — ``fit_each`` returns one
result per model (the population path used by Ballester et al.).

.. autoclass:: TDAPipeline
   :members: from_paper, list_presets, describe_preset, explain,
             fit, fit_model, fit_training, fit_models, fit_each,
             reproduce, plot, compare

The result
----------

Every call to ``.fit()`` returns a :class:`tanc.TopoResult`, which
carries a typed sub-result for whichever tool ran (PH, Mapper, or
dimension) plus a uniform set of accessors and introspection methods.

.. autoclass:: tanc.TopoResult
   :members: diagrams, diagram, mapper, dimension,
             plots_available, describe, plot
   :no-index:

Paper presets
-------------

.. autofunction:: tanc.list_presets
.. autofunction:: tanc.describe_preset

The full preset dictionary is exported as :data:`tanc.PAPER_PRESETS`.