Saving & loading
================

.. currentmodule:: tanc

Every result container in the toolkit —
:class:`~tanc.topo_tools._result.TopoResult`,
:class:`~tanc.graph_builder._bundle.GraphBundle`,
:class:`~tanc.model_extractor._snapshot.ModelSnapshot`, and
:class:`~tanc.model_extractor._snapshot.TrainingView` — gains ``.save(path)``
and a matching ``.load(path)`` classmethod from the mixin below, so you can stop at
any pipeline stage and resume later without recomputing.  Figures save separately,
via ``result.plot(kind, save="fig.pdf")`` and ``pipe.reproduce(..., save="fig.png")``.

A ``.tda`` file is a single pickled, version-tagged envelope.

.. warning::
   ``.tda`` files are pickles — load only files you created or trust, and treat them
   as cache (they are Python-only and not guaranteed stable across major dependency
   upgrades).

.. code-block:: python

   result = pipe.fit(W)
   result.save("runs/ph.tda")                 # diagrams + statistics
   from tanc.topo_tools import TopoResult
   result = TopoResult.load("runs/ph.tda")    # reload — no recompute
   result.plot("diagram", save="figs/ph.pdf") # the figure too

.. autoclass:: tanc._serialization.SaveLoadMixin
   :members:

.. autofunction:: tanc._serialization.save_tda
.. autofunction:: tanc._serialization.load_tda
