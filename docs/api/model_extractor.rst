model_extractor
===============

.. automodule:: tanc.model_extractor
   :members:
   :undoc-members:
   :show-inheritance:

Extractor classes (preferred)
-----------------------------

.. autoclass:: tanc.model_extractor.ModelExtractor
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: tanc.model_extractor.TrainingExtractor
   :members:
   :show-inheritance:
   :no-index:

One-call functions
------------------

.. autofunction:: tanc.model_extractor.extract_model
   :no-index:
.. autofunction:: tanc.model_extractor.extract_training
   :no-index:
.. autofunction:: tanc.model_extractor.inspect
   :no-index:

Snapshots and training views
----------------------------

``ModelSnapshot`` and ``TrainingView`` inherit ``.save()`` / ``.load()`` —
see :doc:`persistence`.

Both are documented in full under *model_extractor* above:
:class:`~tanc.model_extractor.ModelSnapshot` and
:class:`~tanc.model_extractor.TrainingView`.