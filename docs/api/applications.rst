applications
============

Task-level applications built on top of the ``graph_builder`` → ``topo_tools``
pipeline. Currently: persistent-homology network pruning (PHPM, Watanabe &
Yamana 2020) — keep the weight-edges that carry a fully-connected sub-network's
strongest ``H1`` loops and zero the rest, with a global-magnitude-pruning
baseline.

.. automodule:: tanc.applications.pruning
   :members:
   :undoc-members:
   :show-inheritance:
