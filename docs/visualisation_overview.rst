Visualisation overview
======================

.. currentmodule:: tanc.visualisation

Every plotting function lives under :mod:`tanc.visualisation`.
Most users interact with the persistence plots indirectly via
``result.plot(kind)`` on a :class:`~tanc.TopoResult`; the others
take a :class:`~tanc.TrainingView`, a
:class:`~tanc.GraphBundle`, or raw arrays.

Persistence representations
---------------------------

.. autosummary::
   :toctree: _generated

   plot_persistence_diagram
   plot_barcode
   plot_betti_curve
   plot_diagram_comparison
   plot_persistence_landscape
   plot_persistence_image
   plot_betti_layer_bars

Training dynamics
-----------------

.. autosummary::
   :toctree: _generated

   plot_training_curves
   plot_weight_norm_trajectory
   plot_activation_stats_trajectory

Topological trajectories over epochs
------------------------------------

.. autosummary::
   :toctree: _generated

   plot_id_over_training
   plot_id_trajectory_all_layers
   plot_ph_dimension_over_training
   plot_magnitude_dimension_over_training
   plot_ph_statistic_trajectory
   plot_betti_trajectory
   plot_diagram_distance_trajectory
   plot_diagram_distance_matrix
   plot_ph_statistic_pairplot
   pipeline_trajectory
   plot_diagram_evolution

Dashboards and animations
-------------------------

.. autosummary::
   :toctree: _generated

   plot_training_summary
   make_training_animation
   plot_pipeline_diagram

Per-snapshot dimension diagnostics
----------------------------------

.. autosummary::
   :toctree: _generated

   plot_id_across_layers
   plot_ph_scaling
   plot_magnitude_scaling
   plot_2nn_ratio_distribution
   plot_loglog_ratio_distribution

Graphs, embeddings, and TU scores
---------------------------------

.. autosummary::
   :toctree: _generated

   plot_graph_matrix
   plot_node_embedding
   plot_tu_score_distribution
   plot_tu_roc
   plot_pathways_on_network
   h0_signal_pathways
   plot_polyhedral_regions
   plot_id_with_uncertainty
   plot_id_qq

Utility helpers
---------------

.. autosummary::
   :toctree: _generated

   make_figure
   annotate_ph_stats
   format_paper_reference