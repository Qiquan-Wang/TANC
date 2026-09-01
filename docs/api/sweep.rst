Mapper sweeps
=============

API reference for the sweep machinery. See :doc:`../sweep_overview` for the
narrative version — what ``b1_excess`` means, which overlap convention applies,
and how to choose ``eps``.

.. currentmodule:: tanc


Study — the assembled entry point
---------------------------------

.. autoclass:: tanc.pipeline.mapper_study.MapperStudy
   :members: requirements, validate, run
   :no-index:

.. autoclass:: tanc.pipeline.mapper_study.MapperStudyResult
   :members: rows, table, rejected, errors, leading, plateaus, graph, cloud_of, labels_of
   :no-index:


Populations
-----------

.. autofunction:: tanc.model_extractor.population.train_population
   :no-index:

.. autoclass:: tanc.model_extractor.population.SeedPlan
   :members: from_seeds, derive
   :no-index:

.. autoclass:: tanc.model_extractor.population.TrainedPopulation
   :members: cloud, cloud_shape, member_index, layer_names, accuracies,
             trained, untrained, save, load
   :no-index:

.. autoclass:: tanc.model_extractor.population.PopulationMember
   :members: weight
   :no-index:


Reading a matrix as a point cloud
---------------------------------

.. autodata:: tanc.model_extractor._views.VIEWS
   :no-index:

.. autofunction:: tanc.model_extractor._views.matrix_view
   :no-index:

.. autofunction:: tanc.model_extractor._views.stack_views
   :no-index:

.. autofunction:: tanc.model_extractor._views.view_shape

.. autodata:: tanc.model_extractor._views.CONV_VIEWS

.. autofunction:: tanc.model_extractor._views.conv_view

.. autofunction:: tanc.model_extractor._views.conv_view_shape
   :no-index:


The grid
--------

.. autoclass:: tanc.topo_tools.mapper_sweep.MapperGrid
   :members: axes, validate, run
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.validate_config
   :no-index:


Cover and nerve
---------------

.. autofunction:: tanc.topo_tools.mapper_sweep.convert_overlap
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.max_intervals
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.cover_intervals
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.build_cover
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.lens_ranges
   :no-index:

.. autoclass:: tanc.topo_tools.mapper_sweep.CoverPlan
   :members: nerve, dendrogram, release_dendrograms, n_nonempty, empty_frac,
             cell_sizes, cell_size_median, cell_size_max, mean_cells_per_point
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.nerve_of
   :no-index:


Lenses and preprocessing
------------------------

.. autodata:: tanc.topo_tools.mapper_sweep.LENS_BUILDERS
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.resolve_lens
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.resolve_preprocess
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.codensity
   :no-index:


Clusterers
----------

Each clusters within one cover cell. Data-dependent parameters are given as
strategies and resolved once per cloud by
:func:`~tanc.topo_tools.mapper_sweep.calibrate`, because distance scales
differ completely between representations.

.. autoclass:: tanc.topo_tools.mapper_sweep.DBSCANCells
   :no-index:

.. autoclass:: tanc.topo_tools.mapper_sweep.SingleLinkageCells
   :no-index:

.. autoclass:: tanc.topo_tools.mapper_sweep.WardCells

.. autoclass:: tanc.topo_tools.mapper_sweep.FirstGapCells

.. autoclass:: tanc.topo_tools.mapper_sweep.HDBSCANCells
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.calibrate
   :no-index:


Graphs and measures
-------------------

.. autoclass:: tanc.topo_tools.mapper_sweep.MapperGraph
   :members: n_nodes, n_edges, node_sizes, node_lens, to_networkx
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.mapper_graph
   :no-index:

.. autodata:: tanc.topo_tools.mapper_sweep.MEASURE_GROUPS
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.measure_graph
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.save_graph
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.load_graph
   :no-index:

.. autofunction:: tanc.topo_tools.mapper_sweep.stored_nerve
   :no-index:


Colouring and diagnostics
-------------------------

.. autodata:: tanc.visualisation.sweep_plots.COLOURINGS
   :no-index:

.. autofunction:: tanc.visualisation.sweep_plots.node_colour
   :no-index:

.. autofunction:: tanc.visualisation.sweep_plots.plot_mapper_sweep_graph

.. autofunction:: tanc.visualisation.sweep_plots.plot_graph_panel
   :no-index:

.. autofunction:: tanc.visualisation.sweep_plots.plot_stability_heatmap
   :no-index:

.. autofunction:: tanc.visualisation.sweep_plots.plot_cover_degeneracy
   :no-index:

.. autofunction:: tanc.visualisation.sweep_plots.plot_node_size_distribution
   :no-index:

.. autofunction:: tanc.visualisation.sweep_plots.plot_filter_sweep
   :no-index:

.. autofunction:: tanc.visualisation.sweep_plots.plot_population_summary
   :no-index:


The sweep engine
----------------

Generic grid execution, with no Mapper dependency. Any function from a
configuration to a row of numbers can use it.

.. autofunction:: tanc.topo_tools._sweep_engine.run_sweep
   :no-index:

.. autofunction:: tanc.topo_tools._sweep_engine.expand_grid
   :no-index:

.. autoclass:: tanc.topo_tools._sweep_engine.Sweep
   :no-index:

.. autoclass:: tanc.topo_tools._sweep_engine.Stage
   :no-index:

.. autoclass:: tanc.topo_tools._sweep_engine.StagedContext
   :members: for_config, release
   :no-index:

.. autoclass:: tanc.topo_tools._sweep_engine.SweepStore
   :members: completed, rows, record_config, record_result, write_manifest,
             artifact_path
   :no-index:

.. autofunction:: tanc.topo_tools._sweep_engine.config_hash
   :no-index:

.. autofunction:: tanc.topo_tools._sweep_engine.canonical_config
   :no-index:

.. autofunction:: tanc.topo_tools._sweep_engine.display_config
   :no-index:
