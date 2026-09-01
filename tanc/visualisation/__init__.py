from tanc.visualisation.representations import (
    plot_persistence_diagram,
    plot_barcode,
    plot_betti_curve,
    plot_diagram_comparison,
    plot_persistence_landscape,
    plot_persistence_image,
    plot_betti_layer_bars,
    persistence_landscape,
    persistence_image,
)
from tanc.visualisation.sweep_plots import (
    COLOURINGS,
    node_colour,
    plot_mapper_sweep_graph,
    plot_stability_heatmap,
    plot_cover_degeneracy,
    plot_node_size_distribution,
    plot_filter_sweep,
    plot_population_summary,
    plot_graph_panel,
)
from tanc.visualisation.visualisation_utils import (
    make_figure,
    annotate_ph_stats,
    format_paper_reference,
)
from tanc.visualisation.training_plots import (
    plot_training_curves,
    plot_weight_norm_trajectory,
    plot_activation_stats_trajectory,
)
from tanc.visualisation.trajectory_plots import (
    detect_settle,
    plot_id_over_training,
    plot_id_trajectory_all_layers,
    plot_ph_dimension_over_training,
    plot_magnitude_dimension_over_training,
    plot_ph_statistic_trajectory,
    plot_betti_trajectory,
    plot_diagram_distance_trajectory,
    plot_diagram_distance_matrix,
    plot_ph_statistic_pairplot,
    plot_ph_statistics_panel,
    pipeline_trajectory,
    TrajectorySeries,
)
from tanc.visualisation.evolution_plots import (
    plot_diagram_evolution,
)
from tanc.visualisation.graph_plots import (
    plot_graph_matrix,
    plot_node_embedding,
    plot_tu_score_distribution,
    plot_tu_roc,
    plot_pathways_on_network,
    h0_signal_pathways,
    plot_polyhedral_regions,
    plot_id_with_uncertainty,
    plot_id_qq,
)
from tanc.visualisation.summary_plots import (
    plot_training_summary,
    make_training_animation,
)
from tanc.visualisation.pipeline_diagram import (
    plot_pipeline_diagram,
)

# Re-exports from topo_tools so users can find every plot under one namespace.
from tanc.topo_tools.dimension_tool import (
    plot_id_across_layers,
    plot_ph_scaling,
    plot_magnitude_scaling,
    plot_2nn_ratio_distribution,
    plot_loglog_ratio_distribution,
)

__all__ = [
    # Persistence representations
    "plot_persistence_diagram",
    "plot_barcode",
    "plot_betti_curve",
    "plot_diagram_comparison",
    "plot_persistence_landscape",
    "plot_persistence_image",
    "plot_betti_layer_bars",
    "persistence_landscape",
    "persistence_image",
    # Helpers
    "make_figure",
    "annotate_ph_stats",
    "format_paper_reference",
    # Training-dynamics plots
    "plot_training_curves",
    "plot_weight_norm_trajectory",
    "plot_activation_stats_trajectory",
    # Trajectory plots
    "plot_id_over_training",
    "plot_id_trajectory_all_layers",
    "plot_ph_dimension_over_training",
    "plot_magnitude_dimension_over_training",
    "plot_ph_statistic_trajectory",
    "plot_betti_trajectory",
    "plot_diagram_distance_trajectory",
    "detect_settle",
    "plot_diagram_distance_matrix",
    "plot_ph_statistic_pairplot",
    "plot_ph_statistics_panel",
    "pipeline_trajectory",
    "TrajectorySeries",
    # Evolution
    "plot_diagram_evolution",
    # Summary / animation
    "plot_training_summary",
    "make_training_animation",
    # Graph / embedding / TU / ID diagnostics
    "plot_graph_matrix",
    "plot_node_embedding",
    "plot_tu_score_distribution",
    "plot_tu_roc",
    "plot_pathways_on_network",
    "h0_signal_pathways",
    "plot_polyhedral_regions",
    "plot_id_with_uncertainty",
    "plot_id_qq",
    # Pipeline diagram
    "plot_pipeline_diagram",
    # Mapper sweeps — colouring and cover diagnostics
    "COLOURINGS",
    "node_colour",
    "plot_mapper_sweep_graph",
    "plot_stability_heatmap",
    "plot_cover_degeneracy",
    "plot_node_size_distribution",
    "plot_filter_sweep",
    "plot_population_summary",
    "plot_graph_panel",
    # Re-exports from topo_tools.dimension_tool
    "plot_id_across_layers",
    "plot_ph_scaling",
    "plot_magnitude_scaling",
    "plot_2nn_ratio_distribution",
    "plot_loglog_ratio_distribution",
]