"""compatibility.py — cross-module validation for TDAPipeline.

Validates that a chosen Module 2 tool is compatible with the output of a
chosen Module 1 builder and enforces required configuration changes.
"""

from __future__ import annotations

import warnings


def check_compatibility(
    builder_name: str,
    builder_kwargs: dict,
    tool_name: str,
    tool_kwargs: dict,
) -> tuple[dict, dict]:
    """Validate and potentially modify (builder_kwargs, tool_kwargs).

    Rules
    -----
    1. ``tool_name="mapper"`` requires node features.
       If ``builder_kwargs.get("node_feature_fn") is None``, set to
       ``"laplacian_eigenvectors"`` and emit UserWarning.

    2. ``tool_name="ph"`` + ``builder_name="activation_graph"`` +
       ``distance="correlation"``:  emit UserWarning reminding the user
       that PH operates on neuron-neuron distances (Ballester et al. 2024).

    3. ``builder_name="weight_graph"`` + ``induced_paths=True`` requires
       ``graph_scope="full"``.  Raise ValueError otherwise.

    4. ``tool_name="mapper"`` requires giotto-tda; raise ImportError with
       install hint if not available.

    Returns
    -------
    (builder_kwargs, tool_kwargs) — possibly modified.

    Raises
    ------
    ValueError : incompatible configuration.
    ImportError : required dependency missing.
    """
    builder_kwargs = dict(builder_kwargs)
    tool_kwargs = dict(tool_kwargs)

    # Rule 1 — mapper needs node features.
    # weight_graph is the only builder that produces no node features by default;
    # all others (activation_graph, boundary_graph, polyhedral_graph,
    # weight_trajectory) naturally populate node_features from their inputs.
    if tool_name == "mapper" and builder_name == "weight_graph":
        if builder_kwargs.get("node_feature_fn") is None:
            builder_kwargs["node_feature_fn"] = "laplacian_eigenvectors"
            warnings.warn(
                "Mapper requires node features.  "
                "Automatically setting node_feature_fn='laplacian_eigenvectors' "
                "on the builder.  To suppress this warning, explicitly set "
                "node_feature_fn in builder_kwargs.",
                UserWarning,
                stacklevel=4,
            )

    # Rule 2 — correlation activation graph + PH
    if (tool_name == "ph"
            and builder_name == "activation_graph"
            and builder_kwargs.get("distance") == "correlation"):
        warnings.warn(
            "PH is computed on a neuron-neuron correlation distance matrix "
            "(N_neurons × N_neurons), not on sample-sample distances — "
            "consistent with Ballester et al. (2024).",
            UserWarning,
            stacklevel=4,
        )

    # Rule 3 — induced_paths requires full scope
    if (builder_name == "weight_graph"
            and builder_kwargs.get("induced_paths") is True
            and builder_kwargs.get("graph_scope") != "full"):
        raise ValueError(
            "builder_kwargs['induced_paths']=True requires "
            "builder_kwargs['graph_scope']='full', "
            f"but got graph_scope='{builder_kwargs.get('graph_scope')}'."
        )

    # Rule 4 — mapper requires giotto-tda
    if tool_name == "mapper":
        try:
            import gtda  # noqa: F401
        except ImportError:
            raise ImportError(
                "The Mapper tool requires giotto-tda.  "
                "Install with: pip install giotto-tda"
            )

    return builder_kwargs, tool_kwargs
