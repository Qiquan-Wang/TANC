"""model_adapter.py — bridge from model_extractor to pipeline data inputs.

Converts a :class:`~tanc.model_extractor.ModelSnapshot` or
:class:`~tanc.model_extractor.TrainingView` into the exact ``data``
argument that each Module 1 builder (or dimension tool when ``builder=None``)
expects.  Keeps that mapping out of ``TDAPipeline.fit`` so the orchestrator
stays focused on dispatch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tanc.model_extractor._snapshot import ModelSnapshot, TrainingView


# Edge-weight modes that need (weight, pre-synaptic activation) pairs.
_COUPLED_EDGE_WEIGHTS = {"weighted_activation", "correlation"}


def is_model_data(data: Any) -> bool:
    """True when ``data`` is a ModelSnapshot or TrainingView."""
    return isinstance(data, (ModelSnapshot, TrainingView))


def snapshot_to_builder_data(
    obj: ModelSnapshot | TrainingView,
    builder: str | None,
    builder_kwargs: dict,
    tool: str,
    tool_kwargs: dict,
) -> Any:
    """Pull the right array(s) off ``obj`` for the selected builder / tool.

    Parameters
    ----------
    obj
        A ``ModelSnapshot`` (single point in time) or a ``TrainingView``
        (ordered snapshots from a training run).
    builder
        The pipeline's ``builder`` field — builder name, or ``None`` for
        the dimension tool path.
    builder_kwargs
        Already-validated builder kwargs; inspected to decide e.g. whether
        we need coupled ``(weight, activation)`` pairs.
    tool, tool_kwargs
        Used only when ``builder is None`` to route between the activation-
        based and trajectory-based dimension estimators.

    Returns
    -------
    Data formatted for the target builder / tool.
    """
    if builder is None:
        return _dimension_data(obj, tool_kwargs)

    if builder == "weight_graph":
        return _weight_graph_data(obj, builder_kwargs)

    if builder == "activation_graph":
        return _activation_graph_data(obj)

    if builder == "labelled_complex_graph":
        return _labelled_complex_data(obj)

    if builder == "polyhedral_graph":
        return _polyhedral_data(obj)

    if builder == "weight_trajectory":
        return _weight_trajectory_data(obj)

    raise ValueError(
        f"Unknown builder '{builder}' — cannot adapt model snapshot to its "
        "input format."
    )


# ── Per-builder helpers ──────────────────────────────────────────────────────


def _final_snapshot(obj: ModelSnapshot | TrainingView) -> ModelSnapshot:
    return obj.final_snapshot if isinstance(obj, TrainingView) else obj


def _weight_graph_data(
    obj: ModelSnapshot | TrainingView, builder_kwargs: dict
) -> Any:
    snap = _final_snapshot(obj)
    if builder_kwargs.get("edge_weight") in _COUPLED_EDGE_WEIGHTS:
        return snap.coupled_weight_activations()
    return snap.weight_matrices()


def _activation_graph_data(obj: ModelSnapshot | TrainingView) -> list[np.ndarray]:
    snap = _final_snapshot(obj)
    mats = snap.all_activation_matrices()
    if not mats:
        raise ValueError(
            "Snapshot has no activation matrices — re-run the extractor with "
            "aspects including 'activations'."
        )
    return mats


def _labelled_complex_data(
    obj: ModelSnapshot | TrainingView,
) -> tuple[np.ndarray, np.ndarray]:
    snap = _final_snapshot(obj)
    if snap.inputs is None:
        raise ValueError(
            "labelled_complex_graph needs snap.inputs — re-run the extractor "
            "with activations or classifications so the inputs are captured."
        )
    labels = snap.predicted_labels
    if labels is None:
        raise ValueError(
            "labelled_complex_graph needs predicted_labels — re-run the "
            "extractor with aspects including 'classifications'."
        )
    return snap.inputs, labels


def _polyhedral_data(obj: ModelSnapshot | TrainingView) -> np.ndarray:
    snap = _final_snapshot(obj)
    mats = snap.all_activation_matrices()
    if not mats:
        raise ValueError(
            "polyhedral_graph needs hidden-layer activations — re-run the "
            "extractor with aspects including 'activations'."
        )
    n_samples = mats[0].shape[0]
    if not all(m.shape[0] == n_samples for m in mats):
        raise ValueError(
            "All activation matrices must share the sample dimension to "
            "concatenate ReLU patterns; got shapes "
            f"{[m.shape for m in mats]}."
        )
    return np.concatenate(mats, axis=1)


def _weight_trajectory_data(
    obj: ModelSnapshot | TrainingView,
) -> list[list[np.ndarray]]:
    if not isinstance(obj, TrainingView):
        raise ValueError(
            "builder='weight_trajectory' requires a TrainingView — call "
            "pipeline.fit_training(...) or pass the view returned by "
            "extract_training()."
        )
    return obj.weight_trajectory()


def _dimension_data(
    obj: ModelSnapshot | TrainingView, tool_kwargs: dict
) -> Any:
    estimator = tool_kwargs.get("estimator", "activation_id")

    if estimator == "activation_id":
        snap = _final_snapshot(obj)
        mats = snap.all_activation_matrices()
        if not mats:
            raise ValueError(
                "activation_id dimension estimation needs per-layer "
                "activations — re-run the extractor with "
                "aspects=['activations']."
            )
        return mats

    if estimator == "trajectory_dimension":
        if not isinstance(obj, TrainingView):
            raise ValueError(
                "trajectory_dimension needs a TrainingView of weight "
                "snapshots — call pipeline.fit_training(...)."
            )
        return obj.weight_trajectory()

    raise ValueError(
        f"Unknown estimator '{estimator}'. "
        "Valid: 'activation_id', 'trajectory_dimension'."
    )


# ── Supplementary hooks the pipeline reads off a TrainingView ────────────────


def extract_loss_values(obj: ModelSnapshot | TrainingView) -> np.ndarray | None:
    """Return losses for ``method='ph_loss'`` / ``'loss_difference'``.

    Prefers the ``(T, n_eval)`` **per-sample** loss matrix (captured when the
    training extractor was given ``loss_eval_data``) so the Dupuis et al. (2023)
    pseudo-metric ``mean_s|ell_i,s - ell_j,s|`` is used.  Falls back to the
    ``(T,)`` scalar per-step loss otherwise.
    """
    if isinstance(obj, TrainingView):
        per_sample = obj.per_sample_loss_trajectory()
        if per_sample is not None:
            return per_sample
        return obj.loss_trajectory()
    return None
