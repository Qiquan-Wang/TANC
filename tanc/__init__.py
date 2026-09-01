"""TANC — Topological Data Analysis for Neural Networks.

A composable pipeline with an optional upstream extraction stage:

    Neural network model
         ↓
    [model_extractor]  →  ModelSnapshot / TrainingView
         ↓
    [graph_builder]    →  GraphBundle
         ↓
    [topo_tools]       →  TopoResult
         ↓
    [visualisation]    →  Figures

Quick start (full pipeline from raw model)
------------------------------------------
>>> from tanc.model_extractor import extract_model, extract_training
>>> snapshot = extract_model(my_trained_model, X_test)
>>> from tanc import TDAPipeline
>>> pipe   = TDAPipeline.from_paper("watanabe2021")
>>> result = pipe.fit(snapshot.weight_matrices())
>>> result.plot("diagram")

Quick start (graph_builder only)
---------------------------------
>>> from tanc import TDAPipeline
>>> pipe = TDAPipeline.from_paper("watanabe2021")
>>> result = pipe.fit(weight_matrices)
>>> result.plot("diagram")
"""

from tanc.pipeline.pipeline import TDAPipeline
from tanc.pipeline.paper_presets import (
    PAPER_PRESETS, describe_preset, list_presets,
)
from tanc.pipeline.mapper_study import MapperStudy, MapperStudyResult
from tanc.topo_tools.mapper_sweep import MapperGrid
from tanc.model_extractor.population import train_population, TrainedPopulation
from tanc.graph_builder._bundle import GraphBundle
from tanc.topo_tools._result import TopoResult, PersistenceResult, MapperView
from tanc.model_extractor._snapshot import ModelSnapshot, TrainingView
from tanc.model_extractor._inspector import ModelInfo, LayerInfo
from tanc.model_extractor.extractors import ModelExtractor, TrainingExtractor
from tanc._help import tour, help

__version__ = "0.1.0"

__all__ = [
    # Pipeline
    "TDAPipeline",
    "PAPER_PRESETS",
    "describe_preset",
    "list_presets",
    # Sweeps — Mapper across a parameter grid, with the cover reported alongside
    "MapperStudy",
    "MapperStudyResult",
    "MapperGrid",
    "train_population",
    "TrainedPopulation",
    # Graph builder output
    "GraphBundle",
    # Topo tools output
    "TopoResult",
    "PersistenceResult",
    "MapperView",
    # Model extractor output
    "ModelSnapshot",
    "TrainingView",
    "ModelInfo",
    "LayerInfo",
    # Model extractor classes (preferred entry points)
    "ModelExtractor",
    "TrainingExtractor",
    # Onboarding
    "tour",
    "help",
]
