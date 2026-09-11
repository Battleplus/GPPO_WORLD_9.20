"""GPPO world-model contracts and implementation."""

from .contracts import (
    EvidenceItem,
    ExecutionRecord,
    GraphSnapshot,
    Transition,
    WorldModelInput,
)
from .registry import FEATURE_REGISTRY, SCHEMA_VERSION
from .consequence_model import (
    ActionConsequenceWorldModel,
    ConsequenceModelConfig,
    ConsequenceTarget,
    consequence_loss,
)

__all__ = [
    "EvidenceItem",
    "ExecutionRecord",
    "FEATURE_REGISTRY",
    "GraphSnapshot",
    "SCHEMA_VERSION",
    "Transition",
    "WorldModelInput",
    "ActionConsequenceWorldModel",
    "ConsequenceModelConfig",
    "ConsequenceTarget",
    "consequence_loss",
]
