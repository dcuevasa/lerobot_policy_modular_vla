"""LeRobot Modular / Hierarchical VLA Policy Plugin.

A plug-and-play modular architecture for LeRobot that replaces monolithic VLAs
with a VLM agent orchestrator, visual affordance grounding, and dynamically swappable
lightweight micro-experts (ACT, Diffusion, SmolVLA).
"""

from .configuration_modular_vla import ModularVLAConfig, VLMConfig, DetectorConfig
from .modeling_modular_vla import ModularVLAPolicy
from .perception.types import BoundingBox, AffordanceDetection
from .perception.detector import BaseAffordanceDetector, MockAffordanceDetector, YOLODetector, make_detector
from .perception.conditioning import AffordanceConditioner
from .perception.visualization import TelemetryVisualizer
from .skills.schema import SkillDefinition, SkillCatalog
from .skills.swapper import ModelSwapper, MockSkillPolicy
from .skills.router import SkillRouter
from .planner.coordinator import PlanCoordinator
from .vlm.base import BaseVLMClient, PlanStep, PlanResult, VerificationResult
from .vlm.mock_client import MockVLMClient
from .vlm.factory import make_vlm_client

# Aliases for convenience
HierarchicalVLAConfig = ModularVLAConfig
HierarchicalVLAPolicy = ModularVLAPolicy

__all__ = [
    # Core LeRobot Interface
    "ModularVLAConfig",
    "ModularVLAPolicy",
    "HierarchicalVLAConfig",
    "HierarchicalVLAPolicy",
    "VLMConfig",
    "DetectorConfig",
    # Perception
    "BoundingBox",
    "AffordanceDetection",
    "BaseAffordanceDetector",
    "MockAffordanceDetector",
    "YOLODetector",
    "make_detector",
    "AffordanceConditioner",
    "TelemetryVisualizer",
    # Skills & Swapping
    "SkillDefinition",
    "SkillCatalog",
    "ModelSwapper",
    "MockSkillPolicy",
    "SkillRouter",
    "PlanCoordinator",
    # VLM
    "BaseVLMClient",
    "MockVLMClient",
    "PlanStep",
    "PlanResult",
    "VerificationResult",
    "make_vlm_client",
]
