"""Configuration for Modular / Hierarchical VLA Policy."""

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@dataclass
class VLMConfig:
    """Configuration for high-level VLM coordinator."""

    provider: str = "mock"  # "mock", "ollama", "openai", "gemini", "anthropic", "transformers"
    model_name: str = "qwen2.5-vl:7b"
    api_key: str | None = None
    api_base: str | None = None  # e.g. "http://localhost:11434" for Ollama
    temperature: float = 0.1
    timeout: float = 30.0


@dataclass
class DetectorConfig:
    """Configuration for visual affordance perception engine."""

    engine: str = "mock"  # "mock", "yolo", "yolo_world", "yolo_seg"
    model_name: str | None = None  # e.g. "yolov8n.pt", "yolov8s-worldv2.pt", "yolov8n-seg.pt"
    conf_threshold: float = 0.25
    mode: str = "bbox"  # "bbox", "seg", "world"
    device: str | None = None


@dataclass
class ModularVLAConfig(PreTrainedConfig):
    """Configuration class for the Modular Hierarchical Mixture-of-Experts VLA Policy.

    Deconstructs end-to-end monolithic VLAs into:
    1. High-level VLM agent coordinator (Ollama, OpenAI, Gemini, Claude, local transformers, or mock).
    2. Real-time visual affordance grounding (YOLO BBox / Segmentation masks).
    3. Specialized micro-expert motor policies (ACT, Diffusion, SmolVLA, etc.) with dynamic swapping.
    4. Observation conditioning (crops/masks) so micro-experts manipulate spatial regions directly.
    5. Live recording overlays burning bounding boxes and planning HUD onto output videos.
    6. Future-ready support for CORAL / LoRA switching where base backbone stays frozen.
    """

    # VLM Coordinator Settings
    vlm: VLMConfig = field(default_factory=VLMConfig)

    # Perception / Affordance Grounding
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    # Catalogue of available micro-expert skills
    skills: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "pick_mug": {
                "name": "pick_mug",
                "description": "Grasps the mug/cup highlighted in the bounding box and lifts it",
                "policy_type": "mock",
                "pretrained_path": None,
                "target_concept": "cup",
                "max_steps": 85,
                "conditioning_mode": "crop",
                "crop_margin": 0.15,
            },
            "place_mug": {
                "name": "place_mug",
                "description": "Transfers and places the mug into the coffee machine dispenser",
                "policy_type": "mock",
                "pretrained_path": None,
                "target_concept": "coffee_machine",
                "max_steps": 105,
                "conditioning_mode": "crop",
                "crop_margin": 0.20,
            },
            "press_button": {
                "name": "press_button",
                "description": "Reaches out and depresses the brew switch/button",
                "policy_type": "mock",
                "pretrained_path": None,
                "target_concept": "button",
                "max_steps": 50,
                "conditioning_mode": "crop",
                "crop_margin": 0.10,
            },
        }
    )

    # Model Swapping Strategy:
    # - "swap_on_demand": dynamically unloads previous model and mounts new checkpoint (low VRAM)
    # - "cache_all": keeps all policies in memory
    # - "lazy_load": loads on first call and caches
    # - "coral_switching" / "lora_switching": keeps frozen backbone, switches only LoRA head
    swapping_strategy: str = "swap_on_demand"

    # Default task instruction
    default_task: str = "Prepare coffee: grasp the mug and place it under the coffee machine"

    # Primary camera stream for VLM and detections
    primary_camera_key: str = "observation.images.cam_high"

    # Whether to burn HUD and bounding boxes directly onto the recorded video stream
    burn_overlays_into_recording: bool = True

    # Subtask completion verification with VLM
    verification_enabled: bool = False

    # LeRobot Core Policy Specs
    n_obs_steps: int = 1
    action_dim: int = 6
    chunk_size: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.vlm, dict):
            self.vlm = VLMConfig(**self.vlm)
        if isinstance(self.detector, dict):
            self.detector = DetectorConfig(**self.detector)

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        pass

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


# Register aliases in LeRobot's PreTrainedConfig registry
PreTrainedConfig.register_subclass("modular_vla")(ModularVLAConfig)
PreTrainedConfig.register_subclass("hierarchical_vla")(ModularVLAConfig)
