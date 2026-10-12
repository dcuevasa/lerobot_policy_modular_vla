"""Modular / Hierarchical VLA Policy implementation for LeRobot."""

import logging
from pathlib import Path
from typing import Any
import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy

from .annotations.recorder import DatasetAnnotationRecorder
from .configuration_modular_vla import ModularVLAConfig
from .perception.conditioning import AffordanceConditioner
from .perception.detector import make_detector
from .perception.visualization import TelemetryVisualizer
from .planner.coordinator import PlanCoordinator
from .skills.router import SkillRouter
from .skills.schema import SkillCatalog, SkillDefinition
from .skills.swapper import ModelSwapper
from .vlm.factory import make_vlm_client

logger = logging.getLogger(__name__)


class ModularVLAPolicy(PreTrainedPolicy):
    """Hierarchical Modular Mixture-of-Experts VLA Policy.

    Functions as a plug-and-play LeRobot policy, replacing monolithic VLAs with:
    - High-level VLM agent coordinator (Ollama, OpenAI, Gemini, Anthropic, or Mock).
    - Dynamic visual affordance detection (bounding boxes & masks).
    - Specialized motor micro-experts (ACT, Diffusion, SmolVLA, etc.).
    - Model Swapping Manager (low-VRAM anti-forgetting guarantees).
    - Real-time video recording visual overlays (bounding boxes + HUD telemetry).
    """

    config_class = ModularVLAConfig
    name = "modular_vla"

    def __init__(self, config: ModularVLAConfig, **kwargs: Any):
        super().__init__(config)
        self.config = config

        # 1. Initialize VLM Coordinator Client
        vlm_cfg = config.vlm
        self.vlm_client = make_vlm_client(
            provider=vlm_cfg.provider,
            model_name=vlm_cfg.model_name,
            api_key=vlm_cfg.api_key,
            api_base=vlm_cfg.api_base,
            temperature=vlm_cfg.temperature,
            device=config.device,
        )

        # 2. Initialize Visual Affordance Detector
        det_cfg = config.detector
        self.detector = make_detector(
            engine=det_cfg.engine,
            model_name=det_cfg.model_name,
            conf_threshold=det_cfg.conf_threshold,
            device=det_cfg.device or config.device,
            mode=det_cfg.mode,
        )

        # 3. Initialize Visual Affordance Conditioner
        self.conditioner = AffordanceConditioner(
            default_mode="crop",
            crop_margin=0.15,
        )

        # 4. Initialize Telemetry Visualizer (recording overlays)
        self.visualizer = TelemetryVisualizer(
            enabled=True,
            hud_opacity=0.8,
        )

        # 5. Initialize Skills Catalog and Model Swapper
        skills_list = []
        for s_name, s_data in config.skills.items():
            if isinstance(s_data, dict):
                skills_list.append(SkillDefinition.from_dict(s_data))
            elif isinstance(s_data, SkillDefinition):
                skills_list.append(s_data)

        self.catalog = SkillCatalog(skills_list)
        self.swapper = ModelSwapper(
            strategy=config.swapping_strategy,
            device=config.device,
            default_action_dim=config.action_dim,
        )
        self.skill_router = SkillRouter(
            catalog=self.catalog,
            swapper=self.swapper,
        )

        # 6. Initialize Dataset Annotation Recorder (v3.1 & lerobot-dataset-visualizer compatible)
        self.annotation_recorder = DatasetAnnotationRecorder(fps=config.fps)

        # 7. Initialize Master Plan Coordinator
        self.coordinator = PlanCoordinator(
            vlm_client=self.vlm_client,
            detector=self.detector,
            conditioner=self.conditioner,
            visualizer=self.visualizer,
            skill_router=self.skill_router,
            primary_camera_key=config.primary_camera_key,
            burn_overlays_into_recording=config.burn_overlays_into_recording,
            verification_enabled=config.verification_enabled,
            annotation_recorder=self.annotation_recorder,
        )

        # Set default task from config
        self.coordinator.set_task(config.default_task)

        # Tracking state
        self._current_episode_index: int = 0
        self._has_started: bool = False
        self.latest_info: dict[str, Any] = {}

    def reset(self, episode_index: int | None = None) -> None:
        """Called whenever the robot or environment resets.

        Args:
            episode_index: Optional index of the episode. If omitted, increments sequentially.
        """
        if episode_index is not None:
            self._current_episode_index = episode_index
            self._has_started = True
        elif self._has_started:
            self._current_episode_index += 1
        else:
            self._has_started = True
            self._current_episode_index = 0

        self.coordinator.reset(episode_index=self._current_episode_index)
        self.coordinator.set_task(self.config.default_task)
        self.latest_info.clear()

    def save_annotations(self, destination: str | Path | None = None) -> Path:
        """Saves recorded plan, subtask, and affordance bounding box annotations.

        Outputs meta/lerobot_annotations.json conforming to lerobot-dataset-visualizer
        and split_dataset.py formats.

        Args:
            destination: Path to json file or dataset directory. If None, uses config.annotations_dir.

        Returns:
            Path to saved meta/lerobot_annotations.json file.
        """
        dest = destination or self.config.annotations_dir
        if not dest:
            raise ValueError("No destination path provided and config.annotations_dir is not set.")
        return self.coordinator.save_annotations(dest)

    def set_task(self, task: str) -> None:
        """Sets or updates the natural language goal."""
        self.coordinator.set_task(task)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Selects the next action step to command the robot."""
        self.eval()

        # If batch contains language instruction, update task dynamically
        if "task" in batch and isinstance(batch["task"], str):
            self.coordinator.set_task(batch["task"])

        action, info = self.coordinator.step(batch)
        self.latest_info = info
        return action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Returns action chunks from the active micro-expert."""
        if hasattr(self.skill_router.active_policy, "predict_action_chunk"):
            return self.skill_router.active_policy.predict_action_chunk(batch)
        action = self.select_action(batch)
        return action.unsqueeze(1)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Forward pass for training / validation."""
        # The modular coordinator relies on independently trained micro-models
        loss = torch.tensor(0.0, requires_grad=True)
        return loss, {"modular_loss": 0.0}

    def get_optim_params(self) -> list[dict[str, Any]]:
        """Returns trainable parameters."""
        return [{"params": [p for p in self.parameters() if p.requires_grad]}]

    def supports_text_generation(self) -> bool:
        """Declares VLM text generation capability to LeRobot."""
        return True

    def generate_text(self, batch: dict[str, Any]) -> str:
        """Generates text from VLM explaining the scene, current plan, or next subtask."""
        img = self.coordinator._get_primary_image(batch)
        if self.coordinator.current_plan:
            steps_desc = ", ".join([f"{i+1}:{s.skill_name}({s.target_object})" for i, s in enumerate(self.coordinator.current_plan.steps)])
            return f"Plan: {steps_desc}. Status: {self.coordinator.status} (Step {self.coordinator.current_step_idx + 1})"
        return self.vlm_client.describe_scene(img)
