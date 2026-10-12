"""High-Level Plan Coordinator and Control Loop Engine."""

import logging
from pathlib import Path
from typing import Any
import numpy as np
import torch

from ..annotations.recorder import DatasetAnnotationRecorder
from ..perception.conditioning import AffordanceConditioner
from ..perception.detector import BaseAffordanceDetector
from ..perception.types import AffordanceDetection
from ..perception.visualization import TelemetryVisualizer
from ..skills.router import SkillRouter
from ..vlm.base import BaseVLMClient, PlanResult, PlanStep

logger = logging.getLogger(__name__)


class PlanCoordinator:
    """Orchestrates perception, VLM reasoning, visual affordance conditioning, and micro-expert dispatch."""

    def __init__(
        self,
        vlm_client: BaseVLMClient,
        detector: BaseAffordanceDetector,
        conditioner: AffordanceConditioner,
        visualizer: TelemetryVisualizer,
        skill_router: SkillRouter,
        primary_camera_key: str = "observation.images.cam_high",
        burn_overlays_into_recording: bool = True,
        verification_enabled: bool = False,
        annotation_recorder: DatasetAnnotationRecorder | None = None,
    ):
        self.vlm_client = vlm_client
        self.detector = detector
        self.conditioner = conditioner
        self.visualizer = visualizer
        self.skill_router = skill_router
        self.primary_camera_key = primary_camera_key
        self.burn_overlays_into_recording = burn_overlays_into_recording
        self.verification_enabled = verification_enabled
        self.annotation_recorder = annotation_recorder or DatasetAnnotationRecorder()

        # Dynamic Execution State
        self.current_task: str = ""
        self.current_plan: PlanResult | None = None
        self.current_step_idx: int = 0
        self.latest_detections: list[AffordanceDetection] = []
        self.active_detection: AffordanceDetection | None = None
        self.status: str = "IDLE"  # "IDLE", "PLANNING", "EXECUTING", "VERIFYING", "COMPLETED"

    def reset(self, episode_index: int = 0) -> None:
        """Resets coordinator state for a new episode."""
        self.current_plan = None
        self.current_step_idx = 0
        self.latest_detections.clear()
        self.active_detection = None
        self.status = "IDLE"
        self.skill_router.reset()
        if self.annotation_recorder:
            self.annotation_recorder.start_episode(episode_index=episode_index)

    def set_task(self, task: str) -> None:
        """Sets or updates high-level instruction."""
        if task != self.current_task:
            self.current_task = task
            self.current_plan = None
            self.current_step_idx = 0
            self.status = "IDLE"

    def plan_or_replan(self, image: torch.Tensor | np.ndarray) -> None:
        """Executes VLM planning given current scene observation."""
        self.status = "PLANNING"
        # 1. Run affordance detector on visual scene
        available_skills = self.skill_router.catalog.to_vlm_summary()
        candidate_labels = [s.get("target_concept") for s in available_skills if s.get("target_concept")]
        self.latest_detections = self.detector.detect(image, candidate_labels=candidate_labels)

        # 2. Query VLM coordinator for decomposition
        logger.info(f"[PlanCoordinator] Requesting VLM plan for goal: '{self.current_task}'...")
        plan = self.vlm_client.generate_plan(
            task=self.current_task,
            detected_objects=self.latest_detections,
            available_skills=available_skills,
            image=image,
        )
        self.current_plan = plan
        self.current_step_idx = 0

        if plan.steps:
            first_step = plan.steps[0]
            self.skill_router.set_active_skill(first_step.skill_name)
            self._update_active_detection(first_step)
            self.status = "EXECUTING"
            logger.info(f"[PlanCoordinator] Plan established ({len(plan.steps)} steps). Reasoning: {plan.reasoning}")

            # Record plan and first subtask into LeRobot annotations schema
            if self.annotation_recorder:
                self.annotation_recorder.record_plan(plan)
                self.annotation_recorder.record_subtask_start(first_step)
                if self.active_detection:
                    self.annotation_recorder.record_affordance_detection(
                        self.active_detection, camera_key=self.primary_camera_key
                    )
        else:
            self.status = "COMPLETED"
            logger.warning("[PlanCoordinator] VLM returned empty plan.")

    def step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Main control loop execution step."""
        if self.annotation_recorder:
            self.annotation_recorder.step()

        # Extract primary camera image
        image_tensor = self._get_primary_image(batch)

        # 1. Initialize plan if not present
        if self.current_plan is None or not self.current_plan.steps:
            self.plan_or_replan(image_tensor)
            if self.status == "COMPLETED" or not self.current_plan or not self.current_plan.steps:
                return self._neutral_action(batch), {"status": "COMPLETED"}

        # 2. Check if current skill subtask has completed
        active_step = self._get_current_step()
        if active_step and self.skill_router.is_current_skill_finished(max_steps=active_step.max_steps):
            logger.info(f"[PlanCoordinator] Subtask '{active_step.skill_name}' finished budget.")
            self._transition_next_step(image_tensor)

        # Check if entire plan finished
        if self.status == "COMPLETED":
            return self._neutral_action(batch), {"status": "COMPLETED"}

        active_step = self._get_current_step()
        if active_step is None:
            self.status = "COMPLETED"
            return self._neutral_action(batch), {"status": "COMPLETED"}

        # 3. Perception & Affordance Tracking
        candidate_labels = [active_step.target_object] if active_step.target_object else None
        self.latest_detections = self.detector.detect(image_tensor, candidate_labels=candidate_labels)
        self._update_active_detection(active_step)

        # 4. Affordance Conditioning (cropper / mask / bbox overlay)
        active_skill_def = self.skill_router.get_active_skill_definition()
        cond_mode = active_skill_def.conditioning_mode if active_skill_def else "crop"
        margin = active_skill_def.crop_margin if active_skill_def else 0.15

        conditioned_batch = self.conditioner.condition_observation_batch(
            batch=batch,
            detection=self.active_detection,
            target_camera_keys=active_skill_def.camera_keys if active_skill_def else [self.primary_camera_key],
            mode=cond_mode,
        )

        # 5. Burn Visual Overlay onto recording / debug frame
        vis_image = None
        if self.visualizer.enabled:
            vis_image = self.visualizer.render_overlay(
                image=image_tensor,
                active_detection=self.active_detection,
                all_detections=self.latest_detections,
                task=self.current_task,
                active_skill=active_step.skill_name,
                current_step_idx=self.current_step_idx,
                total_steps=len(self.current_plan.steps),
                skill_step_count=self.skill_router.skill_step_count,
                max_skill_steps=active_step.max_steps,
                status=self.status,
            )
            if self.burn_overlays_into_recording:
                overlay_tensor = torch.from_numpy(vis_image).permute(2, 0, 1).to(
                    device=image_tensor.device, dtype=image_tensor.dtype
                )
                if image_tensor.max() <= 1.0:
                    overlay_tensor = overlay_tensor / 255.0
                if image_tensor.ndim == 4:
                    overlay_tensor = overlay_tensor.unsqueeze(0)
                batch[self.primary_camera_key] = overlay_tensor

        # 6. Execute Active Micro-Expert Action
        action = self.skill_router.select_action(conditioned_batch)

        info = {
            "status": self.status,
            "task": self.current_task,
            "step_index": self.current_step_idx,
            "total_steps": len(self.current_plan.steps),
            "skill_name": active_step.skill_name,
            "target_object": active_step.target_object,
            "step_progress": self.skill_router.skill_step_count,
            "max_steps": active_step.max_steps,
            "visualized_frame": vis_image,
        }

        return action, info

    def _transition_next_step(self, image: torch.Tensor | np.ndarray) -> None:
        """Transitions to next plan step and records subtask annotation."""
        if self.verification_enabled:
            self.status = "VERIFYING"
            active_step = self._get_current_step()
            if active_step:
                ver = self.vlm_client.verify_step(active_step, image)
                logger.info(f"[PlanCoordinator] VLM Verification: completed={ver.completed}, notes='{ver.reasoning}'")

        self.current_step_idx += 1
        if self.current_plan and self.current_step_idx < len(self.current_plan.steps):
            next_step = self.current_plan.steps[self.current_step_idx]
            self.skill_router.set_active_skill(next_step.skill_name)
            self._update_active_detection(next_step)
            self.status = "EXECUTING"
            logger.info(f"[PlanCoordinator] Advanced to step [{self.current_step_idx + 1}/{len(self.current_plan.steps)}]: '{next_step.skill_name}'.")

            # Record subtask transition into LeRobot annotations schema
            if self.annotation_recorder:
                self.annotation_recorder.record_subtask_start(next_step)
                if self.active_detection:
                    self.annotation_recorder.record_affordance_detection(
                        self.active_detection, camera_key=self.primary_camera_key
                    )
        else:
            self.status = "COMPLETED"
            logger.info("[PlanCoordinator] All plan subtasks completed successfully.")

    def save_annotations(self, destination: str | Path) -> Path:
        """Saves recorded plan and subtask annotations to meta/lerobot_annotations.json."""
        if not self.annotation_recorder:
            raise RuntimeError("No annotation_recorder configured on PlanCoordinator.")
        return self.annotation_recorder.save_to_json(destination)

    def _update_active_detection(self, step: PlanStep) -> None:
        if not self.latest_detections:
            self.active_detection = None
            return

        if step.target_detection_id is not None:
            for det in self.latest_detections:
                if det.id == step.target_detection_id:
                    self.active_detection = det
                    return

        target_name = (step.target_object or "").lower()
        for det in self.latest_detections:
            if target_name in det.label.lower() or det.label.lower() in target_name:
                self.active_detection = det
                return

        self.active_detection = self.latest_detections[0]

    def _get_current_step(self) -> PlanStep | None:
        if self.current_plan and 0 <= self.current_step_idx < len(self.current_plan.steps):
            return self.current_plan.steps[self.current_step_idx]
        return None

    def _get_primary_image(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.primary_camera_key in batch:
            return batch[self.primary_camera_key]
        for k, v in batch.items():
            if "image" in k.lower() and isinstance(v, torch.Tensor):
                return v
        dev = next((v.device for v in batch.values() if isinstance(v, torch.Tensor)), torch.device("cpu"))
        return torch.zeros((1, 3, 224, 224), device=dev)

    def _neutral_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        dim = self.skill_router.swapper.default_action_dim
        b = 1
        dev = torch.device("cpu")
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                b = v.shape[0] if v.ndim > 0 else 1
                dev = v.device
                break
        return torch.zeros((b, dim), dtype=torch.float32, device=dev)
