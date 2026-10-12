"""Deterministic Mock VLM Client for offline testing, simulations, and unit tests."""

import json
import logging
from typing import Any
import numpy as np
from PIL import Image
import torch

from ..perception.types import AffordanceDetection
from .base import BaseVLMClient, PlanResult, PlanStep, VerificationResult

logger = logging.getLogger(__name__)


class MockVLMClient(BaseVLMClient):
    """Deterministic, rule-based planner that works offline without API keys or models."""

    def __init__(self, predefined_plan: list[PlanStep] | None = None):
        self.predefined_plan = predefined_plan

    def generate_plan(
        self,
        task: str,
        detected_objects: list[AffordanceDetection],
        available_skills: list[dict[str, Any]],
        image: np.ndarray | torch.Tensor | Image.Image | None = None,
    ) -> PlanResult:
        if self.predefined_plan is not None:
            return PlanResult(
                task=task,
                steps=self.predefined_plan,
                reasoning="Using predefined plan.",
                raw_response=json.dumps([s.__dict__ for s in self.predefined_plan]),
            )

        task_lower = task.lower()
        steps: list[PlanStep] = []
        skill_names = [s.get("name") for s in available_skills]
        label_map = {d.label.lower(): d for d in detected_objects}

        # Helper to find matching detection
        def find_det(keywords: list[str]) -> AffordanceDetection | None:
            for k in keywords:
                for lbl, d in label_map.items():
                    if k in lbl:
                        return d
            return detected_objects[0] if detected_objects else None

        # Coffee Task heuristics matching the thesis wiki
        if any(w in task_lower for w in ["coffee", "café", "cafe", "cup", "mug", "taza"]):
            # Step 1: Pick Mug
            pick_skill = next((s for s in skill_names if "pick" in s), skill_names[0] if skill_names else "pick")
            mug_det = find_det(["cup", "mug", "taza"])
            steps.append(
                PlanStep(
                    step_index=len(steps),
                    skill_name=pick_skill,
                    target_object=mug_det.label if mug_det else "cup",
                    target_detection_id=mug_det.id if mug_det else None,
                    description="Grasp and lift the cup/mug",
                    max_steps=100,
                )
            )

            # Step 2: Place Under Machine
            place_skill = next((s for s in skill_names if "place" in s), skill_names[min(1, len(skill_names)-1)])
            machine_det = find_det(["coffee_machine", "machine", "cafetera"])
            steps.append(
                PlanStep(
                    step_index=len(steps),
                    skill_name=place_skill,
                    target_object=machine_det.label if machine_det else "coffee_machine",
                    target_detection_id=machine_det.id if machine_det else None,
                    description="Place cup into coffee machine dispenser",
                    max_steps=120,
                )
            )

            # Step 3 (optional): Press button
            press_skill = next((s for s in skill_names if "button" in s or "press" in s), None)
            if press_skill:
                btn_det = find_det(["button", "switch", "boton"])
                steps.append(
                    PlanStep(
                        step_index=len(steps),
                        skill_name=press_skill,
                        target_object=btn_det.label if btn_det else "button",
                        target_detection_id=btn_det.id if btn_det else None,
                        description="Press brewing button",
                        max_steps=60,
                    )
                )

        else:
            # Generic decomposition: assign each available skill to the next detected object
            for i, skill in enumerate(available_skills):
                s_name = skill.get("name", f"skill_{i}")
                det = detected_objects[i % len(detected_objects)] if detected_objects else None
                steps.append(
                    PlanStep(
                        step_index=len(steps),
                        skill_name=s_name,
                        target_object=det.label if det else "target",
                        target_detection_id=det.id if det else None,
                        description=skill.get("description", ""),
                        max_steps=skill.get("max_steps", 100),
                    )
                )

        return PlanResult(
            task=task,
            steps=steps,
            reasoning=f"Decomposed '{task}' into {len(steps)} sequential subtasks using available skills.",
            raw_response=json.dumps([s.__dict__ for s in steps]),
        )

    def verify_step(
        self,
        current_step: PlanStep,
        image: np.ndarray | torch.Tensor | Image.Image,
    ) -> VerificationResult:
        # Mock always verifies success
        return VerificationResult(
            completed=True,
            confidence=0.95,
            reasoning=f"Mock verified completion of step {current_step.skill_name}.",
            should_replan=False,
        )

    def describe_scene(
        self,
        image: np.ndarray | torch.Tensor | Image.Image,
        prompt: str = "Describe scene",
    ) -> str:
        return "Robotic workspace containing target manipulation objects with spatial bounding box affordances."
