"""Dataset Annotation Recorder for LeRobot Datasets and Dataset Visualizer.

Records high-level VLM plans, subtask execution intervals, and bounding box / mask
affordances as standard LeRobot language atoms, persisting them to
``meta/lerobot_annotations.json`` so they can be inspected in
``lerobot-dataset-visualizer`` and parsed by ``split_dataset.py``.
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any

from ..perception.types import AffordanceDetection
from ..vlm.base import PlanResult, PlanStep
from .schema import EVENT_ONLY_STYLES, PERSISTENT_STYLES, EpisodeAnnotations, LanguageAtom

logger = logging.getLogger(__name__)


class DatasetAnnotationRecorder:
    """Records VLM plans, subtask spans, and affordance bounding boxes into standard LeRobot format."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.episode_index: int = 0
        self.current_episode_annotations = EpisodeAnnotations(episode_index=0)
        self.all_episodes: dict[int, EpisodeAnnotations] = {}
        self.episode_start_time: float | None = None
        self.current_step_count: int = 0

    def start_episode(self, episode_index: int = 0) -> None:
        """Initializes recording for a new episode."""
        self.episode_index = episode_index
        self.current_episode_annotations = EpisodeAnnotations(episode_index=episode_index)
        self.all_episodes[episode_index] = self.current_episode_annotations
        self.episode_start_time = time.perf_counter()
        self.current_step_count = 0
        logger.info(f"[AnnotationRecorder] Started recording episode {episode_index}.")

    def record_plan(self, plan: PlanResult, timestamp: float | None = None) -> None:
        """Records initial high-level plan as a persistent 'plan' atom."""
        ts = self._resolve_timestamp(timestamp)
        steps_summary = " -> ".join([f"{s.skill_name}({s.target_object})" for s in plan.steps])
        content = f"Plan: {steps_summary}. Reasoning: {plan.reasoning}"

        atom = LanguageAtom(
            role="assistant",
            content=content,
            style="plan",
            timestamp=ts,
            camera=None,
            tool_calls=None,
        )
        self.current_episode_annotations.atoms.append(atom)
        logger.info(f"[AnnotationRecorder] Recorded plan atom at t={ts:.2f}s.")

    def record_subtask_start(
        self,
        step: PlanStep,
        timestamp: float | None = None,
    ) -> None:
        """Records the beginning of a subtask as a 'subtask' atom."""
        ts = self._resolve_timestamp(timestamp)
        atom = LanguageAtom(
            role="assistant",
            content=step.skill_name,
            style="subtask",
            timestamp=ts,
            camera=None,
            tool_calls=None,
        )
        self.current_episode_annotations.atoms.append(atom)
        logger.info(f"[AnnotationRecorder] Recorded subtask atom '{step.skill_name}' at t={ts:.2f}s.")

    def record_affordance_detection(
        self,
        detection: AffordanceDetection,
        camera_key: str = "observation.images.cam_high",
        timestamp: float | None = None,
    ) -> None:
        """Records target object bounding box as a view-dependent 'vqa' atom."""
        ts = self._resolve_timestamp(timestamp)
        box = detection.box

        # Standard bounding box tool call format in LeRobot v3.1 / visualizer
        tool_call = {
            "type": "bbox",
            "box": [
                round(float(box.ymin), 4),
                round(float(box.xmin), 4),
                round(float(box.ymax), 4),
                round(float(box.xmax), 4),
            ],
            "label": detection.label,
            "score": round(float(detection.score), 4),
        }

        atom = LanguageAtom(
            role="assistant",
            content=f"Target: {detection.label}",
            style="vqa",
            timestamp=ts,
            camera=camera_key,
            tool_calls=[tool_call],
        )
        self.current_episode_annotations.atoms.append(atom)

    def step(self) -> None:
        """Increments frame/step counter."""
        self.current_step_count += 1

    def _resolve_timestamp(self, timestamp: float | None = None) -> float:
        if timestamp is not None:
            return float(timestamp)
        # Use discrete frame timestamp based on fps
        return float(self.current_step_count) / max(1.0, float(self.fps))

    def save_to_json(self, destination: str | Path) -> Path:
        """Writes recorded annotations to ``meta/lerobot_annotations.json``.

        Args:
            destination: Path to JSON file or directory containing the dataset.

        Returns:
            Absolute Path to written annotations file.
        """
        dest_path = Path(destination)
        if dest_path.is_dir() or not dest_path.suffix:
            # Point to meta/lerobot_annotations.json inside dataset dir
            dest_file = dest_path / "meta" / "lerobot_annotations.json"
        else:
            dest_file = dest_path

        dest_file.parent.mkdir(parents=True, exist_ok=True)

        # Build payload matching lerobot-dataset-visualizer v2 schema
        episodes_payload: dict[str, Any] = {}
        for ep_idx, ep_ann in self.all_episodes.items():
            ep_dict = ep_ann.to_dict()

            # Also synthesize legacy subtask intervals for split_dataset.py compatibility
            subtask_atoms = [a for a in ep_ann.atoms if a.style == "subtask"]
            legacy_subtasks = []
            for i, st in enumerate(subtask_atoms):
                start_t = st.timestamp
                end_t = (
                    subtask_atoms[i + 1].timestamp
                    if i + 1 < len(subtask_atoms)
                    else float(self.current_step_count) / max(1.0, float(self.fps))
                )
                legacy_subtasks.append({
                    "label": st.content,
                    "start": start_t,
                    "end": max(start_t + 0.1, end_t),
                })

            ep_dict["subtasks"] = legacy_subtasks
            episodes_payload[str(ep_idx)] = ep_dict

        payload = {
            "version": 2,
            "schema": {
                "persistent_styles": sorted(PERSISTENT_STYLES),
                "event_styles": sorted(EVENT_ONLY_STYLES),
            },
            "episodes": episodes_payload,
        }

        with open(dest_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"[AnnotationRecorder] Saved LeRobot annotations to: {dest_file.resolve()}")
        return dest_file.resolve()
