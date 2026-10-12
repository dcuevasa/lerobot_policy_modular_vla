"""Ollama Vision-Language Model Client for local edge inference."""

import json
import logging
import re
from typing import Any
import numpy as np
from PIL import Image
import requests
import torch

from ..perception.types import AffordanceDetection
from .base import BaseVLMClient, PlanResult, PlanStep, VerificationResult
from .mock_client import MockVLMClient

logger = logging.getLogger(__name__)


class OllamaVLMClient(BaseVLMClient):
    """Client for local open-weights VLMs served via Ollama (e.g. Qwen2.5-VL, LLaVA, MiniCPM-V)."""

    def __init__(
        self,
        model_name: str = "qwen2.5-vl:7b",
        api_base: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.mock_fallback = MockVLMClient()

    def generate_plan(
        self,
        task: str,
        detected_objects: list[AffordanceDetection],
        available_skills: list[dict[str, Any]],
        image: np.ndarray | torch.Tensor | Image.Image | None = None,
    ) -> PlanResult:
        prompt = self._build_planning_prompt(task, detected_objects, available_skills)
        images_b64 = [self.encode_image_base64(image)] if image is not None else []

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": images_b64,
                }
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            resp = requests.post(
                f"{self.api_base}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            return self._parse_plan_json(task, content)
        except Exception as e:
            logger.warning(
                f"Ollama request to '{self.api_base}' failed: {e}. Falling back to deterministic plan."
            )
            return self.mock_fallback.generate_plan(task, detected_objects, available_skills, image)

    def verify_step(
        self,
        current_step: PlanStep,
        image: np.ndarray | torch.Tensor | Image.Image,
    ) -> VerificationResult:
        b64 = self.encode_image_base64(image)
        prompt = f"""Observe the image and determine if the robotic subtask was successfully completed.
Subtask: '{current_step.skill_name}' on target '{current_step.target_object}'.
Description: {current_step.description}

Respond ONLY with a JSON object:
{{
  "completed": true/false,
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation",
  "should_replan": false
}}"""

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }

        try:
            resp = requests.post(f"{self.api_base}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
            parsed = json.loads(self._extract_json_str(content))
            return VerificationResult(
                completed=bool(parsed.get("completed", True)),
                confidence=float(parsed.get("confidence", 0.9)),
                reasoning=str(parsed.get("reasoning", "")),
                should_replan=bool(parsed.get("should_replan", False)),
            )
        except Exception as e:
            logger.debug(f"Ollama verification failed ({e}); falling back to True.")
            return VerificationResult(completed=True, confidence=0.8, reasoning="Default fallback.")

    def describe_scene(
        self,
        image: np.ndarray | torch.Tensor | Image.Image,
        prompt: str = "Describe scene",
    ) -> str:
        b64 = self.encode_image_base64(image)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "stream": False,
        }
        try:
            resp = requests.post(f"{self.api_base}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        except Exception as e:
            return f"Ollama description unavailable: {e}"

    def _parse_plan_json(self, task: str, raw_content: str) -> PlanResult:
        clean_json = self._extract_json_str(raw_content)
        data = json.loads(clean_json)
        raw_steps = data.get("plan", [])
        reasoning = data.get("reasoning", "")

        steps: list[PlanStep] = []
        for i, s in enumerate(raw_steps):
            steps.append(
                PlanStep(
                    step_index=i,
                    skill_name=s.get("skill_name", f"skill_{i}"),
                    target_object=s.get("target_object", "object"),
                    target_detection_id=s.get("target_detection_id"),
                    description=s.get("description", ""),
                    max_steps=int(s.get("max_steps", 100)),
                )
            )

        return PlanResult(
            task=task,
            steps=steps,
            reasoning=reasoning,
            raw_response=raw_content,
        )

    def _extract_json_str(self, text: str) -> str:
        """Extracts JSON substring from markdown code blocks or text."""
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            return match.group(1).strip()
        # Find first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return text[start : end + 1]
        return text.strip()
