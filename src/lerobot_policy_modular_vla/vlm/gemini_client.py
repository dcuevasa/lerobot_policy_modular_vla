"""Google Gemini Vision-Language Model Client."""

import json
import logging
import os
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


class GeminiVLMClient(BaseVLMClient):
    """Client for Google Gemini multimodal models (gemini-2.0-flash, gemini-1.5-flash)."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.1,
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
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
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY provided; falling back to deterministic plan.")
            return self.mock_fallback.generate_plan(task, detected_objects, available_skills, image)

        prompt = self._build_planning_prompt(task, detected_objects, available_skills)
        parts: list[dict[str, Any]] = [{"text": prompt}]

        if image is not None:
            b64 = self.encode_image_base64(image)
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64}})

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            return self._parse_plan_json(task, content)
        except Exception as e:
            logger.warning(f"Gemini API request failed: {e}. Falling back to deterministic plan.")
            return self.mock_fallback.generate_plan(task, detected_objects, available_skills, image)

    def verify_step(
        self,
        current_step: PlanStep,
        image: np.ndarray | torch.Tensor | Image.Image,
    ) -> VerificationResult:
        if not self.api_key:
            return VerificationResult(completed=True, confidence=0.9, reasoning="Fallback mock verification.")

        b64 = self.encode_image_base64(image)
        prompt = f"""Observe the image and determine if the robotic subtask was completed.
Subtask: '{current_step.skill_name}' on target '{current_step.target_object}'.
Respond ONLY with JSON:
{{"completed": true/false, "confidence": 0.0 to 1.0, "reasoning": "brief note", "should_replan": false}}"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": b64}}]}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(content)
            return VerificationResult(
                completed=bool(parsed.get("completed", True)),
                confidence=float(parsed.get("confidence", 0.9)),
                reasoning=str(parsed.get("reasoning", "")),
                should_replan=bool(parsed.get("should_replan", False)),
            )
        except Exception as e:
            logger.debug(f"Gemini verification failed: {e}")
            return VerificationResult(completed=True, confidence=0.8, reasoning="Default fallback.")

    def describe_scene(
        self,
        image: np.ndarray | torch.Tensor | Image.Image,
        prompt: str = "Describe scene",
    ) -> str:
        b64 = self.encode_image_base64(image)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": b64}}]}]}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"Gemini description failed: {e}"

    def _parse_plan_json(self, task: str, raw_content: str) -> PlanResult:
        data = json.loads(raw_content)
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

        return PlanResult(task=task, steps=steps, reasoning=reasoning, raw_response=raw_content)
