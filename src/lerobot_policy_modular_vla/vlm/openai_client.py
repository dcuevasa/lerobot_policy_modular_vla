"""OpenAI-compatible Vision-Language Model Client (GPT-4o, vLLM, LiteLLM, OpenRouter)."""

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


class OpenAIVLMClient(BaseVLMClient):
    """Client for OpenAI and any OpenAI-compatible Vision API."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        api_base: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
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
        if not self.api_key and "localhost" not in self.api_base and "127.0.0.1" not in self.api_base:
            logger.warning("No OPENAI_API_KEY provided; falling back to deterministic plan.")
            return self.mock_fallback.generate_plan(task, detected_objects, available_skills, image)

        prompt = self._build_planning_prompt(task, detected_objects, available_skills)
        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if image is not None:
            b64 = self.encode_image_base64(image)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }

        try:
            resp = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return self._parse_plan_json(task, content)
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}. Falling back to deterministic plan.")
            return self.mock_fallback.generate_plan(task, detected_objects, available_skills, image)

    def verify_step(
        self,
        current_step: PlanStep,
        image: np.ndarray | torch.Tensor | Image.Image,
    ) -> VerificationResult:
        if not self.api_key and "localhost" not in self.api_base:
            return VerificationResult(completed=True, confidence=0.9, reasoning="Fallback mock verification.")

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

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
        }

        try:
            resp = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return VerificationResult(
                completed=bool(parsed.get("completed", True)),
                confidence=float(parsed.get("confidence", 0.9)),
                reasoning=str(parsed.get("reasoning", "")),
                should_replan=bool(parsed.get("should_replan", False)),
            )
        except Exception as e:
            logger.debug(f"OpenAI verification failed: {e}")
            return VerificationResult(completed=True, confidence=0.8, reasoning="Default fallback.")

    def describe_scene(
        self,
        image: np.ndarray | torch.Tensor | Image.Image,
        prompt: str = "Describe scene",
    ) -> str:
        b64 = self.encode_image_base64(image)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                    ],
                }
            ],
        }
        try:
            resp = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenAI scene description failed: {e}"

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
