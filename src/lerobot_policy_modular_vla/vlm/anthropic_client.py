"""Anthropic Claude Vision-Language Model Client."""

import json
import logging
import os
from typing import Any
import numpy as np
from PIL import Image
import requests
import torch

from ..perception.types import AffordanceDetection
from .base import BaseVLMClient, PlanResult, PlanStep, VerificationResult
from .mock_client import MockVLMClient

logger = logging.getLogger(__name__)


class AnthropicVLMClient(BaseVLMClient):
    """Client for Anthropic Claude multimodal models (Claude 3.5 Sonnet / Haiku)."""

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: str | None = None,
        temperature: float = 0.1,
        timeout: float = 30.0,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
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
            logger.warning("No ANTHROPIC_API_KEY provided; falling back to deterministic plan.")
            return self.mock_fallback.generate_plan(task, detected_objects, available_skills, image)

        prompt = self._build_planning_prompt(task, detected_objects, available_skills)
        content: list[dict[str, Any]] = []

        if image is not None:
            b64 = self.encode_image_base64(image)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })
        content.append({"type": "text", "text": prompt})

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
        }

        try:
            resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"]
            # Extract JSON substring
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start : end + 1]
            data = json.loads(text)
            raw_steps = data.get("plan", [])
            steps = [
                PlanStep(
                    step_index=i,
                    skill_name=s.get("skill_name", f"skill_{i}"),
                    target_object=s.get("target_object", "object"),
                    target_detection_id=s.get("target_detection_id"),
                    description=s.get("description", ""),
                    max_steps=int(s.get("max_steps", 100)),
                )
                for i, s in enumerate(raw_steps)
            ]
            return PlanResult(task=task, steps=steps, reasoning=data.get("reasoning", ""), raw_response=text)
        except Exception as e:
            logger.warning(f"Anthropic API call failed: {e}. Falling back to deterministic plan.")
            return self.mock_fallback.generate_plan(task, detected_objects, available_skills, image)

    def verify_step(
        self,
        current_step: PlanStep,
        image: np.ndarray | torch.Tensor | Image.Image,
    ) -> VerificationResult:
        return self.mock_fallback.verify_step(current_step, image)

    def describe_scene(
        self,
        image: np.ndarray | torch.Tensor | Image.Image,
        prompt: str = "Describe scene",
    ) -> str:
        return self.mock_fallback.describe_scene(image, prompt)
