"""Local Hugging Face Transformers Vision-Language Model Client."""

import json
import logging
from typing import Any
import numpy as np
from PIL import Image
import torch

from ..perception.types import AffordanceDetection
from .base import BaseVLMClient, PlanResult, PlanStep, VerificationResult
from .mock_client import MockVLMClient

logger = logging.getLogger(__name__)


class TransformersVLMClient(BaseVLMClient):
    """Client for running local open VLMs (SmolVLM, Qwen2-VL) via Hugging Face Transformers."""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM-Instruct",
        device: str | None = None,
        load_in_8bit: bool = False,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.processor = None
        self.model = None
        self.mock_fallback = MockVLMClient()
        self._init_model()

    def _init_model(self):
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device != "cpu" else None,
            )
            if self.device == "cpu":
                self.model.to("cpu")
            self.model.eval()
            logger.info(f"Loaded local Transformers VLM '{self.model_name}' on '{self.device}'")
        except Exception as e:
            logger.warning(f"Could not load local transformers model '{self.model_name}': {e}. Using Mock fallback.")
            self.model = None

    def generate_plan(
        self,
        task: str,
        detected_objects: list[AffordanceDetection],
        available_skills: list[dict[str, Any]],
        image: np.ndarray | torch.Tensor | Image.Image | None = None,
    ) -> PlanResult:
        if self.model is None or self.processor is None:
            return self.mock_fallback.generate_plan(task, detected_objects, available_skills, image)

        prompt = self._build_planning_prompt(task, detected_objects, available_skills)
        pil_img = self._to_pil_image(image) if image is not None else Image.new("RGB", (224, 224), (0, 0, 0))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            prompt_len = inputs["input_ids"].shape[1]
            generated_texts = self.processor.batch_decode(
                generated_ids[:, prompt_len:], skip_special_tokens=True
            )
            raw_out = generated_texts[0]

        # Extract JSON
        try:
            start = raw_out.find("{")
            end = raw_out.rfind("}")
            clean_json = raw_out[start : end + 1] if start != -1 and end != -1 else raw_out
            data = json.loads(clean_json)
            steps = [
                PlanStep(
                    step_index=i,
                    skill_name=s.get("skill_name", f"skill_{i}"),
                    target_object=s.get("target_object", "object"),
                    target_detection_id=s.get("target_detection_id"),
                    description=s.get("description", ""),
                    max_steps=int(s.get("max_steps", 100)),
                )
                for i, s in enumerate(data.get("plan", []))
            ]
            return PlanResult(task=task, steps=steps, reasoning=data.get("reasoning", ""), raw_response=raw_out)
        except Exception:
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
