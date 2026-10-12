"""Skill definition and catalogue schemas."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillDefinition:
    """Configuration for an individual specialized micro-expert policy."""

    name: str
    description: str
    policy_type: str = "act"  # "act", "diffusion", "smolvla", "mock"
    pretrained_path: str | None = None
    target_concept: str | None = None  # e.g. "cup", "coffee_machine", "button"
    max_steps: int = 100
    conditioning_mode: str = "crop"  # "crop", "mask_overlay", "bbox_overlay", "passthrough"
    crop_margin: float = 0.15
    camera_keys: list[str] | None = None  # Specific cameras to apply affordance conditioning
    termination_condition: str = "step_limit"  # "step_limit", "gripper_closed", "vlm_verify"
    policy_kwargs: dict[str, Any] = field(default_factory=dict)

    # CORAL / LoRA Switching Options
    lora_adapter_path: str | None = None  # Path to subtask LoRA adapter / head weights
    base_model_path: str | None = None  # Shared frozen backbone model path

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "policy_type": self.policy_type,
            "pretrained_path": self.pretrained_path,
            "target_concept": self.target_concept,
            "max_steps": self.max_steps,
            "conditioning_mode": self.conditioning_mode,
            "crop_margin": self.crop_margin,
            "camera_keys": self.camera_keys,
            "termination_condition": self.termination_condition,
            "policy_kwargs": self.policy_kwargs,
            "lora_adapter_path": self.lora_adapter_path,
            "base_model_path": self.base_model_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillDefinition":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            policy_type=data.get("policy_type", "act"),
            pretrained_path=data.get("pretrained_path"),
            target_concept=data.get("target_concept"),
            max_steps=int(data.get("max_steps", 100)),
            conditioning_mode=data.get("conditioning_mode", "crop"),
            crop_margin=float(data.get("crop_margin", 0.15)),
            camera_keys=data.get("camera_keys"),
            termination_condition=data.get("termination_condition", "step_limit"),
            policy_kwargs=data.get("policy_kwargs", {}),
            lora_adapter_path=data.get("lora_adapter_path"),
            base_model_path=data.get("base_model_path"),
        )


class SkillCatalog:
    """Manages the catalogue of registered micro-expert skills."""

    def __init__(self, skills: list[SkillDefinition] | None = None):
        self._skills: dict[str, SkillDefinition] = {}
        if skills:
            for s in skills:
                self.register(s)

    def register(self, skill: SkillDefinition) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def list_skills(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def to_vlm_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "policy_type": s.policy_type,
                "target_concept": s.target_concept,
                "max_steps": s.max_steps,
            }
            for s in self._skills.values()
        ]
