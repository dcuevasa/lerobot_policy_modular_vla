"""Dynamic Model Swapper and Resource Manager.

Implements the thesis architecture's model swapping mechanism:
- 'swap_on_demand': frees previous skill policy from GPU memory and mounts the
  new checkpoint dynamically to maintain a tiny VRAM footprint on consumer hardware.
- 'cache_all': keeps all micro-experts initialized simultaneously if memory permits.
- 'lazy_load': loads policies on first access and retains them in an LRU/memory cache.
- 'coral_switching' / 'lora_switching': keeps a frozen base backbone resident on GPU
  and dynamically switches only task-specific LoRA adapters / action heads (CORAL paradigm).
"""

import gc
import logging
from typing import Any
import torch
from torch import nn

from .schema import SkillDefinition

logger = logging.getLogger(__name__)


class MockSkillPolicy(nn.Module):
    """Fallback policy producing compliant kinematic actions for testing."""

    def __init__(self, action_dim: int = 6, name: str = "mock"):
        super().__init__()
        self.action_dim = action_dim
        self.name = name
        self.step_idx = 0

    def reset(self):
        self.step_idx = 0

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        b = 1
        dev = torch.device("cpu")
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                b = v.shape[0] if v.ndim > 0 else 1
                dev = v.device
                break

        self.step_idx += 1
        t = float(self.step_idx) * 0.05
        action = torch.zeros((b, self.action_dim), dtype=torch.float32, device=dev)
        action[:, 0] = 0.1 * torch.sin(torch.tensor(t))
        action[:, -1] = 1.0 if "pick" in self.name else 0.0
        return action

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        return torch.tensor(0.0), {}


class ModelSwapper:
    """Manages loading, caching, swapping, and LoRA/CORAL switching of micro-expert models."""

    def __init__(
        self,
        strategy: str = "swap_on_demand",  # "swap_on_demand", "cache_all", "lazy_load", "coral_switching"
        device: str | None = None,
        default_action_dim: int = 6,
        base_backbone_path: str | None = None,
    ):
        self.strategy = strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.default_action_dim = default_action_dim
        self.base_backbone_path = base_backbone_path
        self._loaded_models: dict[str, Any] = {}
        self._active_skill_name: str | None = None
        self._shared_backbone: Any = None
        self._active_lora_adapter: str | None = None

    def get_or_load(self, skill: SkillDefinition) -> Any:
        """Retrieves or activates the policy for the given skill according to the active strategy."""
        # 1. CORAL / LoRA Head Switching
        if self.strategy in ("coral_switching", "lora_switching", "coral"):
            return self._switch_coral_lora(skill)

        # 2. Dynamic Model Swapping (thesis default)
        elif self.strategy == "swap_on_demand":
            if self._active_skill_name == skill.name and skill.name in self._loaded_models:
                return self._loaded_models[skill.name]

            self.unload_all()
            logger.info(f"[ModelSwapper] Dynamic Swap to '{skill.name}' (type: {skill.policy_type})...")
            policy = self._load_policy(skill)
            self._loaded_models[skill.name] = policy
            self._active_skill_name = skill.name
            return policy

        # 3. Cache All or Lazy Load
        elif self.strategy in ("cache_all", "lazy_load"):
            if skill.name in self._loaded_models:
                self._active_skill_name = skill.name
                return self._loaded_models[skill.name]

            logger.info(f"[ModelSwapper] Loading and caching '{skill.name}'...")
            policy = self._load_policy(skill)
            self._loaded_models[skill.name] = policy
            self._active_skill_name = skill.name
            return policy

        else:
            raise ValueError(f"Unknown swapping strategy: {self.strategy}")

    def _switch_coral_lora(self, skill: SkillDefinition) -> Any:
        """CORAL / LoRA switching mechanism: Keeps base model frozen on GPU, activates task LoRA head."""
        logger.info(f"[ModelSwapper:CORAL] Activating LoRA adapter/head for skill '{skill.name}'...")

        # Load shared backbone if not already resident
        if self._shared_backbone is None:
            backbone_path = skill.base_model_path or self.base_backbone_path or skill.pretrained_path
            logger.info(f"[ModelSwapper:CORAL] Initializing shared frozen backbone from '{backbone_path}'...")
            self._shared_backbone = self._load_policy(skill)

        # Check if PEFT or adapter switching is available
        model = self._shared_backbone
        adapter_path = skill.lora_adapter_path or skill.pretrained_path

        if hasattr(model, "set_adapter") and adapter_path:
            try:
                adapter_name = skill.name
                if hasattr(model, "peft_config") and adapter_name not in model.peft_config:
                    model.load_adapter(adapter_path, adapter_name=adapter_name)
                model.set_adapter(adapter_name)
                self._active_lora_adapter = adapter_name
                logger.info(f"[ModelSwapper:CORAL] Switched PEFT adapter to '{adapter_name}'.")
            except Exception as e:
                logger.warning(f"[ModelSwapper:CORAL] Could not set adapter via PEFT: {e}")

        self._active_skill_name = skill.name
        return self._shared_backbone

    def _load_policy(self, skill: SkillDefinition) -> Any:
        """Instantiates policy for a skill, using LeRobot if available, or fallback."""
        if skill.policy_type == "mock" or not skill.pretrained_path:
            return MockSkillPolicy(action_dim=self.default_action_dim, name=skill.name).to(self.device)

        try:
            from lerobot.policies.factory import get_policy_class, make_policy
            if skill.pretrained_path:
                policy_cls = get_policy_class(skill.policy_type)
                policy = policy_cls.from_pretrained(skill.pretrained_path)
                policy.to(self.device)
                policy.eval()
                return policy
        except Exception as e:
            logger.warning(
                f"[ModelSwapper] Could not load checkpoint from '{skill.pretrained_path}' ({e}). "
                f"Falling back to MockSkillPolicy for skill '{skill.name}'."
            )

        return MockSkillPolicy(action_dim=self.default_action_dim, name=skill.name).to(self.device)

    def unload_all(self) -> None:
        """Clears loaded models and frees GPU memory."""
        if self._loaded_models:
            self._loaded_models.clear()
            self._active_skill_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
