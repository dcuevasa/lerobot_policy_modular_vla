"""Skill Router and Micro-Expert Dispatcher."""

import logging
from typing import Any
import torch

from .schema import SkillCatalog, SkillDefinition
from .swapper import ModelSwapper

logger = logging.getLogger(__name__)


class SkillRouter:
    """Routes execution and sensory observations to the active specialized micro-expert."""

    def __init__(
        self,
        catalog: SkillCatalog,
        swapper: ModelSwapper,
    ):
        self.catalog = catalog
        self.swapper = swapper
        self.active_skill_name: str | None = None
        self.active_policy: Any = None
        self.skill_step_count: int = 0

    def set_active_skill(self, skill_name: str) -> None:
        """Swaps or retrieves the requested skill policy and resets its step count."""
        skill = self.catalog.get(skill_name)
        if skill is None:
            raise KeyError(f"Skill '{skill_name}' not found in registered catalogue: {[s.name for s in self.catalog.list_skills()]}")

        self.active_policy = self.swapper.get_or_load(skill)
        self.active_skill_name = skill_name
        self.skill_step_count = 0

        # Reset inner policy action queue if supported
        if hasattr(self.active_policy, "reset"):
            try:
                self.active_policy.reset()
            except Exception as e:
                logger.debug(f"Inner policy reset error: {e}")

        logger.info(f"[SkillRouter] Active skill set to '{skill_name}' (horizon: {skill.max_steps} steps).")

    def get_active_skill_definition(self) -> SkillDefinition | None:
        if self.active_skill_name:
            return self.catalog.get(self.active_skill_name)
        return None

    def select_action(self, conditioned_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Dispatches conditioned observation to active micro-expert."""
        if self.active_policy is None:
            raise RuntimeError("No active skill set in SkillRouter. Call set_active_skill() first.")

        action = self.active_policy.select_action(conditioned_batch)
        self.skill_step_count += 1
        return action

    def is_current_skill_finished(self, max_steps: int | None = None) -> bool:
        """Checks if active skill has consumed its horizon budget."""
        skill = self.get_active_skill_definition()
        budget = max_steps or (skill.max_steps if skill else 100)
        return self.skill_step_count >= budget

    def reset(self) -> None:
        """Resets the router state."""
        self.skill_step_count = 0
        if self.active_policy is not None and hasattr(self.active_policy, "reset"):
            try:
                self.active_policy.reset()
            except Exception:
                pass
