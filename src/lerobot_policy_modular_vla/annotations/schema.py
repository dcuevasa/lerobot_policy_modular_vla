"""Language and subtask annotation schema conforming to LeRobot v3.1 and lerobot-dataset-visualizer."""

from dataclasses import dataclass, field
from typing import Any


PERSISTENT_STYLES = {"subtask", "plan", "memory", "task_aug"}
EVENT_ONLY_STYLES = {"vqa", "interjection"}
KNOWN_STYLES = PERSISTENT_STYLES | EVENT_ONLY_STYLES


@dataclass
class LanguageAtom:
    """An individual language atom conforming to the LeRobot dataset language schema."""

    role: str = "assistant"  # "assistant" or "user"
    content: str | None = None  # Subtask name (e.g. 'pick_mug'), plan summary, or VQA query/answer
    style: str | None = "subtask"  # "subtask", "plan", "vqa", "interjection", "memory"
    timestamp: float = 0.0  # Timestamp in seconds from episode start
    camera: str | None = None  # Target camera key (e.g. 'observation.images.cam_high') for view-dependent atoms
    tool_calls: list[dict[str, Any]] | None = None  # Structured tool calls (e.g. bounding box or say action)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "style": self.style,
            "timestamp": round(float(self.timestamp), 4),
            "camera": self.camera,
            "tool_calls": self.tool_calls,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LanguageAtom":
        return cls(
            role=data.get("role", "assistant"),
            content=data.get("content"),
            style=data.get("style"),
            timestamp=float(data.get("timestamp", 0.0)),
            camera=data.get("camera"),
            tool_calls=data.get("tool_calls"),
        )


@dataclass
class EpisodeAnnotations:
    """Container for an episode's annotations."""

    episode_index: int = 0
    atoms: list[LanguageAtom] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "atoms": [a.to_dict() for a in self.atoms],
        }
