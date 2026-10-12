"""Perception and affordance detection data structures."""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class BoundingBox:
    """Bounding box in normalized coordinates [0.0, 1.0] or pixel coordinates."""

    ymin: float
    xmin: float
    ymax: float
    xmax: float
    normalized: bool = True

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.ymin + self.ymax) / 2.0, (self.xmin + self.xmax) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_pixel(self, img_height: int, img_width: int) -> "BoundingBox":
        """Convert normalized coords [0, 1] to pixel coords."""
        if not self.normalized:
            return self
        return BoundingBox(
            ymin=self.ymin * img_height,
            xmin=self.xmin * img_width,
            ymax=self.ymax * img_height,
            xmax=self.xmax * img_width,
            normalized=False,
        )

    def to_normalized(self, img_height: int, img_width: int) -> "BoundingBox":
        """Convert pixel coords to normalized coords [0, 1]."""
        if self.normalized:
            return self
        h = max(1, img_height)
        w = max(1, img_width)
        return BoundingBox(
            ymin=self.ymin / h,
            xmin=self.xmin / w,
            ymax=self.ymax / h,
            xmax=self.xmax / w,
            normalized=True,
        )

    def pad(self, margin_ratio: float = 0.15, img_height: int = 1, img_width: int = 1) -> "BoundingBox":
        """Expand the bounding box symmetrically by a margin ratio."""
        dw = self.width * margin_ratio
        dh = self.height * margin_ratio
        if self.normalized:
            return BoundingBox(
                ymin=max(0.0, self.ymin - dh),
                xmin=max(0.0, self.xmin - dw),
                ymax=min(1.0, self.ymax + dh),
                xmax=min(1.0, self.xmax + dw),
                normalized=True,
            )
        return BoundingBox(
            ymin=max(0.0, self.ymin - dh),
            xmin=max(0.0, self.xmin - dw),
            ymax=min(float(img_height), self.ymax + dh),
            xmax=min(float(img_width), self.xmax + dw),
            normalized=False,
        )


@dataclass
class AffordanceDetection:
    """Represents a detected object affordance (bounding box, mask, score, and label)."""

    id: int
    label: str
    box: BoundingBox
    score: float = 1.0
    mask: np.ndarray | None = None  # 2D boolean or uint8 mask (H, W)
    keypoints: list[tuple[float, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
