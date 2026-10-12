"""Live Telemetry and Overlay Visualization for Record / Rollout Streams.

Draws live visual overlays (bounding boxes, masks, current plan step, active skill,
step countdown, and VLM reasoning status) directly onto video frames so they are
visible in video recordings, Rerun streams, and user dashboards.
"""

from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .types import AffordanceDetection, BoundingBox


class TelemetryVisualizer:
    """Renders real-time HUD annotations and affordance detections onto observation frames."""

    def __init__(
        self,
        enabled: bool = True,
        hud_opacity: float = 0.75,
        font_size: int = 14,
    ):
        self.enabled = enabled
        self.hud_opacity = hud_opacity
        self.font_size = font_size

    def render_overlay(
        self,
        image: np.ndarray | torch.Tensor | Image.Image,
        active_detection: AffordanceDetection | None = None,
        all_detections: list[AffordanceDetection] | None = None,
        task: str = "",
        active_skill: str = "",
        current_step_idx: int = 0,
        total_steps: int = 0,
        skill_step_count: int = 0,
        max_skill_steps: int = 0,
        status: str = "RUNNING",
    ) -> np.ndarray:
        """Draws bounding boxes, masks, and HUD onto an image frame.

        Returns:
            RGB uint8 numpy array with overlays burned in.
        """
        # Convert to PIL Image
        pil_img = self._to_pil_image(image)
        w, h = pil_img.size

        # Create overlay layer for transparent HUD
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 1. Draw inactive detections in semi-transparent gray/blue
        if all_detections:
            for det in all_detections:
                if active_detection is not None and det.id == active_detection.id:
                    continue
                pbox = det.box.to_pixel(h, w)
                draw.rectangle(
                    [pbox.xmin, pbox.ymin, pbox.xmax, pbox.ymax],
                    outline=(120, 160, 220, 180),
                    width=2,
                )
                draw.text(
                    (pbox.xmin + 4, max(0, pbox.ymin - 16)),
                    f"{det.label} #{det.id} ({det.score:.2f})",
                    fill=(180, 200, 255, 220),
                )

        # 2. Draw ACTIVE target affordance with bold green / yellow box and glow
        if active_detection is not None:
            pbox = active_detection.box.to_pixel(h, w)
            # Active box outline
            draw.rectangle(
                [pbox.xmin, pbox.ymin, pbox.xmax, pbox.ymax],
                outline=(0, 255, 128, 255),
                width=3,
            )
            # Center target crosshair
            cy = (pbox.ymin + pbox.ymax) / 2.0
            cx = (pbox.xmin + pbox.xmax) / 2.0
            r = 6
            draw.line([(cx - r, cy), (cx + r, cy)], fill=(0, 255, 128, 255), width=2)
            draw.line([(cx, cy - r), (cx, cy + r)], fill=(0, 255, 128, 255), width=2)

            # Target tag
            tag_text = f"★ TARGET: {active_detection.label.upper()} (ID {active_detection.id})"
            draw.rectangle(
                [pbox.xmin, max(0, pbox.ymin - 22), pbox.xmin + len(tag_text) * 8 + 10, pbox.ymin],
                fill=(0, 180, 80, 220),
            )
            draw.text(
                (pbox.xmin + 4, max(0, pbox.ymin - 20)),
                tag_text,
                fill=(255, 255, 255, 255),
            )

        # 3. Draw Top-Left HUD Banner
        hud_h = 68
        hud_w = min(w, 480)
        draw.rectangle(
            [0, 0, hud_w, hud_h],
            fill=(15, 20, 30, int(255 * self.hud_opacity)),
        )
        draw.rectangle([0, hud_h - 2, hud_w, hud_h], fill=(0, 180, 255, 255))  # Cyan accent border

        # Text lines
        task_line = f"GOAL: {task[:45]}..." if len(task) > 45 else f"GOAL: {task or 'N/A'}"
        step_line = (
            f"PLAN [{current_step_idx + 1}/{max(1, total_steps)}]: "
            f"Skill='{active_skill}' | Status: {status}"
        )
        progress_line = (
            f"HORIZON: {skill_step_count}/{max_skill_steps} steps "
            f"({'%.1f' % (100.0 * skill_step_count / max(1, max_skill_steps))}%)"
        )

        draw.text((8, 6), task_line, fill=(255, 220, 100, 255))
        draw.text((8, 26), step_line, fill=(100, 220, 255, 255))
        draw.text((8, 46), progress_line, fill=(200, 255, 200, 255))

        # Composite HUD overlay onto original image
        base_rgba = pil_img.convert("RGBA")
        combined = Image.alpha_composite(base_rgba, overlay)
        rgb_img = combined.convert("RGB")
        return np.array(rgb_img)

    def _to_pil_image(self, image: np.ndarray | torch.Tensor | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.ndim == 4:
                t = t[0]
            if t.shape[0] in (1, 3):
                t = t.permute(1, 2, 0)
            arr = t.numpy()
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            if arr.shape[-1] == 1:
                arr = arr.squeeze(-1)
            return Image.fromarray(arr)
        elif isinstance(image, np.ndarray):
            if image.ndim == 4:
                image = image[0]
            if image.shape[0] in (1, 3) and image.shape[2] not in (1, 3):
                image = np.transpose(image, (1, 2, 0))
            if image.dtype != np.uint8 and image.max() <= 1.0:
                image = (image * 255.0).astype(np.uint8)
            return Image.fromarray(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
