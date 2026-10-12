"""Visual Affordance Conditioning Module.

Conditions camera observations using detected bounding boxes or masks so that
underlying micro-experts (ACT, Diffusion, etc.) focus exclusively on the targeted
spatial region without requiring task-specific semantic conditioning.
"""

from typing import Any
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.functional as TF  # noqa: N812

from .types import AffordanceDetection, BoundingBox


class AffordanceConditioner:
    """Prepares and conditions sensory image observations for micro-experts."""

    def __init__(
        self,
        default_mode: str = "crop",  # "crop", "mask_overlay", "bbox_overlay", "passthrough"
        crop_margin: float = 0.15,
        target_size: tuple[int, int] | None = None,  # (H, W) or None to keep original
    ):
        self.default_mode = default_mode
        self.crop_margin = crop_margin
        self.target_size = target_size

    def condition_tensor(
        self,
        image_tensor: torch.Tensor,
        detection: AffordanceDetection | None,
        mode: str | None = None,
        margin: float | None = None,
    ) -> torch.Tensor:
        """Applies visual affordance conditioning to a torch.Tensor image.

        Args:
            image_tensor: Tensor of shape (B, C, H, W) or (C, H, W).
            detection: The active AffordanceDetection (box / mask).
            mode: Conditioning mode ('crop', 'mask_overlay', 'bbox_overlay', 'passthrough').
            margin: Optional crop margin expansion ratio.

        Returns:
            Conditioned image tensor with same batch/channel structure.
        """
        if detection is None:
            return image_tensor

        mode = mode or self.default_mode
        margin = margin if margin is not None else self.crop_margin

        if mode == "passthrough":
            return image_tensor

        has_batch = image_tensor.ndim == 4
        if not has_batch:
            # Add batch dim for uniform processing: (1, C, H, W)
            image_tensor = image_tensor.unsqueeze(0)

        b, c, h, w = image_tensor.shape

        if mode == "crop":
            # Crop bounding box with margin and resize to target_size or original (h, w)
            padded_box = detection.box.pad(margin_ratio=margin).to_pixel(h, w)
            y1 = int(max(0, min(h - 1, round(padded_box.ymin))))
            y2 = int(max(y1 + 1, min(h, round(padded_box.ymax))))
            x1 = int(max(0, min(w - 1, round(padded_box.xmin))))
            x2 = int(max(x1 + 1, min(w, round(padded_box.xmax))))

            cropped = image_tensor[:, :, y1:y2, x1:x2]

            out_size = self.target_size or (h, w)
            conditioned = F.interpolate(
                cropped,
                size=out_size,
                mode="bilinear",
                align_corners=False,
            )

        elif mode == "mask_overlay":
            if detection.mask is not None:
                mask_t = torch.from_numpy(detection.mask).to(
                    device=image_tensor.device, dtype=image_tensor.dtype
                )
                if mask_t.ndim == 2:
                    mask_t = mask_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                # Dim background by 70%, keep object full brightness
                bg_factor = 0.3
                alpha = mask_t + (1.0 - mask_t) * bg_factor
                conditioned = image_tensor * alpha
            else:
                conditioned = image_tensor

        elif mode == "bbox_overlay":
            # Draw visual bounding box onto tensor
            conditioned = image_tensor.clone()
            pbox = detection.box.to_pixel(h, w)
            y1, y2 = int(max(0, pbox.ymin)), int(min(h - 1, pbox.ymax))
            x1, x2 = int(max(0, pbox.xmin)), int(min(w - 1, pbox.xmax))
            thickness = max(1, int(min(h, w) * 0.01))

            # Highlight border in green (R=0, G=1, B=0)
            green = torch.tensor([0.0, 1.0, 0.0], device=image_tensor.device).view(1, 3, 1)
            # Top and bottom borders
            for t in range(thickness):
                if y1 + t < h:
                    conditioned[:, :3, y1 + t, x1:x2] = 0.5 * conditioned[:, :3, y1 + t, x1:x2] + 0.5 * green
                if y2 - t >= 0:
                    conditioned[:, :3, y2 - t, x1:x2] = 0.5 * conditioned[:, :3, y2 - t, x1:x2] + 0.5 * green
                if x1 + t < w:
                    conditioned[:, :3, y1:y2, x1 + t] = 0.5 * conditioned[:, :3, y1:y2, x1 + t] + 0.5 * green
                if x2 - t >= 0:
                    conditioned[:, :3, y1:y2, x2 - t] = 0.5 * conditioned[:, :3, y1:y2, x2 - t] + 0.5 * green

        else:
            conditioned = image_tensor

        if not has_batch:
            conditioned = conditioned.squeeze(0)

        return conditioned

    def condition_observation_batch(
        self,
        batch: dict[str, Any],
        detection: AffordanceDetection | None,
        target_camera_keys: list[str] | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Conditions image keys in an observation dictionary.

        Args:
            batch: Dictionary containing observation tensors (e.g. observation.images.cam_high).
            detection: Active detected object affordance.
            target_camera_keys: List of keys to apply conditioning to. If None, applies to all
                                keys containing 'image'.
            mode: Conditioning mode override.

        Returns:
            Shallow copy of batch with conditioned image tensors.
        """
        out_batch = dict(batch)
        if detection is None:
            return out_batch

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                should_condition = False
                if target_camera_keys is not None:
                    should_condition = k in target_camera_keys
                elif "image" in k.lower():
                    should_condition = True

                if should_condition:
                    out_batch[k] = self.condition_tensor(v, detection, mode=mode)

        # Inject normalized bounding box coordinates into batch metadata
        box = detection.box
        out_batch["target_affordance_bbox"] = torch.tensor(
            [box.ymin, box.xmin, box.ymax, box.xmax],
            dtype=torch.float32,
            device=next((v.device for v in batch.values() if isinstance(v, torch.Tensor)), torch.device("cpu")),
        )

        return out_batch
