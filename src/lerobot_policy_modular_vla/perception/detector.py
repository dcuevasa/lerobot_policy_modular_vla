"""Modular Affordance and Object Detectors."""

import abc
import logging
from typing import Any
import numpy as np
import torch

from .types import AffordanceDetection, BoundingBox

logger = logging.getLogger(__name__)


class BaseAffordanceDetector(abc.ABC):
    """Abstract base class for visual affordance detectors."""

    @abc.abstractmethod
    def detect(
        self,
        image: np.ndarray | torch.Tensor,
        candidate_labels: list[str] | None = None,
    ) -> list[AffordanceDetection]:
        """Detect objects / affordances in an image.

        Args:
            image: Image as RGB numpy array (H, W, 3) with values in [0, 255] or
                   torch.Tensor (C, H, W) or (B, C, H, W).
            candidate_labels: Optional list of text labels/concepts to search for.

        Returns:
            List of AffordanceDetection objects with bounding boxes, masks, and labels.
        """
        raise NotImplementedError

    def _ensure_numpy_image(self, image: np.ndarray | torch.Tensor) -> np.ndarray:
        """Converts tensor (C, H, W) or (B, C, H, W) to (H, W, C) uint8 numpy array."""
        if isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.ndim == 4:
                t = t[0]  # Take first batch item
            if t.shape[0] in (1, 3):
                t = t.permute(1, 2, 0)
            arr = t.numpy()
            if arr.max() <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            return arr
        elif isinstance(image, np.ndarray):
            if image.ndim == 4:
                image = image[0]
            if image.shape[0] in (1, 3) and image.shape[2] not in (1, 3):
                image = np.transpose(image, (1, 2, 0))
            if image.dtype != np.uint8 and image.max() <= 1.0:
                image = (image * 255.0).astype(np.uint8)
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")


class MockAffordanceDetector(BaseAffordanceDetector):
    """Deterministic mock detector for tests, simulations, and offline runs."""

    def __init__(self, predefined_detections: list[AffordanceDetection] | None = None):
        self.predefined_detections = predefined_detections

    def detect(
        self,
        image: np.ndarray | torch.Tensor,
        candidate_labels: list[str] | None = None,
    ) -> list[AffordanceDetection]:
        img = self._ensure_numpy_image(image)
        h, w = img.shape[:2]

        if self.predefined_detections is not None:
            return self.predefined_detections

        # Generate sensible default affordances based on candidate labels
        labels = candidate_labels or ["cup", "coffee_machine", "button"]
        detections: list[AffordanceDetection] = []

        # Default layout on a table:
        # cup at left-center, machine at right, button on top of machine
        layout = {
            "cup": BoundingBox(ymin=0.45, xmin=0.20, ymax=0.75, xmax=0.42, normalized=True),
            "mug": BoundingBox(ymin=0.45, xmin=0.20, ymax=0.75, xmax=0.42, normalized=True),
            "coffee_machine": BoundingBox(ymin=0.20, xmin=0.55, ymax=0.85, xmax=0.90, normalized=True),
            "machine": BoundingBox(ymin=0.20, xmin=0.55, ymax=0.85, xmax=0.90, normalized=True),
            "button": BoundingBox(ymin=0.28, xmin=0.62, ymax=0.38, xmax=0.72, normalized=True),
        }

        det_id = 0
        for lbl in labels:
            box = layout.get(lbl.lower())
            if box is None:
                # Default generic center box
                box = BoundingBox(ymin=0.3, xmin=0.3, ymax=0.7, xmax=0.7, normalized=True)

            # Generate synthetic mask
            pbox = box.to_pixel(h, w)
            mask = np.zeros((h, w), dtype=bool)
            y1, y2 = int(pbox.ymin), int(pbox.ymax)
            x1, x2 = int(pbox.xmin), int(pbox.xmax)
            mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = True

            detections.append(
                AffordanceDetection(
                    id=det_id,
                    label=lbl,
                    box=box,
                    score=0.95,
                    mask=mask,
                )
            )
            det_id += 1

        return detections


class YOLODetector(BaseAffordanceDetector):
    """YOLO detector supporting bounding boxes, segmentation, and open-vocabulary YOLO-World."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        device: str | None = None,
        mode: str = "bbox",  # "bbox", "seg", "world"
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(self.model_name)
            logger.info(f"Loaded YOLO model '{self.model_name}' on device '{self.device}'")
        except ImportError:
            logger.warning(
                "ultralytics package not installed. YOLODetector falling back to MockAffordanceDetector behavior."
            )
            self.model = None

    def detect(
        self,
        image: np.ndarray | torch.Tensor,
        candidate_labels: list[str] | None = None,
    ) -> list[AffordanceDetection]:
        if self.model is None:
            # Fallback mock detector if weights or ultralytics are unavailable
            mock = MockAffordanceDetector()
            return mock.detect(image, candidate_labels)

        img = self._ensure_numpy_image(image)
        h, w = img.shape[:2]

        # For YOLO-World open-vocabulary classes
        if candidate_labels and hasattr(self.model, "set_classes"):
            try:
                self.model.set_classes(candidate_labels)
            except Exception as e:
                logger.debug(f"Could not set YOLO-World classes: {e}")

        results = self.model.predict(
            source=img,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections: list[AffordanceDetection] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        boxes = r.boxes
        masks = getattr(r, "masks", None)
        names = r.names

        if boxes is None:
            return detections

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            label = names.get(cls_id, str(cls_id)) if names else str(cls_id)
            score = float(box.conf[0].item())

            # xyxy in pixels
            xyxy = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            bbox = BoundingBox(
                ymin=ymin / h,
                xmin=xmin / w,
                ymax=ymax / h,
                xmax=xmax / w,
                normalized=True,
            )

            mask_arr = None
            if masks is not None and len(masks) > i:
                m = masks[i].data[0].cpu().numpy()
                mask_arr = m.astype(bool)

            detections.append(
                AffordanceDetection(
                    id=i,
                    label=label,
                    box=bbox,
                    score=score,
                    mask=mask_arr,
                )
            )

        return detections


def make_detector(
    engine: str = "mock",
    model_name: str | None = None,
    conf_threshold: float = 0.25,
    device: str | None = None,
    mode: str = "bbox",
) -> BaseAffordanceDetector:
    """Detector factory function."""
    engine = engine.lower()
    if engine in ("mock", "dummy"):
        return MockAffordanceDetector()
    elif engine in ("yolo", "yolov8", "yolo_world", "yolo_seg"):
        model_path = model_name or ("yolov8n-seg.pt" if "seg" in engine else "yolov8n.pt")
        return YOLODetector(
            model_name=model_path,
            conf_threshold=conf_threshold,
            device=device,
            mode=mode,
        )
    else:
        logger.warning(f"Unknown detector engine '{engine}'. Falling back to MockAffordanceDetector.")
        return MockAffordanceDetector()
