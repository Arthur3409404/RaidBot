from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class YoloDetector:
    def __init__(self, model_path: str | Path):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("Ultralytics is required to use YoloDetector.") from exc

        self.model = YOLO(str(model_path))

    def _prepare_source(self, source: str | Path | np.ndarray):
        if isinstance(source, (str, Path)):
            return str(source)

        array = np.asarray(source)
        if array.ndim == 2:
            return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if array.ndim == 3 and array.shape[2] == 1:
            return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return array.astype(np.uint8)

    def predict(self, image_path: str | Path | np.ndarray, *, conf: float = 0.25, imgsz: int = 640) -> list[Detection]:
        source = self._prepare_source(image_path)
        results = self.model.predict(source=source, conf=float(conf), imgsz=int(imgsz), verbose=False)
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
            classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
            for coords, score, class_id in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = [float(v) for v in coords]
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=float(score),
                        class_id=int(class_id),
                    )
                )
        return detections
