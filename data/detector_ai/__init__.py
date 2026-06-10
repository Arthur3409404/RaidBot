"""Compatibility aliases for detector AI helpers now in ``raid_bot.detector_ai``."""

from __future__ import annotations

from importlib import import_module
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_ALIASES = {
    "dataset_tools": "raid_bot.detector_ai.dataset_tools",
    "label_pictures": "raid_bot.detector_ai.label_pictures",
    "train_yolo": "raid_bot.detector_ai.train_yolo",
    "yolo_detector": "raid_bot.detector_ai.yolo_detector",
}

for legacy_name, target_name in _ALIASES.items():
    sys.modules[f"{__name__}.{legacy_name}"] = import_module(target_name)

_detector_ai = import_module("raid_bot.detector_ai")

AnnotationBox = _detector_ai.AnnotationBox
DatasetItem = _detector_ai.DatasetItem
export_yolo_dataset = _detector_ai.export_yolo_dataset
load_annotation_items = _detector_ai.load_annotation_items

__all__ = [
    "AnnotationBox",
    "DatasetItem",
    "export_yolo_dataset",
    "load_annotation_items",
]
