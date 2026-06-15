"""Prototype-based champion icon recognition pipeline."""

from __future__ import annotations

from typing import Any

from .augmentation import AugmentationSummary, create_noise_data
from .config import RecognitionConfig
from .dataset import ChampionRecognitionError


_LAZY_EXPORTS = {
    "TrainingSummary": (".train", "TrainingSummary"),
    "PostProcessingReport": (".post_processing_ai", "PostProcessingReport"),
    "build_and_save_prototypes": (".prototypes", "build_and_save_prototypes"),
    "analyze_post_training": (".post_processing_ai", "analyze_post_training"),
    "evaluate_saved_recognizer": (".evaluation", "evaluate_saved_recognizer"),
    "predict_champion": (".predict", "predict_champion"),
    "train_recognizer": (".train", "train_recognizer"),
}

__all__ = [
    "AugmentationSummary",
    "ChampionRecognitionError",
    "PostProcessingReport",
    "RecognitionConfig",
    "analyze_post_training",
    "TrainingSummary",
    "build_and_save_prototypes",
    "create_noise_data",
    "evaluate_saved_recognizer",
    "predict_champion",
    "train_recognizer",
]


def __getattr__(name: str) -> Any:
    """Import torch-dependent recognizer pieces only when requested."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
