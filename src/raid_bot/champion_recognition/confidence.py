"""Shared confidence rules for nearest-prototype recognition."""

from __future__ import annotations

from .config import RecognitionConfig


def is_confident_match(
    similarity: float,
    margin: float,
    config: RecognitionConfig,
    *,
    min_similarity_threshold: float | None = None,
) -> bool:
    """Return whether a nearest-prototype match is confident enough to accept."""
    similarity_threshold = config.min_similarity_threshold
    if min_similarity_threshold is not None:
        similarity_threshold = min_similarity_threshold
    if similarity < similarity_threshold:
        return False
    if config.min_margin_threshold > 0 and margin < config.min_margin_threshold:
        return False
    return True
