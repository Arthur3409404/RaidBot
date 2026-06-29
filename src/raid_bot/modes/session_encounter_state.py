from __future__ import annotations

import re
import time
import unicodedata
from typing import Callable

SESSION_LOST_ENCOUNTERS: dict[str, set[str]] = {
    "cursed_city": set(),
    "grim_forest": set(),
}

_SESSION_LOST_ESCAPE_GUARD: dict[str, dict[str, object]] = {
    "cursed_city": {"signature": None, "pressed_at": 0.0},
    "grim_forest": {"signature": None, "pressed_at": 0.0},
}

_VALID_AREAS = frozenset(SESSION_LOST_ENCOUNTERS)
_SAVE_DAILY_STATE_HOOK: Callable[[dict[str, set[str]]], None] | None = None


def _normalize_area(area: str | None) -> str | None:
    key = str(area or "").strip().lower()
    return key if key in _VALID_AREAS else None


def normalize_encounter_name(encounter_name: str | None) -> str:
    """Return a stable, OCR-friendly key for encounter comparison."""
    normalized = unicodedata.normalize("NFKD", str(encounter_name or ""))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.casefold().strip()
    normalized = re.sub(r"[^0-9a-z]+", " ", normalized)
    return " ".join(normalized.split())


def reset_session_lost_encounters():
    for area in SESSION_LOST_ENCOUNTERS:
        SESSION_LOST_ENCOUNTERS[area].clear()
    for guard in _SESSION_LOST_ESCAPE_GUARD.values():
        guard["signature"] = None
        guard["pressed_at"] = 0.0


def set_session_lost_encounter_persistence_hook(
    save_hook: Callable[[dict[str, set[str]]], None] | None,
) -> None:
    """Register a callback that persists the daily avoid snapshot."""
    global _SAVE_DAILY_STATE_HOOK
    _SAVE_DAILY_STATE_HOOK = save_hook


def clear_session_lost_encounter_persistence_hook() -> None:
    global _SAVE_DAILY_STATE_HOOK
    _SAVE_DAILY_STATE_HOOK = None


def _persist_snapshot() -> None:
    if _SAVE_DAILY_STATE_HOOK is None:
        return
    _SAVE_DAILY_STATE_HOOK(get_session_lost_encounter_snapshot())


def set_session_lost_encounter_snapshot(snapshot: dict[str, set[str] | list[str] | tuple[str, ...] | None]) -> None:
    """Replace the in-memory avoid snapshot from persisted daily state."""
    reset_session_lost_encounters()
    for area, values in (snapshot or {}).items():
        key = _normalize_area(area)
        if not key or not values:
            continue
        for value in values:
            normalized = normalize_encounter_name(value)
            if normalized:
                SESSION_LOST_ENCOUNTERS[key].add(normalized)


def add_session_lost_encounter(area: str | None, encounter_name: str | None) -> bool:
    key = _normalize_area(area)
    normalized = normalize_encounter_name(encounter_name)
    if not key or not normalized:
        return False
    SESSION_LOST_ENCOUNTERS[key].add(normalized)
    _persist_snapshot()
    return True


def match_session_lost_encounter(
    area: str | None,
    detected_encounter_name: str | None,
    *,
    threshold: float = 0.88,
) -> str | None:
    key = _normalize_area(area)
    normalized = normalize_encounter_name(detected_encounter_name)
    if not key or not normalized:
        return None

    candidates = SESSION_LOST_ENCOUNTERS[key]
    return normalized if normalized in candidates else None


def is_session_lost_encounter(
    area: str | None,
    detected_encounter_name: str | None,
    *,
    threshold: float = 0.88,
) -> bool:
    return match_session_lost_encounter(area, detected_encounter_name, threshold=threshold) is not None


def should_escape_session_lost_encounter(
    area: str | None,
    detected_encounter_name: str | None,
    *,
    threshold: float = 0.88,
    cooldown_seconds: float = 2.5,
) -> tuple[bool, str, str | None, bool]:
    """Return match state plus whether ESC should be pressed right now."""
    key = _normalize_area(area)
    normalized = normalize_encounter_name(detected_encounter_name)
    if not key or not normalized:
        return False, normalized, None, False

    match = match_session_lost_encounter(key, normalized, threshold=threshold)
    if match is None:
        return False, normalized, None, False

    guard = _SESSION_LOST_ESCAPE_GUARD[key]
    signature = match
    now = time.monotonic()
    last_signature = guard.get("signature")
    last_pressed_at = float(guard.get("pressed_at") or 0.0)
    if last_signature == signature and (now - last_pressed_at) < float(cooldown_seconds):
        return True, normalized, match, False

    guard["signature"] = signature
    guard["pressed_at"] = now
    return True, normalized, match, True


def get_session_lost_encounter_snapshot() -> dict[str, set[str]]:
    return {area: set(values) for area, values in SESSION_LOST_ENCOUNTERS.items()}
