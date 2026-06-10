"""UTC-based push-rank scheduling helpers."""

from __future__ import annotations

from datetime import datetime, timezone


UTC = timezone.utc
CLASSIC_ARENA_PUSH_MODE = "classic_arena"
TAGTEAM_ARENA_PUSH_MODE = "tagteam_arena"


def _coerce_utc(now_utc: datetime) -> datetime:
    if now_utc.tzinfo is None:
        return now_utc.replace(tzinfo=UTC)
    return now_utc.astimezone(UTC)


def is_classic_arena_pushrank_window(now_utc: datetime) -> bool:
    """Return True during the Sunday 18:00-21:00 UTC push window."""
    current = _coerce_utc(now_utc)
    return current.weekday() == 6 and 18 <= current.hour < 21


def is_tagteam_arena_pushrank_window(now_utc: datetime) -> bool:
    """Return True during the daily 21:00-00:00 UTC push window."""
    current = _coerce_utc(now_utc)
    return 21 <= current.hour < 24


def select_pushrank_mode(
    *,
    classic_enabled: bool,
    classic_window_active: bool,
    tagteam_enabled: bool,
    tagteam_window_active: bool,
) -> str | None:
    """Return the forced arena mode using the configured priority order."""
    if classic_enabled and classic_window_active:
        return CLASSIC_ARENA_PUSH_MODE
    if tagteam_enabled and tagteam_window_active:
        return TAGTEAM_ARENA_PUSH_MODE
    return None


def get_forced_pushrank_mode(
    now_utc: datetime,
    *,
    classic_enabled: bool,
    tagteam_enabled: bool,
) -> str | None:
    """Return the arena mode that should override normal scheduling, if any."""
    return select_pushrank_mode(
        classic_enabled=classic_enabled,
        classic_window_active=is_classic_arena_pushrank_window(now_utc),
        tagteam_enabled=tagteam_enabled,
        tagteam_window_active=is_tagteam_arena_pushrank_window(now_utc),
    )
