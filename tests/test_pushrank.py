from datetime import datetime, timezone

from raid_bot.core import pushrank
from raid_bot.utils import file_tools


UTC = timezone.utc


def test_classic_arena_pushrank_window_boundaries():
    assert not pushrank.is_classic_arena_pushrank_window(datetime(2026, 6, 14, 17, 59, tzinfo=UTC))
    assert pushrank.is_classic_arena_pushrank_window(datetime(2026, 6, 14, 18, 0, tzinfo=UTC))
    assert pushrank.is_classic_arena_pushrank_window(datetime(2026, 6, 14, 20, 59, tzinfo=UTC))
    assert not pushrank.is_classic_arena_pushrank_window(datetime(2026, 6, 14, 21, 0, tzinfo=UTC))
    assert not pushrank.is_classic_arena_pushrank_window(datetime(2026, 6, 15, 18, 30, tzinfo=UTC))


def test_tagteam_arena_pushrank_window_boundaries():
    assert not pushrank.is_tagteam_arena_pushrank_window(datetime(2026, 6, 10, 20, 59, tzinfo=UTC))
    assert pushrank.is_tagteam_arena_pushrank_window(datetime(2026, 6, 10, 21, 0, tzinfo=UTC))
    assert pushrank.is_tagteam_arena_pushrank_window(datetime(2026, 6, 10, 23, 59, tzinfo=UTC))
    assert not pushrank.is_tagteam_arena_pushrank_window(datetime(2026, 6, 11, 0, 0, tzinfo=UTC))
    assert not pushrank.is_tagteam_arena_pushrank_window(datetime(2026, 6, 11, 1, 0, tzinfo=UTC))


def test_pushrank_priority_prefers_classic_when_both_are_active():
    forced_mode = pushrank.select_pushrank_mode(
        classic_enabled=True,
        classic_window_active=True,
        tagteam_enabled=True,
        tagteam_window_active=True,
    )

    assert forced_mode == pushrank.CLASSIC_ARENA_PUSH_MODE


def test_get_forced_pushrank_mode_respects_enabled_flags():
    sunday_push = datetime(2026, 6, 14, 18, 30, tzinfo=UTC)
    evening_reset = datetime(2026, 6, 10, 21, 30, tzinfo=UTC)

    assert (
        pushrank.get_forced_pushrank_mode(
            sunday_push,
            classic_enabled=False,
            tagteam_enabled=True,
        )
        is None
    )
    assert (
        pushrank.get_forced_pushrank_mode(
            evening_reset,
            classic_enabled=False,
            tagteam_enabled=True,
        )
        == pushrank.TAGTEAM_ARENA_PUSH_MODE
    )


def test_secondary_profile_defaults_include_disabled_pushrank_flags():
    overrides = file_tools._build_secondary_profile_overrides()

    assert overrides["classic_arena_pushrank"] is False
    assert overrides["tagteam_arena_pushrank"] is False
