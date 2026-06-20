from __future__ import annotations

from types import SimpleNamespace

from raid_bot.modes import arena_tools


def _classic_bot():
    bot = arena_tools.RSL_Bot_ClassicArena.__new__(arena_tools.RSL_Bot_ClassicArena)
    bot.window = SimpleNamespace()
    bot.main_loop_running = True
    bot.running = True
    bot.no_coin_status = False
    bot.classic_arena_multi_refresh = False
    bot.classic_arena_num_multi_refresh = 0
    bot.max_run_duration_seconds = arena_tools.MAX_RUN_DURATION_SECONDS
    bot.recently_skipped_luchar_slots = {}
    bot._classic_arena_scrolled_down = False
    bot.report_run_status = lambda: None
    bot.ensure_arena_coins = lambda: None
    return bot


def test_classic_arena_restores_lower_window_after_lower_list_battle(monkeypatch):
    bot = _classic_bot()
    moves = []
    outcomes = [False, True, False]

    bot.evaluate_arena_enemies = lambda: outcomes.pop(0)
    monkeypatch.setattr(arena_tools.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        arena_tools.window_tools,
        "move_down",
        lambda *_args, **_kwargs: moves.append("down"),
        raising=False,
    )
    monkeypatch.setattr(
        arena_tools.window_tools,
        "move_up",
        lambda *_args, **_kwargs: moves.append("up"),
        raising=False,
    )

    bot.run_classic_arena_until_empty(max_run_duration_seconds=60)

    assert moves == ["down", "down", "up"]
    assert bot._classic_arena_scrolled_down is False


def test_classic_arena_does_not_restore_lower_window_after_top_list_battle(monkeypatch):
    bot = _classic_bot()
    moves = []
    outcomes = [True, False, False]

    bot.evaluate_arena_enemies = lambda: outcomes.pop(0)
    monkeypatch.setattr(arena_tools.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        arena_tools.window_tools,
        "move_down",
        lambda *_args, **_kwargs: moves.append("down"),
        raising=False,
    )
    monkeypatch.setattr(
        arena_tools.window_tools,
        "move_up",
        lambda *_args, **_kwargs: moves.append("up"),
        raising=False,
    )

    bot.run_classic_arena_until_empty(max_run_duration_seconds=60)

    assert moves == ["down", "up"]
    assert bot._classic_arena_scrolled_down is False
