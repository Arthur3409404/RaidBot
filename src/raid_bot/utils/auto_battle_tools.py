# -*- coding: utf-8 -*-
"""Auto-battle watchdog helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Sequence
import time

import pyautogui

import raid_bot.utils.image_tools as image_tools
import raid_bot.utils.window_tools as window_tools


AUTO_BATTLE_ACTIVITY_PIXEL_AREA = [0.7955, 0.9031, 0.0224, 0.0235]
AUTO_BATTLE_BUTTON_AREA = [0.026, 0.899, 0.058, 0.07]
AUTO_BATTLE_SAMPLE_INTERVAL_SECONDS = 10.0
AUTO_BATTLE_STAGNANT_SAMPLE_COUNT = 12
AUTO_BATTLE_CLICK_COOLDOWN_SECONDS = 30.0
PAUSA_SEARCH_AREA = [0.35, 0.10, 0.30, 0.30]
PAUSA_CONFIRMATION_DELAY_SECONDS = 10.0
PAUSA_CENTER_MATCH_MARGIN_PX = 10.0


@dataclass
class AutoBattleWatchdogState:
    samples: Deque[tuple[int, int, int]] = field(
        default_factory=lambda: deque(maxlen=AUTO_BATTLE_STAGNANT_SAMPLE_COUNT)
    )
    last_sample_at: float = 0.0
    last_click_at: float = 0.0


def reset_auto_battle_watchdog(bot) -> None:
    state = AutoBattleWatchdogState()
    state.last_sample_at = time.monotonic()
    setattr(bot, "_auto_battle_watchdog", state)


def ensure_auto_battle_running(
    bot,
    activity_pixel_area: Sequence[float] | None = None,
    auto_button_area: Sequence[float] | None = None,
    sample_interval_seconds: float = AUTO_BATTLE_SAMPLE_INTERVAL_SECONDS,
    stagnant_sample_count: int = AUTO_BATTLE_STAGNANT_SAMPLE_COUNT,
) -> bool:
    """Click auto if the watched battle pixel is unchanged for two minutes.

    This function is intentionally non-blocking. Call it from existing battle
    polling loops; it samples at most once per ``sample_interval_seconds``.
    """
    if not _auto_battle_watchdog_enabled(bot):
        return False

    window = getattr(bot, "window", None)
    if not window:
        return False

    now = time.monotonic()
    state = getattr(bot, "_auto_battle_watchdog", None)
    if state is None or state.samples.maxlen != int(stagnant_sample_count):
        state = AutoBattleWatchdogState(
            samples=deque(maxlen=max(1, int(stagnant_sample_count)))
        )
        state.last_sample_at = now
        setattr(bot, "_auto_battle_watchdog", state)

    if state.last_sample_at and (now - state.last_sample_at) < float(sample_interval_seconds):
        return False

    pixel = sample_center_pixel(window, activity_pixel_area or AUTO_BATTLE_ACTIVITY_PIXEL_AREA)
    state.last_sample_at = now
    if pixel is None:
        return False

    state.samples.append(pixel)
    if len(state.samples) < max(1, int(stagnant_sample_count)):
        return False
    if not _all_pixels_same(state.samples):
        return False

    if state.last_click_at and (now - state.last_click_at) < AUTO_BATTLE_CLICK_COOLDOWN_SECONDS:
        state.samples.clear()
        return False

    click_area = _resolve_auto_button_area(bot, auto_button_area)
    _log(bot, "Auto-battle watchdog detected a stagnant battle pixel; pressing Auto.")
    window_tools.click_center(window, click_area, delay=0.5)
    state.last_click_at = now
    state.samples.clear()
    return True


def handle_stable_pausa(
    bot,
    *,
    search_area: Sequence[float] | None = None,
    confirmation_delay_seconds: float = PAUSA_CONFIRMATION_DELAY_SECONDS,
    center_margin_px: float = PAUSA_CENTER_MATCH_MARGIN_PX,
    threshold: float = 0.68,
) -> bool:
    """Press Esc only when the same Pausa text remains stable for two checks."""
    first_object = find_pausa_object(bot, search_area=search_area, threshold=threshold)
    if first_object is None:
        return False

    time.sleep(float(confirmation_delay_seconds))
    second_object = find_pausa_object(bot, search_area=search_area, threshold=threshold)
    if second_object is None:
        return False

    if not text_object_centers_match(first_object, second_object, margin_px=center_margin_px):
        _log(
            bot,
            "Pausa text moved between checks; ignoring pause recovery. "
            f"First={text_object_center(first_object)}, second={text_object_center(second_object)}.",
        )
        return False

    if not getattr(bot, "_pausa_esc_sent", False):
        window_tools.sendkey("esc", delay=0.2, window=getattr(bot, "window", None))
        bot._pausa_esc_sent = True
    return True


def find_pausa_object(
    bot,
    *,
    search_area: Sequence[float] | None = None,
    threshold: float = 0.68,
):
    try:
        text_objects = image_tools.get_text_in_relative_area(
            bot.reader,
            bot.window,
            search_area=search_area or PAUSA_SEARCH_AREA,
            power_detection=False,
        )
    except Exception:
        return None

    for text_object in text_objects:
        text = (getattr(text_object, "text", "") or "").strip()
        if text and _resembles(bot, text, "Pausa", threshold=threshold):
            return text_object
    return None


def text_object_centers_match(first_object, second_object, margin_px: float = PAUSA_CENTER_MATCH_MARGIN_PX) -> bool:
    first_center = text_object_center(first_object)
    second_center = text_object_center(second_object)
    if first_center is None or second_center is None:
        return False
    return (
        abs(first_center[0] - second_center[0]) <= float(margin_px)
        and abs(first_center[1] - second_center[1]) <= float(margin_px)
    )


def text_object_center(text_object) -> tuple[float, float] | None:
    try:
        return float(text_object.mean_pos_x), float(text_object.mean_pos_y)
    except (AttributeError, TypeError, ValueError):
        return None


def sample_center_pixel(window, relative_area: Sequence[float]) -> tuple[int, int, int] | None:
    if not window or not relative_area or len(relative_area) < 4:
        return None

    rel_left, rel_top, rel_width, rel_height = [float(value) for value in relative_area[:4]]
    abs_x = int(window.left + (rel_left + rel_width / 2.0) * window.width)
    abs_y = int(window.top + (rel_top + rel_height / 2.0) * window.height)

    try:
        screenshot = pyautogui.screenshot(region=(abs_x, abs_y, 1, 1))
        pixel = screenshot.getpixel((0, 0))
    except Exception:
        return None

    return tuple(int(channel) for channel in pixel[:3])


def _all_pixels_same(samples: Iterable[tuple[int, int, int]]) -> bool:
    iterator = iter(samples)
    try:
        first = next(iterator)
    except StopIteration:
        return False
    return all(pixel == first for pixel in iterator)


def _resolve_auto_button_area(bot, explicit_area: Sequence[float] | None = None) -> Sequence[float]:
    if explicit_area is not None:
        return explicit_area

    search_areas = getattr(bot, "search_areas", {}) or {}
    for key in ("auto_battle_button", "stage_auto_battle_button", "doom_tower_auto_battle_button"):
        area = search_areas.get(key)
        if area is not None:
            return area
    return AUTO_BATTLE_BUTTON_AREA


def _auto_battle_watchdog_enabled(bot) -> bool:
    setup = getattr(bot, "setup", None)
    if isinstance(setup, dict) and "auto_battle_watchdog_enabled" in setup:
        return bool(setup.get("auto_battle_watchdog_enabled"))
    return bool(getattr(bot, "auto_battle_watchdog_enabled", False))


def _resembles(bot, text: str, target: str, threshold: float = 0.68) -> bool:
    resembles = getattr(bot, "resembles", None)
    if callable(resembles):
        try:
            return bool(resembles(text, target, threshold=threshold))
        except TypeError:
            return bool(resembles(text, target))
    return text.strip().lower() == target.strip().lower()


def _log(bot, message: str) -> None:
    logger = getattr(bot, "log", None)
    if logger:
        logger.info(message)
    elif getattr(bot, "verbose", False):
        print(message)
