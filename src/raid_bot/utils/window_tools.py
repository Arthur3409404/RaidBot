# -*- coding: utf-8 -*-
"""
Window and Input Utilities for UI Automation

Created on Sat Oct 25 13:46:11 2025
@author: Arthur
"""

import time
import ctypes
from typing import Optional, Tuple

import pyautogui
import pygetwindow as gw
import matplotlib.pyplot as plt


# -------------------- Window Representation -------------------- #

class WindowObject:
    """Represents a rectangular window region on screen."""
    def __init__(self, coords: Tuple[int, int, int, int], title_substring: Optional[str] = None):
        self.left, self.top, self.width, self.height = coords
        self.title_substring = title_substring


def _match_runtime_window(window: WindowObject):
    if not window:
        return None

    title_substring = getattr(window, "title_substring", None)
    candidates = []
    if title_substring:
        try:
            candidates = gw.getWindowsWithTitle(title_substring)
        except Exception:
            candidates = []

    if not candidates and not title_substring:
        try:
            center_x = int(window.left + window.width / 2)
            center_y = int(window.top + window.height / 2)
            candidates = gw.getWindowsAt(center_x, center_y)
        except Exception:
            candidates = []

    if not candidates:
        return None

    def score(candidate):
        try:
            dx = abs(int(candidate.left) - int(window.left))
            dy = abs(int(candidate.top) - int(window.top))
            dw = abs(int(candidate.width) - int(window.width))
            dh = abs(int(candidate.height) - int(window.height))
            return dx + dy + dw + dh
        except Exception:
            return 10**9

    return min(candidates, key=score)


def refresh_window(window: WindowObject) -> Optional[WindowObject]:
    runtime_window = _match_runtime_window(window)
    if not runtime_window:
        return window

    try:
        window.left = int(runtime_window.left)
        window.top = int(runtime_window.top)
        window.width = int(runtime_window.width)
        window.height = int(runtime_window.height)
    except Exception:
        pass
    return window


def ensure_window_focus(window: WindowObject, delay: float = 0.1) -> bool:
    """Best-effort activation of the Raid window before sending input."""
    if not window:
        return False

    title_substring = getattr(window, "title_substring", None)
    try:
        active_window = gw.getActiveWindow()
        active_title = (active_window.title or "") if active_window else ""
        if title_substring and active_title and title_substring.lower() in active_title.lower():
            refresh_window(window)
            return True
    except Exception:
        pass

    runtime_window = _match_runtime_window(window)
    if not runtime_window:
        return False

    try:
        if getattr(runtime_window, "isMinimized", False):
            runtime_window.restore()
            time.sleep(delay)
    except Exception:
        pass

    try:
        runtime_window.activate()
        time.sleep(delay)
    except Exception:
        pass

    refresh_window(window)
    return True


# -------------------- Keyboard & Mouse -------------------- #

def sendkey(key: str, delay: float = 3.0, window: WindowObject | None = None):
    """
    Sends a single key press and waits for a delay.
    """
    try:
        if window:
            ensure_window_focus(window)
        pyautogui.press(key)
        time.sleep(delay)
    except Exception:
        print(f"SendKey failed for '{key}'")


def click_at(x: int, y: int, delay: float = 3.0, window: WindowObject | None = None):
    """Click at absolute screen coordinates."""
    if window:
        ensure_window_focus(window)
    pyautogui.click(x, y)
    time.sleep(delay)


def click_center(window: WindowObject, rel_coords: Tuple[float, float, float, float] = (0.5, 0.5, 0, 0),
                 clicks: int = 1, delay: float = 3.0, settle_delay: float = 0.08):
    """
    Clicks at the center of a relative rectangle inside a window.

    Args:
        window: WindowObject
        rel_coords: (rel_left, rel_top, rel_width, rel_height)
        clicks: Number of clicks
        delay: Pause after clicking
    """
    if not window:
        print("No window provided for center click.")
        return

    ensure_window_focus(window)
    # In full-speed runs, windows can report stale geometry right after activation.
    # A short settle + refresh mirrors debugger pacing and avoids off-target clicks.
    if settle_delay > 0:
        time.sleep(settle_delay)
    refresh_window(window)
    rel_left, rel_top, rel_width, rel_height = rel_coords
    abs_x = int(window.left + (rel_left + rel_width / 2) * window.width)
    abs_y = int(window.top + (rel_top + rel_height / 2) * window.height)

    pyautogui.click(abs_x, abs_y, clicks=clicks)
    time.sleep(delay)


# -------------------- Window Detection -------------------- #

def find_window(window_title: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Finds a window by title and returns its coordinates (left, top, width, height).
    """
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        print(f"Window '{window_title}' not found.")
        return None

    window = windows[0]
    return (window.left, window.top, window.width, window.height)


def test_window(window: WindowObject):
    """
    Captures and displays a screenshot of the given window.
    """
    if not window:
        print("No window provided for testing.")
        return

    screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
    plt.figure(figsize=(10, 6))
    plt.imshow(screenshot)
    plt.title("Captured Game Window")
    plt.axis("off")
    plt.show()


# -------------------- Drag / Movement -------------------- #

# -------------------- Drag / Movement -------------------- #

# Base relative movement per full "step"
BASE_DELTA = 0.49


def _drag(window: WindowObject, start_rel: tuple[float, float], end_rel: tuple[float, float], duration: float = 0.2, delay: float = 5.0):
    """Generic drag helper: from start_rel to end_rel inside window."""
    if not window:
        return

    ensure_window_focus(window)
    start_x = int(window.left + start_rel[0] * window.width)
    start_y = int(window.top + start_rel[1] * window.height)
    end_x = int(window.left + end_rel[0] * window.width)
    end_y = int(window.top + end_rel[1] * window.height)

    pyautogui.moveTo(start_x, start_y)
    mouse_is_down = False
    try:
        pyautogui.mouseDown()
        mouse_is_down = True
        time.sleep(0.1)
        pyautogui.moveTo(end_x, end_y, duration=duration)
    finally:
        if mouse_is_down:
            # Always release in case move/focus operations raise, preventing stuck input state.
            pyautogui.mouseUp()
    time.sleep(delay)  # keep original wait times


def _move(window, dx: float, dy: float, strength: float, relative_x: float = 0.5, relative_y: float = 0.5):
    """Generic move function handling fractional and full strength."""
    if not window or strength <= 0:
        return

    full_steps = int(strength)          # number of full moves
    remainder = strength - full_steps   # fractional part

    # Perform full moves
    for _ in range(full_steps):
        _drag(window, start_rel=(relative_x, relative_y),
              end_rel=(relative_x + dx, relative_y + dy))

    # Perform fractional move if remainder exists
    if remainder > 0:
        _drag(window, start_rel=(relative_x, relative_y),
              end_rel=(relative_x + dx * remainder, relative_y + dy * remainder))


def move_up(window: WindowObject, strength: float = 1.0, relative_x: float = 0.5, relative_y: float = 0.5):
    _move(window, dx=0, dy=BASE_DELTA, strength=strength, relative_x=relative_x, relative_y=relative_y)


def move_down(window: WindowObject, strength: float = 1.0, relative_x: float = 0.5, relative_y: float = 0.5):
    _move(window, dx=0, dy=-BASE_DELTA, strength=strength, relative_x=relative_x, relative_y=relative_y)


def move_right(window: WindowObject, strength: float = 1.0, relative_x: float = 0.5, relative_y: float = 0.5):
    _move(window, dx=-BASE_DELTA, dy=0, strength=strength, relative_x=relative_x, relative_y=relative_y)


def move_left(window: WindowObject, strength: float = 1.0, relative_x: float = 0.5, relative_y: float = 0.5):
    _move(window, dx=BASE_DELTA, dy=0, strength=strength, relative_x=relative_x, relative_y=relative_y)


def scroll_vertical(
    window: WindowObject,
    amount: int,
    relative_x: float = 0.5,
    relative_y: float = 0.5,
    delay: float = 1.0,
):
    """Scroll the mouse wheel at a relative point inside the window."""
    if not window:
        return

    ensure_window_focus(window)
    abs_x = int(window.left + relative_x * window.width)
    abs_y = int(window.top + relative_y * window.height)
    pyautogui.moveTo(abs_x, abs_y)
    pyautogui.scroll(int(amount))
    time.sleep(delay)


def zoom_out(
    window: WindowObject,
    steps: int = 4,
    amount_per_step: int = -600,
    relative_x: float = 0.5,
    relative_y: float = 0.5,
    delay: float = 0.75,
):
    """Zoom out via repeated mouse-wheel scrolls over the game window."""
    if not window:
        return

    for _ in range(max(0, int(steps))):
        scroll_vertical(
            window,
            amount=amount_per_step,
            relative_x=relative_x,
            relative_y=relative_y,
            delay=delay,
        )


# -------------------- Mouse Position & Clicks -------------------- #

def get_mouse_pos():
    user32 = ctypes.windll.user32
    VK_LBUTTON = 0x01
    point = ctypes.wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def wait_for_click():
    user32 = ctypes.windll.user32
    VK_LBUTTON = 0x01
    while True:
        if user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000:
            pos = get_mouse_pos()
            # wait until released
            while user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000:
                time.sleep(0.01)
            return pos
        time.sleep(0.01)


def get_two_clicks():
    print("Click UPPER-LEFT corner...")
    ul = wait_for_click()

    time.sleep(0.2)

    print("Click LOWER-RIGHT corner...")
    lr = wait_for_click()

    return ul, lr


def compile_search_area_from_clicks(ul_px, lr_px, bot):
    left, top, width, height = bot.coords

    x1, y1 = ul_px
    x2, y2 = lr_px

    # Ensure correct ordering
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    rel_x = (x1 - left) / width
    rel_y = (y1 - top) / height
    rel_dx = (x2 - x1) / width
    rel_dy = (y2 - y1) / height

    return [
        round(rel_x, 3),
        round(rel_y, 3),
        round(rel_dx, 3),
        round(rel_dy, 3),
    ]
