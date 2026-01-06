# -*- coding: utf-8 -*-
"""
Window and Input Utilities for UI Automation

Created on Sat Oct 25 13:46:11 2025
@author: Arthur
"""

import time
import ctypes
from typing import Optional, Tuple, List

import pyautogui
import pygetwindow as gw
import matplotlib.pyplot as plt


# -------------------- Window Representation -------------------- #

class WindowObject:
    """Represents a rectangular window region on screen."""
    def __init__(self, coords: Tuple[int, int, int, int]):
        self.left, self.top, self.width, self.height = coords


# -------------------- Keyboard & Mouse -------------------- #

def sendkey(key: str, delay: float = 3.0):
    """
    Sends a single key press and waits for a delay.
    """
    try:
        pyautogui.press(key)
        time.sleep(delay)
    except Exception:
        print(f"SendKey failed for '{key}'")


def click_at(x: int, y: int, delay: float = 3.0):
    """Click at absolute screen coordinates."""
    pyautogui.click(x, y)
    time.sleep(delay)


def click_center(window: WindowObject, rel_coords: Tuple[float, float, float, float] = (0.5, 0.5, 0, 0),
                 clicks: int = 1, delay: float = 3.0):
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

    start_x = int(window.left + start_rel[0] * window.width)
    start_y = int(window.top + start_rel[1] * window.height)
    end_x = int(window.left + end_rel[0] * window.width)
    end_y = int(window.top + end_rel[1] * window.height)

    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    time.sleep(0.1)
    pyautogui.moveTo(end_x, end_y, duration=duration)
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


# -------------------- Mouse Position & Clicks -------------------- #

def get_mouse_pos() -> Tuple[int, int]:
    """Returns the current mouse cursor position."""
    user32 = ctypes.windll.user32
    point = ctypes.wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def wait_for_click() -> Tuple[int, int]:
    """Waits for left mouse click and returns the cursor position."""
    user32 = ctypes.windll.user32
    VK_LBUTTON = 0x01

    while True:
        if user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000:
            pos = get_mouse_pos()
            while user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000:
                time.sleep(0.01)
            return pos
        time.sleep(0.01)


def get_two_clicks() -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Prompt the user for two clicks: upper-left and lower-right corners."""
    print("Click UPPER-LEFT corner...")
    ul = wait_for_click()
    time.sleep(0.2)
    print("Click LOWER-RIGHT corner...")
    lr = wait_for_click()
    return ul, lr


def compile_search_area_from_clicks(ul_px: Tuple[int, int], lr_px: Tuple[int, int], bot) -> List[float]:
    """
    Convert two absolute clicks into relative coordinates inside a bot window.

    Returns: [rel_x, rel_y, rel_width, rel_height]
    """
    left, top, width, height = bot.coords
    x1, y1 = sorted([ul_px[0], lr_px[0]])
    x2, y2 = sorted([ul_px[1], lr_px[1]])

    rel_x = (x1 - left) / width
    rel_y = (y1 - top) / height
    rel_width = (x2 - x1) / width
    rel_height = (y2 - y1) / height

    return [round(rel_x, 3), round(rel_y, 3), round(rel_width, 3), round(rel_height, 3)]