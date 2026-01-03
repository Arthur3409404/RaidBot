# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:46:11 2025

@author: Arthur
"""

import time
import pyautogui
import pygetwindow as gw
import ctypes
import time
import matplotlib.pyplot as plt
import re


class WindowObject:
    def __init__(self, window):
        self.left = window[0]
        self.top = window[1]
        self.width = window[2]
        self.height = window[3]



def sendkey(key, delay=2):
    """
    Sends a single key press and waits for a delay.

    :param key: Key to press (e.g. 'esc', 'enter', 'f1')
    :param delay: Delay after key press in seconds (default: 2)
    """
    try:
        pyautogui.press(key)
        time.sleep(delay)
    except Exception:
        print('SendKey Failed')

def test_window(window):
    """
    Captures a screenshot of the given game window and plots it.

    Args:
        window: A window object (from pygetwindow) with attributes left, top, width, height.
    """
    if window is None:
        print("No window provided for testing.")
        return

    left, top, width, height = window.left, window.top, window.width, window.height
    screenshot = pyautogui.screenshot(region=(left, top, width, height))

    plt.figure(figsize=(10, 6))
    plt.imshow(screenshot)
    plt.title("Captured Game Window")
    plt.axis("off")
    plt.show()


def find_window(window_title):
    """
    Finds a window by its title.

    Args:
        window_title (str): The title (or part of the title) of the window.

    Returns:
        tuple | None: (left, top, width, height) of the window if found, otherwise None.
    """
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        print(f"Window '{window_title}' not found.")
        return None

    window = windows[0]
    # if not window.isActive:
    #     window.activate()
    #     time.sleep(0.3)

    return (window.left, window.top, window.width, window.height)


def click_at(x, y, delay=3):
    """
    Click at the given absolute screen coordinates (x, y).
    """
    pyautogui.click(x, y)
    time.sleep(delay)
    return 

def click_center(window, coords, clicks=1, delay=2):
    """
    Clicks the center of the given window.

    Args:
        coords (tuple): (left, top, width, height) of the window.
        clicks (int): Number of clicks (default 1).
        delay (float): Delay after clicking (default 0.1).
    """
    if not coords:
        print("No window coordinates provided for center click.")
        return

    left, top, width, height = coords
    abs_x = (left + width/2)*window.width + window.left
    abs_y = (top + height/2)*window.height + window.top

    pyautogui.click(abs_x, abs_y, clicks=clicks)
    time.sleep(delay)


def move_up(window, strength = 1, relative_x_pos = 0.5, relative_y_pos = 0.5):
    """
    Click and drag from the center downward.
    
    Args:
        window: Object with attributes left, top, width, height
    """
    if not window:
        return

    start_x = window.left + window.width *relative_x_pos
    start_y = window.top + window.height *relative_y_pos
    end_y = start_y + int(window.height * 0.49 * strength)  # Drag ~49% of height downward

    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    time.sleep(0.1)
    pyautogui.moveTo(start_x, end_y, duration=0.2)
    pyautogui.mouseUp()
    time.sleep(5)
    



def move_down(window, strength = 1, relative_x_pos = 0.5, relative_y_pos = 0.5):
    """
    Click and drag from the center upward.
    
    Args:
        window: Object with attributes left, top, width, height
    """
    if not window:
        return

    start_x = window.left + window.width *relative_x_pos
    start_y = window.top + window.height *relative_y_pos
    end_y = start_y - int(window.height * 0.49 * strength)  # Drag ~49% of height upward

    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    time.sleep(0.1)
    pyautogui.moveTo(start_x, end_y, duration=0.2)
    pyautogui.mouseUp()
    time.sleep(5)
    
    
def move_left(window, strength = 1, relative_x_pos = 0.5, relative_y_pos = 0.5):
    """
    Click and drag from the center downward.
    
    Args:
        window: Object with attributes left, top, width, height
    """
    if not window:
        return

    start_x = window.left + window.width *relative_x_pos
    start_y = window.top + window.height *relative_y_pos
    end_x = start_x + int(window.width * 0.49 * strength)  # Drag ~49% of height downward

    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    time.sleep(0.1)
    pyautogui.moveTo(end_x, start_y, duration=0.2)
    pyautogui.mouseUp()
    time.sleep(5)
    
    
def move_right(window, strength = 1, relative_x_pos = 0.5, relative_y_pos = 0.5):
    """
    Click and drag from the center downward.
    
    Args:
        window: Object with attributes left, top, width, height
    """
    if not window:
        return

    start_x = window.left + window.width *relative_x_pos
    start_y = window.top + window.height *relative_y_pos
    end_x = start_x - int(window.width * 0.49 * strength)  # Drag ~49% of height downward

    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    time.sleep(0.1)
    pyautogui.moveTo(end_x, start_y, duration=0.2)
    pyautogui.mouseUp()
    time.sleep(5)
    
    
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