from __future__ import annotations

import json

import _bootstrap  # noqa: F401
import pyautogui

from raid_bot.utils import window_tools


def main() -> int:
    title = input("Window title [Raid: Shadow Legends]: ").strip() or "Raid: Shadow Legends"
    n = int(input("Number of iterations (n): ").strip())

    coords = window_tools.find_window(title)
    if not coords:
        raise RuntimeError(f"Window not found: {title}")

    window = window_tools.WindowObject(coords, title_substring=title)
    window_tools.refresh_window(window)

    captures = []
    print("\nMove mouse to location and press Enter when prompted.\n")
    for i in range(1, n + 1):
        input(f"Capture {i}/{n}: press Enter to capture current mouse position...")
        x, y = pyautogui.position()
        x, y = int(x), int(y)
        rgb = pyautogui.screenshot(region=(x, y, 1, 1)).getpixel((0, 0))
        if len(rgb) == 4:
            rgb = rgb[:3]

        rel_x_raw = (x - window.left) / window.width
        rel_y_raw = (y - window.top) / window.height
        rel_x = round(rel_x_raw, 3)
        rel_y = round(rel_y_raw, 3)

        rounded_x = int(round(window.left + (rel_x * window.width)))
        rounded_y = int(round(window.top + (rel_y * window.height)))
        rounded_x = max(int(window.left), min(rounded_x, int(window.left + window.width - 1)))
        rounded_y = max(int(window.top), min(rounded_y, int(window.top + window.height - 1)))

        rounded_rgb = pyautogui.screenshot(region=(rounded_x, rounded_y, 1, 1)).getpixel((0, 0))
        if len(rounded_rgb) == 4:
            rounded_rgb = rounded_rgb[:3]

        item = {
            "iteration": i,
            "abs_x": int(x),
            "abs_y": int(y),
            "rel_x": rel_x,
            "rel_y": rel_y,
            "rgb": [int(rgb[0]), int(rgb[1]), int(rgb[2])],
            "rounded_abs_x": int(rounded_x),
            "rounded_abs_y": int(rounded_y),
            "rounded_rgb": [int(rounded_rgb[0]), int(rounded_rgb[1]), int(rounded_rgb[2])],
        }
        captures.append(item)
        print(json.dumps(item))

    print("\nAll captures:")
    print(json.dumps(captures, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
