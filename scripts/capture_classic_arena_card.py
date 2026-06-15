from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

import pyautogui

from raid_bot.utils import image_tools, window_tools


DEFAULT_WINDOW_TITLE = "Raid: Shadow Legends"
DEFAULT_OUTPUT_DIR = Path("temp") / "classic_arena_card_samples"
DEFAULT_SEARCH_AREA = [0.5, 0.0, 0.5, 1.0]


def _make_output_paths(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"classic_arena_card_{stamp}.png"
    annotation_path = output_dir / f"classic_arena_card_{stamp}.json"
    return image_path, annotation_path


def _find_luchar_text(reader, window):
    text_objects = image_tools.get_text_in_relative_area(
        reader,
        window,
        search_area=DEFAULT_SEARCH_AREA,
        power_detection=False,
    )
    filtered = image_tools.filter_text_objects(text_objects)
    for obj in sorted(filtered, key=lambda item: (float(item.mean_pos_y), float(item.mean_pos_x))):
        if (obj.text or "").strip().lower() == "luchar":
            return obj
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one classic arena enemy card and open a 4-box annotator.")
    parser.add_argument("--window-title", default=DEFAULT_WINDOW_TITLE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--card-width", type=int, default=440)
    parser.add_argument("--card-height", type=int, default=130)
    parser.add_argument("--x-offset", type=int, default=500)
    parser.add_argument("--y-offset", type=int, default=65)
    parser.add_argument("--no-annotate", action="store_true", help="Only save the card image and do not open the annotator.")
    args = parser.parse_args()

    window_coords = window_tools.find_window(args.window_title)
    if not window_coords:
        print(f"Could not find window: {args.window_title}")
        return 1

    window = window_tools.WindowObject(window_coords, title_substring=args.window_title)
    window_tools.refresh_window(window)

    try:
        import easyocr
    except ModuleNotFoundError as exc:
        print(f"Missing OCR dependency: {exc.name}")
        return 1

    reader = easyocr.Reader(["en"])
    luchar = _find_luchar_text(reader, window)
    if luchar is None:
        print("Could not find a visible 'Luchar' card in the current classic arena list.")
        return 1

    left = int(luchar.mean_pos_x - int(args.x_offset))
    top = int(luchar.mean_pos_y - int(args.y_offset))
    region = (left, top, int(args.card_width), int(args.card_height))
    screenshot = pyautogui.screenshot(region=region)

    image_path, annotation_path = _make_output_paths(Path(args.output_dir))
    screenshot.save(str(image_path))

    print(f"Saved classic arena card to: {image_path.resolve().as_posix()}")

    if not args.no_annotate:
        annotator_script = Path(__file__).resolve().parent / "annotate_tagteam_portraits.py"
        subprocess.Popen(
            [
                sys.executable,
                str(annotator_script),
                "--image",
                str(image_path),
                "--output",
                str(annotation_path),
                "--boxes-per-slot",
                "4",
                "--slots",
                "1",
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        print(f"Opened annotator for: {annotation_path.resolve().as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
