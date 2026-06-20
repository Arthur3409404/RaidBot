from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pygetwindow as gw

import _bootstrap  # noqa: F401

from raid_bot.modes.grimforest_tools import RSL_Bot_GrimForest, _spiral_direction_for_step
from raid_bot.utils import map_tools, window_tools


DEFAULT_OUTPUT_BASE = Path("data") / "output" / "debug"
DEFAULT_TITLE = "Raid: Shadow Legends"
DEFAULT_SPIRAL_CAPTURE_COUNT = 24


def _find_window(title_substring: str) -> window_tools.WindowObject:
    matches = gw.getWindowsWithTitle(title_substring)
    if not matches:
        raise RuntimeError(f"No window found with title containing {title_substring!r}.")

    runtime_window = matches[0]
    if getattr(runtime_window, "isMinimized", False):
        runtime_window.restore()
        time.sleep(0.5)

    return window_tools.WindowObject(
        (
            int(runtime_window.left),
            int(runtime_window.top),
            int(runtime_window.width),
            int(runtime_window.height),
        ),
        title_substring=title_substring,
    )


def _grim_forest_mask(pov_np: np.ndarray, target_rgb: np.ndarray, dark_tolerance: int) -> np.ndarray:
    base = target_rgb.astype(np.int16)
    difference = pov_np.astype(np.int16) - base
    tolerance = int(dark_tolerance)
    mask = (
        (difference[:, :, 0] >= -tolerance)
        & (difference[:, :, 1] >= -tolerance)
        & (difference[:, :, 2] >= -tolerance)
    )
    return np.where(mask, 255, 0).astype(np.uint8)


def _move(window: window_tools.WindowObject, direction: str, strength: float) -> None:
    move_functions = {
        "up": window_tools.move_up,
        "down": window_tools.move_down,
        "left": window_tools.move_left,
        "right": window_tools.move_right,
    }
    move_function = move_functions.get(direction)
    if move_function is None:
        return
    move_function(window, strength=float(strength))


def _reset_to_bottom_left(
    window: window_tools.WindowObject,
    *,
    left_steps: int,
    down_steps: int,
    strength: float,
    delay: float,
) -> list[str]:
    movements: list[str] = []
    for _ in range(max(0, int(left_steps))):
        _move(window, "left", strength=strength)
        movements.append("left")
        time.sleep(max(0.0, float(delay)))
    for _ in range(max(0, int(down_steps))):
        _move(window, "down", strength=strength)
        movements.append("down")
        time.sleep(max(0.0, float(delay)))
    return movements


def _switch_difficulty(
    bot: RSL_Bot_GrimForest,
    difficulty: str,
    *,
    open_delay: float = 0.8,
    confirm_delay: float = 2.5,
) -> None:
    requested = bot._normalize_difficulty_value(difficulty)
    if requested not in {"normal", "hard"}:
        return

    switch_key = f"mode_difficulty_switch_{requested}"
    window_tools.click_center(
        bot.window,
        bot.search_areas["mode_difficulty_current"],
        delay=float(open_delay),
    )
    window_tools.click_center(
        bot.window,
        bot.search_areas[switch_key],
        delay=float(confirm_delay),
    )


def _serpentine_movements(repeats: int = 10, horizontal_steps: int = 10) -> list[str]:
    movements: list[str] = []
    for _ in range(max(0, int(repeats))):
        movements.extend(["right"] * max(0, int(horizontal_steps)))
        movements.append("up")
        movements.extend(["left"] * max(0, int(horizontal_steps)))
        movements.append("up")
    return movements


def _movement_before_capture(index: int, args: argparse.Namespace) -> str | None:
    if index <= 0 or args.movement == "none":
        return None

    if args.movement == "spiral":
        return _spiral_direction_for_step(index - 1, start_direction_index=int(args.start_direction_index))

    if args.movement == "serpentine":
        sequence = _serpentine_movements(
            repeats=int(args.serpentine_repeats),
            horizontal_steps=int(args.serpentine_horizontal_steps),
        )
        return sequence[index - 1] if index - 1 < len(sequence) else None

    return None


def _capture_count(args: argparse.Namespace) -> int:
    if args.captures is not None:
        return max(1, int(args.captures))
    if args.movement == "serpentine":
        return len(
            _serpentine_movements(
                repeats=int(args.serpentine_repeats),
                horizontal_steps=int(args.serpentine_horizontal_steps),
            )
        ) + 1
    return DEFAULT_SPIRAL_CAPTURE_COUNT


def _difficulty_slug(args: argparse.Namespace) -> str:
    difficulty = str(args.difficulty or "none").strip().lower()
    if difficulty in {"normal", "hard"}:
        return difficulty
    return "unspecified"


def _save_capture(
    *,
    run_dir: Path,
    bot: RSL_Bot_GrimForest,
    window: window_tools.WindowObject,
    relative_area: list[float],
    label: str,
    movement: str | None,
    zoom_steps: int,
) -> dict:
    screenshot, pov_np, pov_region = map_tools.capture_relative_area(window, relative_area)
    mask = _grim_forest_mask(
        pov_np,
        bot.target_bgr_as_rgb,
        dark_tolerance=int(bot.setup.get("dark_tolerance", 40)),
    )

    raw_path = run_dir / "pov.png"
    binary_path = run_dir / "pov_cyan_brighter_binary.png"
    overlay_path = run_dir / "pov_cyan_brighter_binary_overlay.png"
    meta_path = run_dir / "capture.json"

    screenshot.save(raw_path)
    cv2.imwrite(str(binary_path), mask)

    overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        overlay,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(overlay_path), overlay)

    metadata = {
        "label": label,
        "movement_before_capture": movement,
        "zoom_steps_before_session": int(zoom_steps),
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "relative_area": list(relative_area),
        "pov_region": list(pov_region),
        "window": {
            "left": int(window.left),
            "top": int(window.top),
            "width": int(window.width),
            "height": int(window.height),
            "title_substring": window.title_substring,
        },
        "files": {
            "raw": raw_path.as_posix(),
            "binary": binary_path.as_posix(),
            "overlay": overlay_path.as_posix(),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture non-overwriting Grim Forest detector training images after zooming out. "
            "Each capture is saved as a debug run containing pov.png and pov_cyan_brighter_binary.png."
        )
    )
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Raid window title substring.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output collection root. Defaults to data/output/debug/grimforest_zoomed_out_<difficulty>.",
    )
    parser.add_argument("--captures", type=int, default=None, help="Number of viewports to capture.")
    parser.add_argument("--interval", type=float, default=0.35, help="Delay between move and capture.")
    parser.add_argument("--zoom-steps", type=int, default=3, help="Mouse-wheel zoom-out steps before capture.")
    parser.add_argument("--zoom-amount", type=int, default=-600, help="Mouse-wheel amount per zoom step.")
    parser.add_argument("--zoom-delay", type=float, default=0.75, help="Delay after each zoom step.")
    parser.add_argument(
        "--difficulty",
        choices=("none", "normal", "hard"),
        default="normal",
        help="Optionally switch Grim Forest difficulty before zoom/capture.",
    )
    parser.add_argument("--difficulty-open-delay", type=float, default=0.8, help="Delay after opening difficulty dropdown.")
    parser.add_argument("--difficulty-confirm-delay", type=float, default=2.5, help="Delay after choosing difficulty.")
    parser.add_argument("--pan-strength", type=float, default=1.0, help="Drag strength between captures.")
    parser.add_argument(
        "--reset-bottom-left",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move left and down before capture so the first image starts in the lower-left corner.",
    )
    parser.add_argument("--reset-left-steps", type=int, default=12, help="Left moves used for bottom-left reset.")
    parser.add_argument("--reset-down-steps", type=int, default=12, help="Down moves used for bottom-left reset.")
    parser.add_argument(
        "--movement",
        choices=("none", "spiral", "serpentine"),
        default="serpentine",
        help="How to move between captures.",
    )
    parser.add_argument(
        "--start-direction-index",
        type=int,
        default=0,
        help="Spiral start direction index: 0=right, 1=down, 2=left, 3=up.",
    )
    parser.add_argument("--serpentine-repeats", type=int, default=10, help="Number of serpentine row pairs.")
    parser.add_argument(
        "--serpentine-horizontal-steps",
        type=int,
        default=10,
        help="Number of horizontal moves before each upward move in serpentine mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    difficulty_slug = _difficulty_slug(args)
    output_root = (
        Path(args.output_root)
        if args.output_root
        else DEFAULT_OUTPUT_BASE / f"grimforest_zoomed_out_{difficulty_slug}"
    )
    session_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=False)

    window = _find_window(args.title)
    window_tools.ensure_window_focus(window, delay=0.5)

    bot = RSL_Bot_GrimForest(
        reader=None,
        window=window,
        setup={
            "initial_candidate_zoom_out_steps": args.zoom_steps,
            "difficulty": "normal" if args.difficulty == "none" else args.difficulty,
        },
    )
    relative_area = bot.search_areas["pov"]

    print(f"Output: {session_dir}", flush=True)
    print(f"Window: left={window.left} top={window.top} width={window.width} height={window.height}", flush=True)
    if args.difficulty != "none":
        print(f"Switching difficulty to {args.difficulty}.", flush=True)
        _switch_difficulty(
            bot,
            args.difficulty,
            open_delay=float(args.difficulty_open_delay),
            confirm_delay=float(args.difficulty_confirm_delay),
        )
    print(f"Zooming out {args.zoom_steps} step(s).", flush=True)
    window_tools.zoom_out(
        window,
        steps=int(args.zoom_steps),
        amount_per_step=int(args.zoom_amount),
        delay=float(args.zoom_delay),
    )

    reset_movements: list[str] = []
    if args.reset_bottom_left:
        print(
            "Resetting toward bottom-left "
            f"({args.reset_left_steps} left, {args.reset_down_steps} down).",
            flush=True,
        )
        reset_movements = _reset_to_bottom_left(
            window,
            left_steps=int(args.reset_left_steps),
            down_steps=int(args.reset_down_steps),
            strength=float(args.pan_strength),
            delay=float(args.interval),
        )

    total_captures = _capture_count(args)
    captures = []
    for index in range(total_captures):
        movement = _movement_before_capture(index, args)
        if movement:
            _move(window, movement, strength=float(args.pan_strength))
            time.sleep(max(0.0, float(args.interval)))

        run_dir = session_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index + 1:03d}"
        run_dir.mkdir(parents=True, exist_ok=False)
        label = f"grimforest_zoomed_out_{difficulty_slug}_{index + 1:03d}"
        metadata = _save_capture(
            run_dir=run_dir,
            bot=bot,
            window=window,
            relative_area=relative_area,
            label=label,
            movement=movement,
            zoom_steps=int(args.zoom_steps),
        )
        captures.append(metadata)
        print(f"[{index + 1}/{total_captures}] {run_dir}", flush=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "title": args.title,
        "capture_count": len(captures),
        "difficulty": args.difficulty,
        "zoom_steps": int(args.zoom_steps),
        "zoom_amount": int(args.zoom_amount),
        "movement": args.movement,
        "reset_bottom_left": bool(args.reset_bottom_left),
        "reset_movements": reset_movements,
        "serpentine_repeats": int(args.serpentine_repeats),
        "serpentine_horizontal_steps": int(args.serpentine_horizontal_steps),
        "captures": captures,
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Manifest: {session_dir / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
