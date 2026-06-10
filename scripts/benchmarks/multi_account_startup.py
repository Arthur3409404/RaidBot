from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import pyautogui
import pygetwindow as gw

# Ensure repo-root imports work when executed as:
# `python scripts/benchmarks/multi_account_startup.py ...`
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from raid_bot.core import runtime_launch
from raid_bot.utils import image_tools, window_tools

RAID_WINDOW_TITLE = "Raid: Shadow Legends"
PLARIUM_WINDOW_TITLE = "Plarium Play"
PLARIUM_PLAY_EXE_ENV = "PLARIUM_PLAY_EXE"
RAID_SHORTCUT_PATH_ENV = "RAID_SHORTCUT_PATH"
RAID_DESKTOP_SHORTCUT_NAME = "Raid Shadow Legends.lnk"
PLARIUM_LAUNCH_ARGS = ["-gameid=101", "-tray-start"]

# Normalized OCR boxes relative to PlariumPlay window.
PLARIUM_SQUARE_1 = (0.0532, 0.9050, 0.0441, 0.0437)
PLARIUM_SQUARE_2 = (0.0266, 0.7729, 0.0278, 0.0459)
PLARIUM_SQUARE_3 = (0.0616, 0.7718, 0.0266, 0.0459)
PLARIUM_SQUARE_4 = (0.0961, 0.7729, 0.0284, 0.0459)
PLARIUM_SQUARE_5 = (0.1275, 0.7686, 0.0296, 0.0491)
PLARIUM_SQUARE_6 = (0.1644, 0.7686, 0.0272, 0.0524)
PLARIUM_SQUARE_7 = (0.0284, 0.8308, 0.0242, 0.0524)
PLARIUM_SQUARE_8 = (0.0236, 0.7238, 0.0749, 0.0415)

ACCOUNT_SQUARES = (
    PLARIUM_SQUARE_2,
    PLARIUM_SQUARE_3,
    PLARIUM_SQUARE_4,
    PLARIUM_SQUARE_5,
    PLARIUM_SQUARE_6,
    PLARIUM_SQUARE_7,
)
KNOWN_ACCOUNTS = ("artus", "artus2", "artus3", "artus4", "artus5", "artus6", "artus7")
HOVER_SETTLE_SECONDS = 0.35
OCR_AFTER_CLICK_DELAY_SECONDS = 0.9
PLARIUM_POST_LAUNCH_WAIT_SECONDS = 15.0


def _normalize_account_name(value: str | None) -> str:
    if not value:
        return ""
    lowered = value.lower()
    no_whitespace = "".join(lowered.split())
    # Preserve digits while removing punctuation/artifacts from OCR.
    return "".join(ch for ch in no_whitespace if ch.isalnum())


def _account_names_match_exact(observed: str | None, requested: str | None) -> bool:
    observed_norm = _normalize_account_name(observed)
    requested_norm = _normalize_account_name(requested)
    return bool(observed_norm) and observed_norm == requested_norm


def _normalized_box_to_absolute_rect(window, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    rel_left, rel_top, rel_width, rel_height = box
    abs_left = int(round(window.left + (rel_left * window.width)))
    abs_top = int(round(window.top + (rel_top * window.height)))
    abs_width = max(1, int(round(rel_width * window.width)))
    abs_height = max(1, int(round(rel_height * window.height)))
    return abs_left, abs_top, abs_width, abs_height


def _box_center(abs_rect: tuple[int, int, int, int]) -> tuple[int, int]:
    left, top, width, height = abs_rect
    return int(left + (width / 2)), int(top + (height / 2))


def _ocr_box_text(reader, window, box: tuple[float, float, float, float], label: str) -> str:
    abs_rect = _normalized_box_to_absolute_rect(window, box)
    screenshot = pyautogui.screenshot(region=abs_rect)
    raw_text = (image_tools.get_text_from_image(reader, screenshot) or "").strip()
    normalized = _normalize_account_name(raw_text)
    print(
        f"[startup] OCR {label}: rect={abs_rect} raw='{raw_text}' normalized='{normalized}'"
    )
    return raw_text


def _highlight_plarium_window(window, loops: int = 2) -> None:
    window_tools.ensure_window_focus(window, delay=0.2)
    inset = 3
    left = int(window.left + inset)
    top = int(window.top + inset)
    right = int(window.left + window.width - inset)
    bottom = int(window.top + window.height - inset)
    border_points = ((left, top), (right, top), (right, bottom), (left, bottom), (left, top))

    print("[startup] highlighting Plarium Play window")
    for _ in range(max(1, int(loops))):
        for x, y in border_points:
            pyautogui.moveTo(x, y, duration=0.12)
    center_x, center_y = _box_center((window.left, window.top, window.width, window.height))
    pyautogui.moveTo(center_x, center_y, duration=0.1)


def _log_window_bounds(window) -> None:
    print(
        "[startup] PlariumPlay window bounds: "
        f"left={window.left} top={window.top} width={window.width} height={window.height}"
    )


def _find_window(title_substring: str):
    windows = gw.getWindowsWithTitle(title_substring)
    if not windows:
        return None
    window = windows[0]
    return window_tools.WindowObject(
        (int(window.left), int(window.top), int(window.width), int(window.height)),
        title_substring=title_substring,
    )


def _wait_for_window(title_substring: str, timeout_seconds: float):
    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        window = _find_window(title_substring)
        if window:
            return window
        time.sleep(1.0)
    return _find_window(title_substring)


def _move_window_to_default_top_left(title_substring: str, x: int = 10, y: int = 10) -> bool:
    windows = gw.getWindowsWithTitle(title_substring)
    if not windows:
        return False
    try:
        windows[0].moveTo(int(x), int(y))
        return True
    except Exception:
        return False


def _debug_record_relative_squares(window_title: str, num_squares: int = 1) -> None:
    interactive_stdin = bool(getattr(sys.stdin, "isatty", lambda: False)())

    def wait_for_enter_or_timeout(prompt: str, timeout_seconds: float = 3.0) -> None:
        if interactive_stdin:
            try:
                input(prompt)
                return
            except EOFError:
                pass
        print(
            f"{prompt} [stdin non-interactive; capturing mouse position in {timeout_seconds:.1f}s...]",
            flush=True,
        )
        time.sleep(float(timeout_seconds))

    dbg_windows = gw.getWindowsWithTitle(window_title)
    if dbg_windows:
        dbg_win = dbg_windows[0]
        dbg_left, dbg_top, dbg_width, dbg_height = (
            dbg_win.left, dbg_win.top, dbg_win.width, dbg_win.height
        )
        dbg_coords = (dbg_left, dbg_top, dbg_width, dbg_height)
        print(f"window.left/top/width/height = {dbg_coords}")
        for i in range(max(1, int(num_squares))):
            wait_for_enter_or_timeout(f"[{i + 1}/{num_squares}] Move mouse to TOP-LEFT, then Enter...")
            x1, y1 = pyautogui.position()
            wait_for_enter_or_timeout(f"[{i + 1}/{num_squares}] Move mouse to BOTTOM-RIGHT, then Enter...")
            x2, y2 = pyautogui.position()
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            rel = [
                round((left - dbg_left) / dbg_width, 4), round((top - dbg_top) / dbg_height, 4),
                round((right - left) / dbg_width, 4), round((bottom - top) / dbg_height, 4),
            ]
            print(f"square_{i + 1}: {rel}")
    else:
        print(f"Window not found: {window_title}")


def _launch_plarium_play() -> None:
    plarium_path, checked = runtime_launch.resolve_plarium_play_executable(PLARIUM_PLAY_EXE_ENV)
    if not plarium_path:
        checked_paths = ", ".join(checked) if checked else "<none>"
        raise FileNotFoundError(
            "Could not locate PlariumPlay.exe. "
            f"Checked: {checked_paths}. Set {PLARIUM_PLAY_EXE_ENV} if needed."
        )
    subprocess.Popen([plarium_path])


def _build_raid_launch_command() -> list[str]:
    return runtime_launch.build_raid_launch_command(
        plarium_play_exe_env=PLARIUM_PLAY_EXE_ENV,
        raid_shortcut_path_env=RAID_SHORTCUT_PATH_ENV,
        raid_desktop_shortcut_name=RAID_DESKTOP_SHORTCUT_NAME,
        plarium_launch_args=PLARIUM_LAUNCH_ARGS,
    )


def _build_ocr_reader():
    try:
        import easyocr  # type: ignore
    except ModuleNotFoundError:
        return None
    return easyocr.Reader(["en", "es"])


def _format_candidate_reads(reads: Iterable[tuple[int, str, str]]) -> str:
    parts: list[str] = []
    for box_idx, raw, normalized in reads:
        parts.append(f"box{box_idx}: raw='{raw}' normalized='{normalized}'")
    return "; ".join(parts) if parts else "<none>"


def _switch_to_account(reader, window, account_name: str) -> bool:
    requested_norm = _normalize_account_name(account_name)
    known_norm = {_normalize_account_name(name) for name in KNOWN_ACCOUNTS}
    if requested_norm not in known_norm:
        raise ValueError(
            f"Requested account '{account_name}' is unsupported for this benchmark. "
            f"Supported: {', '.join(KNOWN_ACCOUNTS)}"
        )

    print(f"[startup] requested target account: raw='{account_name}' normalized='{requested_norm}'")

    box1_raw = _ocr_box_text(reader, window, PLARIUM_SQUARE_1, "Box 1")
    if _account_names_match_exact(box1_raw, requested_norm):
        print("[startup] Box 1 already matches requested account; no account switching clicks required.")
        return True

    box1_center = _box_center(_normalized_box_to_absolute_rect(window, PLARIUM_SQUARE_1))
    print(
        "[startup] Box 1 mismatch; opening account selector via Box 1 center "
        f"at ({box1_center[0]}, {box1_center[1]})"
    )
    window_tools.click_at(box1_center[0], box1_center[1], delay=OCR_AFTER_CLICK_DELAY_SECONDS, window=window)

    hover_reads: list[tuple[int, str, str]] = []
    for offset, candidate_box in enumerate(ACCOUNT_SQUARES, start=2):
        candidate_rect = _normalized_box_to_absolute_rect(window, candidate_box)
        candidate_center = _box_center(candidate_rect)
        print(
            f"[startup] hover candidate Box {offset}: center=({candidate_center[0]}, {candidate_center[1]})"
        )

        window_tools.ensure_window_focus(window, delay=0.05)
        pyautogui.moveTo(candidate_center[0], candidate_center[1], duration=0.12)
        time.sleep(HOVER_SETTLE_SECONDS)

        box8_raw = _ocr_box_text(reader, window, PLARIUM_SQUARE_8, "Box 8")
        box8_norm = _normalize_account_name(box8_raw)
        hover_reads.append((offset, box8_raw, box8_norm))

        if _account_names_match_exact(box8_norm, requested_norm):
            print(
                f"[startup] match found on Box {offset}; clicking "
                f"center=({candidate_center[0]}, {candidate_center[1]})"
            )
            window_tools.click_at(
                candidate_center[0],
                candidate_center[1],
                delay=OCR_AFTER_CLICK_DELAY_SECONDS,
                window=window,
            )

            verify_raw = _ocr_box_text(reader, window, PLARIUM_SQUARE_1, "Box 1 post-select")
            if _account_names_match_exact(verify_raw, requested_norm):
                print("[startup] post-select verification succeeded via Box 1.")
                return True

            print(
                "[startup] post-select verification failed: "
                f"requested='{requested_norm}' observed='{_normalize_account_name(verify_raw)}'"
            )
            return False

    print(f"[startup] target account not found in candidate boxes. target='{requested_norm}'")
    print(f"[startup] Box 8 OCR reads: {_format_candidate_reads(hover_reads)}")
    return False


def run(
    account_name: str,
    plarium_timeout: float,
    raid_timeout: float,
    debug_squares: bool,
    num_squares: int,
    allow_manual_fallback: bool,
) -> int:
    if bool(allow_manual_fallback):
        print(
            "[startup] note: --allow-manual-fallback is ignored in this benchmark; "
            "strict OCR account verification is required before Raid launch."
        )

    print(f"[startup] launching Plarium Play and switching account to '{account_name}'")
    _launch_plarium_play()
    print(
        "[startup] waiting "
        f"{PLARIUM_POST_LAUNCH_WAIT_SECONDS:.0f}s after Plarium launch before reading window bounds"
    )
    time.sleep(PLARIUM_POST_LAUNCH_WAIT_SECONDS)

    plarium_window = _wait_for_window(PLARIUM_WINDOW_TITLE, timeout_seconds=plarium_timeout)
    if not plarium_window:
        raise TimeoutError(f"Plarium Play window not found within {plarium_timeout:.0f}s.")
    _move_window_to_default_top_left(PLARIUM_WINDOW_TITLE, x=10, y=10)
    time.sleep(1.0)
    plarium_window = _wait_for_window(PLARIUM_WINDOW_TITLE, timeout_seconds=10.0) or plarium_window
    window_tools.ensure_window_focus(plarium_window, delay=0.4)
    _log_window_bounds(plarium_window)
    _highlight_plarium_window(plarium_window)
    if bool(debug_squares):
        _debug_record_relative_squares(PLARIUM_WINDOW_TITLE, num_squares=num_squares)

    print("[startup] initializing OCR reader")
    reader = _build_ocr_reader()
    if reader is None:
        raise ModuleNotFoundError(
            "easyocr is required for benchmark account selection. "
            "Install it with 'pip install easyocr'."
        )

    print("[startup] trying account switch flow")
    switched = _switch_to_account(reader, plarium_window, account_name=account_name)
    if not switched:
        raise RuntimeError(
            f"Could not switch/verify target account '{account_name}'. "
            "Raid launch is blocked to avoid starting on the wrong account."
        )
    print(f"[startup] account '{account_name}' selected and verified")

    print("[startup] launching Raid (verified account)")
    subprocess.Popen(_build_raid_launch_command())

    raid_window = _wait_for_window(RAID_WINDOW_TITLE, timeout_seconds=raid_timeout)
    if not raid_window:
        raise TimeoutError(f"Raid window not found within {raid_timeout:.0f}s.")
    print("[startup] Raid window detected")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch Plarium Play, switch to a target account, and start Raid."
    )
    parser.add_argument("--account", default="artus3", help="Target Plarium account name.")
    parser.add_argument("--plarium-timeout", type=float, default=60.0, help="Seconds to wait for Plarium Play.")
    parser.add_argument("--raid-timeout", type=float, default=180.0, help="Seconds to wait for Raid window.")
    parser.add_argument("--debug-squares", action="store_true", help="Record relative debug squares for Plarium window.")
    parser.add_argument("--num-squares", type=int, default=1, help="Number of debug squares to record.")
    parser.add_argument(
        "--allow-manual-fallback",
        action="store_true",
        default=True,
        help="Deprecated no-op; strict OCR verification is always enforced in this benchmark.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(
        account_name=str(args.account),
        plarium_timeout=float(args.plarium_timeout),
        raid_timeout=float(args.raid_timeout),
        debug_squares=bool(args.debug_squares),
        num_squares=int(args.num_squares),
        allow_manual_fallback=bool(args.allow_manual_fallback),
    )


if __name__ == "__main__":
    raise SystemExit(main())
