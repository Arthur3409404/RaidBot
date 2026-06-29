from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time

import pygetwindow as gw

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(PACKAGE_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

os.chdir(PROJECT_ROOT)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from raid_bot.mainframe import (
    BOT_PID_FILE,
    RAID_WINDOW_TITLE,
    RESTART_HELPER_ARG,
    RSL_Bot_Mainframe,
    _validate_raid_window_size,
    run_restart_helper,
)
from raid_bot.core.runtime_startup import close_discord_desktop_app


def _write_pid_file():
    os.makedirs(os.path.dirname(BOT_PID_FILE), exist_ok=True)
    with open(BOT_PID_FILE, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))


def _remove_pid_file():
    if not os.path.exists(BOT_PID_FILE):
        return

    try:
        with open(BOT_PID_FILE, "r", encoding="utf-8") as handle:
            recorded_pid = handle.read().strip()
    except OSError:
        return

    if recorded_pid != str(os.getpid()):
        return

    try:
        os.remove(BOT_PID_FILE)
    except OSError:
        pass


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False

    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    output = result.stdout or b""
    if isinstance(output, str):
        output = output.encode("utf-8", errors="replace")
    output = output.strip()
    return bool(output and not output.startswith(b"INFO:"))


def _kill_pid_tree(pid: int) -> None:
    if pid <= 0:
        return

    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _wait_for_pid_exit(pid: int, attempts: int = 20) -> bool:
    for _ in range(max(0, attempts)):
        if not _is_pid_running(pid):
            return True
        time.sleep(0.5)
    return not _is_pid_running(pid)


def _read_running_instance_pid() -> int | None:
    if not os.path.exists(BOT_PID_FILE):
        return None

    try:
        with open(BOT_PID_FILE, "r", encoding="utf-8") as handle:
            raw_pid = handle.read().strip()
    except OSError:
        return None

    if not raw_pid:
        return None

    try:
        pid = int(raw_pid)
    except ValueError:
        return None

    if pid == os.getpid():
        return None
    if _is_pid_running(pid):
        return pid
    return None


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == RESTART_HELPER_ARG:
        return run_restart_helper(int(sys.argv[2]))

    existing_pid = _read_running_instance_pid()
    if existing_pid:
        _kill_pid_tree(existing_pid)
        _wait_for_pid_exit(existing_pid)

    close_discord_desktop_app()
    _write_pid_file()
    atexit.register(_remove_pid_file)

    windows = gw.getWindowsWithTitle(RAID_WINDOW_TITLE)
    if not windows:
        print("Raid window not found at startup.", flush=True)
        return 1

    if not _validate_raid_window_size(windows[0], context="startup"):
        return 1

    bot = RSL_Bot_Mainframe()
    bot.start_main_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
