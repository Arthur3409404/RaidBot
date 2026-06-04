from __future__ import annotations

import atexit
import os
import subprocess
import sys

from Raid_Bot import (
    BOT_PID_FILE,
    RESTART_HELPER_ARG,
    RSL_Bot_Mainframe,
    run_restart_helper,
)
from data.lib.core.runtime_startup import close_discord_desktop_app


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
        text=True,
        check=False,
    )
    output = (result.stdout or "").strip()
    return bool(output and not output.startswith("INFO:"))


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
        print(f"RaidBot is already running (PID {existing_pid}). Exiting duplicate launch.")
        return 1

    close_discord_desktop_app()
    _write_pid_file()
    atexit.register(_remove_pid_file)

    bot = RSL_Bot_Mainframe()
    bot.start_main_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
