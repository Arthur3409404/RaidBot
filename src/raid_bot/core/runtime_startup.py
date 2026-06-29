"""Shared startup helpers for RaidBot launchers."""

from __future__ import annotations

import subprocess
import time

DISCORD_DESKTOP_PROCESS_NAMES = [
    "Discord.exe",
    "DiscordCanary.exe",
    "DiscordPTB.exe",
    "DiscordDevelopment.exe",
]


def _terminate_processes(process_names: list[str]) -> None:
    for process_name in process_names:
        subprocess.run(
            ["taskkill", "/F", "/IM", process_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _is_process_running_by_name(process_name: str) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {process_name}", "/FO", "CSV", "/NH"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    output = result.stdout or b""
    if isinstance(output, str):
        output = output.encode("utf-8", errors="replace")
    output = output.strip()
    return bool(output and not output.startswith(b"INFO:"))


def close_discord_desktop_app() -> bool:
    running_processes = [
        process_name
        for process_name in DISCORD_DESKTOP_PROCESS_NAMES
        if _is_process_running_by_name(process_name)
    ]
    if not running_processes:
        return False

    print("Closing Discord desktop app before RaidBot startup.")
    _terminate_processes(running_processes)
    time.sleep(2)
    return True
