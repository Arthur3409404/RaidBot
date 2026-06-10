"""Raid and Plarium launch command discovery helpers."""

from __future__ import annotations

import os
import shutil


def resolve_plarium_play_executable(plarium_play_exe_env: str):
    candidates = []

    custom_path = os.environ.get(plarium_play_exe_env)
    if custom_path:
        candidates.append(custom_path)

    from_path = shutil.which("PlariumPlay.exe")
    if from_path:
        candidates.append(from_path)

    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        candidates.extend(
            [
                os.path.join(local_appdata, "PlariumPlay", "PlariumPlay.exe"),
                os.path.join(local_appdata, "Programs", "PlariumPlay", "PlariumPlay.exe"),
            ]
        )

    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        candidates.append(
            os.path.join(user_profile, "AppData", "Local", "PlariumPlay", "PlariumPlay.exe")
        )

    for env_name in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
        program_files = os.environ.get(env_name, "")
        if not program_files:
            continue
        candidates.extend(
            [
                os.path.join(program_files, "PlariumPlay", "PlariumPlay.exe"),
                os.path.join(program_files, "Plarium Play", "PlariumPlay.exe"),
            ]
        )

    checked_paths = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue

        normalized = os.path.normpath(os.path.expandvars(os.path.expanduser(candidate)))
        key = os.path.normcase(normalized)
        if key in seen:
            continue

        seen.add(key)
        checked_paths.append(normalized)
        if os.path.isfile(normalized):
            return normalized, checked_paths

    return None, checked_paths


def resolve_raid_shortcut(raid_shortcut_path_env: str, raid_desktop_shortcut_name: str):
    candidates = []

    custom_path = os.environ.get(raid_shortcut_path_env)
    if custom_path:
        candidates.append(custom_path)

    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        candidates.append(os.path.join(user_profile, "Desktop", raid_desktop_shortcut_name))

    public_profile = os.environ.get("PUBLIC", "")
    if public_profile:
        candidates.append(os.path.join(public_profile, "Desktop", raid_desktop_shortcut_name))

    checked_paths = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue

        normalized = os.path.normpath(os.path.expandvars(os.path.expanduser(candidate)))
        key = os.path.normcase(normalized)
        if key in seen:
            continue

        seen.add(key)
        checked_paths.append(normalized)
        if os.path.isfile(normalized) and normalized.lower().endswith(".lnk"):
            return normalized, checked_paths

    return None, checked_paths


def describe_raid_launch_command(command):
    if command and command[-1].lower().endswith(".lnk"):
        return command[-1]
    return command[0] if command else "<unresolved>"


def build_raid_launch_command(
    *,
    plarium_play_exe_env: str,
    raid_shortcut_path_env: str,
    raid_desktop_shortcut_name: str,
    plarium_launch_args: list[str],
    shortcut_resolver=None,
    launcher_resolver=None,
):
    resolve_shortcut = shortcut_resolver
    if resolve_shortcut is None:
        def resolve_shortcut():
            return resolve_raid_shortcut(
                raid_shortcut_path_env,
                raid_desktop_shortcut_name,
            )

    resolve_launcher = launcher_resolver
    if resolve_launcher is None:
        def resolve_launcher():
            return resolve_plarium_play_executable(plarium_play_exe_env)

    shortcut_path, checked_paths = resolve_shortcut()
    if shortcut_path:
        return ["cmd", "/c", "start", "", shortcut_path]

    launcher_path, checked_paths = resolve_launcher()
    if launcher_path:
        return [launcher_path, *plarium_launch_args]

    _, shortcut_checked_paths = resolve_shortcut()
    checked_paths = shortcut_checked_paths + checked_paths
    checked_display = ", ".join(checked_paths) if checked_paths else "<no paths checked>"
    raise FileNotFoundError(
        "Could not locate a Raid shortcut or PlariumPlay.exe. "
        f"Checked: {checked_display}. "
        f"Set {raid_shortcut_path_env} to a valid .lnk file or "
        f"{plarium_play_exe_env} to the full launcher path."
    )
