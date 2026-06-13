# -*- coding: utf-8 -*-
"""
Raid Bot mainframe runtime.

Created on Thu Jun 19 20:04:52 2025
Refactored for stability/extensibility in 2026.
"""

from __future__ import annotations

import difflib
import atexit
import inspect
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PROJECT_ROOT)

EXPECTED_CONDA_ENV = "RaidEnv"
_ENV_RELAUNCH_MARKER = "RAID_BOT_ENV_RELAUNCH_ATTEMPTED"
_ENTRYPOINT_RELAUNCH_MARKER = "RAID_BOT_ENTRYPOINT_RELAUNCH"
RAID_ACCOUNT_NAME_ENV = "RAID_ACCOUNT_NAME"
BOT_PID_FILE = os.path.join("data", "tmp", "run_bot.pid")
RAID_WINDOW_TITLE = "Raid: Shadow Legends"
PLARIUM_WINDOW_TITLE = "Plarium Play"
PLARIUM_PLAY_EXE_ENV = "PLARIUM_PLAY_EXE"
RAID_SHORTCUT_PATH_ENV = "RAID_SHORTCUT_PATH"
RAID_DESKTOP_SHORTCUT_NAME = "Raid Shadow Legends.lnk"
RUN_BOT_SCRIPT_NAME = "run_bot.py"
RAID_BOT_SCRIPT_NAME = "Raid_Bot.py"
RESTART_HELPER_ARG = "--restart-helper"
RESTART_NOTIFY_ENV = "RAID_BOT_NOTIFY_RESTART_SUCCESS"
RESTART_REPLACE_PID_ENV = "RAID_BOT_REPLACE_PID"
DISCORD_GUILD_NAME = "Discord_Sandbox"
DISCORD_CHANNEL_NAME = "raid_sandbox"
PLARIUM_LAUNCH_ARGS = ["-gameid=101", "-tray-start"]
CONNECTIVITY_ONLINE_POLL_SECONDS = 5.0
CONNECTIVITY_OUTAGE_CONFIRM_SECONDS = 60.0
CONNECTIVITY_RETRY_INTERVAL_SECONDS = 10 * 60
SLOW_LAPTOP_MARKER_FILE = os.path.join("data", ".raidbot_slow_laptop")
DAILY_TASK_STATE_DIR = os.path.join("data", "state", "daily_tasks")


def _is_running_in_expected_env(expected_env_name=EXPECTED_CONDA_ENV):
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env == expected_env_name:
        return True

    # Fallback for cases where CONDA_DEFAULT_ENV is missing.
    return os.path.basename(sys.prefix).lower() == expected_env_name.lower()


def _resolve_conda_executable():
    candidates = []

    conda_exe_env = os.environ.get("CONDA_EXE")
    if conda_exe_env:
        candidates.append(conda_exe_env)

    conda_from_path = shutil.which("conda")
    if conda_from_path:
        candidates.append(conda_from_path)

    user_profile = os.environ.get("USERPROFILE", "")
    if user_profile:
        candidates.extend(
            [
                os.path.join(user_profile, "anaconda3", "Scripts", "conda.exe"),
                os.path.join(user_profile, "anaconda3", "condabin", "conda.bat"),
                os.path.join(user_profile, "miniconda3", "Scripts", "conda.exe"),
                os.path.join(user_profile, "miniconda3", "condabin", "conda.bat"),
            ]
        )

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _resolve_launch_target():
    if sys.argv and sys.argv[0]:
        candidate = os.path.abspath(sys.argv[0])
        if os.path.isfile(candidate):
            return candidate
    return os.path.abspath(__file__)


def _resolve_run_bot_script():
    return os.path.join(PROJECT_ROOT, RAID_BOT_SCRIPT_NAME)


def _spawn_run_bot_process(*extra_args: str, env: dict | None = None):
    command = [sys.executable, _resolve_run_bot_script(), *extra_args]
    return subprocess.Popen(command, cwd=PROJECT_ROOT, env=env)


def _is_process_running(pid: int) -> bool:
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


def _wait_for_process_exit(pid: int, timeout_seconds: float = 20.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _is_process_running(pid):
            return True
        time.sleep(0.5)
    return not _is_process_running(pid)


def _kill_process_by_pid(pid: int) -> None:
    if pid <= 0:
        return

    subprocess.run(
        ["taskkill", "/F", "/PID", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _kill_process_tree_by_pid(pid: int) -> None:
    if pid <= 0:
        return

    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _clear_pid_file_for_pid(pid: int) -> bool:
    if pid <= 0 or not os.path.exists(BOT_PID_FILE):
        return False

    try:
        with open(BOT_PID_FILE, "r", encoding="utf-8") as handle:
            recorded_pid = handle.read().strip()
    except OSError:
        return False

    if recorded_pid != str(pid):
        return False

    try:
        os.remove(BOT_PID_FILE)
        return True
    except OSError:
        return False


def _write_current_pid_file() -> None:
    os.makedirs(os.path.dirname(BOT_PID_FILE), exist_ok=True)
    with open(BOT_PID_FILE, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))


def _remove_current_pid_file() -> None:
    _clear_pid_file_for_pid(os.getpid())


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
    if _is_process_running(pid):
        return pid
    return None


def _is_restart_replacement_launch() -> bool:
    return os.environ.get(RESTART_NOTIFY_ENV) == "1"


def _is_entrypoint_relaunch() -> bool:
    return os.environ.get(_ENTRYPOINT_RELAUNCH_MARKER) == "1"


def _terminate_processes(process_names: list[str]) -> None:
    for process_name in process_names:
        subprocess.run(
            ["taskkill", "/F", "/IM", process_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _resolve_plarium_play_executable():
    return runtime_launch.resolve_plarium_play_executable(PLARIUM_PLAY_EXE_ENV)


def _resolve_raid_shortcut():
    return runtime_launch.resolve_raid_shortcut(
        RAID_SHORTCUT_PATH_ENV,
        RAID_DESKTOP_SHORTCUT_NAME,
    )


def _describe_raid_launch_command(command):
    return runtime_launch.describe_raid_launch_command(command)


def _build_raid_launch_command():
    return runtime_launch.build_raid_launch_command(
        plarium_play_exe_env=PLARIUM_PLAY_EXE_ENV,
        raid_shortcut_path_env=RAID_SHORTCUT_PATH_ENV,
        raid_desktop_shortcut_name=RAID_DESKTOP_SHORTCUT_NAME,
        plarium_launch_args=PLARIUM_LAUNCH_ARGS,
        shortcut_resolver=_resolve_raid_shortcut,
        launcher_resolver=_resolve_plarium_play_executable,
    )


def _wait_for_raid_window(timeout_seconds: float = 180.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if gw.getWindowsWithTitle(RAID_WINDOW_TITLE):
            return True
        time.sleep(2)
    return bool(gw.getWindowsWithTitle(RAID_WINDOW_TITLE))


def _launch_raid_and_wait_until_window(
    *,
    launch_command,
    cwd=None,
    timeout_seconds: float = 180.0,
    retry_delay_seconds: float = 5.0,
    log=None,
):
    attempt = 0
    while True:
        attempt += 1
        if log:
            log.warning("Launching Raid restart attempt %s.", attempt)
        else:
            print(f"Launching Raid restart attempt {attempt}.", flush=True)

        _terminate_processes(["Raid.exe", "PlariumPlay.exe"])
        time.sleep(retry_delay_seconds)

        subprocess.Popen(launch_command, cwd=cwd)

        if _wait_for_raid_window(timeout_seconds=timeout_seconds):
            return

        message = (
            f"Raid window did not appear within {int(timeout_seconds)} seconds during restart; "
            "retrying launch."
        )
        if log:
            log.warning(message)
        else:
            print(message, flush=True)


def run_restart_helper(parent_pid: int) -> int:
    if parent_pid > 0 and not _wait_for_process_exit(parent_pid, timeout_seconds=15.0):
        _kill_process_by_pid(parent_pid)
        _wait_for_process_exit(parent_pid, timeout_seconds=10.0)
    _clear_pid_file_for_pid(parent_pid)

    _launch_raid_and_wait_until_window(
        launch_command=_build_raid_launch_command(),
        cwd=PROJECT_ROOT,
        timeout_seconds=180.0,
    )

    env = os.environ.copy()
    env[RESTART_NOTIFY_ENV] = "1"
    env[RESTART_REPLACE_PID_ENV] = str(parent_pid)
    _spawn_run_bot_process(env=env)
    return 0


def _ensure_expected_conda_env():
    if _is_running_in_expected_env():
        return

    if os.environ.get(_ENV_RELAUNCH_MARKER) == "1":
        raise RuntimeError(
            f"Unable to start inside '{EXPECTED_CONDA_ENV}'. "
            "Please activate the environment manually and rerun."
        )

    conda_exe = _resolve_conda_executable()
    if not conda_exe:
        raise RuntimeError(
            "Conda was not found. Add Conda to PATH, set CONDA_EXE, or start this bot "
            f"from the '{EXPECTED_CONDA_ENV}' environment."
        )

    relaunch_env = os.environ.copy()
    relaunch_env[_ENV_RELAUNCH_MARKER] = "1"
    relaunch_env[_ENTRYPOINT_RELAUNCH_MARKER] = "1"

    cmd = [
        conda_exe,
        "run",
        "--no-capture-output",
        "-n",
        EXPECTED_CONDA_ENV,
        "python",
        _resolve_launch_target(),
        *sys.argv[1:],
    ]

    try:
        use_shell = conda_exe.lower().endswith(".bat")
        if use_shell:
            cmd = subprocess.list2cmdline(cmd)
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=relaunch_env, check=False, shell=use_shell)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Conda was not found. Please install Conda or start this bot "
            f"from the '{EXPECTED_CONDA_ENV}' environment."
        ) from exc

    sys.exit(result.returncode)


def _load_berlin_timezone():
    try:
        return ZoneInfo("Europe/Berlin")
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(
            "Missing time zone data for 'Europe/Berlin'. Install 'tzdata' in the "
            "active Python environment, or start the bot through Conda so "
            f"'{EXPECTED_CONDA_ENV}' is used."
        ) from exc


_ensure_expected_conda_env()
BERLIN_TZ = _load_berlin_timezone()
UTC_TZ = timezone.utc

import easyocr
import pyautogui
import pygetwindow as gw

import raid_bot.handlers.error_handler as error_handler
import raid_bot.handlers.raid_calendar_handler as raid_calendar_handler
import raid_bot.core.daily_tasks as daily_tasks
import raid_bot.core.pushrank as pushrank
from raid_bot.core import BotCommandRouter, runtime_discord, runtime_launch, runtime_reporting
from raid_bot.core import runtime_connectivity
from raid_bot.core.runtime_startup import close_discord_desktop_app
from raid_bot.modes import (
    arena_tools,
    chimera_tools,
    cursedcity_tools,
    demonlord_tools,
    doomtower_tools,
    dungeon_tools,
    factionwars_tools,
    grimforest_tools,
    hydra_tools,
)
from raid_bot.utils import file_tools, image_tools, window_tools

logger = logging.getLogger(__name__)


def _constructor_config(cls, config: dict) -> dict:
    parameters = inspect.signature(cls.__init__).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(config)

    accepted_keys = {
        key
        for key, param in parameters.items()
        if key != "self"
        and param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return {key: value for key, value in config.items() if key in accepted_keys}


class _DiscordConsoleFilter(logging.Filter):
    """Hide Discord-related log spam from terminal output."""

    DISCORD_MESSAGE_PATTERNS = (
        "[Discord",
        "Discord remote override",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        logger_name = (record.name or "").lower()
        if logger_name.startswith("discord") or "error_handler" in logger_name:
            return False

        message = record.getMessage()
        for pattern in self.DISCORD_MESSAGE_PATTERNS:
            if pattern in message:
                return False
        return True


def _configure_logging(verbose: bool = True) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()

    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        root.setLevel(level)

    for handler in root.handlers:
        if any(isinstance(existing_filter, _DiscordConsoleFilter) for existing_filter in handler.filters):
            continue
        handler.addFilter(_DiscordConsoleFilter())


@dataclass
class ModeSpec:
    key: str
    display_name: str
    run_flag_key: str
    executor: Callable[[dict, float], None]
    condition: Callable[[], bool] | None = None


class RestartRequested(Exception):
    """Raised when Discord or runtime asks for a full Raid process restart."""

    def __init__(self, trigger: str):
        super().__init__(trigger)
        self.trigger = trigger


class NullDiscordRemoteOverride:
    """No-op Discord transport used when token config is unavailable."""

    def start(self):
        return

    def stop(self):
        return

    def get_last_command(self):
        return None

    def pop_last_command(self):
        return None

    def clear_last_command(self):
        return

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        return False

    def is_running(self) -> bool:
        return False

    def send_message(self, text: str):
        logger.debug("[Discord-Disabled] %s", text)

    def send_message_blocking(self, text: str, timeout: float = 15.0) -> bool:
        logger.debug("[Discord-Disabled] %s", text)
        return False

    def send_image(self, image_path: str, caption: str | None = None):
        logger.debug("[Discord-Disabled] image=%s caption=%s", image_path, caption)


def load_discord_token(secret_file=".ssh"):
    """
    Load Discord token from a local root secrets file.
    Supports either:
    - DISCORD_TOKEN=...
    - plain token as first non-empty line
    """
    return runtime_discord.load_discord_token(secret_file)


class RSL_Bot_Mainframe:
    MAIN_MENU_NAMES = {
        "Campaign": "Campana",
        "Dungeons": "Mazmorras",
        "FactionWars": "Guerras de Facciones",
        "Arena": "Arena",
        "ClanBoss1": "Jefes",
        "ClanBoss2": "e Clan",
        "DoomTower": "Torre del Destino",
        "CursedCity": "Ciudad Maldita",
        "Siege": "Asedio",
        "GrimForest": "Bosque Lugubre",
        "GrimmForest": "Bosque Lugubre",
    }

    SEARCH_AREAS = {
        "menu_name": [0.008, 0.034, 0.23, 0.06],
        "go_to_higher_menu": [0.928, 0.031, 0.046, 0.039],
        "go_to_bastion": [0.903, 0.9, 0.064, 0.059],
        "bastion_to_main_menu": [0.808, 0.904, 0.168, 0.074],
        "detect_doomtower_rotation": [0.121, 0.696, 0.189, 0.035],
        "quest_menu": [0.259, 0.911, 0.066, 0.068],
        "quest_menu_name": [0.013, 0.032, 0.125, 0.038],
        "daily_quest_menu": [0.033, 0.105, 0.151, 0.074],
        "weekly_quest_menu": [0.039, 0.214, 0.143, 0.059],
        "monthly_quest_menu": [0.036, 0.313, 0.147, 0.067],
        "advanced_quest_menu": [0.033, 0.417, 0.15, 0.073],
        "claim_quest_rewards": [0.455, 0.924, 0.3, 0.054],
        "time_gated_reward_menu": [0.894, 0.817, 0.051, 0.067],
        "time_gated_reward_menu_name": [0.707, 0.733, 0.142, 0.04],
        "guardian_faction_menu_name": [0.01, 0.033, 0.24, 0.039],
        "guardian_faction_character_1": [0.194, 0.797, 0.147, 0.073],
        "guardian_faction_character_2": [0.353, 0.797, 0.15, 0.071],
        "guardian_faction_character_3": [0.513, 0.796, 0.148, 0.073],
        "guardian_faction_character_4": [0.673, 0.797, 0.147, 0.071],
        "guardian_faction_character_5": [0.833, 0.798, 0.146, 0.071],
        "pov": [0, 0, 1, 1],
        "main_menu_labels": [0.007, 0.27, 0.984, 0.044],
        "classic_arena": [0.055, 0.407, 0.111, 0.173],
        "live_arena": [0.452, 0.351, 0.123, 0.2],
        "tagteam_arena": [0.829, 0.409, 0.105, 0.177],
        "clanboss_DemonLord": [0.007, 0.307, 0.072, 0.196],
        "clanboss_Chimera": [0.907, 0.307, 0.072, 0.196],
        "factionwars_key_popup_toggle": [0.7593, 0.031, 0.1335, 0.0405],
        "factionwars_key_counter": [0.6867, 0.333, 0.0648, 0.0235],
        "advert2": [0.277, 0.622, 0.215, 0.083],
        "buy_mystery_shard": [0.5123, 0.5767, 0.2052, 0.0715],
    }

    def __init__(self, title_substring="Raid: Shadow Legends"):
        requested_account = os.environ.get(
            RAID_ACCOUNT_NAME_ENV,
            file_tools.DEFAULT_MAIN_ACCOUNT_NAME,
        )
        self.profile_resolution = file_tools.resolve_profile_params_file(
            account_name=requested_account,
            allow_main_profile_fallback_for_missing_account=False,
        )
        self.account_name = self.profile_resolution.account_name
        self.profile_account_name = self.profile_resolution.selected_profile_account_name
        self.param_file = str(self.profile_resolution.selected_param_file)
        self.param_store = file_tools.ParameterStore(self.param_file)
        self.params_flat = self.param_store.get_flat_copy()
        self.params = self.param_store.get_grouped_copy()
        self.verbose = bool(self.params.get("mainframe", {}).get("verbose", True))
        _configure_logging(self.verbose)
        self.log = logging.getLogger(self.__class__.__name__)
        self._log_profile_resolution()
        self.daily_log_path = file_tools.get_daily_log_path(datetime.now(UTC_TZ))
        self.daily_task_state_path = os.path.join(
            DAILY_TASK_STATE_DIR,
            f"{file_tools.normalize_account_name(self.profile_account_name)}.json",
        )
        file_tools.ensure_daily_log_header(
            self.daily_log_path,
            self._build_daily_log_header_lines(),
        )
        self._append_daily_log_lines(
            [
                "",
                f"[{self._format_utc_timestamp()}] Mainframe started",
                f"account={self.account_name} profile={self.profile_account_name} param_file={self.param_file}",
            ]
        )
        self._last_dungeon_override_signature = None
        self._last_known_dungeon_tournament = None
        self._dungeon_fusion_active = False

        self.reader = easyocr.Reader(["en"])
        self.running = True
        self.main_loop_running = False
        self.main_loop_stopped = False
        self.manual_mode = False
        self.current_mode = "boot"
        self.start_time = time.time()
        self.handler_init_time = time.time()
        self.last_loop_error: str | None = None
        self.navigate_bastion_once_after_restart = True
        self.search_areas = dict(self.SEARCH_AREAS)
        self._restart_lock = threading.Lock()
        self._restart_in_progress = False
        self._connectivity_pause_notified = False
        self.connectivity_supervisor = None
        self._last_discord_restart_attempt_ts = 0.0
        self._last_connectivity_supervisor_restart_ts = 0.0
        self._is_slow_laptop = os.path.exists(SLOW_LAPTOP_MARKER_FILE)

        self.window = self._resolve_window(title_substring)
        self.coords = (
            (self.window.left, self.window.top, self.window.width, self.window.height)
            if self.window
            else None
        )
        if self.coords:
            self.log.info("Window Coordinates: %s", self.coords)
        else:
            self.log.warning("Raid window not found during initialization.")

        self._synchronize_dungeon_tournament_override(reason="startup")
        self._init_mode_bots()
        self.mode_specs = self._build_mode_specs()

        self.error_handler = error_handler.RSL_Bot_ErrorHandler(
            reader=self.reader,
            window=self.window,
        )
        self.screen_drift = self.params.get("mainframe", {}).get("screen_drift", [0, 0, 0, 0])

        self.discord_override = self._create_discord_override()
        self.discord_override.start()
        self.command_router = BotCommandRouter(self)

        self.raid_path = None
        try:
            self.raid_path = _build_raid_launch_command()
            self.log.info(
                "Raid launcher resolved: %s",
                _describe_raid_launch_command(self.raid_path),
            )
        except FileNotFoundError as exc:
            self.log.warning("Raid launcher path unresolved at startup: %s", exc)

        self._start_error_checker()
        self._start_connectivity_supervisor()

    def _resolve_window(self, title_substring: str):
        detected = window_tools.find_window(title_substring)
        if not detected:
            return None
        return window_tools.WindowObject(detected, title_substring=title_substring)

    def _create_discord_override(self):
        try:
            token = load_discord_token(".ssh")
            return error_handler.DiscordRemoteOverride(
                token=token,
                guild_name=DISCORD_GUILD_NAME,
                channel_name=DISCORD_CHANNEL_NAME,
            )
        except Exception as exc:
            self.log.warning("Discord remote override disabled: %s", exc)
            return NullDiscordRemoteOverride()

    def _init_mode_bots(self):
        self.classic_arena_bot = arena_tools.RSL_Bot_ClassicArena(
            reader=self.reader,
            window=self.window,
            param_file=self.param_file,
            **_constructor_config(
                arena_tools.RSL_Bot_ClassicArena,
                self.params.get("classic_arena", {}),
            ),
        )
        self.tagteam_arena_bot = arena_tools.RSL_Bot_TagTeamArena(
            reader=self.reader,
            window=self.window,
            param_file=self.param_file,
            **_constructor_config(
                arena_tools.RSL_Bot_TagTeamArena,
                self.params.get("tagteam_arena", {}),
            ),
        )
        self.live_arena_bot = arena_tools.RSL_Bot_LiveArena(
            reader=self.reader,
            window=self.window,
            **_constructor_config(
                arena_tools.RSL_Bot_LiveArena,
                self.params.get("live_arena", {}),
            ),
        )
        self.dungeon_bot = dungeon_tools.RSL_Bot_Dungeons(
            reader=self.reader,
            window=self.window,
            **_constructor_config(
                dungeon_tools.RSL_Bot_Dungeons,
                self.params.get("dungeons", {}),
            ),
        )
        self.dungeon_bot.fusion_active = self._dungeon_fusion_active
        self.factionwars_bot = factionwars_tools.RSL_Bot_FactionWars(
            reader=self.reader,
            window=self.window,
            **_constructor_config(
                factionwars_tools.RSL_Bot_FactionWars,
                self.params.get("faction_wars", {}),
            ),
        )
        self.factionwars_bot.persist_stage_update_callback = (
            self._persist_faction_wars_stage_update
        )
        self.demonlord_bot = demonlord_tools.RSL_Bot_DemonLord(
            reader=self.reader,
            window=self.window,
            **_constructor_config(
                demonlord_tools.RSL_Bot_DemonLord,
                self.params.get("demon_lord", {}),
            ),
        )
        self.hydra_bot = hydra_tools.RSL_Bot_Hydra(
            reader=self.reader,
            window=self.window,
            **_constructor_config(
                hydra_tools.RSL_Bot_Hydra,
                self.params.get("hydra", {}),
            ),
        )
        self.chimera_bot = chimera_tools.RSL_Bot_Chimera(
            reader=self.reader,
            window=self.window,
            **_constructor_config(
                chimera_tools.RSL_Bot_Chimera,
                self.params.get("chimera", {}),
            ),
        )
        self.doomtower_bot = doomtower_tools.RSL_Bot_DoomTower(
            reader=self.reader,
            window=self.window,
            **_constructor_config(
                doomtower_tools.RSL_Bot_DoomTower,
                self.params.get("doom_tower", {}),
            ),
        )
        self.cursedcity_bot = cursedcity_tools.RSL_Bot_CursedCity(
            reader=self.reader,
            window=self.window,
            **_constructor_config(
                cursedcity_tools.RSL_Bot_CursedCity,
                self.params.get("cursed_city", {}),
            ),
        )
        self.grimforest_bot = grimforest_tools.RSL_Bot_GrimForest(
            reader=self.reader,
            window=self.window,
            **_constructor_config(
                grimforest_tools.RSL_Bot_GrimForest,
                self.params.get("grim_forest", {}),
            ),
        )

        self.bots = [
            self.classic_arena_bot,
            self.tagteam_arena_bot,
            self.live_arena_bot,
            self.dungeon_bot,
            self.factionwars_bot,
            self.demonlord_bot,
            self.hydra_bot,
            self.chimera_bot,
            self.doomtower_bot,
            self.cursedcity_bot,
            self.grimforest_bot,
        ]

        self.group_to_bot = {
            "classic_arena": self.classic_arena_bot,
            "tagteam_arena": self.tagteam_arena_bot,
            "live_arena": self.live_arena_bot,
            "dungeons": self.dungeon_bot,
            "faction_wars": self.factionwars_bot,
            "demon_lord": self.demonlord_bot,
            "hydra": self.hydra_bot,
            "chimera": self.chimera_bot,
            "doom_tower": self.doomtower_bot,
            "cursed_city": self.cursedcity_bot,
            "grim_forest": self.grimforest_bot,
        }

    def _build_mode_specs(self) -> list[ModeSpec]:
        return [
            ModeSpec(
                key="classic_arena",
                display_name="Classic Arena",
                run_flag_key="classic_arena",
                executor=self._run_mode_classic_arena,
            ),
            ModeSpec(
                key="tagteam_arena",
                display_name="Tag Team Arena",
                run_flag_key="tagteam_arena",
                executor=self._run_mode_tagteam_arena,
            ),
            ModeSpec(
                key="live_arena",
                display_name="Live Arena",
                run_flag_key="live_arena",
                executor=self._run_mode_live_arena,
            ),
            ModeSpec(
                key="dungeons",
                display_name="Dungeons",
                run_flag_key="dungeons",
                executor=self._run_mode_dungeons,
                condition=lambda: not self.params.get("run", {}).get(
                    "effective_unit_leveling", False
                ),
            ),
            ModeSpec(
                key="factionwars",
                display_name="Faction Wars",
                run_flag_key="factionwars",
                executor=self._run_mode_factionwars,
            ),
            ModeSpec(
                key="demonlord",
                display_name="Demon Lord",
                run_flag_key="demonlord",
                executor=self._run_mode_demonlord,
            ),
            ModeSpec(
                key="hydra",
                display_name="Hydra",
                run_flag_key="hydra",
                executor=self._run_mode_hydra,
            ),
            ModeSpec(
                key="chimera",
                display_name="Chimera",
                run_flag_key="chimera",
                executor=self._run_mode_chimera,
            ),
            ModeSpec(
                key="cursedcity",
                display_name="Cursed City",
                run_flag_key="cursedcity",
                executor=self._run_mode_cursedcity,
            ),
            ModeSpec(
                key="grimforest",
                display_name="Grim Forest",
                run_flag_key="grimforest",
                executor=self._run_mode_grimforest,
            ),
            ModeSpec(
                key="doomtower",
                display_name="Doom Tower",
                run_flag_key="doomtower",
                executor=self._run_mode_doomtower,
            ),
        ]

    def _log_profile_resolution(self) -> None:
        if self.profile_resolution.created_profiles_directory:
            self.log.info("Created profile params directory: %s", file_tools.PROFILES_DIR)

        self.log.info(
            (
                "Loaded profile params: account='%s', profile_account='%s', file='%s'."
            ),
            self.account_name,
            self.profile_account_name,
            self.param_file,
        )

        if self.profile_resolution.migrated_legacy:
            self.log.info(
                "Migrated legacy params file '%s' -> '%s'.",
                self.profile_resolution.legacy_param_file,
                self.profile_resolution.main_profile_file,
            )
        elif self.profile_resolution.used_legacy_fallback:
            self.log.warning(
                "Using legacy params fallback file: %s",
                self.profile_resolution.legacy_param_file,
            )

        if self.profile_resolution.used_main_profile_fallback:
            self.log.warning(
                (
                    "Missing profile for requested account '%s'. "
                    "Falling back to main profile '%s'."
                ),
                self.account_name,
                self.profile_resolution.main_profile_file,
            )

        for missing_path in self.profile_resolution.missing_profile_files:
            self.log.warning("Missing profile params file: %s", missing_path)

        for generated_path in self.profile_resolution.generated_secondary_profiles:
            generated_name = generated_path.stem.split("_params_mainframe")[0]
            self.log.info(
                "Generated secondary profile params for account '%s': %s",
                generated_name,
                generated_path,
            )

    def _build_daily_log_header_lines(self) -> list[str]:
        today = datetime.now(UTC_TZ).strftime("%Y-%m-%d")
        return [
            f"# RaidBot daily log for {today} UTC",
            f"# primary_account={self.account_name}",
            f"# profile_account={self.profile_account_name}",
            f"# params_file={self.param_file}",
            "# entries are shared across profiles for the same UTC day",
            "",
        ]

    def _format_utc_timestamp(self) -> str:
        return datetime.now(UTC_TZ).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _append_daily_log_lines(self, lines: list[str] | tuple[str, ...] | str) -> None:
        try:
            file_tools.append_daily_log_lines(self.daily_log_path, lines)
        except Exception as exc:
            self.log.warning("Failed to append daily log entry: %s", exc)

    def _format_stat_line(self, label: str, value) -> str:
        return f"{label}={value}"

    def _format_mode_daily_summary(self, mode_key: str) -> list[str]:
        timestamp = self._format_utc_timestamp()
        lines = [
            "",
            f"[{timestamp}] mode={mode_key}",
            f"account={self.account_name} profile={self.profile_account_name}",
        ]

        if mode_key == "dungeons":
            bot = self.dungeon_bot
            encounter = getattr(bot, "dungeon", "unknown")
            difficulty = getattr(bot, "difficulty", "unknown")
            level = getattr(bot, "level", None)
            event_level = getattr(bot, "eventdungeon_level", None)
            resolved_level = event_level if str(encounter).strip().lower() == "event_dungeon" else level
            lines.extend(
                [
                    f"run={encounter} difficulty={difficulty} level={resolved_level}",
                    self._format_stat_line("battles_done", getattr(bot, "battles_done", 0)),
                    self._format_stat_line("battles_won", getattr(bot, "battles_won", 0)),
                    self._format_stat_line(
                        "battles_lost",
                        max(0, getattr(bot, "battles_done", 0) - getattr(bot, "battles_won", 0)),
                    ),
                ]
            )
            if getattr(bot, "energy", None) is not None:
                lines.append(self._format_stat_line("energy", getattr(bot, "energy")))
            if getattr(bot, "last_run_energy_cost", None) is not None:
                lines.append(
                    self._format_stat_line("last_run_energy_cost", getattr(bot, "last_run_energy_cost"))
                )
            return lines

        if mode_key == "factionwars":
            bot = self.factionwars_bot
            lines.extend(
                [
                    self._format_stat_line("battles_done", getattr(bot, "battles_done", 0)),
                    self._format_stat_line("battles_won", getattr(bot, "battles_won", 0)),
                    self._format_stat_line(
                        "battles_lost",
                        max(0, getattr(bot, "battles_done", 0) - getattr(bot, "battles_won", 0)),
                    ),
                ]
            )
            farm_stages = getattr(bot, "farm_stages", {})
            if isinstance(farm_stages, dict) and farm_stages:
                summary = ", ".join(
                    f"{faction}:{stage}/{difficulty}"
                    for faction, (stage, difficulty) in sorted(farm_stages.items())
                )
                lines.append(f"farm_stages={summary}")
            progress_factions = getattr(bot, "progress_mode_factions", None)
            if progress_factions:
                lines.append(f"progress_mode_factions={', '.join(progress_factions)}")
            return lines

        if mode_key in {"classic_arena", "tagteam_arena", "live_arena"}:
            bot = self.group_to_bot.get(mode_key)
            if bot is not None:
                battles_done = getattr(bot, "battles_done", 0)
                battles_won = getattr(bot, "battles_won", 0)
                lines.extend(
                    [
                        self._format_stat_line("battles_done", battles_done),
                        self._format_stat_line("battles_won", battles_won),
                        self._format_stat_line("battles_lost", max(0, battles_done - battles_won)),
                    ]
                )
            return lines

        if mode_key == "demonlord":
            bot = self.demonlord_bot
            cleared = getattr(bot, "demonlord_encounters_cleared", [])
            if isinstance(cleared, list):
                lines.append(f"cleared_difficulties={', '.join(map(str, cleared)) or 'none'}")
            lines.extend(
                [
                    self._format_stat_line("keys", getattr(bot, "num_of_keys", 0)),
                    self._format_stat_line("encounters_cleared", len(cleared) if isinstance(cleared, list) else 0),
                ]
            )
            return lines

        if mode_key == "hydra":
            bot = self.hydra_bot
            cleared = getattr(bot, "hydra_encounters_cleared", [])
            if isinstance(cleared, list):
                lines.append(f"cleared_difficulties={', '.join(map(str, cleared)) or 'none'}")
            lines.extend(
                [
                    self._format_stat_line("keys", getattr(bot, "num_of_keys", 0)),
                    self._format_stat_line("lost_encounter", getattr(bot, "lost_encounter", False)),
                ]
            )
            return lines

        if mode_key == "chimera":
            bot = self.chimera_bot
            if hasattr(bot, "battles_done"):
                battles_done = getattr(bot, "battles_done", 0)
                battles_won = getattr(bot, "battles_won", 0)
                lines.extend(
                    [
                        self._format_stat_line("battles_done", battles_done),
                        self._format_stat_line("battles_won", battles_won),
                        self._format_stat_line("battles_lost", max(0, battles_done - battles_won)),
                    ]
                )
            return lines

        if mode_key == "cursedcity":
            bot = self.cursedcity_bot
            lines.extend(
                [
                    self._format_stat_line(
                        "executed_stage_count",
                        getattr(bot, "executed_stage_count", 0),
                    ),
                    self._format_stat_line(
                        "mode_transitioned_out",
                        getattr(bot, "mode_transitioned_out", False),
                    ),
                ]
            )
            return lines

        if mode_key == "grimforest":
            bot = self.grimforest_bot
            lines.extend(
                [
                    self._format_stat_line("battles_done", getattr(bot, "battles_done", 0)),
                    self._format_stat_line("battles_won", getattr(bot, "battles_won", 0)),
                    self._format_stat_line(
                        "current_run_difficulty",
                        getattr(bot, "current_run_difficulty", None),
                    ),
                ]
            )
            return lines

        if mode_key == "doomtower":
            bot = self.doomtower_bot
            lines.extend(
                [
                    self._format_stat_line("battles_done", getattr(bot, "battles_done", 0)),
                    self._format_stat_line("battles_won", getattr(bot, "battles_won", 0)),
                    self._format_stat_line(
                        "current_difficulty",
                        getattr(bot, "current_difficulty", None),
                    ),
                    self._format_stat_line(
                        "highest_stage_available",
                        getattr(bot, "highest_stage_available", {}),
                    ),
                ]
            )
            return lines

        return lines

    def _log_daily_mode_summary(self, mode_key: str) -> None:
        self._append_daily_log_lines(self._format_mode_daily_summary(mode_key))

    # =========================
    # Parameter management
    # =========================
    def resolve_parameter_key(self, requested_key: str) -> str:
        normalized = file_tools.normalize_param_key(requested_key)
        if self.param_store.has_key(normalized):
            return normalized

        if not normalized.startswith("run_"):
            run_key = f"run_{normalized}"
            if self.param_store.has_key(run_key):
                return run_key

        raise KeyError(requested_key)

    def get_parameter_value(self, requested_key: str):
        key = self.resolve_parameter_key(requested_key)
        return key, self.param_store.get(key)

    def set_parameter_value(self, requested_key: str, raw_value: str):
        key = self.resolve_parameter_key(requested_key)
        current = self.param_store.get(key)
        value = file_tools.coerce_value(raw_value, current)
        update = self.param_store.set(key, value, persist=True, create_if_missing=False)

        if update.changed:
            self.params_flat[key] = value
            self.params = self.param_store.get_grouped_copy()
            self._apply_runtime_parameter(key, value)
            if key.startswith("dungeons_"):
                self._synchronize_dungeon_tournament_override(
                    reason=f"parameter update ({key})"
                )

        return update

    def toggle_mode(self, requested_mode: str, desired_state: str | None = None):
        mode = file_tools.normalize_param_key(requested_mode)
        run_key = mode if mode.startswith("run_") else f"run_{mode}"

        if not self.param_store.has_key(run_key):
            raise KeyError(requested_mode)

        current = bool(self.param_store.get(run_key))
        if desired_state is None:
            new_value = not current
        else:
            new_value = file_tools.coerce_value(desired_state, True)

        update = self.param_store.set(run_key, bool(new_value), persist=True)
        if update.changed:
            self.params_flat[run_key] = bool(new_value)
            self.params = self.param_store.get_grouped_copy()

        return update

    def reload_configuration(self):
        self.param_store.reload()
        self.params_flat = self.param_store.get_flat_copy()
        self.params = self.param_store.get_grouped_copy()
        self.verbose = bool(self.params.get("mainframe", {}).get("verbose", True))
        _configure_logging(self.verbose)
        for key, value in self.params_flat.items():
            self._apply_runtime_parameter(key, value)

        self._synchronize_dungeon_tournament_override(reason="reload")

    def _get_configured_dungeon_parameters(self) -> dict:
        fallback = self.params.get("dungeons", {})

        difficulty = self.param_store.get(
            "dungeons_difficulty",
            fallback.get("difficulty", "normal"),
        )
        dungeon = self.param_store.get(
            "dungeons_dungeon",
            fallback.get("dungeon", "fire_knight"),
        )
        eventdungeon_level = self.param_store.get(
            "dungeons_eventdungeon_level",
            fallback.get("eventdungeon_level", 29),
        )
        level = self.param_store.get(
            "dungeons_level",
            fallback.get("level"),
        )
        if str(dungeon).strip().lower() == "event_dungeon":
            level = eventdungeon_level

        return {
            "difficulty": difficulty,
            "dungeon": dungeon,
            "level": level,
            "eventdungeon_level": eventdungeon_level,
        }

    def _apply_effective_dungeon_parameters(self, effective_params: dict) -> None:
        dungeons_group = self.params.setdefault("dungeons", {})
        dungeons_group["difficulty"] = effective_params["difficulty"]
        dungeons_group["dungeon"] = effective_params["dungeon"]
        dungeons_group["level"] = effective_params.get("level")
        dungeons_group["eventdungeon_level"] = effective_params.get("eventdungeon_level")

        if hasattr(self, "dungeon_bot"):
            self.dungeon_bot.difficulty = effective_params["difficulty"]
            self.dungeon_bot.dungeon = effective_params["dungeon"]
            self.dungeon_bot.level = effective_params.get("level")
            self.dungeon_bot.eventdungeon_level = effective_params.get("eventdungeon_level", 29)
            self.dungeon_bot.fusion_active = self._dungeon_fusion_active

    def _normalize_fusion_difficulty(self, value, reason: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in {"normal", "hard"}:
            return normalized
        if normalized:
            self.log.warning(
                "Invalid dungeons_fusion_difficulty='%s' (%s). Defaulting to 'normal'.",
                value,
                reason,
            )
        return "normal"

    def _synchronize_dungeon_tournament_override(self, reason: str) -> None:
        today = datetime.now(BERLIN_TZ).date()
        configured = self._get_configured_dungeon_parameters()
        disable_fusion_override = bool(
            self.param_store.get(
                "dungeons_disable_fusion_override",
                self.params.get("dungeons", {}).get("disable_fusion_override", False),
            )
        )
        raw_fusion_difficulty = self.param_store.get(
            "dungeons_fusion_difficulty",
            self.params.get("dungeons", {}).get("fusion_difficulty", "normal"),
        )
        fusion_difficulty = self._normalize_fusion_difficulty(raw_fusion_difficulty, reason)

        active_event = None
        if not disable_fusion_override:
            try:
                active_event = raid_calendar_handler.get_active_dungeon_tournament(today=today)
                self._last_known_dungeon_tournament = active_event
            except Exception as exc:
                cached_event = self._last_known_dungeon_tournament
                if cached_event and cached_event.start_date <= today <= cached_event.end_date:
                    active_event = cached_event
                    self.log.warning(
                        (
                            "Fastidious calendar lookup failed (%s): %s. "
                            "Reusing cached active tournament '%s'."
                        ),
                        reason,
                        exc,
                        cached_event.name,
                    )
                else:
                    self.log.warning("Fastidious calendar lookup failed (%s): %s", reason, exc)

        self._dungeon_fusion_active = bool(active_event)
        if active_event:
            effective = {
                "difficulty": fusion_difficulty,
                "dungeon": active_event.dungeon,
                "level": None,
            }
            signature = (
                "override",
                fusion_difficulty,
                active_event.name,
                active_event.dungeon,
                active_event.start_date.isoformat(),
                active_event.end_date.isoformat(),
            )
        else:
            effective = configured
            signature = (
                "configured",
                str(configured["difficulty"]),
                str(configured["dungeon"]),
                str(configured.get("level")),
                str(disable_fusion_override),
            )

        self._apply_effective_dungeon_parameters(effective)

        if signature == self._last_dungeon_override_signature:
            return

        self._last_dungeon_override_signature = signature
        if active_event:
            self.log.info(
                (
                    "Active dungeon tournament on Fastidious calendar (%s, %s to %s). "
                    "Forcing dungeons to difficulty='%s', dungeon='%s'."
                ),
                active_event.name,
                active_event.start_date.isoformat(),
                active_event.end_date.isoformat(),
                fusion_difficulty,
                active_event.dungeon,
            )
        elif disable_fusion_override:
            self.log.info(
                (
                    "Dungeon fusion override disabled (%s). "
                    "Using configured dungeon params: difficulty='%s', dungeon='%s', level='%s'."
                ),
                reason,
                configured["difficulty"],
                configured["dungeon"],
                configured.get("level"),
            )
        else:
            self.log.info(
                (
                    "No active Fastidious dungeon tournament (%s). "
                    "Using configured dungeon params: difficulty='%s', dungeon='%s', level='%s'."
                ),
                reason,
                configured["difficulty"],
                configured["dungeon"],
                configured.get("level"),
            )

    def _get_configured_daily_dungeon_tasks(self) -> list[daily_tasks.DungeonDailyTask]:
        raw_tasks = self.param_store.get(
            "daily_tasks_dungeons",
            self.params.get("daily_tasks", {}).get("dungeons"),
        )
        tasks = daily_tasks.parse_dungeon_daily_tasks(raw_tasks)
        if raw_tasks and not tasks:
            self.log.warning("Ignoring invalid daily_tasks_dungeons config: %r", raw_tasks)
        return tasks

    def _load_daily_task_state_for_today(self) -> dict:
        utc_date = daily_tasks.utc_date_string(datetime.now(UTC_TZ))
        return daily_tasks.load_daily_task_state(self.daily_task_state_path, utc_date)

    def _save_daily_task_state(self, state: dict) -> None:
        daily_tasks.save_daily_task_state(self.daily_task_state_path, state)

    def _build_daily_dungeon_effective_parameters(
        self,
        task: daily_tasks.DungeonDailyTask,
    ) -> dict:
        configured = self._get_configured_dungeon_parameters()
        eventdungeon_level = configured.get("eventdungeon_level", 29)
        if task.dungeon == "event_dungeon" and task.level is not None:
            eventdungeon_level = task.level

        return {
            "difficulty": task.difficulty or "hard",
            "dungeon": task.dungeon,
            "level": task.level,
            "eventdungeon_level": eventdungeon_level,
        }

    def _run_daily_dungeon_task(
        self,
        task: daily_tasks.DungeonDailyTask,
        remaining_energy: int,
    ) -> int:
        self._apply_effective_dungeon_parameters(
            self._build_daily_dungeon_effective_parameters(task)
        )
        self.dungeon_bot.fusion_active = False

        spent = 0
        try:
            self.navigate_to_menu(self.MAIN_MENU_NAMES["Dungeons"])
            spent = self.dungeon_bot.run_dungeons(
                main_loop_running=self.main_loop_running,
                forced_encounter=task.dungeon,
                energy_target=max(1, int(remaining_energy)),
                ignore_iron_twins_priority=True,
            )
        finally:
            try:
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
            except Exception:
                pass
            self._synchronize_dungeon_tournament_override(reason="after daily dungeon task")

        return int(spent or 0)

    def _run_daily_routines(self) -> None:
        tasks = self._get_configured_daily_dungeon_tasks()
        if not tasks:
            return

        state = self._load_daily_task_state_for_today()
        dungeon_state = state.setdefault("dungeons", {})

        for task in tasks:
            if not self.main_loop_running:
                break
            if not self._connectivity_gate_allows_work():
                break

            progress = dungeon_state.setdefault(
                task.signature,
                {
                    "dungeon": task.dungeon,
                    "difficulty": task.difficulty or "highest",
                    "level": task.level,
                    "energy_target": task.energy,
                    "energy_spent": 0,
                    "completed": False,
                },
            )
            energy_spent = int(progress.get("energy_spent", 0) or 0)
            if progress.get("completed") or energy_spent >= task.energy:
                if not progress.get("completed"):
                    progress["completed"] = True
                    self._save_daily_task_state(state)
                continue

            remaining = max(0, task.energy - energy_spent)
            self.current_mode = "daily_tasks"
            self._append_daily_log_lines(
                [
                    "",
                    f"[{self._format_utc_timestamp()}] daily_task=dungeons started",
                    f"account={self.account_name} profile={self.profile_account_name}",
                    (
                        f"dungeon={task.dungeon} difficulty={task.difficulty or 'highest'} "
                        f"level={task.level} remaining_energy={remaining}"
                    ),
                ]
            )

            spent = self._run_daily_dungeon_task(task, remaining)
            progress["energy_spent"] = energy_spent + spent
            progress["completed"] = progress["energy_spent"] >= task.energy
            self._save_daily_task_state(state)

            self._append_daily_log_lines(
                [
                    f"[{self._format_utc_timestamp()}] daily_task=dungeons finished",
                    f"dungeon={task.dungeon} estimated_energy_spent={progress['energy_spent']}/{task.energy}",
                    f"completed={progress['completed']}",
                ]
            )

            if spent <= 0:
                break

    def _apply_runtime_parameter(self, flat_key: str, value):
        if flat_key in {"verbose", "screen_drift"}:
            if flat_key == "verbose":
                self.verbose = bool(value)
                _configure_logging(self.verbose)
            elif flat_key == "screen_drift":
                self.screen_drift = value
            return

        if flat_key.startswith("run_"):
            return

        for group_name in sorted(self.group_to_bot.keys(), key=len, reverse=True):
            prefix = f"{group_name}_"
            if not flat_key.startswith(prefix):
                continue

            field_name = flat_key[len(prefix):]
            bot = self.group_to_bot[group_name]

            candidates = [f"{group_name}_{field_name}", field_name]
            if group_name == "demon_lord":
                candidates.append(f"demonlord_{field_name}")
            if group_name == "faction_wars":
                candidates.append(f"factionwars_{field_name}")

            if group_name == "chimera" and field_name == "threshold":
                bot.threshold = float(value) * 1e6
                return

            for attr_name in candidates:
                if hasattr(bot, attr_name):
                    setattr(bot, attr_name, value)
                    if group_name == "faction_wars" and field_name == "farm_superraid":
                        bot.multiplier = 2 if bool(value) else 1
                    if (
                        group_name == "faction_wars"
                        and field_name == "progress_mode_factions"
                        and hasattr(bot, "_refresh_progress_mode_faction_set")
                    ):
                        bot._refresh_progress_mode_faction_set()
                    return

            self.log.debug("No runtime attribute binding for %s", flat_key)
            return

    def _persist_faction_wars_stage_update(self, faction_name: str, stage: int, difficulty: str):
        key = "faction_wars_farm_stages"
        current_farm_stages = self.param_store.get(key, {})
        if not isinstance(current_farm_stages, dict):
            current_farm_stages = {}

        updated_farm_stages = dict(current_farm_stages)
        updated_farm_stages[faction_name] = [int(stage), str(difficulty)]

        update = self.param_store.set(
            key,
            updated_farm_stages,
            persist=True,
            create_if_missing=False,
        )

        if update.changed:
            self.params_flat[key] = updated_farm_stages
            self.params = self.param_store.get_grouped_copy()
            self._apply_runtime_parameter(key, updated_farm_stages)

        self._append_daily_log_lines(
            [
                f"Faction Wars stage update: {faction_name} -> stage {int(stage)} ({difficulty})",
                f"profile={self.profile_account_name}",
            ]
        )
        self.log.info(
            "Faction Wars stage updated: %s -> [%s, '%s']",
            faction_name,
            int(stage),
            str(difficulty),
        )

    # =========================
    # Reporting / help
    # =========================
    def _format_value(self, value, max_len: int = 120) -> str:
        return runtime_reporting.format_value(value, max_len=max_len)

    def get_status_snapshot(self) -> dict:
        now_cest = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S")
        run_flags = self.params.get("run", {})
        enabled_modes = sorted([name for name, enabled in run_flags.items() if enabled])
        uptime = int(time.time() - self.start_time)

        return {
            "timestamp_cest": now_cest,
            "uptime_seconds": uptime,
            "manual_mode": self.manual_mode,
            "main_loop_running": self.main_loop_running,
            "main_loop_stopped": self.main_loop_stopped,
            "current_mode": self.current_mode,
            "enabled_modes": enabled_modes,
            "last_error": self.last_loop_error,
        }

    def build_help_lines(self) -> list[str]:
        snapshot = self.get_status_snapshot()
        return runtime_reporting.build_help_lines(
            snapshot,
            self.build_editable_params_overview_lines(),
        )

    def build_editable_params_overview_lines(self, max_line_length: int = 190) -> list[str]:
        return runtime_reporting.build_editable_params_overview_lines(
            self.param_store.keys(),
            max_line_length=max_line_length,
        )

    def build_status_lines(self) -> list[str]:
        return runtime_reporting.build_status_lines(self.get_status_snapshot())

    def build_modes_lines(self) -> list[str]:
        return runtime_reporting.build_modes_lines(self.params.get("run", {}))

    def build_params_lines(self, search_term: str | None = None, max_items: int = 80) -> list[str]:
        return runtime_reporting.build_params_lines(
            self.param_store.keys(),
            self.param_store.get,
            search_term=search_term,
            max_items=max_items,
            value_formatter=self._format_value,
        )

    # =========================
    # Discord command handling
    # =========================
    def _send_discord_messages(self, messages: list[str]):
        for message in messages:
            if message:
                self.discord_override.send_message(message)

    def _announce_restart_success_if_requested(self):
        if os.environ.pop(RESTART_NOTIFY_ENV, None) != "1":
            return

        now_cest = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S")
        message = f"[Bot Status] Restart completed successfully at {now_cest} CEST."

        if hasattr(self.discord_override, "send_message_blocking"):
            self.discord_override.send_message_blocking(message, timeout=15.0)
            return

        self.discord_override.send_message(message)

    def _consume_single_command(self):
        if not hasattr(self.discord_override, "pop_last_command"):
            command = self.discord_override.get_last_command()
            self.discord_override.clear_last_command()
            return command
        return self.discord_override.pop_last_command()

    def _process_pending_command(self):
        command = self._consume_single_command()
        if not command:
            return

        result = self.command_router.route(command)
        self._send_discord_messages(result.messages)

        if result.restart_requested:
            raise RestartRequested("discord restart command")

        if result.enter_manual_mode:
            self._enter_manual_mode()
            return

        if result.exit_manual_mode:
            self.manual_mode = False

    def _enter_manual_mode(self):
        if self.manual_mode:
            return

        self.manual_mode = True
        self.main_loop_running = False
        self.current_mode = "manual_mode"
        self.discord_override.send_message("[Bot Status] Manual play/config mode enabled.")

        while self.running and self.manual_mode:
            command = self._consume_single_command()
            if command:
                result = self.command_router.route(command)
                self._send_discord_messages(result.messages)

                if result.restart_requested:
                    raise RestartRequested("discord restart command (manual mode)")
                if result.exit_manual_mode:
                    self.manual_mode = False
                    break

            time.sleep(1.0)

        if not self.manual_mode:
            now_cest = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S")
            self.discord_override.send_message(f"[Bot Status] Running | {now_cest} CEST")

            self.main_loop_running = True
            self.navigate_bastion_menu(
                self.search_areas["bastion_to_main_menu"],
                self.search_areas["menu_name"],
                "Modos de juego",
            )

    def _send_mode_heartbeat(self, mode_name: str | None = None):
        now_cest = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S")
        if mode_name:
            self.discord_override.send_message(f"[Bot Status] Running {mode_name} | {now_cest} CEST")
        else:
            self.discord_override.send_message(f"[Bot Status] Running | {now_cest} CEST")

    def _set_all_mode_loops(self, running: bool):
        for bot in self.bots:
            bot.main_loop_running = running

    # =========================
    # Error handling
    # =========================
    def _start_error_checker(self):
        def run_loop():
            while self.running:
                try:
                    self.error_handler.run_once()
                except Exception:
                    self.log.exception("Exception in background error checker.")
                time.sleep(1)

        threading.Thread(target=run_loop, daemon=True, name="error-checker").start()

    def _start_connectivity_supervisor(self):
        try:
            self.connectivity_supervisor = runtime_connectivity.ConnectivityRecoverySupervisor(
                online_poll_interval_seconds=CONNECTIVITY_ONLINE_POLL_SECONDS,
                reconnect_check_interval_seconds=CONNECTIVITY_RETRY_INTERVAL_SECONDS,
                outage_confirmation_seconds=CONNECTIVITY_OUTAGE_CONFIRM_SECONDS,
                on_connection_lost=self._on_connectivity_lost,
                on_retry_attempt=self._on_connectivity_retry,
                on_connection_restored=self._on_connectivity_restored,
            )
            self.connectivity_supervisor.start()
        except Exception:
            self.connectivity_supervisor = None
            self.log.exception("Failed to start connectivity supervisor.")

    def _on_connectivity_lost(self, _lost_at_utc):
        self.main_loop_running = False
        self._set_all_mode_loops(False)
        self.log.warning(
            "Internet connectivity lost. Pausing automation and retrying every %d minutes.",
            int(CONNECTIVITY_RETRY_INTERVAL_SECONDS // 60),
        )

    def _on_connectivity_retry(self, attempt: int, retry_delay_seconds: float):
        self.log.warning(
            (
                "Internet still unavailable (retry attempt #%d). "
                "Next connectivity probe in %.0f seconds."
            ),
            attempt,
            retry_delay_seconds,
        )

    def _on_connectivity_restored(self, downtime_seconds: float, retry_attempts: int):
        self.log.info(
            (
                "Internet connectivity restored after %.0f seconds "
                "(retry attempts during outage: %d)."
            ),
            downtime_seconds,
            retry_attempts,
        )

    def _ensure_connectivity_supervisor_running(self):
        supervisor = self.connectivity_supervisor
        if supervisor is None:
            now = time.time()
            if now - self._last_connectivity_supervisor_restart_ts < 30.0:
                return
            self._last_connectivity_supervisor_restart_ts = now
            self.log.warning("Connectivity supervisor is missing. Reinitializing it.")
            self._start_connectivity_supervisor()
            return
        if supervisor.is_running():
            return

        now = time.time()
        if now - self._last_connectivity_supervisor_restart_ts < 30.0:
            return
        self._last_connectivity_supervisor_restart_ts = now

        self.log.warning("Connectivity supervisor stopped unexpectedly. Restarting it.")
        try:
            supervisor.start()
        except Exception:
            self.log.exception("Failed to restart connectivity supervisor.")

    def _ensure_discord_override_running(self):
        if isinstance(self.discord_override, NullDiscordRemoteOverride):
            return

        is_running = getattr(self.discord_override, "is_running", None)
        if callable(is_running) and is_running():
            return

        now = time.time()
        if now - self._last_discord_restart_attempt_ts < 30.0:
            return
        self._last_discord_restart_attempt_ts = now

        self.log.warning("Discord remote override is not running. Attempting restart.")
        try:
            self.discord_override.start()
        except Exception:
            self.log.exception("Failed to restart Discord remote override.")

    @staticmethod
    def _is_likely_network_exception(exc: Exception) -> bool:
        network_error_types = (
            TimeoutError,
            ConnectionError,
            socket.timeout,
            socket.gaierror,
        )
        if isinstance(exc, network_error_types):
            return True

        message = str(exc).lower()
        markers = (
            "connection",
            "timed out",
            "name resolution",
            "temporarily unavailable",
            "network",
            "dns",
            "host unreachable",
            "remote end closed",
        )
        return any(marker in message for marker in markers)

    def _connectivity_gate_allows_work(self) -> bool:
        supervisor = self.connectivity_supervisor
        self._ensure_connectivity_supervisor_running()
        self._ensure_discord_override_running()
        if supervisor is None:
            return True

        if supervisor.consume_recovery_signal():
            self._connectivity_pause_notified = False
            self.log.info(
                "Connectivity recovery confirmed. Triggering full application restart."
            )
            raise RestartRequested("internet connectivity recovered")

        if supervisor.is_paused():
            if not self._connectivity_pause_notified:
                self._connectivity_pause_notified = True
                self.current_mode = "connectivity_wait"
                self._set_all_mode_loops(False)
                self.discord_override.send_message(
                    (
                        "[Bot Status] Internet connection lost. Automation paused. "
                        "Retrying every 10 minutes."
                    )
                )
            time.sleep(1.0)
            return False

        return True

    def wait_for_restart_command(self, *, allow_connectivity_recovery: bool = False) -> str:
        self.discord_override.clear_last_command()
        if allow_connectivity_recovery:
            self.discord_override.send_message(
                (
                    "[Bot Error] Main process stopped due to an error. Waiting for command: restart. "
                    "Automatic restart will also trigger after internet connectivity is restored."
                )
            )
        else:
            self.discord_override.send_message(
                "[Bot Error] Main process stopped due to an error. Waiting for command: restart"
            )
        while self.running:
            self._ensure_connectivity_supervisor_running()
            self._ensure_discord_override_running()

            if allow_connectivity_recovery and self.connectivity_supervisor is not None:
                if self.connectivity_supervisor.consume_recovery_signal():
                    self.log.info(
                        "Connectivity recovered while waiting for restart command. "
                        "Proceeding with automatic restart."
                    )
                    return "connectivity_recovered"

            command = self._consume_single_command()
            if command:
                try:
                    result = self.command_router.route(command)
                    self._send_discord_messages(result.messages)
                    if result.restart_requested:
                        return "manual_restart_command"
                except Exception:
                    self.log.exception("Failed while processing restart-wait command.")

            time.sleep(1.5)

        return "runtime_stopped"

    def _handoff_full_application_restart(self, trigger: str):
        with self._restart_lock:
            if self._restart_in_progress:
                self.log.warning(
                    "Restart handoff already in progress. Ignoring duplicate trigger: %s",
                    trigger,
                )
                raise SystemExit(0)
            self._restart_in_progress = True

        self.log.warning("Starting full application restart handoff (%s).", trigger)

        try:
            _spawn_run_bot_process(RESTART_HELPER_ARG, str(os.getpid()))
        except Exception:
            with self._restart_lock:
                self._restart_in_progress = False
            self.log.exception("Failed to spawn restart helper process.")
            raise

        self.running = False
        self.main_loop_running = False
        self.main_loop_stopped = True
        self._set_all_mode_loops(False)

        try:
            if self.connectivity_supervisor is not None:
                self.connectivity_supervisor.stop()
        except Exception:
            self.log.exception("Failed to stop connectivity supervisor cleanly.")

        try:
            self.discord_override.stop()
        except Exception:
            self.log.exception("Failed to stop Discord remote override cleanly.")

        raise SystemExit(0)

    def _handoff_full_application_restart_with_fallback(self, trigger: str):
        current_trigger = trigger
        while self.running:
            try:
                self._handoff_full_application_restart(current_trigger)
            except SystemExit:
                raise
            except Exception as handoff_error:
                self.last_loop_error = str(handoff_error)
                self.log.exception(
                    "Restart handoff failed after trigger '%s'.", current_trigger
                )
                wait_result = self.wait_for_restart_command(
                    allow_connectivity_recovery=True
                )
                if wait_result == "connectivity_recovered":
                    current_trigger = "connectivity recovered after failed handoff"
                else:
                    current_trigger = "retry after failed handoff"

        raise RuntimeError("Runtime stopped before restart handoff could complete.")

    def restart_raid_process(self):
        self.log.warning("Restarting Raid process.")

        self.raid_path = _build_raid_launch_command()

        try:
            _launch_raid_and_wait_until_window(
                launch_command=self.raid_path,
                timeout_seconds=180.0,
                log=self.log,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Failed to launch Raid with command: {self.raid_path}. "
                f"Set {PLARIUM_PLAY_EXE_ENV} to a valid PlariumPlay.exe path or "
                f"{RAID_SHORTCUT_PATH_ENV} to a valid Raid shortcut."
            ) from exc
        time.sleep(60)

        windows = gw.getWindowsWithTitle(RAID_WINDOW_TITLE)
        if windows:
            win = windows[0]
            self.window = window_tools.WindowObject(
                (win.left, win.top, win.width, win.height),
                title_substring=RAID_WINDOW_TITLE,
            )
            self.coords = (self.window.left, self.window.top, self.window.width, self.window.height)

            for bot in self.bots:
                bot.window = self.window
                bot.coords = self.coords
            self.error_handler.window = self.window
            self.error_handler.coords = self.coords

        self.navigate_bastion_once_after_restart = True

    def capture_error_screenshot(self):
        screenshot_dir = os.path.join("data", "tmp")
        os.makedirs(screenshot_dir, exist_ok=True)
        screenshot_path = os.path.join(screenshot_dir, "raid_error_latest.png")

        if self.window:
            region = (self.window.left, self.window.top, self.window.width, self.window.height)
            image = pyautogui.screenshot(region=region)
        else:
            image = pyautogui.screenshot()

        image.save(screenshot_path)
        return screenshot_path

    def _wait_before_menu_name_read(self):
        if self._is_slow_laptop:
            time.sleep(2)

    # =========================
    # Navigation helpers
    # =========================
    @staticmethod
    def resembles(text, target, threshold=0.8):
        ratio = difflib.SequenceMatcher(None, (text or "").lower(), target.lower()).ratio()
        return ratio >= threshold

    def navigate_bastion_menu(self, button_area, confirm_area, confirm_string, max_attempts=15):
        for attempt in range(max_attempts):
            if not self.main_loop_running:
                break
            try:
                if attempt % 2 == 0:
                    window_tools.sendkey("esc", window=self.window)

                try:
                    advert2 = image_tools.get_text_in_relative_area(
                        self.reader,
                        self.window,
                        self.search_areas["advert2"],
                        power_detection=False,
                    )
                    if advert2 and self.resembles(advert2[0].text, "Cerrar"):
                        window_tools.click_center(self.window, self.search_areas["advert2"])
                except Exception:
                    pass

                window_tools.click_center(self.window, button_area)
                self._wait_before_menu_name_read()
                menu_name = image_tools.get_text_in_relative_area(
                    self.reader,
                    self.window,
                    confirm_area,
                    power_detection=False,
                )
                if menu_name and self.resembles(menu_name[0].text, confirm_string):
                    return True
            except Exception:
                pass

        return False

    def navigate_to_menu(self, menu_name, max_attempts=5, detect_doomtower_rotation=False):
        for _ in range(max_attempts):
            if not self.main_loop_running:
                break
            try:
                texts = image_tools.get_text_in_relative_area(
                    self.reader,
                    self.window,
                    self.search_areas["menu_name"],
                    power_detection=False,
                )

                if texts and "Modos" in texts[0].text:
                    break
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
            except Exception:
                window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

        def _scan_visible_mode_label():
            labels = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                self.search_areas["main_menu_labels"],
                power_detection=False,
            )
            for label in labels:
                if self.resembles(label.text, menu_name):
                    return label
            return None

        def _click_mode_label(label):
            if not label:
                return False
            if detect_doomtower_rotation:
                self.detect_doomtower_rotation()
            window_tools.click_at(label.mean_pos_x, label.mean_pos_y, window=self.window)
            return True

        # First scan without swiping to avoid unnecessary drag actions.
        self._wait_before_menu_name_read()
        visible_label = _scan_visible_mode_label()
        if _click_mode_label(visible_label):
            self.log.debug("navigate_to_menu('%s'): found target without swipe", menu_name)
            return

        for direction in ("left", "right"):
            if not self.main_loop_running:
                break

            move = window_tools.move_left if direction == "left" else window_tools.move_right
            for swipe_index in range(2):
                if not self.main_loop_running:
                    break
                self.log.debug(
                    "navigate_to_menu('%s'): starting swipe direction=%s step=%s/2",
                    menu_name,
                    direction,
                    swipe_index + 1,
                )
                try:
                    pyautogui.mouseUp()
                except Exception:
                    pass
                move(self.window, strength=1.0, relative_x=0.5, relative_y=0.72)
                try:
                    pyautogui.mouseUp()
                except Exception:
                    pass
                self.log.debug(
                    "navigate_to_menu('%s'): finished swipe direction=%s step=%s/2",
                    menu_name,
                    direction,
                    swipe_index + 1,
                )
                time.sleep(2)

                self._wait_before_menu_name_read()
                visible_label = _scan_visible_mode_label()
                if _click_mode_label(visible_label):
                    return

        self.log.warning("Menu '%s' not found", menu_name)

    def navigate_to_bastion(self):
        self.navigate_to_menu(self.MAIN_MENU_NAMES["Dungeons"])
        window_tools.click_center(self.window, self.search_areas["go_to_bastion"])

    # =========================
    # Doom Tower helpers
    # =========================
    def detect_doomtower_rotation(self):
        rotation_text = image_tools.get_text_in_relative_area(
            self.reader,
            self.window,
            self.search_areas["detect_doomtower_rotation"],
        )
        if not rotation_text:
            return

        # OCR can return multiple text boxes for the same area.
        texts = [entry.text for entry in rotation_text if getattr(entry, "text", None)]
        combined_text = " ".join(texts).lower()

        # Priority when multiple names are detected: Hada -> Dragon -> Arana.
        if "hada" in combined_text:
            self.doomtower_bot.current_rotation = "3"
        elif "dragon" in combined_text:
            self.doomtower_bot.current_rotation = "2"
        elif "arana" in combined_text:
            self.doomtower_bot.current_rotation = "1"

    # =========================
    # Rewards
    # =========================
    def collect_quest_rewards(self, delay=2):
        time.sleep(delay)
        self.navigate_to_bastion()

        if not self.navigate_bastion_menu(
            self.search_areas["quest_menu"],
            self.search_areas["quest_menu_name"],
            "Misiones",
        ):
            self.log.warning("Missions menu not found.")
            window_tools.click_center(self.window, self.search_areas["bastion_to_main_menu"])
            return

        for menu in (
            "daily_quest_menu",
            "weekly_quest_menu",
            "monthly_quest_menu",
            "advanced_quest_menu",
        ):
            window_tools.click_center(self.window, self.search_areas[menu])
            window_tools.click_center(self.window, self.search_areas["claim_quest_rewards"])

        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

        window_tools.click_center(self.window, self.search_areas["time_gated_reward_menu"], delay=5)
        window_tools.click_center(self.window, self.search_areas["time_gated_reward_menu_name"], delay=5)
        window_tools.click_center(self.window, self.search_areas["time_gated_reward_menu"], delay=5)

        for _ in range(2):
            objs = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                self.search_areas["pov"],
                power_detection=False,
            )
            window_tools.move_right(self.window, strength=2.)
            window_tools.move_down(self.window, strength=2.)
            for obj in objs:
                if self.resembles(obj.text, "Ring de Guardianes"):
                    window_tools.click_at(obj.mean_pos_x, obj.mean_pos_y, delay=2)
                    for idx in range(1, 6):
                        window_tools.click_center(
                            self.window,
                            self.search_areas[f"guardian_faction_character_{idx}"],
                        )
                    window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
                    break

            window_tools.move_left(self.window, strength=2.)
            objs = image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                self.search_areas["pov"],
                power_detection=False,
            )
            for obj in objs:
                if self.resembles(obj.text, "Mercado") or self.resembles(obj.text, "ercado") or self.resembles(obj.text, "Mercad"):
                    for i in range(2):
                        window_tools.click_at(obj.mean_pos_x, obj.mean_pos_y, delay=2)
                        objs_market = image_tools.get_text_in_relative_area(
                            self.reader,
                            self.window,
                            self.search_areas["pov"],
                            power_detection=False,
                        )
                        for obj_market in objs_market:
                            if self.resembles(obj_market.text, "Fragmento Misterioso"):
                                window_tools.click_at(obj_market.mean_pos_x, obj_market.mean_pos_y, delay=2)
                                buy_options = image_tools.get_text_in_relative_area(
                                    self.reader,
                                    self.window,
                                    self.search_areas["buy_mystery_shard"],
                                    power_detection=False,
                                )
                                if any(self.resembles(option.text, "Obtener") for option in buy_options):
                                    window_tools.click_center(
                                        self.window,
                                        self.search_areas["buy_mystery_shard"],
                                    )
                                break

                        window_tools.move_right(self.window, strength=1.)

                    window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

        self.navigate_bastion_menu(
            self.search_areas["bastion_to_main_menu"],
            self.search_areas["menu_name"],
            "Modos de juego",
        )

    # =========================
    # Mode execution
    # =========================
    def _is_mode_enabled(self, spec: ModeSpec) -> bool:
        run_flags = self.params.get("run", {})
        enabled = bool(run_flags.get(spec.run_flag_key, False))
        if not enabled:
            return False
        if spec.condition:
            return bool(spec.condition())
        return True

    def _find_mode_spec(self, mode_key: str) -> ModeSpec | None:
        for spec in self.mode_specs:
            if spec.key == mode_key:
                return spec
        return None

    def _get_forced_pushrank_mode_spec(self, now_utc: datetime | None = None) -> ModeSpec | None:
        current_utc = now_utc or datetime.now(UTC_TZ)
        forced_mode_key = pushrank.get_forced_pushrank_mode(
            current_utc,
            classic_enabled=bool(self.params.get("classic_arena", {}).get("pushrank", False)),
            tagteam_enabled=bool(self.params.get("tagteam_arena", {}).get("pushrank", False)),
        )
        if forced_mode_key is None:
            return None
        return self._find_mode_spec(forced_mode_key)

    def _refresh_arena_enemy_list_if_due(self, bot, timers: dict, key: str, interval: float):
        now = time.time()
        if timers.get(key) is None:
            timers[key] = now
            if now - self.handler_init_time > interval:
                bot.refresh_enemy_list()
            return

        if now - timers[key] > interval:
            timers[key] = now
            bot.refresh_enemy_list()

    def _run_mode_classic_arena(self, timers, refresh_interval):
        self.navigate_to_menu(self.MAIN_MENU_NAMES["Arena"])
        window_tools.click_center(self.window, self.search_areas["classic_arena"])
        self._refresh_arena_enemy_list_if_due(
            self.classic_arena_bot,
            timers,
            "classic",
            refresh_interval,
        )
        self.classic_arena_bot.run_classic_arena_until_empty(main_loop_running=self.main_loop_running)
        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    def _run_mode_tagteam_arena(self, timers, refresh_interval):
        self.navigate_to_menu(self.MAIN_MENU_NAMES["Arena"])
        window_tools.click_center(self.window, self.search_areas["tagteam_arena"])
        self._refresh_arena_enemy_list_if_due(
            self.tagteam_arena_bot,
            timers,
            "tagteam",
            refresh_interval,
        )
        self.tagteam_arena_bot.run_tagteam_arena_single_cycle(main_loop_running=self.main_loop_running)
        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    def _run_mode_live_arena(self, timers, refresh_interval):
        self.navigate_to_menu(self.MAIN_MENU_NAMES["Arena"])
        window_tools.click_center(self.window, self.search_areas["live_arena"])
        self.live_arena_bot.run_live_arena_loop(main_loop_running=self.main_loop_running)
        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    def _run_mode_dungeons(self, timers, refresh_interval):
        self._synchronize_dungeon_tournament_override(reason="before dungeon run")
        self.navigate_to_menu(self.MAIN_MENU_NAMES["Dungeons"])
        self.dungeon_bot.run_dungeons(main_loop_running=self.main_loop_running)
        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    def _has_enough_factionwars_keys(self, minimum_required: int = 3) -> bool:
        toggle_area = self.search_areas["factionwars_key_popup_toggle"]
        counter_area = self.search_areas["factionwars_key_counter"]
        counter_data = {"current": None, "maximum": None, "raw_texts": []}

        try:
            window_tools.click_center(self.window, toggle_area, delay=0.2)
            time.sleep(3)
            counter_data = image_tools.read_fraction_counter_in_relative_area(
                self.reader,
                self.window,
                counter_area,
            )
        except Exception as exc:
            self.log.warning("Faction Wars key pre-check failed: %s", exc)
        finally:
            try:
                window_tools.click_center(self.window, toggle_area, delay=0.2)
            except Exception:
                pass

        current_keys = counter_data.get("current")
        if current_keys is None:
            self.log.warning(
                "Faction Wars key pre-check OCR failed (raw=%s). Continuing run.",
                counter_data.get("raw_texts"),
            )
            return True

        maximum_keys = counter_data.get("maximum")
        self.log.info("Faction Wars keys detected: %s/%s", current_keys, maximum_keys)
        return current_keys >= minimum_required

    def _run_mode_factionwars(self, timers, refresh_interval):
        if not self._has_enough_factionwars_keys(minimum_required=3):
            self.log.info("Skipping Faction Wars (keys <= 2).")
            self._append_daily_log_lines(
                [
                    "Faction Wars skipped: insufficient keys",
                    f"account={self.account_name} profile={self.profile_account_name}",
                ]
            )
            return
        self.navigate_to_menu(self.MAIN_MENU_NAMES["FactionWars"])
        self.factionwars_bot.run_factionwars(main_loop_running=self.main_loop_running)
        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    def _run_mode_demonlord(self, timers, refresh_interval):
        self.navigate_to_menu(self.MAIN_MENU_NAMES["ClanBoss1"])
        window_tools.click_center(self.window, self.search_areas["clanboss_DemonLord"])
        self.demonlord_bot.run_demonlord(main_loop_running=self.main_loop_running)
        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    def _run_mode_hydra(self, timers, refresh_interval):
        if self.hydra_bot.check_if_wednesday_berlin():
            self.log.info("Skipping Hydra because it is Wednesday in Europe/Berlin.")
            self._append_daily_log_lines(
                [
                    "Hydra skipped: Wednesday in Europe/Berlin",
                    f"account={self.account_name} profile={self.profile_account_name}",
                ]
            )
            return
        self.navigate_to_menu(self.MAIN_MENU_NAMES["ClanBoss1"])
        window_tools.click_center(self.window, self.hydra_bot.search_areas["clanboss_Hydra"])
        self.hydra_bot.run_hydra(main_loop_running=self.main_loop_running)
        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    def _run_mode_chimera(self, timers, refresh_interval):
        self.navigate_to_menu(self.MAIN_MENU_NAMES["ClanBoss1"])
        window_tools.click_center(self.window, self.search_areas["clanboss_Chimera"])
        self.chimera_bot.run_chimera(main_loop_running=self.main_loop_running)
        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    def _run_mode_cursedcity(self, timers, refresh_interval):
        self.navigate_to_menu(self.MAIN_MENU_NAMES["CursedCity"])
        self.cursedcity_bot.run_cursedcity(main_loop_running=self.main_loop_running)
        if not getattr(self.cursedcity_bot, "mode_transitioned_out", False):
            window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    def _run_mode_grimforest(self, timers, refresh_interval):
        self.navigate_to_menu(self.MAIN_MENU_NAMES["GrimForest"])
        self.grimforest_bot.run_grimforest(main_loop_running=self.main_loop_running)
        if not getattr(self.grimforest_bot, "mode_transitioned_out", False):
            window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    def _run_mode_doomtower(self, timers, refresh_interval):
        self.navigate_to_menu(self.MAIN_MENU_NAMES["DoomTower"], detect_doomtower_rotation=True)
        self.doomtower_bot.run_doomtower(main_loop_running=self.main_loop_running)
        window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

    # =========================
    # Main loop
    # =========================
    def run_main_loop(self):
        self.log.info("Starting Main Loop.")
        self.main_loop_stopped = False
        self.main_loop_running = True
        self.current_mode = "cycle_start"

        if self.navigate_bastion_once_after_restart:
            self.navigate_bastion_menu(
                self.search_areas["bastion_to_main_menu"],
                self.search_areas["menu_name"],
                "Modos de juego",
            )
            self.navigate_bastion_once_after_restart = False

        timers = {"classic": None, "tagteam": None, "live": None}
        refresh_interval = 15.1 * 60

        while self.main_loop_running:
            if not self._connectivity_gate_allows_work():
                continue

            self._process_pending_command()
            if self.manual_mode:
                break

            forced_pushrank_spec = self._get_forced_pushrank_mode_spec()
            if forced_pushrank_spec is not None:
                self.current_mode = forced_pushrank_spec.key
                forced_pushrank_spec.executor(timers, refresh_interval)
                self._log_daily_mode_summary(forced_pushrank_spec.key)
                self._send_mode_heartbeat(f"{forced_pushrank_spec.display_name} Pushrank")
                if not self._connectivity_gate_allows_work():
                    continue
                self._process_pending_command()
                if self.manual_mode:
                    break
                continue

            self._run_daily_routines()
            if not self.main_loop_running or self.manual_mode:
                break

            for spec in self.mode_specs:
                if not self.main_loop_running:
                    break
                if not self._connectivity_gate_allows_work():
                    break
                if not self._is_mode_enabled(spec):
                    continue

                self.current_mode = spec.key
                spec.executor(timers, refresh_interval)
                self._log_daily_mode_summary(spec.key)

                self._send_mode_heartbeat(spec.display_name)
                if not self._connectivity_gate_allows_work():
                    break
                self._process_pending_command()
                if self.manual_mode:
                    break

            if not self.main_loop_running or self.manual_mode:
                break
            if not self._connectivity_gate_allows_work():
                continue

            self.current_mode = "quest_rewards"
            self.collect_quest_rewards()
            self._append_daily_log_lines(
                [
                    "Quest rewards collected",
                    f"account={self.account_name} profile={self.profile_account_name}",
                ]
            )
            self._send_mode_heartbeat()
            self._process_pending_command()

        self.current_mode = "stopped"
        self._append_daily_log_lines(
            [
                "Main loop stopped",
                f"account={self.account_name} profile={self.profile_account_name}",
            ]
        )
        self.main_loop_running = False
        self.main_loop_stopped = True

    def start_main_loop(self):
        self._announce_restart_success_if_requested()

        while True:
            try:
                self.run_main_loop()
            except RestartRequested as restart_request:
                self.main_loop_running = False
                self.main_loop_stopped = True
                trigger = getattr(restart_request, "trigger", "runtime restart request")
                self._handoff_full_application_restart_with_fallback(trigger)
            except Exception as exc:
                self.last_loop_error = str(exc)
                error_time = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S")
                self.log.exception("Fatal error in main loop.")
                self.discord_override.send_message(
                    f"[Bot Error] Main process crashed at {error_time} CEST.\nError: {exc}"
                )

                try:
                    screenshot_path = self.capture_error_screenshot()
                    self.discord_override.send_image(
                        screenshot_path,
                        caption=f"[Bot Error] Raid window screenshot at {error_time} CEST",
                    )
                except Exception as screenshot_error:
                    self.log.error("Failed to capture/send screenshot: %s", screenshot_error)

                self.main_loop_running = False
                self.main_loop_stopped = True
                allow_auto_connectivity_recovery = (
                    self._is_likely_network_exception(exc)
                    or bool(
                        self.connectivity_supervisor and self.connectivity_supervisor.is_paused()
                    )
                )
                wait_result = self.wait_for_restart_command(
                    allow_connectivity_recovery=allow_auto_connectivity_recovery
                )

                if wait_result == "connectivity_recovered":
                    self._handoff_full_application_restart_with_fallback(
                        "connectivity recovered after main loop exception"
                    )
                else:
                    self._handoff_full_application_restart_with_fallback(
                        "post-crash restart command"
                    )

            # If run loop exited without restart, wait briefly and resume.
            time.sleep(2)
            self.handler_init_time = time.time()


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == RESTART_HELPER_ARG:
        return run_restart_helper(int(sys.argv[2]))

    existing_pid = _read_running_instance_pid()
    if existing_pid:
        replacement_reason = None
        if _is_restart_replacement_launch():
            replacement_reason = "Restart launch"
        elif _is_entrypoint_relaunch():
            replacement_reason = "Raid_Bot.py entrypoint relaunch"

        if replacement_reason:
            print(f"{replacement_reason} replacing previous RaidBot PID {existing_pid}.")
            _kill_process_tree_by_pid(existing_pid)
            _wait_for_process_exit(existing_pid, timeout_seconds=10.0)
        else:
            print(f"RaidBot is already running (PID {existing_pid}). Exiting duplicate launch.")
            return 1

    _write_current_pid_file()
    atexit.register(_remove_current_pid_file)

    print("ALWAYS RUN THE PROGRAM IN 1280 x 1024")
    close_discord_desktop_app()

    if not gw.getWindowsWithTitle(RAID_WINDOW_TITLE):
        subprocess.Popen(_build_raid_launch_command())

    while not gw.getWindowsWithTitle(RAID_WINDOW_TITLE):
        time.sleep(2)

    windows = gw.getWindowsWithTitle(RAID_WINDOW_TITLE)
    win = windows[0]

    win.moveTo(10, 10)
    time.sleep(10)

    bot = RSL_Bot_Mainframe()
    bot.start_main_loop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
