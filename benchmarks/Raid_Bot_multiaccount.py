from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

# Ensure repo-root imports work when executed as:
# `python benchmarks/Raid_Bot_multiaccount.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.lib.utils import file_tools

if TYPE_CHECKING:
    from Raid_Bot import RSL_Bot_Mainframe

SECONDARY_ACCOUNTS: tuple[str, ...] = (
    "artus2",
    "artus3",
    "artus4",
    "artus5",
    "artus6",
    "artus7",
)
RAID_WINDOW_TITLE = "Raid: Shadow Legends"
PLARIUM_WINDOW_TITLE = "Plarium Play"
RAID_ACCOUNT_NAME_ENV = "RAID_ACCOUNT_NAME"

LOG = logging.getLogger("RaidBotMultiAccountBenchmark")


def _configure_logging(verbose: bool = True) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _terminate_process_image(image_name: str) -> None:
    subprocess.run(
        ["taskkill", "/F", "/IM", image_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _is_process_running_image(image_name: str) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    output = (result.stdout or "").strip()
    return bool(output and not output.startswith("INFO:"))


def _wait_for_window_gone(title_substring: str, timeout_seconds: float = 20.0) -> bool:
    image_name = "Raid.exe" if title_substring == RAID_WINDOW_TITLE else "PlariumPlay.exe"
    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        if not _is_process_running_image(image_name):
            return True
        time.sleep(0.5)
    return not _is_process_running_image(image_name)


def _cleanup_between_accounts(wait_seconds: float = 2.0) -> None:
    LOG.info("cleanup/reset between accounts: closing Raid + Plarium")
    _terminate_process_image("Raid.exe")
    _terminate_process_image("PlariumPlay.exe")
    _wait_for_window_gone(RAID_WINDOW_TITLE, timeout_seconds=20.0)
    _wait_for_window_gone(PLARIUM_WINDOW_TITLE, timeout_seconds=20.0)
    time.sleep(float(wait_seconds))


def _resolve_strict_profile_for_account(account_name: str) -> file_tools.ProfileParamsResolution:
    return file_tools.resolve_profile_params_file(
        account_name=account_name,
        generate_secondary_profiles=False,
        allow_main_profile_fallback_for_missing_account=False,
    )


def _verify_profile_resolution(account_name: str) -> file_tools.ProfileParamsResolution:
    resolution = _resolve_strict_profile_for_account(account_name)
    if resolution.selected_profile_account_name != account_name:
        raise RuntimeError(
            "Profile/account mismatch before launch: "
            f"target='{account_name}' resolved_profile='{resolution.selected_profile_account_name}' "
            f"path='{resolution.selected_param_file}'."
        )

    expected_prefix = f"{account_name}_{file_tools.PARAMS_FILE_STEM}"
    if not resolution.selected_param_file.name.startswith(expected_prefix):
        raise RuntimeError(
            "Profile filename mismatch before launch: "
            f"target='{account_name}' path='{resolution.selected_param_file}'."
        )

    return resolution


def _install_market_guardian_safeguard(bot: "RSL_Bot_Mainframe", account_name: str) -> None:
    def _blocked_collect_quest_rewards(delay=2):
        LOG.info(
            "one-round safeguard active for '%s': skipped Guardian Ring and skipped Mercado/Market.",
            account_name,
        )
        return

    bot.collect_quest_rewards = _blocked_collect_quest_rewards


def _run_exactly_one_round(bot: "RSL_Bot_Mainframe", account_name: str) -> None:
    LOG.info("one-round execution start for '%s'", account_name)
    bot.main_loop_stopped = False
    bot.main_loop_running = True
    bot.current_mode = "cycle_start"

    _install_market_guardian_safeguard(bot, account_name)

    if bot.navigate_bastion_once_after_restart:
        bot.navigate_bastion_menu(
            bot.search_areas["bastion_to_main_menu"],
            bot.search_areas["menu_name"],
            "Modos de juego",
        )
        bot.navigate_bastion_once_after_restart = False

    timers = {"classic": None, "tagteam": None, "live": None}
    refresh_interval = 15.1 * 60

    for spec in bot.mode_specs:
        if not bot.main_loop_running:
            break
        if not bot._is_mode_enabled(spec):
            continue

        bot.current_mode = spec.key
        LOG.info(
            "one-round mode start for '%s': %s (%s)",
            account_name,
            spec.display_name,
            spec.key,
        )
        spec.executor(timers, refresh_interval)
        bot._send_mode_heartbeat(spec.display_name)
        LOG.info(
            "one-round mode end for '%s': %s (%s)",
            account_name,
            spec.display_name,
            spec.key,
        )

    LOG.info(
        "one-round quest/reward phase intentionally skipped for '%s' (Mercado/Market + Guardian Ring safeguards).",
        account_name,
    )
    bot.current_mode = "stopped"
    bot.main_loop_running = False
    bot.main_loop_stopped = True
    LOG.info("one-round execution end for '%s'", account_name)


def _run_account_once(
    account_name: str,
    plarium_timeout: float,
    raid_timeout: float,
    cleanup_wait: float,
) -> bool:
    LOG.info("current account target: %s", account_name)

    try:
        profile_resolution = _verify_profile_resolution(account_name)
    except Exception as exc:
        LOG.error("profile verification failed for '%s': %s", account_name, exc)
        return False

    LOG.info(
        "profile path loaded for '%s': %s",
        account_name,
        profile_resolution.selected_param_file,
    )

    previous_account_env = os.environ.get(RAID_ACCOUNT_NAME_ENV)
    os.environ[RAID_ACCOUNT_NAME_ENV] = account_name
    bot: RSL_Bot_Mainframe | None = None
    success = False

    try:
        import benchmarks.multi_account_startup as startup
        from Raid_Bot import RSL_Bot_Mainframe

        LOG.info("account switch start for '%s'", account_name)
        LOG.info("Raid launch start for '%s'", account_name)
        startup.run(
            account_name=account_name,
            plarium_timeout=plarium_timeout,
            raid_timeout=raid_timeout,
            debug_squares=False,
            num_squares=1,
            allow_manual_fallback=False,
        )
        LOG.info("account switch result for '%s': success", account_name)
        LOG.info("Raid launch result for '%s': success", account_name)

        bot = RSL_Bot_Mainframe()
        loaded_account = str(getattr(bot, "profile_account_name", "") or "").strip().lower()
        if loaded_account != account_name:
            raise RuntimeError(
                "Loaded profile mismatch after startup: "
                f"target='{account_name}' loaded='{loaded_account}' file='{bot.param_file}'."
            )

        LOG.info("runtime profile verified for '%s': %s", account_name, bot.param_file)
        _run_exactly_one_round(bot, account_name)
        success = True
        return True
    except Exception as exc:
        LOG.exception("account failure for '%s': %s", account_name, exc)
        return False
    finally:
        try:
            if bot is not None:
                bot.main_loop_running = False
                bot.main_loop_stopped = True
                try:
                    bot.discord_override.stop()
                except Exception:
                    pass
        finally:
            LOG.info(
                "cleanup/reset between accounts for '%s' (success=%s)",
                account_name,
                success,
            )
            _cleanup_between_accounts(wait_seconds=cleanup_wait)
            if previous_account_env is None:
                os.environ.pop(RAID_ACCOUNT_NAME_ENV, None)
            else:
                os.environ[RAID_ACCOUNT_NAME_ENV] = previous_account_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one benchmark round for each secondary Raid account "
            "(artus2..artus7) with strict account/profile safety checks."
        )
    )
    parser.add_argument(
        "--plarium-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for Plarium Play window.",
    )
    parser.add_argument(
        "--raid-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for Raid window.",
    )
    parser.add_argument(
        "--cleanup-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after cleanup between accounts.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop benchmark immediately on first per-account failure.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(verbose=bool(args.verbose))

    LOG.info("benchmark start")
    LOG.info("account list being processed: %s", ", ".join(SECONDARY_ACCOUNTS))
    LOG.info("main account 'artus' is intentionally excluded")

    total = len(SECONDARY_ACCOUNTS)
    failures: list[str] = []

    for index, account_name in enumerate(SECONDARY_ACCOUNTS, start=1):
        LOG.info("processing account %s/%s: %s", index, total, account_name)
        ok = _run_account_once(
            account_name=account_name,
            plarium_timeout=float(args.plarium_timeout),
            raid_timeout=float(args.raid_timeout),
            cleanup_wait=float(args.cleanup_wait),
        )
        if not ok:
            failures.append(account_name)
            if bool(args.stop_on_error):
                LOG.error(
                    "benchmark aborting due to --stop-on-error after failure on '%s'",
                    account_name,
                )
                break

    if failures:
        LOG.error("benchmark completion with failures: %s", ", ".join(failures))
        return 1

    LOG.info("benchmark completion: all secondary accounts finished one round")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
