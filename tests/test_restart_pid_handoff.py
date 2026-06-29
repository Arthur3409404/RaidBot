from __future__ import annotations

import os
import types
import unittest
from unittest.mock import patch

os.environ.setdefault("CONDA_DEFAULT_ENV", "RaidEnv")

from raid_bot import mainframe, run_bot


class RestartPidHandoffTests(unittest.TestCase):
    def test_restart_handoff_uses_root_raid_bot_entrypoint(self):
        self.assertTrue(mainframe._resolve_run_bot_script().endswith("Raid_Bot.py"))

    def test_regular_duplicate_launch_overwrites_previous_instance(self):
        bot = types.SimpleNamespace(start_main_loop=lambda: None)

        with patch.object(run_bot, "_read_running_instance_pid", return_value=1234):
            with patch.object(run_bot, "_kill_pid_tree") as kill_pid_tree:
                with patch.object(run_bot, "_wait_for_pid_exit", return_value=True) as wait_for_pid_exit:
                    with patch.object(run_bot, "close_discord_desktop_app"):
                        with patch.object(run_bot, "_write_pid_file"):
                            with patch.object(run_bot.atexit, "register"):
                                with patch.object(run_bot.gw, "getWindowsWithTitle", return_value=[types.SimpleNamespace(moveTo=lambda *_: None)]):
                                    with patch.object(run_bot, "_validate_raid_window_size", return_value=True):
                                        with patch.object(run_bot, "RSL_Bot_Mainframe", return_value=bot):
                                            result = run_bot.main()

        self.assertEqual(result, 0)
        kill_pid_tree.assert_called_once_with(1234)
        wait_for_pid_exit.assert_called_once_with(1234)

    def test_restart_duplicate_also_overwrites_previous_pid(self):
        bot = types.SimpleNamespace(start_main_loop=lambda: None)

        with patch.object(run_bot, "_read_running_instance_pid", return_value=1234):
            with patch.object(run_bot, "_kill_pid_tree") as kill_pid_tree:
                with patch.object(run_bot, "_wait_for_pid_exit", return_value=True) as wait_for_pid_exit:
                    with patch.object(run_bot, "close_discord_desktop_app"):
                        with patch.object(run_bot, "_write_pid_file"):
                            with patch.object(run_bot.atexit, "register"):
                                with patch.object(run_bot.gw, "getWindowsWithTitle", return_value=[types.SimpleNamespace(moveTo=lambda *_: None)]):
                                    with patch.object(run_bot, "_validate_raid_window_size", return_value=True):
                                        with patch.object(run_bot, "RSL_Bot_Mainframe", return_value=bot):
                                            result = run_bot.main()

        self.assertEqual(result, 0)
        kill_pid_tree.assert_called_once_with(1234)
        wait_for_pid_exit.assert_called_once_with(1234)

    def test_mainframe_entrypoint_replaces_stale_pid_during_restart(self):
        bot = types.SimpleNamespace(start_main_loop=lambda: None)
        window = types.SimpleNamespace(moveTo=lambda *_: None)

        with patch.object(mainframe, "_read_running_instance_pid", return_value=21288):
            with patch.object(mainframe, "_kill_process_tree_by_pid") as kill_process_tree:
                with patch.object(mainframe, "_wait_for_process_exit", return_value=True) as wait_for_exit:
                    with patch.object(mainframe, "_write_current_pid_file") as write_pid:
                        with patch.object(mainframe.atexit, "register"):
                            with patch.object(mainframe, "close_discord_desktop_app"):
                                with patch.object(mainframe.gw, "getWindowsWithTitle", return_value=[window]):
                                    with patch.object(mainframe, "_validate_raid_window_size", return_value=True):
                                        with patch.object(mainframe.time, "sleep", return_value=None):
                                            with patch.object(mainframe, "RSL_Bot_Mainframe", return_value=bot):
                                                result = mainframe.main()

        self.assertEqual(result, 0)
        kill_process_tree.assert_called_once_with(21288)
        wait_for_exit.assert_called_once_with(21288, timeout_seconds=10.0)
        write_pid.assert_called_once()

    def test_mainframe_entrypoint_relaunch_replaces_stale_pid(self):
        bot = types.SimpleNamespace(start_main_loop=lambda: None)
        window = types.SimpleNamespace(moveTo=lambda *_: None)

        with patch.object(mainframe, "_read_running_instance_pid", return_value=21288):
            with patch.object(mainframe, "_kill_process_tree_by_pid") as kill_process_tree:
                with patch.object(mainframe, "_wait_for_process_exit", return_value=True) as wait_for_exit:
                    with patch.object(mainframe, "_write_current_pid_file") as write_pid:
                        with patch.object(mainframe.atexit, "register"):
                            with patch.object(mainframe, "close_discord_desktop_app"):
                                with patch.object(mainframe.gw, "getWindowsWithTitle", return_value=[window]):
                                    with patch.object(mainframe, "_validate_raid_window_size", return_value=True):
                                        with patch.object(mainframe.time, "sleep", return_value=None):
                                            with patch.object(mainframe, "RSL_Bot_Mainframe", return_value=bot):
                                                result = mainframe.main()

        self.assertEqual(result, 0)
        kill_process_tree.assert_called_once_with(21288)
        wait_for_exit.assert_called_once_with(21288, timeout_seconds=10.0)
        write_pid.assert_called_once()

    def test_main_loop_exception_sends_auto_restart_message_and_handoffs(self):
        bot = mainframe.RSL_Bot_Mainframe.__new__(mainframe.RSL_Bot_Mainframe)
        bot.running = True
        bot.main_loop_running = True
        bot.main_loop_stopped = False
        bot.last_loop_error = None
        bot.connectivity_supervisor = None
        bot.discord_override = types.SimpleNamespace(send_message=lambda *_: None)
        bot.log = types.SimpleNamespace(
            exception=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
        )

        with patch.object(mainframe.session_encounter_state, "reset_session_lost_encounters"):
            with patch.object(bot, "_announce_restart_success_if_requested"):
                with patch.object(bot, "run_main_loop", side_effect=RuntimeError("boom")):
                    with patch.object(bot.discord_override, "send_message") as send_message:
                        with patch.object(bot, "_handoff_full_application_restart_with_fallback", side_effect=SystemExit(0)) as handoff:
                            with self.assertRaises(SystemExit):
                                bot.start_main_loop()

        send_message.assert_called_once_with(
            "Error logged - automatic restart in progress"
        )
        handoff.assert_called_once_with("main loop exception")


if __name__ == "__main__":
    unittest.main()
