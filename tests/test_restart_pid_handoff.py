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

    def test_regular_duplicate_launch_still_exits(self):
        with patch.object(run_bot, "_read_running_instance_pid", return_value=1234):
            result = run_bot.main()

        self.assertEqual(result, 1)

    def test_restart_duplicate_replaces_previous_pid_and_starts(self):
        bot = types.SimpleNamespace(start_main_loop=lambda: None)

        with patch.dict(
            os.environ,
            {
                run_bot.RESTART_NOTIFY_ENV: "1",
                run_bot.RESTART_REPLACE_PID_ENV: "1234",
            },
            clear=False,
        ):
            with patch.object(run_bot, "_read_running_instance_pid", return_value=1234):
                with patch.object(run_bot, "_kill_pid_tree") as kill_pid_tree:
                    with patch.object(run_bot, "_wait_for_pid_exit", return_value=True) as wait_for_pid_exit:
                        with patch.object(run_bot, "close_discord_desktop_app"):
                            with patch.object(run_bot, "_write_pid_file"):
                                with patch.object(run_bot.atexit, "register"):
                                    with patch.object(run_bot, "RSL_Bot_Mainframe", return_value=bot):
                                        result = run_bot.main()

        self.assertEqual(result, 0)
        kill_pid_tree.assert_called_once_with(1234)
        wait_for_pid_exit.assert_called_once_with(1234)

    def test_mainframe_entrypoint_replaces_stale_pid_during_restart(self):
        bot = types.SimpleNamespace(start_main_loop=lambda: None)
        window = types.SimpleNamespace(moveTo=lambda *_: None)

        with patch.object(mainframe, "_read_running_instance_pid", return_value=21288):
            with patch.object(mainframe, "_is_restart_replacement_launch", return_value=True):
                with patch.object(mainframe, "_kill_process_tree_by_pid") as kill_process_tree:
                    with patch.object(mainframe, "_wait_for_process_exit", return_value=True) as wait_for_exit:
                        with patch.object(mainframe, "_write_current_pid_file") as write_pid:
                            with patch.object(mainframe.atexit, "register"):
                                with patch.object(mainframe, "close_discord_desktop_app"):
                                    with patch.object(mainframe.gw, "getWindowsWithTitle", return_value=[window]):
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
            with patch.object(mainframe, "_is_restart_replacement_launch", return_value=False):
                with patch.object(mainframe, "_is_entrypoint_relaunch", return_value=True):
                    with patch.object(mainframe, "_kill_process_tree_by_pid") as kill_process_tree:
                        with patch.object(mainframe, "_wait_for_process_exit", return_value=True) as wait_for_exit:
                            with patch.object(mainframe, "_write_current_pid_file") as write_pid:
                                with patch.object(mainframe.atexit, "register"):
                                    with patch.object(mainframe, "close_discord_desktop_app"):
                                        with patch.object(mainframe.gw, "getWindowsWithTitle", return_value=[window]):
                                            with patch.object(mainframe.time, "sleep", return_value=None):
                                                with patch.object(mainframe, "RSL_Bot_Mainframe", return_value=bot):
                                                    result = mainframe.main()

        self.assertEqual(result, 0)
        kill_process_tree.assert_called_once_with(21288)
        wait_for_exit.assert_called_once_with(21288, timeout_seconds=10.0)
        write_pid.assert_called_once()


if __name__ == "__main__":
    unittest.main()
