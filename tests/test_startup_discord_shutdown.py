from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import patch

from raid_bot.core import runtime_startup


class StartupDiscordShutdownTests(unittest.TestCase):
    def test_close_discord_desktop_app_only_terminates_running_discord_processes(self):
        killed_processes: list[str] = []

        def fake_run(cmd, stdout=None, stderr=None, check=None):
            if cmd[:2] == ["tasklist", "/FI"]:
                process_name = cmd[2].split(" eq ", 1)[1]
                if process_name == "Discord.exe":
                    return types.SimpleNamespace(stdout=b'"Discord.exe","1234"\r\n')
                return types.SimpleNamespace(
                    stdout=b"INFO: No tasks are running which match the specified criteria.\r\n"
                )

            if cmd[:2] == ["taskkill", "/F"] and cmd[2] == "/IM":
                killed_processes.append(cmd[3])
                return types.SimpleNamespace(stdout="", returncode=0)

            raise AssertionError(f"Unexpected command: {cmd}")

        with patch.object(runtime_startup.subprocess, "run", side_effect=fake_run), patch.object(
            runtime_startup.time, "sleep", return_value=None
        ):
            result = runtime_startup.close_discord_desktop_app()

        self.assertTrue(result)
        self.assertEqual(killed_processes, ["Discord.exe"])

    def test_tasklist_helpers_handle_non_utf8_output(self):
        def fake_run(cmd, stdout=None, stderr=None, check=None):
            return types.SimpleNamespace(stdout=b'"Discord.exe","\x81"\r\n')

        with patch.object(runtime_startup.subprocess, "run", side_effect=fake_run):
            self.assertTrue(runtime_startup._is_process_running_by_name("Discord.exe"))

    def test_launchers_call_shared_helper_before_startup(self):
        repo_root = Path(__file__).resolve().parents[1]
        raid_bot_source = (repo_root / "src" / "raid_bot" / "mainframe.py").read_text(encoding="utf-8")
        run_bot_source = (repo_root / "src" / "raid_bot" / "run_bot.py").read_text(encoding="utf-8")

        self.assertIn("close_discord_desktop_app()", raid_bot_source)
        self.assertIn("close_discord_desktop_app()", run_bot_source)
        self.assertIn("from raid_bot.core.runtime_startup import close_discord_desktop_app", run_bot_source)


if __name__ == "__main__":
    unittest.main()
