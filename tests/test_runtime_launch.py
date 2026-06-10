import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from raid_bot.core import runtime_launch


PLARIUM_ENV = "PLARIUM_PLAY_EXE"
SHORTCUT_ENV = "RAID_SHORTCUT_PATH"
SHORTCUT_NAME = "Raid Shadow Legends.lnk"
LAUNCH_ARGS = ["-gameid=101", "-tray-start"]


class RuntimeLaunchTests(unittest.TestCase):
    def _build_command(self):
        return runtime_launch.build_raid_launch_command(
            plarium_play_exe_env=PLARIUM_ENV,
            raid_shortcut_path_env=SHORTCUT_ENV,
            raid_desktop_shortcut_name=SHORTCUT_NAME,
            plarium_launch_args=LAUNCH_ARGS,
        )

    def test_shortcut_path_takes_precedence_over_launcher_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            shortcut_path = Path(directory) / "Raid.lnk"
            launcher_path = Path(directory) / "PlariumPlay.exe"
            shortcut_path.write_text("", encoding="utf-8")
            launcher_path.write_text("", encoding="utf-8")

            environment = {
                SHORTCUT_ENV: str(shortcut_path),
                PLARIUM_ENV: str(launcher_path),
            }
            with patch.dict(os.environ, environment, clear=True), patch.object(
                runtime_launch.shutil, "which", return_value=None
            ):
                command = self._build_command()

            self.assertEqual(command, ["cmd", "/c", "start", "", str(shortcut_path)])
            self.assertEqual(runtime_launch.describe_raid_launch_command(command), str(shortcut_path))

    def test_launcher_executable_is_used_when_no_shortcut_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher_path = Path(directory) / "PlariumPlay.exe"
            launcher_path.write_text("", encoding="utf-8")

            with patch.dict(os.environ, {PLARIUM_ENV: str(launcher_path)}, clear=True), patch.object(
                runtime_launch.shutil, "which", return_value=None
            ):
                command = self._build_command()

            self.assertEqual(command, [str(launcher_path), *LAUNCH_ARGS])
            self.assertEqual(runtime_launch.describe_raid_launch_command(command), str(launcher_path))

    def test_missing_launch_target_reports_existing_environment_variable_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_shortcut = Path(directory) / "missing.lnk"
            missing_launcher = Path(directory) / "missing.exe"
            environment = {
                SHORTCUT_ENV: str(missing_shortcut),
                PLARIUM_ENV: str(missing_launcher),
            }
            with patch.dict(os.environ, environment, clear=True), patch.object(
                runtime_launch.shutil, "which", return_value=None
            ):
                with self.assertRaises(FileNotFoundError) as context:
                    self._build_command()

            message = str(context.exception)
            self.assertIn(SHORTCUT_ENV, message)
            self.assertIn(PLARIUM_ENV, message)
            self.assertIn(str(missing_shortcut), message)
            self.assertIn(str(missing_launcher), message)

    def test_command_builder_accepts_existing_wrapper_resolvers(self):
        shortcut_calls = []
        launcher_calls = []

        def resolve_shortcut():
            shortcut_calls.append(True)
            return None, ["shortcut_checked"]

        def resolve_launcher():
            launcher_calls.append(True)
            return "launcher.exe", ["launcher.exe"]

        command = runtime_launch.build_raid_launch_command(
            plarium_play_exe_env=PLARIUM_ENV,
            raid_shortcut_path_env=SHORTCUT_ENV,
            raid_desktop_shortcut_name=SHORTCUT_NAME,
            plarium_launch_args=LAUNCH_ARGS,
            shortcut_resolver=resolve_shortcut,
            launcher_resolver=resolve_launcher,
        )

        self.assertEqual(command, ["launcher.exe", *LAUNCH_ARGS])
        self.assertEqual(len(shortcut_calls), 1)
        self.assertEqual(len(launcher_calls), 1)


if __name__ == "__main__":
    unittest.main()
