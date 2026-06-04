import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from data.lib.utils import file_tools


class FileToolsTests(unittest.TestCase):
    def test_read_params_parses_literals_and_keeps_unquoted_values_as_text(self):
        with tempfile.TemporaryDirectory() as directory:
            param_path = Path(directory) / "params.txt"
            param_path.write_text(
                "# runtime settings\n"
                "verbose = True\n"
                "screen_drift = [1, 2, 3, 4]\n"
                "dungeons_dungeon = fire_knight\n"
                "ignored line\n",
                encoding="utf-8",
            )

            self.assertEqual(
                file_tools.read_params(param_path),
                {
                    "verbose": True,
                    "screen_drift": [1, 2, 3, 4],
                    "dungeons_dungeon": "fire_knight",
                },
            )

    def test_group_params_uses_runtime_prefixes_and_leaves_mainframe_values_flat(self):
        grouped = file_tools.group_params(
            {
                "verbose": True,
                "run_hydra": False,
                "dungeons_difficulty": "normal",
                "faction_wars_farm_superraid": True,
            }
        )

        self.assertEqual(grouped["mainframe"], {"verbose": True})
        self.assertEqual(grouped["run"], {"hydra": False})
        self.assertEqual(grouped["dungeons"], {"difficulty": "normal"})
        self.assertEqual(grouped["faction_wars"], {"farm_superraid": True})

    def test_coerce_value_follows_the_existing_reference_type(self):
        self.assertTrue(file_tools.coerce_value("on", False))
        self.assertEqual(file_tools.coerce_value("42", 0), 42)
        self.assertEqual(file_tools.coerce_value("2.5", 0.0), 2.5)
        self.assertEqual(file_tools.coerce_value("'hard'", "normal"), "hard")
        self.assertEqual(file_tools.coerce_value("[1, 2]", []), [1, 2])

        with self.assertRaises(ValueError):
            file_tools.coerce_value("maybe", True)
        with self.assertRaises(ValueError):
            file_tools.coerce_value("{'not': 'a list'}", [])

    def test_parameter_store_updates_existing_values_without_reformatting_other_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            param_path = Path(directory) / "params.txt"
            param_path.write_text(
                "# keep this comment\n"
                "run_hydra = True\n"
                "dungeons_difficulty = 'normal'\n",
                encoding="utf-8",
            )
            store = file_tools.ParameterStore(param_path)

            update = store.set("run_hydra", False)
            unchanged = store.set("run_hydra", False)

            self.assertTrue(update.changed)
            self.assertTrue(update.persisted)
            self.assertFalse(unchanged.changed)
            self.assertFalse(unchanged.persisted)
            self.assertEqual(store.get("run_hydra"), False)
            self.assertEqual(
                param_path.read_text(encoding="utf-8"),
                "# keep this comment\n"
                "run_hydra = False\n"
                "dungeons_difficulty = 'normal'\n",
            )

    def test_parameter_store_only_appends_new_keys_when_explicitly_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            param_path = Path(directory) / "params.txt"
            param_path.write_text("verbose = True\n", encoding="utf-8")
            store = file_tools.ParameterStore(param_path)

            memory_only = store.set("new_setting", 1)
            self.assertTrue(memory_only.changed)
            self.assertFalse(memory_only.persisted)
            self.assertEqual(param_path.read_text(encoding="utf-8"), "verbose = True\n")

            persisted = store.set("another_setting", 2, create_if_missing=True)
            self.assertTrue(persisted.changed)
            self.assertTrue(persisted.persisted)
            self.assertEqual(
                param_path.read_text(encoding="utf-8"),
                "verbose = True\nanother_setting = 2\n",
            )

    def test_raidbot_source_mentions_separate_event_dungeon_level_config(self):
        raid_bot_source = Path("Raid_Bot.py").read_text(encoding="utf-8")
        profile_source = Path("data/profiles/artus_params_mainframe.txt").read_text(encoding="utf-8")

        self.assertIn("dungeons_eventdungeon_level", raid_bot_source)
        self.assertIn("eventdungeon_level", raid_bot_source)
        self.assertIn("dungeons_eventdungeon_level = 29", profile_source)

    def test_daily_log_helpers_create_and_append_a_shared_file(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = file_tools.get_daily_log_path(
                datetime(2026, 6, 4, 12, 30, 0),
                log_dir=Path(directory),
            )
            created = file_tools.ensure_daily_log_header(log_path, ["# header", ""])
            file_tools.append_daily_log_lines(log_path, ["entry one", "entry two"])

            self.assertTrue(created)
            self.assertEqual(
                log_path.read_text(encoding="utf-8"),
                "# header\nentry one\nentry two\n",
            )

            created_again = file_tools.ensure_daily_log_header(log_path, ["# other header"])
            self.assertFalse(created_again)
            self.assertEqual(
                log_path.read_text(encoding="utf-8"),
                "# header\nentry one\nentry two\n",
            )


if __name__ == "__main__":
    unittest.main()
