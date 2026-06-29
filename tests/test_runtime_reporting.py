import unittest
import tempfile
from pathlib import Path

from raid_bot.core import runtime_reporting
from raid_bot.utils import file_tools


class RuntimeReportingTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {
            "timestamp_cest": "2026-05-25 19:00:00",
            "uptime_seconds": 42,
            "current_mode": "hydra",
            "manual_mode": False,
            "main_loop_running": True,
            "enabled_modes": ["hydra", "grimforest"],
            "last_error": None,
        }

    def test_help_lines_preserve_command_text_and_append_parameter_overview(self):
        overview = ["", "[Bot Help] Editable parameters: none loaded."]

        self.assertEqual(
            runtime_reporting.build_help_lines(self.snapshot, overview),
            [
                "[Bot Help] Available commands:",
                "- start / resume",
                "- stop / pause",
                "- restart",
                "- status (current daily entry)",
                "- show_stats / stats",
                "- modes",
                "- params [filter]",
                "- get <parameter_name>",
                "- set <parameter_name> <value>",
                "- toggle <mode_name> [on|off]",
                "- reload / reload_config",
                "- ping",
                "",
                "[Bot Help] Runtime snapshot:",
                "- current_mode: hydra",
                "- manual_mode: False",
                "- main_loop_running: True",
                "- enabled_modes: hydra, grimforest",
                "",
                "[Bot Help] Editable parameters: none loaded.",
            ],
        )

    def test_status_lines_only_include_last_error_when_present(self):
        self.assertEqual(
            runtime_reporting.build_status_lines(self.snapshot),
            [
                "[Bot Status]",
                "- timestamp_cest: 2026-05-25 19:00:00",
                "- uptime_seconds: 42",
                "- current_mode: hydra",
                "- manual_mode: False",
                "- main_loop_running: True",
                "- enabled_modes: hydra, grimforest",
            ],
        )

        failed_snapshot = dict(self.snapshot, last_error="failure")
        self.assertEqual(
            runtime_reporting.build_status_lines(failed_snapshot)[-1],
            "- last_error: failure",
        )

    def test_mode_and_parameter_lines_preserve_sort_filter_and_truncation_behavior(self):
        self.assertEqual(
            runtime_reporting.build_modes_lines({"hydra": True, "arena": False}),
            ["[Bot Modes]", "- arena: DISABLED", "- hydra: ENABLED"],
        )

        values = {"run_hydra": True, "run_grimforest": False, "verbose": True}
        self.assertEqual(
            runtime_reporting.build_params_lines(
                sorted(values),
                values.get,
                search_term="run_",
                max_items=1,
            ),
            [
                "[Bot Config] Parameters:",
                "- run_grimforest = False",
                "- ... 1 more (refine filter with `params <text>`)",
            ],
        )

    def test_parameter_overview_and_value_formatting_preserve_layout_rules(self):
        self.assertEqual(
            runtime_reporting.build_editable_params_overview_lines([]),
            ["", "[Bot Help] Editable parameters: none loaded."],
        )
        lines = runtime_reporting.build_editable_params_overview_lines(
            ["run_hydra", "run_grimforest"],
            max_line_length=17,
        )
        self.assertEqual(lines[2:4], ["- `run_hydra`", "- `run_grimforest`"])
        self.assertEqual(runtime_reporting.format_value("abcdefghij", max_len=8), "'abcd...")

    def test_daily_entry_messages_render_pretty_json_chunks(self):
        messages = runtime_reporting.build_daily_entry_messages(
            "2026_05_25",
            {
                "date": "2026_05_25",
                "summary": {"market": {"mystery_shards_bought": 2}},
                "state": {"enemy_avoid": {"cursed_city": ["frost spider"]}},
            },
        )

        self.assertEqual(messages[0], "[Bot Status] Current daily entry for `2026_05_25`")
        self.assertTrue(messages[1].startswith("```json\n"))
        self.assertIn('"mystery_shards_bought": 2', messages[1])
        self.assertIn('"cursed_city": [', messages[1])

    def test_daily_stats_figure_builds_a_multi_panel_image(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            log_path = tmp_path / "raidbot_daily_report.json"
            file_tools.save_daily_log_document(
                log_path,
                {
                    "version": 1,
                    "metadata": {"created_at_utc": "2026-05-25T00:00:00Z", "header_lines": []},
                    "days": {
                        "2026_05_23": {
                            "date": "2026_05_23",
                            "created_at_utc": "2026-05-23T00:00:00Z",
                            "updated_at_utc": "2026-05-23T00:00:00Z",
                            "account": "artus",
                            "profile_account": "artus",
                            "events": [],
                            "summary": {
                                "pvp": {"classic_arena": {"wins": 1, "losses": 0}},
                                "faction_wars": {"wins": 2, "losses": 1, "progress_mode_factions": ["a"]},
                                "dungeons": {
                                    "iron_twins": {
                                        "wins": 3,
                                        "losses": 1,
                                        "energy_spent": 12,
                                        "iron_twins_keys_used": 2,
                                    },
                                    "daily_shogun": {"wins": 4, "losses": 0, "energy_spent": 8},
                                },
                                "keys": {
                                    "cursed_city": {"used_keys": 5},
                                    "grim_forest": {"used_keys": 6},
                                    "doom_tower": {"silver_keys_used": 7, "gold_keys_used": 1},
                                },
                                "market": {"mystery_shards_bought": 1},
                                "guardian_ring": {"successful_entries": 1},
                            },
                            "state": {"enemy_avoid": {"cursed_city": [], "grim_forest": []}},
                        },
                        "2026_05_24": {
                            "date": "2026_05_24",
                            "created_at_utc": "2026-05-24T00:00:00Z",
                            "updated_at_utc": "2026-05-24T00:00:00Z",
                            "account": "artus",
                            "profile_account": "artus",
                            "events": [],
                            "summary": {
                                "pvp": {"classic_arena": {"wins": 2, "losses": 1}},
                                "faction_wars": {"wins": 1, "losses": 2, "progress_mode_factions": ["a", "b"]},
                                "dungeons": {
                                    "iron_twins": {
                                        "wins": 1,
                                        "losses": 0,
                                        "energy_spent": 4,
                                        "iron_twins_keys_used": 1,
                                    },
                                    "spider": {"wins": 3, "losses": 1, "energy_spent": 18},
                                    "daily_shogun": {"wins": 2, "losses": 1, "energy_spent": 7},
                                },
                                "keys": {
                                    "cursed_city": {"used_keys": 2},
                                    "grim_forest": {"used_keys": 3},
                                    "doom_tower": {"silver_keys_used": 4, "gold_keys_used": 2},
                                },
                                "market": {"mystery_shards_bought": 2},
                                "guardian_ring": {"successful_entries": 1},
                            },
                            "state": {"enemy_avoid": {"cursed_city": ["enemy 1"], "grim_forest": []}},
                        },
                    },
                },
            )

            output_path = runtime_reporting.build_daily_stats_figure(
                log_path,
                output_path=tmp_path / "raidbot_daily_stats.png",
                max_plots=16,
            )

            self.assertEqual(output_path.name, "raidbot_daily_stats.png")
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
