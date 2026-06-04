import unittest

from data.lib.core import runtime_reporting


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
                "- status",
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


if __name__ == "__main__":
    unittest.main()
