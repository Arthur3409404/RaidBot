import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data.lib.modes import cursedcity_tools, grimforest_tools


class SimulatedLossBot(grimforest_tools.RSL_Bot_GrimForest):
    candidate = {
        "index": 1,
        "score": 0.91,
        "center_rel_x": 0.25,
        "center_rel_y": 0.30,
        "bbox_rel": {"x": 0.20, "y": 0.25, "width": 0.10, "height": 0.10},
        "center_abs_x": 100,
        "center_abs_y": 120,
    }

    def _perform_startup_check(self):
        return True

    def set_difficulty(self, set_level=None):
        self.current_difficulty = set_level
        return set_level

    def has_grim_forest_keys_remaining(self, retries=3):
        return True

    def detect_candidates_with_random_reposition(self, difficulty=None):
        return [dict(self.candidate)]

    def select_grim_forest_candidate(self, candidate):
        return True

    def click_grim_forest_start_button(self, retries=None):
        return "battle_started"

    def get_battle_outcome(self, timeout_seconds=None, poll_interval_seconds=None):
        return "Derrota"

    def return_to_mode_root_after_battle(self, max_attempts=4):
        return "mode"

    def select_post_battle_stat_reward(self):
        return None

    def exit_grim_forest_to_main_menu(self, reason):
        self.exit_reason = reason
        self.mode_transitioned_out = True
        return True


class GrimForestToolsTests(unittest.TestCase):
    def _setup_paths(self, directory, **extra):
        setup = {
            "run_state_path": str(Path(directory) / "run_state.json"),
            "last_defeat_path": str(Path(directory) / "last_defeat.json"),
            "post_entry_wait_seconds": 0,
        }
        setup.update(extra)
        return setup

    def test_public_tool_module_and_cursed_city_still_import(self):
        self.assertTrue(hasattr(grimforest_tools, "RSL_Bot_GrimForest"))
        self.assertTrue(callable(grimforest_tools.detect_grimforest_like_structures))
        self.assertTrue(hasattr(cursedcity_tools, "RSL_Bot_CursedCity"))

    def test_default_difficulty_rotation_matches_cursed_city_and_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            setup = self._setup_paths(directory)
            first = grimforest_tools.RSL_Bot_GrimForest(setup=setup)
            second = grimforest_tools.RSL_Bot_GrimForest(setup=setup)

            self.assertEqual(first._plan_and_commit_run_difficulty(), "hard")
            self.assertEqual(second._plan_and_commit_run_difficulty(), "normal")
            payload = json.loads(Path(setup["run_state_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["run_counter"], 2)
            self.assertEqual(payload["last_used_difficulty"], "normal")

    def test_difficulty_switch_selects_requested_menu_option(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = grimforest_tools.RSL_Bot_GrimForest(
                window=object(),
                setup=self._setup_paths(directory, difficulty_switch_retries=1),
            )
            self.assertEqual(bot.search_areas["mode_difficulty_current"], [0.03, 0.917, 0.079, 0.043])
            self.assertEqual(bot.search_areas["mode_difficulty_switch_normal"], [0.092, 0.798, 0.08, 0.036])
            self.assertEqual(bot.search_areas["mode_difficulty_switch_hard"], [0.096, 0.865, 0.066, 0.038])
            with patch.object(bot, "detect_current_difficulty", side_effect=["hard", "normal"]):
                with patch.object(grimforest_tools.window_tools, "click_center") as click_center:
                    self.assertEqual(bot.set_difficulty("normal"), "normal")
            self.assertEqual(click_center.call_count, 2)
            self.assertEqual(click_center.call_args_list[1].args[1], bot.search_areas["mode_difficulty_switch_normal"])

    def test_configured_difficulty_is_used_when_alternation_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            setup = self._setup_paths(directory, alternate_difficulty=False, difficulty="normal")
            bot = grimforest_tools.RSL_Bot_GrimForest(setup=setup)
            self.assertEqual(bot._plan_and_commit_run_difficulty(), "normal")

    def test_previous_loss_is_loaded_and_filters_same_location(self):
        with tempfile.TemporaryDirectory() as directory:
            setup = self._setup_paths(directory)
            previous = SimulatedLossBot.candidate
            writer = grimforest_tools.RSL_Bot_GrimForest(setup=setup)
            writer._record_last_defeat_candidate("hard", previous)

            reader = grimforest_tools.RSL_Bot_GrimForest(setup=setup)
            far_candidate = dict(previous)
            far_candidate["bbox_rel"] = {"x": 0.75, "y": 0.75, "width": 0.10, "height": 0.10}
            filtered = reader._filter_candidates_against_last_defeat([previous, far_candidate], "hard")
            self.assertEqual(filtered, [far_candidate])

    def test_failed_run_saves_previous_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            setup = self._setup_paths(directory, alternate_difficulty=False, difficulty="hard")
            bot = SimulatedLossBot(reader=object(), window=object(), setup=setup)

            self.assertTrue(bot.run_grimforest())
            self.assertEqual(bot.exit_reason, "battle_lost")
            payload = json.loads(Path(setup["last_defeat_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["hard"]["outcome"], "Derrota")
            self.assertEqual(payload["hard"]["bbox_rel"], SimulatedLossBot.candidate["bbox_rel"])

    def test_failed_run_saves_loss_even_after_returning_to_game_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            setup = self._setup_paths(directory, alternate_difficulty=False, difficulty="hard")
            bot = SimulatedLossBot(reader=object(), window=object(), setup=setup)
            bot.return_to_mode_root_after_battle = lambda max_attempts=4: "game_modes"

            self.assertTrue(bot.run_grimforest())
            payload = json.loads(Path(setup["last_defeat_path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["hard"]["outcome"], "Derrota")

    def test_raidbot_source_registers_and_dispatches_grim_forest_tool(self):
        source = Path("Raid_Bot.py").read_text(encoding="utf-8")
        self.assertIn("grimforest_tools.RSL_Bot_GrimForest(", source)
        self.assertIn('key="grimforest"', source)
        self.assertIn("self.grimforest_bot.run_grimforest(", source)
        self.assertIn("for swipe_index in range(2):", source)
        self.assertIn("move(self.window, strength=1.0, relative_x=0.5, relative_y=0.72)", source)


if __name__ == "__main__":
    unittest.main()
