import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _install_stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules.setdefault(name, module)
    return sys.modules[name]


_install_stub_module("pyautogui")
_install_stub_module("pygetwindow")
_install_stub_module(
    "raid_bot.utils.window_tools",
    click_at=lambda *args, **kwargs: None,
    click_center=lambda *args, **kwargs: None,
    move_up=lambda *args, **kwargs: None,
    move_down=lambda *args, **kwargs: None,
    move_right=lambda *args, **kwargs: None,
    move_left=lambda *args, **kwargs: None,
    sendkey=lambda *args, **kwargs: None,
    zoom_out=lambda *args, **kwargs: None,
)

from raid_bot.modes import cursedcity_tools, grimforest_tools
from raid_bot.modes import session_encounter_state


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

    def test_startup_check_clears_level_prompt_before_menu_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = grimforest_tools.RSL_Bot_GrimForest(
                window=object(),
                setup=self._setup_paths(
                    directory,
                    startup_check_timeout_seconds=1,
                    startup_check_poll_interval_seconds=0,
                ),
            )

        def read_text_objects(area_key, power_detection=False):
            if area_key == "post_battle_level_prompt":
                return [SimpleNamespace(text="Subida de nivel")]
            if area_key == "post_battle_stat_options":
                return [SimpleNamespace(text="VEL", mean_pos_x=123, mean_pos_y=456)]
            return []

        with patch.object(bot, "_read_menu_name", side_effect=[None, "Bosque Lugubre"]):
            with patch.object(bot, "_read_text_objects", side_effect=read_text_objects):
                with patch.object(grimforest_tools.window_tools, "click_center") as click_center:
                    with patch.object(grimforest_tools.window_tools, "click_at") as click_at:
                        self.assertTrue(bot._perform_startup_check())

        self.assertEqual(click_center.call_count, 2)
        self.assertEqual(click_center.call_args_list[0].args[1], bot.search_areas["post_battle_level_prompt"])
        self.assertEqual(click_center.call_args_list[1].args[1], bot.search_areas["post_battle_stat_confirm"])
        click_at.assert_called_once_with(123, 456, delay=2.0, window=bot.window)

    def test_startup_check_clears_trait_card_prompt_before_menu_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = grimforest_tools.RSL_Bot_GrimForest(
                window=object(),
                setup=self._setup_paths(
                    directory,
                    startup_check_timeout_seconds=1,
                    startup_check_poll_interval_seconds=0,
                ),
            )

        def read_text_objects(area_key, power_detection=False):
            if area_key == "post_battle_level_prompt":
                return [SimpleNamespace(text="Cartas de Rasgo se otorgan a todos")]
            if area_key == "post_battle_stat_options":
                return [SimpleNamespace(text="VEL", mean_pos_x=629, mean_pos_y=667)]
            if area_key == "post_battle_stat_confirm":
                return [SimpleNamespace(text="ELEGIR")]
            return []

        with patch.object(bot, "_read_menu_name", side_effect=[None, "Bosque Lugubre"]):
            with patch.object(bot, "_read_text_objects", side_effect=read_text_objects):
                with patch.object(grimforest_tools.window_tools, "click_center") as click_center:
                    with patch.object(grimforest_tools.window_tools, "click_at") as click_at:
                        self.assertTrue(bot._perform_startup_check())

        click_center.assert_called_once_with(bot.window, bot.search_areas["post_battle_stat_confirm"], delay=2.0)
        click_at.assert_called_once_with(629, 667, delay=2.0, window=bot.window)

    def test_configured_difficulty_is_used_when_alternation_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            setup = self._setup_paths(directory, alternate_difficulty=False, difficulty="normal")
            bot = grimforest_tools.RSL_Bot_GrimForest(setup=setup)
            self.assertEqual(bot._plan_and_commit_run_difficulty(), "normal")

    def test_failed_run_adds_grim_forest_encounter_to_session_avoid_list(self):
        with tempfile.TemporaryDirectory() as directory:
            setup = self._setup_paths(directory, alternate_difficulty=False, difficulty="hard")
            bot = SimulatedLossBot(reader=object(), window=object(), setup=setup)

            self.assertTrue(bot.run_grimforest())
            self.assertEqual(bot.exit_reason, "battle_lost")
            self.assertTrue(
                session_encounter_state.is_session_lost_encounter("grim_forest", bot.current_encounter_name)
            )

    def test_failed_run_adds_encounter_even_after_returning_to_game_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            setup = self._setup_paths(directory, alternate_difficulty=False, difficulty="hard")
            bot = SimulatedLossBot(reader=object(), window=object(), setup=setup)
            bot.return_to_mode_root_after_battle = lambda max_attempts=4: "game_modes"

            self.assertTrue(bot.run_grimforest())
            self.assertTrue(
                session_encounter_state.is_session_lost_encounter("grim_forest", bot.current_encounter_name)
            )

    def test_cursed_city_uses_six_step_expanding_spiral_by_default(self):
        with patch.object(cursedcity_tools.random, "randrange", return_value=0):
            bot = cursedcity_tools.RSL_Bot_CursedCity(window=object())
        self.assertEqual(bot._max_spiral_repositions_when_no_candidates(), 6)
        self.assertEqual(bot._grid_search_size_steps(), 9)

        moves = []
        with patch.object(cursedcity_tools.window_tools, "move_right", side_effect=lambda *_, **__: moves.append("right")):
            with patch.object(cursedcity_tools.window_tools, "move_down", side_effect=lambda *_, **__: moves.append("down")):
                with patch.object(cursedcity_tools.window_tools, "move_left", side_effect=lambda *_, **__: moves.append("left")):
                    with patch.object(cursedcity_tools.window_tools, "move_up", side_effect=lambda *_, **__: moves.append("up")):
                        for _ in range(6):
                            bot._move_random_direction_once()

        self.assertEqual(
            moves,
            ["right", "down", "left", "left", "up", "up"],
        )

    def test_cursed_city_candidate_detection_does_not_zoom_out_initially(self):
        bot = cursedcity_tools.RSL_Bot_CursedCity(
            window=object(),
            setup={"max_spiral_repositions_when_no_candidates": 0},
        )

        with patch.object(bot, "detect_cursed_city_candidates", return_value=[]):
            with patch.object(cursedcity_tools.window_tools, "zoom_out") as zoom_out:
                self.assertEqual(bot.detect_candidates_with_random_reposition("hard"), [])

        zoom_out.assert_not_called()

    def test_grim_forest_uses_twenty_step_expanding_spiral_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(grimforest_tools.random, "randrange", return_value=0):
                bot = grimforest_tools.RSL_Bot_GrimForest(window=object(), setup=self._setup_paths(directory))

        self.assertEqual(bot._max_spiral_repositions_when_no_candidates(), 20)
        self.assertEqual(bot._grid_search_size_steps(), 20)

        moves = []
        with patch.object(grimforest_tools.window_tools, "move_right", side_effect=lambda *_, **__: moves.append("right")):
            with patch.object(grimforest_tools.window_tools, "move_down", side_effect=lambda *_, **__: moves.append("down")):
                with patch.object(grimforest_tools.window_tools, "move_left", side_effect=lambda *_, **__: moves.append("left")):
                    with patch.object(grimforest_tools.window_tools, "move_up", side_effect=lambda *_, **__: moves.append("up")):
                        for _ in range(10):
                            bot._move_random_direction_once()

        self.assertEqual(
            moves,
            ["right", "down", "left", "left", "up", "up", "right", "right", "right", "down"],
        )

    def test_spiral_start_direction_is_randomized_clockwise(self):
        with patch.object(cursedcity_tools.random, "randrange", return_value=2):
            bot = cursedcity_tools.RSL_Bot_CursedCity(window=object())

        moves = []
        with patch.object(cursedcity_tools.window_tools, "move_right", side_effect=lambda *_, **__: moves.append("right")):
            with patch.object(cursedcity_tools.window_tools, "move_down", side_effect=lambda *_, **__: moves.append("down")):
                with patch.object(cursedcity_tools.window_tools, "move_left", side_effect=lambda *_, **__: moves.append("left")):
                    with patch.object(cursedcity_tools.window_tools, "move_up", side_effect=lambda *_, **__: moves.append("up")):
                        for _ in range(4):
                            bot._move_random_direction_once()

        self.assertEqual(moves, ["left", "up", "right", "right"])

    def test_no_detection_escalates_stride_and_detection_resets_it(self):
        bot = cursedcity_tools.RSL_Bot_CursedCity(
            window=object(),
            setup={"max_spiral_repositions_when_no_candidates": 6},
        )
        self.assertEqual(bot._spiral_stride_for_difficulty("hard"), 1)
        bot._record_candidate_scan_result("hard", found=False)
        self.assertEqual(bot._spiral_stride_for_difficulty("hard"), 2)
        bot._record_candidate_scan_result("hard", found=False)
        self.assertEqual(bot._spiral_stride_for_difficulty("hard"), 2)
        bot._record_candidate_scan_result("hard", found=False)
        self.assertEqual(bot._spiral_stride_for_difficulty("hard"), 3)
        bot._record_candidate_scan_result("hard", found=True)
        self.assertEqual(bot._spiral_stride_for_difficulty("hard"), 1)

    def test_stride_controls_moves_before_next_detection(self):
        bot = cursedcity_tools.RSL_Bot_CursedCity(
            window=object(),
            setup={"max_spiral_repositions_when_no_candidates": 6},
        )
        bot.no_candidate_failures_by_difficulty["hard"] = 1

        moves = []
        with patch.object(bot, "detect_cursed_city_candidates", return_value=[]) as detect:
            with patch.object(cursedcity_tools.window_tools, "move_right", side_effect=lambda *_, **__: moves.append("right")):
                with patch.object(cursedcity_tools.window_tools, "move_down", side_effect=lambda *_, **__: moves.append("down")):
                    with patch.object(cursedcity_tools.window_tools, "move_left", side_effect=lambda *_, **__: moves.append("left")):
                        with patch.object(cursedcity_tools.window_tools, "move_up", side_effect=lambda *_, **__: moves.append("up")):
                            self.assertEqual(bot.detect_candidates_with_random_reposition("hard"), [])

        self.assertEqual(len(moves), 6)
        self.assertEqual(detect.call_count, 4)

    def test_cursed_city_scans_current_view_before_first_snake_move(self):
        bot = cursedcity_tools.RSL_Bot_CursedCity(window=object())
        candidate = {"score": 1.0, "bbox_rel": {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1}}

        with patch.object(bot, "detect_cursed_city_candidates", return_value=[candidate]) as detect:
            with patch.object(cursedcity_tools.window_tools, "move_right") as move_right:
                with patch.object(cursedcity_tools.window_tools, "move_down") as move_down:
                    with patch.object(cursedcity_tools.window_tools, "move_left") as move_left:
                        with patch.object(cursedcity_tools.window_tools, "move_up") as move_up:
                            self.assertEqual(bot.detect_candidates_with_random_reposition("hard"), [candidate])

        self.assertEqual(detect.call_count, 1)
        move_right.assert_not_called()
        move_down.assert_not_called()
        move_left.assert_not_called()
        move_up.assert_not_called()

    def test_cursed_city_rejected_candidate_continues_snake_without_reset(self):
        bot = cursedcity_tools.RSL_Bot_CursedCity(
            window=object(),
            setup={"grid_search_size_steps": 2},
        )

        moves = []
        with patch.object(bot, "detect_cursed_city_candidates", side_effect=[[], [{"score": 1.0, "bbox_rel": {}}], [], []]):
            with patch.object(cursedcity_tools.window_tools, "move_right", side_effect=lambda *_, **__: moves.append("right")):
                with patch.object(cursedcity_tools.window_tools, "move_down", side_effect=lambda *_, **__: moves.append("down")):
                    with patch.object(cursedcity_tools.window_tools, "move_left", side_effect=lambda *_, **__: moves.append("left")):
                        with patch.object(cursedcity_tools.window_tools, "move_up", side_effect=lambda *_, **__: moves.append("up")):
                            self.assertEqual(len(bot.detect_candidates_with_random_reposition("hard")), 1)
                            bot._move_random_direction_once()
                            self.assertEqual(bot.detect_candidates_with_random_reposition("hard"), [])

        self.assertEqual(moves[:3], ["left", "left", "down"])

    def test_grim_forest_stride_three_starts_opposite_previous_failed_start(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = grimforest_tools.RSL_Bot_GrimForest(
                window=object(),
                setup=self._setup_paths(directory, max_spiral_repositions_when_no_candidates=3),
            )
        bot.no_candidate_failures_by_difficulty["hard"] = 3
        bot.last_no_candidate_start_direction_by_difficulty["hard"] = 0

        moves = []
        with patch.object(bot, "detect_grim_forest_candidates", return_value=[]):
            with patch.object(grimforest_tools.random, "randrange", return_value=1):
                with patch.object(grimforest_tools.window_tools, "move_right", side_effect=lambda *_, **__: moves.append("right")):
                    with patch.object(grimforest_tools.window_tools, "move_down", side_effect=lambda *_, **__: moves.append("down")):
                        with patch.object(grimforest_tools.window_tools, "move_left", side_effect=lambda *_, **__: moves.append("left")):
                            with patch.object(grimforest_tools.window_tools, "move_up", side_effect=lambda *_, **__: moves.append("up")):
                                self.assertEqual(bot.detect_candidates_with_random_reposition("hard"), [])

        self.assertEqual(moves[0], "left")
        self.assertEqual(bot.last_no_candidate_start_direction_by_difficulty["hard"], 2)

    def test_grim_forest_candidate_detection_resets_stride_and_start_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = grimforest_tools.RSL_Bot_GrimForest(window=object(), setup=self._setup_paths(directory))
        bot.no_candidate_failures_by_difficulty["hard"] = 3
        bot.last_no_candidate_start_direction_by_difficulty["hard"] = 2

        candidate = {"score": 1.0, "bbox_rel": {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1}}
        with patch.object(bot, "detect_grim_forest_candidates", return_value=[candidate]):
            self.assertEqual(bot.detect_candidates_with_random_reposition("hard"), [candidate])

        self.assertEqual(bot._spiral_stride_for_difficulty("hard"), 1)
        self.assertIsNone(bot.last_no_candidate_start_direction_by_difficulty["hard"])

    def test_grim_forest_scans_current_view_before_first_snake_move(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = grimforest_tools.RSL_Bot_GrimForest(window=object(), setup=self._setup_paths(directory))

        candidate = {"score": 1.0, "bbox_rel": {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1}}
        with patch.object(bot, "detect_grim_forest_candidates", return_value=[candidate]) as detect:
            with patch.object(grimforest_tools.window_tools, "move_right") as move_right:
                with patch.object(grimforest_tools.window_tools, "move_down") as move_down:
                    with patch.object(grimforest_tools.window_tools, "move_left") as move_left:
                        with patch.object(grimforest_tools.window_tools, "move_up") as move_up:
                            self.assertEqual(bot.detect_candidates_with_random_reposition("hard"), [candidate])

        self.assertEqual(detect.call_count, 1)
        move_right.assert_not_called()
        move_down.assert_not_called()
        move_left.assert_not_called()
        move_up.assert_not_called()

    def test_grim_forest_rejected_candidate_continues_snake_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = grimforest_tools.RSL_Bot_GrimForest(
                window=object(),
                setup=self._setup_paths(directory, grid_search_size_steps=2),
            )

        moves = []
        with patch.object(bot, "detect_grim_forest_candidates", side_effect=[[], [{"score": 1.0, "bbox_rel": {}}], [], []]):
            with patch.object(grimforest_tools.window_tools, "move_right", side_effect=lambda *_, **__: moves.append("right")):
                with patch.object(grimforest_tools.window_tools, "move_down", side_effect=lambda *_, **__: moves.append("down")):
                    with patch.object(grimforest_tools.window_tools, "move_left", side_effect=lambda *_, **__: moves.append("left")):
                        with patch.object(grimforest_tools.window_tools, "move_up", side_effect=lambda *_, **__: moves.append("up")):
                            self.assertEqual(len(bot.detect_candidates_with_random_reposition("hard")), 1)
                            bot._move_random_direction_once()
                            self.assertEqual(bot.detect_candidates_with_random_reposition("hard"), [])

        self.assertEqual(moves[:3], ["left", "left", "down"])

    def test_grim_forest_hard_rejects_mimeto_encounter_after_candidate_click(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = grimforest_tools.RSL_Bot_GrimForest(
                window=object(),
                setup=self._setup_paths(directory, stage_select_delay_seconds=0),
            )
        bot.current_run_difficulty = "hard"
        candidate = {"center_abs_x": 100, "center_abs_y": 200}

        with patch.object(grimforest_tools.window_tools, "click_at") as click_at:
            with patch.object(grimforest_tools.window_tools, "sendkey") as sendkey:
                with patch.object(bot, "_read_menu_name", return_value="Mimetohadwe"):
                    self.assertFalse(bot.select_grim_forest_candidate(candidate))

        click_at.assert_called_once()
        sendkey.assert_called_once_with("esc", delay=1.0, window=bot.window)

    def test_grim_forest_session_lost_encounter_skips_candidate_once(self):
        session_encounter_state.reset_session_lost_encounters()
        session_encounter_state.add_session_lost_encounter("grim_forest", "Mimeto Dificil")
        with tempfile.TemporaryDirectory() as directory:
            bot = grimforest_tools.RSL_Bot_GrimForest(
                window=object(),
                setup=self._setup_paths(directory, stage_select_delay_seconds=0),
            )
        bot.current_run_difficulty = "hard"
        candidate = {"center_abs_x": 100, "center_abs_y": 200}

        with patch.object(grimforest_tools.window_tools, "click_at"):
            with patch.object(grimforest_tools.window_tools, "sendkey") as sendkey:
                with patch.object(bot, "_read_menu_name", return_value="Mimeto Dificil"):
                    self.assertFalse(bot.select_grim_forest_candidate(candidate))

        sendkey.assert_called_once_with("esc", delay=1.0, window=bot.window)

    def test_grim_forest_normal_allows_mimeto_encounter_name(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = grimforest_tools.RSL_Bot_GrimForest(
                window=object(),
                setup=self._setup_paths(directory, stage_select_delay_seconds=0),
            )
        bot.current_run_difficulty = "normal"
        candidate = {"center_abs_x": 100, "center_abs_y": 200}

        with patch.object(grimforest_tools.window_tools, "click_at"):
            with patch.object(grimforest_tools.window_tools, "sendkey") as sendkey:
                with patch.object(bot, "_read_menu_name", return_value="Mimetohadwe"):
                    self.assertTrue(bot.select_grim_forest_candidate(candidate))

        sendkey.assert_not_called()

    def test_grim_forest_candidate_view_escapes_when_menu_remains_grim_forest(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = grimforest_tools.RSL_Bot_GrimForest(
                window=object(),
                setup=self._setup_paths(directory, stage_select_delay_seconds=0),
            )
        bot.current_run_difficulty = "hard"
        candidate = {"center_abs_x": 100, "center_abs_y": 200}

        with patch.object(grimforest_tools.window_tools, "click_at"):
            with patch.object(grimforest_tools.window_tools, "sendkey") as sendkey:
                with patch.object(bot, "_read_menu_name", return_value="Bosque Lúgubre"):
                    self.assertFalse(bot.select_grim_forest_candidate(candidate))

        sendkey.assert_called_once_with("esc", delay=1.0, window=bot.window)

    def test_grim_forest_continues_to_next_candidate_after_rejecting_first_one(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = grimforest_tools.RSL_Bot_GrimForest(
                reader=object(),
                window=types.SimpleNamespace(left=0, top=0, width=1000, height=1000),
                setup=self._setup_paths(directory, stage_select_delay_seconds=0),
            )
        bot.main_loop_running = True
        bot.running = True
        bot.current_run_difficulty = "hard"
        bot.detect_candidates_with_random_reposition = lambda difficulty=None: [
            {"center_abs_x": 100, "center_abs_y": 200, "index": 1},
            {"center_abs_x": 300, "center_abs_y": 400, "index": 2},
        ]

        select_calls = []

        def fake_select(candidate):
            select_calls.append(candidate["index"])
            return candidate["index"] == 2

        bot.select_grim_forest_candidate = fake_select
        bot.has_grim_forest_keys_remaining = lambda retries=3: True
        bot._perform_startup_check = lambda: True
        bot.update_available_keys = lambda: 30
        bot.set_difficulty = lambda requested: requested
        bot._plan_and_commit_run_difficulty = lambda: "hard"
        bot.click_grim_forest_start_button = lambda retries=None: "battle_started"
        bot.get_battle_outcome = lambda timeout_seconds=None, poll_interval_seconds=None: "Victoria"
        bot.return_to_mode_root_after_battle = lambda max_attempts=4: "mode"
        bot.select_post_battle_stat_reward = lambda: None
        bot._move_random_direction_once = lambda: None
        bot.exit_grim_forest_to_main_menu = lambda reason: True

        original_deadline_check = grimforest_tools._ensure_within_run_deadline
        grimforest_tools._ensure_within_run_deadline = lambda *args, **kwargs: None
        try:
            self.assertTrue(bot.run_grimforest(main_loop_running=True, max_run_duration_seconds=1))
        finally:
            grimforest_tools._ensure_within_run_deadline = original_deadline_check

        self.assertEqual(select_calls, [1, 2])

    def test_grim_forest_exit_to_main_menu_uses_escape_key(self):
        with tempfile.TemporaryDirectory() as directory:
            bot = grimforest_tools.RSL_Bot_GrimForest(
                window=object(),
                setup=self._setup_paths(directory),
            )

        with patch.object(bot, "_read_menu_name", side_effect=["Bosque Lúgubre", "Modos de juego"]):
            with patch.object(grimforest_tools.window_tools, "sendkey") as sendkey:
                with patch.object(grimforest_tools.window_tools, "click_center") as click_center:
                    self.assertTrue(bot.exit_grim_forest_to_main_menu("test"))

        click_center.assert_called_once_with(bot.window, bot.search_areas["go_to_higher_menu"], delay=1.8)
        sendkey.assert_not_called()

    def test_cursed_city_rejects_forbidden_encounter_names(self):
        bot = cursedcity_tools.RSL_Bot_CursedCity(window=object())

        self.assertTrue(bot._is_forbidden_encounter_name("Borgoth the Scarab King"))
        self.assertTrue(bot._is_forbidden_encounter_name("Siroth"))
        self.assertFalse(bot._is_forbidden_encounter_name("Amius"))

    def test_legacy_random_reposition_config_still_controls_spiral_limit(self):
        cursed = cursedcity_tools.RSL_Bot_CursedCity(
            setup={"max_random_repositions_when_no_candidates": 4}
        )
        self.assertEqual(cursed._max_spiral_repositions_when_no_candidates(), 4)

        with tempfile.TemporaryDirectory() as directory:
            setup = self._setup_paths(directory, max_random_repositions_when_no_candidates=7)
            grim = grimforest_tools.RSL_Bot_GrimForest(setup=setup)
        self.assertEqual(grim._max_spiral_repositions_when_no_candidates(), 7)

    def test_raidbot_source_registers_and_dispatches_grim_forest_tool(self):
        source = Path("src/raid_bot/mainframe.py").read_text(encoding="utf-8")
        self.assertIn("grimforest_tools.RSL_Bot_GrimForest(", source)
        self.assertIn('key="grimforest"', source)
        self.assertIn("self.grimforest_bot.run_grimforest(", source)
        self.assertIn("for swipe_index in range(2):", source)
        self.assertIn("move(self.window, strength=1.0, relative_x=0.5, relative_y=0.72)", source)


if __name__ == "__main__":
    unittest.main()
