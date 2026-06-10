from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _install_stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


REPO_ROOT = Path(__file__).resolve().parents[1]

lib_pkg = sys.modules.setdefault("raid_bot", types.ModuleType("raid_bot"))
lib_pkg.__path__ = [str(REPO_ROOT / "src" / "raid_bot")]
utils_pkg = sys.modules.setdefault("raid_bot.utils", types.ModuleType("raid_bot.utils"))
utils_pkg.__path__ = [str(REPO_ROOT / "src" / "raid_bot" / "utils")]
handlers_pkg = sys.modules.setdefault("raid_bot.handlers", types.ModuleType("raid_bot.handlers"))
handlers_pkg.__path__ = [str(REPO_ROOT / "src" / "raid_bot" / "handlers")]
modes_pkg = sys.modules.setdefault("raid_bot.modes", types.ModuleType("raid_bot.modes"))
modes_pkg.__path__ = [str(REPO_ROOT / "src" / "raid_bot" / "modes")]

_install_stub_module("pygetwindow")
_install_stub_module("pyautogui")
_install_stub_module("cv2")
_install_stub_module("keyboard")
_install_stub_module("matplotlib")
_install_stub_module("matplotlib.pyplot")
_install_stub_module("easyocr")

skimage_module = _install_stub_module("skimage")
skimage_metrics = _install_stub_module(
    "skimage.metrics",
    structural_similarity=lambda *args, **kwargs: 1.0,
)
skimage_module.metrics = skimage_metrics

_install_stub_module(
    "raid_bot.handlers.ai_networks_handler",
    EnemyDataset=type("EnemyDataset", (), {}),
    EvaluationNetwork=type("EvaluationNetwork", (), {}),
    EvaluationNetworkCNN_ImageOnly=type("EvaluationNetworkCNN_ImageOnly", (), {}),
    TagTeamEvaluationNetworkCNN=type("TagTeamEvaluationNetworkCNN", (), {}),
)
_install_stub_module(
    "raid_bot.utils.auto_battle_tools",
    reset_auto_battle_watchdog=lambda *args, **kwargs: None,
    ensure_auto_battle_running=lambda *args, **kwargs: None,
)
_install_stub_module(
    "raid_bot.utils.image_tools",
    get_text_in_relative_area=lambda *args, **kwargs: [],
    get_similarities_in_relative_area=lambda *args, **kwargs: [],
    check_startup=lambda *args, **kwargs: True,
)
_install_stub_module(
    "raid_bot.utils.window_tools",
    click_at=lambda *args, **kwargs: None,
    click_center=lambda *args, **kwargs: None,
    move_up=lambda *args, **kwargs: None,
    move_down=lambda *args, **kwargs: None,
    move_right=lambda *args, **kwargs: None,
    move_left=lambda *args, **kwargs: None,
    sendkey=lambda *args, **kwargs: None,
)

from raid_bot.modes import doomtower_tools, dungeon_tools


class DummyText:
    def __init__(self, text: str, mean_pos_x: float = 0.0, mean_pos_y: float = 0.0):
        self.text = text
        self.mean_pos_x = mean_pos_x
        self.mean_pos_y = mean_pos_y


class EventDungeonTests(unittest.TestCase):
    def setUp(self):
        self.bot = dungeon_tools.RSL_Bot_Dungeons(
            reader=object(),
            window=types.SimpleNamespace(left=0, top=0, width=1000, height=1000),
            verbose=False,
            build_name="Other Team",
        )
        self.bot.main_loop_running = True

    def test_event_dungeon_forces_event_team_name(self):
        self.assertEqual(self.bot.get_required_build_name("event_dungeon"), "Event Dungeon")

    def test_default_available_dungeons_include_minotaur_and_event_dungeon(self):
        self.assertIn("minotaur", self.bot.defaults_available)
        self.assertIn("event_dungeon", self.bot.defaults_available)

    def test_event_dungeon_resolves_to_stage_29(self):
        self.assertEqual(self.bot.resolve_dungeon_level("event_dungeon"), 29)

    def test_event_dungeon_is_always_normal_difficulty(self):
        self.assertEqual(
            self.bot.resolve_dungeon_difficulty("event_dungeon", input_difficulty="hard"),
            "normal",
        )

    def test_event_dungeon_uses_its_own_configured_level(self):
        custom_bot = dungeon_tools.RSL_Bot_Dungeons(
            reader=object(),
            window=types.SimpleNamespace(left=0, top=0, width=1000, height=1000),
            verbose=False,
            build_name="Other Team",
            eventdungeon_level=31,
        )

        self.assertEqual(custom_bot.resolve_dungeon_level("event_dungeon"), 31)

    def test_event_dungeon_selects_stage_29_via_ocr_without_static_fallback(self):
        encounter_text = DummyText("Mazmorra de Evento", mean_pos_x=120, mean_pos_y=180)
        stage_28 = DummyText("Etapa 28", mean_pos_x=100, mean_pos_y=200)
        stage_29 = DummyText("Etapa 29", mean_pos_x=150, mean_pos_y=200)
        stage_30 = DummyText("Etapa 30", mean_pos_x=200, mean_pos_y=200)

        def fake_get_text_in_relative_area(*args, **kwargs):
            search_area = kwargs.get("search_area")
            if search_area is None and len(args) >= 3:
                search_area = args[2]
            if search_area == self.bot.search_areas["pov"]:
                return [encounter_text]
            if search_area == self.bot.search_areas["dungeons_etapa_window"]:
                return [stage_28, stage_29, stage_30]
            return []

        with patch.object(dungeon_tools.image_tools, "get_text_in_relative_area", side_effect=fake_get_text_in_relative_area), patch.object(
            dungeon_tools.window_tools, "click_at"
        ) as click_at, patch.object(dungeon_tools.window_tools, "click_center") as click_center, patch.object(
            dungeon_tools.window_tools, "move_right"
        ) as move_right, patch.object(
            dungeon_tools.window_tools, "move_left"
        ) as move_left, patch.object(
            dungeon_tools.window_tools, "move_up"
        ) as move_up, patch.object(
            dungeon_tools.window_tools, "move_down"
        ) as move_down, patch.object(
            dungeon_tools.time, "sleep", return_value=None
        ):
            result = self.bot.select_encounter("event_dungeon", max_attempts=1)

        self.assertTrue(result)
        click_at.assert_called_once()
        click_center.assert_called_once()
        self.assertEqual(move_right.call_count, 0)
        self.assertEqual(move_left.call_count, 0)
        self.assertEqual(move_up.call_count, 0)
        self.assertEqual(move_down.call_count, 0)

        click_area = click_center.call_args.args[1]
        self.assertAlmostEqual(click_area[0], 0.065)
        self.assertAlmostEqual(click_area[1], 0.16)
        self.assertAlmostEqual(click_area[2], 0.17)
        self.assertAlmostEqual(click_area[3], 0.08)

    def test_event_dungeon_requires_event_team(self):
        def fake_get_text_in_relative_area(*args, **kwargs):
            search_area = kwargs.get("search_area")
            if search_area == self.bot.search_areas["dungeon_setup_names"]:
                return []
            return []

        with patch.object(dungeon_tools.image_tools, "get_text_in_relative_area", side_effect=fake_get_text_in_relative_area), patch.object(
            dungeon_tools.image_tools, "get_similarities_in_relative_area", return_value=False
        ), patch.object(dungeon_tools.window_tools, "click_center"), patch.object(
            dungeon_tools.window_tools, "click_at"
        ), patch.object(
            dungeon_tools.window_tools, "move_up"
        ), patch.object(
            dungeon_tools.window_tools, "move_down"
        ):
            result = self.bot.select_build_if_needed("event_dungeon")

        self.assertFalse(result)

    def test_dungeon_build_clicks_found_row_when_other_setup_is_selected(self):
        self.bot.build_name = "Dragon"
        setup = DummyText("Dragon", mean_pos_x=500.0, mean_pos_y=100.0)
        unrelated_selected_marker = DummyText("", mean_pos_y=500.0)

        with patch.object(dungeon_tools.image_tools, "get_text_in_relative_area", return_value=[setup]):
            with patch.object(
                dungeon_tools.image_tools,
                "get_similarities_in_relative_area",
                return_value=[unrelated_selected_marker],
            ):
                with patch.object(dungeon_tools.window_tools, "click_center"):
                    with patch.object(dungeon_tools.window_tools, "move_up"):
                        with patch.object(dungeon_tools.window_tools, "move_down"):
                            with patch.object(dungeon_tools.window_tools, "click_at") as click_at:
                                self.assertTrue(self.bot.select_build_if_needed("dragon"))

        click_at.assert_called_once_with(232.0, 170.0, delay=2, window=self.bot.window)

    def test_dungeon_build_skips_click_when_found_row_is_selected(self):
        self.bot.build_name = "Dragon"
        setup = DummyText("Dragon", mean_pos_x=500.0, mean_pos_y=100.0)
        selected_marker_on_same_row = DummyText("", mean_pos_y=170.0)

        with patch.object(dungeon_tools.image_tools, "get_text_in_relative_area", return_value=[setup]):
            with patch.object(
                dungeon_tools.image_tools,
                "get_similarities_in_relative_area",
                return_value=[selected_marker_on_same_row],
            ):
                with patch.object(dungeon_tools.window_tools, "click_center"):
                    with patch.object(dungeon_tools.window_tools, "move_up"):
                        with patch.object(dungeon_tools.window_tools, "move_down"):
                            with patch.object(dungeon_tools.window_tools, "click_at") as click_at:
                                self.assertTrue(self.bot.select_build_if_needed("dragon"))

        click_at.assert_not_called()

    def test_forced_dungeon_daily_energy_target_runs_until_estimated_target_is_met(self):
        self.bot.dungeon = "shogun"
        self.bot.difficulty = "hard"
        energy_reads = [100, 60]

        def fake_check_energy():
            self.bot.energy = energy_reads.pop(0)
            self.bot.iron_twins_keys = 6

        with patch.object(self.bot, "check_iron_twins_keys_and_energy", side_effect=fake_check_energy), patch.object(
            self.bot, "select_encounter", return_value=True
        ) as select_encounter, patch.object(self.bot, "run_encounter", return_value=True), patch.object(
            self.bot, "print_status"
        ), patch.object(
            dungeon_tools.time, "sleep", return_value=None
        ):
            spent = self.bot.run_dungeons(
                main_loop_running=True,
                forced_encounter="shogun",
                energy_target=60,
                ignore_iron_twins_priority=True,
            )

        self.assertEqual(spent, 80)
        self.assertEqual(select_encounter.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in select_encounter.call_args_list],
            ["shogun", "shogun"],
        )
        self.assertEqual(energy_reads, [])


class DoomTowerBuildSelectionTests(unittest.TestCase):
    def setUp(self):
        self.bot = doomtower_tools.RSL_Bot_DoomTower(
            reader=object(),
            window=types.SimpleNamespace(left=0, top=0, width=1000, height=1000),
            verbose=False,
        )

    def test_doom_tower_build_clicks_found_row_when_other_setup_is_selected(self):
        setup = DummyText("Waves", mean_pos_x=500.0, mean_pos_y=100.0)
        unrelated_selected_marker = DummyText("", mean_pos_y=500.0)

        with patch.object(doomtower_tools.image_tools, "get_text_in_relative_area", return_value=[setup]):
            with patch.object(
                doomtower_tools.image_tools,
                "get_similarities_in_relative_area",
                return_value=[unrelated_selected_marker],
            ):
                with patch.object(doomtower_tools.window_tools, "click_center"):
                    with patch.object(doomtower_tools.window_tools, "move_up"):
                        with patch.object(doomtower_tools.window_tools, "move_down"):
                            with patch.object(doomtower_tools.window_tools, "click_at") as click_at:
                                self.bot.select_encounter_build("Waves")

        click_at.assert_called_once_with(232.0, 170.0)

    def test_doom_tower_build_skips_click_when_found_row_is_selected(self):
        setup = DummyText("Waves", mean_pos_x=500.0, mean_pos_y=100.0)
        selected_marker_on_same_row = DummyText("", mean_pos_y=170.0)

        with patch.object(doomtower_tools.image_tools, "get_text_in_relative_area", return_value=[setup]):
            with patch.object(
                doomtower_tools.image_tools,
                "get_similarities_in_relative_area",
                return_value=[selected_marker_on_same_row],
            ):
                with patch.object(doomtower_tools.window_tools, "click_center"):
                    with patch.object(doomtower_tools.window_tools, "move_up"):
                        with patch.object(doomtower_tools.window_tools, "move_down"):
                            with patch.object(doomtower_tools.window_tools, "click_at") as click_at:
                                self.bot.select_encounter_build("Waves")

        click_at.assert_not_called()


if __name__ == "__main__":
    unittest.main()
