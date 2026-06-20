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

raid_bot_pkg = sys.modules.setdefault("raid_bot", types.ModuleType("raid_bot"))
raid_bot_pkg.__path__ = [str(REPO_ROOT / "src" / "raid_bot")]
utils_pkg = sys.modules.setdefault("raid_bot.utils", types.ModuleType("raid_bot.utils"))
utils_pkg.__path__ = [str(REPO_ROOT / "src" / "raid_bot" / "utils")]
modes_pkg = sys.modules.setdefault("raid_bot.modes", types.ModuleType("raid_bot.modes"))
modes_pkg.__path__ = [str(REPO_ROOT / "src" / "raid_bot" / "modes")]

_install_stub_module("pyautogui")
_install_stub_module("pygetwindow")
_install_stub_module("keyboard")
_install_stub_module("matplotlib")
_install_stub_module("matplotlib.pyplot")
_install_stub_module("easyocr")

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

from raid_bot.modes import factionwars_tools


class DummyText:
    def __init__(self, text=None, mean_pos_x=0.0, mean_pos_y=0.0):
        self.text = text
        self.mean_pos_x = mean_pos_x
        self.mean_pos_y = mean_pos_y


class FactionWarsStageSelectionTests(unittest.TestCase):
    def setUp(self):
        self.bot = factionwars_tools.RSL_Bot_FactionWars(
            reader=object(),
            window=types.SimpleNamespace(left=0, top=0, width=1000, height=1000),
            verbose=False,
        )

    def test_dynamic_stage_selection_interpolates_missing_stage_from_spacing(self):
        stage_4 = DummyText("Etapa 4", mean_pos_x=420.0, mean_pos_y=100.0)
        stage_6 = DummyText("Etapa 6", mean_pos_x=422.0, mean_pos_y=300.0)

        with patch.object(
            factionwars_tools.image_tools,
            "get_text_in_relative_area",
            return_value=[stage_4, stage_6],
        ), patch.object(factionwars_tools.window_tools, "move_down") as move_down, patch.object(
            factionwars_tools.window_tools, "move_up"
        ) as move_up:
            area = self.bot._select_stage_button_area_dynamic(5, max_scroll_attempts=1)

        self.assertIsNotNone(area)
        move_down.assert_not_called()
        move_up.assert_not_called()
        self.assertAlmostEqual(area[0] + area[2] / 2.0, 0.421, places=3)
        self.assertAlmostEqual(area[1] + area[3] / 2.0, 0.200, places=3)

    def test_dynamic_stage_selection_scrolls_down_when_next_stage_is_below_viewport(self):
        stage_4 = DummyText("Etapa 4", mean_pos_x=420.0, mean_pos_y=820.0)
        stage_6 = DummyText("Etapa 6", mean_pos_x=422.0, mean_pos_y=930.0)
        stage_7 = DummyText("Etapa 7", mean_pos_x=423.0, mean_pos_y=860.0)

        calls = [ [stage_4, stage_6], [stage_6, stage_7] ]

        def fake_get_text_in_relative_area(*args, **kwargs):
            return calls.pop(0)

        with patch.object(
            factionwars_tools.image_tools,
            "get_text_in_relative_area",
            side_effect=fake_get_text_in_relative_area,
        ), patch.object(factionwars_tools.window_tools, "move_down") as move_down, patch.object(
            factionwars_tools.window_tools, "move_up"
        ) as move_up:
            area = self.bot._select_stage_button_area_dynamic(7, max_scroll_attempts=2)

        self.assertIsNotNone(area)
        move_up.assert_not_called()
        move_down.assert_called_once_with(self.bot.window, strength=0.2)
        self.assertAlmostEqual(area[1] + area[3] / 2.0, 0.860, places=3)


class FactionWarsBattleOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.bot = factionwars_tools.RSL_Bot_FactionWars(
            reader=object(),
            window=types.SimpleNamespace(left=0, top=0, width=1000, height=1000),
            verbose=False,
        )

    def test_missing_confirmation_waits_for_second_result(self):
        results = ["VICTORIA", None]

        with patch.object(self.bot, "_read_battle_result_once", side_effect=lambda: results.pop(0)), patch.object(
            factionwars_tools.time, "sleep"
        ):
            self.bot.update_battle_outcome()

        self.assertNotEqual(self.bot.battle_status, "Done")
        self.assertEqual(self.bot.battles_done, 0)
        self.assertEqual(self.bot.battles_won, 0)
        self.assertIsNone(self.bot.last_battle_result)

    def test_conflicting_confirmation_keeps_battle_running(self):
        results = ["VICTORIA", "DERROTA"]

        with patch.object(self.bot, "_read_battle_result_once", side_effect=lambda: results.pop(0)), patch.object(
            factionwars_tools.time, "sleep"
        ):
            self.bot.update_battle_outcome()

        self.assertNotEqual(self.bot.battle_status, "Done")
        self.assertEqual(self.bot.battles_done, 0)
        self.assertIsNone(self.bot.last_battle_result)


if __name__ == "__main__":
    unittest.main()
