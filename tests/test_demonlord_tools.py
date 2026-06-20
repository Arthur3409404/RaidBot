import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.modules.setdefault("pyautogui", types.ModuleType("pyautogui"))
window_tools_stub = sys.modules.setdefault(
    "raid_bot.utils.window_tools",
    types.ModuleType("raid_bot.utils.window_tools"),
)
window_tools_stub.click_center = lambda *args, **kwargs: None
window_tools_stub.sendkey = lambda *args, **kwargs: None

from raid_bot.modes import demonlord_tools


class DummyText:
    def __init__(self, text):
        self.text = text


class DemonLordBattleStatusTests(unittest.TestCase):
    def test_missing_result_leaves_battle_running(self):
        bot = demonlord_tools.RSL_Bot_DemonLord(reader=object(), window=object())

        with patch.object(
            demonlord_tools.image_tools,
            "get_text_in_relative_area",
            return_value=[],
        ):
            bot.update_battle_status()

        self.assertFalse(hasattr(bot, "battle_status"))

    def test_resultado_marks_battle_done(self):
        bot = demonlord_tools.RSL_Bot_DemonLord(reader=object(), window=object())
        bot.demonlord_encounter_difficulty = "UNM"
        bot.demonlord_encounters_cleared = []

        with patch.object(
            demonlord_tools.image_tools,
            "get_text_in_relative_area",
            return_value=[DummyText("RESULTADO")],
        ):
            bot.update_battle_status()

        self.assertEqual(bot.battle_status, "Done")
        self.assertEqual(bot.demonlord_encounters_cleared, ["UNM"])


if __name__ == "__main__":
    unittest.main()
