import unittest
from unittest.mock import patch

from raid_bot.modes import hydra_tools


class DummyText:
    def __init__(self, text=None, mean_pos_x=0.0, mean_pos_y=0.0):
        self.text = text
        self.mean_pos_x = mean_pos_x
        self.mean_pos_y = mean_pos_y


class HydraSetupSelectionTests(unittest.TestCase):
    def test_nightmare_setup_clicks_found_row_when_other_setup_is_selected(self):
        bot = hydra_tools.RSL_Bot_Hydra(reader=object(), window=object())
        setup = DummyText("NM", mean_pos_x=500.0, mean_pos_y=100.0)
        unrelated_selected_marker = DummyText(mean_pos_y=500.0)

        with patch.object(hydra_tools.window_tools, "click_center"):
            with patch.object(hydra_tools.window_tools, "move_up"):
                with patch.object(hydra_tools.window_tools, "move_down"):
                    with patch.object(hydra_tools.image_tools, "get_text_in_relative_area", return_value=[setup]):
                        with patch.object(
                            hydra_tools.image_tools,
                            "get_similarities_in_relative_area",
                            return_value=[unrelated_selected_marker],
                        ):
                            with patch.object(hydra_tools.window_tools, "click_at") as click_at:
                                self.assertTrue(bot._select_hydra_setup("Nightmare"))

        click_at.assert_called_once_with(180.0, 170.0, delay=2, window=bot.window)

    def test_nightmare_setup_skips_click_when_found_row_is_already_selected(self):
        bot = hydra_tools.RSL_Bot_Hydra(reader=object(), window=object())
        setup = DummyText("Nightmare", mean_pos_x=500.0, mean_pos_y=100.0)
        selected_marker_on_same_row = DummyText(mean_pos_y=170.0)

        with patch.object(hydra_tools.window_tools, "click_center"):
            with patch.object(hydra_tools.window_tools, "move_up"):
                with patch.object(hydra_tools.window_tools, "move_down"):
                    with patch.object(hydra_tools.image_tools, "get_text_in_relative_area", return_value=[setup]):
                        with patch.object(
                            hydra_tools.image_tools,
                            "get_similarities_in_relative_area",
                            return_value=[selected_marker_on_same_row],
                        ):
                            with patch.object(hydra_tools.window_tools, "click_at") as click_at:
                                self.assertTrue(bot._select_hydra_setup("Nightmare"))

        click_at.assert_not_called()


class HydraDifficultySelectionTests(unittest.TestCase):
    def test_difficulty_click_is_repeated_with_pause(self):
        bot = hydra_tools.RSL_Bot_Hydra(reader=object(), window=object())

        with patch.object(hydra_tools.window_tools, "click_center") as click_center:
            self.assertTrue(bot._click_hydra_difficulty("Brutal"))

        self.assertEqual(click_center.call_count, 3)
        for call in click_center.call_args_list:
            self.assertEqual(call.args[0], bot.window)
            self.assertEqual(call.args[1], bot.search_areas["Hydra_Brutal"])
            self.assertEqual(call.kwargs["delay"], 2.0)

    def test_detect_cleared_difficulties_repeats_difficulty_click_before_name_scan(self):
        bot = hydra_tools.RSL_Bot_Hydra(
            reader=object(),
            window=object(),
            player_names=["TrollPatrol69"],
            difficulty_order=["Nightmare"],
        )

        with patch.object(hydra_tools.window_tools, "click_center") as click_center:
            with patch.object(hydra_tools.window_tools, "move_up"):
                with patch.object(hydra_tools.window_tools, "move_down"):
                    with patch.object(
                        hydra_tools.image_tools,
                        "get_text_in_relative_area",
                        return_value=[DummyText("TrollPatrol69")],
                    ) as read_text:
                        bot.detect_cleared_difficulties()

        self.assertEqual(click_center.call_count, 3)
        self.assertEqual(read_text.call_count, 1)
        self.assertEqual(bot.hydra_encounters_cleared, ["Nightmare"])


if __name__ == "__main__":
    unittest.main()
