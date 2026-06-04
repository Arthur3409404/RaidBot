from __future__ import annotations

import logging
import os
import sys
import traceback

from data.lib.modes.hydra_tools import HYDRA_SEARCH_AREAS, RSL_Bot_Hydra, resembles

PARAM_FILE = os.path.join("data", "params_mainframe.txt")
RAID_WINDOW_TITLE = "Raid: Shadow Legends"

MAIN_MENU_NAMES = {
    "ClanBoss1": "Jefes",
}


class HydraStandaloneRunner:
    """Standalone wrapper that reuses integrated RSL_Bot_Hydra module."""

    def __init__(
        self,
        title_substring: str = RAID_WINDOW_TITLE,
        param_file: str = PARAM_FILE,
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.main_loop_running = True
        self.search_areas = dict(HYDRA_SEARCH_AREAS)

        import pyautogui
        from data.lib.utils import file_tools, image_tools, window_tools

        self.pyautogui = pyautogui
        self.file_tools = file_tools
        self.image_tools = image_tools
        self.window_tools = window_tools

        self.params = self._load_grouped_params(param_file)
        self.reader = self._build_reader()
        self.window = self._resolve_window(title_substring)
        self.hydra_mode = self._build_hydra_mode()

    def _load_grouped_params(self, param_file: str):
        if not os.path.exists(param_file):
            raise RuntimeError(f"Required params file is missing: '{param_file}'.")

        grouped = self.file_tools.ParameterStore(param_file).get_grouped_copy()
        hydra_cfg = dict(grouped.get("hydra", {}))
        missing_required = [
            key
            for key in ("player_names", "difficulty_order", "thresholds")
            if not hydra_cfg.get(key)
        ]
        if missing_required:
            raise RuntimeError(
                "Missing required Hydra parameters in params file: " + ", ".join(missing_required)
            )
        return grouped

    def _build_reader(self):
        import easyocr

        self.log.info("Initializing OCR reader (easyocr, lang='en').")
        return easyocr.Reader(["en"])

    def _resolve_window(self, title_substring: str):
        detected = self.window_tools.find_window(title_substring)
        if not detected:
            raise RuntimeError(f"Raid window not found. Expected title containing: '{title_substring}'.")

        return self.window_tools.WindowObject(detected, title_substring=title_substring)

    def _build_hydra_mode(self):
        hydra_cfg = dict(self.params.get("hydra", {}))
        return RSL_Bot_Hydra(
            reader=self.reader,
            window=self.window,
            **hydra_cfg,
        )

    def navigate_to_menu(self, menu_name: str, max_attempts: int = 5):
        for _ in range(max_attempts):
            if not self.main_loop_running:
                break
            try:
                texts = self.image_tools.get_text_in_relative_area(
                    self.reader,
                    self.window,
                    self.search_areas["menu_name"],
                    power_detection=False,
                )
                if texts and "Modos" in texts[0].text:
                    break
                self.window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])
            except Exception:
                self.window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])

        def scan_visible_mode_label():
            labels = self.image_tools.get_text_in_relative_area(
                self.reader,
                self.window,
                self.search_areas["main_menu_labels"],
                power_detection=False,
            )
            for label in labels:
                if resembles(label.text, menu_name):
                    return label
            return None

        def click_mode_label(label):
            if not label:
                return False
            self.window_tools.click_at(label.mean_pos_x, label.mean_pos_y, window=self.window)
            return True

        visible_label = scan_visible_mode_label()
        if click_mode_label(visible_label):
            return

        for direction in ("left", "right"):
            move = self.window_tools.move_left if direction == "left" else self.window_tools.move_right
            try:
                self.pyautogui.mouseUp()
            except Exception:
                pass
            move(self.window, strength=1.2, relative_x=0.5, relative_y=0.72)
            try:
                self.pyautogui.mouseUp()
            except Exception:
                pass

            visible_label = scan_visible_mode_label()
            if click_mode_label(visible_label):
                return

        self.log.warning("Menu '%s' not found.", menu_name)

    def run(self):
        self.main_loop_running = True
        self.navigate_to_menu(MAIN_MENU_NAMES["ClanBoss1"])
        self.window_tools.click_center(self.window, self.search_areas["clanboss_Hydra"])
        self.hydra_mode.run_hydra(main_loop_running=self.main_loop_running)
        self.window_tools.click_center(self.window, self.search_areas["go_to_higher_menu"])


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        HydraStandaloneRunner().run()
        return 0
    except Exception as exc:
        logging.error("Hydra standalone runner failed: %s", exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
