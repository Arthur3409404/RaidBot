"""Manual run player tools for profile-driven turn execution."""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Dict, List, Optional, Tuple

import pyautogui


class ManualRunError(RuntimeError):
    """Base manual run exception."""


class ManualRunConfigError(ManualRunError):
    """Raised for missing or invalid manual-run configuration."""


class ManualRunExecutionError(ManualRunError):
    """Raised when a configured manual action cannot be executed."""


class ManualRunPlayer:
    """Execute manual turns based on profile config from manual play database."""

    def __init__(
        self,
        profile_name: Optional[str],
        reader,
        window,
        image_tools,
        window_tools,
        search_areas: Optional[Dict[str, List[float]]] = None,
        database_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.reader = reader
        self.window = window
        self.image_tools = image_tools
        self.window_tools = window_tools
        self.search_areas = dict(search_areas or {})
        self.log = logger or logging.getLogger(self.__class__.__name__)

        self.database_path = database_path or self._default_database_path()
        self.database = self._load_database(self.database_path)
        self.defaults = dict(self.database.get("defaults", {}))
        self.profile_name = None
        self.profile = {}
        self.champions = {}
        self.turn_order: List[str] = []
        self.action_buttons: Dict[str, List[float]] = {}
        self.targets: Dict[str, List[float]] = {}
        self.auto_button_area: Optional[List[float]] = None
        self.priority_cycle_counters: Dict[str, int] = {}
        self.pixel_tolerance = int(self.defaults.get("pixel_tolerance", 2))
        self.identify_neighbor_rings = int(self.defaults.get("identify_neighbor_rings", 2))
        self.click_delay_seconds = float(self.defaults.get("click_delay_seconds", 0.35))
        self.manual_difficulties: List[str] = list(self.defaults.get("manual_difficulties", []))

        if not profile_name:
            raise ManualRunConfigError("Missing profile name for ManualRunPlayer initialization.")
        self.load_profile(profile_name)

    def _default_database_path(self) -> str:
        return os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "database_champions",
                "manual_play_database.py",
            )
        )

    def _load_database(self, database_path: str) -> dict:
        if not os.path.exists(database_path):
            raise ManualRunConfigError(f"Missing manual play database: '{database_path}'.")

        spec = importlib.util.spec_from_file_location("manual_play_database", database_path)
        if not spec or not spec.loader:
            raise ManualRunConfigError(
                f"Failed to create import spec for manual play database: '{database_path}'."
            )

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manual_db = getattr(module, "MANUAL_PLAY_DATABASE", None)
        if not isinstance(manual_db, dict):
            raise ManualRunConfigError(
                "Manual play database must define a MANUAL_PLAY_DATABASE dictionary."
            )
        return manual_db

    def _normalize_name(self, value: Optional[str]) -> str:
        return "".join(ch for ch in (value or "").lower() if ch.isalnum())

    def should_run_for_difficulty(self, difficulty_name: Optional[str]) -> bool:
        current = self._normalize_name(difficulty_name)
        if not current:
            return False
        return current in {self._normalize_name(item) for item in self.manual_difficulties}

    def load_profile(self, profile_name: str):
        if not profile_name:
            raise ManualRunConfigError("Missing profile name when loading manual profile.")

        profiles = self.database.get("profiles", {})
        if profile_name not in profiles:
            known_profiles = ", ".join(sorted(profiles.keys())) or "<none>"
            raise ManualRunConfigError(
                f"Unknown manual profile '{profile_name}'. Known profiles: {known_profiles}."
            )

        self.profile_name = profile_name
        self.profile = dict(profiles[profile_name] or {})
        self.champions = dict(self.profile.get("champions", {}))
        if not self.champions:
            raise ManualRunConfigError(
                f"Manual profile '{profile_name}' has no champion definitions."
            )

        self.turn_order = list(self.profile.get("turn_order", []))
        if not self.turn_order:
            self.turn_order = list(self.champions.keys())
        if not self.turn_order:
            raise ManualRunConfigError(
                f"Manual profile '{profile_name}' has no turn order/champion list."
            )

        self.manual_difficulties = list(
            self.profile.get("manual_difficulties", self.defaults.get("manual_difficulties", []))
        )
        self.pixel_tolerance = int(
            self.profile.get("pixel_tolerance", self.defaults.get("pixel_tolerance", 2))
        )
        self.identify_neighbor_rings = int(
            self.profile.get(
                "identify_neighbor_rings",
                self.defaults.get("identify_neighbor_rings", 2),
            )
        )
        self.click_delay_seconds = float(
            self.profile.get("click_delay_seconds", self.defaults.get("click_delay_seconds", 0.35))
        )

        self.action_buttons = dict(self.defaults.get("action_buttons", {}))
        self.action_buttons.update(self.profile.get("action_buttons", {}))

        self.targets = dict(self.defaults.get("targets", {}))
        self.targets.update(self.profile.get("targets", {}))

        self.auto_button_area = self.profile.get("auto_battle_button") or self.search_areas.get(
            "auto_battle_button"
        ) or self.defaults.get("auto_battle_button")

        self._validate_profile()
        self._reset_priority_cycle_counters()
        self.log.info(
            "Manual profile loaded: '%s' | champions=%s | manual_difficulties=%s",
            self.profile_name,
            len(self.champions),
            self.manual_difficulties,
        )

    def _validate_area(self, area_name: str, area_value):
        if (
            not isinstance(area_value, (list, tuple))
            or len(area_value) != 4
            or not all(isinstance(item, (int, float)) for item in area_value)
        ):
            raise ManualRunConfigError(
                f"Missing or invalid pixel location for '{area_name}' in profile '{self.profile_name}'."
            )

    def _validate_priority_entries(
        self,
        champion_name: str,
        targets: Dict[str, str],
        priority: list,
        priority_label: str,
    ):
        if not isinstance(priority, list) or not priority:
            raise ManualRunConfigError(
                f"Missing champion/action {priority_label} for champion '{champion_name}'."
            )

        for action in priority:
            if action not in self.action_buttons:
                raise ManualRunConfigError(
                    f"Missing pixel location for action '{action}' in champion '{champion_name}'."
                )
            target_name = targets.get(action, "Auto")
            if target_name != "Auto" and target_name not in self.targets:
                raise ManualRunConfigError(
                    f"Missing pixel location for target '{target_name}' in champion '{champion_name}'."
                )

    def _reset_priority_cycle_counters(self):
        self.priority_cycle_counters = {}
        for champion_name, champion_cfg in self.champions.items():
            switch_cfg = champion_cfg.get("priority_switch_once")
            if isinstance(switch_cfg, dict):
                self.priority_cycle_counters[champion_name] = 0

    def _resolve_priority_for_turn(
        self,
        champion_name: str,
        champion_cfg: dict,
    ) -> Tuple[List[str], bool]:
        priority = champion_cfg.get("priority")
        if not isinstance(priority, list) or not priority:
            raise ManualRunConfigError(
                f"Missing champion/action priority for champion '{champion_name}'."
            )

        switch_cfg = champion_cfg.get("priority_switch_once")
        if not isinstance(switch_cfg, dict):
            return list(priority), False

        after_base_turns = int(switch_cfg.get("after_base_turns", 0))
        base_turns_used = int(self.priority_cycle_counters.get(champion_name, 0))
        if base_turns_used >= after_base_turns:
            swapped_priority = switch_cfg.get("swapped_priority")
            if not isinstance(swapped_priority, list) or not swapped_priority:
                raise ManualRunConfigError(
                    f"Champion '{champion_name}' priority_switch_once.swapped_priority is missing or empty."
                )
            self.log.info(
                "Manual profile '%s': champion '%s' using swapped priority once "
                "(base_turns_used=%s threshold=%s).",
                self.profile_name,
                champion_name,
                base_turns_used,
                after_base_turns,
            )
            return list(swapped_priority), True

        return list(priority), False

    def _update_priority_cycle_state(self, champion_name: str, used_swapped_priority: bool):
        champion_cfg = self.champions.get(champion_name, {})
        switch_cfg = champion_cfg.get("priority_switch_once")
        if not isinstance(switch_cfg, dict):
            return

        current_count = int(self.priority_cycle_counters.get(champion_name, 0))
        if used_swapped_priority:
            self.priority_cycle_counters[champion_name] = 0
            self.log.info(
                "Manual profile '%s': champion '%s' priority cycle reset after swapped turn.",
                self.profile_name,
                champion_name,
            )
            return

        self.priority_cycle_counters[champion_name] = current_count + 1

    def _validate_profile(self):
        if self.auto_button_area is None:
            raise ManualRunConfigError(
                f"Missing pixel location for auto battle button in profile '{self.profile_name}'."
            )
        self._validate_area("auto_battle_button", self.auto_button_area)

        for action_name in ("A1", "A2", "A3"):
            if action_name not in self.action_buttons:
                raise ManualRunConfigError(
                    f"Missing pixel location for action '{action_name}' in profile '{self.profile_name}'."
                )
            self._validate_area(f"action_button.{action_name}", self.action_buttons[action_name])

        for champion_name, champion_cfg in self.champions.items():
            if not isinstance(champion_cfg, dict):
                raise ManualRunConfigError(
                    f"Champion '{champion_name}' is missing from profile '{self.profile_name}'."
                )

            identify_pixel = champion_cfg.get("identify_pixel")
            if not isinstance(identify_pixel, dict):
                raise ManualRunConfigError(
                    f"Missing pixel location for champion '{champion_name}' identify_pixel."
                )

            for key in ("x", "y", "rgb"):
                if key not in identify_pixel:
                    raise ManualRunConfigError(
                        f"Champion '{champion_name}' identify_pixel is missing '{key}'."
                    )

            targets = champion_cfg.get("targets", {})
            if not isinstance(targets, dict):
                raise ManualRunConfigError(
                    f"Invalid targets mapping for champion '{champion_name}'."
                )
            priority = champion_cfg.get("priority")
            self._validate_priority_entries(
                champion_name=champion_name,
                targets=targets,
                priority=priority,
                priority_label="priority",
            )

            switch_cfg = champion_cfg.get("priority_switch_once")
            if switch_cfg is None:
                continue
            if not isinstance(switch_cfg, dict):
                raise ManualRunConfigError(
                    f"Champion '{champion_name}' priority_switch_once must be a dictionary."
                )

            after_base_turns = switch_cfg.get("after_base_turns")
            if (
                isinstance(after_base_turns, bool)
                or not isinstance(after_base_turns, int)
                or after_base_turns < 1
            ):
                raise ManualRunConfigError(
                    f"Champion '{champion_name}' priority_switch_once.after_base_turns must be an integer >= 1."
                )

            swapped_priority = switch_cfg.get("swapped_priority")
            self._validate_priority_entries(
                champion_name=champion_name,
                targets=targets,
                priority=swapped_priority,
                priority_label="priority_switch_once.swapped_priority",
            )

    def _resolve_absolute_pixel(self, rel_x: float, rel_y: float) -> Tuple[int, int]:
        if self.window is None:
            raise ManualRunExecutionError("Cannot sample pixel without an active game window.")

        abs_x = int(self.window.left + float(rel_x) * self.window.width)
        abs_y = int(self.window.top + float(rel_y) * self.window.height)
        return abs_x, abs_y

    def _sample_pixel(self, rel_x: float, rel_y: float) -> Tuple[int, int, int]:
        abs_x, abs_y = self._resolve_absolute_pixel(rel_x, rel_y)
        pixel = pyautogui.screenshot(region=(abs_x, abs_y, 1, 1)).getpixel((0, 0))
        if isinstance(pixel, tuple) and len(pixel) == 4:
            pixel = pixel[:3]
        return int(pixel[0]), int(pixel[1]), int(pixel[2])

    def _sample_identify_neighborhood(
        self,
        rel_x: float,
        rel_y: float,
        rings: int = 2,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int, int]]]:
        abs_x, abs_y = self._resolve_absolute_pixel(rel_x, rel_y)
        ring_depth = max(0, int(rings))

        left_bound = int(self.window.left)
        top_bound = int(self.window.top)
        right_bound = int(self.window.left + self.window.width - 1)
        bottom_bound = int(self.window.top + self.window.height - 1)

        region_left = max(left_bound, abs_x - ring_depth)
        region_top = max(top_bound, abs_y - ring_depth)
        region_right = min(right_bound, abs_x + ring_depth)
        region_bottom = min(bottom_bound, abs_y + ring_depth)

        region_width = max(1, region_right - region_left + 1)
        region_height = max(1, region_bottom - region_top + 1)

        screenshot = pyautogui.screenshot(
            region=(region_left, region_top, region_width, region_height)
        )

        samples: List[Tuple[Tuple[int, int], Tuple[int, int, int]]] = []
        for ring in range(0, ring_depth + 1):
            for dy in range(-ring, ring + 1):
                for dx in range(-ring, ring + 1):
                    if ring > 0 and max(abs(dx), abs(dy)) != ring:
                        continue
                    target_x = abs_x + dx
                    target_y = abs_y + dy
                    if (
                        target_x < left_bound
                        or target_x > right_bound
                        or target_y < top_bound
                        or target_y > bottom_bound
                    ):
                        continue
                    local_x = target_x - region_left
                    local_y = target_y - region_top
                    pixel = screenshot.getpixel((local_x, local_y))
                    if isinstance(pixel, tuple) and len(pixel) == 4:
                        pixel = pixel[:3]
                    samples.append(
                        (
                            (dx, dy),
                            (int(pixel[0]), int(pixel[1]), int(pixel[2])),
                        )
                    )
        return samples

    def _pixel_distance(self, rgb_left: Tuple[int, int, int], rgb_right: Tuple[int, int, int]) -> int:
        return max(abs(rgb_left[0] - rgb_right[0]), abs(rgb_left[1] - rgb_right[1]), abs(rgb_left[2] - rgb_right[2]))

    def _identify_champion(self) -> Optional[str]:
        matched_champion = None
        matched_distance = None

        for champion_name, champion_cfg in self.champions.items():
            identify = champion_cfg["identify_pixel"]
            expected_rgb = tuple(int(value) for value in identify["rgb"])
            neighborhood = self._sample_identify_neighborhood(
                float(identify["x"]),
                float(identify["y"]),
                rings=1,
            )
            if not neighborhood:
                sampled_rgb = self._sample_pixel(float(identify["x"]), float(identify["y"]))
                best_offset = (0, 0)
                distance = self._pixel_distance(expected_rgb, sampled_rgb)
            else:
                best_offset, sampled_rgb = min(
                    neighborhood,
                    key=lambda sample: self._pixel_distance(expected_rgb, sample[1]),
                )
                distance = self._pixel_distance(expected_rgb, sampled_rgb)

            if distance <= self.pixel_tolerance:
                self.log.info(
                    "Manual profile '%s': identified champion '%s' near pixel (x=%.3f, y=%.3f) "
                    "offset=(%s,%s) expected=%s sampled=%s delta=%s rings=%s.",
                    self.profile_name,
                    champion_name,
                    float(identify["x"]),
                    float(identify["y"]),
                    int(best_offset[0]),
                    int(best_offset[1]),
                    expected_rgb,
                    sampled_rgb,
                    distance,
                    1,
                )
                return champion_name

            if matched_distance is None or distance < matched_distance:
                matched_champion = champion_name
                matched_distance = distance

        self.log.warning(
            "Manual profile '%s': no champion matched pixel tolerance=%s. "
            "Closest='%s' delta=%s. Falling back to auto.",
            self.profile_name,
            self.pixel_tolerance,
            matched_champion,
            matched_distance,
        )
        return None

    def _fallback_to_auto(self) -> Dict[str, str]:
        try:
            self.window_tools.click_center(
                self.window, self.auto_button_area, delay=self.click_delay_seconds
            )
            self.window_tools.click_center(
                self.window, self.auto_button_area, delay=self.click_delay_seconds
            )
            self.log.info(
                "Manual profile '%s': no champion match, auto fallback executed.",
                self.profile_name,
            )
            return {
                "profile": self.profile_name,
                "champion": "Unknown",
                "action": "AUTO_FALLBACK",
                "target": "Auto",
            }
        except Exception as exc:
            raise ManualRunExecutionError(
                f"Failed auto fallback execution: {exc}"
            ) from exc

    def _resolve_action(self, champion_name: str) -> Tuple[str, str, bool]:
        champion_cfg = self.champions.get(champion_name)
        if not champion_cfg:
            raise ManualRunConfigError(
                f"Champion '{champion_name}' does not exist in profile '{self.profile_name}'."
            )

        priority, used_swapped_priority = self._resolve_priority_for_turn(
            champion_name=champion_name,
            champion_cfg=champion_cfg,
        )
        targets = champion_cfg.get("targets", {})
        if not isinstance(targets, dict):
            raise ManualRunConfigError(
                f"Invalid target mapping for champion '{champion_name}'."
            )

        for action_name in priority:
            if action_name not in self.action_buttons:
                continue
            target_name = targets.get(action_name, "Auto")
            if target_name == "Auto" or target_name in self.targets:
                return action_name, target_name, used_swapped_priority

        raise ManualRunConfigError(
            f"Champion '{champion_name}' has no executable action from priority list: {priority}."
        )

    def _execute_action(self, action_name: str, target_name: str):
        action_area = self.action_buttons.get(action_name)
        if action_area is None:
            raise ManualRunConfigError(
                f"Missing pixel location for action '{action_name}' in profile '{self.profile_name}'."
            )
        self._validate_area(f"action_button.{action_name}", action_area)

        try:
            self.log.info(
                "Manual profile '%s': executing action '%s' with target '%s'.",
                self.profile_name,
                action_name,
                target_name,
            )
            self.window_tools.click_center(
                self.window, action_area, delay=self.click_delay_seconds
            )

            if target_name == "Auto":
                self.window_tools.click_center(
                    self.window, self.auto_button_area, delay=self.click_delay_seconds
                )
                self.window_tools.click_center(
                    self.window, self.auto_button_area, delay=self.click_delay_seconds
                )
                self.log.info(
                    "Manual profile '%s': action '%s' resolved via auto target.",
                    self.profile_name,
                    action_name,
                )
                return

            target_area = self.targets.get(target_name)
            if target_area is None:
                raise ManualRunConfigError(
                    f"Missing pixel location for target '{target_name}' in profile '{self.profile_name}'."
                )
            self._validate_area(f"target.{target_name}", target_area)
            self.window_tools.click_center(
                self.window, target_area, delay=self.click_delay_seconds
            )
            self.log.info(
                "Manual profile '%s': action '%s' executed on target '%s'.",
                self.profile_name,
                action_name,
                target_name,
            )
        except Exception as exc:
            raise ManualRunExecutionError(
                f"Failed manual action execution for action '{action_name}' target '{target_name}': {exc}"
            ) from exc

    def take_turn(self) -> Dict[str, str]:
        if not self.profile_name:
            raise ManualRunConfigError("No manual profile loaded.")

        champion_name = self._identify_champion()
        if champion_name is None:
            return self._fallback_to_auto()

        action_name, target_name, used_swapped_priority = self._resolve_action(champion_name)
        self.log.info(
            "Manual profile '%s': selected champion='%s', action='%s', target='%s'.",
            self.profile_name,
            champion_name,
            action_name,
            target_name,
        )
        self._execute_action(action_name, target_name)
        self._update_priority_cycle_state(champion_name, used_swapped_priority)
        return {
            "profile": self.profile_name,
            "champion": champion_name,
            "action": action_name,
            "target": target_name,
        }

    def take_action(self) -> Dict[str, str]:
        """Backward-compatible alias."""
        return self.take_turn()
