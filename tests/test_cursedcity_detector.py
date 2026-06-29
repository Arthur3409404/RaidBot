import unittest
from unittest.mock import patch

import numpy as np

from raid_bot.detector_ai.yolo_detector import Detection
from raid_bot.modes import cursedcity_tools
from raid_bot.modes import session_encounter_state


class DummyDetector:
    def __init__(self, detections):
        self.detections = detections
        self.calls = []

    def predict(self, image, *, conf=0.25, imgsz=640):
        self.calls.append({"image": image, "conf": conf, "imgsz": imgsz})
        if self.detections is None:
            return None
        return list(self.detections)


class CursedCityDetectorTests(unittest.TestCase):
    def test_yolo_confidence_is_the_only_score_gate(self):
        detector = DummyDetector(
            [
                Detection(x1=10.2, y1=20.2, x2=50.6, y2=80.6, confidence=0.33, class_id=0),
                Detection(x1=100.0, y1=120.0, x2=140.0, y2=150.0, confidence=0.78, class_id=0),
            ]
        )
        mask = np.full((200, 200), 255, dtype=np.uint8)

        with patch.object(cursedcity_tools.Path, "exists", return_value=True):
            with patch.object(cursedcity_tools, "_load_yolo_detector", return_value=detector):
                boxes = cursedcity_tools.detect_cursedcity_like_structures(mask)

        self.assertEqual([box.score for box in boxes], [0.78, 0.33])
        self.assertEqual((boxes[1].x, boxes[1].y, boxes[1].width, boxes[1].height), (10, 20, 41, 61))
        self.assertEqual(detector.calls[0]["conf"], cursedcity_tools.DEFAULT_DETECTOR_CONFIDENCE)
        self.assertEqual(detector.calls[0]["imgsz"], cursedcity_tools.DEFAULT_DETECTOR_IMGSZ)

    def test_empty_yolo_result_returns_no_boxes(self):
        detector = DummyDetector([])
        mask = np.full((100, 100), 255, dtype=np.uint8)

        with patch.object(cursedcity_tools.Path, "exists", return_value=True):
            with patch.object(cursedcity_tools, "_load_yolo_detector", return_value=detector):
                boxes = cursedcity_tools.detect_cursedcity_like_structures(mask)

        self.assertEqual(boxes, [])

    def test_none_yolo_result_returns_no_boxes(self):
        detector = DummyDetector(None)
        mask = np.full((100, 100), 255, dtype=np.uint8)

        with patch.object(cursedcity_tools.Path, "exists", return_value=True):
            with patch.object(cursedcity_tools, "_load_yolo_detector", return_value=detector):
                boxes = cursedcity_tools.detect_cursedcity_like_structures(mask)

        self.assertEqual(boxes, [])

    def test_session_lost_encounter_skips_cursed_city_candidate_once(self):
        session_encounter_state.reset_session_lost_encounters()
        session_encounter_state.add_session_lost_encounter("cursed_city", "Frost Spider")
        bot = cursedcity_tools.RSL_Bot_CursedCity(window=object())
        candidate = {"center_abs_x": 100, "center_abs_y": 120}

        with patch.object(cursedcity_tools.window_tools, "click_at"):
            with patch.object(cursedcity_tools.window_tools, "sendkey") as sendkey:
                with patch.object(bot, "_read_visible_encounter_text", return_value="Frost Sp1der"):
                    self.assertFalse(bot.select_cursed_city_candidate(candidate))

        sendkey.assert_called_once_with("esc", delay=1.0, window=bot.window)

    def test_visible_encounter_can_be_added_to_cursed_city_avoid_list(self):
        session_encounter_state.reset_session_lost_encounters()
        bot = cursedcity_tools.RSL_Bot_CursedCity(window=object())

        with patch.object(bot, "_read_visible_encounter_text", return_value="Frost Spider"):
            encounter = bot.add_visible_encounter_to_avoid_list()

        self.assertEqual(encounter, "Frost Spider")
        self.assertTrue(
            session_encounter_state.is_session_lost_encounter("cursed_city", "Frost Spider")
        )


if __name__ == "__main__":
    unittest.main()
