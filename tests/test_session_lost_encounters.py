import unittest
from pathlib import Path
from unittest.mock import patch

from raid_bot.modes import session_encounter_state as session_state


class SessionLostEncounterStateTests(unittest.TestCase):
    def setUp(self):
        session_state.reset_session_lost_encounters()

    def test_session_state_is_area_specific_and_matches_fuzzy_ocr(self):
        session_state.add_session_lost_encounter("cursed_city", "Frost Spider")
        session_state.add_session_lost_encounter("grim_forest", "Mimeto Dificil")

        self.assertTrue(session_state.is_session_lost_encounter("cursed_city", "Frost Sp1der"))
        self.assertTrue(session_state.is_session_lost_encounter("cursed_city", "Frost Spider - Elite"))
        self.assertFalse(session_state.is_session_lost_encounter("cursed_city", "Mimeto Dificil"))
        self.assertTrue(session_state.is_session_lost_encounter("grim_forest", "Mimeto Dificil"))
        self.assertFalse(session_state.is_session_lost_encounter("grim_forest", "Frost Spider"))

    def test_session_state_resets_without_persisting_to_disk(self):
        with patch.object(Path, "write_text", side_effect=AssertionError("session state must not write files")):
            with patch.object(Path, "replace", side_effect=AssertionError("session state must not replace files")):
                session_state.add_session_lost_encounter("cursed_city", "Frost Spider")
                session_state.add_session_lost_encounter("grim_forest", "Mimeto Dificil")
                session_state.reset_session_lost_encounters()

        self.assertEqual(session_state.get_session_lost_encounter_snapshot(), {"cursed_city": set(), "grim_forest": set()})

    def test_escape_guard_blocks_repeated_esc_presses_for_same_screen(self):
        session_state.add_session_lost_encounter("cursed_city", "Frost Spider")

        with patch.object(session_state.time, "monotonic", side_effect=[100.0, 100.5]):
            first = session_state.should_escape_session_lost_encounter(
                "cursed_city",
                "Frost Spider",
                cooldown_seconds=5.0,
            )
            second = session_state.should_escape_session_lost_encounter(
                "cursed_city",
                "Frost Spider",
                cooldown_seconds=5.0,
            )

        self.assertEqual(first, (True, "frost spider", "frost spider", True))
        self.assertEqual(second, (True, "frost spider", "frost spider", False))

    def test_daily_snapshot_round_trip_restores_avoid_lists(self):
        session_state.set_session_lost_encounter_snapshot(
            {
                "cursed_city": ["Frost Spider", "Borgoth"],
                "grim_forest": ["Mimeto Dificil"],
            }
        )

        snapshot = session_state.get_session_lost_encounter_snapshot()
        self.assertIn("frost spider", snapshot["cursed_city"])
        self.assertIn("borgoth", snapshot["cursed_city"])
        self.assertIn("mimeto dificil", snapshot["grim_forest"])


if __name__ == "__main__":
    unittest.main()
