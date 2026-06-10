from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from raid_bot.core import daily_tasks


class DailyTaskConfigTests(unittest.TestCase):
    def test_parse_dungeon_daily_task_accepts_profile_assignment_syntax(self):
        tasks = daily_tasks.parse_dungeon_daily_tasks(
            "[dungeon= shogun, level = 25, energy = 60]"
        )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].dungeon, "shogun")
        self.assertEqual(tasks[0].level, 25)
        self.assertEqual(tasks[0].energy, 60)
        self.assertEqual(tasks[0].difficulty, None)
        self.assertEqual(tasks[0].signature, "shogun:highest:25:60")

    def test_parse_dungeon_daily_task_accepts_literal_dicts(self):
        tasks = daily_tasks.parse_dungeon_daily_tasks(
            [{"dungeon": "dragon", "difficulty": "hard", "level": 10, "energy": 80}]
        )

        self.assertEqual(
            tasks,
            [
                daily_tasks.DungeonDailyTask(
                    dungeon="dragon",
                    difficulty="hard",
                    level=10,
                    energy=80,
                )
            ],
        )

    def test_daily_state_resets_by_utc_date(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artus.json"
            daily_tasks.save_daily_task_state(
                path,
                {"utc_date": "2026-06-04", "dungeons": {"old": {"completed": True}}},
            )

            self.assertEqual(
                daily_tasks.load_daily_task_state(path, "2026-06-05"),
                {"utc_date": "2026-06-05", "dungeons": {}},
            )

    def test_utc_date_string_normalizes_aware_datetimes(self):
        self.assertEqual(
            daily_tasks.utc_date_string(datetime(2026, 6, 5, 0, 30, tzinfo=timezone.utc)),
            "2026-06-05",
        )


if __name__ == "__main__":
    unittest.main()
