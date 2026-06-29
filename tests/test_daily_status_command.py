from __future__ import annotations

import tempfile
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path

from raid_bot import mainframe
from raid_bot.utils import file_tools


class DailyStatusCommandTests(unittest.TestCase):
    def test_mainframe_status_command_returns_current_daily_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "raidbot_daily_report.json"
            file_tools.ensure_daily_log_header(log_path, ["# header"])
            file_tools.update_daily_log_document(
                log_path,
                lambda day: day.update(
                    {
                        "summary": {"market": {"mystery_shards_bought": 2}},
                        "state": {"enemy_avoid": {"cursed_city": ["frost spider"]}},
                    }
                ),
            )

            bot = mainframe.RSL_Bot_Mainframe.__new__(mainframe.RSL_Bot_Mainframe)
            bot.daily_log_path = log_path
            bot.log = types.SimpleNamespace(warning=lambda *args, **kwargs: None)

            messages = bot.build_status_lines()

        day_key = datetime.now(timezone.utc).strftime("%Y_%m_%d")
        self.assertEqual(messages[0], f"[Bot Status] Current daily entry for `{day_key}`")
        combined = "\n".join(messages)
        self.assertIn('"mystery_shards_bought": 2', combined)
        self.assertIn('"cursed_city": [', combined)
        self.assertIn('"frost spider"', combined)


if __name__ == "__main__":
    unittest.main()
