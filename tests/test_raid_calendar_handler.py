from __future__ import annotations

import sys
import types
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import unittest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if "requests" not in sys.modules:
    requests_module = types.ModuleType("requests")
    requests_module.get = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_module
if "bs4" not in sys.modules:
    bs4_module = types.ModuleType("bs4")

    class _DummyBeautifulSoup:
        def __init__(self, *args, **kwargs):
            pass

        def select(self, *args, **kwargs):
            return []

    bs4_module.BeautifulSoup = _DummyBeautifulSoup
    sys.modules["bs4"] = bs4_module

from raid_bot.handlers import raid_calendar_handler


GESTAL_HTML = """
<!DOCTYPE html>
<html>
<body>
<script>
self.__next_f.push([1,"{\\\"id\\\":\\\"449\\\",\\\"name\\\":\\\"Spider Tournament\\\",\\\"kind\\\":\\\"tournament\\\",\\\"startsAt\\\":\\\"2026-07-01T09:00:00+00:00\\\",\\\"endsAt\\\":\\\"2026-07-04T09:00:00+00:00\\\"}"]);
self.__next_f.push([1,"{\\\"id\\\":\\\"454\\\",\\\"name\\\":\\\"Dragon Tournament\\\",\\\"kind\\\":\\\"tournament\\\",\\\"startsAt\\\":\\\"2026-07-05T09:00:00+00:00\\\",\\\"endsAt\\\":\\\"2026-07-08T09:00:00+00:00\\\"}"]);
self.__next_f.push([1,"{\\\"id\\\":\\\"463\\\",\\\"name\\\":\\\"Ice Golem Tournament\\\",\\\"kind\\\":\\\"tournament\\\",\\\"startsAt\\\":\\\"2026-07-12T00:00:00+00:00\\\",\\\"endsAt\\\":\\\"2026-07-15T00:00:00+00:00\\\"}"]);
self.__next_f.push([1,"{\\\"id\\\":\\\"457\\\",\\\"name\\\":\\\"Fire Knight Tournament\\\",\\\"kind\\\":\\\"tournament\\\",\\\"startsAt\\\":\\\"2026-07-09T09:00:00+00:00\\\",\\\"endsAt\\\":\\\"2026-07-12T09:00:00+00:00\\\"}"]);
</script>
</body>
</html>
"""


def _response(html: str):
    return SimpleNamespace(text=html, raise_for_status=lambda: None)


class RaidCalendarHandlerTests(unittest.TestCase):
    def test_gestal_page_detects_all_four_dungeon_tournaments(self):
        with patch.object(raid_calendar_handler.requests, "get", return_value=_response(GESTAL_HTML)):
            events = raid_calendar_handler._fetch_gestal_dungeon_tournament_events(today=date(2026, 7, 4))

        self.assertEqual([event.name for event in events], [
            "Spider Tournament",
            "Dragon Tournament",
            "Ice Golem Tournament",
            "Fire Knight Tournament",
        ])
        self.assertEqual([event.dungeon for event in events], [
            "spider",
            "dragon",
            "ice_golem",
            "fire_knight",
        ])
        self.assertTrue(all(event.source == "gestal" for event in events))

    def test_get_active_dungeon_tournament_prefers_gestal(self):
        with patch.object(raid_calendar_handler.requests, "get", return_value=_response(GESTAL_HTML)):
            active = raid_calendar_handler.get_active_dungeon_tournament(today=date(2026, 7, 4))

        self.assertIsNotNone(active)
        self.assertEqual(active.name, "Spider Tournament")
        self.assertEqual(active.dungeon, "spider")
        self.assertEqual(active.source, "gestal")

    def test_fastidious_backup_is_used_when_gestal_has_no_match(self):
        fake_event = raid_calendar_handler.DungeonTournamentEvent(
            name="Dragon Tournament",
            dungeon="dragon",
            start_date=date(2026, 7, 9),
            end_date=date(2026, 7, 12),
            source="fastidious",
        )

        with patch.object(
            raid_calendar_handler,
            "_fetch_gestal_dungeon_tournament_events",
            return_value=[],
        ), patch.object(
            raid_calendar_handler,
            "_fetch_fastidious_dungeon_tournament_events",
            return_value=[fake_event],
        ):
            active = raid_calendar_handler.get_active_dungeon_tournament(today=date(2026, 7, 10))

        self.assertIsNotNone(active)
        self.assertEqual(active.name, "Dragon Tournament")
        self.assertEqual(active.dungeon, "dragon")
        self.assertEqual(active.source, "fastidious")


if __name__ == "__main__":
    unittest.main()
