import unittest
from unittest.mock import patch

from raid_bot.modes import (
    arena_tools,
    chimera_tools,
    cursedcity_tools,
    demonlord_tools,
    doomtower_tools,
    dungeon_tools,
    factionwars_tools,
    grimforest_tools,
    hydra_tools,
)


SHARED_RUNTIME_LIMIT_MODULES = (
    arena_tools,
    chimera_tools,
    cursedcity_tools,
    demonlord_tools,
    doomtower_tools,
    dungeon_tools,
    factionwars_tools,
    hydra_tools,
)


class DeadlineBot:
    max_run_duration_seconds = 7200.0
    _run_deadline = None


class BotWithoutDuration:
    _run_deadline = 10.0


class ModeRuntimeLimitTests(unittest.TestCase):
    def test_shared_deadline_initializers_preserve_default_and_explicit_duration_behavior(self):
        for module in SHARED_RUNTIME_LIMIT_MODULES:
            with self.subTest(module=module.__name__, duration="default"):
                bot = DeadlineBot()
                with patch.object(module.time, "time", return_value=100.0):
                    module._start_run_deadline(bot)
                self.assertEqual(bot._run_deadline, 7300.0)

            with self.subTest(module=module.__name__, duration="explicit"):
                bot = DeadlineBot()
                with patch.object(module.time, "time", return_value=100.0):
                    module._start_run_deadline(bot, max_run_duration_seconds="42")
                self.assertEqual(bot._run_deadline, 142.0)

    def test_shared_timeout_checks_preserve_fallback_and_error_message_behavior(self):
        for module in SHARED_RUNTIME_LIMIT_MODULES:
            with self.subTest(module=module.__name__, deadline="missing"):
                with patch.object(module.time, "time", return_value=100.0):
                    module._ensure_within_run_deadline(object(), "waiting")

            with self.subTest(module=module.__name__, duration="module fallback"):
                with patch.object(module.time, "time", return_value=100.0):
                    with self.assertRaisesRegex(
                        TimeoutError,
                        r"BotWithoutDuration exceeded max runtime of 3\.5h while waiting\.",
                    ):
                        module._ensure_within_run_deadline(BotWithoutDuration(), "waiting")

    def test_grim_forest_keeps_its_distinct_direct_attribute_contract(self):
        with patch.object(grimforest_tools.time, "time", return_value=100.0):
            with self.assertRaises(AttributeError):
                grimforest_tools._ensure_within_run_deadline(object(), "waiting")


if __name__ == "__main__":
    unittest.main()
