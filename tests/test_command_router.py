import unittest

from raid_bot.core import BotCommandRouter
from raid_bot.utils.file_tools import ParameterUpdate


class FakeBotRuntime:
    def __init__(self):
        self.reload_calls = 0
        self.params_filters = []
        self.set_calls = []
        self.toggle_calls = []

    def build_help_lines(self):
        return ["help output"]

    def build_status_lines(self):
        return ["status output"]

    def build_modes_lines(self):
        return ["modes output"]

    def build_params_lines(self, search_term=None):
        self.params_filters.append(search_term)
        return [f"params output: {search_term}"]

    def get_parameter_value(self, requested_key):
        if requested_key == "unknown":
            raise KeyError(requested_key)
        return "run_hydra", True

    def set_parameter_value(self, requested_key, raw_value):
        self.set_calls.append((requested_key, raw_value))
        return ParameterUpdate(
            key="run_hydra",
            old_value=True,
            new_value=False,
            changed=True,
            persisted=True,
        )

    def toggle_mode(self, requested_mode, desired_state=None):
        self.toggle_calls.append((requested_mode, desired_state))
        return ParameterUpdate(
            key="run_hydra",
            old_value=True,
            new_value=False,
            changed=True,
            persisted=True,
        )

    def reload_configuration(self):
        self.reload_calls += 1


class CommandRouterTests(unittest.TestCase):
    def setUp(self):
        self.bot = FakeBotRuntime()
        self.router = BotCommandRouter(self.bot)

    def test_empty_and_unknown_commands_preserve_router_responses(self):
        self.assertEqual(self.router.route(None).messages, [])
        self.assertEqual(self.router.route("   ").messages, [])
        self.assertEqual(
            self.router.route("missing").messages,
            [
                "[Bot Command] Unknown command: `missing`",
                "Use `help` to see available commands.",
            ],
        )

    def test_status_and_parameter_commands_delegate_without_changing_messages(self):
        self.assertEqual(self.router.route("status").messages, ["status output"])
        self.assertEqual(
            self.router.route("params dungeons hard").messages,
            ["params output: dungeons hard"],
        )
        self.assertEqual(self.bot.params_filters, ["dungeons hard"])
        self.assertEqual(
            self.router.route("get hydra").messages,
            ["[Bot Config] `run_hydra` = `True`"],
        )
        self.assertEqual(
            self.router.route("get unknown").messages,
            ["[Bot Config] Unknown parameter: `unknown`"],
        )

    def test_set_and_toggle_preserve_forwarded_values_and_response_text(self):
        self.assertEqual(
            self.router.route("set run.hydra false").messages,
            ["[Bot Config] `run_hydra` updated from `True` to `False` (saved)."],
        )
        self.assertEqual(self.bot.set_calls, [("run.hydra", "false")])

        self.assertEqual(
            self.router.route("toggle hydra off").messages,
            ["[Bot Mode] `run_hydra` set to `DISABLED` (saved)."],
        )
        self.assertEqual(self.bot.toggle_calls, [("hydra", "off")])

    def test_control_aliases_and_restart_flags_preserve_runtime_intent(self):
        pause = self.router.route("pause")
        resume = self.router.route("resume")
        restart = self.router.route("restart")

        self.assertTrue(pause.enter_manual_mode)
        self.assertEqual(
            pause.messages,
            ["[Bot Status] Manual mode enabled. Automation paused."],
        )
        self.assertTrue(resume.exit_manual_mode)
        self.assertEqual(
            resume.messages,
            ["[Bot Status] Manual mode disabled. Resuming automation."],
        )
        self.assertTrue(restart.restart_requested)
        self.assertEqual(
            restart.messages,
            ["[Bot Status] Restart command received. Restarting bot process and Raid application."],
        )

    def test_reload_alias_calls_runtime_reload(self):
        result = self.router.route("reload_config")

        self.assertEqual(
            result.messages,
            ["[Bot Config] Configuration reloaded from disk."],
        )
        self.assertEqual(self.bot.reload_calls, 1)


if __name__ == "__main__":
    unittest.main()
