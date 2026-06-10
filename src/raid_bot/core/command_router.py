"""Extensible command routing for Discord/manual bot control."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import shlex
from typing import Callable


CommandHandler = Callable[[list[str], str], "CommandResult"]


@dataclass
class CommandResult:
    messages: list[str] = field(default_factory=list)
    enter_manual_mode: bool = False
    exit_manual_mode: bool = False
    restart_requested: bool = False


class BotCommandRouter:
    """Command parser/router with pluggable handlers."""

    def __init__(self, bot_runtime):
        self.bot = bot_runtime
        self._handlers: dict[str, CommandHandler] = {}
        self._register_default_commands()

    def register(
        self,
        name: str,
        handler: CommandHandler,
        aliases: tuple[str, ...] = (),
    ) -> None:
        canonical = name.strip().lower()
        self._handlers[canonical] = handler
        for alias in aliases:
            self._handlers[alias.strip().lower()] = handler

    def route(self, raw_command: str | None) -> CommandResult:
        if not raw_command:
            return CommandResult()

        try:
            parts = shlex.split(raw_command.strip(), posix=False)
        except ValueError:
            parts = raw_command.strip().split()

        if not parts:
            return CommandResult()

        command = parts[0].lower()
        args = parts[1:]
        handler = self._handlers.get(command)

        if not handler:
            return CommandResult(
                messages=[
                    f"[Bot Command] Unknown command: `{command}`",
                    "Use `help` to see available commands.",
                ]
            )

        return handler(args, raw_command)

    def _register_default_commands(self) -> None:
        self.register("help", self._cmd_help, aliases=("commands", "?"))
        self.register("status", self._cmd_status)
        self.register("modes", self._cmd_modes)
        self.register("params", self._cmd_params)
        self.register("get", self._cmd_get)
        self.register("set", self._cmd_set)
        self.register("toggle", self._cmd_toggle, aliases=("mode",))
        self.register("reload", self._cmd_reload, aliases=("reload_config",))
        self.register("start", self._cmd_start, aliases=("resume",))
        self.register("stop", self._cmd_stop, aliases=("pause",))
        self.register("restart", self._cmd_restart)
        self.register("ping", self._cmd_ping)

    def _cmd_help(self, args: list[str], raw: str) -> CommandResult:
        return CommandResult(messages=self.bot.build_help_lines())

    def _cmd_status(self, args: list[str], raw: str) -> CommandResult:
        return CommandResult(messages=self.bot.build_status_lines())

    def _cmd_modes(self, args: list[str], raw: str) -> CommandResult:
        return CommandResult(messages=self.bot.build_modes_lines())

    def _cmd_params(self, args: list[str], raw: str) -> CommandResult:
        search_term = " ".join(args).strip() if args else None
        return CommandResult(messages=self.bot.build_params_lines(search_term=search_term))

    def _cmd_get(self, args: list[str], raw: str) -> CommandResult:
        if not args:
            return CommandResult(messages=["Usage: `get <parameter_name>`"])

        param_key = args[0]
        try:
            normalized_key, value = self.bot.get_parameter_value(param_key)
        except KeyError:
            return CommandResult(messages=[f"[Bot Config] Unknown parameter: `{param_key}`"])

        value_text = repr(value)
        if len(value_text) > 300:
            value_text = f"{value_text[:297]}..."

        return CommandResult(messages=[f"[Bot Config] `{normalized_key}` = `{value_text}`"])

    def _cmd_set(self, args: list[str], raw: str) -> CommandResult:
        if len(args) < 2:
            return CommandResult(messages=["Usage: `set <parameter_name> <value>`"])

        param_key = args[0]
        raw_value = " ".join(args[1:])
        try:
            update = self.bot.set_parameter_value(param_key, raw_value)
        except KeyError:
            return CommandResult(messages=[f"[Bot Config] Unknown parameter: `{param_key}`"])
        except ValueError as exc:
            return CommandResult(messages=[f"[Bot Config] Invalid value: {exc}"])
        except RuntimeError as exc:
            return CommandResult(messages=[f"[Bot Config] Update failed: {exc}"])

        if not update.changed:
            return CommandResult(
                messages=[f"[Bot Config] `{update.key}` unchanged (`{update.new_value}`)."]
            )

        persistence = "saved" if update.persisted else "not saved"
        return CommandResult(
            messages=[
                f"[Bot Config] `{update.key}` updated from `{update.old_value}` to `{update.new_value}` ({persistence})."
            ]
        )

    def _cmd_toggle(self, args: list[str], raw: str) -> CommandResult:
        if not args:
            return CommandResult(messages=["Usage: `toggle <mode_name> [on|off]`"])

        mode_key = args[0]
        desired_state = args[1] if len(args) > 1 else None

        try:
            update = self.bot.toggle_mode(mode_key, desired_state=desired_state)
        except KeyError:
            return CommandResult(messages=[f"[Bot Config] Unknown mode: `{mode_key}`"])
        except ValueError as exc:
            return CommandResult(messages=[f"[Bot Config] {exc}"])
        except RuntimeError as exc:
            return CommandResult(messages=[f"[Bot Config] Update failed: {exc}"])

        state_text = "ENABLED" if bool(update.new_value) else "DISABLED"
        persistence = "saved" if update.persisted else "not saved"
        return CommandResult(
            messages=[f"[Bot Mode] `{update.key}` set to `{state_text}` ({persistence})."]
        )

    def _cmd_reload(self, args: list[str], raw: str) -> CommandResult:
        self.bot.reload_configuration()
        return CommandResult(messages=["[Bot Config] Configuration reloaded from disk."])

    def _cmd_start(self, args: list[str], raw: str) -> CommandResult:
        return CommandResult(
            messages=["[Bot Status] Manual mode disabled. Resuming automation."],
            exit_manual_mode=True,
        )

    def _cmd_stop(self, args: list[str], raw: str) -> CommandResult:
        return CommandResult(
            messages=["[Bot Status] Manual mode enabled. Automation paused."],
            enter_manual_mode=True,
        )

    def _cmd_restart(self, args: list[str], raw: str) -> CommandResult:
        return CommandResult(
            messages=[
                "[Bot Status] Restart command received. "
                "Restarting bot process and Raid application."
            ],
            restart_requested=True,
        )

    def _cmd_ping(self, args: list[str], raw: str) -> CommandResult:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return CommandResult(messages=[f"[Bot Status] pong ({now} UTC)"])
