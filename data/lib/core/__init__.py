"""Core runtime abstractions for orchestration and command handling."""

from .command_router import BotCommandRouter, CommandResult

__all__ = ["BotCommandRouter", "CommandResult"]
