import _bootstrap  # noqa: F401

from raid_bot.handlers.error_handler import RSL_Bot_ErrorHandler


if __name__ == "__main__":
    handler = RSL_Bot_ErrorHandler()
    handler.run_permanently()
