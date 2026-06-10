import tempfile
import unittest
from pathlib import Path

from raid_bot.core import runtime_discord


class RuntimeDiscordTests(unittest.TestCase):
    def test_named_token_takes_precedence_over_plain_non_comment_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            secret_file = Path(directory) / ".ssh"
            secret_file.write_text(
                "# ignored\nfallback-token\nDISCORD_TOKEN=preferred-token\n",
                encoding="utf-8",
            )

            self.assertEqual(
                runtime_discord.load_discord_token(str(secret_file)),
                "preferred-token",
            )

    def test_plain_first_non_empty_line_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            secret_file = Path(directory) / ".ssh"
            secret_file.write_text("# ignored\n\nplain-token\n", encoding="utf-8")

            self.assertEqual(
                runtime_discord.load_discord_token(str(secret_file)),
                "plain-token",
            )

    def test_missing_and_empty_file_failures_preserve_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_file = Path(directory) / "missing.ssh"
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Discord token file not found:",
            ):
                runtime_discord.load_discord_token(str(missing_file))

            empty_file = Path(directory) / "empty.ssh"
            empty_file.write_text("# comments only\n\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "Discord token file is empty:",
            ):
                runtime_discord.load_discord_token(str(empty_file))


if __name__ == "__main__":
    unittest.main()
