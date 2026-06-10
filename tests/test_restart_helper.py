import unittest
from unittest.mock import Mock, patch

from raid_bot import mainframe


class RestartHelperTests(unittest.TestCase):
    def test_launch_wait_retries_after_window_timeout(self):
        wait_results = [False, True]

        with patch.object(mainframe, "_terminate_processes") as terminate:
            with patch.object(mainframe.time, "sleep") as sleep:
                with patch.object(mainframe.subprocess, "Popen") as popen:
                    with patch.object(mainframe, "_wait_for_raid_window", side_effect=wait_results) as wait:
                        mainframe._launch_raid_and_wait_until_window(
                            launch_command=["raid"],
                            cwd="cwd",
                            timeout_seconds=180.0,
                            retry_delay_seconds=5.0,
                            log=Mock(),
                        )

        self.assertEqual(terminate.call_count, 2)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(wait.call_count, 2)
        self.assertEqual(popen.call_args_list[0].kwargs["cwd"], "cwd")


if __name__ == "__main__":
    unittest.main()
