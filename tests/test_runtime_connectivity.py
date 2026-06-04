import threading
import time
import unittest

from data.lib.core import runtime_connectivity


class RuntimeConnectivityTests(unittest.TestCase):
    def test_outage_pause_and_recovery_signal_flow(self):
        states = iter([False, False, True, True, True])
        lost_event = threading.Event()
        restored_event = threading.Event()
        retry_attempts = []

        def probe(timeout_seconds=3.0):
            try:
                return next(states)
            except StopIteration:
                return True

        supervisor = runtime_connectivity.ConnectivityRecoverySupervisor(
            connectivity_probe=probe,
            online_poll_interval_seconds=0.01,
            reconnect_check_interval_seconds=0.01,
            on_connection_lost=lambda *_: lost_event.set(),
            on_retry_attempt=lambda attempt, *_: retry_attempts.append(attempt),
            on_connection_restored=lambda *_: restored_event.set(),
        )

        supervisor.start()
        self.assertTrue(lost_event.wait(timeout=0.5))
        self.assertTrue(supervisor.is_paused())
        self.assertTrue(restored_event.wait(timeout=0.5))

        self.assertFalse(supervisor.is_paused())
        self.assertTrue(supervisor.consume_recovery_signal())
        self.assertFalse(supervisor.consume_recovery_signal())
        self.assertTrue(any(attempt >= 1 for attempt in retry_attempts))

        supervisor.stop()

    def test_probe_exceptions_are_treated_as_offline(self):
        lost_event = threading.Event()

        def failing_probe(timeout_seconds=3.0):
            raise RuntimeError("probe failed")

        supervisor = runtime_connectivity.ConnectivityRecoverySupervisor(
            connectivity_probe=failing_probe,
            online_poll_interval_seconds=0.01,
            reconnect_check_interval_seconds=0.05,
            on_connection_lost=lambda *_: lost_event.set(),
        )

        supervisor.start()
        self.assertTrue(lost_event.wait(timeout=0.5))
        self.assertTrue(supervisor.is_paused())
        supervisor.stop()

    def test_start_and_stop_are_idempotent(self):
        supervisor = runtime_connectivity.ConnectivityRecoverySupervisor(
            connectivity_probe=lambda timeout_seconds=3.0: True,
            online_poll_interval_seconds=0.01,
            reconnect_check_interval_seconds=0.01,
        )
        supervisor.start()
        first_thread = supervisor._thread
        supervisor.start()
        self.assertIs(first_thread, supervisor._thread)
        self.assertTrue(first_thread.is_alive())

        supervisor.stop()
        time.sleep(0.02)
        supervisor.stop()
        self.assertFalse(first_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
