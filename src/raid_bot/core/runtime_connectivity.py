"""Internet connectivity supervision and recovery signaling helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import socket
import threading
import time


logger = logging.getLogger(__name__)


def default_connectivity_probe(timeout_seconds: float = 3.0) -> bool:
    """Return True when at least one well-known network endpoint is reachable."""
    endpoints = (("1.1.1.1", 53), ("8.8.8.8", 53), ("9.9.9.9", 53))
    for host, port in endpoints:
        try:
            with socket.create_connection((host, port), timeout=timeout_seconds):
                return True
        except OSError:
            continue
    return False


class ConnectivityRecoverySupervisor:
    """Background monitor that pauses runtime on outage and signals recovery."""

    def __init__(
        self,
        *,
        connectivity_probe=None,
        online_poll_interval_seconds: float = 5.0,
        reconnect_check_interval_seconds: float = 600.0,
        outage_confirmation_seconds: float = 60.0,
        probe_timeout_seconds: float = 3.0,
        on_connection_lost=None,
        on_retry_attempt=None,
        on_connection_restored=None,
    ):
        self.connectivity_probe = connectivity_probe or default_connectivity_probe
        self.online_poll_interval_seconds = float(max(0.01, online_poll_interval_seconds))
        self.reconnect_check_interval_seconds = float(
            max(0.01, reconnect_check_interval_seconds)
        )
        self.outage_confirmation_seconds = float(max(0.0, outage_confirmation_seconds))
        self.probe_timeout_seconds = float(max(0.5, probe_timeout_seconds))
        self.on_connection_lost = on_connection_lost
        self.on_retry_attempt = on_retry_attempt
        self.on_connection_restored = on_connection_restored

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._offline_since: datetime | None = None
        self._offline_candidate_since_monotonic: float | None = None
        self._retry_attempts = 0
        self._recovery_pending = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="connectivity-recovery",
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop_event.is_set())

    def is_paused(self) -> bool:
        with self._lock:
            return self._offline_since is not None

    def consume_recovery_signal(self) -> bool:
        with self._lock:
            if not self._recovery_pending:
                return False
            self._recovery_pending = False
            return True

    def _run(self) -> None:
        while not self._stop_event.is_set():
            online = self._probe_online()
            wait_seconds = self.online_poll_interval_seconds

            if online:
                self._clear_offline_candidate()
                restored_payload = self._mark_online_if_needed()
                if restored_payload:
                    downtime_seconds, retry_attempts = restored_payload
                    self._invoke_callback(
                        self.on_connection_restored,
                        downtime_seconds,
                        retry_attempts,
                    )
            else:
                wait_seconds = self._mark_offline_and_get_wait()

            if self._stop_event.wait(timeout=wait_seconds):
                break

    def _probe_online(self) -> bool:
        try:
            return bool(self.connectivity_probe(timeout_seconds=self.probe_timeout_seconds))
        except Exception:
            logger.debug("Connectivity probe failed unexpectedly.", exc_info=True)
            return False

    def _mark_online_if_needed(self) -> tuple[float, int] | None:
        with self._lock:
            if self._offline_since is None:
                return None

            offline_since = self._offline_since
            retry_attempts = self._retry_attempts
            self._offline_since = None
            self._retry_attempts = 0
            self._recovery_pending = True

        downtime_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - offline_since).total_seconds(),
        )
        return downtime_seconds, retry_attempts

    def _mark_offline_and_get_wait(self) -> float:
        now_monotonic = time.monotonic()
        now_utc = datetime.now(timezone.utc)

        with self._lock:
            if self._offline_since is None:
                if self._offline_candidate_since_monotonic is None:
                    self._offline_candidate_since_monotonic = now_monotonic
                elapsed_candidate = now_monotonic - self._offline_candidate_since_monotonic
                if elapsed_candidate < self.outage_confirmation_seconds:
                    return self.online_poll_interval_seconds

            if self._offline_since is None:
                self._offline_since = now_utc
                self._retry_attempts = 0
                first_outage = True
                retry_attempt = 0
            else:
                self._retry_attempts += 1
                first_outage = False
                retry_attempt = self._retry_attempts

        if first_outage:
            self._invoke_callback(self.on_connection_lost, datetime.now(timezone.utc))
        else:
            self._invoke_callback(
                self.on_retry_attempt,
                retry_attempt,
                self.reconnect_check_interval_seconds,
            )

        return self.reconnect_check_interval_seconds

    def _clear_offline_candidate(self) -> None:
        with self._lock:
            self._offline_candidate_since_monotonic = None

    @staticmethod
    def _invoke_callback(callback, *args) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            logger.exception("Connectivity callback failed.")
