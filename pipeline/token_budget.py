"""
Sliding-window tokens-per-minute (TPM) tracker.

Used to throttle API calls against providers that enforce a TPM quota.
Single-threaded; no locking needed.
"""

from __future__ import annotations

import time
from collections import deque


class TokenBudget:
    """
    Tracks token usage over a rolling 60-second window.

    Usage pattern:
        budget.wait_if_needed(estimated_tokens)   # before the API call
        response = litellm.completion(...)
        budget.record(response.usage.total_tokens) # after the call
    """

    def __init__(self, tpm_limit: int, window_seconds: float = 60.0) -> None:
        self.tpm_limit = tpm_limit
        self._window = window_seconds
        self._events: deque[tuple[float, int]] = deque()  # (monotonic_ts, tokens)

    def _evict(self) -> None:
        cutoff = time.monotonic() - self._window
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _used(self) -> int:
        self._evict()
        return sum(t for _, t in self._events)

    def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Block until adding estimated_tokens would not exceed tpm_limit."""
        while True:
            self._evict()
            if self._used() + estimated_tokens <= self.tpm_limit:
                return
            # Sleep until the oldest entry expires, then re-check.
            oldest_ts = self._events[0][0]
            sleep_for = (oldest_ts + self._window) - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    def record(self, actual_tokens: int) -> None:
        """Record actual tokens consumed by a completed call."""
        self._events.append((time.monotonic(), actual_tokens))
