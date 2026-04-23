"""
Async rate limiter combining RPM, TPM, and TPD constraints.

Each model (or shared resource group) gets one AsyncRateLimiter instance.
All three limits are enforced before every API call:

  RPM — requests per minute, enforced via evenly-spaced token bucket.
  TPM — tokens per minute, enforced via sliding 60-second window.
  TPD — tokens per day (optional), hard counter that raises DailyLimitExhausted.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque


class DailyLimitExhausted(Exception):
    """Raised when the daily token budget for a model/group is used up."""

    def __init__(self, used: int, limit: int) -> None:
        self.used = used
        self.limit = limit
        super().__init__(f"Daily token limit exhausted: {used:,}/{limit:,}")


class AsyncRateLimiter:
    """
    Per-model (or per-group) async rate limiter.

    Usage::

        limiter = AsyncRateLimiter(rpm=5000, tpm=2_000_000, tpd=1_000_000)

        await limiter.acquire(estimated_tokens)
        response = await litellm.acompletion(...)
        await limiter.record(response.usage.total_tokens)
    """

    def __init__(self, rpm: int, tpm: int, tpd: int | None = None, rpd: int | None = None) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self.tpd = tpd
        self.rpd = rpd

        # RPM: token-bucket — enforce minimum interval between requests.
        # Calculated delay is applied outside the lock so other coroutines
        # can acquire their slot concurrently.
        self._min_interval = 60.0 / rpm
        self._next_request_time = 0.0
        self._rpm_lock = asyncio.Lock()

        # TPM: sliding 60-second window of (monotonic_ts, tokens).
        self._token_events: deque[tuple[float, int]] = deque()
        self._tpm_lock = asyncio.Lock()
        # In-flight pre-reservation: tokens claimed by acquire() but not yet recorded.
        self._in_flight_tokens = 0

        # TPD: cumulative counter (single-threaded asyncio — no lock needed).
        self._tpd_used = 0

        # RPD: cumulative request counter.
        self._rpd_used = 0

    # ── Public API ───────────────────────────────────────────────────────────

    async def acquire(self, estimated_tokens: int) -> None:
        """Block until this request can proceed within all rate limits."""
        # 1. RPD hard check
        if self.rpd is not None and self._rpd_used >= self.rpd:
            raise DailyLimitExhausted(self._rpd_used, self.rpd)

        # 2. TPD hard check (fail fast — no point waiting)
        if self.tpd is not None and self._tpd_used + estimated_tokens > self.tpd:
            raise DailyLimitExhausted(self._tpd_used, self.tpd)

        self._rpd_used += 1

        # 2. TPM sliding-window gate
        await self._wait_tpm(estimated_tokens)

        # 3. RPM token-bucket gate
        await self._wait_rpm()

    async def record(self, actual_tokens: int, estimated_tokens: int = 0) -> None:
        """Record actual token usage after a completed call.

        estimated_tokens: the value passed to acquire() for this call, so the
        pre-reservation can be released and replaced with the true usage.
        """
        now = time.monotonic()
        async with self._tpm_lock:
            self._token_events.append((now, actual_tokens))
            self._in_flight_tokens = max(0, self._in_flight_tokens - estimated_tokens)
        self._tpd_used += actual_tokens

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def tpd_used(self) -> int:
        return self._tpd_used

    @property
    def tpd_remaining(self) -> int | None:
        return None if self.tpd is None else max(0, self.tpd - self._tpd_used)

    @property
    def rpd_remaining(self) -> int | None:
        return None if self.rpd is None else max(0, self.rpd - self._rpd_used)

    # ── Internals ────────────────────────────────────────────────────────────

    async def _wait_rpm(self) -> None:
        """Wait until the next RPM slot is available (evenly-spaced requests)."""
        delay = 0.0
        async with self._rpm_lock:
            now = time.monotonic()
            if now < self._next_request_time:
                delay = self._next_request_time - now
                self._next_request_time += self._min_interval
            else:
                self._next_request_time = now + self._min_interval
        # Sleep outside the lock so others can schedule their slots.
        if delay > 0:
            await asyncio.sleep(delay)

    async def _wait_tpm(self, estimated_tokens: int) -> None:
        """Wait until the TPM sliding window has room for estimated_tokens.

        Pre-reserves estimated_tokens in _in_flight_tokens on success so that
        concurrent coroutines see the correct remaining budget before responses
        arrive. The reservation is released in record() once actual usage is known.
        """
        while True:
            delay = 0.0
            async with self._tpm_lock:
                now = time.monotonic()
                # Evict expired entries
                cutoff = now - 60.0
                while self._token_events and self._token_events[0][0] < cutoff:
                    self._token_events.popleft()
                used = sum(t for _, t in self._token_events)
                if used + self._in_flight_tokens + estimated_tokens <= self.tpm:
                    self._in_flight_tokens += estimated_tokens  # pre-reserve
                    return  # good to go
                # Need to wait — compute how long until enough budget frees up.
                if self._token_events:
                    delay = self._token_events[0][0] + 60.0 - now
            # Sleep outside the lock.
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0.1)
