"""Exponential backoff with jitter.

Pure computation -- nothing here sleeps. A retry is scheduled by writing
``available_at`` on the task row; the claim query simply skips anything not yet
due. That keeps retry scheduling declarative and testable without real time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """Exponential backoff with bounded full jitter.

    Args:
        base: Delay for the first retry, in seconds.
        max_delay: Ceiling applied before jitter, so the cap actually holds.
        jitter: Fraction of the computed delay to randomize, in ``[0, 1]``.
            ``0.2`` spreads each delay over +/-10%.

    >>> BackoffPolicy(jitter=0.0).delay_for(attempt=0)
    1.0
    """

    base: float = 1.0
    max_delay: float = 300.0
    jitter: float = 0.2

    def __post_init__(self) -> None:
        if self.base < 0:
            raise ValueError("base must be >= 0")
        if self.max_delay < self.base:
            raise ValueError("max_delay must be >= base")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter must be in [0, 1]")

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """Seconds to wait before retry number ``attempt`` (0-based).

        Delay doubles per attempt, is capped at ``max_delay``, then jittered.
        Returns a value in ``[delay * (1 - jitter), delay * (1 + jitter)]``.
        """
        if attempt < 0:
            raise ValueError("attempt must be >= 0")
        if self.base == 0:
            return 0.0

        # Compute in float then cap, so a large attempt count cannot overflow
        # before the ceiling is applied.
        delay = min(self.base * (2.0**attempt), self.max_delay)
        if self.jitter == 0.0:
            return delay

        source = rng or random
        spread = delay * self.jitter
        return max(0.0, delay + source.uniform(-spread, spread))

    def next_available_at(
        self, now: float, attempt: int, *, rng: random.Random | None = None
    ) -> float:
        """Absolute unix timestamp at which ``attempt`` becomes claimable."""
        return now + self.delay_for(attempt, rng=rng)


def should_retry(attempts: int, max_attempts: int) -> bool:
    """Whether another attempt is permitted.

    ``attempts`` is the count already consumed. Exhaustion happens when
    ``attempts >= max_attempts``, so ``max_attempts=3`` permits three attempts
    in total -- not one attempt plus three retries.

    The budget has an effective floor of 1: a task cannot be observed to fail
    without being attempted, so ``max_attempts=0`` and ``max_attempts=1`` both
    result in exactly one attempt before dead-lettering.

    >>> should_retry(attempts=2, max_attempts=3)
    True
    >>> should_retry(attempts=3, max_attempts=3)
    False
    >>> should_retry(attempts=1, max_attempts=0)
    False
    """
    return attempts < max_attempts
