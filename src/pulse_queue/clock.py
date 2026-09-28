"""Time source abstraction.

The store never calls :func:`time.time` directly. Backoff and lease-expiry
tests must assert exact timestamps rather than sleeping, so all wall-clock
reads go through a :class:`Clock`.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Returns unix epoch seconds as a float."""

    def now(self) -> float: ...


class SystemClock:
    """Real wall-clock time. The default in production."""

    __slots__ = ()

    def now(self) -> float:
        return time.time()


class FrozenClock:
    """Manually advanced clock for tests.

    >>> clock = FrozenClock(1000.0)
    >>> clock.now()
    1000.0
    >>> clock.advance(60.0)
    >>> clock.now()
    1060.0
    """

    __slots__ = ("_now",)

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot advance a clock backwards")
        self._now += seconds
