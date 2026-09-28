"""Shared fixtures.

Every test runs against a real SQLite file in ``tmp_path`` -- real WAL, real
locking, no mocked DB layer. Time is injected via ``FrozenClock`` so
backoff and lease-expiry assertions are exact instead of sleeping.
"""

from __future__ import annotations

import pytest

from pulse_queue import FrozenClock, Store


@pytest.fixture
def clock() -> FrozenClock:
    """Deterministic clock starting at a fixed epoch."""
    return FrozenClock(1_700_000_000.0)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "pulse.db"


@pytest.fixture
def store(db_path, clock):
    """A migrated store backed by a real file on disk."""
    with Store(db_path, clock=clock) as store:
        yield store


@pytest.fixture
def make_store(db_path):
    """Factory for extra Store handles against the same file.

    Used to simulate a second worker process sharing one database.
    """
    stores: list[Store] = []

    def _make(clock=None, **kwargs) -> Store:
        store = Store(db_path, clock=clock, **kwargs)
        stores.append(store)
        return store

    yield _make
    for store in stores:
        store.close()
