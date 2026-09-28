"""Backoff policy. No sleeping anywhere -- everything is a pure computation."""

from __future__ import annotations

import random

import pytest

from pulse_queue.backoff import BackoffPolicy, should_retry


def test_delay_doubles_per_attempt() -> None:
    policy = BackoffPolicy(jitter=0.0)
    assert [policy.delay_for(n) for n in range(5)] == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_delay_caps_at_max_delay() -> None:
    policy = BackoffPolicy(base=1.0, max_delay=10.0, jitter=0.0)
    assert policy.delay_for(3) == 8.0
    assert policy.delay_for(4) == 10.0
    # Far past the cap: no overflow, still capped.
    assert policy.delay_for(1000) == 10.0


def test_zero_jitter_is_deterministic() -> None:
    policy = BackoffPolicy(jitter=0.0)
    assert policy.delay_for(2) == 4.0
    assert policy.delay_for(2) == 4.0


def test_jitter_stays_within_band() -> None:
    policy = BackoffPolicy(base=1.0, max_delay=1000.0, jitter=0.2)
    rng = random.Random(1234)
    for attempt in range(8):
        expected = min(1.0 * 2.0**attempt, 1000.0)
        for _ in range(50):
            delay = policy.delay_for(attempt, rng=rng)
            assert expected * 0.8 <= delay <= expected * 1.2


def test_jitter_is_reproducible_with_seeded_rng() -> None:
    policy = BackoffPolicy(jitter=0.5)
    first = [policy.delay_for(2, rng=random.Random(7)) for _ in range(5)]
    second = [policy.delay_for(2, rng=random.Random(7)) for _ in range(5)]
    assert first == second


def test_jitter_actually_varies() -> None:
    policy = BackoffPolicy(jitter=0.2)
    rng = random.Random(99)
    samples = {policy.delay_for(3, rng=rng) for _ in range(50)}
    assert len(samples) > 1


def test_jitter_never_produces_negative_delay() -> None:
    policy = BackoffPolicy(base=0.001, max_delay=0.002, jitter=1.0)
    rng = random.Random(3)
    for _ in range(200):
        assert policy.delay_for(0, rng=rng) >= 0.0


def test_zero_base_disables_delay() -> None:
    assert BackoffPolicy(base=0.0).delay_for(5) == 0.0


def test_next_available_at_is_absolute() -> None:
    policy = BackoffPolicy(base=2.0, jitter=0.0)
    assert policy.next_available_at(now=1000.0, attempt=2) == 1008.0


def test_should_retry_boundaries() -> None:
    assert should_retry(attempts=0, max_attempts=3) is True
    assert should_retry(attempts=2, max_attempts=3) is True
    assert should_retry(attempts=3, max_attempts=3) is False
    assert should_retry(attempts=4, max_attempts=3) is False


def test_zero_retries_exhausts_immediately() -> None:
    assert should_retry(attempts=0, max_attempts=0) is False
    assert should_retry(attempts=1, max_attempts=0) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base": -1.0},
        {"base": 10.0, "max_delay": 1.0},
        {"jitter": 1.5},
        {"jitter": -0.1},
    ],
)
def test_invalid_policy_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        BackoffPolicy(**kwargs)


def test_negative_attempt_rejected() -> None:
    with pytest.raises(ValueError):
        BackoffPolicy().delay_for(-1)


def test_backoff_integration_with_store(store, clock) -> None:
    """A failing task becomes claimable exactly when backoff says."""
    from pulse_queue import TaskStatus

    policy = BackoffPolicy(base=5.0, jitter=0.0)
    store.enqueue({"a": 1})
    task = store.lease_next_task("w1")

    next_at = policy.next_available_at(clock.now(), attempt=task.attempts - 1)
    store.fail_task(task.id, "w1", "boom", available_at=next_at)

    clock.advance(4.9)
    assert store.lease_next_task("w2") is None

    clock.advance(0.1)
    assert store.lease_next_task("w2") is not None
    assert store.get_task(task.id).state is TaskStatus.RUNNING
