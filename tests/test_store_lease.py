"""Lease exclusivity, fencing, and recovery.

These are the tests that matter most: they cover the failure modes where a
queue silently corrupts data rather than merely falling over.
"""

from __future__ import annotations

from pulse_queue import Store, TaskStatus


def test_lease_marks_running_and_increments_attempts(store: Store, clock) -> None:
    store.enqueue({"a": 1})
    task = store.lease_next_task("worker-1", lease_ttl=60.0)

    assert task is not None
    assert task.state is TaskStatus.RUNNING
    assert task.attempts == 1
    assert task.lease_owner == "worker-1"
    assert task.lease_expires_at == clock.now() + 60.0
    assert task.started_at == clock.now()


def test_lease_returns_none_when_empty(store: Store) -> None:
    assert store.lease_next_task("worker-1") is None


def test_concurrent_claims_never_return_same_task(store: Store) -> None:
    """The select-then-update race must not exist."""
    for i in range(20):
        store.enqueue({"n": i})

    workers = [f"worker-{i}" for i in range(20)]
    claimed = [store.lease_next_task(w) for w in workers]

    ids = [t.id for t in claimed if t is not None]
    assert len(ids) == 20
    assert len(set(ids)) == 20
    # Every row moved to RUNNING; none are left claimable.
    assert store.count_tasks(TaskStatus.RUNNING) == 20
    assert store.count_tasks(TaskStatus.PENDING) == 0
    assert store.lease_next_task("worker-x") is None


def test_task_is_not_reclaimable_while_running(store: Store) -> None:
    store.enqueue({"a": 1})
    store.lease_next_task("worker-1")

    assert store.lease_next_task("worker-2") is None
    # Still RUNNING and owned -- the second worker got nothing.
    assert store.count_tasks(TaskStatus.RUNNING) == 1


def test_priority_ordering(store: Store) -> None:
    store.enqueue({"n": "low"}, priority=0)
    store.enqueue({"n": "high"}, priority=10)
    store.enqueue({"n": "mid"}, priority=5)

    order = []
    while (task := store.lease_next_task("w")) is not None:
        order.append(task.decode_payload()["n"])

    assert order == ["high", "mid", "low"]


def test_available_at_breaks_priority_ties(store: Store, clock) -> None:
    store.enqueue({"n": "later"}, priority=5, delay=10.0)
    store.enqueue({"n": "sooner"}, priority=5)

    first = store.lease_next_task("w")
    assert first.decode_payload()["n"] == "sooner"

    clock.advance(10.0)
    second = store.lease_next_task("w")
    assert second.decode_payload()["n"] == "later"


def test_queue_filter_restricts_claims(store: Store) -> None:
    store.enqueue({"n": 1}, queue="alpha")
    store.enqueue({"n": 2}, queue="beta")

    task = store.lease_next_task("w", queues=["beta"])
    assert task.decode_payload()["n"] == 2
    assert store.lease_next_task("w", queues=["beta"]) is None

    # The alpha task is untouched and still claimable.
    assert store.lease_next_task("w", queues=["alpha"]) is not None


# ------------------------------------------------------- stale-lease fencing


def test_stale_worker_cannot_complete_after_reclaim(store: Store, clock) -> None:
    """A zombie worker must not clobber the new owner's state.

    This is the single most important invariant in the design: without the
    ``lease_owner`` guard, a paused worker resuming after its lease expired
    would overwrite the result of whoever re-ran the task.
    """
    store.enqueue({"a": 1})

    zombie = store.lease_next_task("zombie", lease_ttl=10.0)
    assert zombie is not None

    # Lease expires; a second worker reclaims the task.
    clock.advance(11.0)
    assert store.reclaim_expired() == 1
    successor = store.lease_next_task("successor", lease_ttl=10.0)
    assert successor is not None
    assert successor.attempts == 2

    # The zombie finishes and tries to report success -- rejected.
    assert store.complete_task(zombie.id, "zombie") is False
    assert store.fail_task(zombie.id, "zombie", "boom") is False

    # Successor still owns it and its state is untouched.
    current = store.get_task(zombie.id)
    assert current.lease_owner == "successor"
    assert current.state is TaskStatus.RUNNING

    assert store.complete_task(successor.id, "successor") is True
    assert store.get_task(successor.id).state is TaskStatus.COMPLETED


def test_complete_requires_matching_owner(store: Store) -> None:
    store.enqueue({"a": 1})
    task = store.lease_next_task("w1")

    assert store.complete_task(task.id, "w2") is False
    assert store.get_task(task.id).state is TaskStatus.RUNNING
    assert store.complete_task(task.id, "w1") is True


def test_complete_is_not_replayable(store: Store) -> None:
    """Completing twice must not double-count or resurrect the row."""
    store.enqueue({"a": 1})
    task = store.lease_next_task("w1")

    assert store.complete_task(task.id, "w1") is True
    assert store.complete_task(task.id, "w1") is False


# ----------------------------------------------------------- crash recovery


def test_reclaim_recovers_crashed_worker(store: Store, clock) -> None:
    """SIGKILL mid-handler must not lose the task."""
    store.enqueue({"a": 1})
    task = store.lease_next_task("crashed", lease_ttl=30.0)
    assert task.state is TaskStatus.RUNNING

    # Before expiry: nothing to reclaim.
    clock.advance(29.0)
    assert store.reclaim_expired() == 0

    clock.advance(2.0)
    assert store.reclaim_expired() == 1

    recovered = store.get_task(task.id)
    assert recovered.state is TaskStatus.RETRY
    assert recovered.lease_owner is None
    assert "reclaimed" in recovered.last_error

    # And it is immediately claimable again.
    again = store.lease_next_task("next")
    assert again is not None
    assert again.attempts == 2  # crashed attempt still counted


def test_reclaim_respects_backoff(store: Store, clock) -> None:
    store.enqueue({"a": 1})
    task = store.lease_next_task("w", lease_ttl=10.0)

    clock.advance(11.0)
    store.reclaim_expired(backoff=60.0)

    # Re-queued for the future, so not immediately claimable.
    assert store.lease_next_task("w") is None
    clock.advance(60.0)
    assert store.lease_next_task("w").id == task.id


def test_renew_lease_extends_and_is_fenced(store: Store, clock) -> None:
    store.enqueue({"a": 1})
    task = store.lease_next_task("w1", lease_ttl=10.0)

    clock.advance(8.0)
    assert store.renew_lease(task.id, "w1", lease_ttl=30.0) is True
    assert store.get_task(task.id).lease_expires_at == clock.now() + 30.0

    # Wrong owner cannot renew.
    assert store.renew_lease(task.id, "impostor", lease_ttl=30.0) is False


def test_reclaim_ignores_unexpired_leases(store: Store) -> None:
    store.enqueue({"a": 1})
    store.lease_next_task("w", lease_ttl=1e9)
    assert store.reclaim_expired() == 0


def test_renew_lease_prevents_reclaim(store: Store, clock) -> None:
    """A handler that heartbeats is never stolen mid-run."""
    store.enqueue({"a": 1})
    task = store.lease_next_task("w", lease_ttl=10.0)

    for _ in range(5):
        clock.advance(8.0)
        assert store.renew_lease(task.id, "w", lease_ttl=10.0) is True
        assert store.reclaim_expired() == 0

    assert store.get_task(task.id).state is TaskStatus.RUNNING


def test_two_stores_share_one_database(store: Store, make_store, clock) -> None:
    """A second worker process sees the first's writes immediately."""
    store.enqueue({"a": 1})
    other = make_store(clock=clock)

    task = other.lease_next_task("worker-2")
    assert task is not None
    # Visible from the original handle too -- WAL readers are not blocked.
    assert store.get_task(task.id).lease_owner == "worker-2"


def test_stored_state_survives_reopen(db_path, clock) -> None:
    first = Store(db_path, clock=clock)
    task = first.enqueue({"a": 1}, task_id="persisted")
    first.lease_next_task("w1")
    first.close()

    second = Store(db_path, clock=clock)
    try:
        restored = second.get_task("persisted")
        assert restored.id == task.id
        assert restored.state is TaskStatus.RUNNING
        assert restored.attempts == 1
    finally:
        second.close()
