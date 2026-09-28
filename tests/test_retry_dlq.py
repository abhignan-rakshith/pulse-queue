"""Retry transitions and dead-letter behaviour."""

from __future__ import annotations

import pytest

from pulse_queue import QueueingError, Store, TaskStatus
from pulse_queue._sqlite import transaction
from pulse_queue.store import DEFAULT_MAX_ATTEMPTS


def test_fail_task_schedules_retry(store: Store, clock) -> None:
    store.enqueue({"a": 1})
    task = store.lease_next_task("w1")

    assert store.fail_task(task.id, "w1", "boom", available_at=clock.now() + 60)

    failed = store.get_task(task.id)
    assert failed.state is TaskStatus.RETRY
    assert failed.last_error == "boom"
    assert failed.available_at == clock.now() + 60
    assert failed.lease_owner is None  # released so it can be re-claimed
    assert failed.finished_at is None


def test_retry_is_not_claimable_until_available_at(store: Store, clock) -> None:
    store.enqueue({"a": 1})
    task = store.lease_next_task("w1")
    store.fail_task(task.id, "w1", "boom", available_at=clock.now() + 30)

    assert store.lease_next_task("w2") is None
    clock.advance(30.0)
    assert store.lease_next_task("w2").id == task.id


def test_attempts_accumulate_across_retries(store: Store) -> None:
    store.enqueue({"a": 1})
    for expected in (1, 2, 3):
        task = store.lease_next_task("w")
        assert task.attempts == expected
        store.fail_task(task.id, "w", "boom")


def test_permanent_failure_skips_retry(store: Store) -> None:
    store.enqueue({"a": 1})
    task = store.lease_next_task("w1")

    assert store.fail_task(task.id, "w1", "bad payload", retry=False)

    failed = store.get_task(task.id)
    assert failed.state is TaskStatus.FAILED
    assert failed.finished_at is not None
    assert failed.lease_owner is None

    # FAILED is terminal: never re-claimed, no backoff wait.
    clock_now = store.get_task(task.id).available_at
    assert store.lease_next_task("w2") is None
    assert store.get_task(task.id).available_at == clock_now


def test_fail_task_is_fenced_by_owner(store: Store) -> None:
    store.enqueue({"a": 1})
    task = store.lease_next_task("w1")

    assert store.fail_task(task.id, "impostor", "boom") is False
    assert store.get_task(task.id).state is TaskStatus.RUNNING


def test_completed_task_cannot_be_failed(store: Store) -> None:
    store.enqueue({"a": 1})
    task = store.lease_next_task("w1")
    store.complete_task(task.id, "w1")

    assert store.fail_task(task.id, "w1", "too late") is False
    assert store.get_task(task.id).state is TaskStatus.COMPLETED


# ------------------------------------------------------------- dead letter


def test_dead_letter_snapshots_and_removes_task(store: Store, clock) -> None:
    store.enqueue({"a": 1}, task_id="doomed", max_attempts=0)
    task = store.lease_next_task("w1")

    assert store.dead_letter(task.id, "w1", "exhausted") is True

    # Row is gone from the hot table and present in the DLQ.
    assert store.find_task("doomed") is None
    assert store.count_tasks() == 0

    entry = store.list_dead_letters()[0]
    assert entry["id"] == "doomed"
    assert entry["attempts"] == 1
    assert entry["last_error"] == "exhausted"
    assert entry["failed_at"] == clock.now()


def test_dead_letter_preserves_idempotency_key_and_payload(
    store: Store,
) -> None:
    """A replayed task must carry the original key to dedupe correctly."""
    payload = {"email": "a@b.c"}
    store.enqueue(payload, task_id="stable-key")
    task = store.lease_next_task("w1")
    store.dead_letter(task.id, "w1", "exhausted")

    entry = store.list_dead_letters()[0]
    assert entry["id"] == "stable-key"
    assert entry["payload"].decode() == '{"email": "a@b.c"}'


def test_dead_letter_lands_after_exactly_max_attempts(store: Store) -> None:
    """DLQ on the Nth failure, not N-1 or N+1."""
    store.enqueue({"a": 1}, task_id="exhaust", max_attempts=2)

    # Attempt 1 fails -> RETRY (not dead yet).
    first = store.lease_next_task("w")
    assert first.attempts == 1
    assert not _is_exhausted(first.attempts, 2)
    store.fail_task(first.id, "w", "boom 1")

    # Attempt 2 fails -> attempts == max_attempts -> dead-letter.
    second = store.lease_next_task("w")
    assert second.attempts == 2
    assert _is_exhausted(second.attempts, 2)
    assert store.dead_letter(second.id, "w", "boom 2") is True

    assert store.find_task("exhaust") is None
    assert len(store.list_dead_letters()) == 1


def _is_exhausted(attempts: int, max_attempts: int) -> bool:
    # Deliberately not the production predicate: reimplementing the boundary
    # here means a bug in should_retry() cannot make this test agree with it.
    return attempts >= max_attempts


def test_task_count_invariant_holds_through_dlq(store: Store) -> None:
    """Every tasks row is in exactly one state, and DLQ rows are disjoint.

    Note the invariant is `count(tasks) == sum(per-state counts)`, *not*
    `count(tasks) == pending + in-flight`: COMPLETED and FAILED rows are
    retained for inspection rather than deleted, so they are legitimately
    still in the table. Only dead-lettering removes a row.
    """
    for i in range(5):
        store.enqueue({"n": i}, max_attempts=0)

    workers = [f"w{i}" for i in range(5)]
    claimed = [store.lease_next_task(w) for w in workers]
    assert store.count_tasks() == 5
    assert store.count_tasks(TaskStatus.RUNNING) == 5

    # Dead-lettering two of them removes them from tasks entirely.
    for task, worker in zip(claimed[:2], workers[:2], strict=True):
        store.dead_letter(task.id, worker, "exhausted")

    assert store.count_tasks() == 3
    assert store.count_tasks(TaskStatus.RUNNING) == 3
    assert len(store.list_dead_letters()) == 2


def test_state_counts_are_disjoint_and_complete(store: Store) -> None:
    """No row is double-counted or lost across states."""
    store.enqueue({"n": "a"})
    store.enqueue({"n": "b"})
    store.enqueue({"n": "c"})

    claimed = store.lease_next_task("w")
    store.complete_task(claimed.id, "w")
    retried = store.lease_next_task("w")
    store.fail_task(retried.id, "w", "boom")

    per_state = {state: store.count_tasks(state) for state in TaskStatus}
    assert sum(per_state.values()) == store.count_tasks() == 3
    assert per_state[TaskStatus.COMPLETED] == 1
    assert per_state[TaskStatus.RETRY] == 1
    assert per_state[TaskStatus.PENDING] == 1

    # The partial claim index must not contain terminal rows.
    ready = store._conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE state IN ('PENDING','RETRY')"
    ).fetchone()[0]
    assert ready == 2


def test_dead_letter_is_fenced_by_owner(store: Store) -> None:
    store.enqueue({"a": 1})
    task = store.lease_next_task("w1")

    assert store.dead_letter(task.id, "impostor", "boom") is False
    assert store.find_task(task.id) is not None
    assert store.list_dead_letters() == []


class _ExplodingConn:
    """Delegates to a real connection but fails one specific statement.

    ``sqlite3.Connection.execute`` is read-only, so the failure is injected by
    swapping the connection object rather than patching a method on it.
    """

    def __init__(self, conn, fail_on: str) -> None:
        self._conn = conn
        self._fail_on = fail_on

    def execute(self, sql, *args, **kwargs):
        if self._fail_on in sql:
            raise RuntimeError("disk exploded")
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        self._conn.__enter__()
        return self

    def __exit__(self, *exc):
        return self._conn.__exit__(*exc)


def test_dead_letter_is_transactional_on_failure(store: Store) -> None:
    """A failure *after* the DLQ insert must roll that insert back.

    The failure is injected on the DELETE -- the last statement -- so the DLQ
    row has already been written when it fires. Injecting it on the INSERT
    instead would prove nothing: no write would have happened yet, so the test
    would pass even with no transaction at all, which is exactly how this test
    used to pass.
    """
    store.enqueue({"a": 1}, task_id="atomic")
    task = store.lease_next_task("w1")

    real_conn = store._conn
    store._conn = _ExplodingConn(real_conn, "DELETE FROM tasks")
    try:
        with pytest.raises(RuntimeError, match="disk exploded"):
            store.dead_letter(task.id, "w1", "boom")
    finally:
        store._conn = real_conn

    # Rollback preserved the original row and wrote no DLQ entry.
    assert store.find_task("atomic") is not None
    assert store.get_task("atomic").state is TaskStatus.RUNNING
    assert store.list_dead_letters() == []
    # The connection must be usable afterwards, not stuck mid-transaction.
    assert store._conn.in_transaction is False


def test_dead_letter_preserves_priority(store: Store) -> None:
    """Priority has to survive into the DLQ, or replay silently demotes it."""
    store.enqueue({"a": 1}, task_id="urgent", priority=7)
    task = store.lease_next_task("w1")

    store.dead_letter(task.id, "w1", "boom")

    assert store.list_dead_letters()[0]["priority"] == 7


def test_dlq_row_is_upserted_not_duplicated(store: Store) -> None:
    store.enqueue({"a": 1}, task_id="dup")
    task = store.lease_next_task("w1")
    store.dead_letter(task.id, "w1", "first failure")

    # Re-enqueue the same key. The old task row was deleted on dead-letter, so
    # this is a fresh row whose attempts restart at 0; the DLQ row survives.
    store.enqueue({"a": 1}, task_id="dup")
    retry = store.lease_next_task("w2")
    assert retry.attempts == 1
    store.dead_letter(retry.id, "w2", "second failure")

    entries = store.list_dead_letters()
    assert len(entries) == 1
    assert entries[0]["last_error"] == "second failure"
    assert entries[0]["attempts"] == 1


def test_dead_letter_listing_is_newest_first(store: Store, clock) -> None:
    for i in range(3):
        store.enqueue({"n": i}, task_id=f"task-{i}")
        task = store.lease_next_task(f"w{i}")
        store.dead_letter(task.id, f"w{i}", f"error {i}")
        clock.advance(10.0)

    assert [e["id"] for e in store.list_dead_letters()] == [
        "task-2",
        "task-1",
        "task-0",
    ]


# ------------------------------------------------------------------- replay


def test_replay_requeues_and_clears_the_dlq(store: Store, clock) -> None:
    store.enqueue({"a": 1}, task_id="replay-me", max_attempts=0)
    task = store.lease_next_task("w1")
    store.dead_letter(task.id, "w1", "boom")

    replayed = store.replay_dead_letter("replay-me")

    assert replayed is not None
    assert replayed.id == "replay-me"
    assert replayed.state is TaskStatus.PENDING
    assert replayed.attempts == 0
    assert replayed.available_at == clock.now()
    assert replayed.lease_owner is None
    assert store.list_dead_letters() == []
    assert store.count_tasks() == 1


def test_replay_preserves_identity_payload_queue_and_priority(store: Store) -> None:
    store.enqueue({"to": "a@b.c"}, task_id="k", queue="mail", priority=9)
    original = store.lease_next_task("w1")
    store.dead_letter(original.id, "w1", "smtp down")

    replayed = store.replay_dead_letter("k")

    # Same id is the whole point: a replayed task must dedupe against the side
    # effects of its earlier attempts.
    assert replayed.id == "k"
    assert replayed.decode_payload() == {"to": "a@b.c"}
    assert replayed.queue == "mail"
    assert replayed.priority == 9
    # created_at is enqueued_at, so a handler sees the original enqueue time.
    assert replayed.created_at == original.created_at


def test_replay_records_provenance_in_last_error(store: Store) -> None:
    store.enqueue({"a": 1}, task_id="k")
    task = store.lease_next_task("w1")
    store.dead_letter(task.id, "w1", "smtp down")

    replayed = store.replay_dead_letter("k")

    # The DLQ row is gone, so the reason for the original failure would
    # otherwise be lost entirely.
    assert "smtp down" in replayed.last_error
    assert replayed.context().last_error == replayed.last_error


def test_replay_defaults_max_attempts_from_the_store(store: Store) -> None:
    store.enqueue({"a": 1}, task_id="k")
    task = store.lease_next_task("w1")
    store.dead_letter(task.id, "w1", "boom")

    assert store.replay_dead_letter("k").max_attempts == DEFAULT_MAX_ATTEMPTS


def test_replay_grants_a_fresh_attempt_budget(store: Store) -> None:
    """Carrying the old attempt count over would re-DLQ it on first failure."""
    store.enqueue({"a": 1}, task_id="k", max_attempts=1)
    task = store.lease_next_task("w1")
    store.dead_letter(task.id, "w1", "boom")

    replayed = store.replay_dead_letter("k", max_attempts=3)
    assert replayed.max_attempts == 3

    # Two further failures must leave it still queued, not dead again.
    for expected in (1, 2):
        claimed = store.lease_next_task("w2")
        assert claimed.attempts == expected
        store.fail_task(claimed.id, "w2", "still broken")
    assert store.find_task("k") is not None

    third = store.lease_next_task("w2")
    assert third.attempts == 3


def test_replay_unknown_id_returns_none(store: Store) -> None:
    assert store.replay_dead_letter("never-existed") is None


def test_replay_refuses_when_that_id_is_already_active(store: Store) -> None:
    """Replaying must never clobber an active task."""
    store.enqueue({"v": 1}, task_id="dup", max_attempts=0, priority=4)
    task = store.lease_next_task("w1")
    store.dead_letter(task.id, "w1", "first failure")
    store.enqueue({"v": 2}, task_id="dup", priority=6)

    with pytest.raises(QueueingError, match="already active"):
        store.replay_dead_letter("dup")

    # Neither side was touched, and the DLQ row was not consumed.
    assert store.find_task("dup").decode_payload() == {"v": 2}
    assert store.find_task("dup").priority == 6
    assert len(store.list_dead_letters()) == 1
    assert store.list_dead_letters()[0]["priority"] == 4
    assert store._conn.in_transaction is False


def test_replay_is_transactional_on_failure(store: Store) -> None:
    """A failure after the task insert must roll it back and keep the DLQ row."""
    store.enqueue({"a": 1}, task_id="k", max_attempts=0)
    task = store.lease_next_task("w1")
    store.dead_letter(task.id, "w1", "boom")

    real_conn = store._conn
    store._conn = _ExplodingConn(real_conn, "DELETE FROM dead_letter_queue")
    try:
        with pytest.raises(RuntimeError, match="disk exploded"):
            store.replay_dead_letter("k")
    finally:
        store._conn = real_conn

    assert store.find_task("k") is None
    assert len(store.list_dead_letters()) == 1
    assert store._conn.in_transaction is False


def test_replay_can_target_a_different_queue(store: Store) -> None:
    store.enqueue({"a": 1}, task_id="k", queue="mail")
    task = store.lease_next_task("w1")
    store.dead_letter(task.id, "w1", "boom")

    assert store.replay_dead_letter("k", queue="mail-bulk").queue == "mail-bulk"


def test_replayed_task_is_claimable_and_runs_again(store: Store) -> None:
    store.enqueue({"a": 1}, task_id="k", max_attempts=0)
    task = store.lease_next_task("w1")
    store.dead_letter(task.id, "w1", "boom")
    assert store.lease_next_task("w2") is None

    store.replay_dead_letter("k")

    claimed = store.lease_next_task("w2")
    assert claimed is not None and claimed.id == "k"
    assert claimed.attempts == 1
    assert claimed.state is TaskStatus.RUNNING


def test_replay_cannot_be_nested_inside_a_transaction(store: Store) -> None:
    """transaction() refuses nesting, so an outer block can't be committed early."""
    store.enqueue({"a": 1}, task_id="k", max_attempts=0)
    task = store.lease_next_task("w1")
    store.dead_letter(task.id, "w1", "boom")

    with pytest.raises(RuntimeError, match="cannot be nested"):
        with transaction(store._conn):
            store.replay_dead_letter("k")
