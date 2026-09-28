"""SQLite-backed task store.

Synchronous and free of asyncio by design: every method is a plain call that
can be exercised without an event loop. :mod:`pulse_queue.worker` wraps these
in ``asyncio.to_thread`` and serializes them behind a single lock.

Concurrency model
-----------------
One connection per :class:`Store`, guarded by an ``asyncio.Lock`` at the worker
layer. Combined with ``busy_timeout=5000`` this bounds contention and avoids
``SQLITE_BUSY`` under normal multi-worker operation against one file.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._sqlite import transaction
from .clock import Clock, SystemClock
from .errors import QueueingError, StoreError
from .models import Task, TaskStatus
from .schema import current_version, migrate

#: Applied to every connection. WAL is a persistent database property, but it
#: is not set if the database is created by a reader-only handle first.
PRAGMAS: tuple[tuple[str, str], ...] = (
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("busy_timeout", "5000"),
    ("foreign_keys", "ON"),
)

DEFAULT_MAX_ATTEMPTS = 3


class Store:
    """Transactional access to the task queue.

    Opening a store creates the database file if absent and applies any pending
    migrations, so it is always safe to construct one at process start.

    Note that WAL is a persistent property of the *file*: an in-memory database
    (``:memory:``) silently uses ``journal_mode=memory`` instead. Point this at
    a real path to get real WAL and real cross-process locking.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        clock: Clock | None = None,
        default_queue: str = "default",
        default_max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.db_path = str(db_path)
        self.clock = clock or SystemClock()
        self.default_queue = default_queue
        self.default_max_attempts = default_max_attempts

        self._conn = self._connect(self.db_path)
        migrate(self._conn)

    # ---------------------------------------------------------------- setup

    @staticmethod
    def _connect(db_path: str) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(
                db_path,
                # Transactions are explicit: see pulse_queue._sqlite. Under
                # isolation_level=None, `with conn:` does not open one.
                isolation_level=None,
                # Lets us address columns by name.
                check_same_thread=False,
                timeout=5.0,
            )
        except sqlite3.Error as exc:
            raise StoreError(f"cannot open database {db_path!r}: {exc}") from exc

        conn.row_factory = sqlite3.Row
        for pragma, value in PRAGMAS:
            # journal_mode returns a row; the rest return nothing.
            conn.execute(f"PRAGMA {pragma} = {value}")
        return conn

    def close(self) -> None:
        """Close the connection. Idempotent."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        return current_version(self._conn)

    def journal_mode(self) -> str:
        """Current journal mode, e.g. ``'wal'``. For tests and diagnostics."""
        return self._conn.execute("PRAGMA journal_mode").fetchone()[0]

    # -------------------------------------------------------------- producer

    def enqueue(
        self,
        payload: Any,
        *,
        queue: str | None = None,
        priority: int = 0,
        max_attempts: int | None = None,
        task_id: str | None = None,
        delay: float = 0.0,
    ) -> Task:
        """Insert a task and return it.

        The returned :attr:`Task.id` is the stable idempotency key. It is
        generated client-side and never reassigned, so a task that fails,
        retries, and is dead-lettered or replayed keeps the same key.

        Args:
            payload: JSON-serializable task body.
            queue: Queue name; defaults to the store's ``default_queue``.
                Must be non-empty.
            priority: Higher runs first among ready tasks.
            max_attempts: Total attempts allowed before dead-lettering. Note
                this is an *attempt* budget, not an extra-retry count:
                ``max_attempts=3`` means three attempts total. Floored at 1,
                since the first attempt always happens.
            task_id: Supply to make enqueue idempotent. Re-enqueueing the same
                ``task_id`` is a no-op that returns the existing task rather
                than raising, so a retried producer request is safe.
            delay: Seconds to hold the task in PENDING before it is claimable.

        Raises:
            StoreError: If the payload is not JSON-serializable.
            QueueingError: If the resolved queue name is empty.
        """
        now = self.clock.now()
        resolved_queue = queue or self.default_queue
        if not resolved_queue or not resolved_queue.strip():
            raise QueueingError("queue name must be a non-empty string")
        resolved_retries = (
            self.default_max_attempts if max_attempts is None else max_attempts
        )
        if resolved_retries < 0:
            raise QueueingError("max_attempts must be >= 0")

        try:
            blob = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise StoreError(f"payload is not JSON-serializable: {exc}") from exc

        # Generated client-side: this id is the handler-facing idempotency key.
        new_id = task_id or uuid.uuid4().hex

        try:
            with transaction(self._conn):
                self._conn.execute(
                    """
                    INSERT INTO tasks (
                        id, queue, payload, state, priority, attempts,
                        max_attempts, available_at, created_at, updated_at
                    ) VALUES (?, ?, ?, 'PENDING', ?, 0, ?, ?, ?, ?)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (
                        new_id,
                        resolved_queue,
                        blob,
                        priority,
                        resolved_retries,
                        now + delay,
                        now,
                        now,
                    ),
                )
        except sqlite3.Error as exc:
            raise StoreError(f"enqueue failed: {exc}") from exc

        return self.get_task(new_id)

    # -------------------------------------------------------------- consumer

    def lease_next_task(
        self,
        worker_id: str,
        *,
        queues: Sequence[str] | None = None,
        lease_ttl: float = 60.0,
    ) -> Task | None:
        """Atomically claim the next ready task, or return ``None``.

        A single ``UPDATE ... WHERE id = (SELECT ...) RETURNING *`` -- there is
        no select-then-update window, so two concurrent callers can never
        receive the same task.

        The claim increments ``attempts`` up front: a worker that crashes
        mid-handler still consumes an attempt, which is what keeps a poison
        task from looping forever.

        It also increments ``lease_epoch``, which is the fencing token for this
        lease. Every later transition must present the epoch returned here;
        see :meth:`complete_task`.

        Args:
            worker_id: Identity recorded as ``lease_owner``. Later state
                transitions require this to match, fencing out a zombie worker
                whose lease already expired and was reclaimed by someone else.
            queues: Restrict to these queues; ``None`` means all.
            lease_ttl: Seconds until the lease expires and the reclaimer can
                re-queue the task. Must exceed the expected handler runtime.
        """
        now = self.clock.now()
        queue_clause, params = _queue_filter(queues)

        try:
            with transaction(self._conn):
                row = self._conn.execute(
                    f"""
                    UPDATE tasks SET
                        state            = 'RUNNING',
                        lease_owner      = ?,
                        lease_expires_at = ?,
                        lease_epoch      = lease_epoch + 1,
                        attempts         = attempts + 1,
                        started_at       = COALESCE(started_at, ?),
                        updated_at       = ?
                    WHERE id = (
                        SELECT id FROM tasks
                        WHERE state IN ('PENDING','RETRY')
                          AND available_at <= ?
                          {queue_clause}
                        ORDER BY priority DESC, available_at, id
                        LIMIT 1
                    )
                    RETURNING *
                    """,
                    (worker_id, now + lease_ttl, now, now, now, *params),
                ).fetchone()
        except sqlite3.Error as exc:
            raise StoreError(f"lease failed: {exc}") from exc

        return Task.from_row(row) if row is not None else None

    def complete_task(self, task_id: str, worker_id: str, lease_epoch: int) -> bool:
        """Mark a task COMPLETED. Returns ``False`` if the lease was lost.

        ``(worker_id, lease_epoch)`` is the fencing token: a worker whose lease
        expired and was reclaimed gets ``False`` and must discard its result
        rather than overwrite the new owner's state. Both halves are required --
        the owner is only a name, and two pools sharing a database can field the
        same one, but the epoch is bumped on every claim and so cannot repeat.
        """
        return self._finish(task_id, worker_id, lease_epoch, TaskStatus.COMPLETED)

    def fail_task(
        self,
        task_id: str,
        worker_id: str,
        lease_epoch: int,
        error: str,
        *,
        retry: bool = True,
        available_at: float | None = None,
    ) -> bool:
        """Record a failed attempt.

        With ``retry=True`` the task moves to RETRY with ``available_at``
        controlling when it becomes claimable again. With ``retry=False`` it
        moves to FAILED and stays visible for inspection.

        Returns ``False`` if the lease was lost, in which case nothing was
        written -- the reclaimer already owns this task. See
        :meth:`complete_task` for why ``lease_epoch`` is required.
        """
        now = self.clock.now()
        if available_at is None:
            available_at = now
        status = TaskStatus.RETRY if retry else TaskStatus.FAILED

        try:
            with transaction(self._conn):
                cur = self._conn.execute(
                    """
                    UPDATE tasks SET
                        state            = ?,
                        last_error       = ?,
                        available_at     = ?,
                        lease_owner      = ?,
                        lease_expires_at = ?,
                        finished_at      = CASE WHEN ? = 'FAILED' THEN ? ELSE NULL END,
                        updated_at       = ?
                    WHERE id = ? AND lease_owner = ? AND lease_epoch = ?
                      AND state = 'RUNNING'
                    """,
                    (
                        status.value,
                        error,
                        available_at,
                        # Both RETRY and FAILED release the lease: RETRY so the
                        # claim query can pick it up, FAILED because it is
                        # terminal and must never be re-claimed.
                        None,
                        None,
                        status.value,
                        now,
                        now,
                        task_id,
                        worker_id,
                        lease_epoch,
                    ),
                )
        except sqlite3.Error as exc:
            raise StoreError(f"fail_task failed: {exc}") from exc

        return cur.rowcount == 1

    def _finish(
        self, task_id: str, worker_id: str, lease_epoch: int, status: TaskStatus
    ) -> bool:
        now = self.clock.now()
        try:
            with transaction(self._conn):
                cur = self._conn.execute(
                    """
                    UPDATE tasks SET
                        state            = ?,
                        lease_owner      = NULL,
                        lease_expires_at = NULL,
                        finished_at      = ?,
                        updated_at       = ?
                    WHERE id = ? AND lease_owner = ? AND lease_epoch = ?
                      AND state = 'RUNNING'
                    """,
                    (status.value, now, now, task_id, worker_id, lease_epoch),
                )
        except sqlite3.Error as exc:
            raise StoreError(f"state transition failed: {exc}") from exc
        return cur.rowcount == 1

    def renew_lease(
        self, task_id: str, worker_id: str, lease_epoch: int, lease_ttl: float
    ) -> bool:
        """Extend a lease held by ``worker_id``. False if the lease was lost."""
        now = self.clock.now()
        try:
            with transaction(self._conn):
                cur = self._conn.execute(
                    """
                    UPDATE tasks SET lease_expires_at = ?, updated_at = ?
                    WHERE id = ? AND lease_owner = ? AND lease_epoch = ?
                      AND state = 'RUNNING'
                    """,
                    (now + lease_ttl, now, task_id, worker_id, lease_epoch),
                )
        except sqlite3.Error as exc:
            raise StoreError(f"lease renewal failed: {exc}") from exc
        return cur.rowcount == 1

    def reclaim_expired(self, *, backoff: float = 0.0) -> int:
        """Re-queue RUNNING tasks whose lease expired. Returns the count.

        This is how work survives ``SIGKILL`` or a handler that blows past
        ``lease_ttl``. Phase 2 wires this into a periodic sweep.
        """
        now = self.clock.now()
        try:
            with transaction(self._conn):
                cur = self._conn.execute(
                    """
                    UPDATE tasks SET
                        state            = 'RETRY',
                        lease_owner      = NULL,
                        lease_expires_at = NULL,
                        available_at     = ?,
                        last_error       = COALESCE(
                            last_error, 'lease expired; task reclaimed'
                        ),
                        updated_at       = ?
                    WHERE state = 'RUNNING' AND lease_expires_at < ?
                    """,
                    (now + backoff, now, now),
                )
        except sqlite3.Error as exc:
            raise StoreError(f"reclaim failed: {exc}") from exc
        return cur.rowcount

    # ----------------------------------------------------------- dead letter

    def dead_letter(
        self,
        task_id: str,
        worker_id: str,
        lease_epoch: int,
        error: str,
    ) -> bool:
        """Move an exhausted task to the DLQ and delete it from ``tasks``.

        The insert and delete share one transaction, so the row is never
        absent from both tables nor present in both.

        ``tasks`` row count therefore always equals pending + in-flight.

        Dead-lettering is terminal and immediate, so there is no ``available_at``
        to defer and therefore no backoff to apply: the row leaves the queue
        rather than being scheduled for another attempt.

        Returns ``False`` if the lease was lost. See :meth:`complete_task` for
        why ``lease_epoch`` is required.
        """
        now = self.clock.now()
        try:
            with transaction(self._conn):
                row = self._conn.execute(
                    """
                    SELECT * FROM tasks
                    WHERE id = ? AND lease_owner = ? AND lease_epoch = ?
                      AND state = 'RUNNING'
                    """,
                    (task_id, worker_id, lease_epoch),
                ).fetchone()
                if row is None:
                    return False

                self._conn.execute(
                    """
                    INSERT INTO dead_letter_queue (
                        id, queue, payload, priority, attempts, last_error,
                        failed_at, task_created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        attempts   = excluded.attempts,
                        last_error = excluded.last_error,
                        failed_at  = excluded.failed_at,
                        priority   = excluded.priority
                    """,
                    (
                        row["id"],
                        row["queue"],
                        row["payload"],
                        row["priority"],
                        row["attempts"],
                        error,
                        now,
                        row["created_at"],
                    ),
                )
                self._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        except sqlite3.Error as exc:
            raise StoreError(f"dead-lettering failed: {exc}") from exc
        return True

    def replay_dead_letter(
        self,
        task_id: str,
        *,
        max_attempts: int | None = None,
        queue: str | None = None,
        priority: int | None = None,
    ) -> Task | None:
        """Move a dead-lettered task back into the active queue.

        The DLQ row is deleted and the task row re-inserted inside one
        transaction, so the task is never in both tables nor missing from
        both. ``id``, ``payload``, ``created_at`` and the recorded queue and
        priority are carried across unchanged, which means a replayed task
        keeps its original idempotency key and dedupes against the side effects
        of its earlier attempts.

        ``attempts`` resets to 0, because giving the task a fresh budget is the
        whole point of replaying it -- carrying the old count over would send
        it straight back to the DLQ on its next failure. The previous error is
        kept in ``last_error``, prefixed, so the replayed row still shows why it
        was dead-lettered until its next outcome overwrites it.

        Args:
            task_id: Id of the dead-lettered task.
            max_attempts: Attempt budget for the replayed task. Defaults to the
                store's ``default_max_attempts``, since the DLQ does not record
                the original budget.
            queue: Replay into a different queue.
            priority: Override the preserved priority.

        Returns:
            The re-queued task, or ``None`` if no DLQ row has that id.

        Raises:
            QueueingError: If a task with this id is already active. Nothing is
                deleted in that case -- the DLQ row is left untouched.
        """
        now = self.clock.now()
        resolved_attempts = (
            self.default_max_attempts if max_attempts is None else max_attempts
        )
        if resolved_attempts < 0:
            raise QueueingError("max_attempts must be >= 0")

        try:
            with transaction(self._conn):
                row = self._conn.execute(
                    "SELECT * FROM dead_letter_queue WHERE id = ?", (task_id,)
                ).fetchone()
                if row is None:
                    return None

                already_active = self._conn.execute(
                    "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
                ).fetchone()
                if already_active is not None:
                    raise QueueingError(
                        f"cannot replay {task_id!r}: a task with that id is "
                        f"already active"
                    )

                self._conn.execute(
                    """
                    INSERT INTO tasks (
                        id, queue, payload, state, priority, attempts,
                        max_attempts, available_at, lease_owner,
                        lease_expires_at, last_error, created_at, updated_at,
                        started_at, finished_at
                    ) VALUES (?, ?, ?, 'PENDING', ?, 0, ?, ?, NULL, NULL, ?, ?, ?, NULL, NULL)
                    """,
                    (
                        row["id"],
                        queue if queue is not None else row["queue"],
                        row["payload"],
                        priority if priority is not None else row["priority"],
                        resolved_attempts,
                        now,
                        f"replayed from DLQ; previous error: {row['last_error']}",
                        row["task_created_at"],
                        now,
                    ),
                )
                self._conn.execute(
                    "DELETE FROM dead_letter_queue WHERE id = ?", (task_id,)
                )
        except sqlite3.Error as exc:
            raise StoreError(f"replay failed: {exc}") from exc

        return self.get_task(task_id)

    # ------------------------------------------------------------- readback

    def get_task(self, task_id: str) -> Task:
        """Load one task by id.

        Raises:
            KeyError: If no such task exists.
        """
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no task with id {task_id!r}")
        return Task.from_row(row)

    def find_task(self, task_id: str) -> Task | None:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return Task.from_row(row) if row is not None else None

    def count_tasks(
        self, state: TaskStatus | None = None, *, queues: Sequence[str] | None = None
    ) -> int:
        queue_clause, params = _queue_filter(queues)
        where = [queue_clause] if queue_clause else []
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"SELECT COUNT(*) FROM tasks {clause}"
        query_params: list[Any] = list(params)
        if state is not None:
            sql += f"{' AND' if where else 'WHERE'} state = ?"
            query_params.append(state.value)
        return int(self._conn.execute(sql, query_params).fetchone()[0])

    def list_dead_letters(self, limit: int | None = 100) -> list[dict[str, Any]]:
        """Dead-lettered tasks, newest first.

        Args:
            limit: Maximum rows, or ``None`` for every row. ``None`` omits the
                clause entirely rather than binding NULL: SQLite rejects
                ``LIMIT NULL`` with "datatype mismatch".
        """
        sql = "SELECT * FROM dead_letter_queue ORDER BY failed_at DESC"
        params: list[Any] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params)
        return [dict(row) for row in rows]


def _queue_filter(queues: Sequence[str] | None) -> tuple[str, list[Any]]:
    """Build a parameterized ``AND queue IN (...)`` clause.

    Parameterized rather than interpolated so queue names are never able to
    alter the statement.
    """
    if not queues:
        return "", []
    placeholders = ",".join("?" for _ in queues)
    return f"AND queue IN ({placeholders})", list(queues)
