"""Core data types shared across the package."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TaskStatus(StrEnum):
    """Lifecycle state of a task row.

    The three *active* states are :attr:`PENDING`, :attr:`RUNNING`, and
    :attr:`RETRY` -- these are the only ones a claim query will select.

    :attr:`COMPLETED` and :attr:`FAILED` are *terminal*: written by
    :meth:`pulse_queue.store.Store.complete_task` and
    :meth:`pulse_queue.store.Store.fail_task`, never selected for re-claim.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRY = "RETRY"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


#: States a claim query may transition from.
CLAIMABLE_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.PENDING, TaskStatus.RETRY}
)

#: States after which a task is never re-queued by the worker loop.
TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED}
)


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Handed to a handler on every attempt.

    Attributes:
        task_id: Stable idempotency key. Generated client-side at
            :meth:`pulse_queue.store.Store.enqueue` and **never reassigned**.
            A task that fails, retries, and is later dead-lettered or replayed
            keeps this same value, so a handler can dedupe its own side effects
            against it. Persist it as a unique constraint or an idempotency-key
            column on any outbound call the handler makes.
        attempt: 1-based count of the current attempt. Incremented by the
            claim query, so a crashed worker still consumes an attempt. Use for
            logging and backoff decisions, not to distinguish work: a retry of
            the *same* task shares the same :attr:`task_id`.
        queue: Queue name this attempt was claimed from.
        payload: Decoded task payload.
        enqueued_at: Unix epoch seconds when the task was first enqueued.
    """

    task_id: str
    attempt: int
    queue: str
    payload: Any
    enqueued_at: float
    max_attempts: int
    #: Filled in by the store when a task is loaded; absent in unit fixtures.
    last_error: str | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class Task:
    """A row in the ``tasks`` table."""

    id: str
    queue: str
    payload: bytes
    state: TaskStatus
    priority: int
    attempts: int
    max_attempts: int
    available_at: float
    lease_owner: str | None
    lease_expires_at: float | None
    last_error: str | None
    created_at: float
    updated_at: float
    started_at: float | None
    finished_at: float | None

    @classmethod
    def from_row(cls, row: Any) -> Task:
        """Build a :class:`Task` from a :class:`sqlite3.Row`."""
        return cls(
            id=row["id"],
            queue=row["queue"],
            payload=row["payload"],
            state=TaskStatus(row["state"]),
            priority=row["priority"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            available_at=row["available_at"],
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def decode_payload(self) -> Any:
        """Decode the JSON payload.

        >>> Task(
        ...     id="t", queue="default", payload=b'{"a": 1}', state=TaskStatus.PENDING,
        ...     priority=0, attempts=0, max_attempts=3, available_at=0.0,
        ...     lease_owner=None, lease_expires_at=None, last_error=None,
        ...     created_at=0.0, updated_at=0.0, started_at=None, finished_at=None,
        ... ).decode_payload()
        {'a': 1}
        """
        return json.loads(self.payload)

    def context(self) -> TaskContext:
        """Build the :class:`TaskContext` passed to a handler."""
        return TaskContext(
            task_id=self.id,
            attempt=self.attempts,
            queue=self.queue,
            payload=self.decode_payload(),
            enqueued_at=self.created_at,
            max_attempts=self.max_attempts,
            last_error=self.last_error,
        )
