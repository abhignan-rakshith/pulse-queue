"""pulse-queue: a SQLite-backed asyncio task queue.

Delivery semantics
------------------
**At-least-once.** A task may run more than once if a worker crashes, if its
lease expires mid-handler, or if shutdown abandons it after the grace period.
There is no exactly-once guarantee, by design.

Deduplicate on :attr:`~pulse_queue.models.TaskContext.task_id`. It is
generated client-side at enqueue time and never reassigned, so retries,
dead-lettering, and replay all preserve it -- which is what makes a replayed
task collide with the earlier attempt's side effects rather than duplicating
them.
"""

from .backoff import BackoffPolicy, should_retry
from .clock import Clock, FrozenClock, SystemClock
from .errors import (
    PermanentTaskError,
    PulseQueueError,
    QueueingError,
    RetryableTaskError,
    StoreError,
    UnknownTaskType,
)
from .models import Task, TaskContext, TaskStatus
from .store import Store
from .worker import (
    AsyncStore,
    Handler,
    HandlerRegistry,
    Worker,
    WorkerPool,
    WorkerStats,
    payload_type_resolver,
)

__version__ = "0.1.0"

__all__ = [
    "AsyncStore",
    "BackoffPolicy",
    "Clock",
    "FrozenClock",
    "Handler",
    "HandlerRegistry",
    "PermanentTaskError",
    "PulseQueueError",
    "QueueingError",
    "RetryableTaskError",
    "Store",
    "StoreError",
    "SystemClock",
    "Task",
    "TaskContext",
    "TaskStatus",
    "UnknownTaskType",
    "Worker",
    "WorkerPool",
    "WorkerStats",
    "__version__",
    "payload_type_resolver",
    "should_retry",
]
