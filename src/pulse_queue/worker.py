"""Asyncio worker engine.

Design notes
------------
**Store calls never run on the event loop.** :class:`~pulse_queue.store.Store`
is synchronous. One sqlite3 connection cannot be used from two threads
concurrently, so :class:`AsyncStore` funnels every call through a single
``asyncio.Lock`` and executes it in a worker thread. The lock is what makes a
single shared connection safe with ``check_same_thread=False``; the thread is
what keeps a contended write (``busy_timeout`` can wait up to 5s) from freezing
the loop.

**Handlers are interrupted by lease expiry, not by cancellation.** During a
normal graceful shutdown an in-flight handler is allowed to finish. Only if it
exceeds the grace period is it cancelled, and on cancellation the worker
releases its lease so the task is immediately re-claimable rather than waiting
for the TTL to lapse. If even that write fails, the lease TTL is the backstop.
This is the at-least-once guarantee in practice: a task can run twice, so
handlers must dedupe on :attr:`~pulse_queue.models.TaskContext.task_id`.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import random
import signal
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .backoff import BackoffPolicy, should_retry
from .errors import PermanentTaskError, StoreError, UnknownTaskType
from .models import Task, TaskContext
from .store import Store

__all__ = [
    "AsyncStore",
    "Handler",
    "HandlerRegistry",
    "Worker",
    "WorkerPool",
    "WorkerStats",
    "payload_type_resolver",
]

logger = logging.getLogger("pulse_queue.worker")

#: Payload key consulted by :func:`payload_type_resolver`.
DEFAULT_TYPE_KEY = "type"


# --------------------------------------------------------------------- types


@runtime_checkable
class Handler(Protocol):
    """An async callable that performs one attempt at a task."""

    async def __call__(self, ctx: TaskContext) -> Any: ...


def payload_type_resolver(payload: Any, *, key: str = DEFAULT_TYPE_KEY) -> str:
    """Derive a task type from the payload's ``"type"`` key.

    The Phase 1 schema has no ``task_type`` column, so the type is carried in
    the payload by convention::

        queue.enqueue({"type": "send_email", "to": "ada@example.com"})

    This is the single place that convention is encoded, so promoting it to a
    real indexed column later is a change to this function plus one migration
    -- not a change to :class:`Worker`.

    Raises:
        UnknownTaskType: If the payload is not a mapping or has no usable
            ``"type"``. Treated as a permanent failure by the worker.
    """
    if isinstance(payload, Mapping):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    raise UnknownTaskType(
        f"payload must be a mapping with a non-empty {key!r} key; got {type(payload).__name__}"
    )


def _is_async_callable(fn: Any) -> bool:
    """True for coroutine functions, including objects with async ``__call__``."""
    if inspect.iscoroutinefunction(fn):
        return True
    return inspect.iscoroutinefunction(getattr(fn, "__call__", None))


class HandlerRegistry:
    """Maps task types to async handlers.

    >>> registry = HandlerRegistry()
    >>> @registry.register("add")
    ... async def add(ctx):
    ...     return ctx.payload["a"] + ctx.payload["b"]
    >>> "add" in registry
    True
    """

    __slots__ = ("_handlers",)

    def __init__(self, handlers: Mapping[str, Handler] | None = None) -> None:
        self._handlers: dict[str, Handler] = {}
        for task_type, handler in (handlers or {}).items():
            self.register(task_type, handler)

    def register(
        self, task_type: str, handler: Handler | None = None
    ) -> Callable[[Handler], Handler] | Handler:
        """Register a handler, directly or as a decorator.

        Raises:
            TypeError: If ``handler`` is not an async callable. Failing here
                means a mis-wired handler is caught at startup rather than on
                the first task that needs it.
        """
        if not task_type or not task_type.strip():
            raise ValueError("task_type must be a non-empty string")

        def _register(fn: Handler) -> Handler:
            if not _is_async_callable(fn):
                raise TypeError(
                    f"handler for {task_type!r} must be async "
                    f"(got {type(fn).__name__}); wrap blocking work in "
                    f"asyncio.to_thread"
                )
            self._handlers[task_type] = fn
            return fn

        if handler is not None:
            _register(handler)
            return handler
        return _register

    def resolve(self, task_type: str) -> Handler:
        """Look up a handler.

        Raises:
            UnknownTaskType: If nothing is registered for ``task_type``.
        """
        try:
            return self._handlers[task_type]
        except KeyError:
            known = ", ".join(sorted(self._handlers)) or "<none>"
            raise UnknownTaskType(
                f"no handler registered for task type {task_type!r}; known types: {known}"
            ) from None

    def types(self) -> frozenset[str]:
        return frozenset(self._handlers)

    def __contains__(self, task_type: object) -> bool:
        return task_type in self._handlers

    def __len__(self) -> int:
        return len(self._handlers)

    def __repr__(self) -> str:
        return f"HandlerRegistry({sorted(self._handlers)!r})"


@dataclass(slots=True)
class WorkerStats:
    """Per-worker counters, for tests and observability."""

    leased: int = 0
    completed: int = 0
    retried: int = 0
    dead_lettered: int = 0
    failed_permanently: int = 0
    lease_lost: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "leased": self.leased,
            "completed": self.completed,
            "retried": self.retried,
            "dead_lettered": self.dead_lettered,
            "failed_permanently": self.failed_permanently,
            "lease_lost": self.lease_lost,
            "errors": self.errors,
        }


# ------------------------------------------------------------- async adapter


class AsyncStore:
    """Serialized async facade over a synchronous :class:`Store`.

    Every call takes one shared lock and runs in a thread, which is what makes
    a single connection safe to share across concurrent workers.
    """

    __slots__ = ("_store", "_lock")

    def __init__(self, store: Store) -> None:
        self._store = store
        self._lock = asyncio.Lock()

    @property
    def sync(self) -> Store:
        """The underlying synchronous store, for setup and assertions."""
        return self._store

    @property
    def clock_now(self) -> float:
        """Current time from the store's injected clock."""
        return self._store.clock.now()

    async def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        async with self._lock:
            return await asyncio.to_thread(fn, *args, **kwargs)

    async def enqueue(self, *args: Any, **kwargs: Any) -> Task:
        return await self._call(self._store.enqueue, *args, **kwargs)

    async def lease_next_task(
        self, worker_id: str, *, queues: Sequence[str] | None = None, lease_ttl: float = 60.0
    ) -> Task | None:
        return await self._call(
            self._store.lease_next_task,
            worker_id,
            queues=queues,
            lease_ttl=lease_ttl,
        )

    async def complete_task(
        self, task_id: str, worker_id: str, lease_epoch: int
    ) -> bool:
        return await self._call(
            self._store.complete_task, task_id, worker_id, lease_epoch
        )

    async def fail_task(
        self,
        task_id: str,
        worker_id: str,
        lease_epoch: int,
        error: str,
        *,
        retry: bool = True,
        available_at: float | None = None,
    ) -> bool:
        return await self._call(
            self._store.fail_task,
            task_id,
            worker_id,
            lease_epoch,
            error,
            retry=retry,
            available_at=available_at,
        )

    async def dead_letter(
        self, task_id: str, worker_id: str, lease_epoch: int, error: str
    ) -> bool:
        return await self._call(
            self._store.dead_letter, task_id, worker_id, lease_epoch, error
        )

    async def renew_lease(
        self, task_id: str, worker_id: str, lease_epoch: int, lease_ttl: float
    ) -> bool:
        return await self._call(
            self._store.renew_lease, task_id, worker_id, lease_epoch, lease_ttl
        )

    async def reclaim_expired(self, *, backoff: float = 0.0) -> int:
        return await self._call(self._store.reclaim_expired, backoff=backoff)

    async def count_tasks(self, *args: Any, **kwargs: Any) -> int:
        return await self._call(self._store.count_tasks, *args, **kwargs)

    async def find_task(self, task_id: str) -> Task | None:
        return await self._call(self._store.find_task, task_id)

    async def list_dead_letters(self, limit: int = 100) -> list[dict[str, Any]]:
        return await self._call(self._store.list_dead_letters, limit)


# -------------------------------------------------------------------- worker


class Worker:
    """Leases tasks and runs their handlers, one at a time.

    A worker is a single logical slot. Concurrency comes from running several
    of them, normally via :class:`WorkerPool`.

    Failure routing:

    ===========================  ====================================
    Outcome                      Transition
    ===========================  ====================================
    handler returns              ``complete_task`` -> COMPLETED
    :class:`PermanentTaskError`  ``fail_task(retry=False)`` -> FAILED
    :class:`UnknownTaskType`     ``fail_task(retry=False)`` -> FAILED
    attempts exhausted           ``dead_letter`` -> DLQ
    any other exception          ``fail_task(retry=True)`` -> RETRY
    cancelled during shutdown    ``fail_task(retry=True)`` -> RETRY
    ===========================  ====================================
    """

    def __init__(
        self,
        store: Store | AsyncStore,
        registry: HandlerRegistry,
        *,
        worker_id: str | None = None,
        queues: Sequence[str] | None = None,
        poll_interval: float = 0.5,
        idle_backoff_max: float | None = None,
        lease_ttl: float = 60.0,
        heartbeat_interval: float | None = None,
        handler_timeout: float | None = None,
        backoff: BackoffPolicy | None = None,
        type_resolver: Callable[[Any], str] = payload_type_resolver,
        release_on_cancel: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        if poll_interval < 0:
            raise ValueError("poll_interval must be >= 0")
        if lease_ttl <= 0:
            raise ValueError("lease_ttl must be > 0")
        if handler_timeout is not None and handler_timeout <= 0:
            raise ValueError("handler_timeout must be > 0")
        if idle_backoff_max is not None and idle_backoff_max < poll_interval:
            raise ValueError("idle_backoff_max must be >= poll_interval")

        # Default heartbeat at a third of the TTL: two consecutive renewals
        # may be lost before the lease is at risk.
        resolved_heartbeat = (
            lease_ttl / 3.0 if heartbeat_interval is None else heartbeat_interval
        )
        if resolved_heartbeat <= 0:
            raise ValueError("heartbeat_interval must be > 0")
        if resolved_heartbeat >= lease_ttl:
            raise ValueError("heartbeat_interval must be < lease_ttl")

        self.store = store if isinstance(store, AsyncStore) else AsyncStore(store)
        self.registry = registry
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.queues = tuple(queues) if queues else None
        self.poll_interval = poll_interval
        self.idle_backoff_max = idle_backoff_max
        self.lease_ttl = lease_ttl
        self.heartbeat_interval = resolved_heartbeat
        self.handler_timeout = handler_timeout
        self.backoff = backoff or BackoffPolicy()
        self.type_resolver = type_resolver
        self.release_on_cancel = release_on_cancel
        self.stats = WorkerStats()
        self._rng = rng or random.Random()

    # ------------------------------------------------------------- lifecycle

    async def run(self, stop_event: asyncio.Event) -> None:
        """Claim and execute tasks until ``stop_event`` is set.

        The event is only checked *between* tasks, so a handler already in
        flight always runs to completion here. That is what makes shutdown
        graceful without a separate drain step.
        """
        idle_delay = self.poll_interval

        while not stop_event.is_set():
            try:
                processed = await self.run_once()
            except StoreError as exc:
                # A store hiccup must degrade the worker, not kill it.
                self.stats.errors += 1
                logger.warning(
                    "store error in worker %s: %s", self.worker_id, exc
                )
                await self._sleep(stop_event, self.poll_interval)
                continue

            if processed:
                idle_delay = self.poll_interval
                continue

            await self._sleep(stop_event, idle_delay)
            if self.idle_backoff_max is not None and idle_delay > 0:
                idle_delay = min(idle_delay * 2, self.idle_backoff_max)

    async def run_once(self) -> bool:
        """Lease and execute a single task. Returns True if one was processed.

        Useful on its own for deterministic tests and one-shot drains.
        """
        task = await self.store.lease_next_task(
            self.worker_id, queues=self.queues, lease_ttl=self.lease_ttl
        )
        if task is None:
            return False

        self.stats.leased += 1
        await self._execute(task)
        return True

    @staticmethod
    async def _sleep(stop_event: asyncio.Event, delay: float) -> None:
        """Sleep, but wake immediately if shutdown is requested."""
        if delay <= 0:
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=delay)

    # ------------------------------------------------------------- execution

    async def _execute(self, task: Task) -> None:
        """Run one attempt and route the outcome to the correct transition."""
        ctx = task.context()
        heartbeat = asyncio.create_task(
            self._heartbeat(task), name=f"heartbeat:{task.id[:8]}"
        )
        try:
            handler = self.registry.resolve(self.type_resolver(ctx.payload))
            await self._invoke(handler, ctx)
        except asyncio.CancelledError:
            # Shutdown ran past its grace period. Hand the task back straight
            # away rather than making the next worker wait out the lease TTL.
            await self._release_after_cancel(task)
            raise
        except PermanentTaskError as exc:
            await self._fail_permanently(task, exc)
        except Exception as exc:  # noqa: BLE001 -- deliberate catch-all
            await self._retry_or_dead_letter(task, exc)
        else:
            await self._complete(task)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await heartbeat

    async def _invoke(self, handler: Handler, ctx: TaskContext) -> Any:
        result = handler(ctx)
        if not inspect.isawaitable(result):
            raise TypeError(
                f"handler for {ctx.payload!r} returned a non-awaitable; "
                f"handlers must be async"
            )
        if self.handler_timeout is None:
            return await result
        # A timeout surfaces as TimeoutError and takes the normal retry path.
        return await asyncio.wait_for(result, timeout=self.handler_timeout)

    async def _heartbeat(self, task: Task) -> None:
        """Keep the lease alive while a long handler runs."""
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                renewed = await self.store.renew_lease(
                    task.id, self.worker_id, task.lease_epoch, self.lease_ttl
                )
            except Exception:  # noqa: BLE001 -- never fail the task over this
                logger.warning(
                    "heartbeat failed for task %s on %s", task.id, self.worker_id,
                    exc_info=True,
                )
                return
            if not renewed:
                # Someone else owns or finished this task. Anything we produce
                # from here is garbage; the fenced writes below will discard it.
                self.stats.lease_lost += 1
                logger.warning(
                    "lease lost for task %s (worker %s); result will be discarded",
                    task.id,
                    self.worker_id,
                )
                return

    # ------------------------------------------------------------ transitions

    async def _complete(self, task: Task) -> None:
        if await self.store.complete_task(task.id, self.worker_id, task.lease_epoch):
            self.stats.completed += 1
            logger.debug("completed task %s", task.id)
            return

        # The (worker_id, lease_epoch) guard rejected the write: the task was
        # reclaimed while we were running it. Dropping the result is correct.
        self.stats.lease_lost += 1
        logger.warning(
            "task %s completed by %s but the lease was lost; result discarded",
            task.id,
            self.worker_id,
        )

    async def _fail_permanently(self, task: Task, exc: BaseException) -> None:
        error = f"{type(exc).__name__}: {exc}"
        ok = await self.store.fail_task(
            task.id, self.worker_id, task.lease_epoch, error, retry=False
        )
        if ok:
            self.stats.failed_permanently += 1
            logger.error("task %s failed permanently: %s", task.id, error)
        else:
            self.stats.lease_lost += 1
            logger.warning("lease lost while failing task %s", task.id)

    async def _retry_or_dead_letter(self, task: Task, exc: BaseException) -> None:
        error = f"{type(exc).__name__}: {exc}"

        if should_retry(task.attempts, task.max_attempts):
            # task.attempts is already incremented by the claim, so this
            # attempt is index attempts-1 for backoff purposes.
            delay = self.backoff.delay_for(task.attempts - 1, rng=self._rng)
            available_at = self.store.clock_now + delay
            ok = await self.store.fail_task(
                task.id,
                self.worker_id,
                task.lease_epoch,
                error,
                retry=True,
                available_at=available_at,
            )
            if ok:
                self.stats.retried += 1
                logger.warning(
                    "task %s attempt %d/%d failed (%s); retrying in %.2fs",
                    task.id,
                    task.attempts,
                    task.max_attempts,
                    error,
                    delay,
                )
            else:
                self.stats.lease_lost += 1
                logger.warning("lease lost while retrying task %s", task.id)
            return

        if await self.store.dead_letter(
            task.id, self.worker_id, task.lease_epoch, error
        ):
            self.stats.dead_lettered += 1
            logger.error(
                "task %s exhausted %d attempt(s); dead-lettered: %s",
                task.id,
                task.attempts,
                error,
            )
        else:
            self.stats.lease_lost += 1
            logger.warning("lease lost while dead-lettering task %s", task.id)

    async def _release_after_cancel(self, task: Task) -> None:
        """Return a cancelled task to the queue immediately."""
        if not self.release_on_cancel:
            return
        try:
            released = await self.store.fail_task(
                task.id,
                self.worker_id,
                task.lease_epoch,
                "worker cancelled before completion (graceful shutdown)",
                retry=True,
                available_at=self.store.clock_now,
            )
        except Exception:  # noqa: BLE001
            # Lease TTL is the backstop; the reclaimer will pick this up.
            logger.warning(
                "could not release task %s after cancellation; "
                "it will be recovered when the lease expires",
                task.id,
                exc_info=True,
            )
            return

        if released:
            self.stats.retried += 1
            logger.info("released cancelled task %s back to the queue", task.id)
        else:
            self.stats.lease_lost += 1


# ---------------------------------------------------------------------- pool


class WorkerPool:
    """Runs N workers plus a lease reclaimer against one store.

    Usage::

        pool = WorkerPool(store, registry, concurrency=8)
        await pool.run()          # blocks until stop() or a signal

    Shutdown is two-phase: workers stop claiming, in-flight handlers are given
    ``grace_period`` seconds to finish, and only survivors are cancelled.
    """

    def __init__(
        self,
        store: Store | AsyncStore,
        registry: HandlerRegistry,
        *,
        concurrency: int = 4,
        worker_id_prefix: str = "worker",
        queues: Sequence[str] | None = None,
        poll_interval: float = 0.5,
        idle_backoff_max: float | None = None,
        lease_ttl: float = 60.0,
        heartbeat_interval: float | None = None,
        handler_timeout: float | None = None,
        backoff: BackoffPolicy | None = None,
        grace_period: float | None = 30.0,
        reclaim_enabled: bool = True,
        reclaim_interval: float = 15.0,
        type_resolver: Callable[[Any], str] = payload_type_resolver,
        release_on_cancel: bool = True,
        handle_signals: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if grace_period is not None and grace_period < 0:
            raise ValueError("grace_period must be >= 0 or None")
        if reclaim_interval <= 0:
            raise ValueError("reclaim_interval must be > 0")

        self.async_store = store if isinstance(store, AsyncStore) else AsyncStore(store)
        self.registry = registry
        self.concurrency = concurrency
        self.queues = tuple(queues) if queues else None
        self.grace_period = grace_period
        self.reclaim_enabled = reclaim_enabled
        self.reclaim_interval = reclaim_interval
        self.handle_signals = handle_signals

        # One rng per pool, so backoff jitter compares across workers.
        shared_rng = rng or random.Random()

        # Worker ids must be unique across every pool sharing a database, not
        # just within this one: lease_owner is a name, and identical names let
        # a worker whose lease was reclaimed present the string the reclaiming
        # worker holds. The lease epoch rejects that outright, but a distinct
        # name keeps lease_owner meaningful in logs and diagnostics. The token
        # mixes the process id (operator correlation) with randomness, which is
        # what actually guarantees uniqueness -- container PIDs all collide at
        # 1, and two pools in one process share a PID.
        pool_token = f"{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.workers: list[Worker] = [
            Worker(
                self.async_store,
                registry,
                worker_id=f"{worker_id_prefix}-{pool_token}-{i}",
                queues=self.queues,
                poll_interval=poll_interval,
                idle_backoff_max=idle_backoff_max,
                lease_ttl=lease_ttl,
                heartbeat_interval=heartbeat_interval,
                handler_timeout=handler_timeout,
                backoff=backoff,
                type_resolver=type_resolver,
                release_on_cancel=release_on_cancel,
                rng=shared_rng,
            )
            for i in range(concurrency)
        ]

        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._signals_installed = False
        self._signal_count = 0
        self._started = False

    # ------------------------------------------------------------- properties

    @property
    def stopping(self) -> bool:
        return self._stop_event.is_set()

    @property
    def signals_installed(self) -> bool:
        """Whether OS signal handlers are currently registered.

        Exposed so callers (and tests) can confirm handlers are in place
        *before* raising a real signal.
        """
        return self._signals_installed

    @property
    def stats(self) -> dict[str, int]:
        """Aggregate counters across all workers."""
        totals = WorkerStats()
        for worker in self.workers:
            for name, value in worker.stats.as_dict().items():
                setattr(totals, name, getattr(totals, name) + value)
        return totals.as_dict()

    # ---------------------------------------------------------------- control

    def stop(self) -> None:
        """Request a graceful shutdown. Safe to call from a signal handler."""
        if not self._stop_event.is_set():
            logger.info("shutdown requested")
        self._stop_event.set()

    async def start(self) -> None:
        """Spawn worker and reclaimer tasks, installing signal handlers.

        Non-blocking. Pair with :meth:`stop` and :meth:`aclose`.
        """
        if self._started:
            raise RuntimeError("WorkerPool.start() called twice")
        self._started = True

        if self.handle_signals:
            self._install_signals(asyncio.get_running_loop())

        for worker in self.workers:
            self._tasks.append(
                asyncio.create_task(
                    self._worker_loop(worker), name=f"pulse.{worker.worker_id}"
                )
            )
        logger.info(
            "started %d worker(s) for queues=%s",
            len(self.workers),
            list(self.queues) if self.queues else "all",
        )

        if self.reclaim_enabled:
            self._tasks.append(
                asyncio.create_task(self._reclaimer_loop(), name="pulse.reclaimer")
            )

    async def wait_stopped(self) -> None:
        """Block until :meth:`stop` has been called.

        The public counterpart to ``start()`` for embedders that own their own
        event loop and want to shut down for a reason other than a signal::

            await pool.start()
            await pool.wait_stopped()
            await pool.aclose()
        """
        await self._stop_event.wait()

    async def aclose(self) -> None:
        """Drain in-flight work, join all tasks, and restore signal handlers.

        Idempotent, and safe to call even if :meth:`start` was never called.
        """
        try:
            await self._drain()
            await self._join()
        finally:
            # Must happen even if draining raised, or a leaked handler would
            # hijack SIGTERM for the rest of the process.
            self._restore_signals(asyncio.get_running_loop())

    async def run(self) -> None:
        """Start, block until stopped, then drain and return.

        Convenience wrapper over :meth:`start` / :meth:`aclose` for the common
        case of a process whose only job is running workers. A second signal
        skips the grace period.
        """
        await self.start()
        try:
            await self._stop_event.wait()
        finally:
            await self.aclose()

    async def _drain(self) -> None:
        """Give in-flight handlers a grace period, then cancel survivors."""
        pending = [task for task in self._tasks if not task.done()]
        if not pending:
            return

        logger.info(
            "draining %d in-flight task(s) (grace=%s)",
            len(pending),
            "unbounded" if self.grace_period is None else f"{self.grace_period}s",
        )
        _, still_running = await asyncio.wait(pending, timeout=self.grace_period)

        if not still_running:
            logger.info("all in-flight task(s) finished within the grace period")
            return

        logger.warning(
            "%d task(s) exceeded the grace period; cancelling. Their leases "
            "will be released for immediate retry.",
            len(still_running),
        )
        for task in still_running:
            task.cancel()
        await asyncio.gather(*still_running, return_exceptions=True)

    async def _join(self) -> None:
        """Collect worker tasks, logging anything that escaped."""
        if not self._tasks:
            return
        # gather rather than TaskGroup: individual workers are cancelled during
        # drain, and gather(return_exceptions=True) makes those cancellations
        # plain results rather than group failures.
        results = await asyncio.gather(*self._tasks, return_exceptions=True)
        for task, result in zip(self._tasks, results, strict=True):
            if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError
            ):
                logger.error(
                    "worker task %s exited with %r", task.get_name(), result
                )

    # ------------------------------------------------------------ worker loop

    async def _worker_loop(self, worker: Worker) -> None:
        try:
            await worker.run(self._stop_event)
        except asyncio.CancelledError:
            logger.debug("worker %s cancelled", worker.worker_id)
            raise
        except Exception:  # noqa: BLE001
            logger.exception("worker %s crashed", worker.worker_id)
            raise

    async def _reclaimer_loop(self) -> None:
        """Periodically re-queue tasks whose lease expired.

        This is what recovers work after SIGKILL, a host reboot, or a handler
        that overran ``lease_ttl``.
        """
        while not self._stop_event.is_set():
            try:
                reclaimed = await self.async_store.reclaim_expired()
                if reclaimed:
                    logger.warning(
                        "reclaimed %d task(s) with expired leases", reclaimed
                    )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("lease reclaim sweep failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.reclaim_interval
                )

    # ---------------------------------------------------------------- signals

    def _install_signals(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register SIGINT/SIGTERM handlers. No-op if unsupported."""
        if not _signals_supported():
            logger.debug("signal handlers not supported here; skipping")
            return

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._on_signal, sig)
            except (NotImplementedError, RuntimeError, ValueError) as exc:
                logger.debug("could not install handler for %s: %s", sig, exc)
                return
        self._signals_installed = True

    def _restore_signals(self, loop: asyncio.AbstractEventLoop) -> None:
        if not self._signals_installed:
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(sig)
        self._signals_installed = False

    def _on_signal(self, sig: signal.Signals) -> None:
        self._signal_count += 1

        if self._signal_count == 1:
            logger.info("received %s; shutting down gracefully", sig.name)
            self.stop()
            return

        # Second signal: the operator is impatient. Skip the grace period.
        logger.warning(
            "received %s again; cancelling in-flight tasks immediately", sig.name
        )
        self.stop()
        for task in self._tasks:
            if not task.done():
                task.cancel()


def _signals_supported() -> bool:
    """Signal handlers need a Unix platform and the main thread."""
    return (
        hasattr(signal, "SIGTERM")
        and threading.current_thread() is threading.main_thread()
    )
