"""Worker engine: execution, failure routing, concurrency, shutdown.

Test discipline: once a pool is running, its store is being touched from
threads via ``asyncio.to_thread``. Assertions must therefore go through
``pool.async_store`` (which shares the serializing lock) rather than reaching
for the raw synchronous ``store`` from the event loop thread. Direct ``store``
access is only safe before a pool starts or after it has fully stopped.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time

import pytest

from pulse_queue import (
    AsyncStore,
    HandlerRegistry,
    PermanentTaskError,
    RetryableTaskError,
    StoreError,
    TaskStatus,
    UnknownTaskType,
    Worker,
    WorkerPool,
)
from pulse_queue.backoff import BackoffPolicy

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------- helpers


async def wait_until(predicate, *, timeout: float = 3.0, interval: float = 0.005) -> None:
    """Poll an async ``predicate`` until it returns truthy, or fail.

    The predicate must be a zero-arg async callable, so that any store access
    inside it is awaited (and therefore serialized) rather than compared as a
    bare coroutine.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"condition not met within {timeout}s")


async def wait_for_completed(pool: WorkerPool, n: int, *, timeout: float = 3.0) -> None:
    """Wait until ``pool`` has completed at least ``n`` tasks."""

    async def check() -> bool:
        return await pool.async_store.count_tasks(TaskStatus.COMPLETED) >= n

    await wait_until(check, timeout=timeout)


def make_registry(**handlers) -> HandlerRegistry:
    registry = HandlerRegistry()
    for task_type, handler in handlers.items():
        registry.register(task_type, handler)
    return registry


@pytest.fixture
def calls() -> list:
    return []


# ------------------------------------------------------------------ execution


async def test_successful_execution_end_to_end(store, calls) -> None:
    async def handle(ctx):
        calls.append((ctx.task_id, ctx.attempt, ctx.payload))

    store.enqueue({"type": "greet", "name": "ada"}, task_id="job-1")
    worker = Worker(store, make_registry(greet=handle), worker_id="w1")

    assert await worker.run_once() is True

    assert calls == [("job-1", 1, {"type": "greet", "name": "ada"})]
    assert store.get_task("job-1").state is TaskStatus.COMPLETED
    assert worker.stats.completed == 1
    assert worker.stats.leased == 1


async def test_run_once_returns_false_when_empty(store) -> None:
    worker = Worker(store, make_registry(), worker_id="w1")
    assert await worker.run_once() is False
    assert worker.stats.leased == 0


async def test_context_exposes_stable_idempotency_key(store) -> None:
    """Retries must present the same task_id so handlers can dedupe."""
    seen: list[str] = []

    async def handle(ctx):
        seen.append(ctx.task_id)
        raise RetryableTaskError("try again")

    store.enqueue({"type": "flaky"}, task_id="stable-key", max_attempts=2)
    worker = Worker(store, make_registry(flaky=handle), worker_id="w1",
                    backoff=BackoffPolicy(base=0.0, jitter=0.0))

    await worker.run_once()
    await worker.run_once()

    assert seen == ["stable-key", "stable-key"]


async def test_attempt_counter_increments_across_retries(store) -> None:
    """ctx.attempt is the 1-based count, and it advances on each retry."""
    attempts: list[int] = []

    async def handle(ctx):
        attempts.append(ctx.attempt)
        if ctx.attempt < 3:
            raise RetryableTaskError("not yet")

    store.enqueue({"type": "watch"}, task_id="t1", max_attempts=5)
    worker = Worker(
        store, make_registry(watch=handle), worker_id="w1",
        backoff=BackoffPolicy(base=0.0, jitter=0.0),
    )

    while await worker.run_once():
        pass

    assert attempts == [1, 2, 3]
    assert store.get_task("t1").state is TaskStatus.COMPLETED


async def test_worker_drains_multiple_tasks(store, calls) -> None:
    async def handle(ctx):
        calls.append(ctx.payload["n"])

    for i in range(5):
        store.enqueue({"type": "count", "n": i})

    worker = Worker(store, make_registry(count=handle), worker_id="w1")
    while await worker.run_once():
        pass

    assert sorted(calls) == [0, 1, 2, 3, 4]
    assert store.count_tasks(TaskStatus.COMPLETED) == 5


async def test_handler_may_return_a_value(store) -> None:
    """Handler return values are discarded, not persisted."""
    results: list = []

    async def handle(ctx):
        results.append(42)
        return 42

    store.enqueue({"type": "answer"}, task_id="t1")
    worker = Worker(store, make_registry(answer=handle), worker_id="w1")
    assert await worker.run_once() is True
    assert results == [42]
    assert store.get_task("t1").state is TaskStatus.COMPLETED


# ------------------------------------------------------------ failure routing


async def test_retryable_failure_schedules_retry_with_backoff(store, clock) -> None:
    async def handle(ctx):
        raise RetryableTaskError("transient")

    store.enqueue({"type": "flaky"}, task_id="t1")
    worker = Worker(
        store,
        make_registry(flaky=handle),
        worker_id="w1",
        backoff=BackoffPolicy(base=4.0, jitter=0.0),
    )

    await worker.run_once()

    task = store.get_task("t1")
    assert task.state is TaskStatus.RETRY
    assert task.lease_owner is None
    assert task.available_at == clock.now() + 4.0  # base * 2**0
    assert "RetryableTaskError" in task.last_error
    assert worker.stats.retried == 1


async def test_backoff_grows_across_attempts(store, clock) -> None:
    async def handle(ctx):
        raise RetryableTaskError("nope")

    store.enqueue({"type": "flaky"}, task_id="t1", max_attempts=5)
    worker = Worker(
        store,
        make_registry(flaky=handle),
        worker_id="w1",
        backoff=BackoffPolicy(base=1.0, jitter=0.0),
    )

    delays = []
    for _ in range(4):
        before = clock.now()
        await worker.run_once()
        delays.append(store.get_task("t1").available_at - before)
        clock.advance(store.get_task("t1").available_at - clock.now())

    assert delays == [1.0, 2.0, 4.0, 8.0]


async def test_retry_becomes_claimable_again(store, clock) -> None:
    async def handle(ctx):
        raise RetryableTaskError("transient")

    store.enqueue({"type": "flaky"}, task_id="t1")
    worker = Worker(
        store, make_registry(flaky=handle), worker_id="w1",
        backoff=BackoffPolicy(base=10.0, jitter=0.0),
    )
    await worker.run_once()

    # Not yet due.
    assert await worker.run_once() is False

    clock.advance(10.0)
    assert await worker.run_once() is True  # attempt 2 ran


async def test_generic_exception_is_treated_as_retryable(store) -> None:
    async def handle(ctx):
        raise ValueError("unexpected")

    store.enqueue({"type": "boom"}, task_id="t1")
    worker = Worker(store, make_registry(boom=handle), worker_id="w1")
    await worker.run_once()

    assert store.get_task("t1").state is TaskStatus.RETRY


async def test_exhausted_retries_route_to_dlq(store, clock) -> None:
    """A task lands in the DLQ on exactly its final permitted attempt."""
    attempts = []

    async def handle(ctx):
        attempts.append(ctx.attempt)
        raise RetryableTaskError(f"fail {ctx.attempt}")

    store.enqueue({"type": "doomed"}, task_id="t1", max_attempts=3)
    worker = Worker(
        store, make_registry(doomed=handle), worker_id="w1",
        backoff=BackoffPolicy(base=0.0, jitter=0.0),
    )

    while await worker.run_once():
        pass

    assert attempts == [1, 2, 3]  # exactly max_attempts attempts
    assert store.find_task("t1") is None
    entries = store.list_dead_letters()
    assert len(entries) == 1
    assert entries[0]["id"] == "t1"
    assert entries[0]["attempts"] == 3
    assert "fail 3" in entries[0]["last_error"]
    assert worker.stats.dead_lettered == 1
    assert worker.stats.retried == 2


async def test_permanent_error_goes_to_failed_not_dlq(store) -> None:
    async def handle(ctx):
        raise PermanentTaskError("payload is malformed")

    store.enqueue({"type": "bad"}, task_id="t1")
    worker = Worker(store, make_registry(bad=handle), worker_id="w1")
    await worker.run_once()

    task = store.get_task("t1")
    assert task.state is TaskStatus.FAILED
    assert task.finished_at is not None
    assert "PermanentTaskError" in task.last_error
    assert store.list_dead_letters() == []
    assert worker.stats.failed_permanently == 1


@pytest.mark.parametrize("max_attempts", [0, 1, 2, 4])
async def test_max_attempts_counts_total_attempts_not_extra_retries(
    store, max_attempts
) -> None:
    """Pins the semantics: ``max_attempts`` is the total attempt budget.

    ``max_attempts=3`` means three attempts total, *not* "three retries". The
    budget is floored at 1, because a task cannot be observed to fail without
    being attempted once -- so ``max_attempts=0`` and ``max_attempts=1`` are
    indistinguishable, and both yield exactly one attempt before the DLQ.
    """
    attempts = 0

    async def handle(ctx):
        nonlocal attempts
        attempts += 1
        raise RetryableTaskError("always fails")

    store.enqueue({"type": "doomed"}, task_id="t1", max_attempts=max_attempts)
    worker = Worker(
        store, make_registry(doomed=handle), worker_id="w1",
        backoff=BackoffPolicy(base=0.0, jitter=0.0),
    )

    while await worker.run_once():
        pass

    expected_attempts = max(max_attempts, 1)
    assert attempts == expected_attempts

    entries = store.list_dead_letters()
    assert len(entries) == 1
    assert entries[0]["attempts"] == expected_attempts


async def test_permanent_error_does_not_burn_retries(store) -> None:
    async def handle(ctx):
        raise PermanentTaskError("nope")

    store.enqueue({"type": "bad"}, task_id="t1", max_attempts=5)
    worker = Worker(store, make_registry(bad=handle), worker_id="w1")
    await worker.run_once()

    # Terminal after one attempt, not five.
    assert store.get_task("t1").attempts == 1
    assert store.get_task("t1").state is TaskStatus.FAILED
    assert await worker.run_once() is False


async def test_unknown_task_type_fails_permanently(store) -> None:
    store.enqueue({"type": "no-such-handler"}, task_id="t1")
    worker = Worker(store, make_registry(other=asyncio.sleep), worker_id="w1")

    assert await worker.run_once() is True

    task = store.get_task("t1")
    assert task.state is TaskStatus.FAILED
    assert "UnknownTaskType" in task.last_error
    assert worker.stats.failed_permanently == 1


async def test_payload_without_type_key_fails_permanently(store) -> None:
    store.enqueue({"nope": 1}, task_id="t1")
    worker = Worker(store, make_registry(), worker_id="w1")
    await worker.run_once()
    assert store.get_task("t1").state is TaskStatus.FAILED


async def test_handler_timeout_routes_to_retry(store) -> None:
    async def handle(ctx):
        await asyncio.sleep(10)

    store.enqueue({"type": "hang"}, task_id="t1")
    worker = Worker(
        store, make_registry(hang=handle), worker_id="w1", handler_timeout=0.05
    )

    await worker.run_once()

    task = store.get_task("t1")
    assert task.state is TaskStatus.RETRY
    assert "TimeoutError" in task.last_error


async def test_failing_handler_does_not_kill_the_worker(store, calls) -> None:
    async def handle(ctx):
        calls.append(ctx.payload["n"])
        if ctx.payload["n"] == 0:
            raise RuntimeError("first one fails")

    for i in range(3):
        store.enqueue({"type": "mixed", "n": i})

    worker = Worker(store, make_registry(mixed=handle), worker_id="w1")
    while await worker.run_once():
        pass

    # Claim order among equal-priority tasks falls back to the random uuid, so
    # completion order is not enqueue order.
    assert sorted(calls) == [0, 1, 2]
    assert store.count_tasks(TaskStatus.COMPLETED) == 2
    assert store.count_tasks(TaskStatus.RETRY) == 1


# --------------------------------------------------------------- lease safety


async def test_lease_lost_mid_handler_discards_result(store, clock) -> None:
    """A reclaimed task must not be overwritten by the original worker."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def handle(ctx):
        started.set()
        await release.wait()

    store.enqueue({"type": "slow"}, task_id="t1")
    async_store = AsyncStore(store)
    worker = Worker(
        async_store, make_registry(slow=handle), worker_id="w1",
        lease_ttl=10.0, heartbeat_interval=5.0,
    )

    running = asyncio.create_task(worker.run_once())
    await asyncio.wait_for(started.wait(), 2.0)

    # Lease expires and the task is reclaimed by someone else.
    clock.advance(11.0)
    assert await async_store.reclaim_expired() == 1

    release.set()
    await asyncio.wait_for(running, 2.0)

    # The stale worker's success was rejected; the task is queued for retry.
    assert store.get_task("t1").state is TaskStatus.RETRY
    assert worker.stats.lease_lost == 1
    assert worker.stats.completed == 0


async def test_heartbeat_keeps_long_handler_alive(store) -> None:
    """A handler outliving lease_ttl must survive on heartbeats alone."""
    renewals = 0

    class CountingStore(AsyncStore):
        async def renew_lease(self, task_id, worker_id, lease_epoch, lease_ttl):
            nonlocal renewals
            renewals += 1
            return await super().renew_lease(
                task_id, worker_id, lease_epoch, lease_ttl
            )

    async def handle(ctx):
        await asyncio.sleep(0.35)

    store.enqueue({"type": "slow"}, task_id="t1")
    worker = Worker(
        CountingStore(store),
        make_registry(slow=handle),
        worker_id="w1",
        lease_ttl=0.15,          # shorter than the handler
        heartbeat_interval=0.05,
    )

    await worker.run_once()

    assert renewals >= 2, f"expected repeated renewals, saw {renewals}"
    assert store.get_task("t1").state is TaskStatus.COMPLETED


async def test_store_error_does_not_kill_worker(store, clock) -> None:
    """A transient DB failure must degrade the worker, not terminate it."""
    calls = 0

    class FlakyStore(AsyncStore):
        async def lease_next_task(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise StoreError("database is locked")
            return await super().lease_next_task(*args, **kwargs)

    async def handle(ctx):
        pass

    store.enqueue({"type": "ok"}, task_id="t1")
    async_store = FlakyStore(store)
    worker = Worker(
        async_store, make_registry(ok=handle), worker_id="w1", poll_interval=0.01
    )

    stop = asyncio.Event()
    runner = asyncio.create_task(worker.run(stop))
    try:
        # Predicate goes through async_store so it is serialized with the
        # worker's own connection use.
        await wait_until(
            lambda: _state_is(async_store, "t1", TaskStatus.COMPLETED)
        )
    finally:
        stop.set()
        await asyncio.wait_for(runner, 2.0)

    assert calls >= 2


async def _state_is(async_store: AsyncStore, task_id: str, state: TaskStatus) -> bool:
    task = await async_store.find_task(task_id)
    return task is not None and task.state is state


# ---------------------------------------------------------------- registry


async def test_registry_rejects_sync_handler() -> None:
    registry = HandlerRegistry()
    with pytest.raises(TypeError, match="must be async"):

        @registry.register("sync")
        def handle(ctx):  # pragma: no cover - never invoked
            pass


async def test_registry_supports_direct_and_decorator_registration() -> None:
    async def alpha(ctx):
        pass

    async def beta(ctx):
        pass

    registry = HandlerRegistry({"alpha": alpha})
    registry.register("beta", beta)

    assert registry.resolve("alpha") is alpha
    assert registry.resolve("beta") is beta
    assert registry.types() == frozenset({"alpha", "beta"})
    assert len(registry) == 2


async def test_registry_resolve_unknown_raises() -> None:
    registry = HandlerRegistry()
    with pytest.raises(UnknownTaskType, match="no handler registered"):
        registry.resolve("missing")


async def test_async_callable_object_is_accepted() -> None:
    class Handler:
        async def __call__(self, ctx):
            return "ok"

    registry = HandlerRegistry()
    registry.register("obj", Handler())
    assert registry.resolve("obj") is not None


async def test_registry_rejects_empty_type() -> None:
    registry = HandlerRegistry()
    with pytest.raises(ValueError, match="non-empty"):
        registry.register("   ", asyncio.sleep)


# -------------------------------------------------------------- concurrency


async def test_pool_runs_each_task_exactly_once(store) -> None:
    """20 tasks, 16 concurrent slots: every task runs once, none twice."""
    seen: dict[str, int] = {}
    lock = asyncio.Lock()

    async def handle(ctx):
        async with lock:
            seen[ctx.task_id] = seen.get(ctx.task_id, 0) + 1
        await asyncio.sleep(0.01)

    for i in range(20):
        store.enqueue({"type": "work", "n": i}, task_id=f"job-{i}")

    pool = WorkerPool(
        store,
        make_registry(work=handle),
        concurrency=16,
        poll_interval=0.005,
        handle_signals=False,
        reclaim_enabled=False,
    )
    runner = asyncio.create_task(pool.run())
    try:
        await wait_for_completed(pool, 20)
    finally:
        pool.stop()
        await asyncio.wait_for(runner, 5.0)

    assert len(seen) == 20
    assert set(seen.values()) == {1}
    assert store.count_tasks(TaskStatus.COMPLETED) == 20


async def test_pool_uses_configured_concurrency(store) -> None:
    active = 0
    peak = 0
    gate = asyncio.Event()

    async def handle(ctx):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await gate.wait()
        active -= 1

    for i in range(4):
        store.enqueue({"type": "work"}, task_id=f"job-{i}")

    pool = WorkerPool(
        store,
        make_registry(work=handle),
        concurrency=4,
        poll_interval=0.005,
        handle_signals=False,
        reclaim_enabled=False,
    )
    runner = asyncio.create_task(pool.run())
    try:
        async def four_in_flight() -> bool:
            return peak == 4

        await wait_until(four_in_flight, timeout=3.0)
    finally:
        gate.set()
        pool.stop()
        await asyncio.wait_for(runner, 5.0)

    assert peak == 4


async def test_concurrency_is_bounded(store) -> None:
    """More tasks than slots must not exceed the slot count."""
    active = 0
    peak = 0

    async def handle(ctx):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1

    for i in range(12):
        store.enqueue({"type": "work"}, task_id=f"job-{i}")

    pool = WorkerPool(
        store,
        make_registry(work=handle),
        concurrency=3,
        poll_interval=0.002,
        handle_signals=False,
        reclaim_enabled=False,
    )
    runner = asyncio.create_task(pool.run())
    try:
        await wait_for_completed(pool, 12)
    finally:
        pool.stop()
        await asyncio.wait_for(runner, 5.0)

    assert peak <= 3


async def test_pool_assigns_distinct_worker_ids(store) -> None:
    pool = WorkerPool(
        store, make_registry(), concurrency=5, handle_signals=False,
        reclaim_enabled=False,
    )
    ids = [w.worker_id for w in pool.workers]
    assert len(ids) == 5
    assert len(set(ids)) == 5
    assert all(worker_id.startswith("worker-") for worker_id in ids)


async def test_pools_sharing_a_database_do_not_share_worker_ids(
    store, make_store, clock
) -> None:
    """Names must be unique across pools, not just within one.

    ``lease_owner`` is what an operator reads to find the process holding a
    task, so two pools both fielding a ``worker-0`` makes that column lie. The
    lease epoch is what actually fences a stale write; the name should not be
    ambiguous either.
    """
    other = make_store(clock=clock)
    first = WorkerPool(store, make_registry(), concurrency=3, handle_signals=False)
    second = WorkerPool(other, make_registry(), concurrency=3, handle_signals=False)

    first_ids = {w.worker_id for w in first.workers}
    second_ids = {w.worker_id for w in second.workers}
    assert len(first_ids) == 3
    assert len(second_ids) == 3
    assert not first_ids & second_ids


async def test_a_zombie_pool_cannot_overwrite_another_pools_live_lease(
    store, make_store, clock
) -> None:
    """The cross-pool failure the lease epoch exists to prevent.

    Pool A leases a task and stalls past its TTL. Pool B reclaims it and starts
    its own run while holding the new lease. Pool A then wakes and reports a
    *permanent failure*. By owner name alone that write is indistinguishable
    from pool B's own, so it would land and mark FAILED a task that is running
    fine. The bumped epoch is what separates the two generations.
    """
    a_started = asyncio.Event()
    b_started = asyncio.Event()
    release_a = asyncio.Event()
    release_b = asyncio.Event()

    async def stall_then_fail(ctx):
        a_started.set()
        await release_a.wait()
        raise PermanentTaskError("zombie: stale terminal write")

    async def stall_then_succeed(ctx):
        b_started.set()
        await release_b.wait()

    store.enqueue({"type": "job"}, task_id="contested", max_attempts=3)

    pool_a = WorkerPool(
        store, make_registry(job=stall_then_fail),
        concurrency=1, lease_ttl=10.0, heartbeat_interval=5.0,
        handle_signals=False, reclaim_enabled=False,
    )
    pool_b = WorkerPool(
        make_store(clock=clock), make_registry(job=stall_then_succeed),
        concurrency=1, lease_ttl=10.0, heartbeat_interval=5.0,
        handle_signals=False, reclaim_enabled=False,
    )
    # Reads go through a pool's AsyncStore, which serializes with its writes.
    read = pool_b.async_store.find_task

    # Pool A claims generation 1 and stalls inside the handler.
    a_run = asyncio.create_task(pool_a.workers[0].run_once())
    await asyncio.wait_for(a_started.wait(), 2.0)
    assert (await read("contested")).lease_epoch == 1

    # Its lease expires; pool B reclaims and claims generation 2.
    clock.advance(11.0)
    assert await pool_b.async_store.reclaim_expired() == 1
    b_run = asyncio.create_task(pool_b.workers[0].run_once())
    await asyncio.wait_for(b_started.wait(), 2.0)

    live = await read("contested")
    assert live.state is TaskStatus.RUNNING
    assert live.lease_epoch == 2
    assert live.attempts == 2

    # The zombie wakes and reports a permanent failure. It must not land.
    release_a.set()
    await asyncio.wait_for(a_run, 2.0)

    still_live = await read("contested")
    assert still_live.state is TaskStatus.RUNNING, "stale write clobbered a lease"
    assert still_live.lease_epoch == 2
    assert pool_a.workers[0].stats.lease_lost == 1
    assert pool_a.workers[0].stats.failed_permanently == 0

    # Pool B's run is untouched, and still completes.
    release_b.set()
    await asyncio.wait_for(b_run, 2.0)
    assert (await read("contested")).state is TaskStatus.COMPLETED
    assert pool_b.workers[0].stats.completed == 1


async def test_queue_filter_is_respected_by_pool(store) -> None:
    served = []

    async def handle(ctx):
        served.append(ctx.payload["n"])

    store.enqueue({"type": "a", "n": 1}, queue="alpha", task_id="a1")
    store.enqueue({"type": "a", "n": 2}, queue="beta", task_id="b1")

    pool = WorkerPool(
        store,
        make_registry(a=handle),
        queues=["alpha"],
        concurrency=2,
        poll_interval=0.005,
        handle_signals=False,
        reclaim_enabled=False,
    )
    runner = asyncio.create_task(pool.run())
    try:
        await wait_for_completed(pool, 1)
    finally:
        pool.stop()
        await asyncio.wait_for(runner, 5.0)

    assert served == [1]
    assert store.get_task("b1").state is TaskStatus.PENDING


# ----------------------------------------------------------- graceful shutdown


async def test_stop_waits_for_inflight_handler_to_finish(store) -> None:
    """The core graceful-shutdown guarantee.

    stop() must not abandon work that is already running.
    """
    started = asyncio.Event()
    release = asyncio.Event()
    finished = []

    async def handle(ctx):
        started.set()
        await release.wait()
        finished.append(ctx.task_id)

    store.enqueue({"type": "slow"}, task_id="t1")
    pool = WorkerPool(
        store,
        make_registry(slow=handle),
        concurrency=1,
        poll_interval=0.005,
        grace_period=5.0,
        handle_signals=False,
        reclaim_enabled=False,
    )
    runner = asyncio.create_task(pool.run())
    await asyncio.wait_for(started.wait(), 2.0)

    pool.stop()
    # Shutdown must be waiting on the handler, not racing past it.
    await asyncio.sleep(0.05)
    assert finished == []
    assert not runner.done()

    release.set()
    await asyncio.wait_for(runner, 3.0)

    assert finished == ["t1"]
    assert store.get_task("t1").state is TaskStatus.COMPLETED


async def test_stop_drains_several_inflight_handlers(store) -> None:
    """All four in-flight handlers finish, even though shutdown was requested."""
    all_started = asyncio.Event()
    release = asyncio.Event()
    started: list[str] = []
    finished: list[str] = []

    async def handle(ctx):
        started.append(ctx.task_id)
        if len(started) == 4:
            all_started.set()
        await release.wait()
        finished.append(ctx.task_id)

    for i in range(4):
        store.enqueue({"type": "slow"}, task_id=f"job-{i}")

    pool = WorkerPool(
        store,
        make_registry(slow=handle),
        concurrency=4,
        poll_interval=0.005,
        grace_period=5.0,
        handle_signals=False,
        reclaim_enabled=False,
    )
    runner = asyncio.create_task(pool.run())
    await asyncio.wait_for(all_started.wait(), 3.0)

    pool.stop()
    await asyncio.sleep(0.05)
    assert not runner.done(), "must still be draining"
    assert finished == []

    release.set()
    await asyncio.wait_for(runner, 3.0)

    assert sorted(finished) == [f"job-{i}" for i in range(4)]
    assert store.count_tasks(TaskStatus.COMPLETED) == 4


async def test_grace_period_cancels_and_releases_for_retry(store) -> None:
    """A handler that overruns the grace period is cancelled, and its task
    becomes immediately re-claimable rather than waiting out the lease TTL."""
    started = asyncio.Event()

    async def handle(ctx):
        started.set()
        await asyncio.sleep(3600)  # never finishes on its own

    store.enqueue({"type": "hang"}, task_id="t1")
    pool = WorkerPool(
        store,
        make_registry(hang=handle),
        concurrency=1,
        poll_interval=0.005,
        grace_period=0.15,
        handle_signals=False,
        reclaim_enabled=False,
    )
    runner = asyncio.create_task(pool.run())
    await asyncio.wait_for(started.wait(), 2.0)

    started_at = time.monotonic()
    pool.stop()
    await asyncio.wait_for(runner, 3.0)
    elapsed = time.monotonic() - started_at

    assert elapsed < 2.0, f"shutdown took {elapsed:.2f}s; grace period ignored"

    task = store.get_task("t1")
    assert task.state is TaskStatus.RETRY, "cancelled task must be re-queued"
    assert task.lease_owner is None
    assert "cancelled" in task.last_error
    # Immediately claimable -- no backoff wait.
    assert task.available_at <= store.clock.now()


async def test_grace_period_none_waits_indefinitely(store) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def handle(ctx):
        started.set()
        await release.wait()

    store.enqueue({"type": "slow"}, task_id="t1")
    pool = WorkerPool(
        store,
        make_registry(slow=handle),
        concurrency=1,
        poll_interval=0.005,
        grace_period=None,
        handle_signals=False,
        reclaim_enabled=False,
    )
    runner = asyncio.create_task(pool.run())
    await asyncio.wait_for(started.wait(), 2.0)

    pool.stop()
    await asyncio.sleep(0.1)
    assert not runner.done(), "grace_period=None must not cancel"

    release.set()
    await asyncio.wait_for(runner, 3.0)
    assert store.get_task("t1").state is TaskStatus.COMPLETED


async def test_zero_grace_period_cancels_immediately(store) -> None:
    started = asyncio.Event()

    async def handle(ctx):
        started.set()
        await asyncio.sleep(3600)

    store.enqueue({"type": "hang"}, task_id="t1")
    pool = WorkerPool(
        store,
        make_registry(hang=handle),
        concurrency=1,
        poll_interval=0.005,
        grace_period=0.0,
        handle_signals=False,
        reclaim_enabled=False,
    )
    runner = asyncio.create_task(pool.run())
    await asyncio.wait_for(started.wait(), 2.0)

    pool.stop()
    await asyncio.wait_for(runner, 2.0)
    assert store.get_task("t1").state is TaskStatus.RETRY


async def test_stop_is_idempotent(store) -> None:
    pool = WorkerPool(
        store, make_registry(), concurrency=1, handle_signals=False,
        reclaim_enabled=False,
    )
    pool.stop()
    pool.stop()
    await asyncio.wait_for(asyncio.create_task(pool.run()), 2.0)
    assert pool.stopping is True


async def test_stop_before_start_returns_immediately(store) -> None:
    pool = WorkerPool(
        store, make_registry(), concurrency=2, handle_signals=False,
        reclaim_enabled=False,
    )
    pool.stop()
    started_at = time.monotonic()
    await asyncio.wait_for(asyncio.create_task(pool.run()), 2.0)
    assert time.monotonic() - started_at < 1.0


async def test_worker_stops_claiming_after_stop(store) -> None:
    """No new work may be leased once shutdown has been requested."""
    processed = []

    async def handle(ctx):
        processed.append(ctx.task_id)
        await asyncio.sleep(0.02)

    pool = WorkerPool(
        store,
        make_registry(work=handle),
        concurrency=2,
        poll_interval=0.005,
        grace_period=5.0,
        handle_signals=False,
        reclaim_enabled=False,
    )
    runner = asyncio.create_task(pool.run())
    await asyncio.sleep(0.05)

    # Enqueue after shutdown is requested: must not be picked up. Routed
    # through async_store because the pool may still be mid-write on the
    # connection; a raw store call here would race it from another thread.
    pool.stop()
    await pool.async_store.enqueue({"type": "work"}, task_id="late")
    await asyncio.wait_for(runner, 3.0)

    assert "late" not in processed
    assert store.get_task("late").state is TaskStatus.PENDING


async def test_start_twice_raises(store) -> None:
    pool = WorkerPool(
        store, make_registry(), concurrency=1, handle_signals=False,
        reclaim_enabled=False,
    )
    await pool.start()
    try:
        with pytest.raises(RuntimeError, match="called twice"):
            await pool.start()
    finally:
        pool.stop()
        await pool.aclose()


async def test_aclose_is_idempotent_without_start(store) -> None:
    """aclose() before start() must not raise or hang."""
    pool = WorkerPool(
        store, make_registry(), concurrency=1, handle_signals=False,
        reclaim_enabled=False,
    )
    await asyncio.wait_for(pool.aclose(), 2.0)
    await asyncio.wait_for(pool.aclose(), 2.0)


async def test_signals_installed_by_start_not_only_run(store) -> None:
    """Embedders using start()/aclose() get signal handling too."""
    pool = WorkerPool(
        store, make_registry(), concurrency=1, handle_signals=True,
        reclaim_enabled=False,
    )
    await pool.start()
    try:
        if not hasattr(signal, "SIGTERM"):
            pytest.skip("requires a Unix platform")
        assert pool.signals_installed is True
    finally:
        pool.stop()
        await pool.aclose()
    assert pool.signals_installed is False


# -------------------------------------------------------------------- signals


@pytest.mark.skipif(
    not hasattr(signal, "SIGTERM"), reason="requires a Unix platform"
)
async def test_sigterm_triggers_graceful_shutdown(store) -> None:
    """A real SIGTERM must drain in-flight work, not abandon it."""
    started = asyncio.Event()
    release = asyncio.Event()
    finished: list[str] = []

    async def handle(ctx):
        started.set()
        await release.wait()
        finished.append(ctx.task_id)

    store.enqueue({"type": "slow"}, task_id="t1")
    pool = WorkerPool(
        store,
        make_registry(slow=handle),
        concurrency=1,
        poll_interval=0.005,
        grace_period=1.0,
        handle_signals=True,
        reclaim_enabled=False,
    )
    runner = asyncio.create_task(pool.run())
    try:
        await asyncio.wait_for(started.wait(), 2.0)
        if not pool.signals_installed:
            pytest.skip("signal handlers could not be installed in this context")

        # Handlers are confirmed in place, so this cannot kill the test run.
        os.kill(os.getpid(), signal.SIGTERM)

        async def stopping() -> bool:
            return pool.stopping

        await wait_until(stopping, timeout=2.0)
        await asyncio.sleep(0.05)
        assert not runner.done(), "grace period should still be draining"

        release.set()
        await asyncio.wait_for(runner, 3.0)
    finally:
        release.set()
        if not runner.done():
            pool.stop()
            await asyncio.gather(runner, return_exceptions=True)

    assert finished == ["t1"]
    assert store.get_task("t1").state is TaskStatus.COMPLETED
    assert pool.signals_installed is False, "handlers must be restored"


@pytest.mark.skipif(
    not hasattr(signal, "SIGINT"), reason="requires a Unix platform"
)
async def test_second_signal_skips_grace_period(store) -> None:
    started = asyncio.Event()

    async def handle(ctx):
        started.set()
        await asyncio.sleep(3600)

    store.enqueue({"type": "hang"}, task_id="t1")
    pool = WorkerPool(
        store,
        make_registry(hang=handle),
        concurrency=1,
        poll_interval=0.005,
        grace_period=30.0,          # long enough that only a 2nd signal ends it
        handle_signals=True,
        reclaim_enabled=False,
    )
    runner = asyncio.create_task(pool.run())
    try:
        await asyncio.wait_for(started.wait(), 2.0)
        if not pool.signals_installed:
            pytest.skip("signal handlers could not be installed in this context")

        os.kill(os.getpid(), signal.SIGTERM)
        async def stopping() -> bool:
            return pool.stopping

        await wait_until(stopping, timeout=2.0)

        started_at = time.monotonic()
        os.kill(os.getpid(), signal.SIGTERM)   # second signal: cancel now

        await asyncio.wait_for(runner, 3.0)
        elapsed = time.monotonic() - started_at
    finally:
        if not runner.done():
            pool.stop()
            await asyncio.gather(runner, return_exceptions=True)

    assert elapsed < 2.0, f"second signal ignored; took {elapsed:.2f}s"
    assert store.get_task("t1").state is TaskStatus.RETRY


async def test_signals_not_installed_when_disabled(store) -> None:
    pool = WorkerPool(
        store, make_registry(), concurrency=1, handle_signals=False,
        reclaim_enabled=False,
    )
    pool.stop()
    await asyncio.wait_for(asyncio.create_task(pool.run()), 2.0)
    assert pool.signals_installed is False


async def test_signal_handlers_restored_after_run(store) -> None:
    """After run() returns, no signal handler is left registered."""
    loop = asyncio.get_running_loop()

    pool = WorkerPool(
        store, make_registry(), concurrency=1, handle_signals=True,
        reclaim_enabled=False,
    )
    pool.stop()
    await asyncio.wait_for(asyncio.create_task(pool.run()), 2.0)

    assert pool.signals_installed is False
    # remove_signal_handler returns False when nothing is registered, so this
    # both asserts cleanup and leaves no trace behind.
    assert loop.remove_signal_handler(signal.SIGTERM) is False
    assert loop.remove_signal_handler(signal.SIGINT) is False


async def test_signal_handlers_restored_even_if_drain_fails(store) -> None:
    """A failure while draining must not leak handlers into the process."""
    loop = asyncio.get_running_loop()

    pool = WorkerPool(
        store, make_registry(), concurrency=1, handle_signals=True,
        reclaim_enabled=False,
    )

    async def exploding_drain() -> None:
        raise RuntimeError("drain exploded")

    pool._drain = exploding_drain  # type: ignore[method-assign]
    await pool.start()
    try:
        with pytest.raises(RuntimeError, match="drain exploded"):
            await pool.aclose()
    finally:
        pool._drain = type(pool)._drain.__get__(pool)  # type: ignore[method-assign]
        pool.stop()

    assert pool.signals_installed is False, "leaked signal handler"
    assert loop.remove_signal_handler(signal.SIGTERM) is False


# ------------------------------------------------------------------ reclaimer


async def test_reclaimer_recovers_abandoned_lease(store) -> None:
    """Work abandoned by a crashed worker is picked back up."""
    # A task leased by a worker that then vanishes, with a lease in the past.
    store.enqueue({"type": "recover"}, task_id="t1")
    store.lease_next_task("ghost-worker", lease_ttl=-1.0)
    assert store.get_task("t1").state is TaskStatus.RUNNING

    ran = asyncio.Event()

    async def handle(ctx):
        ran.set()

    pool = WorkerPool(
        store,
        make_registry(recover=handle),
        concurrency=1,
        poll_interval=0.005,
        reclaim_interval=0.02,
        handle_signals=False,
    )
    runner = asyncio.create_task(pool.run())
    try:
        await asyncio.wait_for(ran.wait(), 3.0)
        await wait_for_completed(pool, 1)
    finally:
        pool.stop()
        await asyncio.wait_for(runner, 3.0)

    assert store.get_task("t1").attempts == 2  # ghost's attempt still counted


async def test_pool_stats_aggregate_across_workers(store) -> None:
    async def handle(ctx):
        pass

    for i in range(6):
        store.enqueue({"type": "work"}, task_id=f"job-{i}")

    pool = WorkerPool(
        store,
        make_registry(work=handle),
        concurrency=3,
        poll_interval=0.005,
        handle_signals=False,
        reclaim_enabled=False,
    )
    runner = asyncio.create_task(pool.run())
    try:
        await wait_for_completed(pool, 6)
    finally:
        pool.stop()
        await asyncio.wait_for(runner, 3.0)

    assert pool.stats["completed"] == 6
    assert pool.stats["leased"] == 6


# -------------------------------------------------------------- construction


async def test_invalid_construction_rejected(store) -> None:
    registry = make_registry()
    with pytest.raises(ValueError, match="concurrency"):
        WorkerPool(store, registry, concurrency=0)
    with pytest.raises(ValueError, match="grace_period"):
        WorkerPool(store, registry, grace_period=-1.0)
    with pytest.raises(ValueError, match="reclaim_interval"):
        WorkerPool(store, registry, reclaim_interval=0)


async def test_worker_validates_lease_settings(store) -> None:
    registry = make_registry()
    with pytest.raises(ValueError, match="lease_ttl"):
        Worker(store, registry, lease_ttl=0)
    with pytest.raises(ValueError, match="heartbeat_interval"):
        Worker(store, registry, lease_ttl=10.0, heartbeat_interval=10.0)
    with pytest.raises(ValueError, match="poll_interval"):
        Worker(store, registry, poll_interval=-1)


async def test_worker_defaults_heartbeat_below_lease_ttl(store) -> None:
    worker = Worker(store, make_registry(), lease_ttl=30.0)
    assert worker.heartbeat_interval == 10.0
