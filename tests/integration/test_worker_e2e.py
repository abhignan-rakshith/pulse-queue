"""End-to-end worker shutdown, across real process lifetimes.

Everything here runs a *real* child process against a *real* WAL database and
sends it a *real* ``SIGTERM``. That is deliberately slower and blunter than the
in-process signal tests in ``tests/test_worker.py``; it covers what those
cannot:

* the database is inspected by a separate process after the child has exited,
  so results must have been durably committed rather than merely observed in
  memory;
* the graceful-shutdown path has to survive ``asyncio.run()`` teardown and
  process exit, not just an event loop;
* work left behind by a stopped process must be picked up by a *later* process,
  which is what "a deploy restarts the worker" actually looks like.

Deselect with ``-m "not integration"``.
"""

from __future__ import annotations

import asyncio
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from pulse_queue import Store, TaskStatus

pytestmark = pytest.mark.integration

#: This file is both the test and the child program.
CHILD = Path(__file__).resolve()

#: Long enough that SIGTERM lands while handlers are still running.
WORK_SECONDS = 0.4

#: Tasks seeded per test. Only the first `concurrency` can be in flight.
WORK_TASKS = 5
CONCURRENCY = 3


# ============================================================== child program
#
# Executed as `python test_worker_e2e.py <mode> <db>`. Everything below runs in
# the child process; nothing here is imported by the parent except through the
# subprocess boundary.


def _build_registry(finished: list[str]):
    from pulse_queue import HandlerRegistry

    registry = HandlerRegistry()

    @registry.register("work")
    async def work(ctx):
        await asyncio.sleep(WORK_SECONDS)
        finished.append(ctx.task_id)

    @registry.register("poison")
    async def poison(ctx):
        raise RuntimeError("always fails")

    return registry


async def _stop_when_drained(pool) -> None:
    """Set the stop event once nothing is pending, running, or retrying."""
    active = (TaskStatus.PENDING, TaskStatus.RETRY, TaskStatus.RUNNING)
    while True:
        remaining = sum(
            [await pool.async_store.count_tasks(state) for state in active]
        )
        if remaining == 0:
            pool.stop()
            return
        await asyncio.sleep(0.02)


def _child_main(mode: str, db: str) -> None:
    from pulse_queue import BackoffPolicy, Store, WorkerPool

    store = Store(db)
    finished: list[str] = []
    pool = WorkerPool(
        store,
        _build_registry(finished),
        concurrency=CONCURRENCY,
        poll_interval=0.01,
        grace_period=5.0,
        reclaim_enabled=True,
        reclaim_interval=0.05,
        # Keep the poison task's single retry from costing the suite a second.
        backoff=BackoffPolicy(base=0.05, jitter=0.0),
    )

    async def main() -> None:
        await pool.start()
        # Only now are workers live and the registry populated, so the parent
        # knows a signal sent after this point lands during real work.
        print("READY", flush=True)

        if mode == "signal":
            await pool.wait_stopped()
        else:
            await _stop_when_drained(pool)

        await pool.aclose()
        store.close()
        print(f"DRAINED:{','.join(sorted(finished))}", flush=True)

    asyncio.run(main())


if __name__ == "__main__":
    # Child entry point. Guarded so pytest can import this module for
    # collection without executing a worker pool.
    _child_main(sys.argv[1], sys.argv[2])


# ================================================================= child driver


def _wait_until(predicate, *, timeout: float, message) -> None:
    """Poll ``predicate`` until true.

    ``message`` may be a callable, and should be whenever it reads a file or
    does any other work that only makes sense at failure time -- an f-string is
    evaluated at the call site, before the wait even starts.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    text = message() if callable(message) else message
    raise AssertionError(text)


def run_child(
    mode: str, db: Path, tmp_path: Path, *, send_signal: bool
) -> tuple[float, list[str]]:
    """Run the child to completion. Returns (seconds, ids finished in-handler).

    The child's stdout is drained by a daemon thread rather than read inline:
    a child that hangs before printing ``READY`` would otherwise block
    ``readline()`` forever and wedge the whole test session. stderr goes
    straight to a file, so it can never fill a pipe buffer and deadlock either.
    """
    stderr_path = tmp_path / f"child-{mode}.stderr"

    with open(stderr_path, "w") as stderr_file:
        proc = subprocess.Popen(
            [sys.executable, str(CHILD), mode, str(db)],
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
        )

        lines: list[str] = []

        def drain() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line)

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()

        try:
            _wait_until(
                lambda: any(ln.strip() == "READY" for ln in lines),
                timeout=30.0,
                message=lambda: (
                    f"child {mode!r} never became ready; "
                    f"stdout={lines!r} stderr={stderr_path.read_text()!r}"
                ),
            )

            started = time.monotonic()
            if send_signal:
                time.sleep(0.25)  # let the handlers get into flight
                proc.send_signal(signal.SIGTERM)

            try:
                proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise AssertionError(
                    f"child {mode!r} did not exit; "
                    f"stdout={lines!r} stderr={stderr_path.read_text()!r}"
                ) from None
            elapsed = time.monotonic() - started
        finally:
            reader.join(timeout=5.0)

    stderr = stderr_path.read_text()
    assert proc.returncode == 0, (
        f"child {mode!r} exited rc={proc.returncode}\n"
        f"stdout={''.join(lines)}\nstderr={stderr}"
    )

    drained = [
        line.split(":", 1)[1]
        for line in lines
        if line.startswith("DRAINED:")
    ]
    finished = drained[0].split(",") if drained and drained[0] else []
    return elapsed, finished


def seed(db: Path) -> Store:
    """Seed a queue: work tasks plus one poison and one unhandled type."""
    store = Store(db)
    for i in range(WORK_TASKS):
        store.enqueue({"type": "work", "n": i}, task_id=f"job-{i}")
    store.enqueue({"type": "poison"}, task_id="poison-1", max_attempts=2)
    store.enqueue({"type": "never-registered"}, task_id="unknown-1")
    assert store.count_tasks() == WORK_TASKS + 2
    return store


def counts(store: Store) -> dict[str, int]:
    return {s.value: store.count_tasks(s) for s in TaskStatus if store.count_tasks(s)}


requires_unix_signals = pytest.mark.skipif(
    not hasattr(signal, "SIGTERM"), reason="requires POSIX signals"
)


# ====================================================================== tests


@requires_unix_signals
def test_sigterm_drains_in_flight_work_and_stops_claiming(tmp_path) -> None:
    """SIGTERM finishes what is running, and claims nothing new."""
    db = tmp_path / "pulse.db"
    seed(db).close()

    elapsed, finished = run_child("signal", db, tmp_path, send_signal=True)

    # Shutdown must be governed by the handler's runtime, not the grace period.
    assert elapsed < 5.0, f"shutdown took {elapsed:.2f}s"

    # Exactly the in-flight handlers finished -- not zero, and not more than
    # the pool had slots for.
    assert len(finished) == CONCURRENCY, (
        f"expected {CONCURRENCY} in-flight handlers to finish, got {finished}"
    )

    with Store(db) as store:
        per_state = counts(store)
        assert per_state.get("RUNNING", 0) == 0, "a task was abandoned RUNNING"
        assert per_state.get("COMPLETED", 0) == CONCURRENCY

        # Shutdown stops claiming: the rest of the queue is untouched, not
        # half-processed and not claimed-then-abandoned.
        assert per_state.get("PENDING", 0) == WORK_TASKS + 2 - CONCURRENCY
        assert per_state.get("RETRY", 0) == 0
        # No task was lost or duplicated: COMPLETED rows are retained, so the
        # total still accounts for every seeded task.
        assert store.count_tasks() == WORK_TASKS + 2


@requires_unix_signals
def test_sigterm_then_restart_completes_the_queue(tmp_path) -> None:
    """The full lifecycle: interrupt mid-flight, then let a new process finish.

    This is the sequence an operator actually cares about -- a restart during a
    deploy must not lose the work that was queued behind the in-flight tasks.
    """
    db = tmp_path / "pulse.db"
    seed(db).close()

    elapsed, finished = run_child("signal", db, tmp_path, send_signal=True)
    assert len(finished) == CONCURRENCY
    assert elapsed < 5.0

    # ---- a fresh process drains what the first one left behind ----
    run_child("drain", db, tmp_path, send_signal=False)

    with Store(db) as store:
        per_state = counts(store)
        assert per_state.get("RUNNING", 0) == 0
        assert per_state.get("PENDING", 0) == 0
        assert per_state.get("RETRY", 0) == 0

        # Every work task completed exactly once, across both lifetimes.
        assert per_state.get("COMPLETED", 0) == WORK_TASKS
        assert sorted(
            row[0]
            for row in store._conn.execute(
                "SELECT id FROM tasks WHERE state = 'COMPLETED'"
            )
        ) == [f"job-{i}" for i in range(WORK_TASKS)]

        # The poison task retried its one remaining attempt, then dead-lettered.
        assert per_state.get("FAILED", 0) == 1
        dead = store.list_dead_letters()
        assert [entry["id"] for entry in dead] == ["poison-1"]
        entry = dead[0]
        assert entry["attempts"] == 2
        assert "RuntimeError" in entry["last_error"]
        assert store.find_task("poison-1") is None

        # An unhandled task type is a permanent failure, retained for
        # inspection rather than silently discarded.
        unknown = store.find_task("unknown-1")
        assert unknown is not None
        assert unknown.state is TaskStatus.FAILED
        assert "UnknownTaskType" in (unknown.last_error or "")
