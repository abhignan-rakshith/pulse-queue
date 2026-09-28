#!/usr/bin/env python3
"""Standalone pulse-queue lifecycle demo.

Runs five tasks through a two-worker pool against a temporary WAL-mode
SQLite database, then prints the final task states and the dead-letter
queue:

* ``send_email``     -- succeeds cleanly                  -> COMPLETED
* ``permanent_fail`` -- raises PermanentTaskError        -> FAILED
* ``flaky_service``  -- raises RuntimeError with
  ``max_attempts=1`` -- exhausts its attempt budget      -> dead-lettered

Usage::

    uv run python demo.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import time
from typing import Any

from pulse_queue import (
    BackoffPolicy,
    HandlerRegistry,
    PermanentTaskError,
    Store,
    TaskStatus,
    WorkerPool,
)

# The demo presents outcomes through its own handler prints and result
# tables; the library's worker log lines would only duplicate them (and land
# before the header, since stderr is unbuffered while stdout is not).
logging.getLogger("pulse_queue.worker").setLevel(logging.CRITICAL)

#: (task_id, payload, priority, max_attempts) for the five demo tasks.
#: Distinct idempotency keys and distinct priorities; ``max_attempts=None``
#: means the store default (3). The flaky tasks budget exactly one attempt,
#: so their first failure dead-letters them immediately.
TASKS: list[tuple[str, dict[str, Any], int, int | None]] = [
    ("email-ada", {"type": "send_email", "to": "ada@example.com"}, 10, None),
    ("email-grace", {"type": "send_email", "to": "grace@example.com"}, 5, None),
    (
        "perm-bad-dom",
        {"type": "permanent_fail", "to": "no-such@example.invalid"},
        7,
        None,
    ),
    ("flaky-one", {"type": "flaky_service", "url": "https://flaky.example"}, 3, 1),
    (
        "flaky-two",
        {"type": "flaky_service", "url": "https://flaky.example/2"},
        1,
        1,
    ),
]


# --------------------------------------------------------------------- handlers


def build_registry() -> HandlerRegistry:
    """Register one handler per task type, each demonstrating one outcome."""
    registry = HandlerRegistry()

    @registry.register("send_email")
    async def send_email(ctx) -> str:
        to = ctx.payload["to"]
        print(f"  send_email: delivering to {to} (attempt {ctx.attempt})")
        return f"delivered to {to}"

    @registry.register("permanent_fail")
    async def permanent_fail(ctx) -> None:
        to = ctx.payload["to"]
        print(f"  permanent_fail: refusing {to!r} (attempt {ctx.attempt})")
        raise PermanentTaskError(
            "recipient domain does not exist; retrying cannot help"
        )

    @registry.register("flaky_service")
    async def flaky_service(ctx) -> None:
        url = ctx.payload["url"]
        print(f"  flaky_service: calling {url} (attempt {ctx.attempt})")
        raise RuntimeError("upstream connection reset by peer")

    return registry


# ------------------------------------------------------------------ table rendering


def _ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a fixed-width ASCII table with a header rule."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def sep() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def line(cells: list[str]) -> str:
        padded = (cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
        return "| " + " | ".join(padded) + " |"

    lines = [sep(), line(headers), sep()]
    lines.extend(line(row) for row in rows)
    lines.append(sep())
    return "\n".join(lines)


# ------------------------------------------------------------------------- demo


async def _wait_until_settled(pool: WorkerPool, *, timeout: float = 30.0) -> None:
    """Block until no task is PENDING, RUNNING, or RETRY.

    A task in one of those states is still scheduled or in flight; once the
    count hits zero, every task has reached a terminal state (COMPLETED or
    FAILED) or left the ``tasks`` table for the dead-letter queue.
    """
    active = (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.RETRY)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        remaining = sum(
            [await pool.async_store.count_tasks(state) for state in active]
        )
        if remaining == 0:
            return
        if loop.time() >= deadline:
            raise TimeoutError(
                f"{remaining} task(s) still active after {timeout:.1f}s"
            )
        await asyncio.sleep(0.02)


def _print_results(store: Store, pool: WorkerPool) -> None:
    """Print the final task states, the DLQ contents, and pool stats."""
    dlq = {entry["id"]: entry for entry in store.list_dead_letters()}

    rows: list[list[str]] = []
    for task_id, payload, _priority, _max_attempts in TASKS:
        task = store.find_task(task_id)
        if task is not None:
            rows.append(
                [
                    task.id,
                    payload["type"],
                    str(task.priority),
                    str(task.attempts),
                    task.state.value,
                    task.last_error or "-",
                ]
            )
        else:
            # Dead-lettered tasks are deleted from ``tasks``; their final
            # record lives in the DLQ instead.
            entry = dlq[task_id]
            rows.append(
                [
                    task_id,
                    payload["type"],
                    str(entry["priority"]),
                    str(entry["attempts"]),
                    "DEAD_LETTERED",
                    entry["last_error"],
                ]
            )

    print()
    print("FINAL TASK STATES")
    print(
        _ascii_table(
            ["ID", "TYPE", "PRIORITY", "ATTEMPTS", "STATE", "LAST ERROR"],
            rows,
        )
    )

    print()
    print("DEAD LETTER QUEUE")
    if dlq:
        dlq_rows = [
            [
                entry["id"],
                json.loads(entry["payload"])["type"],
                entry["queue"],
                str(entry["priority"]),
                str(entry["attempts"]),
                time.strftime("%H:%M:%S", time.localtime(entry["failed_at"])),
                entry["last_error"],
            ]
            for entry in sorted(dlq.values(), key=lambda e: e["id"])
        ]
        print(
            _ascii_table(
                [
                    "ID", "TYPE", "QUEUE", "PRIORITY",
                    "ATTEMPTS", "FAILED AT", "LAST ERROR",
                ],
                dlq_rows,
            )
        )
    else:
        print("(empty)")

    stats = pool.stats
    print()
    print(
        "pool stats: "
        f"leased={stats['leased']} completed={stats['completed']} "
        f"retried={stats['retried']} dead_lettered={stats['dead_lettered']} "
        f"failed_permanently={stats['failed_permanently']}"
    )


async def main() -> int:
    # A real file, not :memory: -- WAL is a persistent property of the file,
    # and an in-memory database silently uses journal_mode=memory instead.
    tmp = tempfile.NamedTemporaryFile(
        prefix="pulse-demo-", suffix=".db", delete=False
    )
    db_path = tmp.name
    tmp.close()

    store = Store(db_path)
    try:
        # WAL must survive: it is what gives the demo real cross-process
        # locking semantics rather than an in-memory shortcut.
        assert store.journal_mode() == "wal", "WAL mode was not enabled"

        print("pulse-queue lifecycle demo")
        print("==========================")
        print(f"database: {db_path}")
        print(f"journal mode: {store.journal_mode()}")

        for task_id, payload, priority, max_attempts in TASKS:
            store.enqueue(
                payload, task_id=task_id, priority=priority, max_attempts=max_attempts
            )
        print(f"enqueued {len(TASKS)} tasks (pool concurrency=2)\n")

        pool = WorkerPool(
            store,
            build_registry(),
            concurrency=2,
            poll_interval=0.01,
            grace_period=5.0,
            reclaim_enabled=True,
            reclaim_interval=0.05,
            # No retries are expected (the flaky tasks budget exactly one
            # attempt), but keep any scheduled retry delay tiny anyway.
            backoff=BackoffPolicy(base=0.05, jitter=0.0),
        )

        try:
            await pool.start()
            await _wait_until_settled(pool, timeout=30.0)
        finally:
            pool.stop()
            await pool.aclose()

        _print_results(store, pool)

        # Self-check: the demo is only a success if every task reached the
        # terminal state its handler was designed to produce.
        assert store.count_tasks(TaskStatus.COMPLETED) == 2
        assert store.count_tasks(TaskStatus.FAILED) == 1
        assert len(store.list_dead_letters()) == 2
        print("\nOK: 2 completed, 1 failed permanently, 2 dead-lettered.")
    finally:
        store.close()
        for suffix in ("", "-wal", "-shm"):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(db_path + suffix)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
