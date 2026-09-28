# pulse-queue

[![CI](https://github.com/abhignan-rakshith/pulse-queue/actions/workflows/ci.yml/badge.svg)](https://github.com/abhignan-rakshith/pulse-queue/actions/workflows/ci.yml)

A standalone background task queue built on SQLite and `asyncio`. One file on
disk is the whole broker: no Redis, no separate server process, and no runtime
dependencies beyond the standard library.

- **WAL-mode SQLite** with transactional lease and state transitions
- **Retry with exponential backoff**, then a **dead letter queue**
- **At-least-once delivery** with a stable idempotency key per task
- **Graceful shutdown** on `SIGINT`/`SIGTERM`, draining in-flight work
- **Lease expiry recovery**, so `SIGKILL` or a host reboot does not lose work
- **Fenced leases**, so a stalled worker whose lease was reclaimed cannot
  overwrite the result of whoever took over
- **DLQ replay** from the CLI, preserving the original task id

## Install

```sh
uv add pulse-queue
```

or, with pip:

```sh
pip install pulse-queue
```

Requires Python 3.12+. No runtime dependencies beyond the standard library.

## Quickstart

Define handlers against a registry, enqueue a payload, then run a pool. The
snippet below follows [`demo.py`](demo.py), which pushes five tasks through a
two-worker pool -- two successes, one permanent failure, and two that exhaust
their attempt budget -- and prints every final state plus the dead letter
queue:

```python
import asyncio

from pulse_queue import HandlerRegistry, Store, TaskStatus, WorkerPool

registry = HandlerRegistry()


@registry.register("send_email")
async def send_email(ctx) -> str:
    print(f"delivering to {ctx.payload['to']} (attempt {ctx.attempt})")
    return f"delivered to {ctx.payload['to']}"


async def main() -> None:
    store = Store("pulse.db")
    try:
        store.enqueue({"type": "send_email", "to": "ada@example.com"}, priority=10)

        pool = WorkerPool(store, registry, concurrency=2)
        await pool.start()
        try:
            active = (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.RETRY)
            pending = sum([await pool.async_store.count_tasks(s) for s in active])
            while pending:
                await asyncio.sleep(0.02)
                pending = sum([await pool.async_store.count_tasks(s) for s in active])
        finally:
            pool.stop()
            await pool.aclose()

        print("completed:", store.count_tasks(TaskStatus.COMPLETED))
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
```

A handler's outcome selects the transition:

| Outcome | Result |
| --- | --- |
| returns | `COMPLETED` |
| `PermanentTaskError` | `FAILED` (no retry) |
| anything else | `RETRY` with backoff, then `DLQ` once the budget is spent |

`demo.py` asserts those transitions at the end, so it doubles as an end-to-end
smoke test:

```sh
uv run python demo.py
```

## Architecture

Everything lives in one SQLite file. `Store` owns a single connection opened in
WAL mode (`journal_mode=wal`, `busy_timeout=5000`, foreign keys on) and applies
forward-only migrations on construction. `WorkerPool` wraps the store in
`AsyncStore`, an async facade that serializes calls through one lock and runs
them in a worker thread, so a single connection stays safe for concurrent
workers.

| Component | Role |
| --- | --- |
| `tasks` table | Canonical queue: payload, state, priority, attempt budget, availability, lease columns |
| `dead_letter_queue` table | Terminal record for tasks that spent their attempt budget; replayed by id |
| `Store` | Migrations, transactional enqueue, claim, complete/fail/retry/dead-letter |
| `Worker` / `WorkerPool` | Worker slots, lease heartbeats, expiry reclamation, graceful drain |

A claim is a single `UPDATE ... RETURNING` that selects the highest-priority
available task and stamps a fresh lease, so two workers can never be handed the
same row. The lease is a fencing token -- `(lease_owner, lease_epoch)` -- and
every transition (complete, fail, retry, dead-letter, heartbeat) must present
the epoch it was claimed with. Migrations run one transaction per version with
the version row written last, so a migration that fails rolls back completely
and is retried on the next open.

### Delivery semantics

Delivery is **at-least-once**, not exactly-once. A task can run twice if a
worker crashes, if its lease expires mid-handler, or if shutdown gives up on it
after the grace period.

`ctx.task_id` is the idempotency key that makes this safe. It is generated at
enqueue time and never reassigned, so retries, dead-lettering, and replay all
carry the same value. Persist it as a unique constraint on any outbound call
your handler makes.

### Leases, fencing, and more than one pool

A claim is a single `UPDATE ... RETURNING`, so two workers can never be handed
the same task. What the worker holds afterwards is a lease *token*:
`(lease_owner, lease_epoch)`. The epoch is a per-task counter bumped on every
claim, so it identifies the generation of the lease and not merely its holder.
Every transition -- complete, fail, retry, dead-letter, heartbeat -- must
present the epoch it was claimed with.

That is what makes running more than one pool against the same database safe.
Worker ids are names, and names repeat: two pools built from the same prefix
would both field a `worker-0`. `WorkerPool` therefore builds ids as
`worker-<pid>-<uuid6>-<i>`, unique per pool, so the column stays unambiguous --
but the epoch is what actually enforces the fence. A worker whose lease expired
is rejected even when it presents exactly the same name as its successor, and
even when it reclaims its own lease.

```python
# Two processes on one database file. One Store per process.
store_a, store_b = Store("pulse.db"), Store("pulse.db")
pool_a = WorkerPool(store_a, registry, concurrency=4)
pool_b = WorkerPool(store_b, registry, concurrency=4)
```

Set `lease_ttl` above your slowest handler's runtime; the heartbeat renews it
every third of the TTL by default.

### Schema and migrations

The schema is currently **v4**. Migrations are forward-only and applied on
`Store` construction, one transaction per migration with the version row written
last -- so a migration that fails rolls back completely and is retried on the
next open, instead of being recorded as applied with half its DDL missing.

| Version | Change |
| --- | --- |
| v1 | `tasks`, `dead_letter_queue`, and the partial claim and lease indexes |
| v2 | `max_retries` renamed to `max_attempts`; `priority` carried into the DLQ |
| v3 | dropped the never-written `revived_at` and `revive_count` DLQ columns |
| v4 | added `tasks.lease_epoch`, the monotonic lease fencing token |

Epoch `0` means "never claimed". The first claim sets it to `1`, and every later
claim of that row increments it, including a reclaim after expiry. Because the
value never repeats for a row, a stale epoch is rejected no matter which worker
presents it.

### `max_attempts` is an attempt budget

`max_attempts=3` means three attempts in total, not one attempt plus three
retries. The budget floors at 1, because a task cannot be observed to fail
without being attempted once.

## CLI

```sh
# Run workers. Handlers come from an import path, because a worker with no
# registry has nothing to run.
pulse-queue --db pulse.db work --handlers myapp.tasks --concurrency 8

# Enqueue from a shell. Prints the task id; --json gives full detail.
pulse-queue enqueue send_email '{"to": "ada@example.com"}' --priority 5
pulse-queue enqueue export '{}' --idempotency-key nightly-2026-09-28

# Inspect and replay dead letters.
pulse-queue dlq list
pulse-queue dlq replay <task_id>
pulse-queue dlq replay --all --limit 10
```

`myapp.tasks` must expose either a `registry` attribute or a
`build_registry()` function returning a `HandlerRegistry`. Point at a specific
attribute with `--handlers myapp.tasks:hooks`.

`--db` is accepted before or after the subcommand, and falls back to
`$PULSE_QUEUE_DB` and then `pulse.db`.

## Benchmarks

Measured on macOS arm64 (10 CPUs), Python 3.14.7, SQLite 3.53.4, schema v4.
Handlers sleep 10-50 ms so claims and completions genuinely overlap.

| Workload | Result |
| --- | --- |
| 500 tasks, one pool, `concurrency=8` | 500/500 `COMPLETED`, **0 lockouts**, 1.96 s, **255 tasks/s** |
| 500 tasks, 2 processes x 4 workers, one DB file | 500/500 `COMPLETED`, **0 lockouts**, 1.24 s wall |
| 5,000 tasks, 4 threads with separate connections | 5,000/5,000 `COMPLETED`, **0 lockouts**, 2.09 s |

Every run reported `attempts == 1` for all tasks and an empty dead letter queue:
no lease expired, so nothing was reclaimed and nothing ran twice. In the
single-pool run, exactly-once-per-key was checked across 500 handler
invocations -- 500 unique ids, 0 duplicates, 0 missing.

Priority ordering was verified against *claim* order rather than completion
order: all 124,750 task pairs came back in descending priority with **0
inversions**. Completion order trails claim order by at most 8 ranks (Spearman
0.9998), which is exactly the configured concurrency rather than reordering.

No `SQLITE_BUSY` or "database is locked" surfaced in any configuration. Within a
single pool that is by design -- one lock serializes every store call -- while
across separate connections it is `BEGIN IMMEDIATE` plus `busy_timeout=5000`
absorbing the contention.

## Development

**202 tests** covering the store, lease fencing, retry and DLQ routing, the
worker engine, the CLI, the migration engine, and shutdown across real
subprocess lifetimes. The full suite runs in about 3.7 s.

```sh
uv run pytest                      # everything
uv run pytest -m "not integration" # skip the subprocess tests
uv run ruff check .                # lint
```
