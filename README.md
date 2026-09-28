# pulse-queue

A standalone background task queue built on SQLite and `asyncio`. One file on
disk is the whole broker: no Redis, no separate server process.

- **WAL-mode SQLite** with transactional lease and state transitions
- **Retry with exponential backoff**, then a **dead letter queue**
- **At-least-once delivery** with a stable idempotency key per task
- **Graceful shutdown** on `SIGINT`/`SIGTERM`, draining in-flight work
- **Lease expiry recovery**, so `SIGKILL` or a host reboot does not lose work

## Install

```sh
uv add pulse-queue
```

Requires Python 3.14+. No runtime dependencies beyond the standard library.

## Library

Define handlers against a registry, then run a pool:

```python
from pulse_queue import HandlerRegistry, Store, WorkerPool

registry = HandlerRegistry()


@registry.register("send_email")
async def send_email(ctx):
    await smtp_send(ctx.payload["to"], idempotency_key=ctx.task_id)


store = Store("pulse.db")
store.enqueue({"type": "send_email", "to": "ada@example.com"})

pool = WorkerPool(store, registry, concurrency=8)
asyncio.run(pool.run())  # blocks until SIGINT/SIGTERM, then drains
```

A handler's outcome selects the transition:

| Outcome | Result |
| --- | --- |
| returns | `COMPLETED` |
| `PermanentTaskError` | `FAILED` (no retry) |
| anything else | `RETRY` with backoff, then `DLQ` once the budget is spent |

### Delivery semantics

Delivery is **at-least-once**, not exactly-once. A task can run twice if a
worker crashes, if its lease expires mid-handler, or if shutdown gives up on it
after the grace period.

`ctx.task_id` is the idempotency key that makes this safe. It is generated at
enqueue time and never reassigned, so retries, dead-lettering, and replay all
carry the same value. Persist it as a unique constraint on any outbound call
your handler makes.

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

## Development

```sh
uv run pytest                      # everything
uv run pytest -m "not integration" # skip the subprocess tests
```
