"""Command-line interface.

Four operations, all against one SQLite file:

``work``
    Run a worker pool until interrupted.
``enqueue``
    Put a task on the queue from a shell.
``dlq list``
    Show dead-lettered tasks.
``dlq replay``
    Put dead-lettered tasks back into the active queue.

Handlers are supplied by import path -- ``--handlers myapp.tasks`` -- because a
worker without a registry cannot run anything. The module must expose either a
``registry`` attribute or a ``build_registry()`` function returning a
:class:`~pulse_queue.worker.HandlerRegistry`.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from . import __version__
from .errors import PulseQueueError, QueueingError
from .store import DEFAULT_MAX_ATTEMPTS, Store
from .worker import HandlerRegistry, WorkerPool

#: Where ``--handlers`` comes from when the flag is absent.
ENV_HANDLERS = "PULSE_QUEUE_HANDLERS"

#: Where ``--db`` comes from when the flag is absent.
ENV_DB = "PULSE_QUEUE_DB"

DEFAULT_DB = "pulse.db"

#: Error text is clipped to this many characters in ``dlq list``.
ERROR_COLUMN_WIDTH = 56


class CliError(PulseQueueError):
    """A problem with the invocation or its inputs, not with the queue."""


# --------------------------------------------------------------------- helpers


def merge_payload(task_type: str, body: Any) -> dict[str, Any]:
    """Attach the positional task type to a JSON object payload.

    The worker resolves a handler by looking up ``payload["type"]``, so the
    type has to be *in* the payload, not just alongside it.

    >>> merge_payload("send_email", {"to": "ada@example.com"})
    {'type': 'send_email', 'to': 'ada@example.com'}

    A conflicting ``type`` inside the body is rejected rather than silently
    overridden, because the two disagreeing means one of them is a mistake:

    >>> merge_payload("send_email", {"type": "send_sms"})
    Traceback (most recent call last):
        ...
    pulse_queue.cli.CliError: payload type 'send_sms' conflicts with 'send_email'
    """
    if not isinstance(body, dict):
        raise CliError(
            f"payload must be a JSON object, got {type(body).__name__}"
        )
    declared = body.get("type")
    if declared is not None and declared != task_type:
        raise CliError(
            f"payload type {declared!r} conflicts with {task_type!r}"
        )
    return {"type": task_type, **body}


def parse_payload(raw: str) -> Any:
    """Parse a ``payload_json`` argument, treating ``-`` as stdin."""
    if raw == "-":
        raw = sys.stdin.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CliError(f"payload is not valid JSON: {exc}") from exc


def load_registry(spec: str) -> HandlerRegistry:
    """Import a handler registry from ``module`` or ``module:attribute``.

    Raises:
        CliError: If the module cannot be imported, exposes no registry, or
            registers no handlers.
    """
    module_name, _, attribute = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise CliError(f"cannot import handler module {module_name!r}: {exc}") from exc

    if attribute:
        try:
            found = getattr(module, attribute)
        except AttributeError:
            raise CliError(
                f"{module_name!r} has no attribute {attribute!r}"
            ) from None
        registry = _coerce_registry(found, spec)
    elif isinstance(getattr(module, "registry", None), HandlerRegistry):
        registry = module.registry
    elif callable(getattr(module, "build_registry", None)):
        registry = _coerce_registry(module.build_registry(), spec)
    elif hasattr(module, "registry"):
        # A `registry` of the wrong type is worth naming precisely: "exposes
        # neither" would send the reader looking for a missing attribute that
        # is in fact right there.
        raise CliError(
            f"{module_name!r} has a `registry` attribute of type "
            f"{type(module.registry).__name__}, expected HandlerRegistry"
        )
    else:
        raise CliError(
            f"{module_name!r} exposes neither a `registry` HandlerRegistry nor a "
            f"`build_registry()` function; point at one explicitly with "
            f"--handlers {module_name}:<name>"
        )

    if not len(registry):
        raise CliError(f"{spec!r} registered no handlers")
    return registry


def _coerce_registry(candidate: Any, spec: str) -> HandlerRegistry:
    if isinstance(candidate, HandlerRegistry):
        return candidate
    if callable(candidate):
        result = candidate()
        if isinstance(result, HandlerRegistry):
            return result
        raise CliError(
            f"{spec!r} returned {type(result).__name__}, expected HandlerRegistry"
        )
    raise CliError(
        f"{spec!r} is a {type(candidate).__name__}, expected HandlerRegistry "
        f"or a callable returning one"
    )


def _format_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _clip(text: str, width: int) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 1] + "\u2026"


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells: Sequence[str]) -> str:
        return "  ".join(
            cell.ljust(widths[index]) for index, cell in enumerate(cells)
        ).rstrip()

    rendered = [line(headers), "  ".join("-" * width for width in widths)]
    rendered.extend(line(row) for row in rows)
    return "\n".join(rendered)


def _dead_letter_row(entry: dict[str, Any], *, wide: bool) -> list[str]:
    error = str(entry["last_error"])
    if not wide:
        error = _clip(error, ERROR_COLUMN_WIDTH)
    return [
        entry["id"],
        str(entry["queue"]),
        str(entry["priority"]),
        f"{entry['attempts']}",
        _format_timestamp(entry["failed_at"]),
        error,
    ]


def _dead_letter_json(entry: dict[str, Any]) -> dict[str, Any]:
    """DLQ row as JSON-safe data, with the payload BLOB decoded."""
    failed_at = entry["failed_at"]
    try:
        payload = json.loads(entry["payload"])
    except (TypeError, ValueError):
        payload = entry["payload"].decode("utf-8", errors="replace")
    return {
        "id": entry["id"],
        "queue": entry["queue"],
        "priority": entry["priority"],
        "attempts": entry["attempts"],
        "last_error": entry["last_error"],
        "failed_at": failed_at,
        "failed_at_iso": _format_timestamp(failed_at),
        "task_created_at": entry["task_created_at"],
        "payload": payload,
    }


def _emit(data: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=False))
    else:
        print(data)


# -------------------------------------------------------------------- commands


def cmd_work(args: argparse.Namespace) -> int:
    """Run a worker pool until SIGINT/SIGTERM, then drain and exit."""
    spec = args.handlers or os.environ.get(ENV_HANDLERS)
    if not spec:
        raise CliError(
            "no handlers given; pass --handlers <module>[:<attribute>] or set "
            f"{ENV_HANDLERS}"
        )

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    registry = load_registry(spec)
    queues = (
        [q.strip() for q in args.queues.split(",") if q.strip()]
        if args.queues
        else None
    )

    log = logging.getLogger("pulse_queue.cli")
    log.info(
        "handlers=%s types=%s db=%s concurrency=%d lease_ttl=%.1fs",
        spec,
        ", ".join(sorted(registry.types())),
        args.db,
        args.concurrency,
        args.lease_ttl,
    )

    with Store(args.db) as store:
        pool = WorkerPool(
            store,
            registry,
            concurrency=args.concurrency,
            queues=queues,
            poll_interval=args.poll_interval,
            lease_ttl=args.lease_ttl,
            heartbeat_interval=args.heartbeat_interval,
            handler_timeout=args.handler_timeout,
            grace_period=args.grace_period,
            reclaim_enabled=not args.no_reclaim,
            reclaim_interval=args.reclaim_interval,
        )
        try:
            asyncio.run(pool.run())
        except KeyboardInterrupt:
            # Only reachable if the loop could not install signal handlers.
            log.warning("interrupted")
            return 130

    log.info("stopped; %s", pool.stats)
    return 0


def cmd_enqueue(args: argparse.Namespace) -> int:
    """Insert one task from the command line."""
    payload = merge_payload(args.type, parse_payload(args.payload_json))

    with Store(args.db) as store:
        task = store.enqueue(
            payload,
            queue=args.queue,
            priority=args.priority,
            # None means "use the store's default", so the CLI does not
            # duplicate the constant.
            max_attempts=args.max_attempts,
            task_id=args.idempotency_key,
            delay=args.delay,
        )

    if args.json:
        _emit(
            {
                "id": task.id,
                "queue": task.queue,
                "state": task.state.value,
                "priority": task.priority,
                "max_attempts": task.max_attempts,
                "available_at": task.available_at,
                "payload": task.decode_payload(),
            },
            as_json=True,
        )
    else:
        # Bare id on stdout so it can be captured: ID=$(pulse-queue enqueue ...)
        print(task.id)
    return 0


def cmd_dlq_list(args: argparse.Namespace) -> int:
    """List dead-lettered tasks, newest first."""
    with Store(args.db) as store:
        entries = store.list_dead_letters(limit=args.limit)

    if args.json:
        _emit([_dead_letter_json(entry) for entry in entries], as_json=True)
        return 0

    if not entries:
        print("no dead-lettered tasks")
        return 0

    headers = ["ID", "QUEUE", "PRI", "ATTEMPTS", "FAILED AT", "ERROR"]
    rows = [_dead_letter_row(entry, wide=args.wide) for entry in entries]
    print(_render_table(headers, rows))
    print(f"\n{len(entries)} task(s)")
    return 0


def cmd_dlq_replay(args: argparse.Namespace) -> int:
    """Re-enqueue dead-lettered tasks and remove them from the DLQ."""
    if args.all and args.task_id:
        raise CliError("pass either a task id or --all, not both")
    if not args.all and not args.task_id:
        raise CliError("pass a task id, or --all to replay every dead letter")

    with Store(args.db) as store:
        if not args.all:
            task = store.replay_dead_letter(
                args.task_id, max_attempts=args.max_attempts
            )
            if task is None:
                raise CliError(f"no dead-lettered task with id {args.task_id!r}")
            print(
                f"replayed {task.id} (queue={task.queue}, "
                f"attempts=0/{task.max_attempts})"
            )
            return 0

        # Snapshot first: replaying mutates the DLQ while we iterate it.
        pending = [entry["id"] for entry in store.list_dead_letters(limit=args.limit)]
        replayed: list[str] = []
        skipped: list[tuple[str, str]] = []
        for task_id in pending:
            try:
                task = store.replay_dead_letter(
                    task_id, max_attempts=args.max_attempts
                )
            except QueueingError as exc:
                # One conflicted row must not block the rest of the batch.
                skipped.append((task_id, str(exc)))
                continue
            if task is None:
                skipped.append((task_id, "no longer dead-lettered"))
            else:
                replayed.append(task.id)

    for task_id, reason in skipped:
        print(f"skipped {task_id}: {reason}", file=sys.stderr)
    print(f"replayed {len(replayed)} task(s), skipped {len(skipped)}")
    return 0 if replayed or not pending else 1


# --------------------------------------------------------------------- parser


def _add_db_argument(
    parser: argparse.ArgumentParser, *, suppress_default: bool = False
) -> None:
    """Add ``--db``.

    Declared on the top-level parser *and* on every leaf, so both
    ``pulse-queue --db X enqueue ...`` and ``pulse-queue enqueue --db X ...``
    work. The leaves use ``SUPPRESS`` rather than a real default: argparse
    applies a subparser's defaults *after* the parent's, so a leaf default
    would silently overwrite a ``--db`` given before the subcommand.
    """
    parser.add_argument(
        "--db",
        default=(
            argparse.SUPPRESS
            if suppress_default
            else os.environ.get(ENV_DB, DEFAULT_DB)
        ),
        metavar="PATH",
        help=(
            "SQLite database file (default: $%s or %r). Created and migrated "
            "if absent." % (ENV_DB, DEFAULT_DB)
        ),
    )


def _add_json_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pulse-queue",
        description="SQLite-backed task queue with leases, retries, and a DLQ.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    _add_db_argument(parser)
    subcommands = parser.add_subparsers(dest="command", required=True)

    # ---------------------------------------------------------------- work
    work = subcommands.add_parser(
        "work", help="run a worker pool until interrupted"
    )
    _add_db_argument(work, suppress_default=True)
    work.add_argument(
        "--handlers",
        metavar="MODULE[:ATTR]",
        default=None,
        help=(
            "module exposing a `registry` HandlerRegistry or a "
            "`build_registry()` function (default: $%s)" % ENV_HANDLERS
        ),
    )
    work.add_argument(
        "-c", "--concurrency", type=int, default=4, help="parallel slots (default: 4)"
    )
    work.add_argument(
        "--queues",
        default=None,
        metavar="A,B",
        help="comma-separated queues to serve (default: all)",
    )
    work.add_argument(
        "--lease-ttl",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="lease lifetime; must exceed handler runtime (default: 60)",
    )
    work.add_argument(
        "--heartbeat-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help="lease renewal cadence (default: lease-ttl / 3)",
    )
    work.add_argument(
        "--handler-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="abandon and retry a handler exceeding this (default: none)",
    )
    work.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="idle sleep between claims (default: 0.5)",
    )
    work.add_argument(
        "--grace-period",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="shutdown drain window before cancelling (default: 30)",
    )
    work.add_argument(
        "--reclaim-interval",
        type=float,
        default=15.0,
        metavar="SECONDS",
        help="expired-lease sweep cadence (default: 15)",
    )
    work.add_argument(
        "--no-reclaim",
        action="store_true",
        help="disable the expired-lease reclaimer",
    )
    work.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: INFO)",
    )
    work.set_defaults(func=cmd_work)

    # ------------------------------------------------------------- enqueue
    enqueue = subcommands.add_parser(
        "enqueue", help="add a task to the queue from the shell"
    )
    _add_db_argument(enqueue, suppress_default=True)
    enqueue.add_argument("type", metavar="TYPE", help="task type; selects the handler")
    enqueue.add_argument(
        "payload_json",
        metavar="PAYLOAD_JSON",
        help="JSON object of task arguments; '-' reads stdin",
    )
    enqueue.add_argument(
        "--idempotency-key",
        metavar="KEY",
        default=None,
        help=(
            "stable task id. Re-using a key is a no-op if the task still "
            "exists, which makes a retried enqueue safe."
        ),
    )
    enqueue.add_argument("--queue", default=None, help="queue name (default: default)")
    enqueue.add_argument(
        "--priority", type=int, default=0, help="higher runs first (default: 0)"
    )
    enqueue.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        metavar="N",
        help=(
            "total attempts before dead-lettering (default: %d). This is an "
            "attempt budget, not a retry count." % DEFAULT_MAX_ATTEMPTS
        ),
    )
    enqueue.add_argument(
        "--delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="hold the task before it becomes claimable (default: 0)",
    )
    _add_json_argument(enqueue)
    enqueue.set_defaults(func=cmd_enqueue)

    # ----------------------------------------------------------------- dlq
    dlq = subcommands.add_parser("dlq", help="inspect and replay dead letters")
    dlq_commands = dlq.add_subparsers(dest="dlq_command", required=True)

    dlq_list = dlq_commands.add_parser("list", help="show dead-lettered tasks")
    _add_db_argument(dlq_list, suppress_default=True)
    dlq_list.add_argument(
        "--limit", type=int, default=50, help="maximum rows (default: 50)"
    )
    dlq_list.add_argument(
        "--wide",
        action="store_true",
        help="do not truncate the error column",
    )
    _add_json_argument(dlq_list)
    dlq_list.set_defaults(func=cmd_dlq_list)

    replay = dlq_commands.add_parser(
        "replay", help="re-enqueue dead-lettered tasks"
    )
    _add_db_argument(replay, suppress_default=True)
    replay.add_argument(
        "task_id",
        nargs="?",
        default=None,
        metavar="TASK_ID",
        help="task to replay; omit with --all",
    )
    replay.add_argument(
        "--all", action="store_true", help="replay every dead-lettered task"
    )
    replay.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="with --all, replay at most N tasks (default: no limit)",
    )
    replay.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        metavar="N",
        help=(
            "attempt budget for the replayed task (default: the store's "
            "default, %d)" % DEFAULT_MAX_ATTEMPTS
        ),
    )
    replay.set_defaults(func=cmd_dlq_replay)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handler: Callable[[argparse.Namespace], int] = args.func
    try:
        return handler(args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except PulseQueueError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
