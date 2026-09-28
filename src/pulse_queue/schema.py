"""Schema definition and forward-only migration runner.

Migrations are applied in ascending version order, each inside a single
transaction, and are recorded in ``schema_migrations``. Re-running :func:`migrate`
on an already-current database is a no-op.

Two rules keep this honest:

1. **Never edit an applied migration.** Existing databases have already run it,
   and the runner will not re-run it -- so editing the text fixes nothing and
   makes fresh databases disagree with upgraded ones. Corrections go in a new
   migration.
2. **The version row is written last.** Recording a version before its DDL
   succeeds would leave a database that claims to be migrated but is not, and
   the failure would never be retried.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from ._sqlite import transaction
from .errors import StoreError

#: Bump when adding a migration below.
SCHEMA_VERSION = 4

#: Minimum SQLite needed to apply pending migrations.
#:
#: Migration v3 uses ``ALTER TABLE ... DROP COLUMN`` (SQLite 3.35.0, 2021).
#: Python does not guarantee which SQLite it was built against, so an older one
#: is detected explicitly rather than left to fail with a bare
#: ``near "COLUMN": syntax error``.
MIN_SQLITE_VERSION: tuple[int, int, int] = (3, 35, 0)

#: Frozen history -- see rule 1 in the module docstring. Fresh databases replay
#: v1 followed by v2, so the current shape is v1 + v2 rather than v1 edited.
_MIGRATION_V1 = """
CREATE TABLE tasks (
    id               TEXT    PRIMARY KEY,
    queue            TEXT    NOT NULL DEFAULT 'default',
    payload          BLOB    NOT NULL,
    state            TEXT    NOT NULL DEFAULT 'PENDING'
        CHECK (state IN ('PENDING','RUNNING','RETRY','COMPLETED','FAILED')),
    priority         INTEGER NOT NULL DEFAULT 0,
    attempts         INTEGER NOT NULL DEFAULT 0,
    max_retries      INTEGER NOT NULL DEFAULT 3,
    available_at     REAL    NOT NULL,
    lease_owner      TEXT,
    lease_expires_at REAL,
    last_error       TEXT,
    created_at       REAL    NOT NULL,
    updated_at       REAL    NOT NULL,
    started_at       REAL,
    finished_at      REAL
);

-- Partial index matching the claim query's WHERE + ORDER BY exactly. Stays
-- small because terminal rows never enter it.
CREATE INDEX idx_tasks_ready ON tasks (priority DESC, available_at, id)
    WHERE state IN ('PENDING','RETRY');

-- Drives the lease reclaimer sweep; partial so it only holds RUNNING rows.
CREATE INDEX idx_tasks_lease ON tasks (lease_expires_at)
    WHERE state = 'RUNNING';

CREATE TABLE dead_letter_queue (
    id              TEXT    PRIMARY KEY,
    queue           TEXT    NOT NULL,
    payload         BLOB    NOT NULL,
    attempts        INTEGER NOT NULL,
    last_error      TEXT    NOT NULL,
    failed_at       REAL    NOT NULL,
    task_created_at REAL    NOT NULL,
    revived_at      REAL,
    revive_count    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX idx_dlq_failed_at ON dead_letter_queue (failed_at DESC);
"""

#: Renames ``max_retries`` to ``max_attempts``. The old name read as "retries"
#: but was enforced as a total attempt budget, so ``max_retries=3`` meant three
#: attempts rather than one attempt plus three retries.
#:
#: Also carries ``priority`` across to the DLQ. Without it, replaying a
#: dead-lettered task would silently reset it to priority 0 and drop it behind
#: everything else in the queue.
_MIGRATION_V2 = """
ALTER TABLE tasks RENAME COLUMN max_retries TO max_attempts;
ALTER TABLE dead_letter_queue ADD COLUMN priority INTEGER NOT NULL DEFAULT 0;
"""

#: Drops ``revived_at`` and ``revive_count``.
#:
#: They were designed for an audit trail where replaying a dead letter marked
#: the DLQ row as revived instead of consuming it. Replay went the other way --
#: it deletes the row -- so nothing ever wrote them, and a column with no writer
#: is worse than no column: it advertises a history that does not exist.
#:
#: Uses ``ALTER TABLE ... DROP COLUMN``, hence :data:`MIN_SQLITE_VERSION`.
#: Dropping rewrites the table, but SQLite preserves every remaining column and
#: the ``idx_dlq_failed_at`` index is untouched.
_MIGRATION_V3 = """
ALTER TABLE dead_letter_queue DROP COLUMN revived_at;
ALTER TABLE dead_letter_queue DROP COLUMN revive_count;
"""

#: Adds the fencing token that every lease-bound transition must present.
#:
#: ``lease_owner`` alone cannot fence, because it is a *name* rather than an
#: identity. Two pools sharing a database are constructed from the same default
#: prefix, so both field a ``worker-0``; a worker whose lease expired could then
#: present exactly the string the reclaiming worker holds and have its stale
#: result accepted, overwriting newer state. Bumping a per-task counter at claim
#: time gives every lease a generation that cannot repeat, so a stale holder is
#: rejected whatever it calls itself -- including a pool racing with *itself*
#: after a reclaim.
#:
#: Backfills to 0 for existing rows. Unclaimed tasks sit at 0; the first claim
#: makes it 1.
_MIGRATION_V4 = """
ALTER TABLE tasks ADD COLUMN lease_epoch INTEGER NOT NULL DEFAULT 0;
"""

_MIGRATIONS: tuple[tuple[int, str], ...] = (
    (1, _MIGRATION_V1),
    (2, _MIGRATION_V2),
    (3, _MIGRATION_V3),
    (4, _MIGRATION_V4),
)


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations. Returns the resulting schema version."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    INTEGER PRIMARY KEY,
            applied_at REAL    NOT NULL
        )
        """
    )
    applied = {
        row[0] for row in conn.execute("SELECT version FROM schema_migrations")
    }

    pending = [version for version, _ in _MIGRATIONS if version not in applied]
    if pending and sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
        required = ".".join(str(part) for part in MIN_SQLITE_VERSION)
        raise StoreError(
            f"cannot apply schema migrations: SQLite >= {required} is required "
            f"(found {sqlite3.sqlite_version}), because migration v3 uses "
            f"ALTER TABLE ... DROP COLUMN"
        )

    for version, ddl in _MIGRATIONS:
        if version in applied:
            continue
        # One transaction per migration, with the version row written last: a
        # failure rolls the whole migration back and retries it on the next
        # open, rather than recording a version whose DDL never completed.
        with transaction(conn):
            for statement in _statements(ddl):
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations VALUES (?, ?)", (version, _now())
            )

    return current_version(conn)


def _statements(script: str) -> list[str]:
    """Split a migration script into individual statements.

    ``executescript`` cannot be used here: it issues an unconditional ``COMMIT``
    before running, which would end the very transaction the migration is meant
    to be atomic inside. Scripts are ours, so a line-comment-aware split on
    ``;`` is sufficient -- no migration may contain a semicolon inside a string
    literal.
    """
    decommented = "\n".join(
        line.split("--", 1)[0] for line in script.splitlines()
    )
    return [stmt.strip() for stmt in decommented.split(";") if stmt.strip()]


def current_version(conn: sqlite3.Connection) -> int:
    """Highest applied migration version, or 0 if none."""
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _now() -> float:
    # Deliberately the raw clock: migrate() is a low-level entry point that
    # runs before any Store exists to inject a test clock.
    import time

    return time.time()


def table_names(conn: sqlite3.Connection) -> Sequence[str]:
    """User table names, for tests and diagnostics."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return sorted(row[0] for row in rows)
