"""Schema, pragmas, and enqueue behaviour."""

from __future__ import annotations

import json
import sqlite3

import pytest

from pulse_queue import QueueingError, Store, StoreError, TaskStatus
from pulse_queue import schema as schema_module
from pulse_queue._sqlite import transaction
from pulse_queue.schema import (
    SCHEMA_VERSION,
    _MIGRATIONS,
    _statements,
    current_version,
    migrate,
    table_names,
)


def test_wal_is_enabled(store: Store) -> None:
    assert store.journal_mode() == "wal"


def test_migration_applied_and_idempotent(store: Store) -> None:
    assert store.schema_version == SCHEMA_VERSION
    # Re-running must not raise or double-apply.
    assert migrate(store._conn) == SCHEMA_VERSION
    assert migrate(store._conn) == SCHEMA_VERSION


def test_tasks_table_uses_the_max_attempts_column(store: Store) -> None:
    """The rename has to reach the SQL, not just the Python attribute."""
    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(tasks)")}
    assert "max_attempts" in columns
    assert "max_retries" not in columns


def test_dlq_carries_priority(store: Store) -> None:
    columns = {
        row[1]
        for row in store._conn.execute("PRAGMA table_info(dead_letter_queue)")
    }
    assert "priority" in columns


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _apply_through(conn: sqlite3.Connection, target: int) -> None:
    """Apply migrations one at a time, up to and including ``target``.

    Deliberately not :func:`migrate`: the point is to inspect each intermediate
    schema, which ``migrate`` jumps straight past.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
    )
    applied = {
        row[0] for row in conn.execute("SELECT version FROM schema_migrations")
    }
    for version, ddl in _MIGRATIONS:
        if version > target or version in applied:
            continue
        with transaction(conn):
            for statement in _statements(ddl):
                conn.execute(statement)
            conn.execute("INSERT INTO schema_migrations VALUES (?, ?)", (version, 0.0))


def _legacy_v1_database(path) -> sqlite3.Connection:
    """A real v1 database holding one row in each table.

    Built from the frozen v1 DDL rather than from the current runner, so it is
    genuinely what an old deployment would have on disk.
    """
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    _apply_through(conn, 1)
    conn.execute(
        """
        INSERT INTO tasks (id, queue, payload, state, attempts, max_retries,
                           available_at, created_at, updated_at)
        VALUES ('legacy', 'default', X'7b7d', 'PENDING', 2, 7, 0, 0, 0)
        """
    )
    conn.execute(
        """
        INSERT INTO dead_letter_queue
            (id, queue, payload, attempts, last_error, failed_at, task_created_at)
        VALUES ('old-dlq', 'default', X'7b7d', 3, 'gone', 0, 0)
        """
    )
    return conn


def test_upgrade_path_v1_to_v2_to_v3(tmp_path) -> None:
    """Every step of the upgrade is inspectable and preserves existing rows."""
    conn = _legacy_v1_database(tmp_path / "legacy.db")

    # ---- v1: the original shape -----------------------------------------
    assert current_version(conn) == 1
    assert "max_retries" in _columns(conn, "tasks")
    dlq_columns = _columns(conn, "dead_letter_queue")
    assert {"revived_at", "revive_count"} <= dlq_columns
    assert "priority" not in dlq_columns

    # ---- v2: rename the attempt budget, carry priority across -----------
    _apply_through(conn, 2)
    assert current_version(conn) == 2

    task_columns = _columns(conn, "tasks")
    assert "max_attempts" in task_columns
    assert "max_retries" not in task_columns

    dlq_columns = _columns(conn, "dead_letter_queue")
    assert "priority" in dlq_columns
    # v2 leaves the vestigial columns alone; dropping them is v3's job, so an
    # upgrade that stopped here would still be a valid v2 database.
    assert {"revived_at", "revive_count"} <= dlq_columns

    # The rename carried the value across, and priority was backfilled from the
    # column default because v1 had nowhere to record it.
    row = conn.execute("SELECT max_attempts, attempts FROM tasks").fetchone()
    assert (row["max_attempts"], row["attempts"]) == (7, 2)
    priority = conn.execute("SELECT priority FROM dead_letter_queue").fetchone()[0]
    assert priority == 0

    # ---- v3: drop the columns nothing ever wrote ------------------------
    _apply_through(conn, 3)
    assert current_version(conn) == 3

    dlq_columns = _columns(conn, "dead_letter_queue")
    assert "revived_at" not in dlq_columns
    assert "revive_count" not in dlq_columns

    # DROP COLUMN rewrites the table, so prove the row itself survived intact.
    dead = conn.execute("SELECT * FROM dead_letter_queue").fetchone()
    assert dead["id"] == "old-dlq"
    assert dead["attempts"] == 3
    assert dead["last_error"] == "gone"
    assert dead["priority"] == 0

    # Dropping a column must not disturb the index the listing query uses.
    indexes = {
        row[1] for row in conn.execute("PRAGMA index_list(dead_letter_queue)")
    }
    assert "idx_dlq_failed_at" in indexes

    # And the whole path is idempotent once complete.
    assert migrate(conn) == SCHEMA_VERSION == 3
    conn.close()


def test_fresh_database_lands_on_the_current_schema(store: Store) -> None:
    """A new database replays every migration and keeps none of the debris."""
    assert store.schema_version == SCHEMA_VERSION == 3
    assert "max_retries" not in _columns(store._conn, "tasks")
    assert "revived_at" not in _columns(store._conn, "dead_letter_queue")
    assert "revive_count" not in _columns(store._conn, "dead_letter_queue")


def test_failed_migration_rolls_back_and_is_retried(tmp_path, monkeypatch) -> None:
    """A migration that fails must leave no DDL behind and must not be recorded.

    Recording the version before running the DDL would leave a database that
    claims to be migrated but is not, and the failure would never be retried.
    """
    db = tmp_path / "broken.db"
    conn = sqlite3.connect(str(db), isolation_level=None)
    conn.row_factory = sqlite3.Row

    real = _MIGRATIONS
    broken_version = SCHEMA_VERSION + 1
    monkeypatch.setattr(
        schema_module,
        "_MIGRATIONS",
        real
        + (
            (
                broken_version,
                "CREATE TABLE half_applied (a INTEGER); "
                "SELECT this_column_does_not_exist;",
            ),
        ),
    )

    with pytest.raises(sqlite3.Error):
        migrate(conn)

    assert broken_version not in {
        row[0] for row in conn.execute("SELECT version FROM schema_migrations")
    }
    assert "half_applied" not in table_names(conn)
    assert conn.in_transaction is False

    # The migration is retryable: a corrected version of it now succeeds.
    monkeypatch.setattr(
        schema_module,
        "_MIGRATIONS",
        real + ((broken_version, "CREATE TABLE eventually (a INTEGER);"),),
    )
    assert migrate(conn) == broken_version
    assert "eventually" in table_names(conn)
    conn.close()


def test_pending_migrations_require_a_recent_sqlite(tmp_path, monkeypatch) -> None:
    """An old SQLite must fail with a clear message, not a syntax error.

    Migration v3 uses ``ALTER TABLE ... DROP COLUMN``, which needs SQLite
    3.35.0. Without this check the user would see ``near "COLUMN": syntax
    error`` and have no idea which migration or which requirement was at fault.
    """
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 0))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.34.0")

    conn = sqlite3.connect(str(tmp_path / "old.db"), isolation_level=None)
    with pytest.raises(StoreError, match=r"SQLite >= 3\.35\.0"):
        migrate(conn)

    # Nothing was applied, so a later attempt on a modern SQLite starts clean.
    assert current_version(conn) == 0
    assert conn.in_transaction is False
    conn.close()


def test_an_already_current_database_ignores_the_sqlite_floor(
    store: Store, monkeypatch
) -> None:
    """No pending migrations means no DDL, so an old SQLite is still fine."""
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 20, 0))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.20.0")

    assert migrate(store._conn) == SCHEMA_VERSION


def test_expected_tables_exist(store: Store) -> None:
    names = set(table_names(store._conn))
    assert {"tasks", "dead_letter_queue", "schema_migrations"} <= names


def test_state_check_constraint_rejects_unknown(store: Store) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "UPDATE tasks SET state = 'NONSENSE' WHERE id = ?",
            (store.enqueue({"a": 1}).id,),
        )


def test_enqueue_returns_pending_task(store: Store, clock) -> None:
    task = store.enqueue({"hello": "world"})

    assert task.state is TaskStatus.PENDING
    assert task.attempts == 0
    assert task.queue == "default"
    assert task.max_attempts == 3
    assert task.available_at == clock.now()
    assert task.created_at == clock.now()
    assert task.lease_owner is None
    assert task.context().payload == {"hello": "world"}


def test_enqueue_generates_unique_ids(store: Store) -> None:
    ids = {store.enqueue({"n": i}).id for i in range(100)}
    assert len(ids) == 100


def test_enqueue_is_idempotent_on_explicit_id(store: Store) -> None:
    first = store.enqueue({"v": 1}, task_id="stable-key")
    second = store.enqueue({"v": 2}, task_id="stable-key")

    # Same key, original row preserved -- a retried producer is a no-op.
    assert first.id == second.id == "stable-key"
    assert second.decode_payload() == {"v": 1}
    assert store.count_tasks() == 1


def test_enqueue_rejects_unserializable_payload(store: Store) -> None:
    with pytest.raises(StoreError, match="JSON"):
        store.enqueue({"bad": object()})


def test_enqueue_rejects_empty_queue(store: Store) -> None:
    with pytest.raises(QueueingError):
        store.enqueue({"a": 1}, queue="   ")


def test_enqueue_rejects_negative_retries(store: Store) -> None:
    with pytest.raises(QueueingError):
        store.enqueue({"a": 1}, max_attempts=-1)


def test_delay_holds_task_until_available_at(store: Store, clock) -> None:
    task = store.enqueue({"a": 1}, delay=30.0)

    assert task.available_at == clock.now() + 30.0
    assert store.lease_next_task("w1") is None

    clock.advance(30.0)
    assert store.lease_next_task("w1") is not None


def test_payload_roundtrips_unicode(store: Store) -> None:
    payload = {"emoji": "🎉", "nested": {"list": [1, 2, 3]}}
    task = store.enqueue(payload)
    assert task.context().payload == payload
    assert json.loads(task.payload) == payload


def test_context_exposes_idempotency_key(store: Store) -> None:
    task = store.enqueue({"a": 1}, task_id="key-abc")
    ctx = task.context()

    assert ctx.task_id == "key-abc"
    assert ctx.attempt == 0
    assert ctx.max_attempts == 3


def test_get_task_raises_on_missing(store: Store) -> None:
    with pytest.raises(KeyError):
        store.get_task("does-not-exist")
    assert store.find_task("does-not-exist") is None


def test_store_is_reusable_as_context_manager(db_path, clock) -> None:
    with Store(db_path, clock=clock) as store:
        store.enqueue({"a": 1})
    # Closing twice must not raise.
    store.close()


def test_pragmas_survive_reopen(db_path, clock) -> None:
    first = Store(db_path, clock=clock)
    first.enqueue({"a": 1})
    first.close()

    second = Store(db_path, clock=clock)
    try:
        assert second.journal_mode() == "wal"
        assert second.count_tasks() == 1
    finally:
        second.close()
