"""Explicit SQLite transactions.

:class:`~pulse_queue.store.Store` opens its connection with
``isolation_level=None`` so that transactions are explicit rather than implicit.
That choice has a sharp edge worth spelling out, because it is silent:

    Under ``isolation_level=None``, ``with conn:`` does **not** open a
    transaction. It only commits or rolls back whatever is already open -- and
    in autocommit there is nothing open, so a failure part-way through a
    multi-statement operation leaves the earlier statements permanently
    written.

Every operation that writes more than one statement must therefore go through
:func:`transaction`. Single-statement operations are atomic on their own and do
not need it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block inside one write transaction, rolling back on any exception.

    ``BEGIN IMMEDIATE`` takes the write lock up front. A deferred transaction
    that reads first and only then upgrades to a write can fail with
    ``SQLITE_BUSY`` at the upgrade point without ever consulting
    ``busy_timeout``; taking the lock at ``BEGIN`` makes contention wait its
    turn instead of failing.

    Args:
        conn: Connection to run the block on.

    Raises:
        RuntimeError: If a transaction is already open. Nesting would let the
            inner ``COMMIT`` release the outer transaction's writes early,
            silently breaking the outer block's atomicity.
    """
    if conn.in_transaction:
        raise RuntimeError(
            "transaction() cannot be nested: a transaction is already open"
        )

    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        # Roll back only if the transaction is still open: a statement that
        # aborted the transaction outright must not mask the original error.
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
