"""Exception types controlling retry behaviour."""

from __future__ import annotations


class PulseQueueError(Exception):
    """Base class for all errors raised by this package."""


class RetryableTaskError(PulseQueueError):
    """Failure that should be retried with exponential backoff.

    This is the default for unexpected exceptions. Raise this (or let an
    arbitrary exception propagate) when the task should get another attempt
    before being dead-lettered.
    """


class PermanentTaskError(PulseQueueError):
    """Failure that must not be retried.

    The task moves to ``FAILED`` and stays visible in the ``tasks`` table for
    inspection. Use for malformed payloads, missing referenced records, and
    other conditions that will fail identically on every retry.
    """


class StoreError(PulseQueueError):
    """Database-level failure (locking, corruption, constraint violations)."""


class UnknownTaskType(PermanentTaskError):
    """No handler is registered for a task's type.

    Permanent by default: retrying cannot help, because the registry only
    changes when the process is redeployed. The task stays in the ``tasks``
    table as ``FAILED`` so it can be inspected and replayed once a handler
    exists. Subclassing :class:`PermanentTaskError` means the worker treats it
    as non-retryable without any special-casing.
    """


class QueueingError(PulseQueueError):
    """Invalid queue name, or a queue operation that is not permitted."""
