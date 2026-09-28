"""CLI behaviour: argument handling, output, exit codes, and DLQ replay.

Most tests call :func:`pulse_queue.cli.main` in-process. That is fast and lets
them assert exact exit codes and captured output. The two things an in-process
call *cannot* prove are checked separately: that ``[project.scripts]`` produces
a working executable, and that ``work`` really runs a pool -- both of which go
through a real subprocess.
"""

from __future__ import annotations

import importlib
import io
import itertools
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from pulse_queue import FrozenClock, Store, TaskStatus
from pulse_queue.cli import main
from pulse_queue.schema import SCHEMA_VERSION
from pulse_queue.worker import HandlerRegistry, WorkerPool, payload_type_resolver

# --------------------------------------------------------------------- helpers


def run(argv: list[str], capsys) -> tuple[int, str, str]:
    """Invoke the CLI in-process. Returns ``(exit_code, stdout, stderr)``."""
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def console_script() -> Path:
    """Path of the installed ``pulse-queue`` script.

    Derived from the interpreter rather than ``PATH``, so the test works even
    when the venv's bin directory is not exported.
    """
    candidate = Path(sys.executable).parent / "pulse-queue"
    if not candidate.exists():  # pragma: no cover - depends on the environment
        pytest.skip(f"console script not installed at {candidate}")
    return candidate


REGISTRY_MODULE = """
from pulse_queue import HandlerRegistry

registry = HandlerRegistry()

@registry.register("work")
async def work(ctx):
    pass
"""

BUILD_REGISTRY_MODULE = """
from pulse_queue import HandlerRegistry

def build_registry():
    registry = HandlerRegistry()

    @registry.register("work")
    async def work(ctx):
        pass

    return registry
"""

EMPTY_REGISTRY_MODULE = """
from pulse_queue import HandlerRegistry

registry = HandlerRegistry()
"""

NO_REGISTRY_MODULE = """
answer = 42
"""

NOT_A_REGISTRY_MODULE = """
registry = "not a registry"
"""


@pytest.fixture
def handler_module(tmp_path, monkeypatch):
    """Factory writing an importable handler module into ``tmp_path``."""
    created: list[str] = []
    counter = itertools.count()

    def make(source: str, *, name: str | None = None) -> str:
        module_name = name or f"pq_cli_handlers_{next(counter)}"
        (tmp_path / f"{module_name}.py").write_text(textwrap.dedent(source))
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()
        created.append(module_name)
        return module_name

    yield make
    for module_name in created:
        sys.modules.pop(module_name, None)


@pytest.fixture
def seeded_dlq(db_path):
    """Two dead-lettered tasks, with distinct priorities and failure times.

    Time is frozen and advanced between dead-letters so "newest first" is
    deterministic rather than depending on ties in ``failed_at``.
    """
    clock = FrozenClock(1_700_000_000.0)
    with Store(db_path, clock=clock) as store:
        for index, priority in enumerate((0, 3)):
            store.enqueue(
                {"type": "work", "i": index},
                task_id=f"dead-{index}",
                priority=priority,
                max_attempts=0,
            )
            task = store.lease_next_task(f"w{index}")
            store.dead_letter(
                task.id, f"w{index}", task.lease_epoch, f"failure {index}"
            )
            clock.advance(10.0)
    return db_path


@pytest.fixture
def fake_run(monkeypatch) -> dict[str, Any]:
    """Keep ``work`` from actually starting a pool.

    The real constructor still runs, so option plumbing is asserted against a
    genuinely built ``WorkerPool``; only ``run()`` is neutralised, which avoids
    an event loop, signal handlers, and real workers.
    """
    recorded: dict[str, Any] = {}
    real_init = WorkerPool.__init__

    def spy_init(self, *args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        real_init(self, *args, **kwargs)

    async def fake_run(self):
        return None

    monkeypatch.setattr(WorkerPool, "__init__", spy_init)
    monkeypatch.setattr(WorkerPool, "run", fake_run)
    return recorded


# ==================================================================== enqueue


def test_enqueue_prints_a_bare_task_id(db_path, capsys) -> None:
    """Bare id on stdout so it can be captured in a shell variable."""
    code, out, err = run(
        ["--db", str(db_path), "enqueue", "send_email", '{"to": "a@b.c"}'], capsys
    )

    assert code == 0
    assert err == ""
    task_id = out.strip()
    assert len(task_id) == 32
    assert all(character in "0123456789abcdef" for character in task_id)

    with Store(db_path) as store:
        assert store.get_task(task_id).state is TaskStatus.PENDING


def test_enqueue_injects_the_type_where_the_worker_looks_for_it(
    db_path, capsys
) -> None:
    """The positional TYPE must end up in the payload, not beside it.

    ``payload_type_resolver`` reads ``payload["type"]``, so a task enqueued
    from the shell has to be indistinguishable from one enqueued in Python.
    """
    _, out, _ = run(
        ["--db", str(db_path), "enqueue", "send_email", '{"to": "a@b.c"}'], capsys
    )

    with Store(db_path) as store:
        task = store.get_task(out.strip())

    assert task.decode_payload() == {"type": "send_email", "to": "a@b.c"}
    assert payload_type_resolver(task.decode_payload()) == "send_email"


def test_enqueue_json_reports_the_full_task(db_path, capsys) -> None:
    code, out, _ = run(
        [
            "--db", str(db_path), "enqueue", "add", '{"a": 1}',
            "--queue", "math",
            "--priority", "5",
            "--max-attempts", "9",
            "--json",
        ],
        capsys,
    )

    assert code == 0
    body = json.loads(out)
    assert body["queue"] == "math"
    assert body["priority"] == 5
    assert body["max_attempts"] == 9
    assert body["state"] == "PENDING"
    assert body["payload"] == {"type": "add", "a": 1}


def test_enqueue_defaults_max_attempts_to_the_store_default(db_path, capsys) -> None:
    _, out, _ = run(["--db", str(db_path), "enqueue", "t", "{}", "--json"], capsys)
    assert json.loads(out)["max_attempts"] == 3


def test_enqueue_is_idempotent_on_the_idempotency_key(db_path, capsys) -> None:
    first, out1, _ = run(
        ["--db", str(db_path), "enqueue", "add", '{"a": 1}', "--idempotency-key", "k"],
        capsys,
    )
    second, out2, _ = run(
        ["--db", str(db_path), "enqueue", "add", '{"a": 2}', "--idempotency-key", "k"],
        capsys,
    )

    assert (first, second) == (0, 0)
    assert out1.strip() == out2.strip() == "k"

    with Store(db_path) as store:
        assert store.count_tasks() == 1
        # The original row wins: a retried enqueue must not mutate the task.
        assert store.get_task("k").decode_payload() == {"type": "add", "a": 1}


def test_enqueue_reads_the_payload_from_stdin(db_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO('{"a": 5}'))
    code, out, _ = run(["--db", str(db_path), "enqueue", "add", "-"], capsys)

    assert code == 0
    with Store(db_path) as store:
        assert store.get_task(out.strip()).decode_payload() == {"type": "add", "a": 5}


def test_enqueue_delay_holds_the_task_out_of_reach(db_path, capsys) -> None:
    before = time.time()
    code, out, _ = run(
        ["--db", str(db_path), "enqueue", "add", "{}", "--delay", "30"], capsys
    )

    assert code == 0
    with Store(db_path) as store:
        task = store.get_task(out.strip())
        assert task.available_at >= before + 30
        assert store.lease_next_task("w") is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("{not json}", "not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ('"just a string"', "must be a JSON object"),
        ("null", "must be a JSON object"),
    ],
)
def test_enqueue_rejects_malformed_payloads(db_path, capsys, payload, expected) -> None:
    code, _, err = run(["--db", str(db_path), "enqueue", "add", payload], capsys)

    assert code == 1
    assert expected in err
    with Store(db_path) as store:
        assert store.count_tasks() == 0


def test_enqueue_rejects_a_conflicting_type(db_path, capsys) -> None:
    """Two disagreeing sources of truth is a mistake, not something to guess at."""
    code, _, err = run(
        ["--db", str(db_path), "enqueue", "add", '{"type": "other"}'], capsys
    )

    assert code == 1
    assert "conflicts" in err
    with Store(db_path) as store:
        assert store.count_tasks() == 0


def test_enqueue_accepts_a_matching_type(db_path, capsys) -> None:
    code, _, _ = run(
        ["--db", str(db_path), "enqueue", "add", '{"type": "add", "a": 1}'], capsys
    )
    assert code == 0
    with Store(db_path) as store:
        assert store.count_tasks() == 1


def test_enqueue_rejects_negative_max_attempts(db_path, capsys) -> None:
    code, _, err = run(
        ["--db", str(db_path), "enqueue", "add", "{}", "--max-attempts", "-1"], capsys
    )
    assert code == 1
    assert "max_attempts" in err


def test_enqueue_creates_and_migrates_the_database(tmp_path, capsys) -> None:
    db = tmp_path / "fresh.db"
    assert not db.exists()

    code, _, _ = run(["--db", str(db), "enqueue", "add", "{}"], capsys)

    assert code == 0
    assert db.exists()
    with Store(db) as store:
        assert store.schema_version == SCHEMA_VERSION


def test_a_missing_parent_directory_is_a_clean_error(tmp_path, capsys) -> None:
    """sqlite3 cannot create directories; the user should not see a traceback."""
    code, _, err = run(["--db", str(tmp_path / "nope" / "x.db"), "dlq", "list"], capsys)

    assert code == 1
    assert err.startswith("error:")


# ======================================================================= work


def test_work_requires_handlers(db_path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("PULSE_QUEUE_HANDLERS", raising=False)
    code, _, err = run(["--db", str(db_path), "work"], capsys)

    assert code == 1
    assert "no handlers given" in err


def test_work_plumbs_options_into_the_pool(
    db_path, capsys, handler_module, fake_run
) -> None:
    module = handler_module(REGISTRY_MODULE)
    code, _, _ = run(
        [
            "--db", str(db_path), "work",
            "--handlers", module,
            "--concurrency", "7",
            "--queues", "alpha, beta",
            "--lease-ttl", "12.5",
            "--handler-timeout", "3",
            "--grace-period", "1.5",
            "--poll-interval", "0.25",
            "--reclaim-interval", "9",
            "--no-reclaim",
            "--log-level", "ERROR",
        ],
        capsys,
    )

    assert code == 0
    _, registry = fake_run["args"]
    assert isinstance(registry, HandlerRegistry)
    assert registry.types() == {"work"}

    kwargs = fake_run["kwargs"]
    assert kwargs["concurrency"] == 7
    assert kwargs["queues"] == ["alpha", "beta"]
    assert kwargs["lease_ttl"] == 12.5
    assert kwargs["handler_timeout"] == 3.0
    assert kwargs["grace_period"] == 1.5
    assert kwargs["poll_interval"] == 0.25
    assert kwargs["reclaim_interval"] == 9.0
    assert kwargs["reclaim_enabled"] is False


def test_work_defaults_are_sane(db_path, capsys, handler_module, fake_run) -> None:
    module = handler_module(REGISTRY_MODULE)
    run(
        ["--db", str(db_path), "work", "--handlers", module, "--log-level", "ERROR"],
        capsys,
    )

    kwargs = fake_run["kwargs"]
    assert kwargs["concurrency"] == 4
    assert kwargs["queues"] is None
    assert kwargs["lease_ttl"] == 60.0
    assert kwargs["grace_period"] == 30.0
    assert kwargs["reclaim_enabled"] is True
    assert kwargs["heartbeat_interval"] is None
    assert kwargs["handler_timeout"] is None


def test_work_reads_handlers_from_the_environment(
    db_path, capsys, handler_module, fake_run, monkeypatch
) -> None:
    module = handler_module(REGISTRY_MODULE)
    monkeypatch.setenv("PULSE_QUEUE_HANDLERS", module)

    code, _, _ = run(["--db", str(db_path), "work", "--log-level", "ERROR"], capsys)

    assert code == 0
    assert fake_run["args"][1].types() == {"work"}


def test_work_accepts_a_build_registry_function(
    db_path, capsys, handler_module, fake_run
) -> None:
    module = handler_module(BUILD_REGISTRY_MODULE)
    code, _, _ = run(
        ["--db", str(db_path), "work", "--handlers", module, "--log-level", "ERROR"],
        capsys,
    )
    assert code == 0
    assert fake_run["args"][1].types() == {"work"}


def test_work_accepts_an_explicit_attribute(
    db_path, capsys, handler_module, fake_run
) -> None:
    module = handler_module(REGISTRY_MODULE)
    code, _, _ = run(
        [
            "--db", str(db_path), "work",
            "--handlers", f"{module}:registry",
            "--log-level", "ERROR",
        ],
        capsys,
    )
    assert code == 0
    assert fake_run["args"][1].types() == {"work"}


def test_work_rejects_a_module_without_a_registry(
    db_path, capsys, handler_module
) -> None:
    module = handler_module(NO_REGISTRY_MODULE)
    code, _, err = run(["--db", str(db_path), "work", "--handlers", module], capsys)

    assert code == 1
    assert "neither a `registry`" in err


def test_work_rejects_an_empty_registry(db_path, capsys, handler_module) -> None:
    """An empty registry means the wiring is wrong, not that work is idle."""
    module = handler_module(EMPTY_REGISTRY_MODULE)
    code, _, err = run(["--db", str(db_path), "work", "--handlers", module], capsys)

    assert code == 1
    assert "registered no handlers" in err


def test_work_rejects_a_non_registry_attribute(db_path, capsys, handler_module) -> None:
    module = handler_module(NOT_A_REGISTRY_MODULE)
    code, _, err = run(["--db", str(db_path), "work", "--handlers", module], capsys)

    assert code == 1
    assert "expected HandlerRegistry" in err


def test_work_rejects_an_explicitly_named_non_registry(
    db_path, capsys, handler_module
) -> None:
    module = handler_module(NOT_A_REGISTRY_MODULE)
    code, _, err = run(
        ["--db", str(db_path), "work", "--handlers", f"{module}:registry"], capsys
    )

    assert code == 1
    assert "expected HandlerRegistry" in err


def test_work_reports_an_unimportable_module(db_path, capsys) -> None:
    code, _, err = run(
        ["--db", str(db_path), "work", "--handlers", "definitely.not.here"], capsys
    )
    assert code == 1
    assert "cannot import handler module" in err


def test_work_reports_a_missing_attribute(db_path, capsys, handler_module) -> None:
    module = handler_module(REGISTRY_MODULE)
    code, _, err = run(
        ["--db", str(db_path), "work", "--handlers", f"{module}:nope"], capsys
    )
    assert code == 1
    assert "has no attribute 'nope'" in err


# =================================================================== dlq list


def test_dlq_list_reports_an_empty_queue(db_path, capsys) -> None:
    code, out, err = run(["--db", str(db_path), "dlq", "list"], capsys)
    assert code == 0
    assert "no dead-lettered tasks" in out
    assert err == ""


def test_dlq_list_renders_a_table(seeded_dlq, capsys) -> None:
    code, out, _ = run(["--db", str(seeded_dlq), "dlq", "list"], capsys)

    assert code == 0
    for header in ("ID", "QUEUE", "PRI", "ATTEMPTS", "FAILED AT", "ERROR"):
        assert header in out
    assert "dead-0" in out and "dead-1" in out
    assert "failure 0" in out and "failure 1" in out
    assert "2 task(s)" in out


def test_dlq_list_is_newest_first(seeded_dlq, capsys) -> None:
    _, out, _ = run(["--db", str(seeded_dlq), "dlq", "list"], capsys)
    body = [line for line in out.splitlines() if line.startswith("dead-")]
    assert [line.split()[0] for line in body] == ["dead-1", "dead-0"]


def test_dlq_list_clips_long_errors_unless_wide(db_path, capsys) -> None:
    with Store(db_path) as store:
        store.enqueue({"type": "x"}, task_id="t", max_attempts=0)
        claimed = store.lease_next_task("w")
        store.dead_letter("t", "w", claimed.lease_epoch, "E" * 200)

    _, clipped, _ = run(["--db", str(db_path), "dlq", "list"], capsys)
    assert "\u2026" in clipped
    assert "E" * 200 not in clipped

    _, wide, _ = run(["--db", str(db_path), "dlq", "list", "--wide"], capsys)
    assert "E" * 200 in wide
    assert "\u2026" not in wide


def test_dlq_list_json_is_machine_readable(seeded_dlq, capsys) -> None:
    code, out, _ = run(["--db", str(seeded_dlq), "dlq", "list", "--json"], capsys)

    assert code == 0
    entries = json.loads(out)
    assert [entry["id"] for entry in entries] == ["dead-1", "dead-0"]

    newest = entries[0]
    assert newest["priority"] == 3
    assert newest["attempts"] == 1
    assert newest["last_error"] == "failure 1"
    assert newest["payload"] == {"type": "work", "i": 1}
    assert newest["failed_at_iso"]
    # The payload BLOB must be decoded, not left as a non-serializable bytes.
    assert isinstance(newest["payload"], dict)
    # Vestigial columns were dropped in schema v3; the API must not advertise
    # a revive history that nothing ever recorded.
    assert "revive_count" not in newest
    assert "revived_at" not in newest


def test_dlq_list_honours_limit(seeded_dlq, capsys) -> None:
    _, out, _ = run(
        ["--db", str(seeded_dlq), "dlq", "list", "--limit", "1", "--json"], capsys
    )
    assert len(json.loads(out)) == 1


# ================================================================= dlq replay


def test_replay_moves_the_task_back_into_the_queue(seeded_dlq, capsys) -> None:
    code, out, _ = run(["--db", str(seeded_dlq), "dlq", "replay", "dead-0"], capsys)

    assert code == 0
    assert "replayed dead-0" in out

    with Store(seeded_dlq) as store:
        task = store.get_task("dead-0")
        assert task.state is TaskStatus.PENDING
        assert task.attempts == 0
        assert task.available_at <= time.time()
        # Removed from the DLQ, and the other entry is untouched.
        assert [entry["id"] for entry in store.list_dead_letters()] == ["dead-1"]


def test_replay_preserves_priority(seeded_dlq, capsys) -> None:
    run(["--db", str(seeded_dlq), "dlq", "replay", "dead-1"], capsys)
    with Store(seeded_dlq) as store:
        assert store.get_task("dead-1").priority == 3


def test_replay_honours_max_attempts(seeded_dlq, capsys) -> None:
    run(
        [
            "--db", str(seeded_dlq), "dlq", "replay", "dead-0",
            "--max-attempts", "9",
        ],
        capsys,
    )
    with Store(seeded_dlq) as store:
        assert store.get_task("dead-0").max_attempts == 9


def test_replay_unknown_id_is_an_error(seeded_dlq, capsys) -> None:
    code, _, err = run(["--db", str(seeded_dlq), "dlq", "replay", "nope"], capsys)

    assert code == 1
    assert "no dead-lettered task with id 'nope'" in err
    with Store(seeded_dlq) as store:
        assert len(store.list_dead_letters()) == 2


def test_replay_requires_an_id_or_all(seeded_dlq, capsys) -> None:
    code, _, err = run(["--db", str(seeded_dlq), "dlq", "replay"], capsys)
    assert code == 1
    assert "pass a task id" in err


def test_replay_rejects_an_id_and_all_together(seeded_dlq, capsys) -> None:
    code, _, err = run(
        ["--db", str(seeded_dlq), "dlq", "replay", "dead-0", "--all"], capsys
    )
    assert code == 1
    assert "not both" in err


def test_replay_all_drains_the_dlq(seeded_dlq, capsys) -> None:
    code, out, _ = run(["--db", str(seeded_dlq), "dlq", "replay", "--all"], capsys)

    assert code == 0
    assert "replayed 2 task(s), skipped 0" in out
    with Store(seeded_dlq) as store:
        assert store.list_dead_letters() == []
        assert store.count_tasks() == 2


def test_replay_all_on_an_empty_dlq_is_a_no_op(db_path, capsys) -> None:
    code, out, _ = run(["--db", str(db_path), "dlq", "replay", "--all"], capsys)

    assert code == 0
    assert "replayed 0 task(s), skipped 0" in out


def test_replay_all_honours_limit(seeded_dlq, capsys) -> None:
    code, out, _ = run(
        ["--db", str(seeded_dlq), "dlq", "replay", "--all", "--limit", "1"], capsys
    )

    assert code == 0
    assert "replayed 1 task(s)" in out
    with Store(seeded_dlq) as store:
        assert len(store.list_dead_letters()) == 1


def test_replay_all_skips_conflicts_without_aborting(db_path, capsys) -> None:
    """One conflicting row must not block the rest of the batch."""
    with Store(db_path) as store:
        for index in (0, 1):
            store.enqueue(
                {"type": "x", "i": index}, task_id=f"c-{index}", max_attempts=0
            )
            task = store.lease_next_task(f"w{index}")
            store.dead_letter(task.id, f"w{index}", task.lease_epoch, "boom")
        # c-0 is active again, so replaying it would clobber a live task.
        store.enqueue({"type": "x", "v": 2}, task_id="c-0")

    code, out, err = run(["--db", str(db_path), "dlq", "replay", "--all"], capsys)

    assert code == 0
    assert "replayed 1 task(s), skipped 1" in out
    assert "skipped c-0" in err
    assert "already active" in err

    with Store(db_path) as store:
        # The conflicted row survives for a human to look at, and the live task
        # was not overwritten.
        assert [entry["id"] for entry in store.list_dead_letters()] == ["c-0"]
        assert store.get_task("c-0").decode_payload() == {"type": "x", "v": 2}


def test_replay_all_exits_nonzero_when_everything_is_skipped(db_path, capsys) -> None:
    with Store(db_path) as store:
        store.enqueue({"type": "x"}, task_id="c", max_attempts=0)
        claimed = store.lease_next_task("w")
        store.dead_letter("c", "w", claimed.lease_epoch, "boom")
        store.enqueue({"type": "x", "v": 2}, task_id="c")

    code, out, err = run(["--db", str(db_path), "dlq", "replay", "--all"], capsys)

    assert code == 1
    assert "replayed 0 task(s), skipped 1" in out
    assert "skipped c" in err


# ============================================================== wiring, misc


def test_db_option_is_accepted_before_and_after_the_subcommand(
    tmp_path, capsys
) -> None:
    before = tmp_path / "before.db"
    after = tmp_path / "after.db"

    assert run(["--db", str(before), "enqueue", "t", "{}"], capsys)[0] == 0
    assert run(["enqueue", "--db", str(after), "t", "{}"], capsys)[0] == 0
    assert before.exists() and after.exists()


def test_leaf_db_option_overrides_the_global_one(tmp_path, capsys) -> None:
    """argparse applies subparser defaults last: the leaf must not clobber."""
    unused = tmp_path / "unused.db"
    used = tmp_path / "used.db"

    code, _, _ = run(
        ["--db", str(unused), "enqueue", "--db", str(used), "t", "{}"], capsys
    )

    assert code == 0
    assert used.exists()
    assert not unused.exists()


def test_env_var_supplies_the_database(tmp_path, capsys, monkeypatch) -> None:
    db = tmp_path / "env.db"
    monkeypatch.setenv("PULSE_QUEUE_DB", str(db))

    code, _, _ = run(["enqueue", "t", "{}"], capsys)

    assert code == 0
    assert db.exists()


def test_unknown_subcommand_is_a_usage_error(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["bogus"])
    assert excinfo.value.code == 2


def test_version_exits_zero(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "pulse-queue" in capsys.readouterr().out


# ========================================================= subprocess coverage


@pytest.mark.integration
def test_console_script_is_wired_up(tmp_path) -> None:
    """``[project.scripts]`` must produce a working ``pulse-queue`` command."""
    script = console_script()

    version = subprocess.run(
        [str(script), "--version"], capture_output=True, text=True
    )
    assert version.returncode == 0
    assert "pulse-queue" in version.stdout

    db = tmp_path / "cli.db"
    enqueued = subprocess.run(
        [str(script), "--db", str(db), "enqueue", "add", '{"a": 1}'],
        capture_output=True,
        text=True,
    )
    assert enqueued.returncode == 0, enqueued.stderr
    task_id = enqueued.stdout.strip()

    listed = subprocess.run(
        [str(script), "--db", str(db), "dlq", "list"], capture_output=True, text=True
    )
    assert listed.returncode == 0
    assert "no dead-lettered tasks" in listed.stdout

    with Store(db) as store:
        assert store.get_task(task_id).state is TaskStatus.PENDING


@pytest.mark.integration
def test_work_command_processes_tasks_and_stops_cleanly(tmp_path) -> None:
    """The whole CLI path: import handlers, run a pool, drain, exit 0."""
    module = tmp_path / "cli_e2e_handlers.py"
    module.write_text(textwrap.dedent(REGISTRY_MODULE))

    db = tmp_path / "pulse.db"
    with Store(db) as store:
        for index in range(3):
            store.enqueue({"type": "work", "i": index}, task_id=f"w-{index}")

    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    proc = subprocess.Popen(
        [
            str(console_script()), "--db", str(db), "work",
            "--handlers", "cli_e2e_handlers",
            "--concurrency", "2",
            "--poll-interval", "0.02",
            "--lease-ttl", "10",
            "--grace-period", "2",
            "--log-level", "ERROR",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    stderr = ""
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            with Store(db) as store:
                if store.count_tasks(TaskStatus.COMPLETED) == 3:
                    break
            time.sleep(0.05)
        else:
            proc.kill()
            proc.wait(timeout=10)
            raise AssertionError("tasks never completed")

        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)

    assert proc.returncode == 0, stderr
    with Store(db) as store:
        assert store.count_tasks(TaskStatus.COMPLETED) == 3
        assert store.count_tasks(TaskStatus.RUNNING) == 0
