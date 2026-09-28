---
name: release-check
description: Pre-release validation pipeline for pulse-queue. Runs linters, types, migrations, unit tests, and wheel validation.
---

# Release Check Skill

When invoked:
1. Run `uv run ruff check .`.
2. Run syntax verification via `uv run python -m compileall src tests`.
3. Run the full pytest suite with `uv run pytest`.
4. Run `uv build` and test the CLI invocation with `uv run pulse-queue --help`.
5. Print the release readiness Markdown table.
