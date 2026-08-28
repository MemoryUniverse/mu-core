## What this changes

<!-- One paragraph. What is different after this PR that was not true before it? -->

## Why

<!-- The problem, not the patch. Link the issue if there is one: Fixes #123 -->

## How to see it fail without this change

<!--
The single most useful thing in this box. Name the test, or the exact command and the output you
saw before the fix. "Tests added" is not evidence; a test you watched go red is.
-->

## Gates

Run locally, on this branch:

- [ ] `uv run --no-sync ruff check .`
- [ ] `uv run --no-sync ruff format --check .`
- [ ] `uv run --no-sync lint-imports`
- [ ] `uv run --no-sync mypy packages/mu-contracts/src packages/mu-engine/src packages/mu-local/src packages/mu-engine-server/src`
- [ ] `uv run --no-sync pytest -m "not integration"`
- [ ] Integration tests (`pytest -m integration`, real stores) — ran / not applicable / could not run (say which)

## Checks that are not automatable

- [ ] **No memory content in logs, traces, events, metrics or error messages.** If this PR touches
      an observability or error path, say here what it emits.
- [ ] Every new store access is namespace-scoped.
- [ ] New I/O is async, has a timeout, and handles cancellation.
- [ ] Any new test can actually fail — I mutated the line it guards and watched it go red.
- [ ] Nothing in this PR weakens or disables an existing gate. (If a gate had to change, that is the
      PR's headline, not a footnote.)

## Anything a reviewer should push back on

<!-- Shortcuts taken, a design question you are unsure about, a boundary you had to bend. Saying it
     here is faster than being asked. -->
