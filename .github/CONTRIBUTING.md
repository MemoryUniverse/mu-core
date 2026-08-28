# Contributing to mu-core

Thanks for looking. This repo is early and under active development, and the honest version of
"how to contribute" is short: **the gates are the contract.** If `ruff`, `ruff-format`,
`import-linter`, `mypy --strict` and the container-free test tier are green, and your change
does not cross a package boundary it should not cross, a maintainer can review it quickly.

## Before you open a large PR

Open an issue first. This repo is being built against a design set that is not (yet) public, and
a change that is right in isolation can still contradict a decision recorded elsewhere. A short
issue saves you from writing a patch that has to be turned down for a reason you had no way to see.

Small fixes — a bug, a wrong docstring, a broken example, a missing type — just send them.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv will fetch the interpreter for you;
this repo pins `>=3.12,<3.13`).

```bash
git clone -b dev/mlm-build https://github.com/MemoryUniverse/mu-core.git
cd mu-core
uv sync --locked
```

**`-b dev/mlm-build` is load-bearing**, and it is the same coupling `mu-client`'s and
`mu-sdk-python`'s CONTRIBUTING files and CI (`MU_CORE_REF`) already carry. GitHub's *default*
branch here is `main`; the *integration trunk* every other repo is built and tested against is
`dev/mlm-build`. A plain `git clone` lands you on `main`, where `uv sync --locked` still exits `0`
and installs **three** packages, not four — `packages/mu-engine-server/` does not exist there, and
`mu_contracts.contracts` is an empty scaffold, so `import mu_contracts.contracts.recall` raises
`ModuleNotFoundError`. Work started on `main` is work against a tree that cannot run.

`uv sync --locked` installs the four workspace packages (`mu-contracts`, `mu-engine`, `mu-local`,
`mu-engine-server`) plus the dev toolchain, and fails if `uv.lock` has drifted from
`pyproject.toml`. That is deliberate — a lockfile you have to remember to regenerate is a lockfile
that is wrong.

## The gates — run them before you push

These are exactly the commands CI runs, in the same order:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync lint-imports
uv run --no-sync mypy packages/mu-contracts/src packages/mu-engine/src \
                       packages/mu-local/src packages/mu-engine-server/src
uv run --no-sync pytest -m "not integration"
```

`--no-sync` matters. Without it `uv run` re-resolves the environment and can hand you a *different*
tool version than the one the lockfile pins — we have watched an unpinned `ruff` report format
diffs that the pinned `ruff` does not.

To fix formatting rather than just check it: `uv run --no-sync ruff format .`

## What CI actually runs

One workflow, [`ci.yml`](workflows/ci.yml), two jobs. Every command in it was run against a clean
clone of this repo — no sibling checkouts, no containers, no secrets — before the workflow was
committed:

| Gate | Command | Measured on a clean clone |
|---|---|---|
| lint | `ruff check .` | `All checks passed!` |
| format | `ruff format --check .` | `437 files already formatted` |
| boundaries | `lint-imports` | `Contracts: 5 kept, 0 broken.` (280 files, 1741 dependencies) |
| types | `mypy --strict` (4 src roots) | `Success: no issues found in 238 source files` (~4 min) |
| tests | `pytest -m "not integration"` | `1546 passed, 1 skipped, 274 deselected` (~60 s) |
| packaging | `uv build --all-packages` | 4 sdists + 4 wheels |

The five import-linter contracts are the load-bearing ones — they are how the package split stays
real rather than aspirational:

- `core-layers` — `mu_engine` may import `mu_contracts`, never the reverse.
- `mu-local-layers` — `mu_local` sits above both.
- `contracts imports nothing in-project` — `mu_contracts` is pydantic and nothing else.
- `mu-engine-server-boundary` — the HTTP server sees `mu_contracts` + `mu_engine` only; never
  `mu_local`, never `mu_server`, never `mu_client`.
- `persona-off-the-hot-path` — nothing that runs while a user is waiting may import the persona
  subsystem.

If your change breaks one of these, the fix is almost never to edit `.importlinter`.

## What CI does not run, and why

**Integration tests.** They are marked `integration` and are deselected in CI. They talk to real
Valkey, Qdrant, FalkorDB and Postgres instances — no mocks, ever, not even one — so they need a
machine with those containers up. They run on the project's dev VM. If you are working on
storage adapters and cannot run them, say so in the PR and a maintainer will run them for you.
They are not skipped because they are optional; they are skipped because a GitHub runner is the
wrong machine.

**The `acceptance` dependency group.** One acceptance test (F1, portability parity) drives the
public Python SDK against the containerized engine-server and needs the sibling `mu-sdk-python`
repo checked out next to this one. On a clean clone it reports itself skipped, with the reason,
rather than passing quietly:

```
SKIPPED [1] packages/mu-engine-server/tests/acceptance/test_f1_portability_parity.py:46:
  F1 needs the public mu-sdk-python package: run `uv sync --group acceptance` from the mu-core root
```

**A dependency vulnerability + license scan.** The project's engineering standards call for one.
It is not here because it does not pass today, and a gate that is red the day it lands is a gate
everyone learns to ignore. It is tracked as work, not hidden as a green checkmark.

## Conventions

- **Commits** follow Conventional Commits: `type(scope): subject` — `fix(recall): …`,
  `docs(readme): …`, `feat(storage)!: …` for a breaking change. Subject in the imperative.
- **Async everywhere** on I/O paths, with timeouts and cancellation handled, never blocking calls
  in the event loop.
- **pydantic, not dataclasses**, for anything that crosses a boundary.
- **No memory content in logs, traces, events or metrics.** This is not a style preference — it is
  the property that lets people run this on their own machine and mean it. A log line, error
  message or metric label that could carry a user's remembered text is a bug, and it is the one
  kind of bug reviewers here will block a PR over without discussion.
- **Tests must be able to fail.** If you add a test, satisfy yourself that mutating the line it
  guards actually turns it red. A test that cannot fail is worse than no test.

## Licensing

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](../LICENSE), the same license as this repository.
