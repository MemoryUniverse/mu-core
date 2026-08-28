# Releasing — proposed convention

**Status: a proposal, not yet in force.** Nothing in Memory Universe has ever been released:
there are **zero git tags in every repository**, and of the distributions this repo ships,
`mu-contracts`, `mu-engine`, `mu-local` and `mu-engine-server` are all **unclaimed on PyPI**
(all four `404` as of 2026-08-28). This file exists so the first tag is a decision someone made
on purpose, rather than whatever the first person to type `git tag` happened to choose.

A maintainer should ratify or amend this before the first tag is cut. **Do not cut one against
this file while it still says "proposal".**

## Versioning

Semantic Versioning, `v`-prefixed tags: `v0.1.0`, `v0.2.0`, `v1.0.0`.

Pre-1.0 the usual pre-1.0 rules apply and should be said out loud rather than assumed:

- `0.MINOR.PATCH`. A **minor** bump may break anything, including the wire contracts.
- A **patch** bump never changes a contract, a schema, or an on-disk format.
- No compatibility promise is made until `v1.0.0`. The README says the project is early; the
  version number should say the same thing.

## One tag, four distributions

`mu-core` is a uv workspace that ships four distributions. They are versioned **in lockstep** and
released under a **single repository-level tag**, because they are not independently useful:
`mu-engine` is meaningless without the exact `mu-contracts` it was built against, and a consumer
resolving them separately is a bug report waiting to happen.

So: tag `v0.1.0`, publish `mu-contracts==0.1.0`, `mu-engine==0.1.0`, `mu-local==0.1.0`,
`mu-engine-server==0.1.0` from it.

If a distribution ever genuinely needs its own cadence, the escape hatch is a package-scoped tag
(`mu-contracts/v0.3.1`) — reserved deliberately here so that the day it is needed, the convention
already exists and the repo-level tags stay unambiguous.

## The procedure

1. Land everything on the integration trunk. CI green.
2. Bump `version` in all four `packages/*/pyproject.toml` in one commit:
   `chore(release): v0.1.0`.
3. `uv lock` and commit the lockfile in the same commit.
4. Annotated tag on that commit: `git tag -a v0.1.0 -m "mu-core v0.1.0"`.
5. Push the commit, then the tag.
6. GitHub Release from the tag. Notes grouped by Conventional-Commit type (`feat`, `fix`,
   `BREAKING CHANGE` first), written for someone deciding whether to upgrade — not a commit dump.
7. Attach the sdists and wheels the `build` job already produced.

Steps 4-7 are manual today, on purpose. A publish workflow is the right thing to add **after** the
first release has been done by hand once and the blockers below are cleared — automating a
procedure nobody has performed yet is how repositories acquire a `release.yml` that has never run.

## Cross-repo ordering

Four public repos depend on this one by **filesystem path**, not by version. That has to be undone
in a fixed order or the first release is unusable:

1. `mu-core` publishes `mu-contracts`, `mu-engine`, `mu-local`, `mu-engine-server`, and tags.
2. `mu-client` and `mu-sdk-python` replace their `[tool.uv.sources]` path dependencies with version
   ranges (`mu-contracts>=0.1,<0.2`), re-lock, and only then tag.
3. `mu-sdk-js` has no cross-repo path dependency and can tag independently, but should match the
   Python SDK's minor version so `0.1.x` means the same wire surface in both languages.

Tagging step 2 before step 1 produces a package that installs on exactly one laptop.

## Blockers before `v0.1.0`

Verified 2026-08-28; each is a fact, not an estimate:

- **`mu-sdk` is taken on PyPI** (`pypi.org/pypi/mu-sdk` → `200`, an unrelated project). The Python
  SDK cannot publish under its current name. The npm name `mu-sdk` **is** free (`404`). Resolving
  this renames a package in two SDK repos, both READMEs, and every example.
- **The path dependencies above.** A fresh `git clone` of `mu-client` or `mu-sdk-python` followed by
  the `uv sync` its README prints cannot resolve `mu-contracts`, because there is no `../mu-core`.
- **`mu-sdk-python` ships no `py.typed`.** All four `mu-core` distributions do; the Python SDK does
  not, so a typed consumer gets `Any` from a package whose whole selling point is a typed client.
- **No dependency vulnerability or license scan runs anywhere.** `mu-engine` depends on FalkorDB,
  whose license is SSPL — a license the project has already committed to revisiting *before* the
  open-source gate. Publishing without that decision recorded is the one blocker here that is not
  merely mechanical.

## What a release is not

A tag is not a claim that the hosted plane works, that a migration has been run, or that anything
in this repo has been deployed. It is a claim that this exact tree built, passed its gates, and
was published under that version number. Everything else belongs in the release notes as prose,
where it can be qualified.
