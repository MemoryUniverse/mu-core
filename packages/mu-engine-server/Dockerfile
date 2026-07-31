# syntax=docker/dockerfile:1.7
#
# mu-engine-server — thin single-tenant HTTP server over the SurfaceFacade (mu-engine),
# containerized (design 2026-07-31-sdk-engine-server-design.md §7.1, build-plan §4 Stage E, E1).
#
# Import boundary this image respects (design §2.2/§8, .importlinter mu-engine-server-boundary):
# mu_engine_server -> {mu_contracts, mu_engine} ONLY. mu-local's workspace member must be
# PRESENT during the builder stage (root pyproject.toml's [tool.uv.workspace] declares 4 members
# and `uv sync` refuses to resolve if one is missing on disk) but its source is never copied into
# the runtime stage and its package is never installed into the venv `--package mu-engine-server`
# builds (verify: `docker run --rm mu-engine-server:latest python -c "import mu_local"` ->
# ModuleNotFoundError; `docker run --rm mu-engine-server:latest python -c "import mu_server"` /
# `mu_client` likewise — those two aren't even present in this workspace).
#
# Build context MUST be the mu-core workspace root, not this directory — mu-engine-server depends
# on sibling workspace packages that live outside packages/mu-engine-server/:
#   docker build -f packages/mu-engine-server/Dockerfile -t mu-engine-server:latest .   (run from mu-core/)
# docker-compose.yml / docker-compose.embedded.yml (this directory) set this via
# `build: {context: ../.., dockerfile: packages/mu-engine-server/Dockerfile}`.

FROM python:3.12-slim AS builder

# uv, pinned to the version this Dockerfile was authored/verified against.
COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

# WORKDIR is /app here — the SAME absolute path the runtime stage below uses — not /workspace.
# `uv sync` bakes an ABSOLUTE shebang (`#!<venv-path>/bin/python`) into every installed console
# script (verified empirically: `.venv/bin/uvicorn`'s first line was `#!/workspace/.venv/bin/
# python` when this stage's WORKDIR was `/workspace` and the runtime stage copied the venv to
# `/app/.venv` — every script in it, uvicorn included, then failed at container start with `exec
# /app/.venv/bin/uvicorn: no such file or directory`, since the interpreter path the shebang
# pointed at did not exist in the runtime image). Building at the SAME path the venv is later
# copied to makes the shebang correct in both stages with zero extra flags/patching.
WORKDIR /app

# Root workspace manifest + lockfile first — better layer-cache reuse on source-only edits below.
COPY pyproject.toml uv.lock ./

# All FOUR workspace members must be physically present for `uv sync` to resolve the workspace
# manifest (see header comment) — mu-local is present here ONLY for that resolution; it is never
# installed into this image (--package mu-engine-server below installs only mu-engine-server +
# its own declared dependency closure: mu-contracts, mu-engine).
COPY packages/mu-contracts packages/mu-contracts
COPY packages/mu-engine packages/mu-engine
COPY packages/mu-local packages/mu-local
COPY packages/mu-engine-server packages/mu-engine-server

# --no-editable: build real wheels for the workspace-local members instead of editable/path
# installs, so the runtime stage below needs only the resulting .venv — none of the source trees
# copied above (keeps mu-local's source, tests/, and dev tooling out of the shipped image).
RUN uv sync --frozen --no-dev --no-editable --package mu-engine-server

# -------------------------------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root (design §7.1 "thin image"; standard container hardening — nothing in this process needs
# root once dependencies are installed).
RUN useradd --create-home --home-dir /home/mu-engine-server --uid 1000 \
        --shell /usr/sbin/nologin mu-engine-server

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

# Stage E's uvicorn entrypoint shim — see docker/serve.py's own docstring for exactly why this is
# needed (no module-level `app` / no `/health` route exists anywhere in the landed
# mu_engine_server package for uvicorn to import directly; mu_engine_server/app.py's own docstring
# hands this wiring job to Stage E by name).
COPY packages/mu-engine-server/docker/serve.py /app/serve.py

RUN chown -R mu-engine-server:mu-engine-server /app /home/mu-engine-server

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HOME=/home/mu-engine-server

USER mu-engine-server
EXPOSE 8300

# --host 0.0.0.0 INSIDE the container is REQUIRED, not optional (design §7.2 FIX 5, REVIEW-2
# BLOCKER): bridge-network NAT maps a compose/`docker run -p` host-side port publish to the
# container's eth0, never to its lo — a process bound to 127.0.0.1 in-container is unreachable via
# any host port-map, silently ("container up, port unreachable"). The host-side loopback
# restriction (`ports: ["127.0.0.1:8300:8300"]` in the compose files, NOT this Dockerfile) is the
# SOLE, sufficient network-exposure control; binding 0.0.0.0 in here does not reopen it.
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8300"]
