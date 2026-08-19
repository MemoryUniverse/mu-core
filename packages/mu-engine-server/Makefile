# mu-engine-server — `make up` deliverable (design §7.3, build-plan §4 Stage E, E1).
#
# `make up` is the one-command promise: after it returns, `mode="local_server",
# endpoint="http://localhost:8300"` (or the JS equivalent) works immediately against real stores.
#
# COMPOSE_FILE selects the profile: docker-compose.yml (real stores, RECOMMENDED DEFAULT) or
# docker-compose.embedded.yml (option, not recommended — see that file's own header for the
# honesty note on what "embedded" actually means here). Override: `make up COMPOSE_FILE=docker-compose.embedded.yml`.

SHELL := /bin/bash
COMPOSE_FILE ?= docker-compose.yml
COMPOSE := docker compose -f $(COMPOSE_FILE)

TOKEN_DIR := $(HOME)/.memory-universe
TOKEN_FILE := $(TOKEN_DIR)/engine-server.token

# The image's runtime user is built with the HOST's uid/gid so it can read the 0600 bearer token
# bind-mounted into the container (a 0600 file is readable only by its owning uid). Without this
# the server builds and reports healthy but 401s every authenticated request on any host whose
# user is not uid 1000.
export MU_APP_UID := $(shell id -u)
export MU_APP_GID := $(shell id -g)

.PHONY: up down logs reset mint-token check-boundary

# `make up`: (1) mint the per-process bearer token to disk if absent, mode 0600 (design §2.4/§11.2,
#            §13 item 7 — "minted on first `make up`"); (2) bring the compose stack up;
#            (3) print the endpoint + how to point an SDK at it.
up: mint-token
	$(COMPOSE) up -d --build
	@echo ""
	@echo "mu-engine-server is up: http://127.0.0.1:8300"
	@echo "bearer token:           $(TOKEN_FILE)"
	@echo ""
	@echo "Point an SDK at it (auth auto-loads the token from disk, design §1.2 FIX 4):"
	@echo '  Python: MemoryClient(mode="local_server", endpoint="http://localhost:8300")'
	@echo '  JS:     new MemoryClient({ mode: "local_server", endpoint: "http://localhost:8300" })'
	@echo ""
	@echo "Or curl it directly:"
	@echo '  curl http://127.0.0.1:8300/health'
	@echo '  curl -H "Authorization: Bearer $$(cat $(TOKEN_FILE))" http://127.0.0.1:8300/profile'

# Mint the per-process bearer token (design §2.4 candidate (a): a per-process token generated
# locally and written to disk on first `make up`) — idempotent, never overwrites an existing token
# (rotating it would invalidate every SDK client already holding the old value with no notice).
mint-token:
	@mkdir -p "$(TOKEN_DIR)"
	@if [ ! -s "$(TOKEN_FILE)" ]; then \
		umask 177; \
		python3 -c "import secrets; print(secrets.token_urlsafe(32))" > "$(TOKEN_FILE)"; \
		chmod 600 "$(TOKEN_FILE)"; \
		echo "minted new bearer token -> $(TOKEN_FILE) (mode 0600)"; \
	else \
		chmod 600 "$(TOKEN_FILE)"; \
		echo "bearer token already present -> $(TOKEN_FILE) (left unchanged)"; \
	fi

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f engine

# Wipes store volumes (valkey/qdrant/falkordb data + the HF embedder cache), fresh start. Does NOT
# touch the bearer token on disk (that is identity, not store state) — use `rm $(TOKEN_FILE)` +
# `make up` if a fresh token is genuinely wanted.
reset:
	$(COMPOSE) down -v
	$(COMPOSE) up -d --build

# E2 — dependency-free grep backstop (design §8, build-plan §4 Stage E, E2). The `import-linter`
# contract `mu-engine-server-boundary` (mu-core/.importlinter, already landed by C0 — this target
# does not modify it, only adds a second, tool-independent check of the same invariant) already
# enforces `mu_engine_server -> {mu_server, mu_client, mu_local}` = forbidden. This target is the
# "dependency-free" backstop named in design §8's own bash snippet — it needs nothing installed
# beyond grep, so it still catches a violation even in a context where `import-linter` itself
# wasn't run/installed (a plain `grep` step in any CI job, no Python env setup required). Wire this
# target into CI as its own step (in addition to, never instead of, `import-linter` itself).
check-boundary:
	@! grep -rEn '^\s*(from|import)\s+mu_(server|client|local)\b' src \
		&& echo "check-boundary: OK — no mu_server/mu_client/mu_local imports under mu-engine-server/src"
