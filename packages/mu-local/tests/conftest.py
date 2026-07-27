"""mu-local test fixtures. The REAL end-to-end test drives ``LocalMemory`` over the live mu-dev-*
containers (redis + qdrant + falkordb) with the REAL offline MiniLM embedder — ZERO mocks
(DEV-STANDARDS: non-negotiable). Host ports come from the central ``Settings`` tree (``.env.test``),
never a hardcoded literal.
"""

from __future__ import annotations

import uuid

import pytest

from mu_contracts.config import Settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture
def uid() -> str:
    return uuid.uuid4().hex[:12]
