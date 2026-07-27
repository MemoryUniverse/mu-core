"""StoreRegistry + build-time refusal tests (spec §8.4-§8.5, §8.10) — marked unit."""

from __future__ import annotations

import pytest

from mu_engine.storage.domain.namespace import Visibility
from mu_engine.storage.errors import (
    InvalidBackendError,
    MandatoryBackendMissingError,
    UnknownBackendError,
    VectorNotFilterableError,
)
from mu_engine.storage.registry import StoreRegistry, assert_mandatory_roles

pytestmark = pytest.mark.unit


def test_register_and_build() -> None:
    reg = StoreRegistry()

    @reg.register("vector", "fake")
    def _build(**cfg: object) -> str:
        return f"built:{cfg}"

    assert reg.build("vector", "fake", x=1) == "built:{'x': 1}"
    assert reg.known("vector") == frozenset({"fake"})


def test_unknown_backend_raises_named() -> None:
    reg = StoreRegistry()
    with pytest.raises(UnknownBackendError):
        reg.build("vector", "nope")


def test_mandatory_role_missing_refused() -> None:
    with pytest.raises(MandatoryBackendMissingError):
        assert_mandatory_roles(
            {"relational": "postgres", "vector": "qdrant"}, plane=Visibility.PRIVATE
        )  # graph missing


def test_graph_fold_refused() -> None:
    with pytest.raises(InvalidBackendError):
        assert_mandatory_roles(
            {"relational": "postgres", "vector": "qdrant", "graph": "sqlfold"},
            plane=Visibility.PRIVATE,
        )


def test_faiss_on_shared_refused() -> None:
    with pytest.raises(VectorNotFilterableError):
        assert_mandatory_roles(
            {"relational": "postgres", "vector": "faiss", "graph": "falkordb"},
            plane=Visibility.SHARED,
        )


def test_faiss_on_private_allowed() -> None:
    # PRIVATE plane: authorized_ids=None, isolation is the physical plane split (D3 note).
    assert_mandatory_roles(
        {"relational": "postgres", "vector": "faiss", "graph": "falkordb"},
        plane=Visibility.PRIVATE,
    )
