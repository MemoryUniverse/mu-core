"""L5a — WarmLocalSingleton DCL single-load, verified on the REAL MiniLM (spec §8 test 4).

This mirrors MemOS `hf_singleton.py` behaviour against a real (small, offline) model — not a mock:
the same `model_id` loads ONCE (`get_instance_count() == 1`), a different id makes a second
instance, and `clear_all()` resets. The real load is the sentence-transformers MiniLM (kind=EMBED);
the singleton MECHANISM is identical to the causal-LM path (same class).
"""

from __future__ import annotations

import pytest

from mu_engine.providers.catalog import ModelKind, WarmLocalConfig
from mu_engine.providers.warm_local import WarmLocalSingleton, warm_singleton_info

pytestmark = pytest.mark.unit

MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def _embed_cfg(model_id: str) -> WarmLocalConfig:
    return WarmLocalConfig(model_id=model_id, kind=ModelKind.EMBED, model_load_path=MINILM)


def test_same_model_id_loads_once() -> None:
    cfg = _embed_cfg("minilm_local")
    a = WarmLocalSingleton(cfg)
    b = WarmLocalSingleton(cfg)
    c = WarmLocalSingleton(_embed_cfg("minilm_local"))  # same id, fresh cfg object
    assert a is b is c  # identity — DCL returned the cached instance
    assert WarmLocalSingleton.get_instance_count() == 1


def test_different_model_id_makes_second_instance() -> None:
    WarmLocalSingleton(_embed_cfg("minilm_local"))
    WarmLocalSingleton(_embed_cfg("minilm_alias"))  # same weights path, different deployment id
    assert WarmLocalSingleton.get_instance_count() == 2
    info = warm_singleton_info()
    assert info["instance_count"] == 2
    assert set(info["instance_info"]) == {"minilm_local", "minilm_alias"}


def test_clear_all_resets() -> None:
    WarmLocalSingleton(_embed_cfg("minilm_local"))
    assert WarmLocalSingleton.get_instance_count() == 1
    WarmLocalSingleton.clear_all()
    assert WarmLocalSingleton.get_instance_count() == 0


def test_singleton_actually_loaded_the_real_model() -> None:
    s = WarmLocalSingleton(_embed_cfg("minilm_local"))
    assert s.embedding_dimension() == 384  # real MiniLM dimension, read from the model
    vecs = s.embed(["hello", "world"])
    assert len(vecs) == 2
    assert all(len(v) == 384 for v in vecs)
