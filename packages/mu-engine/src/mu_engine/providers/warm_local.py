"""L5 — Startup-warm in-process local models (model-layer-spec §2.5).

The ONLY layer that holds weights in OUR process (LiteLLM calls endpoints, never holds weights).
Two parts:
  (a) `WarmLocalSingleton` — a FAITHFUL port of MemOS `HFSingletonLLM` (the double-checked-locking
      singleton) + `HFLLM` (the transformers load + generate loop).
  (b) `WarmLocalCustomLLM` — wraps the singleton as a LiteLLM `CustomLLM` and is registered in
      `custom_provider_map`, so the Router addresses warm weights as an ordinary order-1
      deployment with NO HTTP endpoint (research §1.5 Path A).

PORT CITATIONS (read line-for-line from the read-only reference, CODE-ADOPTION step 2):
  singleton:  other_repos/MemOS/src/memos/llms/hf_singleton.py
              :13,19-20,22-42,44-55,57-70,72-100,104-114
  load+gen :  other_repos/MemOS/src/memos/llms/hf.py:33-45,57-73,95-126

DELIBERATE DEVIATIONS (CODE-ADOPTION step 4):
  1. Config-key source: MemOS keys on `HFLLMConfig.model_name_or_path` (hf_singleton.py:69); we key
     on our `WarmLocalConfig.model_id` (the deployment id). Recorded at `_get_config_key`.
  2. Kind dispatch: MemOS's singleton loads only a causal LM; we extend the SAME singleton to also
     load a sentence-transformers embedder (`kind=EMBED`) for the EmbeddingPort seam. The causal
     path is a byte-faithful port; the embed path is our documented extension.
  3. Heavy imports (`transformers`, `sentence_transformers`, `torch`) are LAZY (inside `_load`) so
     importing this module never pulls torch — DEV-STANDARDS hygiene; behaviour unchanged.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, ClassVar

from mu_engine.providers._contracts import Message, Vector
from mu_engine.providers.catalog import ModelKind, WarmLocalConfig

__all__ = ["WarmLocalCustomLLM", "WarmLocalSingleton", "warm_singleton_info"]


class WarmLocalSingleton:
    """Singleton HF/ST model loader (ports MemOS `HFSingletonLLM` + `HFLLM`).

    Loads weights ONCE per `model_id`; repeated construction with the same id returns the same
    instance (hf_singleton.py:22-42 DCL). `kind=LLM` → causal generate (hf.py); `kind=EMBED` →
    sentence-transformers embed (our extension).
    """

    # hf_singleton.py:19-20 — class-level instance table + lock (double-checked locking)
    _instances: ClassVar[dict[str, WarmLocalSingleton]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, cfg: WarmLocalConfig) -> WarmLocalSingleton:
        # hf_singleton.py:22-42 — reuse existing instance if the config key is known; DCL otherwise
        config_key = cls._get_config_key(cfg)
        if config_key in cls._instances:
            return cls._instances[config_key]
        with cls._lock:
            if config_key in cls._instances:  # double-check under the lock
                return cls._instances[config_key]
            instance = super().__new__(cls)
            cls._instances[config_key] = instance
            return instance

    def __init__(self, cfg: WarmLocalConfig) -> None:
        # hf_singleton.py:50-51 — initialise only once (weights load ONCE, hf.py:40-45)
        if hasattr(self, "_initialized"):
            return
        self.config = cfg
        self._model: Any = None
        self._tokenizer: Any = None
        self._load()
        self._initialized = True

    @classmethod
    def _get_config_key(cls, cfg: WarmLocalConfig) -> str:
        # hf_singleton.py:57-70 — DEVIATION: key on our deployment id, not model_name_or_path
        return cfg.model_id

    # ---- load (hf.py:33-45 for LLM; sentence-transformers for EMBED) ----------------------
    def _load(self) -> None:
        if self.config.kind is ModelKind.EMBED:
            self._load_embedder()
        else:
            self._load_causal_lm()

    def _load_causal_lm(self) -> None:
        # hf.py:40-45 — AutoModelForCausalLM + AutoTokenizer, once
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy (deviation 3)

        # transformers' `from_pretrained` is untyped; bind through Any at this third-party edge.
        model_cls: Any = AutoModelForCausalLM
        tokenizer_cls: Any = AutoTokenizer

        self._model = model_cls.from_pretrained(
            self.config.model_load_path,
            torch_dtype=self.config.torch_dtype,  # hf.py:41
            device_map=self.config.device_map,  # hf.py:41
        )
        self._tokenizer = tokenizer_cls.from_pretrained(
            self.config.model_load_path, use_fast=True
        )  # hf.py:43-45

    def _load_embedder(self) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self._model = SentenceTransformer(self.config.model_load_path, device=self.config.device)

    # ---- generate (hf.py:57-73, 95-126) ---------------------------------------------------
    def generate(self, messages: Sequence[Message]) -> str:
        """Port of `HFLLM.generate`/`_generate_full` (hf.py:57-73, 95-126) — full-prompt path
        (no KV cache). Returns the decoded NEW tokens only (hf.py:116-120)."""
        if self.config.kind is not ModelKind.LLM:
            raise TypeError(f"generate() on a non-LLM singleton (kind={self.config.kind})")
        chat = [{"role": m.role.value, "content": m.content} for m in messages]
        prompt = self._tokenizer.apply_chat_template(  # hf.py:66-67
            chat, tokenize=False, add_generation_prompt=self.config.add_generation_prompt
        )
        return self._generate_full(prompt)

    def _generate_full(self, prompt: str) -> str:
        # hf.py:95-126 — build inputs, gen_kwargs, decode NEW tokens only
        inputs = self._tokenizer([prompt], return_tensors="pt").to(self._model.device)  # :103
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_tokens,  # hf.py:105
            "do_sample": self.config.do_sample,  # hf.py:106
        }
        if self.config.do_sample:  # hf.py:108-111
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_k"] = self.config.top_k
            gen_kwargs["top_p"] = self.config.top_p
        gen_ids = self._model.generate(**inputs, **gen_kwargs)  # hf.py:112-115
        new_ids = [
            out_ids[len(src_ids) :]
            for src_ids, out_ids in zip(inputs.input_ids, gen_ids, strict=False)
        ]  # hf.py:116-119
        return str(self._tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0])  # hf.py:120

    # ---- embed (our EMBED extension) ------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if self.config.kind is not ModelKind.EMBED:
            raise TypeError(f"embed() on a non-EMBED singleton (kind={self.config.kind})")
        vecs = self._model.encode(
            list(texts), normalize_embeddings=self.config.normalize_embeddings
        )
        return [[float(x) for x in row] for row in vecs]

    def embedding_dimension(self) -> int:
        if self.config.kind is not ModelKind.EMBED:
            raise TypeError("embedding_dimension() on a non-EMBED singleton")
        return int(self._model.get_sentence_embedding_dimension())

    # ---- class methods (hf_singleton.py:72-100) -------------------------------------------
    @classmethod
    def get_instance_count(cls) -> int:  # hf_singleton.py:72-80
        return len(cls._instances)

    @classmethod
    def get_instance_info(cls) -> dict[str, str]:  # hf_singleton.py:82-90
        return {key: inst.config.model_load_path for key, inst in cls._instances.items()}

    @classmethod
    def clear_all(cls) -> None:  # hf_singleton.py:92-100
        with cls._lock:
            cls._instances.clear()


def warm_singleton_info() -> dict[str, Any]:
    """Ports `get_hf_singleton_info()` (hf_singleton.py:104-114)."""
    return {
        "instance_count": WarmLocalSingleton.get_instance_count(),
        "instance_info": WarmLocalSingleton.get_instance_info(),
    }


class WarmLocalCustomLLM:
    """LiteLLM `CustomLLM` wrapper so the Router addresses a warm singleton as a normal
    deployment (model-layer-spec §2.5b, LiteLLM `custom_llm_server` docs). Inherits `CustomLLM`
    LAZILY (the base is only bound at construction) so importing this module needs no litellm.
    """

    def __new__(cls, singleton: WarmLocalSingleton) -> Any:
        import litellm  # lazy — only when a warm handler is actually built

        # Build a concrete CustomLLM subclass bound to this singleton. Faithful to LiteLLM's
        # documented in-process path: completion returns a ModelResponse; embedding returns an
        # EmbeddingResponse; NO HTTP endpoint is opened.
        singleton_ref = singleton

        class _Handler(litellm.CustomLLM):  # type: ignore[misc, name-defined]
            def _text(self, kwargs: dict[str, Any], args: tuple[Any, ...]) -> str:
                raw = kwargs.get("messages")
                if raw is None and len(args) >= 2:
                    raw = args[1]
                msgs = [Message(role=m["role"], content=m["content"]) for m in (raw or [])]
                return singleton_ref.generate(msgs)

            def _model_response(self, text: str, model: str) -> Any:
                from litellm.types.utils import Choices, ModelResponse
                from litellm.types.utils import Message as LMessage

                resp = ModelResponse()
                resp.model = model
                resp.choices = [
                    Choices(
                        index=0,
                        message=LMessage(role="assistant", content=text),
                        finish_reason="stop",
                    )
                ]
                return resp

            def completion(self, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model") or (args[0] if args else "mu-local")
                return self._model_response(self._text(kwargs, args), model)

            async def acompletion(self, *args: Any, **kwargs: Any) -> Any:
                model = kwargs.get("model") or (args[0] if args else "mu-local")
                return self._model_response(self._text(kwargs, args), model)

            def embedding(self, *args: Any, **kwargs: Any) -> Any:
                from litellm.types.utils import EmbeddingResponse

                model = kwargs.get("model") or (args[0] if args else "mu-local")
                texts = kwargs.get("input") or (args[1] if len(args) >= 2 else [])
                vecs = singleton_ref.embed(list(texts))
                resp = EmbeddingResponse()
                resp.model = model
                resp.data = [
                    {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)
                ]
                return resp

            async def aembedding(self, *args: Any, **kwargs: Any) -> Any:
                return self.embedding(*args, **kwargs)

        return _Handler()
