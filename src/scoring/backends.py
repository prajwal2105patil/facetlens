"""LLM backends behind one narrow interface.

Two implementations:

  OllamaBackend  local qwen2.5:7b-instruct (Apache-2.0, 7.6B params)
  MockBackend    deterministic, no model, no network

MockBackend is not a toy. It lets the whole pipeline - routing, batching,
parsing, validation, verification, evaluation - be tested and demonstrated on a
machine with no model installed, and it can be told to emit malformed payloads
so the error paths are exercised on purpose rather than hoped about.

Responses are cached on disk keyed by (model, options, prompt). The cache is
committed to the repository so the benchmark report reproduces offline; see
DECISIONS.md D5.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Protocol

CACHE_DIR = Path("artifacts/llm_cache")


class LLMBackend(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        """Return the raw text of the model's response."""
        ...


def _cache_key(model: str, system: str, user: str, options: dict) -> str:
    digest = hashlib.sha256()
    for part in (model, system, user, json.dumps(options, sort_keys=True)):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:32]


class OllamaBackend:
    """Local inference via the Ollama HTTP API.

    Determinism: temperature=0 and a fixed seed. Structured decoding is
    requested by passing the Pydantic JSON schema as `format`, which constrains
    generation rather than merely asking politely for JSON. Parsing is still
    defensive - see parser.py - because constrained decoding can still truncate.
    """

    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b-instruct", seed: int = 42,
                 temperature: float = 0.0, num_ctx: int = 4096,
                 use_cache: bool = True) -> None:
        self.model = model
        self.use_cache = use_cache
        self.options = {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
        }

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        import ollama

        # The schema is part of the request (it constrains decoding), so its
        # CONTENT must be in the cache key. Keying on merely whether a schema
        # was passed would silently serve responses generated under an older
        # grammar after the contract changed.
        key = _cache_key(self.model, system, user, self.options | {"schema": schema})
        cache_path = CACHE_DIR / f"{key}.json"
        if self.use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["response"]

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=schema if schema else "json",
            options=self.options,
        )
        text = response["message"]["content"]

        if self.use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {"model": self.model, "options": self.options,
                     "system": system, "user": user, "response": text},
                    indent=2, ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return text


class MockBackend:
    """Deterministic stand-in used by the test suite and offline demos.

    `mode` selects the failure being exercised:
      "valid"      well-formed verdicts for every requested facet
      "malformed"  syntactically broken JSON
      "partial"    valid JSON that omits some requested facets
      "bad_schema" valid JSON violating the contract (score set while abstaining)
      "fabricated" a quote that does NOT appear in the conversation
    """

    name = "mock"

    def __init__(self, model: str = "mock", mode: str = "valid") -> None:
        self.model = model
        self.mode = mode

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        facet_ids = re.findall(r"facet_id=(\S+)", user)

        if self.mode == "malformed":
            return '{"verdicts": [{"facet_id": "F0001", "status": '  # truncated

        if self.mode == "bad_schema":
            return json.dumps({"verdicts": [
                {"facet_id": fid, "status": "insufficient_evidence", "score": 3,
                 "confidence": 0.5, "evidence_quote": "", "reason": "contradictory"}
                for fid in facet_ids
            ]})

        if self.mode == "fabricated":
            return json.dumps({"verdicts": [
                {"facet_id": fid, "status": "scored", "score": 4, "confidence": 0.9,
                 "evidence_quote": "a sentence that is definitely not in the transcript",
                 "reason": "invented evidence"}
                for fid in facet_ids
            ]})

        if self.mode == "partial":
            facet_ids = facet_ids[:1]

        # "valid": echo back a plausible, contract-compliant verdict. The quote is
        # taken from the real conversation so the evidence verifier passes.
        match = re.search(r'"""\n(.*?)\n"""', user, re.DOTALL)
        conversation = match.group(1) if match else ""
        quote = conversation.split(".")[0].strip()[:80]
        return json.dumps({"verdicts": [
            {"facet_id": fid, "status": "scored", "score": 3, "confidence": 0.6,
             "evidence_quote": quote, "reason": "mock deterministic verdict"}
            for fid in facet_ids
        ]})


def get_backend(name: str, model: str, **kwargs) -> LLMBackend:
    if name == "ollama":
        return OllamaBackend(model=model, **kwargs)
    if name == "mock":
        return MockBackend(model=model, **kwargs)
    raise ValueError(f"Unknown backend {name!r}; expected 'ollama' or 'mock'")
