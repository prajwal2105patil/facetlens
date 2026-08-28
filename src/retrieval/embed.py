"""Facet embedding index.

Embeddings are computed once and cached to disk, keyed by a hash of the exact
input texts plus the model name. Changing the enriched catalogue or the model
invalidates the cache automatically; nothing else does.

Model: sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0, 22.7M params).
Chosen because it is small, CPU-fast (399 facets in ~0.13s on this machine) and
already vendored in the local HuggingFace cache, so retrieval has no network
dependency at run time.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Cosmetic only: this cache legitimately cannot use symlinks on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ENRICHED_CSV = Path("data/processed/enriched_facets.csv")
CACHE_DIR = Path("artifacts/embeddings")


def _cache_paths(text_column: str) -> tuple[Path, Path]:
    """One cache slot per retrieval-text variant.

    The ablation builds an index over a different column; sharing a single path
    would make the two variants evict each other on every alternation. The
    fingerprint check keeps that *correct* either way, but it wastes a full
    rebuild each time.
    """
    stem = f"facet_matrix__{text_column}"
    return CACHE_DIR / f"{stem}.npy", CACHE_DIR / f"{stem}.meta.json"


@dataclass
class FacetIndex:
    """Facet rows, their embedding matrix, and a lazily-built BM25 index."""

    facet_ids: list[str]
    rows: list[dict]
    matrix: np.ndarray  # shape (n_facets, dim), unit-norm rows
    text_column: str = "retrieval_text"
    _bm25: object = None

    def by_id(self, facet_id: str) -> dict:
        return self.rows[self.facet_ids.index(facet_id)]

    @property
    def bm25(self):
        """Built on first use: ~10ms for 399 facets, and only the hybrid
        retriever needs it, so dense-only runs never pay for it."""
        if self._bm25 is None:
            from .lexical import BM25Index

            self._bm25 = BM25Index([r[self.text_column] for r in self.rows])
        return self._bm25


def load_enriched(path: Path = ENRICHED_CSV) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.pipeline enrich` first."
        )
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # csv gives strings; restore the booleans the rest of the pipeline relies on.
    for row in rows:
        for flag in ("conversation_observable", "is_header_like",
                     "has_numeric_prefix", "has_encoding_artifact"):
            row[flag] = row[flag] == "True"
    return rows


def _fingerprint(texts: list[str], model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for text in texts:
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


@functools.lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load the encoder once per process, from the local snapshot when possible.

    Two separate problems are being solved here, both measured:

    1. Loading MiniLM costs ~25s; embedding a query afterwards costs ~30ms.
       Without the lru_cache, a re-load per call dominates the entire run.

    2. Passing a *repo id* makes sentence-transformers resolve it against the
       HuggingFace Hub even when the weights are already cached. Unauthenticated
       Hub requests get rate-limited, and the retries pushed one index build to
       410s. Setting HF_HUB_OFFLINE from inside this function is too late -
       huggingface_hub freezes that flag into a module constant at import time.
       Resolving the cached snapshot to a concrete local path sidesteps Hub
       resolution entirely. See DEBUGGING.md #2.
    """
    from sentence_transformers import SentenceTransformer

    try:
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(model_name, local_files_only=True)
        return SentenceTransformer(local_path)
    except Exception:
        # Not in the local cache (e.g. a genuinely fresh clone) - allow one
        # online fetch rather than failing with an opaque offline error.
        return SentenceTransformer(model_name)


def build_index(force: bool = False, model_name: str = EMBED_MODEL,
                text_column: str = "retrieval_text") -> FacetIndex:
    """Return the facet index, rebuilding the embedding cache only if stale."""
    rows = load_enriched()
    texts = [row[text_column] for row in rows]
    fingerprint = _fingerprint(texts, f"{model_name}:{text_column}")
    matrix_path, meta_path = _cache_paths(text_column)

    if not force and matrix_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            matrix = np.load(matrix_path)
            if matrix.shape[0] == len(rows):
                return FacetIndex([r["facet_id"] for r in rows], rows,
                                  matrix, text_column)

    model = _load_model(model_name)
    matrix = model.encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
    ).astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix)
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "model": model_name,
                "text_column": text_column,
                "n_facets": len(rows),
                "dim": int(matrix.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return FacetIndex([r["facet_id"] for r in rows], rows, matrix, text_column)


def embed_query(text: str, model_name: str = EMBED_MODEL) -> np.ndarray:
    model = _load_model(model_name)
    return model.encode([text], normalize_embeddings=True).astype(np.float32)[0]
