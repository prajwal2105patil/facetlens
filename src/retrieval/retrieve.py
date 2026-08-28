"""Gate 1 (retrieval) and Gate 2 (observability).

The central discipline of this system is that these are DIFFERENT QUESTIONS:

  Gate 1  "Is this facet semantically related to the conversation?"
  Gate 2  "Can this facet legitimately be evidenced by a conversation at all?"

Semantic similarity is never treated as evidence. A conversation about feeling
tired retrieves 'FSH level' perfectly happily; Gate 2 is what stops it from
being scored. Because Gate 2 reads a precomputed catalogue column rather than
asking the model, the abstention guarantee does not depend on the model
behaving well - and it costs zero tokens.

Similarity search is exact (numpy dot product over unit-norm rows). See
DECISIONS.md D3 for why an ANN index would be premature at this scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embed import FacetIndex, build_index, embed_query
from .lexical import fuse
from .rerank import DEFAULT_POOL, rerank


@dataclass(frozen=True)
class Candidate:
    """A facet that survived retrieval, with its similarity score."""

    facet_id: str
    facet: str
    facet_type: str
    conversation_observable: bool
    abstention_reason: str | None
    sensitivity: str
    special_category: str | None
    scoring_definition: str
    score_anchors: str
    retrieval_score: float


@dataclass(frozen=True)
class RoutedFacets:
    """Result of running both gates over one conversation."""

    scorable: list[Candidate]      # -> Gate 3 (the LLM)
    gated_out: list[Candidate]     # -> deterministic not_observable, no LLM call
    policy_blocked: list[Candidate]  # -> refused by policy, no LLM call
    top_k: int


def _to_candidate(row: dict, score: float) -> Candidate:
    reason = row["abstention_reason"] or None
    return Candidate(
        facet_id=row["facet_id"],
        facet=row["facet_normalized"],
        facet_type=row["facet_type"],
        conversation_observable=row["conversation_observable"],
        abstention_reason=reason,
        sensitivity=row["sensitivity"],
        special_category=row.get("special_category") or None,
        scoring_definition=row["scoring_definition"],
        score_anchors=row["score_anchors"],
        retrieval_score=round(float(score), 4),
    )


def retrieve(conversation: str, index: FacetIndex | None = None,
             top_k: int = 25, mode: str = "dense") -> list[Candidate]:
    """Gate 1: top-K candidate facets. Relevance only, never evidence.

    `mode` selects the signal:
      "dense"   cosine over MiniLM embeddings
      "lexical" BM25 over the same facet text
      "hybrid"  reciprocal rank fusion of both
      "rerank"  dense to a wide pool, then cross-encoder reordering

    DEFAULT IS "dense", and every default here is justified by a number in
    artifacts/ablation_retrieval.md rather than by intuition. That rule exists
    because an earlier version defaulted to "hybrid" on the assumption that
    fusing signals must help; the ablation measured hybrid as WORSE than dense
    and the default was never updated to match, so the pipeline ran a
    configuration the evidence contradicted and the docs did not describe.
    See DEBUGGING.md #11.

    "rerank" is available and measurably retrieves better (40.0% vs 34.5%
    recall@25), but it is NOT the default: it scores worse end to end
    (81.8% vs 87.3% status agreement). See DECISIONS.md D12 - improving
    retrieval cannot improve agreement here, because labelled facets are
    force-included regardless of what retrieval finds.
    """
    index = index or build_index()
    if mode in ("dense", "rerank"):
        scores = index.matrix @ embed_query(conversation)
    elif mode == "lexical":
        scores = index.bm25.score(conversation)
    elif mode == "hybrid":
        scores = fuse(index.matrix @ embed_query(conversation),
                      index.bm25.score(conversation))
    else:
        raise ValueError(f"unknown retrieval mode {mode!r}")

    if mode == "rerank":
        # Retrieve a wide pool cheaply, then reorder it with a model that sees
        # conversation and facet together. The cosine score is kept for
        # reporting because the cross-encoder's scale means nothing here - only
        # its ordering is trusted. See rerank.py.
        pool = min(max(DEFAULT_POOL, top_k), len(scores))
        candidates = np.argpartition(-scores, pool - 1)[:pool]
        candidates = candidates[np.argsort(-scores[candidates])]
        top = rerank(conversation, candidates, index.rows)[:top_k]
        return [_to_candidate(index.rows[i], scores[i]) for i in top]

    k = min(top_k, len(scores))
    # argpartition is O(n); full sort only over the K we keep.
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [_to_candidate(index.rows[i], scores[i]) for i in top]


def route(conversation: str, index: FacetIndex | None = None, top_k: int = 25,
          allow_sensitive: bool = False, mode: str = "dense") -> RoutedFacets:
    """Gate 1, then Gate 2, then Gate 2b. Only `scorable` may reach the LLM.

    Gate 2b (policy) runs AFTER observability because the two refusals mean
    different things and should be reported separately: `not_observable` says
    the conversation cannot evidence the facet, `policy_blocked` says it might
    well be inferable and we decline to infer it anyway.
    """
    candidates = retrieve(conversation, index=index, top_k=top_k, mode=mode)
    gated_out = [c for c in candidates if not c.conversation_observable]
    observable = [c for c in candidates if c.conversation_observable]

    if allow_sensitive:
        return RoutedFacets(observable, gated_out, [], top_k)

    scorable = [c for c in observable if c.special_category is None]
    blocked = [c for c in observable if c.special_category is not None]
    return RoutedFacets(scorable, gated_out, blocked, top_k)


def near_duplicates(index: FacetIndex | None = None,
                    threshold: float = 0.85) -> list[tuple[str, str, float]]:
    """Find semantically near-duplicate facets, reusing the retrieval matrix.

    The catalogue has no exact or normalisation-level duplicates, but it does
    contain several constructs expressed more than once ('Depression Symptoms'
    vs 'Depression (DEP)'). Detecting them costs one matrix multiply against an
    index we already built.
    """
    index = index or build_index()
    similarity = index.matrix @ index.matrix.T
    # Keep the strict upper triangle so each unordered pair appears once.
    rows, cols = np.triu_indices(len(index.rows), k=1)
    hits = similarity[rows, cols] >= threshold
    pairs = [
        (
            index.rows[i]["facet_raw"],
            index.rows[j]["facet_raw"],
            round(float(similarity[i, j]), 4),
        )
        for i, j in zip(rows[hits], cols[hits])
    ]
    return sorted(pairs, key=lambda p: -p[2])
