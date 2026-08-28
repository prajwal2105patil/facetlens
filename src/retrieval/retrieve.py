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


@dataclass(frozen=True)
class Candidate:
    """A facet that survived retrieval, with its similarity score."""

    facet_id: str
    facet: str
    facet_type: str
    conversation_observable: bool
    abstention_reason: str | None
    sensitivity: str
    scoring_definition: str
    score_anchors: str
    retrieval_score: float


@dataclass(frozen=True)
class RoutedFacets:
    """Result of running both gates over one conversation."""

    scorable: list[Candidate]      # -> Gate 3 (the LLM)
    gated_out: list[Candidate]     # -> deterministic not_observable, no LLM call
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
        scoring_definition=row["scoring_definition"],
        score_anchors=row["score_anchors"],
        retrieval_score=round(float(score), 4),
    )


def retrieve(conversation: str, index: FacetIndex | None = None,
             top_k: int = 25) -> list[Candidate]:
    """Gate 1: top-K facets by cosine similarity. Relevance only."""
    index = index or build_index()
    query = embed_query(conversation)
    scores = index.matrix @ query
    k = min(top_k, len(scores))
    # argpartition is O(n); full sort only over the K we keep.
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [_to_candidate(index.rows[i], scores[i]) for i in top]


def route(conversation: str, index: FacetIndex | None = None,
          top_k: int = 25) -> RoutedFacets:
    """Gate 1 then Gate 2. Only `scorable` may reach the LLM."""
    candidates = retrieve(conversation, index=index, top_k=top_k)
    scorable = [c for c in candidates if c.conversation_observable]
    gated_out = [c for c in candidates if not c.conversation_observable]
    return RoutedFacets(scorable=scorable, gated_out=gated_out, top_k=top_k)


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
