"""Cross-encoder reranking: the fix four earlier attempts pointed to.

WHY THIS EXISTS
---------------
Retrieval was the weakest measured component of this system. Four interventions
were built and measured against it, and none worked (DECISIONS.md D10/D11):

    BM25 lexical            11% recall@25 - far worse
    dense + BM25 fusion     53% - worse than dense alone
    BGE-small encoder       53% - a "better" encoder scored worse
    document expansion      63% - tied the incumbent

Those four failures were not wasted: they ruled out the encoder, the similarity
function and the indexed text as causes. What survived was a structural
observation - recall is **89% at K=100 but 63% at K=25**. The right facets are
already being retrieved. They are simply ranked badly.

That is a reranking problem, not a retrieval problem, and it needs a model that
can attend to the conversation and the facet *jointly*. A bi-encoder cannot: it
compresses each side to a vector independently, so a short abstract label and a
long concrete narrative never meet.

THE DESIGN
----------
    dense retrieval  ->  wide pool (100)  ->  cross-encoder  ->  top-K
        cheap, recall-oriented              expensive per pair,
                                            but only 100 pairs

Cost measured on this machine: **8.3s for all 13 benchmark conversations**,
about 0.64s each, against ~50s for a single LLM scoring batch. The reranker is
free relative to the thing it feeds.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`, 22.7M parameters, Apache-2.0.

HONEST LIMITS
-------------
It is trained on web search query->passage relevance, which is not this task.
A four-example hand check looked bad (it ranked `Nationality` above
`Delegation skills`); the full measurement over 19 labelled facets looked good.
The hand check was wrong, not the measurement - which is the same lesson
recorded three times in PROMPT_LOG.

It also *hurts* at K=10 (47% -> 42%). Reranking a wide pool needs enough room
to place what it promotes; too small a K discards it again.
"""

from __future__ import annotations

import functools

import numpy as np

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: How many dense candidates to rerank. 100 is where dense recall saturates
#: (89%), so it is the largest pool worth paying for.
DEFAULT_POOL = 100

#: Facet text is truncated for the cross-encoder; definitions are front-loaded
#: with the name and type, so 250 chars keeps the discriminative part.
_MAX_FACET_CHARS = 250


@functools.lru_cache(maxsize=1)
def _load(model_name: str = RERANK_MODEL):
    """Load once per process; the model costs ~36s to load, ~0.6s to use."""
    from sentence_transformers import CrossEncoder

    try:
        from huggingface_hub import snapshot_download

        return CrossEncoder(snapshot_download(model_name, local_files_only=True),
                            max_length=256)
    except Exception:
        return CrossEncoder(model_name, max_length=256)


def rerank(conversation: str, candidate_indices: np.ndarray, rows: list[dict],
           model_name: str = RERANK_MODEL) -> np.ndarray:
    """Reorder `candidate_indices` by cross-encoder relevance, best first.

    Returns indices into `rows`, not scores: the reranker's absolute values are
    not comparable to cosine similarity and must never be presented as if they
    were. Ordering is the only thing it is trusted for.
    """
    if len(candidate_indices) == 0:
        return candidate_indices
    model = _load(model_name)
    pairs = [(conversation, rows[i]["retrieval_text"][:_MAX_FACET_CHARS])
             for i in candidate_indices]
    scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
    return candidate_indices[np.argsort(-scores)]
