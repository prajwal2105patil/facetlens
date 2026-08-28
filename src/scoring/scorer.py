"""Gate 3: batched LLM scoring, wired to the gates in front of it.

Flow for one conversation:

    route()                  Gate 1 + Gate 2
      |-- gated_out  -----> deterministic not_observable verdicts (no LLM call)
      `-- scorable   -----> compact batches -> LLM -> parse -> verify evidence

Facets the model silently omits from a batch are NOT dropped: they become
explicit ERROR verdicts, so a lazy or truncated response is visible in the
report rather than looking like a clean run.
"""

from __future__ import annotations

from ..retrieval.embed import FacetIndex
from ..retrieval.retrieve import Candidate, route
from .backends import LLMBackend, get_backend
from .parser import error_verdict, parse_batch, to_facet_verdict
from .prompts import SYSTEM_PROMPT, build_batch_prompt
from .schema import BatchResponse, ConversationResult, FacetVerdict, Status

#: Confidence attached to deterministic, rule-derived abstentions.
#: This is NOT a model probability. It reflects that the observability gate is a
#: deterministic catalogue lookup: given the catalogue, the verdict is certain.
#: Whether the *catalogue* is right is a separate question the audit addresses.
GATE_CONFIDENCE = 1.0


def _gate_verdict(candidate: Candidate) -> FacetVerdict:
    return FacetVerdict(
        facet_id=candidate.facet_id,
        facet=candidate.facet,
        facet_type=candidate.facet_type,
        status=Status.NOT_OBSERVABLE,
        score=None,
        confidence=GATE_CONFIDENCE,
        reason=(
            f"Not scorable from conversation: classified as "
            f"{candidate.facet_type} ({candidate.abstention_reason}). "
            f"Establishing it needs evidence outside the transcript."
        ),
        origin="observability_gate",
        retrieval_score=candidate.retrieval_score,
    )


def _chunk(items: list[Candidate], size: int) -> list[list[Candidate]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def score_batch(conversation: str, batch: list[Candidate],
                backend: LLMBackend) -> list[FacetVerdict]:
    """Score one compact batch, isolating every failure to the facet it affects."""
    user_prompt = build_batch_prompt(conversation, batch)
    schema = BatchResponse.model_json_schema()

    try:
        raw = backend.complete(SYSTEM_PROMPT, user_prompt, schema=schema)
    except Exception as exc:
        # Network/daemon failure is reported per facet, not raised. The run
        # continues and the report shows exactly which facets were affected.
        return [error_verdict(c, f"{type(exc).__name__}: {exc}") for c in batch]

    verdicts, problems, repaired = parse_batch(raw)
    by_id = {v.facet_id: v for v in verdicts}

    results: list[FacetVerdict] = []
    for candidate in batch:
        verdict = by_id.get(candidate.facet_id)
        if verdict is None:
            detail = "; ".join(problems) if problems else "facet missing from response"
            results.append(error_verdict(candidate, detail))
        else:
            results.append(to_facet_verdict(
                verdict, candidate, conversation,
                schema_repaired=candidate.facet_id in repaired))
    return results


def score_conversation(conversation: str, *, conversation_id: str = "adhoc",
                       top_k: int = 25, batch_size: int = 5,
                       backend_name: str = "ollama",
                       model: str = "qwen2.5:7b-instruct",
                       backend: LLMBackend | None = None,
                       index: FacetIndex | None = None) -> ConversationResult:
    """Run all three gates over one conversation."""
    backend = backend or get_backend(backend_name, model)
    routed = route(conversation, index=index, top_k=top_k)

    verdicts = [_gate_verdict(c) for c in routed.gated_out]
    for batch in _chunk(routed.scorable, batch_size):
        verdicts.extend(score_batch(conversation, batch, backend))

    # Stable ordering: strongest retrieval first, so reports diff cleanly.
    verdicts.sort(key=lambda v: (-(v.retrieval_score or 0.0), v.facet_id))

    return ConversationResult(
        conversation_id=conversation_id,
        conversation=conversation,
        model=f"{backend.name}:{backend.model}",
        top_k=top_k,
        batch_size=batch_size,
        verdicts=verdicts,
    )
