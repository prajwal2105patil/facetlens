"""Defensive parsing of model output, plus the evidence verifier.

Two independent safety layers live here.

PARSING - a malformed batch must never take down a run. Recovery is layered:
  1. parse the whole response as JSON
  2. strip ```json fences and retry
  3. scan for the first balanced {...} object and retry
  4. validate against the schema; per-facet violations are isolated
Anything still unparseable becomes an explicit ERROR verdict for each facet in
that batch. Errors are visible in the output and counted in the report - never
swallowed.

EVIDENCE VERIFICATION - the model is required to quote verbatim. If the quote
is not actually present in the conversation, the "evidence" was fabricated, and
a fabricated citation cannot support a score. Such verdicts are downgraded to
insufficient_evidence. This is a programmatic hallucination check that does not
rely on the model being honest about its own reliability.
"""

from __future__ import annotations

import json
import re
import unicodedata

from pydantic import ValidationError

from ..retrieval.retrieve import Candidate
from .schema import BatchResponse, FacetVerdict, ModelVerdict, Status

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

#: A quote must be at least this long to be checkable. Very short quotes match
#: by accident, so we do not treat them as verification failures.
MIN_VERIFIABLE_QUOTE = 12


def _first_json_object(text: str) -> str | None:
    """Extract the first balanced {...} block, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_json(text: str) -> dict | None:
    """Layered JSON recovery. Returns None if nothing usable is found."""
    attempts = [text, _fence_body(text), _first_json_object(text)]
    for attempt in attempts:
        if not attempt:
            continue
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _fence_body(text: str) -> str | None:
    match = _FENCE.search(text)
    return match.group(1) if match else None


def parse_batch(text: str) -> tuple[list[ModelVerdict], list[str]]:
    """Parse a batch response into valid verdicts plus human-readable problems."""
    payload = extract_json(text)
    if payload is None:
        return [], ["response was not parseable as JSON"]

    try:
        return BatchResponse.model_validate(payload).verdicts, []
    except ValidationError:
        pass

    # The envelope failed, but individual verdicts may still be salvageable.
    # Isolating them means one bad row does not discard the whole batch.
    raw_items = payload.get("verdicts")
    if not isinstance(raw_items, list):
        return [], ["response JSON had no 'verdicts' list"]

    verdicts: list[ModelVerdict] = []
    problems: list[str] = []
    for index, item in enumerate(raw_items):
        try:
            verdicts.append(ModelVerdict.model_validate(item))
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            problems.append(
                f"verdict[{index}] invalid: "
                f"{'.'.join(str(p) for p in first.get('loc', ()))} "
                f"{first.get('msg', 'schema violation')}"
            )
    return verdicts, problems


def _normalise_for_match(text: str) -> str:
    """Fold case, unicode form and whitespace so quoting is robust to cosmetics."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = folded.translate(dict.fromkeys(map(ord, "‐‑‒–—―−"), "-"))
    folded = folded.translate({ord("’"): "'", ord("‘"): "'",
                               ord("“"): '"', ord("”"): '"'})
    return re.sub(r"\s+", " ", folded).strip()


def verify_evidence(quote: str, conversation: str) -> bool | None:
    """Is the quote genuinely present in the conversation?

    Returns True/False, or None when the quote is too short to check
    meaningfully (we do not want to punish a legitimate one-word quote).
    """
    cleaned = quote.strip().strip('"').strip()
    if len(cleaned) < MIN_VERIFIABLE_QUOTE:
        return None
    return _normalise_for_match(cleaned) in _normalise_for_match(conversation)


def to_facet_verdict(verdict: ModelVerdict, candidate: Candidate,
                     conversation: str) -> FacetVerdict:
    """Apply the evidence verifier and build the final verdict."""
    verified = (
        verify_evidence(verdict.evidence_quote, conversation)
        if verdict.status == "scored"
        else None
    )

    if verdict.status == "scored" and verified is False:
        return FacetVerdict(
            facet_id=candidate.facet_id,
            facet=candidate.facet,
            facet_type=candidate.facet_type,
            status=Status.INSUFFICIENT_EVIDENCE,
            score=None,
            confidence=verdict.confidence,
            reason=(
                "Downgraded by the evidence verifier: the cited quote "
                f"{verdict.evidence_quote[:60]!r} does not appear in the "
                f"conversation. Original model reason: {verdict.reason}"
            ),
            evidence_quote=verdict.evidence_quote,
            origin="evidence_verifier",
            retrieval_score=candidate.retrieval_score,
            evidence_verified=False,
        )

    return FacetVerdict(
        facet_id=candidate.facet_id,
        facet=candidate.facet,
        facet_type=candidate.facet_type,
        status=Status(verdict.status),
        score=verdict.score,
        confidence=verdict.confidence,
        reason=verdict.reason,
        evidence_quote=verdict.evidence_quote,
        origin="llm",
        retrieval_score=candidate.retrieval_score,
        evidence_verified=verified,
    )


def error_verdict(candidate: Candidate, problem: str) -> FacetVerdict:
    """An explicit, visible failure for one facet. Never silently dropped."""
    return FacetVerdict(
        facet_id=candidate.facet_id,
        facet=candidate.facet,
        facet_type=candidate.facet_type,
        status=Status.ERROR,
        score=None,
        confidence=0.0,
        reason=f"Could not obtain a valid verdict: {problem}",
        origin="parser",
        retrieval_score=candidate.retrieval_score,
    )
