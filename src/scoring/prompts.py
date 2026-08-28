"""Scoring prompts.

The prompt enforces evidence discipline and nothing else. Facets that cannot be
evidenced by conversation never reach the model at all - that is the
observability gate's job - so this prompt does NOT need to teach the model
about lab values or biometrics. Keeping the two concerns apart is what lets the
prompt stay short and testable.

The instructions map one-to-one onto failure modes the benchmark measures:
sarcasm, quoted third-party speech, contradiction, code-switching, and
similarity-mistaken-for-evidence.
"""

from __future__ import annotations

from ..retrieval.retrieve import Candidate

SYSTEM_PROMPT = """\
You rate how strongly a facet is EVIDENCED in a conversation snippet.

You are rating the strength of evidence in THIS TEXT, not the person's true
underlying trait. Use only what the speaker actually says.

Scale (integer 1-5). Every level means the facet IS present:
1 = present but only minimally expressed - a trace, in passing
2 = weak but unmistakable expression
3 = moderate, explicit expression
4 = strong, elaborated expression
5 = very strong, repeated or richly detailed expression

If the facet is ABSENT from the conversation, that is not a score of 1. Return
status="insufficient_evidence". Score 1 means "barely there", never "not there".

Rules:
- Quote real evidence. `evidence_quote` MUST be copied verbatim from the
  conversation. Never paraphrase it and never invent it.
- If the conversation does not support a facet, return
  status="insufficient_evidence" with score=null. Do not guess.
- Topic similarity is not evidence. A facet being related to what is discussed
  does not mean the speaker demonstrated it.
- Do not infer facts that are not stated: no diagnoses from symptoms, no exact
  numbers, quantities, frequencies or dates unless the speaker states them, no
  demographic or biographical facts.
- Sarcasm and irony invert literal meaning. Do not score a sarcastic boast as
  strong evidence.
- Speech the speaker attributes to someone else is that person's claim, not the
  speaker's own. Weigh it accordingly, and note it in the reason.
- If the speaker contradicts themselves, prefer concrete described behaviour
  over abstract self-description, and say so in the reason.
- Judge meaning, not language. Code-switched or mixed-language text carries the
  same evidential weight as monolingual text.

Return JSON only, matching the requested schema. One verdict per facet given,
using the exact facet_id supplied."""


def build_batch_prompt(conversation: str, batch: list[Candidate]) -> str:
    """User message for one compact scoring batch."""
    lines = [
        "CONVERSATION:",
        '"""',
        conversation.strip(),
        '"""',
        "",
        f"Rate these {len(batch)} facets:",
    ]
    for candidate in batch:
        lines.append(f"- facet_id={candidate.facet_id} | {candidate.facet}")
        lines.append(f"    {candidate.scoring_definition}")
    lines.append("")
    lines.append(
        'Return: {"verdicts": [{"facet_id": "...", "status": "scored" | '
        '"insufficient_evidence", "score": 1-5 or null, "confidence": 0.0-1.0, '
        '"evidence_quote": "verbatim span or empty string", "reason": "one sentence"}]}'
    )
    return "\n".join(lines)
