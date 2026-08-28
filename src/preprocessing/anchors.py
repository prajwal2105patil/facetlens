"""Scoring definitions and 1-5 anchors, generated from facet_type templates.

WHAT IS BEING SCORED
--------------------
The score is the *strength of evidenced expression of the facet in this
conversation* - NOT the person's true underlying trait level.

This framing is what makes abstention coherent. "How strongly is leadership
evidenced in these three sentences?" is answerable from text. "What is this
person's leadership ability?" is not, and inviting the model to answer it is
exactly how confident hallucination happens.

Anchors are produced by code from per-type templates rather than hand-written
per row, so all 399 rows stay mutually consistent and the catalogue is
regenerable from the raw CSV.
"""

from __future__ import annotations

# Generic ordinal ladder, shared by every observable type.
SCORE_LEVELS: dict[int, str] = {
    1: "no or very weak evidence",
    2: "weak / passing evidence",
    3: "moderate, explicit evidence",
    4: "strong, elaborated evidence",
    5: "very strong, repeated or richly detailed evidence",
}

# Per-type phrasing of *what* counts as evidence. Keeps the ladder concrete.
_EVIDENCE_TEMPLATE: dict[str, str] = {
    "personality_trait":
        "how strongly the speaker's own words display the disposition '{name}'",
    "behavioral_tendency":
        "how strongly the speaker describes acting in ways characteristic of '{name}'",
    "communication_style":
        "how strongly the speaker's own manner of speaking demonstrates '{name}'",
    "interpersonal":
        "how strongly the speaker describes relating to others in terms of '{name}'",
    "emotional_state":
        "how strongly the speaker expresses or reports the emotional state '{name}'",
    "motivation_value":
        "how strongly the speaker expresses '{name}' as a driver or held value",
    "preference_lifestyle":
        "how strongly the speaker states a preference or habit reflecting '{name}'",
    "cognitive_style":
        "how strongly the speaker's reasoning in this conversation exhibits '{name}'",
    "spiritual_religious_disposition":
        "how strongly the speaker expresses the disposition '{name}', "
        "without inferring any specific tradition or affiliation",
}

_DEFAULT_TEMPLATE = "how strongly '{name}' is evidenced by the speaker's own words"


def scoring_definition(facet_name: str, facet_type: str, observable: bool,
                       abstention_reason: str | None) -> str:
    """One-line definition of what a score on this facet would mean."""
    if not observable:
        return (
            f"NOT SCORABLE FROM CONVERSATION. '{facet_name}' is classified as "
            f"{facet_type}; establishing it requires evidence outside the "
            f"transcript ({abstention_reason})."
        )
    template = _EVIDENCE_TEMPLATE.get(facet_type, _DEFAULT_TEMPLATE)
    return "Rate " + template.format(name=facet_name) + ", on a 1-5 scale."


def anchors(observable: bool) -> str:
    """Compact, pipe-delimited anchor ladder stored in the enriched CSV."""
    if not observable:
        return ""
    return " | ".join(f"{level}={text}" for level, text in SCORE_LEVELS.items())


def retrieval_text(facet_name: str, facet_type: str, definition: str) -> str:
    """The string that actually gets embedded for retrieval.

    MEASURED DESIGN DECISION (see DECISIONS.md D2 / DEBUGGING.md #1): embedding
    the bare facet name ranks 'FSH level' at cosine 0.110 against a leadership
    conversation - level with 'Assertiveness...' (0.110) and below 'Nationality'
    (0.134). Appending the type and definition drops 'FSH level' to -0.005 and
    lets the genuinely relevant facets take the top ranks.
    """
    readable_type = facet_type.replace("_", " ")
    return f"{facet_name}. {readable_type}. {definition}"
