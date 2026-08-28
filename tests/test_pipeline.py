"""Tests that need no model and no network.

Everything here runs against MockBackend or pure functions, so a reviewer can
clone the repo and verify the safety-critical behaviour immediately.
"""

from __future__ import annotations

import json

import pytest

from src.evaluation.evaluate import CONVERSATIONS, REFERENCE, load_facet_list, load_jsonl
from src.preprocessing.enrich import build_records, read_raw
from src.preprocessing.normalize import normalize_facet
from src.preprocessing.taxonomy import classify
from src.scoring.backends import MockBackend
from src.scoring.parser import (extract_json, parse_batch, repair_verdict,
                                verify_evidence)
from src.scoring.schema import FacetVerdict, ModelVerdict, Status


# --------------------------------------------------------------------- normalise
def test_numeric_prefix_is_extracted_not_deleted():
    result = normalize_facet("800. Sufi practice: Sufi retreat attendance count")
    assert result.catalogue_id == "800"
    assert result.source_qualifier == "Sufi practice"
    assert result.facet_normalized == "Sufi retreat attendance count"
    assert result.facet_raw.startswith("800.")  # raw is preserved verbatim


def test_trailing_colon_flags_header():
    assert normalize_facet("Leadership Potential:").is_header_like


def test_singular_behaviour_is_not_a_header():
    """Regression: 'Adventure-Seeking Behavior' is a facet, not a section."""
    assert not normalize_facet("Adventure-Seeking Behavior").is_header_like
    assert normalize_facet("Work Styles").is_header_like


def test_encoding_artifacts_are_flagged():
    assert normalize_facet("926. Bahá'í spiritual metric: Ridván festival participation"
                           ).has_encoding_artifact


# ---------------------------------------------------------------------- taxonomy
@pytest.mark.parametrize("raw,observable", [
    ("FSH level", False),
    ("Basophil count", False),
    ("Sleep Apnea", False),
    ("Intelligence Quotient (IQ)", False),
    ("Nationality", False),
    ("Passport-stamps count", False),
    ("Leadership Potential:", False),
    ("Delegation skills", True),
    ("Talkativeness", True),
    ("Irritability", True),
])
def test_observability_gate(raw, observable):
    assert classify(normalize_facet(raw)).conversation_observable is observable


def test_non_observable_always_has_a_reason():
    for raw in ("FSH level", "Nationality", "Work Styles"):
        result = classify(normalize_facet(raw))
        assert result.abstention_reason, f"{raw} must explain why it is gated"


def test_plural_keywords_match():
    """Regression: \\brelationship\\b failed on '...in relationships'."""
    result = classify(normalize_facet("Assertiveness and control in relationships"))
    assert result.facet_type == "interpersonal"


# ------------------------------------------------------------------------ schema
def test_scored_requires_a_score():
    with pytest.raises(ValueError):
        ModelVerdict(facet_id="F1", status="scored", score=None,
                     confidence=0.8, evidence_quote="x", reason="y")


def test_abstention_must_not_carry_a_score():
    with pytest.raises(ValueError):
        ModelVerdict(facet_id="F1", status="insufficient_evidence", score=3,
                     confidence=0.8, evidence_quote="", reason="y")


def test_score_field_is_required_by_the_json_schema():
    """The schema is used as a decoding grammar; optional fields get skipped."""
    from src.scoring.schema import BatchResponse
    schema = BatchResponse.model_json_schema()
    assert "score" in schema["$defs"]["ModelVerdict"]["required"]


def test_confidence_is_bounded():
    with pytest.raises(ValueError):
        FacetVerdict(facet_id="F1", facet="x", facet_type="t",
                     status=Status.INSUFFICIENT_EVIDENCE, confidence=1.4,
                     reason="y", origin="llm")


# ------------------------------------------------------------------------ parser
def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert extract_json("not json at all") is None


def test_extract_json_ignores_braces_inside_strings():
    assert extract_json('{"a": "} not the end"}') == {"a": "} not the end"}


def test_malformed_response_yields_problems_not_exceptions():
    verdicts, problems, _ = parse_batch('{"verdicts": [{"facet_id": ')
    assert verdicts == []
    assert problems


def test_repair_only_moves_toward_abstention():
    repaired = repair_verdict({"facet_id": "F1", "status": "scored", "score": None})
    assert repaired["status"] == "insufficient_evidence"
    # A genuine score is never touched.
    assert repair_verdict({"facet_id": "F1", "status": "scored", "score": 4}) is None


def test_one_bad_verdict_does_not_discard_the_batch():
    payload = json.dumps({"verdicts": [
        {"facet_id": "F1", "status": "scored", "score": 4, "confidence": 0.9,
         "evidence_quote": "a real quote here", "reason": "ok"},
        {"facet_id": "F2", "status": "scored", "score": 99, "confidence": 0.9,
         "evidence_quote": "q", "reason": "bad score"},
    ]})
    verdicts, problems, _ = parse_batch(payload)
    assert [v.facet_id for v in verdicts] == ["F1"]
    assert len(problems) == 1


# -------------------------------------------------------------- evidence verifier
def test_verify_evidence_detects_fabrication():
    conversation = "I led a team of five engineers and assigned tasks."
    assert verify_evidence("I led a team of five engineers", conversation) is True
    assert verify_evidence("I have a doctorate in astrophysics", conversation) is False


def test_verify_evidence_tolerates_cosmetic_differences():
    assert verify_evidence("I LED a  team", "I led a team of five") is True


def test_verify_evidence_skips_unverifiably_short_quotes():
    assert verify_evidence("ok", "everything is ok") is None


# ------------------------------------------------------------ end-to-end (mocked)
@pytest.mark.parametrize("mode,expected_status", [
    ("valid", "scored"),
    ("malformed", "error"),
    ("bad_schema", "error"),
    ("fabricated", "insufficient_evidence"),
])
def test_pipeline_survives_every_backend_failure_mode(mode, expected_status):
    from src.retrieval.embed import build_index
    from src.scoring.scorer import score_conversation

    result = score_conversation(
        "I led a team of five engineers and assigned tasks based on strengths.",
        top_k=6, batch_size=6, backend=MockBackend(mode=mode),
        index=build_index(),
    )
    assert result.verdicts, "pipeline must always return verdicts"
    llm_side = [v for v in result.verdicts if v.origin != "observability_gate"]
    assert any(v.status.value == expected_status for v in llm_side)


def test_fabricated_evidence_is_downgraded_never_scored():
    from src.retrieval.embed import build_index
    from src.scoring.scorer import score_conversation

    result = score_conversation(
        "I led a team of five engineers.", top_k=6, batch_size=6,
        backend=MockBackend(mode="fabricated"), index=build_index(),
    )
    assert not result.scored, "a fabricated quote must never survive as a score"


# -------------------------------------------------------------- benchmark data
def test_every_benchmark_facet_exists_verbatim_in_the_catalogue():
    catalogue = {row for row in read_raw()}
    missing = [f for f in load_facet_list() if f not in catalogue]
    assert not missing, f"invented facets: {missing}"


def test_benchmark_covers_every_required_category():
    required = {"clear", "ambiguous", "contradictory", "quoted", "sarcastic",
                "code-switched", "low-evidence", "hallucination-trap"}
    conversations = load_jsonl(CONVERSATIONS)
    assert len(conversations) >= 10
    assert required <= {c["category"] for c in conversations}


def test_benchmark_has_both_observable_and_non_observable_facets():
    records = {r["facet_raw"]: r for r in build_records(read_raw())}
    flags = [records[f]["conversation_observable"] for f in load_facet_list()]
    assert any(flags) and not all(flags)


def test_reference_labels_are_well_formed():
    catalogue = set(read_raw())
    conversation_ids = {c["conversation_id"] for c in load_jsonl(CONVERSATIONS)}
    for label in load_jsonl(REFERENCE):
        assert label["conversation_id"] in conversation_ids
        assert label["facet_raw"] in catalogue
        assert label["rationale"].strip(), "every label needs a rationale"
        if label["expected_status"] == "scored":
            assert 1 <= label["expected_score"] <= 5
        else:
            assert label["expected_score"] is None


def test_preprocessing_is_deterministic():
    assert build_records(read_raw()) == build_records(read_raw())


# ------------------------------------------------------- special-category policy
@pytest.mark.parametrize("raw,category", [
    ("Kink-interest diversity", "sex_life_or_orientation"),
    ("Holiness", "religious_or_philosophical_belief"),
    ("Ethnocentrism", "racial_or_ethnic_origin"),
    ("Liberalism", "political_opinion"),
    ("FSH level", "genetic_or_biometric"),
])
def test_special_categories_are_detected(raw, category):
    from src.preprocessing.sensitivity import special_category
    assert special_category(raw) == category


@pytest.mark.parametrize("raw", [
    "Value orientation: Universalism",   # 'orientation' is not sex life
    "Patient care orientation",
    "Neuroticism",                       # a Big Five trait, not a health record
    "Delegation skills",
    "Talkativeness",
])
def test_ordinary_facets_are_not_special_category(raw):
    from src.preprocessing.sensitivity import special_category
    assert special_category(raw) is None


def test_special_category_forces_high_sensitivity():
    """A trait-looking facet must not stay 'low' just because its type is."""
    result = classify(normalize_facet("Kink-interest diversity"))
    assert result.special_category == "sex_life_or_orientation"
    assert result.sensitivity == "high"


def test_policy_gate_blocks_special_category_before_the_llm():
    from src.retrieval.embed import build_index
    from src.scoring.scorer import score_conversation

    text = "I meditate every morning and I lean pretty liberal politically."
    result = score_conversation(text, top_k=10, batch_size=5,
                                backend=MockBackend(mode="valid"),
                                index=build_index())
    blocked = [v for v in result.verdicts if v.status == Status.POLICY_BLOCKED]
    assert blocked, "special-category facets must be refused by default"
    assert all(v.origin == "policy_gate" and v.score is None for v in blocked)
    # A refused facet must never also appear as a score.
    assert not ({v.facet_id for v in blocked} & {v.facet_id for v in result.scored})


def test_allow_sensitive_is_an_explicit_opt_in():
    from src.retrieval.embed import build_index
    from src.retrieval.retrieve import route

    text = "I meditate every morning and I lean pretty liberal politically."
    index = build_index()
    assert route(text, index, top_k=10).policy_blocked
    assert not route(text, index, top_k=10, allow_sensitive=True).policy_blocked


def test_verifier_accepts_a_quote_truncated_with_punctuation():
    """Regression: a verbatim quote that stops early and closes with a full
    stop was scored as fabricated, costing two legitimate scores on the
    code-switched benchmark case. See DEBUGGING.md #8."""
    conversation = ("Team meeting mein maine sabka opinion suna, phir hum sab "
                    "ne milkar decide kiya ki naya architecture use karenge.")
    truncated = ("Team meeting mein maine sabka opinion suna, phir hum sab ne "
                 "milkar decide kiya.")
    assert verify_evidence(truncated, conversation) is True
    assert verify_evidence("...maine sabka opinion suna...", conversation) is True
    # Loosening the edges must NOT loosen the words.
    assert verify_evidence("I have a doctorate in astrophysics.", conversation) is False


def test_evidence_level_policy_gate_closes_the_red_team_leak():
    """Regression for adversarial a05: the policy gate filtered facets but not
    the evidence they were scored from, so 'Patience: Resistance to anger'
    scored 5 quoting 'my faith is the main thing keeping me steady'."""
    from src.retrieval.retrieve import Candidate
    from src.scoring.parser import to_facet_verdict
    from src.scoring.schema import ModelVerdict

    conversation = ("I pray twice a day and honestly my faith is the main "
                    "thing keeping me steady right now.")
    candidate = Candidate(
        facet_id="F0001", facet="Patience: Resistance to anger",
        facet_type="emotional_state", conversation_observable=True,
        abstention_reason=None, sensitivity="low", special_category=None,
        scoring_definition="d", score_anchors="a", retrieval_score=0.3,
    )
    verdict = ModelVerdict(
        facet_id="F0001", status="scored", score=5, confidence=1.0,
        evidence_quote="my faith is the main thing keeping me steady",
        reason="speaker reports steadiness",
    )
    result = to_facet_verdict(verdict, candidate, conversation)
    assert result.status == Status.POLICY_BLOCKED
    assert result.score is None
    assert result.origin == "policy_gate"


def test_evidence_gate_does_not_block_ordinary_evidence():
    from src.retrieval.retrieve import Candidate
    from src.scoring.parser import to_facet_verdict
    from src.scoring.schema import ModelVerdict

    conversation = "I led a team of five engineers and assigned tasks by strength."
    candidate = Candidate(
        facet_id="F0002", facet="Delegation skills", facet_type="interpersonal",
        conversation_observable=True, abstention_reason=None, sensitivity="low",
        special_category=None, scoring_definition="d", score_anchors="a",
        retrieval_score=0.3,
    )
    verdict = ModelVerdict(
        facet_id="F0002", status="scored", score=4, confidence=0.9,
        evidence_quote="assigned tasks by strength", reason="delegation described",
    )
    assert to_facet_verdict(verdict, candidate, conversation).status == Status.SCORED


# ------------------------------------------------------------------- reranking
def test_rerank_mode_returns_top_k_and_preserves_contract():
    from src.retrieval.embed import build_index
    from src.retrieval.retrieve import retrieve

    index = build_index()
    text = "I led a team of five engineers and assigned tasks based on strengths."
    got = retrieve(text, index, top_k=8, mode="rerank")
    assert len(got) == 8
    # retrieval_score must stay the cosine value, never the cross-encoder's:
    # the two are on incomparable scales and the report shows this number.
    assert all(-1.0 <= c.retrieval_score <= 1.0 for c in got)
    assert len({c.facet_id for c in got}) == 8


def test_rerank_changes_ordering_but_not_the_candidate_universe():
    """Reranking must reorder a wide pool, not invent or drop facets."""
    from src.retrieval.embed import build_index
    from src.retrieval.retrieve import retrieve

    index = build_index()
    text = "I led a team of five engineers and assigned tasks based on strengths."
    dense = [c.facet_id for c in retrieve(text, index, top_k=100, mode="dense")]
    reranked = [c.facet_id for c in retrieve(text, index, top_k=100, mode="rerank")]
    assert set(dense) == set(reranked), "rerank must permute the same pool"
    assert dense != reranked, "rerank should actually change the order"


def test_unknown_retrieval_mode_is_rejected():
    from src.retrieval.embed import build_index
    from src.retrieval.retrieve import retrieve

    with pytest.raises(ValueError):
        retrieve("x", build_index(), top_k=3, mode="not_a_mode")
