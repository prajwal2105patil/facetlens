"""Rule-based facet taxonomy and observability gate.

This module answers two separate questions:

  1. facet_type              - what KIND of construct is this?
  2. conversation_observable - can a short conversation legitimately evidence it?

These are deliberately not the same question. A category can contain both
observable and non-observable members depending on the exact wording, which is
why observability is derived from the *type assigned by a named rule* rather
than guessed per row.

HONESTY NOTE: this is a heuristic keyword/pattern classifier, not a trained
model. Every row records which rule fired (`classification_rule`) so any
classification can be audited and challenged. `enrich.py` reports the real
disagreement rate from a seeded manual spot-check; it is not claimed to be 100%
correct. See artifacts/audit_report.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .normalize import NormalizedFacet
from .sensitivity import special_category

# --------------------------------------------------------------------------
# Facet types. Observability is a property of the type (see _OBSERVABLE below).
# --------------------------------------------------------------------------
FACET_TYPES = (
    # --- conversation-observable ---
    "personality_trait",
    "behavioral_tendency",
    "communication_style",
    "interpersonal",
    "emotional_state",
    "motivation_value",
    "preference_lifestyle",
    "cognitive_style",
    "spiritual_religious_disposition",
    # --- NOT conversation-observable ---
    "cognitive_test_ability",
    "psychometric_instrument_score",
    "spiritual_religious_practice_metric",
    "clinical_mental_health",
    "biometric_physiological",
    "demographic_biographical",
    "quantified_activity_metric",
    "instrument_or_scale_header",
    "other",
)

_OBSERVABLE: frozenset[str] = frozenset(
    {
        "personality_trait",
        "behavioral_tendency",
        "communication_style",
        "interpersonal",
        "emotional_state",
        "motivation_value",
        "preference_lifestyle",
        "cognitive_style",
        "spiritual_religious_disposition",
    }
)

# Why each non-observable type cannot be scored from conversation alone.
_ABSTENTION_REASON: dict[str, str] = {
    "cognitive_test_ability": "requires_standardised_assessment",
    "psychometric_instrument_score": "psychometric_instrument_scale",
    "spiritual_religious_practice_metric": "requires_quantified_self_report",
    "clinical_mental_health": "requires_clinical_assessment",
    "biometric_physiological": "requires_biometric_measurement",
    "demographic_biographical": "demographic_fact_not_inferable",
    "quantified_activity_metric": "requires_external_records",
    "instrument_or_scale_header": "malformed_or_header_row",
    "other": "unclassified_construct",
}

_HIGH_SENSITIVITY = frozenset(
    {"clinical_mental_health", "biometric_physiological", "demographic_biographical"}
)
_MEDIUM_SENSITIVITY = frozenset(
    {
        "spiritual_religious_disposition",
        "spiritual_religious_practice_metric",
        "emotional_state",
        "preference_lifestyle",
    }
)


@dataclass(frozen=True)
class Rule:
    """A single named classification rule. Order is significant."""

    name: str
    facet_type: str
    test: Callable[[str], bool]


def _kw(*words: str) -> Callable[[str], bool]:
    """Match if any whole-word keyword appears, tolerating a plural suffix.

    The plural suffix is NOT cosmetic. Without it `\\brelationship\\b` fails to
    match "Assertiveness and control in relationships", which silently dropped
    that row into the catch-all fallback. See DEBUGGING.md #3.
    """
    alternatives = "|".join(re.escape(word) for word in words)
    pattern = re.compile(rf"\b(?:{alternatives})(?:e?s)?\b", re.IGNORECASE)
    return lambda text: bool(pattern.search(text))


def _rx(pattern: str) -> Callable[[str], bool]:
    compiled = re.compile(pattern, re.IGNORECASE)
    return lambda text: bool(compiled.search(text))


# --------------------------------------------------------------------------
# The ruleset. ORDER MATTERS: most specific and most safety-critical first.
# A row is classified by the FIRST rule that fires; ties are impossible.
# --------------------------------------------------------------------------
RULES: tuple[Rule, ...] = (
    # 1. Lab values, biomarkers, genetics, physiology. Highest precedence:
    #    misclassifying one of these as observable is the worst failure mode.
    Rule("biomarker_named_analyte", "biometric_physiological",
         _kw("fsh", "parathyroid", "basophil", "serotonin", "chromatin",
             "polygenic", "hormone", "immune", "macronutrient", "metabolic",
             "cortisol", "glucose", "eosinophil")),
    Rule("biomarker_genetic", "biometric_physiological",
         _kw("gene", "genetic", "genome", "allele", "snp")),
    Rule("biomarker_body_measure", "biometric_physiological",
         _rx(r"\b(blood|heart[- ]rate|bmi|body[- ]fat|oxygen|apnea|apnoea)\b")),

    # 2. Clinical / psychiatric constructs requiring assessment or diagnosis.
    Rule("clinical_diagnosis", "clinical_mental_health",
         _kw("diagnosis", "diagnosed", "disorder", "syndrome", "pathology")),
    Rule("clinical_named_scale", "clinical_mental_health",
         _rx(r"\b(depression|hypomania|hysteria|psychoticism|psychopath|"
             r"burnout|compassion fatigue|chronic pain|drug[- ]use|"
             r"physical[- ]violence|trauma|suicid)\w*")),
    Rule("clinical_symptom_language", "clinical_mental_health",
         _rx(r"\bsymptoms?\b|\bpresence of\b.*\bpain\b")),

    # 3. Standardised-test abilities. A conversation cannot establish these;
    #    they require an instrument. Distinguished from cognitive *style* below.
    Rule("cognitive_test_score", "cognitive_test_ability",
         _rx(r"\b(iq|intelligence quotient|quotient)\b|\bindex\b|"
             r"\b(working memory|divided attention|auditory memory|"
             r"sequential memory|memory for sounds|information retention|"
             r"spatial perception|mental arithmetic|rapid cognitive|"
             r"precision of movements|spelling accuracy|estimating calculations|"
             r"alphabetical filing|numeric filing|comparing alphanumeric|"
             r"logical sequence|analogies)\b")),
    Rule("cognitive_test_academic", "cognitive_test_ability",
         _rx(r"\b(mathematical (concepts|formulas)|numerical reasoning|"
             r"statistical reasoning|anatomy knowledge|material properties|"
             r"network basics|mechanical concepts)\b")),

    # 3b. Psychometric instrument outputs. Found by the seeded manual spot-check
    #     (see artifacts/audit_report.md S8): "Sense-of-coherence score",
    #     "Honesty-humility trait score", "Ethical leadership rating" and
    #     "Eye-Contact avoidance score" were all being routed to the LLM as
    #     observable. A score/rating/scale IS an instrument output - it is
    #     produced by administering a questionnaire or by third-party raters,
    #     never by reading a conversation. This was 4 of 30 sampled rows, and
    #     every one erred in the dangerous direction.
    Rule("psychometric_instrument_output", "psychometric_instrument_score",
         _rx(r"\b(score|rating|ratings|scale|inventory|questionnaire|"
             r"assessment|quotient)\b")),

    # 4. Countable / measurable activity. The giveaway is a UNIT or a COUNT
    #    noun: these need diaries, logs or platform records, never a snippet.
    Rule("quantified_unit_bearing", "quantified_activity_metric",
         _rx(r"(/\s*(day|week|month|year|h)\b)|\bper (day|week|month|year)\b|"
             r"\(\s*(mg|h|hours|km|%)\s*/?\s*\w*\s*\)|\b(km|mg)\b|%\s*$")),
    Rule("quantified_count_noun", "quantified_activity_metric",
         _rx(r"\bcounts?\b|\bhours?\b|\byears?\b|\bmonths?\b|\bdays?\b|"
             r"\bsessions?\b|\bvisits?\b|\bcycles?\b|\bverses?\b|"
             r"\brepetitions?\b|\bcontributions?\b|\bendorsements?\b|"
             r"\bsubscribers?\b|\bstamps?\b|\bfrequency\b|\bduration\b|"
             r"\bconsistency\b|\bmemorized\b|\bobserved\b|\btemperature\b|"
             r"\bkm\b|\bintake\b")),

    # 5. Religion / spirituality. Practice METRICS were caught by rule 4 above;
    #    what reaches here is dispositional and can surface in conversation.
    Rule("spiritual_practice_named", "spiritual_religious_practice_metric",
         _rx(r"\b(hexagram|sephira|rising sign|aura[- ]colou?r|khatam|"
             r"lulav|ridv|vrata|dhikr|reiki|channeling|archon)\w*")),
    Rule("spiritual_disposition", "spiritual_religious_disposition",
         _kw("spiritual", "spirituality", "holiness", "sacred", "divine",
             "faith", "prayer", "meditation", "mindfulness", "satya",
             "ego dissolution", "pilgrimage", "scripture")),

    # 6. Fixed facts about a person that a conversation cannot establish.
    Rule("demographic_fact", "demographic_biographical",
         _kw("nationality", "ethnicity", "citizenship", "age", "gender",
             "income", "salary", "marital", "childhood", "cultural identity",
             "multiculturalism", "patriotism", "passport", "consent")),

    # 7. Catalogue structure rather than a facet (also flagged in normalize.py).
    Rule("structural_header", "instrument_or_scale_header",
         lambda text: False),  # populated from NormalizedFacet.is_header_like

    # 8. Observable behaviour and disposition. Broadest rules go last.
    Rule("communication", "communication_style",
         _kw("communication", "listening", "talkativeness", "brevity",
             "outspokenness", "storytelling", "language use", "frankness",
             "sentence structure", "eye-contact", "non-verbal", "verbal",
             "comprehension of spoken", "sensationalism", "coarseness")),
    Rule("interpersonal", "interpersonal",
         _kw("relationship", "collaboration", "cooperation", "teamwork",
             "social", "interpersonal", "empathy", "compassion", "affection",
             "trust", "affiliation", "chivalrousness", "civility", "cordiality",
             "hostility", "disrespect", "peer", "group", "community",
             "participation", "delegation", "leadership", "mentoring")),
    Rule("emotional", "emotional_state",
         _kw("emotion", "emotional", "mood", "happiness", "joyfulness",
             "merriness", "sadness", "moroseness", "irritability", "anxiety",
             "fearfulness", "contentment", "discontentment", "blissfulness",
             "desperation", "boredom", "enthusiasm", "vivacity", "affect",
             "peacefulness", "stress", "hopelessness")),
    Rule("motivation", "motivation_value",
         _kw("motivation", "motivational", "ambition", "achievement", "desire",
             "goal", "aspiration", "drive", "significance", "purpose",
             "ethical", "ethics", "morality", "moral", "justice", "dignity",
             "integrity", "honesty", "value", "universalism", "liberalism",
             "conservatism", "esteem needs")),
    Rule("preference_lifestyle", "preference_lifestyle",
         _kw("preference", "preferred", "habits", "lifestyle", "diet",
             "dietary", "eating", "cooking", "culinary", "snacking", "sleep",
             "travel", "tourism", "hobby", "leisure", "aesthetic",
             "aestheticism", "appreciation", "arts", "music", "dance",
             "learning style", "orderliness", "organized")),
    Rule("cognitive_style", "cognitive_style",
         _kw("reasoning", "critical", "synthesis", "analysis", "evaluating",
             "judging", "decision-making", "problem", "creativity", "creative",
             "originality", "innovation", "ideas", "epistemology",
             "troubleshooting", "perceiving", "insight", "curiosity",
             "intellect", "openness")),
    Rule("behavioral_tendency", "behavioral_tendency",
         _kw("behavior", "behaviour", "behavioral", "tendency", "initiative",
             "persistence", "perseverance", "risktaking", "risk-taking",
             "impulsivity", "impulsiveness", "hesitation", "procrastination",
             "compulsive", "adventure", "competition", "conformity",
             "rebelliousness", "safety", "compliance", "structure",
             "self-control", "selfcontrol", "controlling", "managing")),
)

# Rules whose keyword lists cannot express the condition are handled in code.
_STRUCTURAL_RULE = "structural_header"


@dataclass(frozen=True)
class Classification:
    facet_type: str
    conversation_observable: bool
    sensitivity: str
    abstention_reason: str | None
    classification_rule: str
    #: GDPR Art. 9(1) special category this facet concerns, if any.
    special_category: str | None


def classify(facet: NormalizedFacet) -> Classification:
    """Assign a facet type and derive observability from it.

    The first matching rule wins. Header-like rows short-circuit everything:
    a catalogue section header is not a facet at all, so classifying its
    *content* would be meaningless.
    """
    if facet.is_header_like:
        return _finalize("instrument_or_scale_header", _STRUCTURAL_RULE, None)

    # Match against the normalised name plus any stripped source qualifier, so
    # that "926. Bahá'í spiritual metric: Ridván festival participation" is
    # still recognisable as a religious-practice row after the prefix is removed.
    haystack = facet.facet_normalized
    if facet.source_qualifier:
        haystack = f"{facet.source_qualifier} {haystack}"

    category = special_category(haystack)

    for rule in RULES:
        if rule.name == _STRUCTURAL_RULE:
            continue
        if rule.test(haystack):
            return _finalize(rule.facet_type, rule.name, category)

    # Explicit fallback. Bare trait nouns ("Naivety", "Cunningness", "Dignity")
    # legitimately land here: they are single-word dispositions with no keyword
    # signal. Treating them as observable personality traits is the defensible
    # default for a *personality* catalogue, but it IS a default, and the audit
    # report counts how many rows depend on it.
    return _finalize("personality_trait", "fallback_bare_trait_noun", category)


def _finalize(facet_type: str, rule_name: str,
              category: str | None = None) -> Classification:
    observable = facet_type in _OBSERVABLE
    # A special category always forces high sensitivity, whatever the type says.
    # This is what catches 'Kink-interest diversity', which the type-derived
    # scale rated 'low' because it looks like an ordinary personality trait.
    if category is not None:
        sensitivity = "high"
    elif facet_type in _HIGH_SENSITIVITY:
        sensitivity = "high"
    elif facet_type in _MEDIUM_SENSITIVITY:
        sensitivity = "medium"
    else:
        sensitivity = "low"
    return Classification(
        facet_type=facet_type,
        conversation_observable=observable,
        sensitivity=sensitivity,
        abstention_reason=None if observable else _ABSTENTION_REASON[facet_type],
        classification_rule=rule_name,
        special_category=category,
    )
