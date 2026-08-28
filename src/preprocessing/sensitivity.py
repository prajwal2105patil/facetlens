"""Special-category detection, and the policy that follows from it.

WHY THIS EXISTS
---------------
The first version of this pipeline derived `sensitivity` from `facet_type`
alone. That was too crude, and the audit showed exactly how it failed:

    Kink-interest diversity        -> personality_trait, sensitivity=low, OBSERVABLE
    516. Religious coping - Negative -> personality_trait, sensitivity=low, OBSERVABLE

Both would have been retrieved, passed the observability gate, and scored. A
system that infers someone's sexual interests or religious coping style from a
conversation is not a scoring bug - it is a product that should not ship.

THE FRAMEWORK
-------------
Rather than invent a sensitivity scale, this classifies facets against the
special categories of personal data in **GDPR Article 9(1)**: racial or ethnic
origin, political opinions, religious or philosophical beliefs, trade union
membership, genetic data, biometric data, health data, and data concerning sex
life or sexual orientation.

Article 9 prohibits processing these by default and requires a specific lawful
basis - usually explicit consent - to proceed. That maps directly onto the
engineering decision: **refuse by default, allow only on explicit opt-in.**

This is a defensible line precisely because it is not mine. It is the standard
that already applies to a wellness product handling this kind of data.

WHAT THIS IS NOT
----------------
This is not legal advice and not a compliance implementation. It is a
conservative technical control that makes the default behaviour safe and the
unsafe behaviour explicit, logged and opt-in. Real deployment needs a lawful
basis, a consent record, retention limits and a DPIA - none of which live in a
take-home baseline.
"""

from __future__ import annotations

import re

#: GDPR Art. 9(1) special categories, in the order they are tested.
#: Ordered most-specific first so a facet lands in its most precise category.
SPECIAL_CATEGORIES: tuple[tuple[str, str], ...] = (
    (
        "sex_life_or_orientation",
        # Bare "orientation" is deliberately NOT matched: it swept in
        # "Value orientation: Universalism" and "Patient care orientation",
        # which have nothing to do with sex life. Require the specific sense.
        r"\b(kink|sexual|sexuality|sex[- ]life|libido|erotic|fetish|"
        r"celibacy)\b",
    ),
    (
        "religious_or_philosophical_belief",
        r"\b(religio\w*|spiritual\w*|faith|pray\w*|worship\w*|meditat\w*|"
        r"mindfulness|holiness|sacred|divine|pilgrimage|scripture|quran|"
        r"bible|zohar|torah|gita|sufi|dhikr|sikh|hindu|jewish|buddhist|"
        r"islamic|christian|bahá|bahai|gnostic|kabbalah|sephira|astrology|"
        r"i ching|hexagram|reiki|channeling|satya|vrata|khatam|shabbat|"
        r"sukkot|ridv|seerah|archon|theolog\w*|epistemolog\w*|karma|dharma)\b",
    ),
    (
        "health_physical_or_mental",
        r"\b(health|medical|clinical|diagnos\w*|disorder|syndrome|symptom\w*|"
        r"depress\w*|anxiet\w*|hypomania|hysteria|psychotic\w*|psychoticism|"
        r"burnout|fatigue|apnea|apnoea|chronic pain|drug[- ]use|addiction|"
        # "neurotic" is deliberately NOT matched: Neuroticism is a Big Five
        # personality dimension, not a health condition, and matching it swept
        # an ordinary observable trait into the special-category bucket.
        r"substance|suicid\w*|trauma|therapy|disab\w*|illness|"
        r"sleep[- ]disorder)\b",
    ),
    (
        "genetic_or_biometric",
        r"\b(gene|genetic|genome|allele|polygenic|chromatin|hormone|fsh|"
        r"parathyroid|basophil|eosinophil|serotonin|cortisol|glucose|"
        r"biometric|metabolic|macronutrient|immune)\b",
    ),
    (
        "racial_or_ethnic_origin",
        r"\b(ethnic\w*|ethnocentr\w*|racial|race|nationality|citizenship|"
        r"multicultural\w*|cultural identity|ancestry|caste|indigenous)\b",
    ),
    (
        "political_opinion",
        r"\b(political|politics|liberalism|conservatism|patriotism|"
        r"ideolog\w*|partisan)\b",
    ),
    (
        "trade_union_membership",
        r"\b(trade[- ]union|unioni[sz]ed|collective bargaining)\b",
    ),
)

_COMPILED = tuple(
    (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in SPECIAL_CATEGORIES
)

#: Facets carrying a special category are refused unless explicitly allowed.
POLICY_REASON = (
    "Refused by default policy: this facet concerns {category}, a special "
    "category of personal data under GDPR Article 9(1). Inferring it from "
    "conversation without an explicit lawful basis is not permitted. Re-run "
    "with --allow-sensitive to override for evaluation purposes."
)


def special_category(facet_text: str) -> str | None:
    """Return the Art. 9 special category this facet concerns, or None.

    Matched against the facet name and its source qualifier. Deliberately
    over-inclusive: a false positive costs one abstention, a false negative
    means silently profiling someone on protected data.
    """
    for name, pattern in _COMPILED:
        if pattern.search(facet_text):
            return name
    return None


def policy_reason(category: str) -> str:
    return POLICY_REASON.format(category=category.replace("_", " "))
