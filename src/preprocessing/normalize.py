"""Deterministic normalisation of raw facet strings.

The raw catalogue value is never mutated. Every transform here is pure and
reproducible: running it twice on the same input yields byte-identical output.

Design note: normalisation is kept *separate* from classification. Normalising
answers "what is this string, tidied?"; the taxonomy answers "what kind of thing
is it, and can a conversation reveal it?". Mixing the two makes both untestable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Rows like "800. Sufi practice: Sufi retreat attendance count" carry a catalogue
# ID that is metadata, not part of the facet name.
_NUMERIC_PREFIX = re.compile(r"^\s*(\d+)\.\s+")

# Rows like "Psychological construct: Hope scale" carry a source-instrument
# qualifier before the colon. It is useful metadata but noise in the facet name.
_QUALIFIER_PREFIX = re.compile(
    r"^(?P<qualifier>"
    r"Psychological construct|Social-cognition variable|Cognitive measure|"
    r"Character strength|Emotional-intelligence measure|Value orientation|"
    r"Defense-mechanism tendency|Well-being component|Mindfulness facet|"
    r"Attachment style|Practice frequency|Religious practice|Sufi practice|"
    r"Islamic practice|Buddhist practice|Hindu spiritual metric|"
    r"Jewish spiritual metric|Sikh spiritual metric|Bahá'í spiritual metric|"
    r"New-Age spiritual metric|Gnostic spiritual metric|Spiritual virtue|"
    r"Sacred text engagement|Energy-healing practice|Kabbalah sephira balance|"
    r"Astrology|Polygenic risk|Macronutrient ratio|Depression|Religious coping|"
    r"Big Five facet|HEXACO domain"
    r"):\s*(?P<rest>.+)$"
)

# Unicode characters that appear in this catalogue and confuse naive matching.
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_QUOTES = {ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"'}


@dataclass(frozen=True)
class NormalizedFacet:
    """The output of normalising one raw catalogue row."""

    facet_raw: str
    facet_normalized: str
    facet_key: str
    catalogue_id: str | None
    source_qualifier: str | None
    is_header_like: bool
    has_numeric_prefix: bool
    has_encoding_artifact: bool


def _strip_accents_for_key(text: str) -> str:
    """Fold accents so 'Bahá'í' and 'Baha'i' collide in duplicate detection."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def make_key(text: str) -> str:
    """Aggressive casefolded key used ONLY for duplicate detection, never display."""
    folded = _strip_accents_for_key(text).casefold()
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def _looks_like_header(raw: str) -> bool:
    """Detect catalogue section headers masquerading as facets.

    Two independent signals, because neither alone is sufficient:
      1. A trailing colon  ("Leadership Potential:") - 30 rows in this catalogue.
      2. A plural grouping noun ("... Subcomponents", "... Facets", "... Types").

    Signal 2 catches rows like "Work Styles" that have no trailing colon.
    """
    stripped = raw.strip()
    if stripped.endswith(":"):
        return True
    # The grouping noun must be PLURAL. Allowing the singular made
    # "Adventure-Seeking Behavior" - a real facet - register as a section
    # header and get gated out as non-observable. A header names a group.
    grouping = (
        r"\b(subcomponents|components|facets|themes|types|styles|"
        r"parameters|end\s+points|drivers|behaviors|behaviours)\s*$"
    )
    return bool(re.search(grouping, stripped, re.IGNORECASE))


def normalize_facet(raw: str) -> NormalizedFacet:
    """Normalise one raw facet string into its display and matching forms."""
    has_artifact = any(ord(ch) > 127 for ch in raw)

    # NFKC folds compatibility forms; then unify the dash/quote zoo.
    text = unicodedata.normalize("NFKC", raw).translate(_DASHES).translate(_QUOTES)
    text = text.strip()

    catalogue_id = None
    if (m := _NUMERIC_PREFIX.match(text)) is not None:
        catalogue_id = m.group(1)
        text = text[m.end():]

    header_like = _looks_like_header(text)

    source_qualifier = None
    if (m := _QUALIFIER_PREFIX.match(text)) is not None:
        source_qualifier = m.group("qualifier")
        text = m.group("rest").strip()

    # A trailing colon is a header marker, not part of the name.
    text = text.rstrip(":").strip()
    text = re.sub(r"\s+", " ", text)

    return NormalizedFacet(
        facet_raw=raw,
        facet_normalized=text,
        facet_key=make_key(text),
        catalogue_id=catalogue_id,
        source_qualifier=source_qualifier,
        is_header_like=header_like,
        has_numeric_prefix=catalogue_id is not None,
        has_encoding_artifact=has_artifact,
    )
