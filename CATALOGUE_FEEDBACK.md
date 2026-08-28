# Notes for the facet catalogue owner

Written while building the scoring pipeline. None of this blocked the work -
the pipeline handles every case below - but a catalogue is a product surface,
and these are the things I would raise with whoever maintains it.

All figures come from `artifacts/audit_report.md` and
`artifacts/near_duplicates.md`, both generated from the raw CSV.

---

## 1. Only ~59% of the catalogue is scorable from conversation

Of 399 rows, **235 (58.9%)** can legitimately be evidenced by conversation
text. The remaining 164 need something else entirely:

| what it needs | rows | examples |
|---|---:|---|
| external records or logs | 53 | `Passport-stamps count`, `Commute time/day` |
| a standardised test | 28 | `Intelligence Quotient (IQ)`, `Spatial perception` |
| clinical assessment | 13 | `Sleep-disorder diagnosis`, `Compassion Fatigue` |
| a lab or biometric measurement | 12 | `FSH level`, `Basophil count` |
| a questionnaire or third-party rater | 10 | `Ethical leadership rating` |
| nothing - they are section headers | 31 | `Leadership Potential:` |

**Why this matters commercially.** If the catalogue is presented as "facets we
can assess", roughly 4 in 10 entries cannot be delivered from conversation
alone. That is a scoping question, not a modelling one. The options are to
split the catalogue by evidence source, to collect the other sources, or to
mark the difference in the product.

---

## 2. 31 rows are section headers, not facets

`Leadership Potential:`, `HEXACO Personality Inventory Facets:`,
`Time Orientation End Points:`, `Work Styles`, and 27 others are structural
labels from whatever documents the catalogue was assembled from.

They are detectable - most end in a colon, the rest end in a plural grouping
noun - but they are indistinguishable from facets to any consumer that just
reads the column. **They rank highly in retrieval:** `Leadership Potential:` was
the single top hit for a leadership conversation during development, ahead of
every real facet.

**Suggestion.** A `row_type` column (`facet` / `section_header`) would remove
the guesswork. Detecting them heuristically works, but it is inference where a
column would be fact.

---

## 3. The catalogue carries a numeric ID that only some rows have

31 rows are prefixed with a catalogue ID (`800. Sufi practice: ...`,
`754. I Ching hexagram 36 ...`); the other 368 are not. The IDs run into the
900s, so this looks like a merge of a numbered source into an unnumbered one.

**Suggestion.** Promote the ID to its own column and give every row one.
Stable IDs make it possible to version the catalogue, and right now nothing
identifies a facet except its exact text.

---

## 4. Redundancy that no string comparison will find

There are **zero exact duplicates** - the catalogue is clean by any textual
test. But cosine similarity over facet embeddings surfaces **37 pairs above
0.85**:

- `Character strength: Perseverance` and `Perseverance` - **cosine 1.000**,
  the same construct entered twice under different source instruments.
- **Seven `I Ching hexagram N resonance level` rows** at 0.98-0.99 with each
  other. They differ only by hexagram number; to an embedding model they are
  one facet.
- `Depression Symptoms`, `Depression (DEP)` and
  `Depression: Feelings of sadness and hopelessness` - three rows, one construct.
- `Auditory memory` / `Auditory Memory Recall`,
  `Delegation skills` / `Delegation Ability`.

**Practical cost.** Retrieval returns several near-identical facets in one
top-K, so scoring budget is spent answering the same question repeatedly, and
any per-facet aggregate double-counts whatever the cluster measures.

**Deliberately not fixed here.** A pipeline should not silently deduplicate
someone else's instrument - two rows that look redundant may be a real
distinction in the scale they came from. The pairs are reported in
`artifacts/near_duplicates.md` for a domain owner to adjudicate.

---

## 5. Five rows are encoding-sensitive

```
Big Five facet Openness – Artistic Interests
926. Bahá'í spiritual metric: Ridván festival participation
Psychological construct: Cultural Intelligence – Behavioral
516. Religious coping – Negative
823. Buddhist practice: Eightfold Path – Right Intention level
```

These contain en-dashes (`–`, not `-`) and accented characters. The file is
valid UTF-8, but anything reading it as cp1252 - which is the Windows Excel
default - will mojibake them, and a mojibaked facet name will silently fail any
exact-match lookup.

**Suggestion.** Normalise the punctuation to ASCII hyphens where the dash is
not semantically meaningful, and keep the accents but document the encoding.

---

## 6. Mixed granularity and mixed frames of reference

The catalogue mixes several kinds of thing that behave very differently:

- **Bare disposition nouns** - `Naivety`, `Cunningness`, `Dignity`. 100 rows
  (25%) match no classification keyword at all, because there is nothing to
  match; they are single words.
- **Fully-specified metrics** - `793. Sufi practice: Dhikr repetitions / day`.
- **Instrument sub-scales** - `Big Five facet Openness - Artistic Interests`.
- **Compound statements** - `Assertiveness and control in relationships` is two
  constructs joined by "and".

A scorer has to treat these very differently, and a single flat column gives no
signal about which is which. The enrichment step infers it, but inference is
what a schema is supposed to make unnecessary.

---

## 7. The part I would escalate: special-category data

**71 of 399 rows (18%)** concern data that GDPR Article 9(1) treats as special
category - religious or philosophical belief (39), health (13), genetic or
biometric (11), racial or ethnic origin (4), political opinion (3), sex life (1).

Some are unmistakable: `Kink-interest diversity`, `Drug-use history`,
`Physical-violence exposure`, `Polygenic risk: cardiovascular disease`,
`Depression (DEP)`, and religious practice metrics spanning Hindu, Islamic,
Jewish, Sikh, Bahá'í, Buddhist, Sufi, Kabbalistic and Gnostic traditions.

**This is not a data-quality note, it is a compliance one.** Article 9
prohibits processing these categories without a specific lawful basis, usually
explicit consent. Inferring them from conversation - rather than asking - does
not avoid that; arguably it makes it worse, because the person never disclosed
the information at all.

This pipeline therefore **refuses these by default** (`policy_blocked`,
overridable only with `--allow-sensitive`), and the enriched catalogue carries a
`special_category` column so the classification is inspectable.

**What I would want to know before shipping:** what lawful basis covers these
facets, whether consent is collected per-category, and whether the religious
practice metrics in particular are intended for inference or only for
self-report. Sixteen tradition-specific religious facets is a deliberate design
choice by someone, and it should be an explicit one.

---

## Summary of suggested columns

| column | why |
|---|---|
| `row_type` | separate the 31 section headers from real facets |
| `facet_id` | stable identity for all 399 rows, not just the numbered 31 |
| `evidence_source` | conversation / instrument / lab / records / self-report |
| `special_category` | Art. 9 flag, for consent routing |
| `canonical_id` | group the near-duplicate clusters without deleting rows |

The pipeline computes all of these today. They would be more reliable as data
than as inference.
