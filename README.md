# FacetLens

Scores conversation text against a heterogeneous 399-facet catalogue - and, more
importantly, decides which facets may legitimately be scored at all.

Built for the Ahoum AI/ML Engineer take-home. The hard part of this assignment is
not prompting an LLM for scores; it is refusing to produce scores that the
conversation does not support.

---

## The problem

The supplied catalogue mixes things a conversation can genuinely evidence
(`Talkativeness`, `Delegation skills`) with things it cannot: lab values
(`FSH level`, `Basophil count`), clinical constructs (`Sleep Apnea`), test scores
(`Intelligence Quotient (IQ)`), psychometric instrument outputs
(`Honesty-humility trait score`, `Ethical leadership rating`), biographical
facts (`Nationality`), activity counts requiring records
(`Passport-stamps count`), and 31 catalogue *section headers* that are not
facets at all (`Leadership Potential:`).

A retriever is perfectly happy to return `FSH level` for "I've been so tired
lately". The system's job is to know that retrieving it and being able to score
it are different questions.

---

## Architecture

```
conversation
     |
[Gate 1] RETRIEVAL - semantic relevance only
     |   MiniLM cosine over enriched facet text -> top-K
     |   Similarity is NEVER treated as evidence.
     v
[Gate 2] OBSERVABILITY - deterministic, pre-LLM, zero tokens
     |   conversation_observable == False  ->  not_observable + reason
     |   Reads a precomputed catalogue column. No model call.
     v
[Gate 3] SCORING - LLM, compact batches of 5, schema-constrained
     |   May still return insufficient_evidence.
     v
[Gate 3b] EVIDENCE VERIFIER - programmatic
         The model must quote verbatim. If the quote is not actually in the
         conversation, the citation was fabricated -> insufficient_evidence.
```

**The central claim.** Abstention for non-observable facets is a property of the
catalogue, not a judgement by the model. It costs no tokens and does not depend
on a 7B model reliably declining to answer. Gate 3b then catches fabricated
evidence on the facets that *are* observable.

The honest cost of that design: a taxonomy misclassification becomes a **false
abstention**. The benchmark measures that rate rather than assuming it is zero.

---

## Model and licence

| | |
|---|---|
| Scoring model | `qwen2.5:7b-instruct` via Ollama |
| Parameters | **7,615,616,512 (7.6B)** - under the 16B limit |
| Licence | **Apache-2.0** |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, Apache-2.0, 22.7M params |
| Decoding | `temperature=0`, `seed=42`, JSON-schema-constrained |

Both the parameter count and the licence were read from the local daemon
(`/api/show`), not from a model card:

```bash
curl -s http://127.0.0.1:11434/api/show -d '{"model":"qwen2.5:7b-instruct"}'
# general.parameter_count : 7615616512
# license                 : Apache License, Version 2.0
```

The machine's pre-installed model (`qwen3.6`, **36.0B**) was rejected for
breaking the size limit - see DEBUGGING.md #1.

---

## Setup

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct
```

CPU-only is fine; that is what this was developed and benchmarked on
(Ryzen 7 6800H, no GPU). Expect ~8 tok/s generation.

**No model? Everything still runs.** `--backend mock` and the test suite are
fully offline, and the committed LLM cache reproduces the benchmark report
exactly.

---

## Running it

```bash
python -m src.pipeline enrich
```
Rebuilds `data/processed/enriched_facets.csv` and `artifacts/audit_report.md`
from the raw CSV. Deterministic - the raw file is never modified.

```bash
python -m src.pipeline score --text "I led a team of five engineers."
```
Scores one conversation. Add `--backend mock` to run without a model.

```bash
python -m src.pipeline benchmark
```
Runs all 13 benchmark conversations and writes
`artifacts/benchmark_report.md`. Roughly 30 minutes cold on CPU; near-instant
from the committed cache.

```bash
python -m pytest tests/ -q      # 38 tests, no model or network needed
```

### Reproducibility check (actually performed)

Cloned this repository to a clean directory and verified from scratch:

- `enrich` regenerates `enriched_facets.csv` **byte-identically**
  (md5 `0ae85114...` before and after, `git status` clean)
- **38/38 tests pass** in the fresh clone with no setup beyond `pip install`
- `artifacts/embeddings/*.npy` is gitignored, was absent in the clone, and was
  **regenerated automatically** on first use
- no `.env`, `.key`, `.pem`, or token files are tracked

---

## Part 1 - facet audit

`data/processed/enriched_facets.csv` is generated entirely by code. Columns:
`facet_raw` (verbatim), `facet_normalized`, `facet_key`, `catalogue_id`,
`source_qualifier`, `facet_type`, `conversation_observable`, `sensitivity`,
`abstention_reason`, `scoring_definition`, `score_anchors`, `retrieval_text`,
`is_header_like`, `has_numeric_prefix`, `has_encoding_artifact`,
`classification_rule`.

What the audit found in 399 rows (full detail in `artifacts/audit_report.md`):

- **31 header-like rows** - catalogue sections, not facets. Detected by two
  independent signals, because a trailing colon alone misses `Work Styles`.
  Retained and flagged, never deleted.
- **31 numeric catalogue-ID prefixes** (`800. Sufi practice: ...`) - extracted
  into `catalogue_id`.
- **5 non-ASCII rows** - en-dashes and accents (`Bahá'í`, `Ridván`). Must be read
  as UTF-8; cp1252 mojibakes them.
- **0 exact duplicates and 0 normalisation collisions on the raw values.**
  Reported as a null result rather than dressed up.
- **1 collision exposed by qualifier stripping**: `Character strength:
  Perseverance` vs `Perseverance`, cosine **1.000**.
- **37 semantic near-duplicate pairs** found by reusing the embedding matrix -
  pairs no string test can catch, including a cluster of six
  `I Ching hexagram N resonance level` rows at cosine 0.98-0.99. Written to
  [`artifacts/near_duplicates.md`](artifacts/near_duplicates.md) by
  `python -m src.pipeline embed`. They are reported, never merged: deduplicating
  someone else's instrument is not this pipeline's call.

- **10 psychometric instrument outputs** (`... score`, `... rating`) gated
  non-observable. These were originally misclassified as observable traits and
  were caught by the manual spot-check - see DEBUGGING.md #7.

Of 399 rows, **235 are conversation-observable**. Every row records which named
rule classified it, so any classification can be challenged. **100 rows (25%)
match no keyword rule** and fall through to a default of observable
`personality_trait` - mostly genuine bare disposition nouns (`Naivety`,
`Cunningness`), but that is a default, and it is the most likely remaining
source of error. No row is forced into `other`; the fallback absorbs ambiguity
instead.

**Scoring anchors.** `1` no/very weak - `2` weak - `3` moderate - `4` strong -
`5` very strong. The quantity scored is *strength of evidenced expression in
this conversation*, explicitly not the person's true trait level. That framing
is what makes abstention coherent: "how strongly is leadership evidenced in
these three sentences?" is answerable from text; "what is this person's
leadership ability?" is not.

---

## Output schema

```json
{
  "facet_id": "F0284", "facet": "Delegation skills",
  "facet_type": "interpersonal", "status": "scored", "score": 4,
  "confidence": 0.8,
  "reason": "The speaker describes assigning tasks based on team strengths.",
  "evidence_quote": "assigned tasks based on their strengths",
  "origin": "llm", "retrieval_score": 0.31,
  "evidence_verified": true, "schema_repaired": false
}
```

```json
{
  "facet_id": "F0032", "facet": "FSH level",
  "facet_type": "biometric_physiological", "status": "not_observable",
  "score": null, "confidence": 1.0,
  "reason": "Not scorable from conversation: classified as biometric_physiological (requires_biometric_measurement).",
  "origin": "observability_gate"
}
```

Statuses: `scored`, `not_observable`, `insufficient_evidence`, `error`.
Enforced by Pydantic, not merely documented: `score` is an integer 1-5 exactly
when `status == "scored"` and null otherwise, `confidence` is in [0,1], `reason`
is non-empty, unknown fields are rejected.

`origin` records which gate produced the verdict, which is what makes the
failure analysis attributable.

**On confidence:** for `observability_gate` verdicts, confidence is 1.0 because
the gate is a deterministic lookup - given the catalogue, the verdict is
certain. Whether the *catalogue* is right is a separate question the audit
addresses. For LLM verdicts it is the model's **self-reported, uncalibrated**
number. The benchmark reports it bucketed against actual agreement so you can
see how much it is worth.

---

## Part 3 - benchmark and results

13 conversations covering clear, ambiguous, contradictory, quoted, sarcastic,
code-switched and low-evidence cases, plus 3 hallucination traps. 30 facets, all
verbatim catalogue rows (14 observable / 16 not). 55 human-reviewed reference
pairs, each with a written rationale.

**Full generated results: [`artifacts/benchmark_report.md`](artifacts/benchmark_report.md).**
Every number there is produced by `src/evaluation/evaluate.py`; none is
hardcoded.

| metric | result |
|---|---|
| Status agreement (scored vs abstained vs refused) | 41/55 (74.5%) |
| Exact score agreement | 5/11 (45.5%) |
| **Within +/-1** | **11/11 (100%)** |
| Correct abstentions | 30/36 (83.3%) |
| Missed abstentions (scored something unsupported) | 6 |
| False abstentions (abstained where a score was expected) | 8 |
| Verdicts ending in `error` | **1 of 362** |
| Facets answered with **no LLM call** | 136 (38%) |

**Zero hallucination-trap failures.** Across all three traps, no medical, lab,
financial-count or religious-practice facet was scored. Every `not_observable`
gate held, in every run.

### The scale change that made the numbers worse

These figures are **lower than an earlier configuration of the same system**,
and the reason is a deliberate fix that did not pay off. Both configurations
were measured on the identical benchmark:

| | anchor v1 (shipped earlier) | **anchor v2 (shipped now)** |
|---|---|---|
| level 1 means | "no or very weak evidence" | "present but minimally expressed" |
| status agreement | **85.5%** | 74.5% |
| exact score agreement | **56.2%** | 45.5% |
| within +/-1 | 93.8% | **100%** |
| false abstentions | **3** | 8 |
| missed abstentions | 5 | 6 |

**Why v2 exists.** v1 contained a genuine specification contradiction: "no
evidence" was simultaneously the definition of score 1 *and* the definition of
`insufficient_evidence`. The model was offered two correct answers. It was
caught on *"Things are okay"*, where `Enthusiasm` scored 1 with the reason *"the
statement is neutral and does not express enthusiasm"* - correct reasoning,
wrong output.

**What v2 cost.** The contradiction is gone (that conversation now scores
nothing at all, which is right), but raising the bar for level 1 made the model
markedly more conservative overall: **5 additional false abstentions**, on
facets the reference expects to score 2 or 3.

**Why v2 ships anyway.** Three reasons, and I would defend all three:

1. The errors moved in the **safe** direction. A false abstention costs
   coverage; a missed abstention invents a fact. Missed abstentions rose by
   only 1, while false abstentions absorbed the change.
2. **Within +/-1 is now 100%.** When the system does commit to a score, it is
   never more than one level from the human label. v1 scored more often and
   less precisely.
3. v1's headline number was partly earned by a defect. A scale that lets
   "absent" be scored as 1 will agree with a reference set more often *and* be
   wrong about what it is measuring.

**The honest counter-argument**, which I am not hiding: 74.5% is worse than
85.5%, the fix was found late, and there was no time to re-tune the anchor
wording after seeing its effect. A better v3 probably exists - level 1 as
"present but minimal" with an explicit instruction that hedged self-report
still counts as evidence - and it was not attempted because validating it
needs another full benchmark run.

### Where it actually fails

The five missed abstentions are not five instances of the same problem, and
counting them as one number would hide that.

**Two are genuine reasoning failures, both sarcasm.** On *"I'm the world's
greatest communicator... three people shipped to the wrong environment"*, the
system scored `Talkativeness` 5 and `Enthusiasm` 4. The prompt explicitly says
sarcasm inverts meaning; a 7B model read the exaggeration literally anyway. One
reason is revealing - *"the speaker is highly talkative, providing multiple
statements and elaborating on an anecdote"* - it scored the **act of writing a
long message** rather than what the message says. Sarcasm is the clearest
capability gap in this system.

**One is a flaw in my own scale, not the model's reasoning.** On *"Things are
okay"*, `Enthusiasm` was scored **1** with the reason *"The statement is neutral
and does not express enthusiasm."* The model reasoned correctly and still did
not abstain - because anchor 1 is defined as *"no or very weak evidence"*, which
collides directly with `insufficient_evidence`. It was offered two correct
answers and picked the other one. **This is a design bug I introduced.** The fix
is to redefine anchor 1 as *"the facet is clearly present but minimally
expressed"*, reserving absence for abstention. It is not applied here: changing
the anchors changes every prompt, invalidating the whole LLM cache and costing a
40-minute re-run with no time left to validate the result. Documented rather
than quietly left.

**Two are over-inference** - `Withdrawnness` from a contradictory statement, and
`Self-improvement` from a description of weekly planning.

**All three false abstentions trace to one debatable convention in my reference
labels,** not to system error. I decided a hedged self-report (*"I guess I'm
pretty good at working with people"*) earns a 2; the system says no behaviour is
described, so abstain. That is arguably the system being *more* evidence-
disciplined than my own labels. It is recorded as a disagreement rather than
silently resolved in the system's favour.

### Hallucination traps

Each is built from real catalogue rows, and each must abstain:

| Conversation | What a naive scorer infers | Required behaviour |
|---|---|---|
| "so tired the last few weeks..." | anaemia / depression / sleep apnea from `FSH level`, `Basophil count`, `Sleep Apnea`, `Sleep-disorder diagnosis`, `Compassion Fatigue` | all `not_observable` - a symptom is not a diagnosis |
| "cut my spending... putting something aside" | an income figure or a `Subscription count` | `not_observable` - no quantity is stated |
| "calm and centred after my morning practice" | a religion *and* a frequency, from `892. Hindu... Yoga discipline hours / week`, `793. Sufi practice: Dhikr repetitions / day`, `Pilgrimage participation count` | all `not_observable` - the word "practice" names no tradition and no count |

A fourth guard is embedded in the code-switched case: Hindi-English mixing must
not be converted into an inferred `Nationality`.

### Red-team: attacking my own defences

Beyond the three required hallucination traps, six attacks were run against the
system's own protections. Full results:
[`artifacts/adversarial_report.md`](artifacts/adversarial_report.md).

| attack | result |
|---|---|
| instruction injection inside the conversation | **held** - 0 scored |
| speaker reciting facet names to bait scoring | **held** - 0 of 20 scored |
| conversation stating the medical facts the gate blocks | **held**, but the gate is evidence-insensitive by design |
| third-party "you obviously have depression" | **held** - 0 clinical scores |
| voluntary disclosure of religious belief | **LEAKED** |
| evidence verifier given a real-but-irrelevant quote | **bypassed**, as documented in D6 |

**The leak is the important one.** On a conversation disclosing religious
practice, six religious facets were correctly refused - and then
`Patience: Resistance to anger` was scored **5 at confidence 1.00**, quoting
*"my faith is the main thing keeping me steady"*. `Peacefulness` scored 4 on the
same sentence.

The policy gate filters **facets, not evidence**. It asks "may we score this
facet?" and never "may we use this sentence?". So religion still shapes the
profile, via a facet that looks innocuous, with nothing in the output to
indicate a protected attribute was the input. Under the Article 9 reasoning the
gate is built on, inferring *from* special-category data is itself processing
it.

**Fix, shipped (Gate 3c).** `special_category()` now runs over the
`evidence_quote` as well as the facet name, so a verdict whose supporting
evidence is itself a special-category disclosure is refused with
`policy_blocked`. Two regression tests cover both directions: a faith quote is
refused, ordinary delegation evidence is not.

The table above is the run that **found** the leak, and is kept as the evidence
that it was real. `artifacts/adversarial_report.md` is regenerated by
`python -m src.pipeline redteam` against the current code.

Worth recording: I predicted in advance that the *medical-facts* and *verifier*
attacks would fail. The medical one held. The leak was in the attack I expected
to pass, and it surfaced only because the attack was executed rather than
reasoned about.

### Retrieval is the weakest component, and four attempts did not fix it

Measured **before** any labelled facet is force-included, so misses are visible
rather than absorbed. Full table:
[`artifacts/ablation_retrieval.md`](artifacts/ablation_retrieval.md).

Recall of facets the reference says **should be scored** (n=19):

| K | bare name | enriched | BM25 | dense+BM25 | **expansions (shipped)** |
|---:|---|---|---|---|---|
| 10 | 8/19 | 9/19 | 1/19 | 6/19 | **9/19** |
| 25 | 11/19 | 12/19 | 2/19 | 10/19 | **12/19 (63%)** |
| 40 | 14/19 | 12/19 | 4/19 | 13/19 | **13/19** |
| 100 | 16/19 | 16/19 | 12/19 | 16/19 | **17/19 (89%)** |

**Four interventions were built and measured. None materially moved recall.**
Two made it actively worse (BM25 alone, and dense+BM25 fusion). A fifth,
swapping MiniLM for the stronger BGE-small-en-v1.5, scored 10/19 at K=25
against MiniLM's 12/19 and is recorded in DECISIONS.md D10. Document expansion
ties at four values of K and gains one facet at two; it ships because it never
loses, not because it works.

**Why the expansion idea misled me.** It was validated by hand first -
appending example utterances moved three facets from rank 7->2, 7->2 and 9->3.
But I wrote those examples *after reading the target conversation*. Generated
blind from a definition, `Collaboration` gets *"I enjoy working in teams"* -
a perfectly good example, and nothing like *"we worked through it together
until we had something everyone could live with"*. The hand test measured the
ceiling, not the method.

**What the failures did buy: a localised diagnosis.** It is not the encoder,
not the similarity function, not the indexed text - swapping each in turn
changed almost nothing. A bi-encoder cannot bridge a short abstract label to a
long concrete narrative. Recall reaches 89% at K=100, so the candidates *are*
retrievable; they are just not ranked. That points squarely at a **cross-encoder
reranker over a wide candidate set**, which is the top next step and was not
attempted because it adds a second model to a machine already generating at
8 tokens/second.

**This does not corrupt the scoring metrics.** Reference-labelled facets that
retrieval misses are force-included before scoring (DECISIONS.md D8), so
agreement measures scoring quality given the right facets, reported alongside
the recall figure above rather than blended into it.

## Limitations

1. **The taxonomy is heuristic, and it has a measured error rate.** Keyword and
   pattern rules, no training. A seeded 30-row manual review found **4
   misclassifications (13.3%), all in the dangerous direction** - marked
   observable when they were not (DEBUGGING.md #7). They were fixed, but that
   sample can no longer estimate the post-fix rate, and ~27% of rows still
   depend on a fallback default. Assume roughly a 10% observability error rate
   until a fresh sample says otherwise.
2. **Retrieval recall is poor** - 63% at K=25 on facets that should be scored.
   This is the single biggest weakness.
3. **Confidence is not calibrated.** LLM confidence is self-reported. The report
   shows agreement per confidence bucket instead of asserting reliability.
4. **The evidence verifier is a substring check.** A paraphrasing model loses a
   legitimate score; a model quoting something real but irrelevant still passes.
   It is a cheap lower bound on evidence integrity, not proof of relevance.
5. **The reference set is small (55 pairs) and partial**, so percentages move a
   lot per item. Counts are always reported with denominators.
6. **Single model, single seed.** No variance estimate across runs or models.
7. **This is a baseline, not a production system.** No auth, rate limiting,
   observability, or PII handling - and `sensitivity` is recorded in the
   catalogue but not yet enforced as a policy.

---

## Scaling to 5,000 facets

**Indexing.** Embedding cost is linear and trivial: 399 facets encode in 0.13s,
so 5,000 is ~1.6s, one time. The matrix is 5000 x 384 float32 = **7.7 MB**, and
it is cached to disk under a hash of its inputs.

**Retrieval.** A query is one matrix-vector product - well under a millisecond
at this size, with `argpartition` keeping the top-K selection O(n). No ANN index
is warranted; see DECISIONS.md D3. That changes somewhere around 10^5-10^6
facets, where an IVF/HNSW index and sharding start to earn their complexity.

**Where the cost actually is.** Retrieval and gating are O(catalogue); LLM calls
are O(observable facets retrieved), which is bounded by `top_k`, **not** by
catalogue size. Going from 399 to 5,000 facets therefore leaves the number of
LLM calls unchanged at a fixed `top_k`. That is the property that makes this
design scale.

**The first bottleneck is LLM inference, and it is not close.** Measured on this
machine: one batch of 5 facets takes ~50s (21s prompt evaluation at 33 tok/s,
~30s generation at 8.2 tok/s). Vector search is sub-millisecond. The ratio is
roughly **3,000x**. Any effort spent optimising retrieval before inference is
misdirected.

**Consequences at 5,000 facets:**
- **Batching** stays at 5 facets/call; total calls depend only on `top_k`.
- **Caching** matters more, not less - keyed on model, options, schema and
  prompt, so identical conversation/facet pairs are never re-inferred.
- **Latency** is dominated by the ~50s/batch serial cost. The fixes are
  concurrent batch requests, a GPU (this ran CPU-only), or a smaller model for
  a first pass with escalation only on low-confidence facets.
- **The observability gate is also a cost control.** In this benchmark it
  answers a large share of all verdicts with zero tokens, and that share grows
  with a catalogue as medical- and metric-heavy as this one.

---

## With another day

1. **Re-tune the anchor scale (v3).** v2 fixed a real contradiction and cost 11
   points of status agreement by over-abstaining. Keep level 1 as "present but
   minimal" and add one instruction: a hedged self-report is still evidence and
   belongs at 2, not at abstention. That targets the 8 false abstentions
   directly. One benchmark run to validate.
2. **A cross-encoder reranker over K=100.** Recall is 89% at K=100 and 63% at
   K=25, so the right facets are retrieved and simply not ranked. Four
   bi-encoder-side interventions failed (DECISIONS.md D10/D11); this is the
   remaining hypothesis.
3. **Second-model agreement** as a confidence signal, since self-reported
   confidence measured as uninformative (0.9 bucket -> 44% agreement). Try
   `bge-small-en-v1.5`, add lexical BM25 as a hybrid channel (abstract trait
   nouns are exactly where dense retrieval underperforms), and expand each facet
   with 2-3 generated example phrasings before embedding.
4. **Adjudicate the fallback bucket** - hand-label a stratified sample of the
   ~27% of rows that match no rule, and use the disagreement rate to decide
   whether the default is defensible or needs a classifier.
5. **Tune batch size empirically** - 5 is reasoned, not measured. Sweep 3/5/8/12
   for wall-clock against malformed-output rate.
6. **Strengthen the evidence verifier** - fuzzy span matching to stop punishing
   near-verbatim quotes, plus a check that the quote is *relevant* to the facet
   rather than merely present.
7. **Second-model agreement** - run a second open-weight model and treat
   disagreement as a low-confidence signal, which is a far better confidence
   estimate than self-report.
8. **Grow the reference set** and get a second annotator, so agreement can be
   reported with an inter-annotator baseline.

---

## Reference labels - provenance

`data/benchmark/reference_labels.jsonl` was **drafted with AI assistance and
requires review by the candidate** before submission. Each label carries a
rationale, and the contested categories follow stated rules: contradiction
resolves toward described behaviour over stated preference; quoted praise is
attributed to whoever said it, not its subject; sarcasm is not read literally;
code-switching neither reduces evidential weight nor implies a nationality.
Stating this is the honest position - see PROMPT_LOG.md.

---

## Repository

```
data/raw/            immutable source CSV
data/processed/      generated enriched catalogue
data/benchmark/      conversations, facet selection, reference labels
src/preprocessing/   normalize, taxonomy, anchors, enrich
src/retrieval/       embed (cached index), retrieve (Gates 1 + 2)
src/scoring/         schema, prompts, backends, parser, scorer (Gate 3 + 3b)
src/evaluation/      benchmark runner and report generation
tests/               38 tests, no model or network required
artifacts/           audit report, near-duplicates, benchmark report,
                     ablation, LLM cache
```

`DECISIONS.md` - 9 non-trivial decisions with trade-offs.
`DEBUGGING.md` - 7 real issues, each with symptom, root cause, fix, verification.
`PROMPT_LOG.md` - how AI was used, and four concrete things it got wrong.
