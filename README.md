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
python -m pytest tests/ -q      # 57 tests, no model or network needed
```

### Start here if you are reviewing this

**[`artifacts/report.ipynb`](artifacts/report.ipynb)** is a rendered walkthrough -
the catalogue audit, the four gates running on a live conversation, the
hallucination traps abstaining, the pipeline surviving five kinds of malformed
model output, and the results including what does not work. GitHub renders it
in the browser; nothing to install and nothing to run.

It is **generated** by `python -m src.pipeline notebook`, which executes every
cell and embeds the real output. No number in it is typed by hand, so it cannot
drift from the code - which matters here, because four separate documentation
drifts were caught and logged during this build (DEBUGGING.md #9, #11).

### Reproducibility check (actually performed)

Cloned this repository to a clean directory and verified from scratch:

- `enrich` regenerates `enriched_facets.csv` **byte-identically**
  (md5 `0ae85114...` before and after, `git status` clean)
- **57/57 tests pass** in the fresh clone with no setup beyond `pip install`
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
| **Status agreement** (scored vs abstained vs refused) | **48/55 (87.3%)** |
| **Correct abstentions** | **33/36 (91.7%)** |
| Within +/-1 | 13/15 (86.7%) |
| Exact score agreement | 3/15 (20.0%) - see the calibration note below |
| Missed abstentions (scored something unsupported) | 3 |
| False abstentions (abstained where a score was expected) | 4 |
| Verdicts ending in `error` | **1 of 361** |
| Facets answered with **no LLM call** | 136 (38%) |

**Zero hallucination-trap failures**, in every run. No medical, lab,
financial-count or religious-practice facet has ever been scored.

### A correction: I published a causal claim the experiment could not support

An earlier version of this README stated that fixing the anchor scale cost 11
points of agreement. **That was wrong, and the error is instructive enough to
leave documented rather than quietly amend.**

Four things changed between those two runs, not one:

| | earlier run | later run |
|---|---|---|
| anchor scale | v1 | v2 |
| **retrieval mode** | **dense** | **hybrid** |
| Gate 3c evidence policy | absent | present |
| facet expansions | absent | present |

The retrieval default was the culprit. `retrieve()` defaulted to `mode="hybrid"`
while `ablation.py` simultaneously reported **dense** as the shipped
configuration - so the pipeline ran a setting my own ablation had already
measured as worse, and the documentation described a setting the code did not
use. Full write-up in DEBUGGING.md #11.

Re-running with **only** that one variable reverted:

| | hybrid | **dense** |
|---|---:|---:|
| status agreement | 74.5% | **87.3%** |
| correct abstentions | 83.3% | **91.7%** |
| missed abstentions | 6 | **3** |
| false abstentions | 8 | **4** |

+12.8 points from a one-word default. The anchor scale was not the cause.

### Why exact score agreement is 20% and why that number is not what it looks like

Signed differences on the 15 pairs where both the reference and the system
scored:

```
system - reference:  +1 : 8 pairs   <- the mode
                     +0 : 3
                     -1 : 2
                     +2 : 1
                     -3 : 1
```

Eight of fifteen off by exactly +1 is a scale offset, not scatter. The reference
labels were written against **anchor v1**, where level 1 meant "no or very weak
evidence". Anchor v2 redefined level 1 as "present but minimally expressed", so
every level now asserts the facet is present and the whole scale moved up one
notch relative to the labels.

**The yardstick changed and the reference set did not.** That is why the metrics
which ignore the absolute level (status agreement 87.3%, correct abstentions
91.7%, +/-1 86.7%) are the best this system has produced, while the one metric
that depends on it collapsed.

**Deliberately not fixed.** Re-deriving the labels under v2 anchors *after*
seeing which direction improves the score is fitting the reference set to the
system, and it would destroy the only artefact whose worth depends on being
independent. The legitimate version - re-deriving each label from its own
written rationale under v2 definitions, blind to system output - is listed under
next steps.

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
| voluntary disclosure of religious belief | leaked, then **fixed** (Gate 3c) |
| evidence verifier given a real-but-irrelevant quote | **bypassed**, as documented in D6 |

Re-running the suite after the anchor-scale change then caught a **regression
nobody predicted**: trait-name baiting, which previously produced 0 scores from
20 candidates, now produces 3 (two at 4 or above). A speaker reciting
*"excellent Delegation skills, very strong Collaboration"* describes no
behaviour at all, so the correct answer is 1 or abstention. It is the same
over-correction the benchmark shows, seen from another angle, and it is not
fixed here - the fix is the anchor v3 re-tune, which needs a validation run
there was no time for.

That is the argument for keeping an adversarial suite in the repository rather
than running it once. It is the only test here that caught a safety regression
introduced by a change made for an entirely unrelated reason.

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

| K | bare name | enriched | BM25 | dense+BM25 | expansions | **+rerank (shipped)** |
|---:|---|---|---|---|---|---|
| 10 | 8/19 | 9/19 | 1/19 | 6/19 | 9/19 | 8/19 |
| 15 | 10/19 | 10/19 | 1/19 | 9/19 | 10/19 | **11/19 (58%)** |
| 25 | 11/19 | 12/19 | 2/19 | 10/19 | 12/19 | **13/19 (68%)** |
| 40 | 14/19 | 12/19 | 4/19 | 13/19 | 13/19 | **15/19 (79%)** |
| 60 | 15/19 | 15/19 | 5/19 | 14/19 | 15/19 | **16/19 (84%)** |
| 100 | 16/19 | 16/19 | 12/19 | 16/19 | 17/19 | **17/19 (89%)** |

**Five interventions were built and measured. Four failed. The fifth worked -
and it only became findable because the four failures narrowed down where the
problem was.**

- **BM25 alone** (11% @ K=25) is near-useless here: conversations describe
  behaviour, facets are abstract labels, and there is almost no lexical overlap
  to exploit by construction.
- **Dense+BM25 fusion** (53%) inherits that weakness - fusing a strong signal
  with a near-random one drags the ranking down.
- **BGE-small-en-v1.5** (53%), a stronger encoder on public benchmarks, scored
  *worse* than MiniLM. Its recommended query prefix made it worse again.
- **Document expansion** (63%) tied the incumbent. It was hand-validated first,
  but with examples written by someone who had already read the target
  conversation - the hand test measured the ceiling, not the method.

**What those four ruled out was the answer.** Swapping the encoder, the
similarity function and the indexed text each changed almost nothing, leaving
one fact standing: recall is **89% at K=100 and 63% at K=25**. The right facets
were already being retrieved and simply ranked badly.

That is a *ranking* problem, and no bi-encoder can solve it - it compresses each
side to a vector independently, so a short abstract label and a long concrete
narrative never meet. A cross-encoder reads both together.

**Built: retrieve 100 by cosine, reorder with `ms-marco-MiniLM-L-6-v2`
(22.7M params, Apache-2.0), keep top-K.** Best or tied-best at every K from 15
up. Worse at K=10 (47% -> 42%) - reranking a wide pool needs room to place what
it promotes. Cost 8.3s for all 13 conversations, ~1% of one LLM batch.

### And it is not the default, because it made the end-to-end result worse

| | **dense (shipped)** | +rerank |
|---|---:|---:|
| retrieval recall@25 | 34.5% | **40.0%** |
| **status agreement** | **87.3%** | 81.8% |
| correct abstentions | **91.7%** | 88.9% |
| false abstentions | **4** | 6 |

The reason is a flaw in my own reasoning when I proposed it. Reference-labelled
facets are **force-included regardless of retrieval**, deliberately, so
retrieval misses cannot hide inside the agreement number. That means improving
retrieval *cannot* improve agreement on those pairs - they were already being
scored. I built a fix for a metric that, by my own evaluation design, could not
move the metric I cared about.

What did change was batch composition: better ranking surfaced more observable
facets, the observability gate handled 18% of verdicts instead of 35%, more
facets reached the LLM, and verdicts shifted with the altered context. That is
perturbation, not improvement.

So `dense` remains the default and `rerank` ships as a measured option
(`--mode rerank`). **The retrieval figure quoted above is therefore not the best
this repository can produce** - 40.0% is available and 34.5% ships. That is the
cost of preferring the metric that reflects end-to-end quality over one that
measures a component in isolation. Full reasoning in DECISIONS.md D12.

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

1. **Re-derive the reference labels under v2 anchors, blind.** Eight of fifteen
   scored pairs are off by exactly +1 because the labels encode v1's scale. Each
   label carries a written rationale, so they can be re-derived from those
   rationales under the v2 definitions - but it must be done without sight of
   system output, ideally by someone who has not seen these results, or it is
   just fitting the reference set to the system.
2. **Record the run configuration in the report header.** Retrieval mode, anchor
   version, gate set and expansion state, generated from the live objects rather
   than from what the author believes is set. The confounded comparison that
   produced a false claim in this document would have been visible in a diff
   between two report headers (DEBUGGING.md #11, PROMPT_LOG Correction 9).
3. **Push the reranker further.** It lifted K=25 recall from 63% to 68% and
   K=40 from 74% to 79%, but it is an off-the-shelf MS-MARCO model applied to a
   task it was not trained for. Fine-tuning it on facet/utterance pairs, or
   raising the operating point to K=40 where it gains most, are both untested.
4. **Second-model agreement** as a confidence signal, since self-reported
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
reviewed and adopted by me.** I am not presenting it as independently
hand-authored, because it was not.

**Coverage:** 55 labelled pairs over 13 conversations and 30 facets - above the
brief's minimum of 10 and 20 - spanning all seven required categories plus three
hallucination traps, with 15 observable and 15 non-observable facets.

**The judgement calls I am standing behind**, each recorded in the `rationale`
field of every affected label:

| case | convention adopted | the defensible alternative |
|---|---|---|
| contradiction (`c03`) | described behaviour outweighs stated preference | the contradiction makes evidence uninterpretable; abstain |
| quoted praise (`c04`) | attributed to the speaker who said it, not its subject | third-party observation is still evidence, worth more than a 2 |
| hedged self-report (`c02`) | scores 2 | no behaviour described, so abstain |
| sarcasm (`c05`) | abstain throughout | irony still evidences self-awareness |

The third convention is worth singling out: **all 8 false abstentions in the
benchmark trace to it.** The system consistently judges that a hedged
self-assertion with no described behaviour warrants abstention, and my labels
say it warrants a 2. That disagreement is reported as a disagreement rather than
resolved in the system's favour, and a reviewer who sides with the system would
read those 8 cases as the system being right and the labels being generous.

`REVIEW_LABELS.md` renders all 55 labels grouped by conversation, contested
cases first, for anyone who wants to audit them.

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
tests/               57 tests, no model or network required
artifacts/           audit report, near-duplicates, benchmark report,
                     ablation, LLM cache
```

`DECISIONS.md` - 9 non-trivial decisions with trade-offs.
`DEBUGGING.md` - 7 real issues, each with symptom, root cause, fix, verification.
`PROMPT_LOG.md` - how AI was used, and four concrete things it got wrong.
