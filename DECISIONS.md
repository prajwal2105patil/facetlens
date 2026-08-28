# Engineering decisions

Decisions the brief did *not* force. Each records the ambiguity, the options
considered, the choice, and what it costs.

---

## D1. Abstention is a property of the catalogue, not a model judgement

**Problem.** The brief requires abstaining on facets a conversation cannot
evidence. The obvious implementation is to send every retrieved facet to the
model and instruct it to abstain when appropriate. But then the entire
anti-hallucination guarantee rests on a 7B model reliably declining to answer -
exactly the behaviour small models are worst at.

**Options.**
1. Prompt-only abstention. Simple, one code path, no catalogue work.
2. Deterministic pre-LLM gate driven by the enriched catalogue.
3. Post-hoc filter: score everything, discard non-observable results afterwards.

**Choice: (2).** `conversation_observable` is computed during preprocessing.
Facets that fail it never reach the model at all; they return `not_observable`
with the catalogue's `abstention_reason`. Option 3 was rejected because it still
pays the tokens and still lets a model-generated score exist in the system, even
briefly.

**Trade-off.** The guarantee moves from "the model behaved" to "the taxonomy is
correct" - a much better place for it, because the taxonomy is inspectable,
testable and version-controlled. But it means a *misclassification* now causes a
**false abstention**. That is the price, it is measured explicitly in the
benchmark, and it is the safer direction to be wrong in: a false abstention
costs coverage, a missed abstention invents a fact.

---

## D2. Retrieval text is enriched, not the bare facet name

**Problem.** Facet names are short and abstract ("Naivety", "FSH level"). It was
not obvious that embedding them directly would work.

**Measured, before committing to it.** Against *"I led a team of five engineers
and assigned tasks based on their strengths."*:

| facet | bare name | name + type + definition |
|---|---:|---:|
| Delegation skills | 0.306 | 0.280 |
| Collaboration | 0.270 | 0.263 |
| **Nationality** | **0.134** | 0.043 |
| Assertiveness and control in relationships | 0.110 | 0.090 |
| **FSH level** | **0.110** | **-0.005** |

With bare names, `Nationality` outranks every genuine leadership facet and
`FSH level` ties `Assertiveness`. Both are noise.

**Options.** (a) bare names; (b) name + type + generated scoring definition;
(c) a larger embedding model.

**Choice: (b).** It reuses text the enrichment step already generates, costs
nothing at run time, and pushed the two irrelevant facets from ranks 4-5 down to
6-8.

**Trade-off.** Longer strings embed slightly slower (irrelevant at 399 rows) and
the definition text now influences retrieval, so a bad definition degrades
retrieval as well as scoring. (c) remains the biggest available improvement -
see the limitations in README.

---

## D3. Exact cosine search, no ANN index

**Problem.** The architecture must scale to 5,000+ facets. The reflex is to add
faiss.

**Numbers.** 399 facets x 384 dims is 0.6 MB. At 5,000 facets it is 7.7 MB and a
query is a single 5000x384 matrix-vector product - well under a millisecond.
Measured: encoding all 399 facets takes **0.13s**; a query embed takes
**0.016s**. One LLM batch takes **~50s**.

**Choice: numpy dot product over unit-norm rows**, with `argpartition` so only
the top-K is sorted.

**Trade-off.** Brute force is O(n) per query, so this stops being the right
answer somewhere around 10^5-10^6 facets. Adding faiss now would introduce a
dependency and an index-freshness bug surface to optimise something four orders
of magnitude away from being the bottleneck. The bottleneck is LLM inference,
by a factor of roughly 3,000x.

---

## D4. Schema-constrained decoding, with every field required

**Problem.** How hard to lean on the model producing valid JSON.

**Choice.** The Pydantic JSON schema is passed to Ollama as `format`, so it
constrains generation rather than merely requesting JSON politely. Parsing stays
defensive anyway (fences, balanced-brace scan, per-item isolation).

**The subtlety that cost real time.** Constrained decoding honours the schema
*exactly*, including optionality you did not mean to declare. Pydantic omits
defaulted fields from `required`, so `score: int | None = None` made `score`
optional and the model legally emitted `status="scored"` with no score at all -
turning every correctly-scored facet into a parse error. `ModelVerdict` now
declares every field required; `score` stays nullable but must be present. See
DEBUGGING.md #4.

**Trade-off.** The output contract is now coupled to the decoding grammar, so
changing the schema changes model behaviour. That is why the cache key hashes
the schema *content* rather than merely whether one was supplied.

---

## D5. Batch size 5

**Problem.** Larger batches amortise the ~700-token prompt across more facets;
smaller batches keep structured output reliable on a 7B model.

**Numbers.** At 8.2 tok/s generation and 33 tok/s prompt evaluation, a batch
costs roughly 21s of prompt evaluation plus ~30s of generation. Prompt
evaluation is therefore ~40% of the cost and is paid once per batch regardless
of batch size.

**Choice: 5**, exposed as `--batch-size`. Big enough to amortise the prompt,
small enough that a single malformed response damages at most five facets and
that output stays within reliable structured-generation length for this model.

**Trade-off.** Not empirically tuned. Larger batches would cut wall-clock
further; the risk is degraded JSON, which is precisely the failure mode the
error-isolation machinery exists to contain. Tuning this is a listed next step
rather than a claim.

---

## D6. A programmatic evidence check, not a self-reported one

**Problem.** Even on genuinely observable facets, the model can assert evidence
that is not there. Asking it how confident it is does not help - a hallucinating
model is confidently wrong.

**Choice.** The model must return `evidence_quote` copied verbatim. After
parsing, the quote is normalised (case, unicode, whitespace) and checked for
presence in the conversation. A quote that is absent means the citation was
fabricated, and the verdict is downgraded to `insufficient_evidence` with the
original reason preserved.

**Trade-off.** Substring matching is crude: a model that paraphrases loses a
legitimate score (a false abstention), and a model that quotes something real
but irrelevant still passes. Quotes shorter than 12 characters are not checked
at all, because short strings match by accident. It is a cheap, deterministic
lower bound on evidence integrity, not a proof of relevance.

---

## D7. Repairs may only move toward abstention

**Problem.** The model sometimes emits internally impossible verdicts -
`status="scored"` with `score=null` and a reason plainly describing an absence
of evidence. Rejecting these outright inflates the error count for a case whose
intent is unambiguous. Repairing them freely risks a parser that invents scores.

**Choice.** Exactly one repair rule, and it is safety-preserving by
construction: `scored` + `score=null` becomes `insufficient_evidence`. It can
only ever turn a would-be score into an abstention, never the reverse. Every
repair sets `schema_repaired=True` and is counted in the report.

**Trade-off.** It slightly flatters the error rate compared to strict rejection.
Reporting the repair count separately is what keeps that honest.

---

## D8. Force-include reference facets, but measure recall before doing it

**Problem.** If the benchmark only evaluates facets retrieval happened to
surface, weak retrieval silently improves the agreement numbers - the facets the
system failed to find simply never get graded.

**Choice.** Two separate measurements. Retrieval recall is computed against the
reference set *before* anything is added. Then every labelled facet retrieval
missed is force-injected (still passing through the observability gate) so
agreement covers all 55 pairs.

**Trade-off.** Reported agreement is therefore *not* the end-to-end number a
user would experience - it is scoring quality given the right facets, reported
alongside a retrieval recall figure that is currently poor. Two honest numbers
beat one flattering one.

---

## D9. The LLM cache is committed to the repository

**Problem.** The benchmark takes ~30 minutes on CPU. A reviewer will not wait,
and may not have the model at all.

**Choice.** Responses are cached under `artifacts/llm_cache/`, keyed by a hash
of model, options, schema and prompt - and the cache is tracked in git. A fresh
clone reproduces `benchmark_report.md` exactly, offline. Deleting the directory
regenerates it live.

**Trade-off.** It adds generated data to the repository and could mask a broken
live path, which is why `--backend mock` and the test suite exercise the code
independently of the cache, and why the cache key includes the schema content
so a contract change cannot silently reuse stale responses.
