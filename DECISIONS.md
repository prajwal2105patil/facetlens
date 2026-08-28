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
testable and version-controlled. But it means a *misclassification* now becomes
the system's dominant error mode.

**And the taxonomy is measurably imperfect.** A seeded 30-row manual review
found **4 errors (13.3%)** - and, uncomfortably for this design, all four were
*false-observable*: rows wrongly sent onward to the LLM rather than wrongly
gated (DEBUGGING.md #7). They were fixed, but the honest reading is that the
gate leaks in the dangerous direction as well as the safe one, at something like
a 10% rate.

That does not invalidate the choice - a prompt-only gate would leak at least as
much, with no audit trail and no way to fix a class of errors in one line. But
it does mean the correct claim is "abstention is *auditable and fixable*", not
"abstention is guaranteed". The benchmark measures false abstentions rather than
assuming they are zero, and the spot-check measures the other direction.

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

**Then I tested it properly, and it did not hold up as well as claimed.**
`artifacts/ablation_retrieval.md` runs both variants across 6 values of K over
all 13 conversations and 55 labelled pairs. On recall of facets that *should* be
scored, enriched text wins at 2 values of K, **loses at 1** (K=40: 12/19 vs
14/19), and ties at 3.

| K | bare | enriched |
|---:|---|---|
| 10 | 8/19 | **9/19** |
| 25 | 11/19 | **12/19** |
| 40 | **14/19** | 12/19 |
| 60 | 15/19 | 15/19 |

**What the single example actually measured.** Enriching the text reliably
*demotes obviously irrelevant medical and demographic facets* - that part
reproduces. It is not the same thing as *surfacing more of the right facets*.
I conflated the two when making the decision.

**Kept anyway, but downgraded to weakly supported.** Demoting non-observable
facets out of the candidate set is independently useful - it reduces what Gate 2
has to catch - and it costs nothing at run time. But this is now recorded as a
decision the evidence does not strongly justify rather than a validated win.

**Trade-off.** The definition text now influences retrieval, so a bad definition
degrades retrieval as well as scoring.

**Update after D10.** Option (c), a stronger embedding model, was the obvious
next move from here and it was tested: BGE-small-en-v1.5 scored *worse* than
MiniLM (10/19 vs 12/19 at K=25). The real fix turned out to be changing the
indexed text rather than the encoder - see D11.

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

---

## D10. Rejecting three retrieval "improvements" on measurement

**Problem.** Retrieval was the weakest component: 63% recall on facets that
should be scored, at K=25. Three standard fixes suggested themselves.

**Measured, all three:**

| approach | recall@25 (should-score) |
|---|---|
| MiniLM dense, enriched text (incumbent) | **12/19 (63%)** |
| BM25 lexical, with suffix normalisation | 2/19 (11%) |
| dense + BM25, reciprocal rank fusion | 10/19 (53%) |
| BGE-small-en-v1.5 (stronger public benchmarks) | 10/19 (53%) |
| BGE-small + its recommended query prefix | 9/19 (47%) |

**Choice: ship none of them.** Every alternative was worse. BM25 is close to
useless here because conversations describe behaviour while facets are abstract
labels, so there is almost no lexical overlap to exploit. Fusion then drags a
strong signal down with a near-random one. And a "better" encoder was worse,
which is the result that mattered most - it ruled out the encoder as the
problem.

**Trade-off.** Three implementations of which only one shipped, and it shipped
disabled. The BM25 module stays in the tree as an ablation arm rather than
being deleted, because the negative result is the evidence for D11.

**What it bought.** Eliminating the encoder and the similarity function as
causes is what identified the real one: an abstract label and a concrete
narrative do not occupy the same region of *any* embedding space. That is a
property of the task, not of the tooling, and it needs a different kind of fix.

**That fix is D12.** These four failures are the reason it was findable.

---

## D11. Document expansion, generated once and committed as data

**Problem.** Given D10, the fix has to change what is *indexed*, not how it is
compared.

**Options.** (a) hand-write example phrasings for 399 facets; (b) generate them
with the LLM once and cache; (c) query-side expansion (HyDE) at run time;
(d) accept the recall ceiling.

**Choice: (b).** Each facet is indexed as its enriched text plus two generated
first-person utterances that would evidence it. Validated by hand before
building: appending examples to three facets moved their rank against a
leadership conversation from 7->2, 7->2 and 9->3.

(c) was rejected because it adds an LLM call to every query - on a machine
generating 8 tokens/second, that is the whole latency budget - while (b) is
paid once, offline, and cached.

**Guarding against leakage.** The generator sees a facet's name, type and
definition and nothing else. It never sees a benchmark conversation or a
reference label. Expansions were generated for **all 399 facets**, not only the
observable ones, because expanding only the observable half would confound the
result: recall could improve merely by demoting non-observable facets rather
than by surfacing the right ones.

**Trade-off.** A one-time generation cost (~45 minutes on this CPU) and a new
failure surface: a bad expansion actively harms retrieval for that facet, and
nothing validates the generated utterances beyond their being cached as
inspectable, committed data. At 5,000 facets this cost scales linearly and
would want batching on a GPU or a hosted endpoint - but it remains a one-off,
not a per-query cost.

---

## D12. A cross-encoder reranker: built, measured, and NOT made the default

**Problem.** After D10 and D11, retrieval was still the weakest measured
component (63% should-score recall at K=25) and four interventions had failed.
What survived them was a structural fact: recall is **89% at K=100** and 63% at
K=25. The facets were being retrieved. They were being ranked badly.

**Choice.** Retrieve a wide pool (100) by cosine, then reorder it with
`ms-marco-MiniLM-L-6-v2` (22.7M params, Apache-2.0). A bi-encoder embeds each
side independently, so an abstract label and a concrete narrative are compared
only after both are compressed; a cross-encoder attends to them jointly, which
is exactly the comparison the four failures showed was missing.

**It worked on the metric it targeted.** Should-score recall, best or tied-best
at every K from 15 up:

| K | dense | +rerank |
|---:|---|---|
| 25 | 12/19 (63%) | **13/19 (68%)** |
| 40 | 13/19 (68%) | **15/19 (79%)** |
| 60 | 15/19 (79%) | **16/19 (84%)** |

Cost: 8.3s for all 13 conversations, ~1% of one LLM scoring batch.

**And it made the end-to-end result worse.**

| | dense (shipped) | +rerank |
|---|---:|---:|
| retrieval recall@25 | 34.5% | **40.0%** |
| **status agreement** | **87.3%** | 81.8% |
| correct abstentions | **91.7%** | 88.9% |
| false abstentions | **4** | 6 |

**Why - and this is a flaw in my reasoning when I proposed it.** Labelled facets
are **force-included regardless of retrieval** (D8), deliberately, so retrieval
misses cannot hide inside the agreement number. But that means improving
retrieval *cannot* improve agreement on labelled pairs: they were already being
scored. I proposed a fix for a metric that, by my own evaluation design, could
not move the metric I cared about.

What did change is batch composition. Better ranking surfaced more *observable*
facets, so the observability gate handled only 66 verdicts (18%) instead of 126
(35%), more facets reached the LLM, and verdicts shifted with the altered
context. That is perturbation, not improvement.

**Decision: `dense` remains the default; `rerank` ships as a measured option.**
Defaulting to a configuration with worse end-to-end numbers in order to chase a
secondary metric would repeat exactly the mistake that caused DEBUGGING #11 -
an unjustified default contradicted by an available measurement.

**Trade-off, stated plainly.** The retrieval number in the README is therefore
*not* the best this repository can produce. 40.0% is available and 34.5% ships.
That is the honest cost of preferring the metric that reflects end-to-end
quality over the one that reflects a component in isolation.

**On sample size.** 87.3% vs 81.8% is a 3-pair difference on 55 pairs, so some
of this gap is noise. The direction is consistent across status agreement,
correct abstentions and false abstentions, which is why it decided the default -
but a larger reference set could overturn it, and that is listed as next work.
