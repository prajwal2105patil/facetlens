# Prompt log

## How AI was used on this assignment

**Tool:** Claude Code (Claude Opus 5), used as a pair-programming agent with
filesystem and shell access, in a single working session on 2026-08-28.

**Honest summary of the split.** The AI wrote the majority of the code text.
What made the system correct was not the generation - it was the loop of
*running things and reading the actual output*: inspecting the real CSV before
designing the taxonomy, reading the Ollama manifest instead of trusting a model
tag, timing the index build, and opening cached raw model responses when
verdicts looked wrong. Every issue in DEBUGGING.md was found that way. Several
of the AI's first proposals were measurably wrong and are recorded below.

This log describes what actually happened. No interactions are invented, and
nothing that failed has been quietly dropped.

---

## Material prompts, in order

### 1. Establish the source of truth
> *"Plan out everything... make sure each and every possible outcome stated in
> the document is completely taken into account."* (with the assignment `.docx`
> and `Facets Assignment.csv` attached)

**Used:** the assignment text was extracted from the `.docx` directly rather
than working from a summary, and treated as authoritative wherever the
accompanying `CLAUDE.md` expanded on it.

**Verified:** read the raw CSV before any design work - 399 rows, one column,
30 colon-terminated rows, 31 numeric-ID prefixes, 5 non-ASCII rows, **zero**
exact or normalised duplicates.

**Why that mattered:** the instinct was to write a duplicate-detection section
into the audit. The data does not support one. The audit now reports the null
result explicitly, because inventing a finding would have been worse than
having none.

---

### 2. Environment and feasibility check *(before writing code)*
> Probe: Python, git, GPU, Ollama, installed models, API keys, RAM.

**Result:** no NVIDIA GPU (`torch.cuda.is_available()` -> `False`, Radeon
integrated only), 15.3 GB RAM, and the only Ollama model present was
**Qwen3.5-MoE 36B**.

**Changed as a result:** the whole inference plan. See DEBUGGING.md #1 - that
model breaks the <=16B rule *and* exceeds available RAM. Replaced with
`qwen2.5:7b-instruct` (7.6B, Apache-2.0), both facts read from the local
daemon's `/api/show` rather than from a model card.

---

### 3. "Design the retrieval layer"

**AI's first proposal:** embed the bare facet names and retrieve top-K.

**Rejected after measuring.** Against *"I led a team of five engineers..."*:

```
cos(FSH level, conversation)     = 0.110
cos(Assertiveness..., conversation) = 0.110
cos(Nationality, conversation)   = 0.134
```

A blood-test facet tied a leadership facet, and a demographic facet beat both.

**Changed to:** embedding `name + facet_type + scoring_definition`, which drops
`FSH level` to **-0.005** and `Nationality` to 0.043. This became DECISIONS.md
D2 and DEBUGGING.md #1.

**Also rejected:** an early suggestion to add **faiss**. At 399 facets - and at
the 5,000 the brief asks about - a numpy dot product is sub-millisecond while
one LLM batch takes ~50s. The dependency would have optimised something ~3,000x
away from the bottleneck. See DECISIONS.md D3.

---

### 4. "Write the scoring prompt"

**AI's first draft** was long, and included instructions telling the model not
to infer lab values, diagnoses or biometric facts.

**Cut, deliberately.** Facets that require lab values never reach the model -
the observability gate removes them first. Those instructions were defending
against a case that cannot occur, and a longer prompt costs prompt-evaluation
time on every batch (~40% of per-batch cost at 33 tok/s). The prompt now only
enforces what the model alone can get wrong: sarcasm, quoted speech,
contradiction, code-switching, and treating topical similarity as evidence.

---

### 5. "Make output parsing robust"

**Used:** the layered recovery strategy (direct parse -> fenced block ->
balanced-brace scan -> per-item validation) largely as proposed.

**Added beyond the proposal:** facets the model *omits* from a batch response
become explicit `error` verdicts. The first version silently returned fewer
verdicts than requested, which reads as a clean run in the report while facets
quietly vanish. `MockBackend(mode="partial")` exists specifically to test this.

---

## What AI got wrong, and what I corrected

### Correction 1 - the schema silently made `score` optional

The AI wrote `score: int | None = Field(default=None, ge=1, le=5)`, which is
idiomatic Pydantic and looks completely fine.

That schema is also handed to Ollama as a **constrained-decoding grammar**.
Pydantic omits defaulted fields from `required`, so the grammar made `score`
optional, so the model emitted `status="scored"` with no score at all. Four of
ten facets in the first live run became parse errors - and the model had
actually scored them correctly.

Diagnosed by opening the cached raw response rather than trusting the error
message. Fixed by making every field required; `score` stays nullable but must
be present. A test now asserts `score` remains in the schema's `required` list.

While fixing it: the cache key hashed only *whether* a schema was passed, not
its content - so the contract change would have silently reused responses
generated under the old grammar. Fixed too. Full write-up: DEBUGGING.md #4.

### Correction 2 - "just set HF_HUB_OFFLINE" did not work

The index build took **410s**. The AI's fix was to set `HF_HUB_OFFLINE=1` inside
the model loader. That does nothing: `huggingface_hub` freezes the flag into a
module constant at import time, which has already happened by then. The build
stayed slow.

The actual fix was to stop handing the library a repo id at all - resolve the
cached snapshot to a filesystem path and load from there, so no Hub resolution
is attempted. Plus an `lru_cache`, because the encoder was being reloaded on
every single query embed at ~25s each. **410s -> 48s**, measured both ways.
Full write-up: DEBUGGING.md #2.

### Correction 3 - the taxonomy looked right and was not

The generated keyword rules were singular (`relationship`, `behavior`) while the
catalogue is not. Two consequences, neither of which produced an error message:

- `\brelationship\b` never matched `...in relationships`, so that row fell
  through to the catch-all bucket (DEBUGGING.md #3).
- `behaviors?` matched the singular `Behavior`, so the real facet
  `Adventure-Seeking Behavior` was classified as a *section header* and gated
  out as non-observable (DEBUGGING.md #5).

Both were found by auditing the `classification_rule` column - which exists
because rule attribution was designed in - not by reading the rules. The
classifications themselves looked plausible in isolation.

### Correction 4 - I over-trusted my own single measurement

The enriched-retrieval-text decision (Correction 3's sibling, DECISIONS.md D2)
was made on **one** conversation where it dropped `FSH level` from 0.110 to
-0.005. That looked conclusive, and the AI and I both treated it as settled.

Running the proper ablation across all 13 conversations and 6 values of K
(`artifacts/ablation_retrieval.md`) showed it wins 2 times, **loses once**, and
ties 3 times on should-score recall. The single example measured *demotion of
irrelevant facets*, which is real; I had generalised it to *better retrieval of
relevant facets*, which it does not support.

The decision is kept - demoting non-observable facets is independently useful
and free - but DECISIONS.md D2 now records it as weakly supported rather than
validated. Worth noting the ablation was originally a "brownie point" item; it
ended up correcting a decision I had already written up as settled.

### Correction 5 - measuring one number instead of two

The AI's benchmark design scored only the facets retrieval surfaced. That
quietly flatters the result: facets the retriever misses never get graded, so
weak retrieval improves the agreement score. Changed to measure retrieval recall
*first*, then force-include every labelled facet so agreement covers the full
reference set. This is what exposed that retrieval recall on should-score facets
is only 12/19 at K=25 - the system's largest real weakness, which the original
design would have hidden. See DECISIONS.md D8.

---

### Correction 6 - the scoring scale had two right answers

The AI proposed the anchor ladder and I accepted it: `1 = no or very weak
evidence` through `5 = very strong`. It reads fine. It is also broken, and the
benchmark proved it.

On *"Things are okay. Not much to report this week."* the system scored
`Enthusiasm` **1**, with the reason *"The statement is neutral and does not
express enthusiasm."* The model reasoned perfectly and still failed the case,
because "no evidence" is simultaneously the definition of anchor 1 **and** the
definition of `insufficient_evidence`. Two correct answers were on the table and
nothing in the design said which to prefer.

That is a specification bug, not a model bug, and no amount of prompt tuning
fixes it. The correct anchor 1 is *"the facet is clearly present but minimally
expressed"*, leaving absence entirely to abstention.

**Not applied.** Changing anchors changes every prompt, invalidating the LLM
cache and costing a ~40 minute re-run I could not then validate. Recorded in
README as the first thing to fix, with the specific replacement text. The brief
permits a justified decision not to fix under time pressure; this is that
justification, made explicitly rather than by omission.

### Correction 7 - I nearly wrote a plausible wrong root cause

Two verdicts flipped when the policy gate landed, on facets the policy gate does
not touch. My first explanation - batch composition changed, so the prompt
changed, so the verdict changed - was coherent, mechanically plausible, and
would have made a respectable finding about batch-context contamination.

It was also wrong. Reading the actual verdict records showed
`evidence_verified: false`: the *evidence verifier* had downgraded them, and
diffing the quote against the conversation gave the exact divergence at
character 78 - a full stop the model added when it truncated an otherwise
verbatim quote (DEBUGGING.md #8).

Worth logging because the failure mode is subtle: a confident, plausible causal
story is exactly what an LLM produces on demand, and the only defence is to go
and read the data. Had I written it up without checking, the failure analysis
would have contained a fabricated explanation - in the document whose entire
purpose is honest failure analysis.

### Correction 8 - I made Correction 4's mistake again, on a bigger scale

Correction 4 records that I generalised a single cosine measurement into a
design decision that the full ablation later contradicted. I wrote that up,
noted the lesson, and then did the same thing again about forty minutes later.

Retrieval recall was the system's weakest number, so I hand-tested document
expansion: appending example utterances to three facets moved their rank
against a leadership conversation from 7->2, 7->2 and 9->3. That looked
decisive. I built the generator, spent ~45 minutes of compute generating
expansions for all 399 facets, and wired it into the index.

Measured result at K=25: **12/19, identical to the incumbent.** Ties at four
values of K, one extra facet at two others.

The flaw was the same shape as before and I still did not see it: I wrote those
three example utterances *after reading the conversation they were meant to
match*. The generated ones are written blind from a facet definition, which is
the only honest setup and a far harder one. The hand test measured the ceiling
of the idea, not the method as it would actually be deployed.

**What I would do differently:** the validation should have been run the way
the system runs - generate expansions for a handful of facets from definitions
alone, then measure - which would have cost five minutes and saved forty-five.
A hand-built proof-of-concept that has seen the answer is not evidence about a
blind method.

Kept in the submission because it never loses (it ties or marginally wins at
every K), and because four measured failures in a row are what localised the
real bottleneck: a bi-encoder cannot bridge an abstract label to a concrete
narrative, and the fix is a reranker, not better inputs to the same comparison.

## Reference labels - provenance

`data/benchmark/reference_labels.jsonl` was **drafted with AI assistance** and
requires the candidate's own review before submission. Each of the 55 labels
carries a written rationale, and the contested categories follow stated rules:
contradiction resolves toward described behaviour over stated preference;
quoted praise is attributed to whoever said it; sarcasm is not read literally;
code-switching neither reduces evidential weight nor implies a nationality.

Flagging this rather than presenting the labels as independently human-authored
is the honest position, and the rationales are written so each label can be
argued individually.
