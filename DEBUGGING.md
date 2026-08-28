# Debugging log

Real issues hit while building this, in the order they were found. Each was
exposed by measurement or by testing, not by reading the code and imagining
what might go wrong.

---

## #1. The installed model silently violated the <=16B constraint

**Symptom.** Ollama was installed with one model already pulled, `qwen3.6:latest`.
The tag says nothing about size, and it would have been easy to just use it.

**Diagnosis.** Read the manifest config blob rather than trusting the tag, then
confirmed against the running daemon:

```
$ curl -s http://127.0.0.1:11434/api/tags
{"name":"qwen3.6:latest","details":{"family":"qwen35moe",
 "parameter_size":"36.0B","quantization_level":"Q4_K_M"},"size":23938333577}
```

**Root cause.** Two independent disqualifications: **36.0B** parameters against a
hard <=16B limit, and a 23.9 GB weight file on a machine with 15.3 GB of RAM, so
it would have thrashed swap even if it had been permitted.

**Fix.** Pulled `qwen2.5:7b-instruct` and verified the replacement the same way
instead of trusting the model card:

```
general.parameter_count : 7615616512      # 7.6B, under the limit
license (head)          : Apache License Version 2.0
```

**Verification.** Both facts come from `/api/show` on the local daemon and are
quoted in README. The constraint is checked against the artefact actually being
run, not against documentation.

---

## #2. Embedding index build took 410 seconds instead of ~25

**Symptom.** Building the 399-facet index took **410.0s**. An earlier standalone
experiment had loaded the same model in 24s. Nothing about the data had changed.

**Diagnosis.** The log showed the process reaching for the network:

```
UserWarning: You are sending unauthenticated requests to the HF Hub.
```

The weights were already in the local cache, so this was Hub *resolution*, not
downloading. Unauthenticated requests are rate-limited, and the retries were the
entire cost.

**Root cause.** Two compounding mistakes. `SentenceTransformer(model_name)` with
a repo id resolves the id against the Hub even when weights are cached. And
setting `HF_HUB_OFFLINE=1` inside the loader function was too late -
`huggingface_hub` freezes that flag into a module constant when it is imported,
which had already happened.

**Fix.** Resolve the cached snapshot to a concrete filesystem path and hand
*that* to the loader, so no Hub resolution occurs:

```python
local_path = snapshot_download(model_name, local_files_only=True)
return SentenceTransformer(local_path)
```

Also added `functools.lru_cache` - the model was being reloaded on every query
embed, at ~25s each.

**Verification.** Forced rebuild after the fix: **48.2s**, of which ~45s is the
one-time model load. A subsequent query embed is **0.016s**. 410s -> 48s.

---

## #3. Keyword matching was blind to plurals

**Symptom.** Not a crash - found by auditing the `classification_rule` column
rather than the classifications themselves. 107 of 399 rows (26.8%) had fallen
through every rule into `fallback_bare_trait_noun`. Scanning them,
`Assertiveness and control in relationships` stood out: the `interpersonal` rule
explicitly lists `relationship`.

**Diagnosis.** `\brelationship\b` does not match `relationships`. The `\b` after
the keyword requires a word boundary, and the trailing `s` is a word character,
so there is none.

**Root cause.** Hand-written keyword lists were singular; the catalogue is not.

**Fix.** `_kw()` now builds `\b(?:word1|word2)(?:e?s)?\b`.

**Verification.** That row now classifies as `interpersonal`; fallback count
107 -> 106. The immediate impact was one row and observability did not change
(both types are observable), so this is reported as a *latent correctness bug*
rather than a large win - it would scale badly on a larger catalogue. A
regression test pins it.

**What this really validates** is recording which named rule fired for every
row. The classifications looked fine; the rule attribution is what exposed the
bug.

---

## #4. Constrained decoding honoured optionality I did not mean to declare

**Symptom.** The first live run produced 4 parser errors out of 10 facets, and
every other verdict was `insufficient_evidence`. It looked like the model simply
could not score anything.

**Diagnosis.** Inspected the cached raw response instead of guessing. The model
was scoring correctly:

```json
{ "facet_id": "F0331", "status": "scored", "confidence": 0.8,
  "reason": "The speaker demonstrates the ability to lead and assign tasks...",
  "evidence_quote": "I led a team of five engineers and assigned tasks..." }
```

There is no `score` key. Validation rejected it, correctly - and threw away a
genuinely good verdict.

**Root cause.** The Pydantic schema is handed to Ollama as a decoding grammar.
`score: int | None = Field(default=None, ...)` gives the field a default, so
Pydantic omits it from `required`, so the grammar made it optional, so the model
skipped it. The model followed my schema exactly. The schema was wrong.

**Fix.** Every field in `ModelVerdict` is now required. `score` stays nullable
but must be present. Confirmed at the schema level:

```
required: ['facet_id', 'status', 'score', 'confidence', 'evidence_quote', 'reason']
```

**Verification.** Re-ran the same conversation: `Desire to influence others`
scored 4 with a quote that passes evidence verification, `Hardworking` scored 4,
`Need for Achievement level` scored 4. A test asserts `score` stays in the
schema's `required` list, so this cannot silently regress.

**Follow-on bug found while fixing it.** The cache key hashed only *whether* a
schema was passed, not its content - so changing the contract would have served
responses generated under the old grammar. Now the schema content is in the key.

---

## #5. Header detection swallowed a real facet

**Symptom.** Selecting benchmark facets, `Adventure-Seeking Behavior` came back
typed `instrument_or_scale_header` and gated non-observable. It is plainly a
facet.

**Diagnosis.** Header detection uses two signals: a trailing colon, or a
grouping noun at the end of the string. The grouping pattern was
`behaviors?` - the optional `s` matched the singular `Behavior`.

**Root cause.** A header *names a group*, so the grouping noun must be plural.
Allowing the singular made the rule match ordinary facet names ending in a
category word.

**Fix.** The grouping alternation now requires plural forms only
(`behaviors|behaviours|styles|types|facets|...`).

**Verification.** `Adventure-Seeking Behavior` -> `behavioral_tendency`,
observable. `Work Styles` (a genuine header with no colon) is still detected.
Header-like rows 35 -> 31; observable 241 -> 245. Both directions are pinned by
tests.

---

## #6. A misclassification that survived into routing

**Symptom.** Running the routing gate over the "so tired lately" conversation,
`Sleep-environment temperature` appeared in the **scorable** list, typed
`preference_lifestyle`.

**Diagnosis.** It matched the `preference_lifestyle` rule on the keyword `sleep`
before reaching any quantitative rule. But it is a measured environmental
quantity, not a stated preference - a conversation cannot establish it.

**Root cause.** The quantitative rules keyed on units and count nouns and had no
entry for bare physical measurements like `temperature`.

**Fix.** Added `temperature`, `intake` and `km` to the quantified-metric
patterns.

**Verification.** Now `quantified_activity_metric`, non-observable, gated before
the LLM. Observable count 242 -> 241.

**Why this one matters.** It is exactly the failure mode the architecture exists
to prevent, and it was caught by *looking at the routing output* rather than by
reading the ruleset. It is also honest evidence that the taxonomy is heuristic:
this one was found, and others are certainly still there. That is why the
benchmark measures false abstentions instead of assuming the gate is perfect.
