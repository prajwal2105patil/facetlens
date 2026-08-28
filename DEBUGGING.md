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

---

## #7. The spot-check I had promised but not yet run found a 13% error rate

**Symptom.** No symptom - that is the point. `audit_report.md` had a section
promising a seeded 30-row manual review, and its text said "reviewed by hand;
see README for the resulting disagreement count". The review had not actually
been performed. The claim was written before the work.

**Diagnosis.** Performed it properly: printed all 30 sampled rows (seed 42) and
judged each one's observability verdict by hand.

**Result: 4 of 30 wrong (13.3%), and every single one in the dangerous
direction** - classified observable when they are not, so all four would have
been sent to the LLM and scored:

| row | classified as | reality |
|---|---|---|
| `Sense-of-coherence score` | `personality_trait`, observable | Antonovsky SOC instrument output |
| `Psychological construct: Eye-Contact avoidance score` | `communication_style`, observable | instrument output - and eye contact is not present in *text* at all |
| `Honesty-humility trait score` | `motivation_value`, observable | HEXACO instrument output |
| `Ethical leadership rating` | `interpersonal`, observable | a rating, typically by third-party raters |

**Root cause.** One shared blind spot: the words `score`, `rating` and `scale`
were never treated as non-observability signals. Every one of these is an
*instrument output* - produced by administering a questionnaire or by external
raters - and no amount of conversation can produce one. The rules had been
written around subject matter (medical, biometric, demographic) and completely
missed the *measurement-artefact* category cutting across all of them.

**Fix.** Added the `psychometric_instrument_output` rule and a corresponding
`psychometric_instrument_score` type, placed after the cognitive-test rules so
the cognitive distinction survives.

**Verification.** All four now gate correctly; `Sportsmanship rating` was caught
as a bonus. Observable rows 245 -> 235. Legitimately observable facets
(`Happiness`, `Delegation skills`, `Collaboration`, `Irritability`,
`Orderliness`) were checked and are unaffected, so this is not over-correction.
No reference label broke, and all 38 tests pass.

**The caveat that matters.** Having driven the fix, this sample is no longer an
unbiased estimator - its post-fix error count is 0 by construction and quoting
that would be meaningless. The honest number is the pre-fix **4/30**, which
suggests a real observability error rate around 10%. A fresh seed is needed for
a post-fix estimate, and that is listed as remaining work rather than claimed.

**What this really says.** The most valuable audit finding in this project came
from doing a boring thing I had already written down as done. It is also a
reminder that the observability gate is only as good as the taxonomy behind it -
which is exactly why DECISIONS.md D1 accepts false abstentions as the price of
never inventing a score.

---

## #8. The evidence verifier called a verbatim quote a fabrication

**Symptom.** Adding the policy gate changed two benchmark verdicts that had
nothing to do with policy. On the code-switched conversation, `Collaboration`
and `Cooperation` went from `scored 4` to `insufficient_evidence`. Neither is a
special-category facet, so the policy gate should not have touched them.

**First hypothesis, and why it was wrong.** The obvious explanation was batch
composition: removing facets changes which facets share a batch, which changes
the prompt, which changes the cache key and forces a fresh call. Plausible, and
it would have been a genuine finding about batch-context contamination. It was
also wrong, and asserting it without checking would have put a fabricated
explanation into the failure analysis.

**Diagnosis.** Read the actual verdicts instead. Both carried
`evidence_verified: false` and had been downgraded by the *evidence verifier*,
not by batching. Diffing the quote against the conversation, normalised, gave
the exact divergence point:

```
quote : 'team meeting mein maine sabka opinion suna, phir hum sab ne milkar decide kiya.'
conv  : 'team meeting mein maine sabka opinion suna, phir hum sab ne milkar decide kiya ki naya architecture use ...'
first divergence at index 78: quote='.'  conv=' '
```

**Root cause.** The quote is verbatim for all 78 characters. The model stopped
mid-sentence and closed with a full stop the source does not contain. My strict
substring match treated one added character as evidence of fabrication, and the
verifier did what it was built to do: refuse the score.

The failure mode is *over*-rejection - the safe direction - which is precisely
why it went unnoticed. It produced more abstentions, and abstentions look like
caution rather than a bug.

**Fix.** Strip edge punctuation from the quote before matching. This loosens
punctuation only, never words, so a genuine fabrication is still rejected.

**Verification.** Measured on the full benchmark with the policy gate active:

| metric | before | after |
|---|---|---|
| status agreement | 81.8% | **85.5%** |
| exact score agreement | 50.0% | **56.2%** |
| false abstentions | 5 | **3** |
| quotes flagged as fabricated | 4 | **2** |

The two remaining flags are genuine fabrications. No prompt changed, so the LLM
cache stayed valid and the re-run cost nothing. A regression test pins the
truncation case.

**Two things worth taking from this.** First, a safety check that fails closed
is still a bug, and it is harder to spot than one that fails open, because its
symptom is behaviour you were hoping for. Second, the bug landed hardest on the
non-English conversation. That is not because the verifier is language-aware -
it is not - but it is a reminder that a brittle string comparison will
concentrate its damage wherever the text is least like what you tested on.
