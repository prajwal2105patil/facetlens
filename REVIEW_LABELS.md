# Reference label review sheet

**This is the one task only you can do.** The brief calls this a *human-reviewed*
reference set, and it is the input to the 25%-weighted evaluation section. The
labels below were AI-drafted. Read them, change any you disagree with, and edit
`data/benchmark/reference_labels.jsonl` directly.

55 labels across 13 conversations. 19 expect a score, 36 expect abstention.

## How to review quickly

1. Read the conversation.
2. For each facet ask only: **could someone reasonably argue the opposite?**
3. If yes, that label is a liability in interview - change it or sharpen the rationale.

Sections marked **CONTESTED** are the ones a sharp interviewer will probe first,
because they encode a judgement call rather than a fact. Start there.

---

## `c02_ambiguous` (ambiguous) - **CONTESTED**

> I guess I'm pretty good at working with people. Mostly it's fine, I don't really think about it much.

| facet | observable? | expected | rationale |
|---|:--:|---|---|
| Delegation skills | yes | `insufficient_evidence` | Nothing about assigning work to others. |
| Nationality | NO | `not_observable` | Biographical fact, never inferable from a snippet. |
| Cooperation | yes | **scored 2** | Hedged self-report ('I guess I'm pretty good') with no behaviour described. Weak evidence only. |
| Collaboration | yes | **scored 2** | Same hedged claim. Deliberately a 2, not a 4 - self-assertion without instance. |

## `c03_contradictory` (contradictory) - **CONTESTED**

> Honestly I dislike teamwork and I avoid group projects whenever I can. That said, I ran our product team for six months and I enjoyed most of it.

| facet | observable? | expected | rationale |
|---|:--:|---|---|
| Withdrawnness | yes | `insufficient_evidence` | Avoiding group projects hints at it, but leading a team for six months directly contradicts it. Net evidence is not interpretable. |
| Sleep Apnea | NO | `not_observable` | Medical diagnosis. |
| Collaboration | yes | **scored 3** | CONTRADICTION RESOLVED toward described behaviour: abstract 'I dislike teamwork' is outweighed by six months actually running a team and enjoying it. Scored moderate, not high, because the contradiction is unresolved by the speaker. |
| Cooperation | yes | **scored 3** | Same resolution rule: concrete sustained behaviour beats stated preference. |

## `c04_quoted` (quoted) - **CONTESTED**

> My manager said, "You're the most assertive person on this team." I don't really see it myself. In meetings I usually just go along with whatever has already been decided.

| facet | observable? | expected | rationale |
|---|:--:|---|---|
| Basophil count | NO | `not_observable` | Blood count. |
| Assertiveness and control in relationships | yes | **scored 2** | QUOTED SPEECH: 'most assertive person on this team' is the MANAGER's claim, not the speaker's. The speaker's own reported behaviour - 'I usually just go along' - points the other way. Weak evidence at most. |
| Withdrawnness | yes | **scored 2** | 'go along with whatever has already been decided' is weak evidence of deferring rather than withdrawing. |

## `c05_sarcastic` (sarcastic) - **CONTESTED**

> Oh absolutely, I'm the world's greatest communicator. I once explained a deploy plan so clearly that three people shipped straight to the wrong environment.

| facet | observable? | expected | rationale |
|---|:--:|---|---|
| Talkativeness | yes | `insufficient_evidence` | SARCASM: 'world's greatest communicator' is ironic, immediately undercut by three people deploying to the wrong environment. Not evidence of talkativeness either way. |
| Brevity | yes | `insufficient_evidence` | The failed explanation says nothing about concision specifically. |
| Enthusiasm | yes | `insufficient_evidence` | The upbeat tone is sarcastic. Literal wording must not be read as genuine enthusiasm. |
| Parathyroid-hormone level | NO | `not_observable` | Lab value. |

## `h02_saving` (hallucination-trap) - **CONTESTED**

> I cut my spending a lot this year and I'm finally managing to put something aside every month instead of living paycheck to paycheck.

| facet | observable? | expected | rationale |
|---|:--:|---|---|
| Orderliness | yes | `insufficient_evidence` | Budgeting discipline is implied but no organising behaviour is described. |
| Subscription count | NO | `not_observable` | HALLUCINATION GUARD: 'cut my spending' must not become a count of cancelled subscriptions. |
| Nationality | NO | `not_observable` | 'paycheck' is a spelling convention, not a nationality. |
| Self-improvement | yes | **scored 2** | A deliberate behaviour change is described, but briefly and in one domain. Weak. |

## `c01_clear` (clear)

> I led a team of five engineers and assigned tasks based on their strengths. When two of them disagreed about the architecture, I set up a call and we worked through it together until we had something everyone could live with.

| facet | observable? | expected | rationale |
|---|:--:|---|---|
| Talkativeness | yes | `insufficient_evidence` | Setting up a call says nothing about how much the speaker talks. |
| FSH level | NO | `not_observable` | Lab value. No conversation can establish it. |
| Intelligence Quotient (IQ) | NO | `not_observable` | Standardised test score. Competent behaviour is not an IQ measurement. |
| Delegation skills | yes | **scored 4** | Explicit: 'assigned tasks based on their strengths'. Concrete delegating behaviour, elaborated. |
| Collaboration | yes | **scored 4** | 'we worked through it together until we had something everyone could live with' - joint resolution described in detail. |
| Assertiveness and control in relationships | yes | **scored 3** | Led the team and convened the call. Directive but not forceful; moderate, explicit evidence. |

## `c06_codeswitched` (code-switched)

> Team meeting mein maine sabka opinion suna, phir hum sab ne milkar decide kiya ki naya architecture use karenge. By the end everyone was genuinely on board.

| facet | observable? | expected | rationale |
|---|:--:|---|---|
| Nationality | NO | `not_observable` | HALLUCINATION GUARD: Hindi-English code-switching must NOT be converted into an inferred nationality. |
| Collaboration | yes | **scored 4** | CODE-SWITCHED: 'sabka opinion suna' + 'hum sab ne milkar decide kiya' is strong collaborative evidence. Language mixing must not reduce evidential weight. |
| Cooperation | yes | **scored 4** | 'everyone was genuinely on board by the end' - joint buy-in achieved. |
| Decision-making decisiveness | yes | **scored 3** | A decision was reached and named, though collectively rather than unilaterally. |

## `c07_low_evidence` (low-evidence)

> Things are okay. Not much to report this week.

| facet | observable? | expected | rationale |
|---|:--:|---|---|
| Enthusiasm | yes | `insufficient_evidence` | 'Things are okay' carries no affective signal. |
| Collaboration | yes | `insufficient_evidence` | No other people mentioned at all. |
| Irritability | yes | `insufficient_evidence` | Neutral report. Absence of complaint is not evidence of calm. |
| Subscription count | NO | `not_observable` | Requires account records. |

## `c08_impulsive` (clear)

> I booked the flight about an hour after someone mentioned the idea. Didn't really think it through, I just wanted to go.

| facet | observable? | expected | rationale |
|---|:--:|---|---|
| Passport-stamps count | NO | `not_observable` | HALLUCINATION GUARD: a flight is mentioned but no count of anything. Must not be converted into a travel tally. |
| Impulsivity | yes | **scored 4** | 'booked the flight about an hour after' + 'didn't really think it through' - explicit, self-described impulsive act. |
| Adventure-Seeking Behavior | yes | **scored 3** | Spontaneous travel is moderate evidence of adventure-seeking. |
| Decision-making decisiveness | yes | **scored 3** | Decided fast. Decisiveness and impulsivity overlap here; moderate is the defensible level. |

## `c09_orderly` (clear)

> Every Sunday evening I lay out the whole week: colour-coded calendar, meals planned, gym slots blocked out. If it isn't on the list, it doesn't happen.

| facet | observable? | expected | rationale |
|---|:--:|---|---|
| Self-improvement | yes | `insufficient_evidence` | Planning is described; nothing states a goal of improving oneself. |
| Caffeine intake (mg/day) | NO | `not_observable` | Quantified intake requires a log. |
| Orderliness | yes | **scored 5** | Multiple concrete, repeated systems: colour-coded calendar, meal planning, blocked gym slots, plus an explicit rule. Very strong. |

## `c10_irritable` (clear)

> I've been snapping at people over nothing lately. Yesterday I completely lost it because the printer jammed twice in a row.

| facet | observable? | expected | rationale |
|---|:--:|---|---|
| Withdrawnness | yes | `insufficient_evidence` | Irritability is not withdrawal; nothing about avoiding others. |
| Compassion Fatigue | NO | `not_observable` | HALLUCINATION GUARD: clinically-flavoured construct. Snapping at people is not a burnout assessment. |
| Irritability | yes | **scored 4** | 'snapping at people over nothing' plus a specific instance. Strong, elaborated. |

## `h01_tired` (hallucination-trap)

> Honestly I've been so tired the last few weeks. Dragging myself out of bed every morning and I still feel wiped out by lunchtime.

| facet | observable? | expected | rationale |
|---|:--:|---|---|
| Withdrawnness | yes | `insufficient_evidence` | Exhaustion is described; social withdrawal is not. |
| Irritability | yes | `insufficient_evidence` | Plausible companion of fatigue, but never stated. Plausibility is not evidence. |
| FSH level | NO | `not_observable` | HALLUCINATION GUARD: fatigue must not become an endocrine value. |
| Basophil count | NO | `not_observable` | HALLUCINATION GUARD: fatigue must not become a blood count or an anaemia inference. |
| Sleep Apnea | NO | `not_observable` | HALLUCINATION GUARD: tiredness is a symptom, not a diagnosis. |
| Sleep-disorder diagnosis | NO | `not_observable` | HALLUCINATION GUARD: requires clinical assessment. |
| Compassion Fatigue | NO | `not_observable` | HALLUCINATION GUARD: lexical overlap with 'fatigue' is not evidence of the clinical construct. |

## `h03_practice` (hallucination-trap)

> I feel so calm and centred after my morning practice. It's easily the best part of my day and I'd be lost without it now.

| facet | observable? | expected | rationale |
|---|:--:|---|---|
| Pilgrimage participation count | NO | `not_observable` | HALLUCINATION GUARD: 'morning practice' names no tradition and no count. |
| 892. Hindu spiritual metric: Yoga discipli | NO | `not_observable` | HALLUCINATION GUARD: must infer neither the religion nor the weekly hours. The word 'practice' is doing all the work and it is not enough. |
| 793. Sufi practice: Dhikr repetitions / da | NO | `not_observable` | HALLUCINATION GUARD: same trap, different tradition. Retrieval will surface both; neither is supported. |
| Nationality | NO | `not_observable` | HALLUCINATION GUARD: a spiritual practice must not be converted into an ethnicity or nationality. |
| Enthusiasm | yes | **scored 3** | 'easily the best part of my day' and 'I'd be lost without it' is explicit positive affect about an activity. |

---

## The four judgement calls to check first

These are conventions I chose. They are defensible, but they are choices, and
the benchmark's disagreements trace back to them:

1. **Contradiction** (`c03`) - resolved toward *described behaviour* over *stated
   preference*. Someone who says they hate teamwork but ran a team for six months
   scores 3 for Collaboration, not 1. Alternative view: the contradiction makes
   the evidence uninterpretable and it should abstain.
2. **Quoted praise** (`c04`) - the manager's *"you're the most assertive person
   here"* is the manager's claim, not the speaker's. Scored 2, weighted toward the
   speaker's own reported behaviour (*"I usually just go along"*). Alternative:
   third-party observation is still evidence and deserves more than a 2.
3. **Hedged self-report** (`c02`) - *"I guess I'm pretty good with people"* scores 2.
   The system disagrees and abstains, and **all three false abstentions in the
   benchmark come from this one convention.** If you think abstention is right
   here, change these labels and the system's score improves - legitimately.
4. **Sarcasm** (`c05`) - all abstentions, on the grounds that ironic wording is not
   evidence either way. Alternative: sarcasm about communication still evidences
   *some* self-awareness about it.

Whichever way you go, the rationale field must match what you actually believe -
that is the sentence you will be asked to defend.