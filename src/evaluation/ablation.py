"""Ablation: what does the enriched retrieval text actually buy?

Compares two retrieval-text choices on the same reference set, same embedding
model, same K values:

    bare      facet_normalized                                  ("FSH level")
    enriched  facet_normalized + facet_type + scoring_definition

Recall is reported split by what the reference set expects, because the two
directions mean opposite things:

  * should-score facets   - a miss is a real loss. The system never sees the
                            facet, so it cannot score it.
  * should-abstain facets - a miss is harmless, arguably good. Not retrieving
                            'FSH level' is a better outcome than retrieving it
                            and gating it, because it costs nothing.

Reporting a single blended recall number would hide that asymmetry. No LLM
calls are involved, so this runs in seconds.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..retrieval.embed import build_index
from ..retrieval.retrieve import retrieve
from .evaluate import CONVERSATIONS, REFERENCE, load_jsonl

REPORT = Path("artifacts/ablation_retrieval.md")
K_VALUES = (10, 15, 25, 40, 60, 100)

#: Each arm is (label, text_column, use_expansions, mode).
#: Every one of these was actually built and measured; three of the four
#: alternatives to the shipped configuration made recall WORSE, and are kept
#: here as evidence rather than deleted.
ARMS = (
    ("bare name", "facet_normalized", False, "dense"),
    ("enriched", "retrieval_text", False, "dense"),
    ("BM25 only", "retrieval_text", False, "lexical"),
    ("dense+BM25", "retrieval_text", False, "hybrid"),
    ("enriched+expansions", "retrieval_text", True, "dense"),
)


def measure(text_column: str, use_expansions: bool = False,
            mode: str = "dense") -> dict[int, tuple[int, int, int, int]]:
    """Return {K: (score_hits, score_total, abstain_hits, abstain_total)}."""
    index = build_index(text_column=text_column, use_expansions=use_expansions)
    conversations = {c["conversation_id"]: c["text"] for c in load_jsonl(CONVERSATIONS)}
    by_raw = {row["facet_raw"]: row for row in index.rows}

    labels_by_conv = defaultdict(list)
    for label in load_jsonl(REFERENCE):
        labels_by_conv[label["conversation_id"]].append(label)

    results = {}
    for k in K_VALUES:
        score_hits = score_total = abstain_hits = abstain_total = 0
        for conversation_id, text in conversations.items():
            retrieved = {c.facet_id for c in retrieve(text, index, top_k=k,
                                                      mode=mode)}
            for label in labels_by_conv[conversation_id]:
                row = by_raw.get(label["facet_raw"])
                if row is None:
                    continue
                hit = row["facet_id"] in retrieved
                if label["expected_status"] == "scored":
                    score_total += 1
                    score_hits += hit
                else:
                    abstain_total += 1
                    abstain_hits += hit
        results[k] = (score_hits, score_total, abstain_hits, abstain_total)
    return results


def run() -> Path:
    measurements = {
        label: measure(column, expansions, mode)
        for label, column, expansions, mode in ARMS
    }
    labels = [a[0] for a in ARMS]
    shipped = "enriched+expansions"

    def table(index: int, denominator_index: int) -> list[str]:
        rows = ["| K | " + " | ".join(labels) + " |",
                "|---:|" + "---|" * len(labels)]
        for k in K_VALUES:
            cells = []
            for label in labels:
                hits = measurements[label][k][index]
                total = measurements[label][k][denominator_index]
                cells.append(f"{hits}/{total} ({hits / total:.0%})" if total else "n/a")
            rows.append(f"| {k} | " + " | ".join(cells) + " |")
        return rows

    lines = [
        "# Ablation: five retrieval configurations",
        "",
        "Same embedding model, same reference set, same K values. Only the "
        "indexed text and the similarity signal change. No LLM calls; generated "
        "by `python -m src.pipeline ablation`.",
        "",
        "| arm | what it is |",
        "|---|---|",
        "| bare name | embed the facet name alone |",
        "| enriched | name + type + generated scoring definition |",
        "| BM25 only | lexical match over the enriched text, no embeddings |",
        "| dense+BM25 | reciprocal rank fusion of the two |",
        "| **enriched+expansions** | enriched text + 2 generated example "
        "utterances per facet (**shipped**) |",
        "",
        "## Recall on facets the reference says SHOULD be scored",
        "",
        "The number that matters. A miss here is a real loss: the facet never "
        "reaches the scorer, so it cannot be scored however good the scorer is.",
        "",
    ]
    lines += table(0, 1)

    lines += [
        "",
        "## Recall on facets the reference says should be ABSTAINED",
        "",
        "A miss here is harmless and mildly good: a non-observable facet that is "
        "never retrieved costs nothing, whereas one that is retrieved must still "
        "be gated. Reporting a single blended recall would hide this asymmetry.",
        "",
    ]
    lines += table(2, 3)

    baseline = measurements["enriched"]
    best = measurements[shipped]
    at25 = (best[25][0], best[25][1], baseline[25][0])

    lines += [
        "",
        "## What each attempt was worth",
        "",
        f"At K=25, the shipped configuration retrieves **{at25[0]}/{at25[1]}** of "
        f"should-score facets against **{at25[2]}/{at25[1]}** for the previous "
        "default.",
        "",
        "**Three of the four alternatives made retrieval worse and were not "
        "shipped.** They are kept here because a rejected experiment with "
        "numbers attached is more useful than a clean-looking report:",
        "",
        "- **BM25 alone** is close to useless on this catalogue. Conversations "
        "describe behaviour (*'I assigned tasks based on their strengths'*) "
        "while facets are abstract labels (*'Delegation skills'*). There is "
        "almost no lexical overlap to exploit, by construction.",
        "- **Dense+BM25 fusion** inherits that weakness: fusing a strong signal "
        "with a near-random one drags the ranking down at most values of K.",
        "- **BGE-small-en-v1.5**, a stronger encoder on public retrieval "
        "benchmarks, scored 10/19 at K=25 against MiniLM's 12/19, and its "
        "recommended query prefix made it worse again. Not included as an arm "
        "here because it needs a second model download; the measurement is "
        "recorded in DECISIONS.md D10.",
        "",
        "Those three failures are what identified the real problem. The "
        "bottleneck was never the encoder or the similarity function - it is "
        "that an abstract label and a concrete narrative do not occupy the same "
        "region of any embedding space. Document expansion attacks that "
        "directly by making each facet look more like the conversations that "
        "should match it.",
        "",
        "## Honesty notes",
        "",
        "- The expansions are generated from a facet's **name, type and "
        "definition only**. The generator never sees a benchmark conversation "
        "or a reference label, so this is not test-set leakage.",
        "- Expansions were generated for **all 399 facets**, not only the "
        "observable ones. Expanding only the observable half would have "
        "confounded the result: recall could have improved merely by demoting "
        "non-observable facets rather than by promoting the right ones.",
        "- The reference set is 55 pairs, 19 of which expect a score. Single-"
        "facet changes move these percentages by ~5 points, so treat small "
        "differences between arms as noise.",
        "",
    ]

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return REPORT
