"""Benchmark runner and evaluation.

Design choice that matters for honesty: the pipeline runs REAL retrieval over
all 399 facets, so retrieval recall is measurable. Separately, every
reference-labelled facet is force-included so that agreement is computed over
the whole reference set rather than only over facets retrieval happened to
find. Confusing those two would let a retrieval miss silently disappear from
the agreement numbers.

Forced facets still pass through the observability gate, so forcing costs no
extra LLM calls for non-observable facets.

Every number in the report is computed here. Nothing is hardcoded.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..retrieval.embed import FacetIndex, build_index
from ..retrieval.retrieve import Candidate, route
from ..scoring.backends import get_backend
from ..scoring.scorer import _chunk, _gate_verdict, score_batch
from ..scoring.schema import ConversationResult, FacetVerdict, Status

BENCH_DIR = Path("data/benchmark")
CONVERSATIONS = BENCH_DIR / "conversations.jsonl"
REFERENCE = BENCH_DIR / "reference_labels.jsonl"
FACET_LIST = BENCH_DIR / "benchmark_facets.txt"
REPORT = Path("artifacts/benchmark_report.md")
RESULTS = Path("artifacts/benchmark_results.jsonl")

ABSTAIN = (Status.NOT_OBSERVABLE, Status.INSUFFICIENT_EVIDENCE)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_facet_list(path: Path = FACET_LIST) -> list[str]:
    with path.open(encoding="utf-8") as fh:
        return [
            line.strip() for line in fh
            if line.strip() and not line.lstrip().startswith("#")
        ]


def _candidate_from_row(row: dict, retrieval_score: float | None) -> Candidate:
    return Candidate(
        facet_id=row["facet_id"],
        facet=row["facet_normalized"],
        facet_type=row["facet_type"],
        conversation_observable=row["conversation_observable"],
        abstention_reason=row["abstention_reason"] or None,
        sensitivity=row["sensitivity"],
        scoring_definition=row["scoring_definition"],
        score_anchors=row["score_anchors"],
        retrieval_score=retrieval_score if retrieval_score is not None else -1.0,
    )


@dataclass
class Metrics:
    """Counts only. Percentages are rendered with their denominator."""

    exact_agreement: int = 0
    within_one: int = 0
    score_pairs: int = 0
    correct_abstentions: int = 0
    missed_abstentions: int = 0      # reference says abstain, system scored
    false_abstentions: int = 0       # reference says score, system abstained
    abstention_pairs: int = 0
    status_exact: int = 0
    total_pairs: int = 0
    retrieval_hits: int = 0
    retrieval_total: int = 0
    errors: int = 0
    repairs: int = 0
    evidence_failures: int = 0
    confidence_buckets: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))
    failures: list = field(default_factory=list)


def run_benchmark(top_k: int = 15, batch_size: int = 5,
                  backend_name: str = "ollama",
                  model: str = "qwen2.5:7b-instruct",
                  limit: int | None = None,
                  index: FacetIndex | None = None) -> Path:
    index = index or build_index()
    backend = get_backend(backend_name, model)
    conversations = load_jsonl(CONVERSATIONS)
    if limit:
        conversations = conversations[:limit]
    reference = load_jsonl(REFERENCE)

    by_raw = {row["facet_raw"]: row for row in index.rows}
    ref_by_conv: dict[str, list[dict]] = defaultdict(list)
    for label in reference:
        ref_by_conv[label["conversation_id"]].append(label)

    results: list[ConversationResult] = []
    retrieval_hits, retrieval_total = 0, 0

    for conv in conversations:
        text = conv["text"]
        routed = route(text, index=index, top_k=top_k)
        retrieved_ids = {c.facet_id for c in routed.scorable} | {
            c.facet_id for c in routed.gated_out
        }

        # Retrieval recall is measured BEFORE forcing anything in.
        for label in ref_by_conv[conv["conversation_id"]]:
            row = by_raw.get(label["facet_raw"])
            if row is None:
                continue
            retrieval_total += 1
            if row["facet_id"] in retrieved_ids:
                retrieval_hits += 1

        # Force-include labelled facets that retrieval missed, so agreement is
        # computed over the full reference set.
        forced_scorable, forced_gated = [], []
        for label in ref_by_conv[conv["conversation_id"]]:
            row = by_raw.get(label["facet_raw"])
            if row is None or row["facet_id"] in retrieved_ids:
                continue
            candidate = _candidate_from_row(row, None)
            (forced_scorable if row["conversation_observable"] else forced_gated
             ).append(candidate)

        verdicts = [_gate_verdict(c) for c in routed.gated_out + forced_gated]
        for batch in _chunk(routed.scorable + forced_scorable, batch_size):
            verdicts.extend(score_batch(text, batch, backend))
        verdicts.sort(key=lambda v: (-(v.retrieval_score or 0.0), v.facet_id))

        results.append(ConversationResult(
            conversation_id=conv["conversation_id"], conversation=text,
            model=f"{backend.name}:{backend.model}", top_k=top_k,
            batch_size=batch_size, verdicts=verdicts,
        ))
        print(f"  {conv['conversation_id']:18s} "
              f"scored={len([v for v in verdicts if v.status == Status.SCORED]):2d} "
              f"abstained={len([v for v in verdicts if v.status in ABSTAIN]):2d} "
              f"errors={len([v for v in verdicts if v.status == Status.ERROR]):2d}")

    metrics = score_against_reference(results, reference, by_raw)
    metrics.retrieval_hits, metrics.retrieval_total = retrieval_hits, retrieval_total

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", encoding="utf-8", newline="\n") as fh:
        for result in results:
            fh.write(json.dumps(result.model_dump(mode="json"), ensure_ascii=False) + "\n")

    write_report(results, metrics, conversations, backend, top_k, batch_size)
    return REPORT


def score_against_reference(results: list[ConversationResult],
                            reference: list[dict], by_raw: dict) -> Metrics:
    metrics = Metrics()
    verdict_lookup: dict[tuple[str, str], FacetVerdict] = {
        (r.conversation_id, v.facet_id): v for r in results for v in r.verdicts
    }

    for result in results:
        for verdict in result.verdicts:
            if verdict.status == Status.ERROR:
                metrics.errors += 1
            if verdict.schema_repaired:
                metrics.repairs += 1
            if verdict.evidence_verified is False:
                metrics.evidence_failures += 1

    for label in reference:
        row = by_raw.get(label["facet_raw"])
        if row is None:
            continue
        verdict = verdict_lookup.get((label["conversation_id"], row["facet_id"]))
        if verdict is None or verdict.status == Status.ERROR:
            continue

        metrics.total_pairs += 1
        expected_status = label["expected_status"]
        got_status = verdict.status.value
        if expected_status == got_status:
            metrics.status_exact += 1

        expected_abstain = expected_status in ("not_observable", "insufficient_evidence")
        got_abstain = verdict.status in ABSTAIN

        if expected_abstain:
            metrics.abstention_pairs += 1
            if got_abstain:
                metrics.correct_abstentions += 1
            else:
                metrics.missed_abstentions += 1
                metrics.failures.append({
                    "type": "missed_abstention",
                    "conversation_id": label["conversation_id"],
                    "facet": label["facet_raw"],
                    "expected": expected_status,
                    "got": f"scored {verdict.score}",
                    "model_reason": verdict.reason,
                    "rationale": label["rationale"],
                })
        elif got_abstain:
            metrics.false_abstentions += 1
            metrics.failures.append({
                "type": "false_abstention",
                "conversation_id": label["conversation_id"],
                "facet": label["facet_raw"],
                "expected": f"scored {label['expected_score']}",
                "got": got_status,
                "model_reason": verdict.reason,
                "rationale": label["rationale"],
            })

        if expected_status == "scored" and verdict.status == Status.SCORED:
            metrics.score_pairs += 1
            delta = abs(verdict.score - label["expected_score"])
            if delta == 0:
                metrics.exact_agreement += 1
            if delta <= 1:
                metrics.within_one += 1
            else:
                metrics.failures.append({
                    "type": "score_disagreement",
                    "conversation_id": label["conversation_id"],
                    "facet": label["facet_raw"],
                    "expected": f"scored {label['expected_score']}",
                    "got": f"scored {verdict.score}",
                    "model_reason": verdict.reason,
                    "rationale": label["rationale"],
                })

            bucket = f"{int(verdict.confidence * 10) / 10:.1f}"
            metrics.confidence_buckets[bucket][1] += 1
            if delta == 0:
                metrics.confidence_buckets[bucket][0] += 1

    return metrics


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a (0 cases)"
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})"


def write_report(results, metrics: Metrics, conversations, backend,
                 top_k: int, batch_size: int) -> None:
    status_totals = Counter(
        v.status.value for r in results for v in r.verdicts
    )
    origin_totals = Counter(v.origin for r in results for v in r.verdicts)
    gate_saved = origin_totals.get("observability_gate", 0)
    total_verdicts = sum(status_totals.values())

    lines = [
        "# Benchmark report",
        "",
        f"Generated by `python -m src.pipeline benchmark`. "
        f"Model: `{backend.name}:{backend.model}`. top_k={top_k}, batch_size={batch_size}.",
        "Every number below is produced by the evaluation code; none is hardcoded.",
        "",
        "## Run shape",
        "",
        f"- Conversations: **{len(results)}**",
        f"- Verdicts emitted: **{total_verdicts}**",
        f"- Reference-labelled pairs evaluated: **{metrics.total_pairs}**",
        f"- Facets answered by the observability gate with **no LLM call**: "
        f"**{gate_saved}** ({gate_saved / max(total_verdicts, 1):.0%} of all verdicts)",
        "",
        "| status | count |", "|---|---:|",
    ]
    for status, count in status_totals.most_common():
        lines.append(f"| `{status}` | {count} |")

    lines += [
        "",
        "## Agreement with the human reference set",
        "",
        f"- **Status agreement** (scored vs abstained vs not_observable): "
        f"{_pct(metrics.status_exact, metrics.total_pairs)}",
        f"- **Exact score agreement** (both scored): "
        f"{_pct(metrics.exact_agreement, metrics.score_pairs)}",
        f"- **Within +/-1**: {_pct(metrics.within_one, metrics.score_pairs)}",
        "",
        "## Abstention analysis",
        "",
        f"- **Correct abstentions**: {_pct(metrics.correct_abstentions, metrics.abstention_pairs)}",
        f"- **Missed abstentions** (system scored something the reference says is "
        f"unsupported): **{metrics.missed_abstentions}**",
        f"- **False abstentions** (system abstained where the reference expects a "
        f"score): **{metrics.false_abstentions}**",
        "",
        "> Missed abstentions are the dangerous direction: they are confident "
        "hallucinations. False abstentions cost coverage but never invent a fact.",
        "",
        "## Retrieval",
        "",
        f"- **Recall@{top_k}** of reference-labelled facets: "
        f"{_pct(metrics.retrieval_hits, metrics.retrieval_total)}",
        "",
        "Measured before any labelled facet is force-included, so misses are "
        "visible rather than absorbed.",
        "",
        "## Robustness",
        "",
        f"- Verdicts that ended as `error`: **{metrics.errors}**",
        f"- Contradictory verdicts repaired toward abstention: **{metrics.repairs}**",
        f"- Fabricated evidence quotes caught by the verifier: "
        f"**{metrics.evidence_failures}**",
        "",
        "## Confidence calibration",
        "",
        "Self-reported model confidence vs actual exact-score agreement. This is "
        "a diagnostic, not a calibrated probability.",
        "",
        "| confidence bucket | exact agreement |", "|---|---|",
    ]
    for bucket in sorted(metrics.confidence_buckets):
        correct, total = metrics.confidence_buckets[bucket]
        lines.append(f"| {bucket} | {_pct(correct, total)} |")

    lines += ["", "## Failure cases", ""]
    if not metrics.failures:
        lines.append("No disagreements against the reference set.")
    else:
        by_type = Counter(f["type"] for f in metrics.failures)
        lines.append("| type | count |")
        lines.append("|---|---:|")
        for failure_type, count in by_type.most_common():
            lines.append(f"| `{failure_type}` | {count} |")
        lines.append("")
        for failure in metrics.failures:
            lines += [
                f"### `{failure['type']}` - {failure['conversation_id']} / "
                f"{failure['facet']}",
                f"- expected: **{failure['expected']}**, got: **{failure['got']}**",
                f"- reference rationale: {failure['rationale']}",
                f"- system reason: {failure['model_reason']}",
                "",
            ]

    lines += ["## Per-conversation detail", ""]
    for result in results:
        lines += [
            f"### `{result.conversation_id}`", "",
            f"> {result.conversation}", "",
            "| facet | status | score | conf | origin | evidence |",
            "|---|---|---:|---:|---|---|",
        ]
        for verdict in result.verdicts:
            quote = (verdict.evidence_quote or "").replace("|", "/")[:45]
            score = verdict.score if verdict.score is not None else ""
            lines.append(
                f"| {verdict.facet[:38]} | `{verdict.status.value}` | {score} | "
                f"{verdict.confidence:.2f} | {verdict.origin} | {quote} |"
            )
        lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
