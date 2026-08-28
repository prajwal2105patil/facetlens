"""FacetLens CLI.

    python -m src.pipeline enrich                     # rebuild enriched catalogue + audit
    python -m src.pipeline embed                      # precompute facet embeddings
    python -m src.pipeline score --text "..."         # score one conversation
    python -m src.pipeline benchmark                  # run the benchmark + report
"""

from __future__ import annotations

import argparse
import json
import sys


def cmd_enrich(args: argparse.Namespace) -> int:
    from .preprocessing.enrich import AUDIT_MD, OUT_CSV, run

    findings = run()
    n = findings["n_rows"]
    print(f"Wrote {OUT_CSV}  ({n} rows)")
    print(f"Wrote {AUDIT_MD}")
    print(f"  observable          : {findings['observable']}/{n}")
    print(f"  header-like         : {len(findings['header_like'])}")
    print(f"  numeric-prefixed    : {findings['numeric_prefixed']}")
    print(f"  encoding artifacts  : {len(findings['encoding_artifacts'])}")
    print(f"  exact duplicates    : {len(findings['exact_duplicates'])}")
    print(f"  normalised collisions: {len(findings['normalized_collisions'])}")
    print("  types:")
    for facet_type, count in findings["type_counts"].most_common():
        print(f"    {facet_type:38s} {count:4d}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    from .retrieval.embed import build_index

    index = build_index(force=args.force)
    print(f"Embedded {len(index.facet_ids)} facets -> {index.matrix.shape}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from .scoring.scorer import score_conversation

    result = score_conversation(
        args.text,
        top_k=args.top_k,
        batch_size=args.batch_size,
        backend_name=args.backend,
        model=args.model,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .evaluation.evaluate import run_benchmark

    report_path = run_benchmark(
        top_k=args.top_k,
        batch_size=args.batch_size,
        backend_name=args.backend,
        model=args.model,
        limit=args.limit,
    )
    print(f"Wrote {report_path}")
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    from .evaluation.ablation import run

    print(f"Wrote {run()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="src.pipeline", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("enrich", help="rebuild enriched catalogue + audit report"
                   ).set_defaults(func=cmd_enrich)

    p_embed = sub.add_parser("embed", help="precompute facet embeddings")
    p_embed.add_argument("--force", action="store_true",
                         help="rebuild even if the cached matrix is current")
    p_embed.set_defaults(func=cmd_embed)

    def add_runtime_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--top-k", type=int, default=25,
                       help="candidates retrieved before the observability gate")
        p.add_argument("--batch-size", type=int, default=5,
                       help="facets per LLM scoring call")
        p.add_argument("--backend", default="ollama", choices=["ollama", "mock"])
        p.add_argument("--model", default="qwen2.5:7b-instruct")

    p_score = sub.add_parser("score", help="score a single conversation")
    p_score.add_argument("--text", required=True)
    add_runtime_flags(p_score)
    p_score.set_defaults(func=cmd_score)

    p_bench = sub.add_parser("benchmark", help="run the benchmark and write the report")
    add_runtime_flags(p_bench)
    p_bench.add_argument("--limit", type=int, default=None,
                         help="only run the first N conversations (dev loop)")
    p_bench.set_defaults(func=cmd_benchmark)

    sub.add_parser("ablation",
                   help="compare bare vs enriched retrieval text (no LLM calls)"
                   ).set_defaults(func=cmd_ablation)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
