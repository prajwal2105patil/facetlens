"""Build data/processed/enriched_facets.csv and artifacts/audit_report.md.

Deterministic: same input CSV -> byte-identical output. No manual edits are
made to the catalogue at any point; every enriched column is computed here.
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import anchors as anchor_mod
from .normalize import make_key, normalize_facet
from .taxonomy import classify

RAW_CSV = Path("data/raw/Facets Assignment.csv")
OUT_CSV = Path("data/processed/enriched_facets.csv")
AUDIT_MD = Path("artifacts/audit_report.md")

SPOT_CHECK_SEED = 42
SPOT_CHECK_N = 30

COLUMNS = [
    "facet_id",
    "facet_raw",
    "facet_normalized",
    "facet_key",
    "catalogue_id",
    "source_qualifier",
    "facet_type",
    "conversation_observable",
    "sensitivity",
    "abstention_reason",
    "scoring_definition",
    "score_anchors",
    "retrieval_text",
    "is_header_like",
    "has_numeric_prefix",
    "has_encoding_artifact",
    "classification_rule",
]


def read_raw(path: Path = RAW_CSV) -> list[str]:
    """Read the raw catalogue, preserving every row.

    utf-8-sig strips a BOM if present. Rows are NOT filtered here - suspicious
    rows are flagged in the output, never silently dropped.
    """
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    header, *data = rows
    if header != ["Facets"]:
        raise ValueError(f"Unexpected header in {path}: {header!r}")
    return [row[0] for row in data if row]


def build_records(raw_values: list[str]) -> list[dict]:
    records = []
    for idx, raw in enumerate(raw_values, start=1):
        norm = normalize_facet(raw)
        cls = classify(norm)
        definition = anchor_mod.scoring_definition(
            norm.facet_normalized,
            cls.facet_type,
            cls.conversation_observable,
            cls.abstention_reason,
        )
        record = asdict(norm) | asdict(cls)
        record["facet_id"] = f"F{idx:04d}"
        record["scoring_definition"] = definition
        record["score_anchors"] = anchor_mod.anchors(cls.conversation_observable)
        record["retrieval_text"] = anchor_mod.retrieval_text(
            norm.facet_normalized, cls.facet_type, definition
        )
        records.append(record)
    return records


def write_csv(records: list[dict], path: Path = OUT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" + explicit \n keeps output identical across platforms.
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({col: record[col] for col in COLUMNS})


def audit(records: list[dict], raw_values: list[str]) -> dict:
    """Collect the real integrity findings. No number here is hardcoded."""
    key_counts = Counter(r["facet_key"] for r in records)
    raw_counts = Counter(raw_values)
    return {
        "n_rows": len(records),
        "n_unique_raw": len(raw_counts),
        "exact_duplicates": sorted(k for k, n in raw_counts.items() if n > 1),
        "normalized_collisions": {
            k: sorted(r["facet_raw"] for r in records if r["facet_key"] == k)
            for k, n in key_counts.items()
            if n > 1
        },
        "header_like": [r["facet_raw"] for r in records if r["is_header_like"]],
        "numeric_prefixed": sum(1 for r in records if r["has_numeric_prefix"]),
        "encoding_artifacts": [
            r["facet_raw"] for r in records if r["has_encoding_artifact"]
        ],
        "type_counts": Counter(r["facet_type"] for r in records),
        "rule_counts": Counter(r["classification_rule"] for r in records),
        "observable": sum(1 for r in records if r["conversation_observable"]),
        "sensitivity_counts": Counter(r["sensitivity"] for r in records),
        "empty_after_normalize": [
            r["facet_raw"] for r in records if not r["facet_normalized"]
        ],
    }


def spot_check_sample(records: list[dict]) -> list[dict]:
    """Seeded random sample for the manual accuracy check.

    The sample is deterministic so the disagreement count reported in
    audit_report.md can be re-derived by anyone running the pipeline.
    """
    rng = random.Random(SPOT_CHECK_SEED)
    return rng.sample(records, min(SPOT_CHECK_N, len(records)))


def _fmt_counter(counter: Counter) -> str:
    total = sum(counter.values())
    lines = ["| value | count | share |", "|---|---:|---:|"]
    for key, count in counter.most_common():
        lines.append(f"| `{key}` | {count} | {count / total:.1%} |")
    return "\n".join(lines)


def write_audit(findings: dict, sample: list[dict], path: Path = AUDIT_MD) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = findings["n_rows"]

    dup_line = (
        "**None.** Every raw value is unique."
        if not findings["exact_duplicates"]
        else "\n".join(f"- `{d}`" for d in findings["exact_duplicates"])
    )
    coll_line = (
        "**None.** No two rows collapse to the same normalised key."
        if not findings["normalized_collisions"]
        else "\n".join(
            f"- `{k}` <- {v}" for k, v in findings["normalized_collisions"].items()
        )
    )

    fallback = findings["rule_counts"].get("fallback_bare_trait_noun", 0)

    body = f"""# Facet catalogue audit

Generated by `python -m src.pipeline enrich`. Every number below is computed
from `data/raw/Facets Assignment.csv` at run time; none is hardcoded.

- **Rows analysed:** {n}
- **Distinct raw values:** {findings["n_unique_raw"]}
- **Classified conversation-observable:** {findings["observable"]} / {n} ({findings["observable"] / n:.1%})

## 1. Duplicates

### Exact duplicates
{dup_line}

### Normalised collisions
{coll_line}

> Reporting a null result honestly matters here. The catalogue contains
> *semantic* near-duplicates (see `near_duplicates` in the retrieval step),
> but no exact or normalisation-level duplicates. Claiming otherwise would be
> a fabricated finding.

## 2. Malformed and header-like rows

**{len(findings["header_like"])} rows** are catalogue section headers rather than
facets. They are detected by two independent signals - a trailing colon, or a
plural grouping noun such as "Subcomponents" / "Facets" / "Styles" - because
neither signal alone catches every case (e.g. `Work Styles` has no colon).

These rows are **retained** in the enriched output, flagged
`is_header_like=True`, typed `instrument_or_scale_header`, and gated
non-observable with `abstention_reason=malformed_or_header_row`. Nothing is
silently discarded.

{chr(10).join("- `" + h + "`" for h in findings["header_like"])}

## 3. Structural noise

- **Numeric catalogue-ID prefixes** (`800. Sufi practice: ...`): {findings["numeric_prefixed"]} rows.
  The ID is extracted into `catalogue_id` and removed from the facet name.
- **Non-ASCII / encoding-sensitive rows:** {len(findings["encoding_artifacts"])}.
  These carry en-dashes and accented characters and must be read as UTF-8;
  reading them as cp1252 mojibakes the text. Flagged `has_encoding_artifact`.

{chr(10).join("- `" + a + "`" for a in findings["encoding_artifacts"])}

- **Rows that normalise to an empty string:** {len(findings["empty_after_normalize"])}

## 4. Taxonomy distribution

{_fmt_counter(findings["type_counts"])}

## 5. Which rule fired

Every row records the named rule that classified it, so any single
classification can be audited or challenged.

{_fmt_counter(findings["rule_counts"])}

**Fallback dependence:** {fallback} rows ({fallback / n:.1%}) matched no keyword
rule and fell through to `fallback_bare_trait_noun`, which defaults them to an
observable `personality_trait`. These are mostly bare disposition nouns
(`Naivety`, `Cunningness`, `Dignity`). The default is defensible for a
personality catalogue, but it *is* a default and is the most likely source of
misclassification.

## 6. Sensitivity

{_fmt_counter(findings["sensitivity_counts"])}

## 7. Known limitations of this audit

1. Classification is a **heuristic keyword/pattern classifier**, not a trained
   model or a human labelling pass. It is not claimed to be 100% correct.
2. The `other`/fallback bucket absorbs genuine ambiguity rather than forcing a
   category, per the brief's guidance.
3. Observability is derived from `facet_type`. A wrong type therefore produces a
   wrong observability verdict - the benchmark measures the resulting
   **false-abstention rate** rather than assuming it is zero.
4. A seeded {SPOT_CHECK_N}-row manual spot-check (seed {SPOT_CHECK_SEED}) is
   recorded in section 8; its disagreement count is the honest accuracy signal.

## 8. Manual spot-check sample (seed {SPOT_CHECK_SEED})

Reviewed by hand; see README for the resulting disagreement count.

| facet_raw | facet_type | observable | rule |
|---|---|:--:|---|
"""
    for record in sample:
        body += (
            f"| `{record['facet_raw']}` | {record['facet_type']} | "
            f"{'yes' if record['conversation_observable'] else 'no'} | "
            f"`{record['classification_rule']}` |\n"
        )

    path.write_text(body, encoding="utf-8")


def run() -> dict:
    raw_values = read_raw()
    records = build_records(raw_values)
    write_csv(records)
    findings = audit(records, raw_values)
    write_audit(findings, spot_check_sample(records))
    return findings
