"""Summarize individual expert ratings for a SummEval model."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ("coherence", "fluency", "consistency", "relevance")
SCORES = range(1, 6)
EXPECTED_EXPERTS = 3


def distribution(values):
    counts = Counter(values)
    total = len(values)
    return {
        score: {
            "count": counts[score],
            "percent": round(100 * counts[score] / total, 2) if total else None,
        }
        for score in SCORES
    }


def complete_expert_triplets(items, dimension):
    """Return items with exactly three valid ratings for the dimension."""
    triplets = (item["ratings"][dimension] for item in items)
    return [
        ratings for ratings in triplets
        if len(ratings) == EXPECTED_EXPERTS
        and all(type(score) is int and score in SCORES for score in ratings)
    ]


def screen(path, model_id):
    items = {}
    duplicates = Counter()
    matched = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if row.get("model_id") != model_id:
                continue
            matched += 1
            source_id = row["id"]
            if source_id in items:
                duplicates[source_id] += 1
                continue
            experts = row.get("expert_annotations")
            experts = experts if isinstance(experts, list) else []
            items[source_id] = {
                "source_id": source_id,
                "expert_count": len(experts),
                "ratings": {
                    dimension: [
                        expert.get(dimension) if isinstance(expert, dict) else None
                        for expert in experts
                    ]
                    for dimension in DIMENSIONS
                },
            }

    if not items:
        raise ValueError(f"No {model_id} rows found.")
    dimensions = {}
    for dimension in DIMENSIONS:
        pooled, references = [], []
        missing = invalid = 0
        for item in items.values():
            ratings = item["ratings"][dimension]
            valid = [score for score in ratings if type(score) is int and score in SCORES]
            missing += max(0, EXPECTED_EXPERTS - len(ratings)) + ratings.count(None)
            invalid += len(ratings) - len(valid) - ratings.count(None)
            pooled.extend(valid)
            if len(ratings) == len(valid) == EXPECTED_EXPERTS:
                references.append(median(valid))
        dimensions[dimension] = {
            "valid_rating_count": len(pooled),
            "missing_rating_count": missing,
            "invalid_rating_count": invalid,
            "reference_count": len(references),
            "items_without_reference": len(items) - len(references),
            "expert_ratings": distribution(pooled),
            "expert_medians": distribution(references),
        }

    return {
        "model_id": model_id,
        "matched_rows": matched,
        "unique_summaries": len(items),
        "distinct_source_documents": len(items),
        "duplicate_rows_by_source_id": dict(duplicates),
        "unexpected_expert_count_items": sum(
            item["expert_count"] != EXPECTED_EXPERTS for item in items.values()
        ),
        "dimensions": dimensions,
        "items": list(items.values()),
    }


def print_report(report):
    print(f"{report['model_id']} | {report['matched_rows']} matched rows | "
          f"{report['unique_summaries']} unique summaries | "
          f"{report['distinct_source_documents']} source documents")
    print(f"Duplicate rows: {sum(report['duplicate_rows_by_source_id'].values())} "
          f"(first row kept per source) | "
          f"Items with other than three expert entries: {report['unexpected_expert_count_items']}")
    print("Expert ratings are pooled because the release has no rater IDs.")
    for dimension, stats in report["dimensions"].items():
        print(f"\n{dimension.title()}")
        print(f"Valid ratings: {stats['valid_rating_count']} | "
              f"Missing: {stats['missing_rating_count']} | Invalid: {stats['invalid_rating_count']} | "
              f"Items with an expert median: {stats['reference_count']}")
        print(f"{'Score':>5}  {'Expert ratings':>16}  {'Expert medians':>16}")
        for score in SCORES:
            cells = []
            for field in ("expert_ratings", "expert_medians"):
                entry = stats[field][score]
                percent = "n/a" if entry["percent"] is None else f"{entry['percent']:.1f}%"
                cells.append(f"{entry['count']} ({percent})")
            print(f"{score:>5}  {cells[0]:>16}  {cells[1]:>16}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "SummEval_data" / "model_annotations.aligned.jsonl")
    parser.add_argument("--model", required=True, help="Exact SummEval model ID")
    parser.add_argument("--output", type=Path, required=True, help="Path to the JSON report")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must refer to different files")
    try:
        report = screen(args.input, args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"SummEval screening failed: {exc}\n")
    print_report(report)
    print(f"\nJSON report saved to {args.output}")


if __name__ == "__main__":
    main()
