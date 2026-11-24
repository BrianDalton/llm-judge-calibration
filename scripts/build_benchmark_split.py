"""Create a stratified SummEval benchmark JSONL from paired annotations.

Run with --input PAIRED.jsonl --output BENCHMARK.jsonl --model MODEL_ID
--dimension DIMENSION. Optional --cal-size and --seed default to 30 and 42.
The output contains expert labels and split assignments. Terminal output
summarizes the split by publisher, median band, rating spread, and length.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

# Allow direct invocation from a checkout to import the shared model package.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.benchmark_annotation import BenchmarkAnnotation
from scripts.screen_summeval_single_model import DIMENSIONS, EXPECTED_EXPERTS, SCORES


def load_examples(path, model, dimension):
    examples = {}
    article_texts = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["model_id"] != model:
                continue

            article_id = row["id"]
            if article_id in examples:
                raise ValueError(f"Duplicate article ID for {model}: {article_id}")
            article = " ".join(row["text"].split())
            if not article:
                raise ValueError(f"Missing article text for {article_id}")
            if article in article_texts:
                raise ValueError(f"Duplicate article text at {article_id}")
            article_texts.add(article)

            ratings = [expert.get(dimension) for expert in row["expert_annotations"]]
            if len(ratings) != EXPECTED_EXPERTS or any(
                type(score) is not int or score not in SCORES for score in ratings
            ):
                raise ValueError(f"Expected three valid {dimension} ratings for {article_id}")
            score = median(ratings)
            band = "1–2" if score <= 2 else "3–4" if score <= 4 else "5"
            publisher = Path(row["filepath"]).parts[1]
            if publisher not in {"cnn", "dailymail"}:
                raise ValueError(f"Unrecognized publisher for {article_id}: {publisher}")
            examples[article_id] = {
                "benchmark": {
                    "article_id": article_id,
                    "model_id": model,
                    "dimension": dimension,
                    "expert_ratings": ratings,
                    "expert_median": score,
                },
                "publisher": publisher,
                "band": band,
                "spread": min(max(ratings) - min(ratings), 2),
                "article_words": len(article.split()),
                "summary_words": len(row["decoded"].split()),
            }
    return examples


def assign_splits(examples, cal_size, seed):
    total = len(examples)
    if not 0 < cal_size < total:
        raise ValueError(f"Calibration size must be between 1 and {total - 1} for {total} articles.")

    strata = defaultdict(list)
    for article_id, example in sorted(examples.items()):
        strata[example["publisher"], example["band"]].append(article_id)

    quotas = {group: divmod(cal_size * len(ids), total) for group, ids in strata.items()}
    remaining = cal_size - sum(base for base, remainder in quotas.values())
    extra_slots = set(sorted(strata, key=lambda group: (-quotas[group][1], group))[:remaining])

    rng = random.Random(seed)
    for group, ids in sorted(strata.items()):
        rng.shuffle(ids)
        quota = quotas[group][0] + (group in extra_slots)
        for index, article_id in enumerate(ids):
            examples[article_id]["benchmark"]["split"] = "cal" if index < quota else "eval"


def print_balance(examples):
    cal = [item for item in examples.values() if item["benchmark"]["split"] == "cal"]
    eval_items = [item for item in examples.values() if item["benchmark"]["split"] == "eval"]
    for name, items in (("cal", cal), ("eval", eval_items)):
        print(f"\n{name}: {len(items)} articles")
        for field, label, categories in (
            ("publisher", "Publisher", ("cnn", "dailymail")),
            ("band", "Expert median band", ("1–2", "3–4", "5")),
            ("spread", "Expert rating spread", (0, 1, 2)),
        ):
            counts = Counter(item[field] for item in items)
            cells = [
                f"{'2+' if field == 'spread' and category == 2 else category}={counts[category]}"
                for category in categories
            ]
            print(f"  {label}: {', '.join(cells)}")
        for field, label in (("article_words", "Article"), ("summary_words", "Summary")):
            lengths = [item[field] for item in items]
            print(f"  {label} words: {min(lengths)}–{max(lengths)} (median {median(lengths):g})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="Paired SummEval annotations JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Benchmark labels JSONL")
    parser.add_argument("--model", required=True, help="Exact SummEval model ID")
    parser.add_argument("--dimension", required=True, choices=DIMENSIONS)
    parser.add_argument("--cal-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must be different files")
    try:
        examples = load_examples(args.input, args.model, args.dimension)
        if not examples:
            raise ValueError(f"No annotations found for {args.model}")
        assign_splits(examples, args.cal_size, args.seed)
        annotations = [
            BenchmarkAnnotation.model_validate(example["benchmark"])
            for _, example in sorted(examples.items())
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output:
            for annotation in annotations:
                output.write(json.dumps(annotation.model_dump(mode="json")) + "\n")
    except (OSError, ValueError, KeyError, IndexError) as exc:
        parser.exit(1, f"Benchmark split failed: {exc}\n")

    print(f"{args.model} {args.dimension} | seed {args.seed}")
    print_balance(examples)
    print(f"\nSaved benchmark labels to {args.output}")


if __name__ == "__main__":
    main()
