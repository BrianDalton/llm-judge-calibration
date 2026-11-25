"""Score saved judge predictions against the expert median without API calls."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pvariance, quantiles

from models.judge_prediction import JudgePrediction
from scripts.judge_data import read_benchmark

LEVELS = range(1, 6)


def read_predictions(path, expected, split):
    rows = {}
    settings = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = JudgePrediction.model_validate_json(line)
            key = row.article_id, row.model_id
            if row.split != split or key not in expected:
                raise ValueError(f"Unexpected prediction for {key} in split {row.split}")
            if key in rows:
                raise ValueError(f"Duplicate prediction for {key}")
            rows[key] = row
            settings.add((
                row.prompt_sha256, row.judge_model, row.response_model,
                row.reasoning_effort, row.max_output_tokens,
            ))
    missing = expected.keys() - rows.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} predictions, e.g. {min(missing)}")
    if len(settings) != 1:
        raise ValueError("Predictions must use one prompt, judge model, and settings")
    return rows, settings.pop()


def metrics(reference, judged):
    """Compute exact and distance-sensitive metrics for aligned scores."""
    table = [[0] * 5 for _ in LEVELS]
    for expert, judge in zip(reference, judged, strict=True):
        table[expert - 1][judge - 1] += 1

    reference_counts = [sum(row) for row in table]
    judge_counts = [sum(column) for column in zip(*table)]
    n = len(reference)
    observed = sum(
        (i - j) ** 2 * table[i][j] for i in range(5) for j in range(5)
    )
    expected = sum(
        (i - j) ** 2 * reference_counts[i] * judge_counts[j]
        for i in range(5) for j in range(5)
    )
    gaps = [judge - expert for expert, judge in zip(reference, judged)]
    return {
        "mae": mean(abs(gap) for gap in gaps),
        "qwk": 1 - n * observed / expected if expected else None,
        "exact_match": sum(gap == 0 for gap in gaps) / n,
        "within_one": sum(abs(gap) <= 1 for gap in gaps) / n,
        "mean_signed_error": mean(gaps),
        "confusion_matrix_expert_rows_judge_columns": table,
        "expert_score_counts": reference_counts,
        "judge_score_counts": judge_counts,
        "expert_score_variance": pvariance(reference),
    }


def bootstrap(reference, judged, publishers, samples, seed, baseline=None):
    """Resample complete article records within publisher, keeping runs paired."""
    groups = defaultdict(list)
    for index, publisher in enumerate(publishers):
        groups[publisher].append(index)
    rng = random.Random(seed)
    mae, qwk, delta = [], [], []
    for _ in range(samples):
        indices = [
            i for publisher in sorted(groups)
            for i in rng.choices(groups[publisher], k=len(groups[publisher]))
        ]
        drawn_reference = [reference[i] for i in indices]
        result = metrics(drawn_reference, [judged[i] for i in indices])
        mae.append(result["mae"])
        if result["qwk"] is not None:
            qwk.append(result["qwk"])
        if baseline is not None:
            baseline_mae = mean(abs(baseline[i] - reference[i]) for i in indices)
            delta.append(result["mae"] - baseline_mae)

    def interval(values):
        if len(values) < 2:
            return None
        cuts = quantiles(values, n=40, method="inclusive")
        return [cuts[0], cuts[-1]]

    return {
        "samples": samples,
        "seed": seed,
        "mae_95_ci": interval(mae),
        "qwk_95_ci_defined_draws": interval(qwk),
        "undefined_qwk_draws": samples - len(qwk),
        "mae_change_vs_baseline_95_ci": interval(delta),
    }


def score(args):
    benchmark = read_benchmark(args.benchmark)
    expected = {key: row for key, row in benchmark.items() if row.split == args.split}
    if not expected:
        raise ValueError(f"No benchmark records for {args.split}")
    cal = [row.expert_median for row in benchmark.values() if row.split == "cal"]
    if not cal:
        raise ValueError("Calibration scores are required to select the constant reference")
    frequencies = Counter(cal)
    constant = min(LEVELS, key=lambda level: (
        sum(abs(level - label) for label in cal), -frequencies[level], level
    ))

    predictions, settings = read_predictions(args.predictions, expected, args.split)
    keys = sorted(expected)
    reference = [expected[key].expert_median for key in keys]
    judged = [predictions[key].result.score for key in keys]
    publishers = [predictions[key].publisher for key in keys]
    baseline_scores = None
    if args.baseline:
        baseline, baseline_settings = read_predictions(args.baseline, expected, args.split)
        if settings[1:] != baseline_settings[1:]:
            raise ValueError("Prompt comparison requires matching judge model and settings")
        if any(baseline[key].publisher != predictions[key].publisher for key in keys):
            raise ValueError("Publisher metadata differs between runs")
        baseline_scores = [baseline[key].result.score for key in keys]

    result = {
        "split": args.split,
        "article_count": len(keys),
        "publishers": dict(Counter(publishers)),
        "prediction_file": str(args.predictions),
        "prompt_sha256": settings[0],
        "judge_model": settings[1],
        "response_model": settings[2],
        "metrics": metrics(reference, judged),
        "constant_score_reference": {"score": constant, "metrics": metrics(
            reference, [constant] * len(keys)
        )},
        "bootstrap": bootstrap(
            reference, judged, publishers, args.resamples, args.seed, baseline_scores
        ),
        "articles": [
            {
                "article_id": key[0],
                "model_id": key[1],
                "expert_ratings": expected[key].expert_ratings,
                "expert_median": expected[key].expert_median,
                "judge_score": predictions[key].result.score,
                "rationale": predictions[key].result.rationale,
            }
            for key in keys
        ],
    }
    if baseline_scores is not None:
        baseline_metrics = metrics(reference, baseline_scores)
        result["baseline"] = {
            "prediction_file": str(args.baseline),
            "prompt_sha256": baseline_settings[0],
            "metrics": baseline_metrics,
            "bootstrap": bootstrap(reference, baseline_scores, publishers, args.resamples, args.seed),
        }
        result["mae_change_vs_baseline"] = result["metrics"]["mae"] - baseline_metrics["mae"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("cal", "eval"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, help="Optional frozen v1 predictions for the same split")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.resamples < 2:
        parser.error("--resamples must be at least 2")
    try:
        result = score(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Scoring failed: {exc}\n")
    statistics = result["metrics"]
    print(f"{args.split} | {result['article_count']} articles | MAE {statistics['mae']:.3f} | "
          f"QWK {statistics['qwk']} | exact {statistics['exact_match']:.1%}")
    print(f"Saved metrics and article-level comparisons to {args.output}")


if __name__ == "__main__":
    main()
