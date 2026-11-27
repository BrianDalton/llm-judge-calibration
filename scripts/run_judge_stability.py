"""Score every calibration summary three times with the frozen judge.

Run from the repository root with python -m scripts.run_judge_stability.
Pass --input (paired articles), --benchmark (split assignments), --baseline
(complete v1 calibration scores), and --prompt (the original prompt).
The baseline supplies settings only. All three scores per summary are new.
--output saves response JSONL and --report saves the stability metrics.
Use --dry-run to validate inputs and show the call count without the API.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

from models.stability_prediction import StabilityPrediction
from scripts.judge_data import read_benchmark, read_sources
from scripts.run_judge import judge_article
from scripts.score_judge import read_predictions

PASSES = (1, 2, 3)


def request_order(keys, seed):
    """Shuffle all article IDs independently before each complete pass."""
    rng = random.Random(seed)
    ordered_keys = sorted(keys)
    for pass_number in PASSES:
        shuffled = ordered_keys.copy()
        rng.shuffle(shuffled)
        for key in shuffled:
            yield pass_number, key


def summarize(scores):
    """Summarize three scores per summary and use summaries as the main denominator."""
    if not scores or any(set(by_pass) != set(PASSES) for by_pass in scores.values()):
        raise ValueError("Stability metrics require all three scores for every summary")
    spreads = Counter()
    exact_pairs = 0
    articles = []
    for (article_id, model_id), by_pass in sorted(scores.items()):
        values = [by_pass[pass_number] for pass_number in PASSES]
        spread = max(values) - min(values)
        spreads[spread] += 1
        exact_pairs += sum(first == second for first, second in combinations(values, 2))
        articles.append({
            "article_id": article_id, "model_id": model_id,
            "scores_by_pass": values, "score_spread": spread,
        })
    count = len(articles)
    pair_count = count * len(tuple(combinations(PASSES, 2)))
    return {
        "summary_count": count,
        "passes": len(PASSES),
        "all_three_match": spreads[0],
        "all_three_match_rate": spreads[0] / count,
        "score_spread_counts": {spread: spreads[spread] for spread in range(5)},
        "spread_two_or_more": sum(spreads[spread] for spread in range(2, 5)),
        "exact_score_pairs": exact_pairs,
        "score_pairs": pair_count,
        "exact_pair_agreement": exact_pairs / pair_count,
        "articles": articles,
    }


def run(args):
    if not args.dry_run:
        destinations = {args.output.resolve(), args.report.resolve()}
        if len(destinations) != 2:
            raise ValueError("--output and --report must be different files")
        for path in destinations:
            if path.exists():
                raise FileExistsError(f"Choose a new output path: {path}")

    benchmark = read_benchmark(args.benchmark)
    calibration = {key: row for key, row in benchmark.items() if row.split == "cal"}
    if not calibration:
        raise ValueError("No calibration summaries found")
    _, settings = read_predictions(args.baseline, calibration, "cal")
    prompt_hash, judge_model, response_model, effort, max_tokens = settings
    prompt_bytes = args.prompt.read_bytes()
    if hashlib.sha256(prompt_bytes).hexdigest() != prompt_hash:
        raise ValueError("Prompt file does not match the frozen v1 calibration prompt hash")
    prompt = prompt_bytes.decode("utf-8").strip()
    sources = read_sources(args.input, set(calibration))
    call_count = len(calibration) * len(PASSES)
    print(f"{len(calibration)} calibration summaries | {len(PASSES)} fresh passes | "
          f"{call_count} calls | {judge_model}")
    if args.dry_run:
        return

    from openai import OpenAI

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    frozen = argparse.Namespace(
        judge_model=judge_model, reasoning_effort=effort, max_output_tokens=max_tokens
    )
    scores = {key: {} for key in calibration}
    response_ids = set()
    with OpenAI() as client, args.output.open("x", encoding="utf-8") as output:
        for index, (pass_number, key) in enumerate(request_order(calibration, args.seed), start=1):
            article, summary, publisher = sources[key]
            try:
                response = judge_article(client, frozen, prompt, article, summary, disable_prompt_cache=True)
                cached = response.usage.input_tokens_details.cached_tokens
                if cached != 0:
                    raise ValueError(f"Expected no cached input tokens, got {cached}")
                if response.model != response_model:
                    raise ValueError(f"Expected response model {response_model}, got {response.model}")
                if response.id in response_ids:
                    raise ValueError(f"Duplicate API response ID: {response.id}")
                record = StabilityPrediction(
                    article_id=key[0], model_id=key[1], split="cal", publisher=publisher,
                    prompt=args.prompt.name, prompt_sha256=prompt_hash, judge_model=judge_model,
                    response_model=response.model, reasoning_effort=effort, max_output_tokens=max_tokens,
                    response_id=response.id, input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens, result=response.output_parsed,
                    pass_number=pass_number, article_sha256=hashlib.sha256(article.encode()).hexdigest(),
                    summary_sha256=hashlib.sha256(summary.encode()).hexdigest(), cached_input_tokens=cached,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Stopped at pass {pass_number} for {key} after saving {index - 1} of "
                    f"{call_count} responses to {args.output}: {exc}"
                ) from exc
            output.write(record.model_dump_json() + "\n")
            output.flush()
            scores[key][pass_number] = record.result.score
            response_ids.add(response.id)
            print(f"Scored {index}/{call_count} (pass {pass_number})", flush=True)

    report = {
        "order_seed": args.seed, "judge_model": judge_model, "response_model": response_model,
        "reasoning_effort": effort, "max_output_tokens": max_tokens, "prompt_sha256": prompt_hash,
        "response_count": len(response_ids), "cached_input_tokens": 0, **summarize(scores),
    }
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(f"All three scores matched: {report['all_three_match']}/{report['summary_count']} | "
          f"Spread of two or more points: {report['spread_two_or_more']} | "
          f"Exact score-pair agreement: {report['exact_score_pairs']}/{report['score_pairs']}")
    print(f"Saved responses to {args.output} and stability metrics to {args.report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="Paired source and summary JSONL")
    parser.add_argument("--benchmark", type=Path, required=True, help="Benchmark JSONL with split assignments")
    parser.add_argument("--baseline", type=Path, required=True, help="Complete v1 calibration predictions")
    parser.add_argument("--prompt", type=Path, required=True, help="Frozen v1 prompt")
    parser.add_argument("--output", type=Path, required=True, help="New response JSONL")
    parser.add_argument("--report", type=Path, required=True, help="New stability metrics JSON")
    parser.add_argument("--seed", type=int, default=42, help="Seed for request order")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Judge stability run failed: {exc}\n")


if __name__ == "__main__":
    main()
