"""Run the frozen consistency judge on reviewed original/padded pairs.

Use python -m scripts.run_length_bias --help from the repo root.
--pairs is a JSONL of LengthBiasPair records. --baseline is the complete
v1 calibration predictions file and supplies the frozen judge settings.
--dry-run validates inputs and reports the call count without using the API.
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from models.length_bias_pair import LengthBiasPair
from models.length_bias_prediction import LengthBiasPrediction
from scripts.judge_data import read_benchmark, read_sources
from scripts.run_judge import judge_article
from scripts.score_judge import read_predictions


def read_pairs(path, baseline):
    pairs = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            pair = LengthBiasPair.model_validate_json(line)
            key = pair.article_id, pair.model_id
            if key in pairs:
                raise ValueError(f"Duplicate length pair for {key}")
            if key not in baseline or baseline[key].result.score >= 5:
                raise ValueError(f"Expected a calibration summary with a prior v1 score below 5: {key}")
            pairs[key] = pair
    if not pairs:
        raise ValueError("The pairs file has no reviewed examples")
    return pairs


def request_order(keys, seed):
    """Randomize the calls while balancing which variant is scored first."""
    rng = random.Random(seed)
    randomized_keys = sorted(keys)
    rng.shuffle(randomized_keys)
    first_variants = ("original", "padded")
    if rng.randrange(2):
        first_variants = first_variants[::-1]
    pending = {
        key: [first_variants[index % 2], first_variants[(index + 1) % 2]]
        for index, key in enumerate(randomized_keys)
    }
    while pending:
        key = rng.choice(list(pending))
        yield key, pending[key].pop(0)
        if not pending[key]:
            del pending[key]


def summarize(scores):
    """Summarize signed score differences with an article as the unit."""
    differences = [
        {
            "article_id": article_id, "model_id": model_id,
            "original_score": variants["original"], "padded_score": variants["padded"],
            "padded_minus_original": variants["padded"] - variants["original"],
        }
        for (article_id, model_id), variants in sorted(scores.items())
    ]
    gaps = [row["padded_minus_original"] for row in differences]
    counts = Counter(gaps)
    return {
        "pairs": len(gaps),
        "mean_signed_change": mean(gaps),
        "increases": sum(gap > 0 for gap in gaps),
        "decreases": sum(gap < 0 for gap in gaps),
        "ties": counts[0],
        "articles": differences,
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
    baseline, settings = read_predictions(args.baseline, calibration, "cal")
    pairs = read_pairs(args.pairs, baseline)
    sources = read_sources(args.input, set(pairs))
    for key, pair in pairs.items():
        if pair.original_summary != sources[key][1].strip():
            raise ValueError(f"Original summary does not match the paired annotations: {key}")

    prompt_hash, judge_model, response_model, effort, max_tokens = settings
    prompt_bytes = args.prompt.read_bytes()
    if hashlib.sha256(prompt_bytes).hexdigest() != prompt_hash:
        raise ValueError("Prompt file does not match the frozen v1 calibration prompt hash")
    prompt = prompt_bytes.decode("utf-8").strip()
    print(f"{len(pairs)} reviewed calibration pairs | {2 * len(pairs)} calls | {judge_model}")
    if args.dry_run:
        return

    from openai import OpenAI

    client = OpenAI()
    frozen = argparse.Namespace(
        judge_model=judge_model, reasoning_effort=effort, max_output_tokens=max_tokens
    )
    scores = defaultdict(dict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for index, (key, variant) in enumerate(request_order(pairs, args.seed), start=1):
            article, _, publisher = sources[key]
            pair = pairs[key]
            summary = pair.original_summary if variant == "original" else pair.padded_summary
            try:
                response = judge_article(client, frozen, prompt, article, summary, disable_prompt_cache=True)
                cached_tokens = response.usage.input_tokens_details.cached_tokens
                if cached_tokens != 0:
                    raise ValueError(f"Expected no cached input tokens, got {cached_tokens}")
                if response.model != response_model:
                    raise ValueError(f"Expected response model {response_model}, got {response.model}")
                record = LengthBiasPrediction(
                    article_id=key[0], model_id=key[1], split="cal", publisher=publisher,
                    prompt=args.prompt.name, prompt_sha256=prompt_hash, judge_model=judge_model,
                    response_model=response.model, reasoning_effort=effort, max_output_tokens=max_tokens,
                    response_id=response.id, input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens, result=response.output_parsed,
                    variant=variant, summary_sha256=hashlib.sha256(summary.encode()).hexdigest(),
                    order_seed=args.seed, cached_input_tokens=cached_tokens,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Stopped at {key} ({variant}) after saving {index - 1} responses "
                    f"to {args.output}: {exc}"
                ) from exc
            output.write(record.model_dump_json() + "\n")
            output.flush()
            scores[key][variant] = record.result.score
            print(f"Scored {index}/{2 * len(pairs)}", flush=True)

    report = {"order_seed": args.seed, "cached_input_tokens": 0, **summarize(scores)}
    for row in report["articles"]:
        pair = pairs[row["article_id"], row["model_id"]]
        row["original_words"] = len(pair.original_summary.split())
        row["padded_words"] = len(pair.padded_summary.split())
    report["original_scores_at_ceiling"] = sum(row["original_score"] == 5 for row in report["articles"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(f"Padded minus original: {report['mean_signed_change']:+.3f} points | "
          f"{report['increases']} increases, {report['decreases']} decreases, {report['ties']} ties")
    print(f"Saved responses to {args.output} and paired results to {args.report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="Paired source articles JSONL")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True, help="Reviewed original/padded pairs JSONL")
    parser.add_argument("--baseline", type=Path, required=True, help="Complete v1 calibration predictions")
    parser.add_argument("--prompt", type=Path, required=True, help="Frozen v1 prompt")
    parser.add_argument("--output", type=Path, required=True, help="New response JSONL")
    parser.add_argument("--report", type=Path, required=True, help="New paired-results JSON")
    parser.add_argument("--seed", type=int, default=42, help="Seed for request order")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Length bias run failed: {exc}\n")


if __name__ == "__main__":
    main()
