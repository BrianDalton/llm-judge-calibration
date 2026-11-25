"""Run one factual-consistency prompt on a saved split, using the Responses API.

Run from the repository root with python -m scripts.run_judge --help.
"""

import argparse
import hashlib
import json
from pathlib import Path

from models.judge_output import JudgeOutput
from models.judge_prediction import JudgePrediction
from scripts.judge_data import read_benchmark, read_sources


def judge_article(client, args, prompt, article, summary):
    """Send only the rubric and two texts. Benchmark metadata stays local."""
    response = client.responses.parse(
        model=args.judge_model,
        input=[
            {"role": "developer", "content": prompt},
            {"role": "user", "content": json.dumps(
                {"article": article, "candidate_summary": summary}, ensure_ascii=False
            )},
        ],
        text_format=JudgeOutput,
        reasoning={"effort": args.reasoning_effort},
        max_output_tokens=args.max_output_tokens,
        store=False,
    )
    if response.status != "completed" or response.output_parsed is None or response.usage is None:
        raise ValueError(f"Response {response.id} returned no complete score (status={response.status})")
    return response


def run(args):
    prompt_bytes = args.prompt.read_bytes()
    prompt = prompt_bytes.decode("utf-8").strip()
    if not prompt:
        raise ValueError("Prompt is empty")
    prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
    benchmark = read_benchmark(args.benchmark)
    keys = sorted(key for key, row in benchmark.items() if row.split == args.split)
    if not keys:
        raise ValueError(f"No articles found for split {args.split}")
    keys = keys[:args.limit]
    sources = read_sources(args.input, set(keys))
    print(f"{args.split}: {len(keys)} articles | {args.judge_model} | {args.prompt.name}")
    if args.dry_run:
        return

    # Keep dry runs and offline scoring usable without the API SDK.
    from openai import OpenAI

    client = OpenAI()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for index, (article_id, model_id) in enumerate(keys, start=1):
            article, summary, publisher = sources[article_id, model_id]
            try:
                response = judge_article(client, args, prompt, article, summary)
                record = JudgePrediction(
                    article_id=article_id, model_id=model_id, split=args.split,
                    publisher=publisher, prompt=args.prompt.name, prompt_sha256=prompt_hash,
                    judge_model=args.judge_model, response_model=response.model,
                    reasoning_effort=args.reasoning_effort, max_output_tokens=args.max_output_tokens,
                    response_id=response.id, input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens, result=response.output_parsed,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Stopped at {article_id} after saving {index - 1} of {len(keys)} scores "
                    f"to {args.output}: {exc}"
                ) from exc
            output.write(record.model_dump_json() + "\n")
            output.flush()
            print(f"Scored {index}/{len(keys)}", flush=True)
    print(f"Saved judge scores to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="Paired source and summary JSONL")
    parser.add_argument("--benchmark", required=True, type=Path, help="Benchmark JSONL for split membership")
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("cal", "eval"))
    parser.add_argument("--output", required=True, type=Path, help="New predictions JSONL")
    parser.add_argument("--judge-model", default="gpt-6.2-sol")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, help="Score only the first N calibration articles")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs without calling the API")
    args = parser.parse_args()
    if args.limit is not None and (args.limit < 1 or args.split != "cal"):
        parser.error("--limit requires a positive count and --split cal")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Judge run failed: {exc}\n")


if __name__ == "__main__":
    main()
