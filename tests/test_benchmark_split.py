import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from models.benchmark_annotation import BenchmarkAnnotation
from scripts.build_benchmark_split import load_examples

ROOT = Path(__file__).resolve().parents[1]


def make_row(number, publisher="cnn", score=2):
    return {
        "model_id": "M20",
        "id": f"article-{number}",
        "text": f"Source article {number}.",
        "decoded": "A candidate summary.",
        "filepath": f"cnndm/{publisher}/stories/{number}.story",
        "expert_annotations": [{"consistency": score}] * 3,
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_split_cli_preserves_labels_and_seeded_strata(tmp_path):
    rows = [make_row(n) for n in range(3)]
    rows += [make_row(n, "dailymail", 5) for n in range(3, 8)]
    source = tmp_path / "paired.jsonl"
    output = tmp_path / "benchmark.jsonl"
    command = [
        sys.executable, str(ROOT / "scripts/build_benchmark_split.py"),
        "--input", str(source), "--output", str(output), "--model", "M20",
        "--dimension", "consistency", "--cal-size", "3", "--seed", "42",
    ]

    write_jsonl(source, rows)
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    first_output = output.read_text()
    records = [BenchmarkAnnotation.model_validate_json(line) for line in first_output.splitlines()]
    assert len(records) == len({row.article_id for row in records}) == 8
    assert Counter(row.split for row in records) == {"cal": 3, "eval": 5}
    assert Counter(row.expert_median for row in records if row.split == "cal") == {2: 1, 5: 2}
    assert {row.article_id: list(row.expert_ratings) for row in records} == {
        row["id"]: [expert["consistency"] for expert in row["expert_annotations"]] for row in rows
    }
    assert all(set(json.loads(line)) == set(BenchmarkAnnotation.model_fields)
               for line in first_output.splitlines())

    write_jsonl(source, reversed(rows))
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert output.read_text() == first_output


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"id": "article-0"}, "Duplicate article ID"),
        ({"text": "  Source  article  0. "}, "Duplicate article text"),
        ({"expert_annotations": [{"consistency": True}] * 3}, "Expected three valid"),
    ],
)
def test_split_rejects_bad_input(tmp_path, change, message):
    source = tmp_path / "paired.jsonl"
    write_jsonl(source, [make_row(0), {**make_row(1), **change}])
    with pytest.raises(ValueError, match=message):
        load_examples(source, "M20", "consistency")


@pytest.mark.parametrize(
    "change",
    [
        {"expert_median": 5},
        {"expert_ratings": [True, 2, 3]},
        {"expert_ratings": [1, 2]},
        {"expert_ratings": [1, 2, 6]},
        {"split": "test"},
        {"text": "Raw source text"},
    ],
)
def test_benchmark_schema_rejects_invalid_labels(change):
    valid = {
        "article_id": "article-0", "model_id": "M20", "dimension": "consistency",
        "expert_ratings": [1, 2, 3], "expert_median": 2, "split": "cal",
    }
    with pytest.raises(ValidationError):
        BenchmarkAnnotation.model_validate({**valid, **change})
