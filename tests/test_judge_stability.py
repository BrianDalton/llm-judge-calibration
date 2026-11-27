import hashlib
import json
import sys
from argparse import Namespace
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from models.judge_output import JudgeOutput
from scripts.run_judge_stability import request_order, run, summarize


def test_request_order_covers_each_key_once_per_pass():
    keys = [(str(i), "M20") for i in range(30)]
    order = list(request_order(keys, 42))
    assert order == list(request_order(reversed(keys), 42))
    assert [pass_number for pass_number, _ in order] == [1] * 30 + [2] * 30 + [3] * 30
    for pass_number in (1, 2, 3):
        assert sorted(key for number, key in order if number == pass_number) == sorted(keys)


def test_summary_counts_exact_agreement_and_score_spread():
    result = summarize({
        ("a", "M20"): {1: 2, 2: 2, 3: 2},
        ("b", "M20"): {1: 1, 2: 2, 3: 2},
        ("c", "M20"): {1: 1, 2: 3, 3: 5},
    })
    assert result["all_three_match"] == 1
    assert result["all_three_match_rate"] == pytest.approx(1 / 3)
    assert result["score_spread_counts"] == {0: 1, 1: 1, 2: 0, 3: 0, 4: 1}
    assert result["spread_two_or_more"] == 1
    assert result["exact_pair_agreement"] == pytest.approx(4 / 9)
    assert result["articles"][2]["scores_by_pass"] == [1, 3, 5]


def test_summary_rejects_incomplete_passes():
    with pytest.raises(ValueError, match="all three scores"):
        summarize({("a", "M20"): {1: 2, 3: 2}})


@pytest.fixture
def stability_files(tmp_path):
    paths = {name: tmp_path / name for name in (
        "input", "benchmark", "baseline", "prompt", "output", "report"
    )}
    paths["prompt"].write_text("Freeze this rubric.")
    prompt_hash = hashlib.sha256(paths["prompt"].read_bytes()).hexdigest()

    def write(name, rows):
        paths[name].write_text("".join(json.dumps(row) + "\n" for row in rows))

    write("benchmark", [
        {"article_id": key, "model_id": "M20", "dimension": "consistency",
         "expert_ratings": [3, 3, 3], "expert_median": 3, "split": split}
        for key, split in [("a", "cal"), ("b", "cal"), ("c", "eval")]
    ])
    write("input", [
        {"id": key, "model_id": "M20", "filepath": f"cnndm/cnn/stories/{key}",
         "text": f"Article {key}.", "decoded": f"Summary {key}.", "expert_annotations": [1, 2, 3]}
        for key in ("a", "b", "c")
    ])
    write("baseline", [
        {"article_id": key, "model_id": "M20", "split": "cal", "publisher": "cnn",
         "prompt": "prompt", "prompt_sha256": prompt_hash, "judge_model": "frozen-model",
         "response_model": "frozen-model", "reasoning_effort": "low", "max_output_tokens": 2048,
         "response_id": f"baseline-{key}", "input_tokens": 1, "output_tokens": 1,
         "result": {"score": 5, "rationale": "Historical score."}}
        for key in ("a", "b")
    ])
    return Namespace(**paths, seed=42, dry_run=False)


def mock_client(monkeypatch, cached_tokens=0):
    client = MagicMock()
    client.__enter__.return_value = client
    response_count = 0

    def respond(**kwargs):
        nonlocal response_count
        response_count += 1
        summary = json.loads(kwargs["input"][1]["content"])["candidate_summary"]
        return SimpleNamespace(
            id=f"new-{response_count}", status="completed", model="frozen-model",
            output_parsed=JudgeOutput(score=2 if summary == "Summary a." else 3, rationale="Explanation."),
            usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                                  input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens)),
        )

    client.responses.parse.side_effect = respond
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))
    return client


def test_runner_rescores_all_calibration_items_three_times(stability_files, monkeypatch):
    client = mock_client(monkeypatch)
    run(stability_files)
    requests = [call.kwargs for call in client.responses.parse.call_args_list]
    assert len(requests) == 6
    assert all(r["model"] == "frozen-model" and r["reasoning"] == {"effort": "low"} for r in requests)
    assert all(r["extra_body"] == {"prompt_cache_options": {"mode": "explicit"}} for r in requests)
    assert all(r["store"] is False and "previous_response_id" not in r for r in requests)
    payloads = [json.loads(r["input"][1]["content"]) for r in requests]
    assert all(set(payload) == {"article", "candidate_summary"} for payload in payloads)
    assert Counter(payload["candidate_summary"] for payload in payloads) == {"Summary a.": 3, "Summary b.": 3}

    records = [json.loads(line) for line in stability_files.output.read_text().splitlines()]
    assert len({r["response_id"] for r in records}) == 6
    assert {(r["article_id"], r["pass_number"]) for r in records} == {
        (key, number) for key in ("a", "b") for number in (1, 2, 3)
    }
    assert all(r["cached_input_tokens"] == 0 for r in records)
    report = json.loads(stability_files.report.read_text())
    assert report["all_three_match"] == 2
    assert [row["scores_by_pass"] for row in report["articles"]] == [[2, 2, 2], [3, 3, 3]]
    with pytest.raises(FileExistsError):
        run(stability_files)
    assert client.responses.parse.call_count == 6


def test_runner_stops_if_input_cache_was_used(stability_files, monkeypatch):
    mock_client(monkeypatch, cached_tokens=8)
    with pytest.raises(RuntimeError, match="Expected no cached input tokens"):
        run(stability_files)
    assert not stability_files.report.exists()


def test_dry_run_verifies_frozen_prompt_without_api(stability_files, monkeypatch):
    client = mock_client(monkeypatch)
    stability_files.dry_run = True
    run(stability_files)
    client.responses.parse.assert_not_called()
    assert not stability_files.output.exists()
    stability_files.prompt.write_text("Changed rubric.")
    with pytest.raises(ValueError, match="frozen v1 calibration prompt hash"):
        run(stability_files)
