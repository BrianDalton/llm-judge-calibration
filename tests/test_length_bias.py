import hashlib
import json
import sys
from argparse import Namespace
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from models.judge_output import JudgeOutput
from models.length_bias_pair import LengthBiasPair
from scripts.run_length_bias import request_order, run, summarize

PAIR = {
    "article_id": "a",
    "model_id": "M20",
    "original_summary": "The launch happened on Monday.",
    "padded_summary": "The launch took place on Monday. The launch happened on Monday.",
}


def test_request_order_is_reproducible_and_balanced():
    keys = [(str(i), "M20") for i in range(15)]
    order = list(request_order(keys, 42))
    assert order == list(request_order(reversed(keys), 42))
    assert Counter(order) == {(key, variant): 1 for key in keys for variant in ("original", "padded")}
    first = {}
    for key, variant in order:
        first.setdefault(key, variant)
    assert sorted(Counter(first.values()).values()) == [7, 8]


def test_signed_changes_use_padded_minus_original():
    result = summarize({
        ("a", "M20"): {"original": 2, "padded": 4},
        ("b", "M20"): {"original": 3, "padded": 2},
        ("c", "M20"): {"original": 3, "padded": 3},
    })
    assert result["mean_signed_change"] == pytest.approx(1 / 3)
    assert (result["increases"], result["decreases"], result["ties"]) == (1, 1, 1)


@pytest.mark.parametrize("padded", [
    PAIR["original_summary"],
    "The launch took place on a Monday.",
])
def test_pair_requires_padding_and_preserves_original(padded):
    with pytest.raises(ValidationError):
        LengthBiasPair.model_validate({**PAIR, "padded_summary": padded})


@pytest.fixture
def bias_files(tmp_path):
    paths = {name: tmp_path / name for name in (
        "benchmark", "input", "pairs", "baseline", "prompt", "output", "report"
    )}
    paths["prompt"].write_text("Frozen consistency rubric.")
    paths["pairs"].write_text(json.dumps(PAIR))
    paths["benchmark"].write_text(json.dumps({
        "article_id": "a", "model_id": "M20", "dimension": "consistency",
        "split": "cal", "expert_ratings": [2, 2, 3], "expert_median": 2,
    }))
    paths["input"].write_text(json.dumps({
        "id": "a", "model_id": "M20", "text": "The launch happened on Tuesday.",
        "decoded": PAIR["original_summary"], "filepath": "cnndm/cnn/stories/x",
    }))
    paths["baseline"].write_text(json.dumps({
        "article_id": "a", "model_id": "M20", "split": "cal", "publisher": "cnn",
        "prompt": "prompt", "prompt_sha256": hashlib.sha256(paths["prompt"].read_bytes()).hexdigest(),
        "judge_model": "frozen-model", "response_model": "frozen-model",
        "reasoning_effort": "low", "max_output_tokens": 2048, "response_id": "old",
        "input_tokens": 10, "output_tokens": 10, "result": {"score": 2, "rationale": "Old rationale."},
    }))
    return Namespace(**paths, seed=42, dry_run=False)


@pytest.mark.parametrize("cached_tokens", [0, 8])
def test_runner_keeps_conditions_paired_and_uses_frozen_settings(
    bias_files, monkeypatch, cached_tokens
):
    client = Mock()

    def respond(**kwargs):
        summary = json.loads(kwargs["input"][1]["content"])["candidate_summary"]
        score = 3 if summary == PAIR["padded_summary"] else 2
        return SimpleNamespace(
            status="completed", output_parsed=JudgeOutput(score=score, rationale="Rationale."),
            usage=SimpleNamespace(input_tokens=10, output_tokens=10,
                                  input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens)),
            id=f"response-{score}",
            model="frozen-model",
        )

    client.responses.parse.side_effect = respond
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))
    if cached_tokens:
        with pytest.raises(RuntimeError, match="Expected no cached input tokens"):
            run(bias_files)
        assert not bias_files.report.exists()
        return
    run(bias_files)
    requests = [call.kwargs for call in client.responses.parse.call_args_list]
    assert len(requests) == 2
    assert all(r["extra_body"] == {"prompt_cache_options": {"mode": "explicit"}} for r in requests)
    assert all(r["store"] is False and "previous_response_id" not in r for r in requests)
    assert all(json.loads(line)["cached_input_tokens"] == 0
               for line in bias_files.output.read_text().splitlines())
    assert all(r["model"] == "frozen-model" and r["reasoning"] == {"effort": "low"} for r in requests)
    payloads = [json.loads(r["input"][1]["content"]) for r in requests]
    assert all(payload["article"] == "The launch happened on Tuesday." for payload in payloads)
    assert all(set(payload) == {"article", "candidate_summary"} for payload in payloads)
    assert {payload["candidate_summary"] for payload in payloads} == {
        PAIR["padded_summary"], PAIR["original_summary"]
    }
    assert json.loads(bias_files.report.read_text())["mean_signed_change"] == 1


@pytest.mark.parametrize(("file", "old", "new", "message"), [
    ("benchmark", '"cal"', '"eval"', "Unexpected prediction"),
    ("input", "The launch happened on Monday.", "Different original.", "Original summary does not match"),
    ("baseline", '"score": 2', '"score": 5', "prior v1 score below 5"),
    ("prompt", "Frozen", "Changed", "frozen v1 calibration prompt hash"),
])
def test_preflight_rejects_invalid_inputs(bias_files, file, old, new, message):
    path = getattr(bias_files, file)
    path.write_text(path.read_text().replace(old, new))
    bias_files.dry_run = True
    with pytest.raises(ValueError, match=message):
        run(bias_files)
    assert not bias_files.output.exists()
