import json
import sys
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from models.judge_output import JudgeOutput
from scripts.run_judge import run
from scripts.score_judge import bootstrap, metrics, score


@pytest.mark.parametrize("bad_score", [0, 6, True, "5", 3.2])
def test_model_rejects_invalid_scores(bad_score):
    with pytest.raises(ValidationError):
        JudgeOutput(score=bad_score, rationale="Some rationale.")


def test_mocked_judge_sends_only_allowed_content_and_scores_by_article(tmp_path, monkeypatch):
    benchmark = tmp_path / "labels.jsonl"
    paired = tmp_path / "paired.jsonl"
    prompt = tmp_path / "prompt.txt"
    output = tmp_path / "run.jsonl"
    labels = [
        {"article_id": "a", "split": "cal", "expert_ratings": [1, 2, 3], "expert_median": 2},
        {"article_id": "b", "split": "cal", "expert_ratings": [4, 5, 5], "expert_median": 5},
        {"article_id": "c", "split": "eval", "expert_ratings": [1, 1, 2], "expert_median": 1},
    ]
    benchmark.write_text("".join(
        json.dumps({**label, "model_id": "M20", "dimension": "consistency"}) + "\n"
        for label in labels
    ))
    paired.write_text("".join(
        json.dumps({
            "id": label["article_id"], "model_id": "M20", "text": f"Article {i}.",
            "decoded": f"Summary {i}.", "filepath": f"cnndm/cnn/stories/{i}.story",
            "expert_annotations": "must never enter request",
        }) + "\n"
        for i, label in reversed(list(enumerate(labels)))
    ))
    prompt.write_text("Test rubric.")
    client = Mock()
    client.responses.parse.side_effect = [
        SimpleNamespace(
            status="completed", output_parsed=JudgeOutput(score=value, rationale="Evidence."),
            usage=SimpleNamespace(input_tokens=50, output_tokens=20), id=f"resp_{value}",
            model="test-model",
        )
        for value in [3, 5]
    ]
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))
    run(Namespace(
        benchmark=benchmark, input=paired, prompt=prompt, output=output, split="cal",
        judge_model="test-model", reasoning_effort="low", max_output_tokens=2048,
        dry_run=False, limit=None,
    ))
    requests = [call.kwargs for call in client.responses.parse.call_args_list]
    assert len(requests) == 2
    assert requests[0]["input"][0] == {"role": "developer", "content": "Test rubric."}
    assert [json.loads(request["input"][1]["content"]) for request in requests] == [
        {"article": "Article 0.", "candidate_summary": "Summary 0."},
        {"article": "Article 1.", "candidate_summary": "Summary 1."},
    ]
    assert requests[0]["text_format"] is JudgeOutput
    assert requests[0]["store"] is False
    args = Namespace(benchmark=benchmark, predictions=output, split="cal",
                     baseline=None, resamples=10, seed=42)
    result = score(args)
    assert result["metrics"]["mae"] == 0.5
    assert result["constant_score_reference"]["score"] == 2
    assert [row["article_id"] for row in result["articles"]] == ["a", "b"]

    original = output.read_text()
    output.write_text(original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Missing 1 predictions"):
        score(args)
    output.write_text(original + original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Duplicate prediction"):
        score(args)


def test_metrics_match_hand_calculation():
    # Observed mean squared gap is 2. A random pairing has expected squared gap 6.
    result = metrics([1, 5], [1, 3])
    assert result["qwk"] == pytest.approx(2 / 3)
    assert result["mae"] == 1
    assert result["mean_signed_error"] == -1
    assert result["exact_match"] == result["within_one"] == 0.5
    assert result["expert_score_variance"] == 4
    assert result["confusion_matrix_expert_rows_judge_columns"][4][2] == 1
    assert metrics([5, 5], [5, 5])["qwk"] is None


def test_paired_bootstrap_counts_undefined_draws_and_preserves_publisher_mix():
    result = bootstrap([1, 5], [1, 5], ["cnn", "dailymail"], 20, 42, baseline=[2, 4])
    assert result["mae_95_ci"] == [0, 0]
    assert result["mae_change_vs_baseline_95_ci"] == [-1, -1]
    assert result["qwk_95_ci_defined_draws"] == [1, 1]
    tied = bootstrap([5, 5], [5, 5], ["cnn", "dailymail"], 20, 42)
    assert tied["qwk_95_ci_defined_draws"] is None
    assert tied["undefined_qwk_draws"] == 20
