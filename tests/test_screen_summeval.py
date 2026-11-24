import json

from scripts.screen_summeval_single_model import screen


def test_screen_counts_duplicates_missing_and_invalid_ratings(tmp_path):
    source = tmp_path / "annotations.jsonl"
    row = {
        "id": "one", "model_id": "M20", "expert_annotations": [
            {"consistency": 2, "relevance": 2},
            {"consistency": 4, "relevance": True},
            {"consistency": 5},
        ],
    }
    source.write_text("\n".join(map(json.dumps, [row, row, {**row, "model_id": "M22"}])))
    result = screen(source, "M20")
    assert result["matched_rows"] == 2
    assert result["unique_summaries"] == 1
    assert result["duplicate_rows_by_source_id"] == {"one": 1}
    assert result["dimensions"]["consistency"]["expert_medians"][4]["count"] == 1
    relevance = result["dimensions"]["relevance"]
    assert relevance["missing_rating_count"] == 1
    assert relevance["invalid_rating_count"] == 1
    assert relevance["reference_count"] == 0
