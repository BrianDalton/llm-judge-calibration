"""Rank SummEval model-dimension candidates using the released expert ratings."""

import json
import sys
from pathlib import Path

# Allow direct invocation as well as imports from the scripts package.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_expert_agreement import agreement
from scripts.screen_summeval_single_model import (
    DIMENSIONS,
    ROOT,
    complete_expert_triplets,
    screen,
)

INPUT = ROOT / "SummEval_data" / "model_annotations.aligned.jsonl"
OUTPUT = ROOT / "local_files" / "summeval_model_screening.json"
MAX_MODAL_SHARE = 0.80


def main():
    with INPUT.open(encoding="utf-8") as source:
        models = sorted({json.loads(line)["model_id"] for line in source if line.strip()})

    rows = []
    for model in models:
        report = screen(INPUT, model)
        for dimension in DIMENSIONS:
            scores = report["dimensions"][dimension]
            triplets = complete_expert_triplets(report["items"], dimension)
            interval, ordinal, exact, within_one, pairs = (
                agreement(triplets) if triplets else (None, None, 0, 0, 0)
            )
            distribution = scores["expert_medians"]
            modal_share = (
                max(entry["count"] for entry in distribution.values()) / len(triplets)
                if triplets else None
            )
            rows.append({
                "model_id": model,
                "dimension": dimension,
                **scores,
                "interval_alpha": interval,
                "ordinal_alpha": ordinal,
                "expert_pairs": pairs,
                "exact_pair_matches": exact,
                "pair_matches_within_one": within_one,
                "modal_median_share": modal_share,
            })

    eligible = [
        row for row in rows if row["interval_alpha"] is not None
        and row["modal_median_share"] <= MAX_MODAL_SHARE
    ]
    eligible.sort(key=lambda row: (-row["interval_alpha"], row["modal_median_share"]))
    best_by_model = {}
    for row in eligible:
        best_by_model.setdefault(row["model_id"], row)
    top_three = list(best_by_model.values())[:3]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "model_count": len(models),
        "max_modal_median_share": MAX_MODAL_SHARE,
        "top_three": top_three,
        "results": rows,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"Screened {len(models)} models and {len(rows)} model-dimension pairs.")
    print(f"Top distinct models by interval alpha, with modal share ≤ {MAX_MODAL_SHARE:.0%}:")
    for row in top_three:
        print(f"{row['model_id']} {row['dimension']}: "
              f"interval alpha {row['interval_alpha']:.3f}, "
              f"ordinal alpha {row['ordinal_alpha']:.3f}, "
              f"modal share {row['modal_median_share']:.0%}")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
