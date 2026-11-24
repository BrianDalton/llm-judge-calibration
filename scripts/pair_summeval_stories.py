"""Attach local CNN/Daily Mail articles to the bundled SummEval annotations."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "SummEval_data" / "model_annotations.aligned.jsonl"
OUTPUT = ROOT / "local_files" / "model_annotations.aligned.paired.jsonl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("story_root", type=Path, help="Directory containing cnn/ and dailymail/")
    args = parser.parse_args()

    try:
        with ANNOTATIONS.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]

        articles = {}
        for filepath in sorted({row["filepath"] for row in rows}):
            # The release paths start with cnndm/; story_root points at that level.
            story = args.story_root / Path(filepath).relative_to("cnndm")
            content = story.read_text(encoding="utf-8").split("@highlight", 1)[0]
            article = " ".join(line.strip() for line in content.splitlines() if line.strip())
            if not article:
                raise ValueError(f"Empty article: {story}")
            articles[filepath] = article

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("w", encoding="utf-8") as output:
            for row in rows:
                row["text"] = articles[row["filepath"]]
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"SummEval pairing failed: {exc}\n")

    print(f"Paired {len(rows):,} annotated summaries with {len(articles):,} stories.")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
