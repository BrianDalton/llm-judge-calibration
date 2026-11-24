"""Measure expert agreement for one SummEval model and rubric dimension.

Usage:
    python3 scripts/analyze_expert_agreement.py --input ANNOTATIONS.jsonl \
        --model MODEL_ID --dimension DIMENSION

--input is the path to a SummEval annotations JSONL file.
--model selects the exact model ID in the file.
--dimension selects coherence, fluency, consistency, or relevance.

Prints interval Krippendorff's alpha as the primary result, ordinal alpha as
a sensitivity check, and exact and within-one-point expert-pair agreement.
Only summaries with three valid expert scores for the selected dimension
are included. The script reports the number excluded.
"""

import argparse
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

# Allow direct invocation as well as imports from the scripts package.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.screen_summeval_single_model import (
    DIMENSIONS,
    complete_expert_triplets,
    screen,
)


def agreement(triplets):
    """Compute expert agreement from three valid scores per summary.

    Pass a nonempty collection of triplets, one per summary. Return interval
    alpha, ordinal alpha, the exact-match count, the within-one-point count
    (including exact matches), and the total number of expert pairs, in that
    order. Alpha is None when there is no expected disagreement.
    """
    # Pool scores across summaries to count how often each score level occurs.
    marginals = Counter(score for ratings in triplets for score in ratings)
    levels = sorted(marginals)

    # Each summary contributes its three unordered expert pairs to observed
    # disagreement. Sort the scores so b - a is always nonnegative.
    pairs = [pair for ratings in triplets for pair in combinations(sorted(ratings), 2)]

    # For ordinal alpha, imagine the pooled ratings sorted into one block per
    # score. A block's midpoint is the number of ratings in lower blocks plus
    # half the number in that block. These midpoints determine score distances.
    midpoint = {}
    cumulative = 0
    for score in levels:
        midpoint[score] = cumulative + marginals[score] / 2
        cumulative += marginals[score]

    def alpha(coordinates):
        # Use the same squared-distance rule for both variants. Only the
        # mapping from scores to coordinates changes.
        observed = sum((coordinates[b] - coordinates[a]) ** 2 for a, b in pairs) / len(pairs)

        # Expected disagreement uses all unordered pairs of pooled ratings.
        # There are count[a] * count[b] cross-score pairs. Equal-score pairs
        # add zero to the numerator and still count in the denominator.
        expected = sum(
            marginals[a] * marginals[b] * (coordinates[b] - coordinates[a]) ** 2
            for a, b in combinations(levels, 2)
        ) / (cumulative * (cumulative - 1) / 2)
        return 1 - observed / expected if expected else None

    exact = sum(a == b for a, b in pairs)
    within_one = sum(b - a <= 1 for a, b in pairs)
    # Interval alpha uses the raw score levels. Ordinal alpha uses the pooled
    # distribution's category midpoints. Return interval first.
    return alpha({score: score for score in levels}), alpha(midpoint), exact, within_one, len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="SummEval annotations JSONL")
    parser.add_argument("--model", required=True, help="Exact SummEval model ID")
    parser.add_argument("--dimension", choices=DIMENSIONS, required=True)
    args = parser.parse_args()
    try:
        items = screen(args.input, args.model)["items"]
        triplets = complete_expert_triplets(items, args.dimension)
        if not triplets:
            parser.error(f"No complete expert rating triplets for {args.model} {args.dimension}")

        interval, ordinal, exact, within_one, pairs = agreement(triplets)
        interval_text = f"{interval:.3f}" if interval is not None else "undefined"
        ordinal_text = f"{ordinal:.3f}" if ordinal is not None else "undefined"
        print(f"{args.model} {args.dimension} | {len(triplets)} summaries | "
              f"{len(items) - len(triplets)} excluded | interval alpha {interval_text} | "
              f"ordinal alpha {ordinal_text} | exact {exact}/{pairs} ({exact / pairs:.1%}) | "
              f"within one point {within_one}/{pairs} ({within_one / pairs:.1%})")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Expert agreement analysis failed: {exc}\n")


if __name__ == "__main__":
    main()
