from fractions import Fraction
from itertools import combinations

import pytest

from scripts.analyze_expert_agreement import agreement


@pytest.mark.parametrize(
    ("triplets", "expected"),
    [
        ([[1, 1, 1], [5, 5, 5]], (1, 1, 6, 6, 6)),
        ([[5, 5, 5]], (None, None, 3, 3, 3)),
        ([[1, 2, 5]], (0, 0, 0, 1, 3)),
        # Each item has the same mixed scores, so within-item disagreement
        # exceeds the expected disagreement from the pooled ratings.
        ([[1, 2, 5], [1, 2, 5]], (-0.25, -0.25, 0, 2, 6)),
    ],
)
def test_agreement_boundaries(triplets, expected):
    interval, ordinal, *counts = agreement(triplets)
    assert tuple(counts) == expected[2:]
    for actual, reference in zip((interval, ordinal), expected[:2]):
        if reference is None:
            assert actual is None
        else:
            assert actual == pytest.approx(reference)


def test_agreement_matches_independent_pair_enumeration():
    triplets = [[5, 1, 5], [4, 2, 4], [5, 5, 4], [1, 2, 2]]
    pooled = [score for item in triplets for score in item]
    observed = [pair for item in triplets for pair in combinations(item, 2)]
    expected = list(combinations(pooled, 2))

    def interval_distance(left, right):
        return (left - right) ** 2

    def ordinal_distance(left, right):
        if left == right:
            return 0
        low, high = sorted((left, right))
        interior = sum(low < score < high for score in pooled)
        return (2 * interior + pooled.count(low) + pooled.count(high)) ** 2

    def reference_alpha(distance):
        observed_mean = Fraction(sum(distance(*pair) for pair in observed), len(observed))
        expected_mean = Fraction(sum(distance(*pair) for pair in expected), len(expected))
        return 1 - observed_mean / expected_mean

    interval, ordinal, exact, within_one, pairs = agreement(triplets)
    assert interval == pytest.approx(float(reference_alpha(interval_distance)))
    assert ordinal == pytest.approx(float(reference_alpha(ordinal_distance)))
    assert interval != pytest.approx(ordinal)
    assert (exact, within_one, pairs) == (4, 8, 12)
