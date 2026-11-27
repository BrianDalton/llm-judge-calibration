"""One response from the three-pass judge stability experiment."""

from typing import Literal

from models.judge_prediction import JudgePrediction


class StabilityPrediction(JudgePrediction):
    split: Literal["cal"]
    pass_number: Literal[1, 2, 3]
    article_sha256: str
    summary_sha256: str
    cached_input_tokens: Literal[0]
