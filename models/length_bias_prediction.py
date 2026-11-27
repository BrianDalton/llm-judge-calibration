"""One recorded judge response for a matched length variant."""

from typing import Literal

from models.judge_prediction import JudgePrediction


class LengthBiasPrediction(JudgePrediction):
    variant: Literal["original", "padded"]
    summary_sha256: str
    order_seed: int
    cached_input_tokens: Literal[0]
