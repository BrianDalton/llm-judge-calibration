"""Schema for expert reference labels and split assignments."""

from statistics import median
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Score = Annotated[int, Field(strict=True, ge=1, le=5)]


class BenchmarkAnnotation(BaseModel):
    """One summary's expert ratings and its calibration or evaluation split."""

    model_config = ConfigDict(extra="forbid")

    article_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    dimension: Literal["coherence", "fluency", "consistency", "relevance"]
    expert_ratings: tuple[Score, Score, Score]
    expert_median: Score
    split: Literal["cal", "eval"]

    @model_validator(mode="after")
    def check_expert_median(self):
        if self.expert_median != median(self.expert_ratings):
            raise ValueError("expert_median must equal the median of expert_ratings")
        return self
