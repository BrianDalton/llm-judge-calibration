"""Structured response from the judge."""

from pydantic import BaseModel, ConfigDict, Field

from models.benchmark_annotation import Score


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    score: Score
    rationale: str = Field(min_length=1)
