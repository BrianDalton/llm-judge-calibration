"""A validated score and the metadata needed to reproduce its request."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from models.judge_output import JudgeOutput


class JudgePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    article_id: str
    model_id: str
    split: Literal["cal", "eval"]
    publisher: Literal["cnn", "dailymail"]
    prompt: str
    prompt_sha256: str
    judge_model: str
    response_model: str
    reasoning_effort: str
    max_output_tokens: int = Field(gt=0)
    response_id: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    result: JudgeOutput
