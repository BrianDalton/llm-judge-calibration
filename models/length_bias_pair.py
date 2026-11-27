"""Original and padded summaries for a controlled length comparison."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LengthBiasPair(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    article_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    original_summary: str = Field(min_length=1)
    padded_summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_padding(self):
        if not self.padded_summary.endswith(" " + self.original_summary):
            raise ValueError("Padded summary must prepend text to the unchanged original summary")
        return self
