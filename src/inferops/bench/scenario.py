"""Benchmark scenario: where to send requests, what they look like, and how hard to push."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Target(BaseModel):
    """The server under test."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://localhost:8000/v1"
    # Must match the name the server serves the model under.
    model: str
    # Name of the environment variable that holds the API key. The key itself never goes in a file.
    api_key_env: str | None = None


class RequestShape(BaseModel):
    """What every request looks like."""

    model_config = ConfigDict(extra="forbid")

    # Synthetic prompt length in words. The real token count is read from the server's usage.
    prompt_words: int = Field(default=200, ge=1)
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    # Start every prompt with a unique marker so the server's prefix cache cannot serve it.
    # Turn this off only when the prefix cache is the thing being measured.
    unique_prefix: bool = True
    # Extra fields merged into the request body, e.g. vLLM's `ignore_eos`.
    extra_body: dict[str, Any] = Field(default_factory=dict)


class Load(BaseModel):
    """How hard to push: one closed-loop run per concurrency level."""

    model_config = ConfigDict(extra="forbid")

    concurrency: list[int] = Field(min_length=1)
    warmup_seconds: float = Field(default=5.0, ge=0.0)
    duration_seconds: float = Field(default=30.0, gt=0.0)
    # Upper bound on one whole request, from send to last token.
    request_timeout_seconds: float = Field(default=120.0, gt=0.0)

    @field_validator("concurrency")
    @classmethod
    def _positive_and_unique(cls, levels: list[int]) -> list[int]:
        if any(level < 1 for level in levels):
            raise ValueError("every concurrency level must be >= 1")
        if len(set(levels)) != len(levels):
            raise ValueError("concurrency levels must be unique")
        return levels


class Scenario(BaseModel):
    """A complete, reproducible benchmark definition."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    target: Target
    request: RequestShape = Field(default_factory=RequestShape)
    load: Load


def load_scenario(path: Path) -> Scenario:
    """Read and validate a scenario file. Unknown keys are an error, not ignored."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return Scenario.model_validate(data)
