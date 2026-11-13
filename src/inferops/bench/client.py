"""One streamed chat completion against an OpenAI-compatible server, with timestamps."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """Timestamps and token counts of one request. Times come from a monotonic clock."""

    started_at: float
    finished_at: float
    first_token_at: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int = 0
    # True when the server sent no usage and streamed chunks were counted instead.
    output_tokens_estimated: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def ttft(self) -> float | None:
        """Time to first token: from sending the request to the first streamed token."""
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.started_at

    @property
    def latency(self) -> float:
        """End-to-end latency: from sending the request to the end of the stream."""
        return self.finished_at - self.started_at

    @property
    def tpot(self) -> float | None:
        """Time per output token after the first one (the first is already counted by TTFT)."""
        if self.first_token_at is None or self.output_tokens < 2:
            return None
        return (self.finished_at - self.first_token_at) / (self.output_tokens - 1)


def _has_text(delta: dict[str, Any]) -> bool:
    """A chunk counts as a token if it carries answer text or reasoning text."""
    return any(delta.get(key) for key in ("content", "reasoning_content", "reasoning"))


async def stream_chat(
    client: httpx.AsyncClient,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    extra_body: dict[str, Any] | None = None,
    timeout_seconds: float = 120.0,
    clock: Clock = time.perf_counter,
) -> RequestRecord:
    """Send one streaming chat completion and time it.

    Failures are returned as a record with `error` set, never raised, so one bad request
    cannot end a whole benchmark run and every failure is still counted.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # Ask the server for real token counts in the last chunk.
        "stream_options": {"include_usage": True},
        **(extra_body or {}),
    }
    first_token_at: float | None = None
    chunks = 0
    usage: dict[str, Any] | None = None
    error: str | None = None

    started_at = clock()
    try:
        async with (
            asyncio.timeout(timeout_seconds),
            client.stream("POST", "chat/completions", json=payload) as response,
        ):
            if response.status_code != 200:
                body = (await response.aread()).decode("utf-8", errors="replace")
                error = f"HTTP {response.status_code}: {body[:200]}"
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        error = f"server error in stream: {str(event['error'])[:200]}"
                        break
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        if _has_text(choice.get("delta") or {}):
                            chunks += 1
                            if first_token_at is None:
                                first_token_at = clock()
    except TimeoutError:
        error = f"timed out after {timeout_seconds:g}s"
    except httpx.HTTPError as exc:
        error = f"{type(exc).__name__}: {exc}"
    except json.JSONDecodeError as exc:
        error = f"malformed stream chunk: {exc}"
    finished_at = clock()

    if error is None and first_token_at is None:
        error = "stream ended without any token"

    completion_tokens = (usage or {}).get("completion_tokens")
    prompt_tokens = (usage or {}).get("prompt_tokens")
    has_usage = isinstance(completion_tokens, int)
    return RequestRecord(
        started_at=started_at,
        finished_at=finished_at,
        first_token_at=first_token_at,
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        output_tokens=completion_tokens if has_usage else chunks,
        output_tokens_estimated=not has_usage,
        error=error,
    )
