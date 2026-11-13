"""A mock OpenAI-compatible chat server with a simple, explainable performance model.

It exists so the benchmark path (client, load generator, metrics, report) can be exercised
without a GPU: in tests, in CI and in demos. It does not imitate a real engine's numbers.

Performance model:

- `max_concurrency` requests are served at once; the rest wait in a queue. Waiting shows up
  as a higher time to first token, the same shape a saturated engine produces.
- A served request waits `ttft_seconds`, emits its first token, then one token every
  `token_interval_seconds` until `max_tokens` tokens are out.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from inferops import __version__

DEFAULT_MODEL = "mock-model"
# What `GET /version` reports, the way vLLM reports its release.
MOCK_VERSION = f"inferops-mock {__version__}"


def _chunk(model: str, delta: dict[str, Any], finish_reason: str | None = None) -> str:
    event = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(event)}\n\n"


def create_app(
    *,
    model: str = DEFAULT_MODEL,
    ttft_seconds: float = 0.05,
    token_interval_seconds: float = 0.01,
    max_concurrency: int = 8,
) -> Starlette:
    """Build the mock server. All timing knobs are explicit so tests can reason about them."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    slots = asyncio.Semaphore(max_concurrency)

    async def health(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def version(_: Request) -> Response:
        return JSONResponse({"version": MOCK_VERSION})

    async def models(_: Request) -> Response:
        return JSONResponse({"object": "list", "data": [{"id": model, "object": "model"}]})

    async def chat_completions(request: Request) -> Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _error(400, "request body is not valid JSON")
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return _error(400, "`messages` must be a list")
        if body.get("model") != model:
            return _error(404, f"The model `{body.get('model')}` does not exist.")

        # Clients use either name: `max_completion_tokens` is the newer OpenAI field.
        max_tokens = body.get("max_completion_tokens", body.get("max_tokens", 16))
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            return _error(400, "`max_tokens` must be a positive integer")
        prompt_tokens = sum(
            len(str(message.get("content", "")).split())
            for message in body["messages"]
            if isinstance(message, dict)
        )
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": max_tokens,
            "total_tokens": prompt_tokens + max_tokens,
        }
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        if not body.get("stream"):
            async with slots:
                await asyncio.sleep(ttft_seconds + token_interval_seconds * (max_tokens - 1))
            return JSONResponse(
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "tok " * max_tokens},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": usage,
                }
            )

        async def stream() -> AsyncIterator[str]:
            # Queue time is spent here, before the first token: it becomes TTFT.
            async with slots:
                await asyncio.sleep(ttft_seconds)
                yield _chunk(model, {"role": "assistant", "content": "tok"})
                for _ in range(max_tokens - 1):
                    await asyncio.sleep(token_interval_seconds)
                    yield _chunk(model, {"content": " tok"})
                yield _chunk(model, {}, finish_reason="length")
            if include_usage:
                final = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [],
                    "usage": usage,
                }
                yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/version", version, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        ]
    )


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error", "code": status}},
        status_code=status,
    )


@asynccontextmanager
async def running_mock_server(**app_options: Any) -> AsyncIterator[str]:
    """Serve the mock on a free local port inside the current event loop.

    Yields the base URL (ending in `/v1`). Used by tests and by `inferops bench --spawn-mock`.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]

    config = uvicorn.Config(create_app(**app_options), log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            if task.done():
                # Startup failed: surface the real exception instead of waiting forever.
                task.result()
                raise RuntimeError("mock server exited before it started")
            await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.should_exit = True
        await task
        sock.close()


def serve(host: str, port: int, **app_options: Any) -> None:
    """Run the mock server in the foreground until interrupted."""
    uvicorn.run(create_app(**app_options), host=host, port=port, log_level="info")
