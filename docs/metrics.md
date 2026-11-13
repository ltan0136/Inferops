# Metrics

Every number `inferops bench` reports is defined here. The definitions are covered by
`tests/test_metrics.py` with exact values.

## Per request

All timestamps come from a monotonic clock on the client.

| Metric | Definition | What it tells you |
| --- | --- | --- |
| TTFT (time to first token) | first streamed token − request sent | Queueing plus prefill. This is what a user feels as "it started answering". |
| TPOT (time per output token) | (end of stream − first token) / (output tokens − 1) | Decode speed once the request is running. Undefined for a one-token answer. |
| Latency | end of stream − request sent | The whole request. Roughly TTFT + TPOT × (output tokens − 1). |

A token is any streamed chunk that carries answer text or reasoning text, so models that think
before answering are timed from their first reasoning token.

Output token counts come from the `usage` block the server sends at the end of the stream. If a
server sends no usage, streamed chunks are counted instead and the report says so, because one
chunk is not always one token.

## Per concurrency level

| Metric | Definition |
| --- | --- |
| Request throughput | successful requests / measured seconds |
| Output token throughput | output tokens of successful requests / measured seconds |
| p50 / p90 / p99 | percentiles over successful requests, linear interpolation between ranks |
| Failed | requests that returned an error, timed out, or produced no token |

Measured time runs from the start of the measurement window to the last completion. Requests
that start during warmup are sent but not recorded. Requests that start inside the window are
recorded even if they finish after it ends: dropping them would hide the slowest requests.

Failed requests never enter latency or throughput. A level with no successful request aborts
the run instead of reporting zeros.

## Server-side counterparts

vLLM exports its own view of the same quantities on `/metrics`. Client and server numbers
should move together; the gap between them is network and HTTP overhead.

| This repo | vLLM metric |
| --- | --- |
| TTFT | `vllm:time_to_first_token_seconds` |
| TPOT | `vllm:request_time_per_output_token_seconds` |
| Latency | `vllm:e2e_request_latency_seconds` |
| Requests in flight | `vllm:num_requests_running`, `vllm:num_requests_waiting` |
| KV cache pressure | `vllm:kv_cache_usage_perc` |
| Prefix cache hit rate | `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` |

## Known limits

- **Closed loop only.** Each worker sends its next request when the previous one ends, so the
  offered load drops when the server slows down. Latency under a fixed arrival rate (open loop)
  is worse than these numbers suggest once the server is saturated. Open-loop load is planned
  for M2 and will come from a standard tool rather than from this one.
- **Prompt length is set in words.** The real prompt token count is read from the server and
  reported as `mean_prompt_tokens` in `summary.json`.
- **One client process.** The load generator shares one event loop. At very high concurrency the
  client itself can become the bottleneck; cross-check with GuideLLM (`make guidellm`) before
  trusting a curve that flattens.
