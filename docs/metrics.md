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
| Request throughput | successful requests started in the window / window length |
| Output token throughput | output tokens of those requests / window length |
| p50 / p90 / p99 | percentiles over successful requests, linear interpolation between ranks |
| Failed | requests that returned an error, timed out, or produced no token |

Each level runs a warmup, then a measurement window of `duration_seconds`. Requests that start
during warmup are sent but not recorded. Requests that start inside the window are recorded
even if they finish after it ends: dropping them would hide the slowest requests.

Throughput divides by the window's length, not by the time to the last completion. In a closed
loop a worker starts its next request the moment the previous one ends, so requests start at
the rate they finish. After the window closes no new request starts and the level drains with
fewer than `concurrency` requests in flight; counting that drain would understate throughput
by up to latency / (window + latency). The error is largest past saturation, where queueing
makes latency long, so it would bend the curve down exactly where the knee is. On a mock with
capacity 8 and about 1 s per request, 16 streams read 28% low in a 3 s window and 11% low in a
12 s window; divided by the window, both read 2% low.

Counting whole requests in a fixed window has a resolution of one request per worker, so a
level can be off by up to latency / window. On a real server latencies vary, workers drift out
of step and the errors average out; keep the window many times longer than one request's
latency. The mock's timings are exact, so its levels visibly step in multiples of
concurrency / window from run to run.

Failed requests never enter latency or throughput. A level with no successful request stops
the run instead of reporting zeros; the levels already finished are still written, and
`summary.json` marks the run as incomplete with the reason.

## Run metadata

`summary.json` starts with a `run` block: the `inferops` version, the start time (UTC), the
version the server reports at `GET /version` (vLLM serves it; absent if the server does not),
whether the sweep completed, and why it stopped if it did not.

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
