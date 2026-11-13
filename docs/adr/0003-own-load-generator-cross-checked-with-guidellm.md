# ADR 0003: An in-repo load generator, cross-checked with GuideLLM

- Status: accepted
- Date: 2026-11-13

## Context

The product of M1 is a set of curves: throughput and latency against concurrency. Their value
depends on trusting the instrument that drew them. Two more constraints apply: the benchmark
path must run in CI without a GPU, and the definitions of TTFT and TPOT should be something
this project can state and test, not inherit.

The original design named GuideLLM as the main benchmark tool.

## Decision

A small asyncio load generator in this repository (`src/inferops/bench/`) is the primary
instrument for v0.1. It does one thing: a closed-loop concurrency sweep against an
OpenAI-compatible server. A mock server with an explainable queueing model exercises it in
tests and CI.

GuideLLM, pinned to 0.8.1, is the reference instrument. `make guidellm` runs the same sweep,
and the two tools have to agree on the shape of the curve before a result is published.

## First cross-check

Both tools against the mock server on 2026-11-13: 20 output tokens per request, 5 seconds per
level, mock capacity 8.

| Streams | `inferops bench` req/s | GuideLLM 0.8.1 req/s | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 3.95 | 3.80 | 4% |
| 4 | 15.33 | 14.40 | 6% |
| 16 | 29.59 | 30.40 | 3% |

They agree within 7% on short runs, and both show throughput saturating past the mock's
capacity. The same comparison on the real server is an open M1 item.

### Addendum (2026-11-13): throughput denominator changed

The `inferops bench` column above divided by the time from the window's start to the last
completion, which understates throughput by up to latency / (window + latency)
([metrics](../metrics.md#per-concurrency-level)). It now divides by the window's length.
Rerun under the same conditions (three runs; GuideLLM not rerun):

| Streams | `inferops bench` req/s, now | GuideLLM 0.8.1 req/s (above) | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 4.00 | 3.80 | 5% |
| 4 | 16.00 | 14.40 | 11% |
| 16 | 30.40 to 32.00 | 30.40 | 0 to 5% |

The new numbers sit at the mock's theoretical rate (4 streams / 0.25 s per request = 16 req/s),
but the 7% agreement no longer holds at 4 streams. Following the rule above, this is a finding
to investigate, not to average: rerun GuideLLM on the mock, check how it computes its request
rate, and settle the gap before the real-server cross-check.

## Alternatives considered

| Option | Good at | Why not the only tool |
| --- | --- | --- |
| GuideLLM alone | Rate sweeps, saturation detection, HTML reports | Installs about 100 packages (5.6 GB with torch) and downloads a tokenizer, which is heavy for CI; the instrument stays a black box. |
| AIPerf | NVIDIA's benchmark tool, detailed token metrics | Same trade-off as GuideLLM; no reason to hold two external references. |
| inference-perf | Benchmarks at the gateway level on Kubernetes | Fits M2, where the gateway is the thing under test. |
| Locust or k6 | General HTTP load | No token-level metrics without writing them, which is the same work as this tool. |

## Consequences

- The metric definitions are owned here (`docs/metrics.md`) and tested with exact numbers.
- The tool stays small on purpose. Open-loop arrival patterns, needed in M2, come from
  GuideLLM or inference-perf instead of growing this one.
- If the two instruments disagree, that is a finding to investigate, not a number to average.
- This decision is revisited at M2: if the in-repo tool has stopped teaching anything, it
  shrinks to the smoke test and the standard tools take over.
