"""Command line entry point: `inferops bench` and `inferops mock-server`."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import yaml

from inferops import __version__
from inferops.bench.loadgen import BenchmarkError, LevelResult, run_scenario
from inferops.bench.metrics import summarize
from inferops.bench.report import render_markdown, write_run
from inferops.bench.scenario import Scenario, load_scenario

_MOCK_HINT = 'the mock server needs extra packages: pip install "inferops[mock]"'


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _with_target(scenario: Scenario, **changes: str) -> Scenario:
    target = scenario.target.model_copy(update=changes)
    return scenario.model_copy(update={"target": target})


def _report_level(result: LevelResult) -> None:
    ok = sum(1 for record in result.records if record.ok)
    failed = len(result.records) - ok
    print(f"concurrency {result.concurrency}: {ok} ok, {failed} failed", file=sys.stderr)


async def _run(
    scenario: Scenario, api_key: str | None, *, spawn_mock: bool
) -> tuple[Scenario, list[LevelResult]]:
    if not spawn_mock:
        return scenario, await run_scenario(scenario, api_key=api_key, on_level=_report_level)

    from inferops.mock.server import running_mock_server

    async with running_mock_server(model=scenario.target.model) as base_url:
        scenario = _with_target(scenario, base_url=base_url)
        return scenario, await run_scenario(scenario, api_key=api_key, on_level=_report_level)


def _mock_available() -> bool:
    """The mock server lives behind an optional extra; the benchmark itself does not need it."""
    try:
        import inferops.mock.server  # noqa: F401
    except ImportError:
        return False
    return True


def _bench(args: argparse.Namespace) -> int:
    try:
        scenario = load_scenario(args.scenario)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return _fail(f"cannot load scenario {args.scenario}: {exc}")

    if args.base_url:
        scenario = _with_target(scenario, base_url=args.base_url)
    if args.model:
        scenario = _with_target(scenario, model=args.model)

    api_key: str | None = None
    if scenario.target.api_key_env:
        api_key = os.environ.get(scenario.target.api_key_env)
        if not api_key:
            return _fail(f"environment variable {scenario.target.api_key_env} is not set")

    if args.spawn_mock and not _mock_available():
        return _fail(_MOCK_HINT)

    try:
        scenario, results = asyncio.run(_run(scenario, api_key, spawn_mock=args.spawn_mock))
    except BenchmarkError as exc:
        return _fail(str(exc))

    summaries = [summarize(r.concurrency, r.records, r.window_start) for r in results]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or Path("results") / f"{scenario.name}-{stamp}"
    write_run(out_dir, scenario, results, summaries)

    print(render_markdown(summaries))
    print(f"wrote {out_dir}")
    return 0


def _mock_server(args: argparse.Namespace) -> int:
    if not _mock_available():
        return _fail(_MOCK_HINT)
    from inferops.mock.server import serve

    serve(
        args.host,
        args.port,
        model=args.model,
        ttft_seconds=args.ttft_ms / 1000,
        token_interval_seconds=args.token_interval_ms / 1000,
        max_concurrency=args.max_concurrency,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferops", description="Benchmark tooling for a single-GPU inference stack."
    )
    parser.add_argument("--version", action="version", version=f"inferops {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    bench = commands.add_parser(
        "bench", help="run a concurrency sweep against an OpenAI-compatible server"
    )
    bench.add_argument("--scenario", type=Path, required=True, help="path to a scenario YAML file")
    bench.add_argument("--base-url", help="override target.base_url, e.g. http://localhost:8000/v1")
    bench.add_argument("--model", help="override target.model")
    bench.add_argument("--out", type=Path, help="output directory (default: results/<name>-<utc>)")
    bench.add_argument(
        "--spawn-mock",
        action="store_true",
        help="start an in-process mock server and benchmark it (no GPU needed)",
    )
    bench.set_defaults(handler=_bench)

    mock = commands.add_parser("mock-server", help="run the mock OpenAI-compatible server")
    mock.add_argument("--host", default="127.0.0.1")
    mock.add_argument("--port", type=int, default=8001)
    mock.add_argument("--model", default="mock-model", help="model name the mock answers to")
    mock.add_argument("--ttft-ms", type=float, default=50.0, help="time to first token")
    mock.add_argument("--token-interval-ms", type=float, default=10.0, help="time between tokens")
    mock.add_argument(
        "--max-concurrency", type=int, default=8, help="requests served at once; the rest queue"
    )
    mock.set_defaults(handler=_mock_server)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
