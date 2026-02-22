#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

# Keep benchmark output concise by default. Must run before enzu imports.
os.environ.setdefault("ENZU_LOGFIRE", "0")
os.environ.setdefault("ENZU_LOGFIRE_CONSOLE", "0")
os.environ.setdefault("ENZU_LOGFIRE_STREAM", "0")
os.environ.setdefault("ENZU_TELEMETRY_STDERR", "0")

from enzu.isolation.container import (
    ContainerConfig,
    ContainerSandbox,
    is_container_available,
)
from enzu.isolation.runner import IsolatedSandbox, SandboxConfig
from enzu.scalability import (
    USLFit,
    fit_usl,
    littles_law as little_law_l,
    percentile,
    relative_error,
)
from enzu.models import Budget, ProviderResult, SuccessCriteria, TaskSpec
from enzu.providers.base import BaseProvider
from enzu.repl.sandbox import PythonSandbox
from enzu.rlm.engine import RLMEngine

ModeName = Literal["inprocess", "subprocess", "container"]
WorkloadName = Literal[
    "tiny_compute",
    "baseline",
    "monty_import",
    "rlm_tiny_compute",
    "rlm_baseline",
    "rlm_monty_import",
    "rlm_subcall",
]


@dataclass
class RequestSample:
    request_id: int
    success: bool
    latency_seconds: float
    error: Optional[str] = None


@dataclass
class PopulationSample:
    timestamp_seconds: float
    in_flight: int


@dataclass
class BenchmarkRun:
    run_model: str
    mode: str
    workload: str
    concurrency: int
    repeat: int
    requests: int
    total_seconds: float
    throughput_total_rps: float
    throughput_success_rps: float
    success_rate: float
    mean_latency_seconds: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    p99_latency_seconds: float
    avg_population_observed: float
    avg_population_little: float
    little_law_error_pct: float
    errors: Dict[str, int]


@dataclass
class BenchmarkAggregate:
    mode: str
    workload: str
    concurrency: int
    repeats: int
    requests_per_repeat_mean: float
    throughput_success_rps_mean: float
    throughput_success_rps_std: float
    throughput_total_rps_mean: float
    success_rate_mean: float
    mean_latency_seconds_mean: float
    p95_latency_seconds_mean: float
    avg_population_observed_mean: float
    avg_population_little_mean: float
    little_law_error_pct_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark sandbox scaling and compare modes with USL and Little's Law."
        )
    )
    parser.add_argument(
        "--modes",
        default="all",
        help="Comma-separated modes: inprocess,subprocess,container or 'all'.",
    )
    parser.add_argument(
        "--workloads",
        default="all",
        help=(
            "Comma-separated workloads: tiny_compute,baseline,monty_import,"
            "rlm_tiny_compute,rlm_baseline,rlm_monty_import,rlm_subcall or 'all'."
        ),
    )
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8,16",
        help="Comma-separated concurrency levels.",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=2000,
        help=(
            "Legacy open-loop request count. In closed-loop mode this is ignored, "
            "use --measure-seconds."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Benchmark repeats per mode/concurrency point.",
    )
    parser.add_argument(
        "--run-model",
        choices=["closed_loop", "open_loop"],
        default="closed_loop",
        help=(
            "closed_loop = steady-state virtual users (recommended); "
            "open_loop = burst submit fixed requests."
        ),
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=1.5,
        help="Warmup duration for closed-loop mode before collecting metrics.",
    )
    parser.add_argument(
        "--measure-seconds",
        type=float,
        default=6.0,
        help="Measurement duration for closed-loop mode.",
    )
    parser.add_argument(
        "--sample-interval-ms",
        type=float,
        default=10.0,
        help="Independent in-flight sampler interval for Little's Law checks.",
    )
    parser.add_argument(
        "--think-time-ms",
        type=float,
        default=0.0,
        help="Closed-loop think time per virtual user between requests.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=8,
        help="Warmup requests per mode before measurement.",
    )
    parser.add_argument(
        "--skip-unavailable",
        action="store_true",
        default=True,
        help="Skip mode/workload combinations that cannot run in this environment.",
    )
    parser.add_argument(
        "--no-skip-unavailable",
        action="store_false",
        dest="skip_unavailable",
        help="Do not skip unavailable combinations; include failures in the report.",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path. Defaults to artifacts/benchmarks/ timestamped file.",
    )
    return parser.parse_args()


def parse_modes(value: str) -> List[ModeName]:
    supported = ("inprocess", "subprocess", "container")
    if value.strip().lower() == "all":
        return list(supported)  # type: ignore[return-value]
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("At least one mode is required.")
    invalid = [mode for mode in parsed if mode not in set(supported)]
    if invalid:
        raise ValueError(f"Unsupported mode(s): {', '.join(invalid)}")
    dedup: List[str] = []
    for mode in parsed:
        if mode not in dedup:
            dedup.append(mode)
    return [mode for mode in dedup if mode in set(supported)]  # type: ignore[return-value]


def parse_workloads(value: str) -> List[WorkloadName]:
    supported = (
        "tiny_compute",
        "baseline",
        "monty_import",
        "rlm_tiny_compute",
        "rlm_baseline",
        "rlm_monty_import",
        "rlm_subcall",
    )
    if value.strip().lower() == "all":
        return list(supported)  # type: ignore[return-value]
    parsed = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("At least one workload is required.")
    invalid = [workload for workload in parsed if workload not in set(supported)]
    if invalid:
        raise ValueError(f"Unsupported workload(s): {', '.join(invalid)}")
    dedup: List[str] = []
    for workload in parsed:
        if workload not in dedup:
            dedup.append(workload)
    return [workload for workload in dedup if workload in set(supported)]  # type: ignore[return-value]


def parse_concurrency(value: str) -> List[int]:
    points = [int(part.strip()) for part in value.split(",") if part.strip()]
    points = sorted(set(points))
    if not points or any(n <= 0 for n in points):
        raise ValueError("Concurrency levels must be positive integers.")
    return points


def benchmark_run(
    *,
    mode: ModeName,
    workload: WorkloadName,
    concurrency: int,
    requests: int,
    repeat: int,
    run_model: str,
    warmup_seconds: float,
    measure_seconds: float,
    sample_interval_seconds: float,
    think_time_seconds: float,
) -> BenchmarkRun:
    if run_model == "open_loop":
        return _benchmark_run_open_loop(
            mode=mode,
            workload=workload,
            concurrency=concurrency,
            requests=requests,
            repeat=repeat,
            sample_interval_seconds=sample_interval_seconds,
        )
    return _benchmark_run_closed_loop(
        mode=mode,
        workload=workload,
        concurrency=concurrency,
        repeat=repeat,
        warmup_seconds=warmup_seconds,
        measure_seconds=measure_seconds,
        sample_interval_seconds=sample_interval_seconds,
        think_time_seconds=think_time_seconds,
    )


def _benchmark_run_open_loop(
    *,
    mode: ModeName,
    workload: WorkloadName,
    concurrency: int,
    requests: int,
    repeat: int,
    sample_interval_seconds: float,
) -> BenchmarkRun:
    request_code = _workload_code(workload)
    allowed_imports = _allowed_imports(workload)
    lock = threading.Lock()
    samples: List[RequestSample] = []
    population_samples: List[PopulationSample] = []
    submitted = 0
    completed = 0
    stop_sampling = threading.Event()
    start = time.perf_counter()

    def sample_population() -> None:
        while not stop_sampling.is_set():
            now = time.perf_counter()
            with lock:
                in_system = submitted - completed
            population_samples.append(
                PopulationSample(timestamp_seconds=now - start, in_flight=in_system)
            )
            time.sleep(sample_interval_seconds)
        now = time.perf_counter()
        with lock:
            in_system = submitted - completed
        population_samples.append(
            PopulationSample(timestamp_seconds=now - start, in_flight=in_system)
        )

    sampler = threading.Thread(target=sample_population, daemon=True)
    sampler.start()

    def mark_completed() -> None:
        nonlocal completed
        with lock:
            completed += 1

    def worker(request_id: int, arrival: float) -> RequestSample:
        error: Optional[str] = None
        success = False
        try:
            success, error = _execute_request(
                mode=mode,
                workload=workload,
                code=request_code,
                request_id=request_id,
                allowed_imports=allowed_imports,
            )
        except Exception as exc:  # noqa: BLE001
            success = False
            error = f"{type(exc).__name__}: {exc}"
        finally:
            finish = time.perf_counter()
            mark_completed()
        return RequestSample(
            request_id=request_id,
            success=success,
            latency_seconds=finish - arrival,
            error=error,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for request_id in range(requests):
            arrival = time.perf_counter()
            with lock:
                submitted += 1
            futures.append(pool.submit(worker, request_id, arrival))
        for fut in futures:
            samples.append(fut.result())

    stop_sampling.set()
    sampler.join(timeout=2.0)

    total_seconds = time.perf_counter() - start
    latencies = [s.latency_seconds for s in samples]
    success_count = sum(1 for s in samples if s.success)
    throughput_total = len(samples) / total_seconds if total_seconds > 0 else 0.0
    throughput_success = success_count / total_seconds if total_seconds > 0 else 0.0
    success_rate = success_count / len(samples) if samples else 0.0
    mean_latency = mean(latencies) if latencies else 0.0
    avg_population_observed = _mean_population_in_window(
        population_samples=population_samples,
        window_start=0.0,
        window_end=total_seconds,
    )
    avg_population_little = little_law_l(throughput_total, mean_latency)
    error_pct = relative_error(avg_population_observed, avg_population_little) * 100.0

    errors: Dict[str, int] = {}
    for sample in samples:
        if sample.error:
            errors[sample.error] = errors.get(sample.error, 0) + 1

    return BenchmarkRun(
        run_model="open_loop",
        mode=mode,
        workload=workload,
        concurrency=concurrency,
        repeat=repeat,
        requests=requests,
        total_seconds=total_seconds,
        throughput_total_rps=throughput_total,
        throughput_success_rps=throughput_success,
        success_rate=success_rate,
        mean_latency_seconds=mean_latency,
        p50_latency_seconds=percentile(latencies, 50.0),
        p95_latency_seconds=percentile(latencies, 95.0),
        p99_latency_seconds=percentile(latencies, 99.0),
        avg_population_observed=avg_population_observed,
        avg_population_little=avg_population_little,
        little_law_error_pct=error_pct,
        errors=errors,
    )


def _benchmark_run_closed_loop(
    *,
    mode: ModeName,
    workload: WorkloadName,
    concurrency: int,
    repeat: int,
    warmup_seconds: float,
    measure_seconds: float,
    sample_interval_seconds: float,
    think_time_seconds: float,
) -> BenchmarkRun:
    if measure_seconds <= 0:
        raise ValueError("measure_seconds must be > 0")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be > 0")

    request_code = _workload_code(workload)
    allowed_imports = _allowed_imports(workload)
    samples: List[RequestSample] = []
    population_samples: List[PopulationSample] = []

    lock = threading.Lock()
    request_id_lock = threading.Lock()
    in_flight = 0
    next_request_id = 0
    stop_sampling = threading.Event()

    start = time.perf_counter()
    warmup_end = start + warmup_seconds
    measure_end = warmup_end + measure_seconds

    def take_request_id() -> int:
        nonlocal next_request_id
        with request_id_lock:
            rid = next_request_id
            next_request_id += 1
            return rid

    def sample_population() -> None:
        while not stop_sampling.is_set():
            now = time.perf_counter()
            with lock:
                current = in_flight
            population_samples.append(
                PopulationSample(timestamp_seconds=now - start, in_flight=current)
            )
            time.sleep(sample_interval_seconds)
        now = time.perf_counter()
        with lock:
            current = in_flight
        population_samples.append(
            PopulationSample(timestamp_seconds=now - start, in_flight=current)
        )

    sampler = threading.Thread(target=sample_population, daemon=True)
    sampler.start()

    def worker() -> List[RequestSample]:
        nonlocal in_flight
        local_samples: List[RequestSample] = []
        while True:
            now = time.perf_counter()
            if now >= measure_end:
                break
            req_id = take_request_id()
            req_start = time.perf_counter()
            with lock:
                # Track requests currently executing for independent population sampling.
                in_flight += 1
            error: Optional[str] = None
            success = False
            try:
                success, error = _execute_request(
                    mode=mode,
                    workload=workload,
                    code=request_code,
                    request_id=req_id,
                    allowed_imports=allowed_imports,
                )
            except Exception as exc:  # noqa: BLE001
                success = False
                error = f"{type(exc).__name__}: {exc}"
            finally:
                req_end = time.perf_counter()
                with lock:
                    in_flight -= 1

            if req_start >= warmup_end and req_end <= measure_end:
                local_samples.append(
                    RequestSample(
                        request_id=req_id,
                        success=success,
                        latency_seconds=req_end - req_start,
                        error=error,
                    )
                )

            if think_time_seconds > 0:
                time.sleep(think_time_seconds)
        return local_samples

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(concurrency)]
        for future in futures:
            samples.extend(future.result())

    stop_sampling.set()
    sampler.join(timeout=2.0)

    total_seconds = measure_seconds
    latencies = [s.latency_seconds for s in samples]
    success_count = sum(1 for s in samples if s.success)
    throughput_total = len(samples) / total_seconds if total_seconds > 0 else 0.0
    throughput_success = success_count / total_seconds if total_seconds > 0 else 0.0
    success_rate = success_count / len(samples) if samples else 0.0
    mean_latency = mean(latencies) if latencies else 0.0
    avg_population_observed = _mean_population_in_window(
        population_samples=population_samples,
        window_start=warmup_seconds,
        window_end=warmup_seconds + measure_seconds,
    )
    avg_population_little = little_law_l(throughput_total, mean_latency)
    error_pct = relative_error(avg_population_observed, avg_population_little) * 100.0

    errors: Dict[str, int] = {}
    for sample in samples:
        if sample.error:
            errors[sample.error] = errors.get(sample.error, 0) + 1

    return BenchmarkRun(
        run_model="closed_loop",
        mode=mode,
        workload=workload,
        concurrency=concurrency,
        repeat=repeat,
        requests=len(samples),
        total_seconds=total_seconds,
        throughput_total_rps=throughput_total,
        throughput_success_rps=throughput_success,
        success_rate=success_rate,
        mean_latency_seconds=mean_latency,
        p50_latency_seconds=percentile(latencies, 50.0),
        p95_latency_seconds=percentile(latencies, 95.0),
        p99_latency_seconds=percentile(latencies, 99.0),
        avg_population_observed=avg_population_observed,
        avg_population_little=avg_population_little,
        little_law_error_pct=error_pct,
        errors=errors,
    )


def _mean_population_in_window(
    *,
    population_samples: Sequence[PopulationSample],
    window_start: float,
    window_end: float,
) -> float:
    if window_end <= window_start:
        return 0.0
    values = [
        sample.in_flight
        for sample in population_samples
        if window_start <= sample.timestamp_seconds <= window_end
    ]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def aggregate_runs(runs: Sequence[BenchmarkRun]) -> List[BenchmarkAggregate]:
    grouped: Dict[Tuple[str, str, int], List[BenchmarkRun]] = {}
    for run in runs:
        key = (run.mode, run.workload, run.concurrency)
        grouped.setdefault(key, []).append(run)

    output: List[BenchmarkAggregate] = []
    for (mode, workload, concurrency), group in sorted(grouped.items()):
        success_tp = [x.throughput_success_rps for x in group]
        total_tp = [x.throughput_total_rps for x in group]
        success_rates = [x.success_rate for x in group]
        lat_means = [x.mean_latency_seconds for x in group]
        p95s = [x.p95_latency_seconds for x in group]
        l_obs = [x.avg_population_observed for x in group]
        l_little = [x.avg_population_little for x in group]
        ll_err = [x.little_law_error_pct for x in group]
        output.append(
            BenchmarkAggregate(
                mode=mode,
                workload=workload,
                concurrency=concurrency,
                repeats=len(group),
                requests_per_repeat_mean=mean([x.requests for x in group]),
                throughput_success_rps_mean=mean(success_tp),
                throughput_success_rps_std=pstdev(success_tp)
                if len(success_tp) > 1
                else 0.0,
                throughput_total_rps_mean=mean(total_tp),
                success_rate_mean=mean(success_rates),
                mean_latency_seconds_mean=mean(lat_means),
                p95_latency_seconds_mean=mean(p95s),
                avg_population_observed_mean=mean(l_obs),
                avg_population_little_mean=mean(l_little),
                little_law_error_pct_mean=mean(ll_err),
            )
        )
    return output


def fit_usl_by_mode_workload(
    aggregates: Iterable[BenchmarkAggregate],
) -> Dict[str, USLFit]:
    grouped: Dict[str, List[BenchmarkAggregate]] = {}
    for row in aggregates:
        key = f"{row.workload}:{row.mode}"
        grouped.setdefault(key, []).append(row)

    fits: Dict[str, USLFit] = {}
    for key, rows in grouped.items():
        points = sorted(
            (row.concurrency, row.throughput_success_rps_mean)
            for row in rows
            if row.success_rate_mean > 0.0 and row.throughput_success_rps_mean > 0.0
        )
        if len(points) < 3:
            continue
        try:
            fits[key] = fit_usl(points)
        except ValueError:
            continue
    return fits


def best_measured(
    aggregates: Sequence[BenchmarkAggregate],
) -> Dict[str, BenchmarkAggregate]:
    best: Dict[str, BenchmarkAggregate] = {}
    for row in aggregates:
        key = f"{row.workload}:{row.mode}"
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        if row.success_rate_mean > current.success_rate_mean:
            best[key] = row
            continue
        if (
            row.success_rate_mean == current.success_rate_mean
            and row.throughput_success_rps_mean > current.throughput_success_rps_mean
        ):
            best[key] = row
    return best


def print_summary(
    *,
    aggregates: Sequence[BenchmarkAggregate],
    fits: Dict[str, USLFit],
) -> None:
    by_workload: Dict[str, List[BenchmarkAggregate]] = {}
    for row in aggregates:
        by_workload.setdefault(row.workload, []).append(row)

    for workload, rows in sorted(by_workload.items()):
        print(f"\nWorkload: {workload}")
        by_mode: Dict[str, List[BenchmarkAggregate]] = {}
        for row in rows:
            by_mode.setdefault(row.mode, []).append(row)
        for mode, mode_rows in sorted(by_mode.items()):
            print(f"  Mode: {mode}")
            print(
                "    N   reqs/run   success%   throughput(req/s)   mean(ms)   p95(ms)   "
                "L_obs   L_little   LL_err%"
            )
            for row in sorted(mode_rows, key=lambda x: x.concurrency):
                print(
                    f"    {row.concurrency:>2}  "
                    f"{row.requests_per_repeat_mean:>8.1f}  "
                    f"{row.success_rate_mean*100:>8.2f}  "
                    f"{row.throughput_success_rps_mean:>18.2f}  "
                    f"{row.mean_latency_seconds_mean*1000:>8.1f}  "
                    f"{row.p95_latency_seconds_mean*1000:>7.1f}  "
                    f"{row.avg_population_observed_mean:>6.2f}  "
                    f"{row.avg_population_little_mean:>8.2f}  "
                    f"{row.little_law_error_pct_mean:>7.2f}"
                )
            fit_key = f"{workload}:{mode}"
            fit = fits.get(fit_key)
            if fit is not None:
                print(
                    "    USL fit:"
                    f" sigma={fit.sigma:.4f}, kappa={fit.kappa:.5f},"
                    f" R2={fit.r2:.4f}, N_opt≈{fit.n_opt:.2f},"
                    f" Xmax≈{fit.max_throughput_estimate:.2f} req/s"
                )


def _workload_code(workload: WorkloadName) -> str:
    if _is_rlm_workload(workload):
        return "FINAL('rlm_workload_uses_engine')\n"
    if workload == "tiny_compute":
        return (
            "values = list(range(256))\n"
            "FINAL(sum(values))\n"
        )
    if workload == "monty_import":
        return (
            "import pydantic_monty\n"
            "from json.decoder import JSONDecoder\n"
            "import json\n"
            "payload = {'idx': data.get('idx', 0), 'values': list(range(128))}\n"
            "decoder = JSONDecoder()\n"
            "decoded = decoder.decode(json.dumps(payload))\n"
            "FINAL(sum(decoded['values']))\n"
        )
    return (
        "from json.decoder import JSONDecoder\n"
        "import json\n"
        "payload = {'idx': data.get('idx', 0), 'values': list(range(128))}\n"
        "decoder = JSONDecoder()\n"
        "decoded = decoder.decode(json.dumps(payload))\n"
        "FINAL(sum(decoded['values']))\n"
    )


def _allowed_imports(workload: WorkloadName) -> List[str]:
    imports = ["json"]
    if workload in {"tiny_compute", "rlm_tiny_compute", "rlm_subcall"}:
        return []
    if workload in {"monty_import", "rlm_monty_import"}:
        imports.append("pydantic_monty")
    return imports


def _is_rlm_workload(workload: WorkloadName) -> bool:
    return workload.startswith("rlm_")


class _BenchmarkRLMProvider(BaseProvider):
    name = "benchmark_rlm_mock"

    def __init__(self, workload: WorkloadName, marker: str) -> None:
        self._workload = workload
        self._marker = marker

    def generate(self, task: TaskSpec) -> ProviderResult:
        prompt = task.input_text or ""
        if task.metadata.get("_subcall_prompt") is not None or prompt.startswith(
            "SUBCALL:"
        ):
            output = (
                "```python\n"
                f"FINAL('subcall::{self._marker}')\n"
                "```"
            )
        else:
            request_id = int(task.metadata.get("request_id", 0))
            output = self._main_response(request_id=request_id)

        token_count = max(len(output.split()), 1)
        return ProviderResult(
            output_text=output,
            raw={"mock": True},
            usage={"output_tokens": token_count, "total_tokens": token_count + 12},
            provider=self.name,
            model=task.model,
        )

    def _main_response(self, *, request_id: int) -> str:
        code = _rlm_workload_code(
            workload=self._workload,
            marker=self._marker,
            request_id=request_id,
        )
        return f"```python\n{code}\n```"


def _rlm_workload_code(*, workload: WorkloadName, marker: str, request_id: int) -> str:
    if workload == "rlm_tiny_compute":
        return (
            "values = list(range(256))\n"
            f"FINAL('{marker}:' + str(sum(values)))"
        )

    if workload == "rlm_baseline":
        return (
            "import json\n"
            "from json.decoder import JSONDecoder\n"
            f"payload = {{'idx': {request_id}, 'values': list(range(128))}}\n"
            "decoded = JSONDecoder().decode(json.dumps(payload))\n"
            f"FINAL('{marker}:' + str(sum(decoded['values'])))"
        )

    if workload == "rlm_monty_import":
        return (
            "import pydantic_monty\n"
            "import json\n"
            "from json.decoder import JSONDecoder\n"
            f"payload = {{'idx': {request_id}, 'values': list(range(128))}}\n"
            "decoded = JSONDecoder().decode(json.dumps(payload))\n"
            f"FINAL('{marker}:' + str(sum(decoded['values'])))"
        )

    if workload == "rlm_subcall":
        return (
            "analysis = llm_query('SUBCALL: summarize this request')\n"
            f"FINAL('{marker}:' + analysis)"
        )

    raise ValueError(f"unsupported RLM workload: {workload}")


def _build_rlm_sandbox_factory(mode: ModeName) -> Callable[..., Any]:
    def factory(
        *,
        isolation: Optional[str],
        data: str,
        context: Any,
        namespace: Dict[str, Any],
        allowed_imports: List[str],
        output_char_limit: int,
        timeout_seconds: Optional[float],
        inject_search_tools: bool,
        enable_pip: bool,
        llm_query: Callable[[str], str],
        llm_batch: Optional[Callable[[list], list]] = None,
        sandbox_image: Optional[Any] = None,
    ) -> Any:
        del isolation, inject_search_tools, enable_pip, sandbox_image
        imports = list(allowed_imports)

        if mode == "inprocess":
            return PythonSandbox(
                data=data,
                context=context,
                llm_query=llm_query,
                llm_batch=llm_batch,
                namespace=namespace,
                allowed_imports=imports,
                output_char_limit=output_char_limit,
                timeout_seconds=None,
                inject_search_tools=False,
            )

        base_timeout = timeout_seconds or 30.0
        isolated_ns = dict(namespace)
        isolated_ns["context"] = context

        if mode == "subprocess":
            config = SandboxConfig(
                max_cpu_seconds=base_timeout,
                timeout_seconds=base_timeout + 30.0,
                allowed_imports=set(imports),
            )
            return IsolatedSandbox(
                data=data,
                namespace=isolated_ns,
                config=config,
                llm_query=llm_query,
                llm_batch=llm_batch,
            )

        config = ContainerConfig(
            max_cpu_seconds=base_timeout,
            timeout_seconds=base_timeout + 30.0,
            allowed_imports=set(imports),
        )
        return ContainerSandbox(
            data=data,
            namespace=isolated_ns,
            config=config,
            fallback_to_subprocess=False,
            llm_query=llm_query,
            llm_batch=llm_batch,
        )

    return factory


def _execute_rlm_request(
    *,
    mode: ModeName,
    workload: WorkloadName,
    request_id: int,
) -> Tuple[bool, Optional[str]]:
    marker = f"BENCH-{workload}-{request_id}"
    provider = _BenchmarkRLMProvider(workload, marker)
    allowed_imports = _allowed_imports(workload)
    required_substrings = [marker]
    if workload == "rlm_subcall":
        required_substrings.append(f"subcall::{marker}")

    recursive_subcalls = workload == "rlm_subcall"
    engine = RLMEngine(
        max_steps=2,
        verify_on_final=True,
        recursive_subcalls=recursive_subcalls,
        max_recursion_depth=1 if recursive_subcalls else 0,
        subcall_max_steps=1,
        subcall_verify_on_final=False,
        inject_search_tools=False,
        subcall_inject_search_tools=False,
        allowed_imports=allowed_imports,
        isolation=None,
    )

    task = TaskSpec(
        task_id=f"bench-rlm-{request_id}",
        input_text=(
            "Generate Python code that calls FINAL() with a string containing "
            f"exact marker {marker}."
        ),
        model="benchmark-rlm-model",
        responses={},
        budget=Budget(
            max_tokens=4096,
            max_total_tokens=None,
            max_seconds=20.0,
        ),
        success_criteria=SuccessCriteria(required_substrings=required_substrings),
        max_output_tokens=512,
        metadata={
            "request_id": request_id,
            "benchmark_workload": workload,
        },
    )

    report = engine.run(
        task,
        provider,
        data={"idx": request_id},
        sandbox_factory=_build_rlm_sandbox_factory(mode),
    )
    if report.success:
        return True, None

    error = "; ".join(report.errors) if report.errors else "rlm_run_failed"
    return False, error


def _execute_request(
    *,
    mode: ModeName,
    workload: WorkloadName,
    code: str,
    request_id: int,
    allowed_imports: Sequence[str],
) -> Tuple[bool, Optional[str]]:
    if _is_rlm_workload(workload):
        return _execute_rlm_request(
            mode=mode,
            workload=workload,
            request_id=request_id,
        )

    imports = list(allowed_imports)
    if mode == "inprocess":
        sandbox = PythonSandbox(
            data={"idx": request_id},
            llm_query=lambda _prompt: "",
            allowed_imports=imports,
            inject_search_tools=False,
        )
        result = sandbox.exec(code)
        if result.error:
            return False, result.error
        if not sandbox.answer.get("ready"):
            return False, "FINAL() not called"
        return True, None

    sandbox = IsolatedSandbox(
        data={"idx": request_id},
        allowed_imports=set(imports),
    ) if mode == "subprocess" else ContainerSandbox(
        data={"idx": request_id},
        allowed_imports=set(imports),
        fallback_to_subprocess=False,
    )
    result = sandbox.exec(code)
    if result.error:
        return False, result.error
    if result.final_answer is None:
        return False, "FINAL() not called"
    return True, None


def _default_output_path() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = Path("artifacts") / "benchmarks"
    return base / f"sandbox-scaling-{stamp}.json"


def _warmup_mode(mode: ModeName, workload: WorkloadName, warmup_requests: int) -> None:
    if warmup_requests <= 0:
        return
    code = _workload_code(workload)
    allowed_imports = _allowed_imports(workload)
    for request_id in range(warmup_requests):
        _execute_request(
            mode=mode,
            workload=workload,
            code=code,
            request_id=request_id,
            allowed_imports=allowed_imports,
        )


def _is_mode_available(mode: ModeName) -> bool:
    if mode != "container":
        return True
    return is_container_available()


def main() -> None:
    args = parse_args()
    if args.run_model == "closed_loop":
        if args.warmup_seconds < 0:
            raise ValueError("--warmup-seconds must be >= 0")
        if args.measure_seconds <= 0:
            raise ValueError("--measure-seconds must be > 0")
        if args.sample_interval_ms <= 0:
            raise ValueError("--sample-interval-ms must be > 0")
        if args.think_time_ms < 0:
            raise ValueError("--think-time-ms must be >= 0")
    modes = parse_modes(args.modes)
    workloads = parse_workloads(args.workloads)
    concurrency_points = parse_concurrency(args.concurrency)
    output_path = Path(args.output) if args.output else _default_output_path()

    runs: List[BenchmarkRun] = []
    skipped: List[Dict[str, str]] = []
    for workload in workloads:
        for mode in modes:
            available = _is_mode_available(mode)
            if not available and args.skip_unavailable:
                skipped.append(
                    {
                        "mode": mode,
                        "workload": workload,
                        "reason": "mode unavailable in current environment",
                    }
                )
                print(
                    f"{mode:>10} workload={workload:<12} skipped: mode unavailable"
                )
                continue

            try:
                _warmup_mode(mode, workload, args.warmup_requests)
            except Exception as exc:  # noqa: BLE001
                reason = f"warmup failed: {type(exc).__name__}: {exc}"
                if args.skip_unavailable:
                    skipped.append(
                        {
                            "mode": mode,
                            "workload": workload,
                            "reason": reason,
                        }
                    )
                    print(
                        f"{mode:>10} workload={workload:<12} skipped: {reason}"
                    )
                    continue
                raise

            for concurrency in concurrency_points:
                for repeat in range(1, args.repeats + 1):
                    run = benchmark_run(
                        mode=mode,
                        workload=workload,
                        concurrency=concurrency,
                        requests=args.requests_per_level,
                        repeat=repeat,
                        run_model=args.run_model,
                        warmup_seconds=args.warmup_seconds,
                        measure_seconds=args.measure_seconds,
                        sample_interval_seconds=args.sample_interval_ms / 1000.0,
                        think_time_seconds=args.think_time_ms / 1000.0,
                    )
                    runs.append(run)
                    print(
                        f"{mode:>10} workload={workload:<12} "
                        f"N={concurrency:<3} repeat={repeat}/{args.repeats} "
                        f"success={run.success_rate*100:5.1f}% "
                        f"throughput={run.throughput_success_rps:8.2f} req/s "
                        f"p95={run.p95_latency_seconds*1000:7.1f} ms "
                        f"requests={run.requests}"
                    )

    aggregates = aggregate_runs(runs)
    fits = fit_usl_by_mode_workload(aggregates)
    best = best_measured(aggregates)
    print_summary(aggregates=aggregates, fits=fits)

    winners_by_workload: Dict[str, BenchmarkAggregate] = {}
    for workload in workloads:
        candidates = [row for row in aggregates if row.workload == workload]
        if not candidates:
            continue
        winners_by_workload[workload] = max(
            candidates,
            key=lambda row: (row.success_rate_mean, row.throughput_success_rps_mean),
        )

    winner_overall = max(
        winners_by_workload.values(),
        key=lambda row: (row.success_rate_mean, row.throughput_success_rps_mean),
        default=None,
    )

    payload = {
        "config": {
            "modes": modes,
            "workloads": workloads,
            "concurrency": concurrency_points,
            "requests_per_level": args.requests_per_level,
            "repeats": args.repeats,
            "warmup_requests": args.warmup_requests,
            "run_model": args.run_model,
            "warmup_seconds": args.warmup_seconds,
            "measure_seconds": args.measure_seconds,
            "sample_interval_ms": args.sample_interval_ms,
            "think_time_ms": args.think_time_ms,
            "skip_unavailable": args.skip_unavailable,
        },
        "skipped": skipped,
        "runs": [asdict(run) for run in runs],
        "aggregates": [asdict(row) for row in aggregates],
        "usl_fits": {key: asdict(fit) for key, fit in fits.items()},
        "best_measured": {key: asdict(row) for key, row in best.items()},
        "winners_by_workload": {
            workload: asdict(row)
            for workload, row in winners_by_workload.items()
        },
        "winner_overall": asdict(winner_overall) if winner_overall is not None else None,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark report: {output_path}")


if __name__ == "__main__":
    main()
