from __future__ import annotations

import concurrent.futures

import pytest

from scripts import sandbox_scaling_benchmark as bench


def _aggregate(*, concurrency: int, throughput: float, success_rate: float) -> bench.BenchmarkAggregate:
    return bench.BenchmarkAggregate(
        mode="subprocess",
        workload="rlm_baseline",
        concurrency=concurrency,
        repeats=1,
        requests_per_repeat_mean=1.0,
        throughput_success_rps_mean=throughput,
        throughput_success_rps_std=0.0,
        throughput_total_rps_mean=throughput,
        success_rate_mean=success_rate,
        mean_latency_seconds_mean=0.01,
        p95_latency_seconds_mean=0.02,
        avg_population_observed_mean=1.0,
        avg_population_little_mean=1.0,
        little_law_error_pct_mean=0.0,
    )


def test_parse_workloads_all_includes_rlm() -> None:
    workloads = set(bench.parse_workloads("all"))
    assert {
        "rlm_tiny_compute",
        "rlm_baseline",
        "rlm_monty_import",
        "rlm_subcall",
    }.issubset(workloads)


@pytest.mark.parametrize(
    "workload",
    ["rlm_tiny_compute", "rlm_baseline", "rlm_monty_import", "rlm_subcall"],
)
def test_execute_rlm_request_inprocess_succeeds(workload: str) -> None:
    success, error = bench._execute_rlm_request(
        mode="inprocess", workload=workload, request_id=1
    )
    assert success, error
    assert error is None


@pytest.mark.parametrize("workload", ["rlm_baseline", "rlm_monty_import", "rlm_subcall"])
def test_execute_rlm_request_subprocess_succeeds(workload: str) -> None:
    success, error = bench._execute_rlm_request(
        mode="subprocess", workload=workload, request_id=1
    )
    assert success, error
    assert error is None


def test_fit_usl_by_mode_workload_ignores_zero_throughput() -> None:
    aggregates = [
        _aggregate(concurrency=1, throughput=0.0, success_rate=0.0),
        _aggregate(concurrency=2, throughput=0.0, success_rate=0.0),
        _aggregate(concurrency=4, throughput=0.0, success_rate=0.0),
    ]
    fits = bench.fit_usl_by_mode_workload(aggregates)
    assert fits == {}


def test_execute_rlm_subcall_inprocess_succeeds_in_worker_thread() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            bench._execute_rlm_request,
            mode="inprocess",
            workload="rlm_subcall",
            request_id=7,
        )
        success, error = future.result()
    assert success, error
    assert error is None
