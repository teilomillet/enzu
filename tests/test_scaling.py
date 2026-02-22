from __future__ import annotations

from enzu.scalability import (
    average_population,
    fit_usl,
    littles_law as little_law_l,
    percentile,
    relative_error,
    usl_throughput,
)


def test_percentile_interpolates() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50.0) == 25.0
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 100.0) == 40.0


def test_average_population_matches_hand_computation() -> None:
    events = [
        (0.0, +1),
        (0.5, +1),
        (1.0, -1),
        (1.5, -1),
    ]
    observed = average_population(events, total_seconds=2.0)
    expected = 1.0
    assert abs(observed - expected) < 1e-9


def test_little_law_helpers_consistent() -> None:
    throughput = 4.0
    mean_latency = 0.25
    predicted_l = little_law_l(throughput, mean_latency)
    assert predicted_l == 1.0
    assert relative_error(1.0, predicted_l) == 0.0


def test_fit_usl_recovers_synthetic_curve() -> None:
    x1 = 120.0
    sigma = 0.09
    kappa = 0.012
    points = [
        (1, usl_throughput(1, x1, sigma, kappa)),
        (2, usl_throughput(2, x1, sigma, kappa)),
        (4, usl_throughput(4, x1, sigma, kappa)),
        (8, usl_throughput(8, x1, sigma, kappa)),
        (12, usl_throughput(12, x1, sigma, kappa)),
        (16, usl_throughput(16, x1, sigma, kappa)),
    ]
    fit = fit_usl(points)
    assert abs(fit.sigma - sigma) < 0.03
    assert abs(fit.kappa - kappa) < 0.01
    assert fit.r2 > 0.99
