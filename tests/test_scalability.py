"""Tests for enzu.scalability — USL model + Little's Law."""

from __future__ import annotations

import math

import pytest

from enzu.scalability import (
    LittlesLawResult,
    USLModel,
    littles_law,
    littles_law_check,
)


# ============================================================
# USL math tests
# ============================================================


class TestUSLThroughput:
    """Core throughput formula: X(N) = λN / (1 + σ(N-1) + κN(N-1))."""

    def test_n1_always_returns_lambda(self):
        model = USLModel(lambda_=100.0, sigma=0.3, kappa=0.01)
        assert model.throughput(1) == pytest.approx(100.0)

    def test_n0_returns_zero(self):
        model = USLModel(lambda_=100.0, sigma=0.3, kappa=0.01)
        assert model.throughput(0) == 0.0

    def test_negative_n_returns_zero(self):
        model = USLModel(lambda_=100.0, sigma=0.3, kappa=0.01)
        assert model.throughput(-5) == 0.0

    def test_linear_scaling_no_contention(self):
        """σ=0, κ=0 → X(N) = λ·N (perfect linear scaling)."""
        model = USLModel(lambda_=50.0, sigma=0.0, kappa=0.0)
        for n in [1, 2, 4, 8, 16, 64]:
            assert model.throughput(n) == pytest.approx(50.0 * n)

    def test_amdahls_limit(self):
        """σ>0, κ=0 → throughput bounded by λ/σ."""
        model = USLModel(lambda_=100.0, sigma=0.1, kappa=0.0)
        limit = 100.0 / 0.1  # 1000

        # Throughput at very high N should approach the limit
        x_1000 = model.throughput(1000)
        assert x_1000 < limit
        assert x_1000 > limit * 0.98  # within 2% of asymptote

    def test_retrograde_throughput_decreases_past_peak(self):
        """κ>0 → throughput eventually decreases."""
        model = USLModel(lambda_=100.0, sigma=0.05, kappa=0.005)
        n_star = model.peak_concurrency()

        # Past the peak, throughput should drop
        x_peak = model.throughput(n_star)
        x_far = model.throughput(n_star * 5)
        assert x_far < x_peak

    def test_known_values(self):
        """Spot-check against hand-computed values."""
        model = USLModel(lambda_=100.0, sigma=0.1, kappa=0.01)

        # N=2: denom = 1 + 0.1*(1) + 0.01*2*1 = 1.12
        expected = 100.0 * 2 / 1.12
        assert model.throughput(2) == pytest.approx(expected)

        # N=4: denom = 1 + 0.1*3 + 0.01*4*3 = 1.42
        expected = 100.0 * 4 / 1.42
        assert model.throughput(4) == pytest.approx(expected)


class TestUSLPeak:
    """Peak concurrency: N* = floor(sqrt((1-σ)/κ))."""

    def test_peak_concurrency_basic(self):
        model = USLModel(lambda_=100.0, sigma=0.05, kappa=0.005)
        # sqrt((1-0.05)/0.005) ≈ 13.78; discrete peak is at 14
        n_star = model.peak_concurrency()
        assert n_star in (13, 14)
        # Must be the actual discrete maximum
        assert model.throughput(n_star) >= model.throughput(n_star - 1)
        assert model.throughput(n_star) >= model.throughput(n_star + 1)

    def test_peak_concurrency_known(self):
        # (1-0.1)/0.01 = 90, sqrt(90) ≈ 9.49
        model = USLModel(lambda_=100.0, sigma=0.1, kappa=0.01)
        n_star = model.peak_concurrency()
        assert n_star in (9, 10)

    def test_peak_concurrency_no_kappa_raises(self):
        model = USLModel(lambda_=100.0, sigma=0.1, kappa=0.0)
        with pytest.raises(ValueError, match="kappa <= 0"):
            model.peak_concurrency()

    def test_peak_throughput_matches_formula(self):
        model = USLModel(lambda_=100.0, sigma=0.1, kappa=0.01)
        n_star = model.peak_concurrency()
        assert model.peak_throughput() == pytest.approx(model.throughput(n_star))

    def test_peak_is_local_maximum(self):
        """X(N*) >= X(N*-1) and X(N*) >= X(N*+1)."""
        model = USLModel(lambda_=100.0, sigma=0.05, kappa=0.005)
        n_star = model.peak_concurrency()

        x_peak = model.throughput(n_star)
        if n_star > 1:
            assert x_peak >= model.throughput(n_star - 1)
        assert x_peak >= model.throughput(n_star + 1)


# ============================================================
# USL fitting tests
# ============================================================


class TestUSLFit:
    """Fit USL from (concurrency, throughput) samples."""

    def test_fit_linear_data(self):
        """Perfect linear scaling → σ≈0, κ≈0."""
        lambda_ = 100.0
        data = [(n, lambda_ * n) for n in [1, 2, 4, 8, 16]]
        model = USLModel.fit(data)

        assert model.lambda_ == pytest.approx(lambda_, rel=0.01)
        assert model.sigma == pytest.approx(0.0, abs=0.01)
        assert model.kappa == pytest.approx(0.0, abs=0.01)

    def test_fit_recovers_contention(self):
        """Amdahl-only data → recovers known σ."""
        true = USLModel(lambda_=100.0, sigma=0.1, kappa=0.0)
        data = [(n, true.throughput(n)) for n in [1, 2, 4, 8, 16, 32]]
        fitted = USLModel.fit(data)

        assert fitted.lambda_ == pytest.approx(100.0, rel=0.02)
        assert fitted.sigma == pytest.approx(0.1, abs=0.02)
        assert fitted.kappa == pytest.approx(0.0, abs=0.01)

    def test_fit_recovers_retrograde(self):
        """Retrograde data → recovers κ > 0."""
        true = USLModel(lambda_=100.0, sigma=0.05, kappa=0.005)
        data = [(n, true.throughput(n)) for n in [1, 2, 4, 8, 16, 32, 64]]
        fitted = USLModel.fit(data)

        assert fitted.lambda_ == pytest.approx(100.0, rel=0.02)
        assert fitted.sigma == pytest.approx(0.05, abs=0.02)
        assert fitted.kappa == pytest.approx(0.005, abs=0.002)

    def test_fit_roundtrip_predictions(self):
        """Fitted model reproduces input data."""
        true = USLModel(lambda_=200.0, sigma=0.08, kappa=0.002)
        data = [(n, true.throughput(n)) for n in [1, 2, 4, 8, 16, 32]]
        fitted = USLModel.fit(data)

        for n, x in data:
            assert fitted.throughput(n) == pytest.approx(x, rel=0.01)

    def test_fit_insufficient_data(self):
        with pytest.raises(ValueError, match="Need >= 3"):
            USLModel.fit([(1, 100), (2, 190)])

    def test_fit_single_point_raises(self):
        with pytest.raises(ValueError, match="Need >= 3"):
            USLModel.fit([(1, 100)])

    def test_fit_zero_throughput_raises(self):
        with pytest.raises(ValueError, match="Throughput must be positive"):
            USLModel.fit([(1, 100), (2, 0), (4, 300)])

    def test_fit_exactly_three_points(self):
        """Minimum viable fit with exactly 3 points."""
        true = USLModel(lambda_=100.0, sigma=0.1, kappa=0.01)
        data = [(1, true.throughput(1)), (4, true.throughput(4)), (16, true.throughput(16))]
        fitted = USLModel.fit(data)

        # Should be reasonably close
        assert fitted.lambda_ == pytest.approx(100.0, rel=0.05)
        assert fitted.sigma == pytest.approx(0.1, abs=0.05)
        assert fitted.kappa == pytest.approx(0.01, abs=0.005)


# ============================================================
# Little's Law tests
# ============================================================


class TestLittlesLaw:
    """L = λ · W."""

    def test_basic(self):
        assert littles_law(10.0, 5.0) == pytest.approx(50.0)

    def test_zero_throughput(self):
        assert littles_law(0.0, 5.0) == pytest.approx(0.0)

    def test_zero_wait(self):
        assert littles_law(10.0, 0.0) == pytest.approx(0.0)

    def test_high_latency(self):
        """High latency → proportionally large queue depth."""
        assert littles_law(10.0, 100.0) == pytest.approx(1000.0)

    def test_fractional(self):
        assert littles_law(3.5, 2.0) == pytest.approx(7.0)


class TestLittlesLawCheck:
    """Validation of observed metrics against Little's Law."""

    def test_within_tolerance(self):
        # predicted = 10 * 5 = 50, observed = 48, deviation = 2/48 ≈ 0.042
        result = littles_law_check(queue_depth=48.0, throughput=10.0, avg_wait=5.0)
        assert result.within_tolerance is True
        assert result.predicted_queue_depth == pytest.approx(50.0)
        assert result.deviation == pytest.approx(2.0 / 48.0, rel=0.01)

    def test_exceeds_tolerance(self):
        # predicted = 10 * 5 = 50, observed = 30, deviation = 20/30 ≈ 0.667
        result = littles_law_check(queue_depth=30.0, throughput=10.0, avg_wait=5.0)
        assert result.within_tolerance is False
        assert result.deviation > 0.15

    def test_exact_match(self):
        result = littles_law_check(queue_depth=50.0, throughput=10.0, avg_wait=5.0)
        assert result.within_tolerance is True
        assert result.deviation == pytest.approx(0.0)

    def test_zero_observed_and_predicted(self):
        result = littles_law_check(queue_depth=0.0, throughput=0.0, avg_wait=5.0)
        assert result.within_tolerance is True
        assert result.deviation == 0.0

    def test_zero_observed_nonzero_predicted(self):
        result = littles_law_check(queue_depth=0.0, throughput=10.0, avg_wait=5.0)
        assert result.within_tolerance is False
        assert result.deviation == float("inf")

    def test_custom_tolerance(self):
        # deviation ≈ 0.042, tolerance=0.03 → should fail
        result = littles_law_check(
            queue_depth=48.0, throughput=10.0, avg_wait=5.0, tolerance=0.03
        )
        assert result.within_tolerance is False

    def test_result_fields(self):
        result = littles_law_check(queue_depth=100.0, throughput=20.0, avg_wait=5.0)
        assert isinstance(result, LittlesLawResult)
        assert result.observed_queue_depth == 100.0
        assert result.throughput == 20.0
        assert result.avg_wait == 5.0
        assert result.predicted_queue_depth == pytest.approx(100.0)


# ============================================================
# Integration: metric shape compatibility
# ============================================================


class TestMetricShapeIntegration:
    """Verify models work with data shaped like existing enzu metrics."""

    def test_usl_from_concurrency_stats_shape(self):
        """Build USL from ConcurrencyStats-shaped data (concurrent, throughput)."""
        stats = [
            {"concurrent": 1, "throughput": 100.0},
            {"concurrent": 2, "throughput": 185.0},
            {"concurrent": 4, "throughput": 320.0},
            {"concurrent": 8, "throughput": 480.0},
            {"concurrent": 16, "throughput": 550.0},
        ]
        data = [(s["concurrent"], s["throughput"]) for s in stats]
        model = USLModel.fit(data)

        # Model should be usable for prediction
        assert model.throughput(1) == pytest.approx(model.lambda_, rel=0.05)
        assert model.throughput(32) > 0

    def test_littles_law_with_queue_stats_shape(self):
        """Validate against QueueStats-shaped data."""
        queue_stats = {
            "queue_depth": 25.0,
            "avg_latency": 0.5,
            "throughput": 48.0,
        }
        result = littles_law_check(
            queue_depth=queue_stats["queue_depth"],
            throughput=queue_stats["throughput"],
            avg_wait=queue_stats["avg_latency"],
        )
        # predicted = 48 * 0.5 = 24, observed = 25, deviation = 1/25 = 0.04
        assert result.within_tolerance is True
        assert result.predicted_queue_depth == pytest.approx(24.0)

    def test_usl_predict_then_validate_queue(self):
        """End-to-end: fit USL, predict throughput, then validate with Little's Law."""
        true_model = USLModel(lambda_=200.0, sigma=0.06, kappa=0.001)
        samples = [(n, true_model.throughput(n)) for n in [1, 2, 4, 8, 16, 32]]

        fitted = USLModel.fit(samples)
        predicted_throughput = fitted.throughput(10)

        # Simulate observed queue: 10 concurrent, avg_wait = 0.2s
        avg_wait = 0.2
        observed_depth = predicted_throughput * avg_wait * 1.05  # 5% noise

        result = littles_law_check(
            queue_depth=observed_depth,
            throughput=predicted_throughput,
            avg_wait=avg_wait,
        )
        assert result.within_tolerance is True
