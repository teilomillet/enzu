"""
Scalability models for capacity planning.

Provides:
- USLModel: Universal Scalability Law — predict throughput at concurrency N
- LittlesLawResult: Validate queue steady-state via L = λ·W
- littles_law / littles_law_check: Convenience functions

No external dependencies — uses pure Python math for fitting.

Usage:
    from enzu.scalability import USLModel, littles_law_check

    # Predict throughput from measured samples
    samples = [(1, 100), (2, 190), (4, 340), (8, 500)]
    model = USLModel.fit(samples)
    print(model.throughput(16))
    print(model.peak_concurrency())

    # Validate queue metrics
    result = littles_law_check(queue_depth=50, throughput=10, avg_wait=4.8)
    print(result.within_tolerance)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class USLModel:
    """Universal Scalability Law model.

    X(N) = lambda_ * N / (1 + sigma * (N - 1) + kappa * N * (N - 1))

    Parameters:
        lambda_: throughput at N=1 (single-thread baseline)
        sigma: contention coefficient (serialization, 0 <= sigma <= 1)
        kappa: coherency coefficient (crosstalk, 0 <= kappa <= 1)
    """

    lambda_: float
    sigma: float
    kappa: float

    def __post_init__(self) -> None:
        if self.lambda_ <= 0:
            raise ValueError(f"lambda_ must be positive, got {self.lambda_}")

    def throughput(self, n: int) -> float:
        """Predicted throughput at concurrency level N."""
        if n <= 0:
            return 0.0
        denom = 1.0 + self.sigma * (n - 1) + self.kappa * n * (n - 1)
        return self.lambda_ * n / denom

    def peak_concurrency(self) -> int:
        """N* where throughput peaks (discrete), near sqrt((1 - sigma) / kappa)."""
        if self.kappa <= 0:
            raise ValueError(
                "peak_concurrency undefined when kappa <= 0 (no retrograde region)"
            )
        if self.sigma >= 1.0:
            raise ValueError(
                "peak_concurrency undefined when sigma >= 1 (no peak exists)"
            )
        continuous = math.sqrt((1.0 - self.sigma) / self.kappa)
        lo = max(1, int(math.floor(continuous)))
        hi = lo + 1
        if self.throughput(hi) > self.throughput(lo):
            return hi
        return lo

    def peak_throughput(self) -> float:
        """Maximum achievable throughput X(N*)."""
        n_star = self.peak_concurrency()
        return self.throughput(n_star)

    @classmethod
    def fit(cls, data: List[Tuple[int, float]]) -> USLModel:
        """Fit USL coefficients from measured (concurrency, throughput) samples.

        Uses linearized least squares (no external dependencies).

        Rewrite: N/X(N) = 1/lambda + (sigma/lambda)*(N-1) + (kappa/lambda)*N*(N-1)

        Let y = N/X, and regressors:
            c0 = 1          -> coefficient = 1/lambda
            c1 = N - 1      -> coefficient = sigma/lambda
            c2 = N*(N - 1)  -> coefficient = kappa/lambda

        Solve via normal equations: (A^T A) b = A^T y
        """
        if len(data) < 3:
            raise ValueError(f"Need >= 3 data points for USL fit, got {len(data)}")

        m = len(data)

        # Build A (m x 3) and y (m x 1)
        A: List[List[float]] = []
        y: List[float] = []
        for n, x in data:
            if n < 1:
                raise ValueError(f"Concurrency must be >= 1, got {n}")
            if x <= 0:
                raise ValueError(f"Throughput must be positive, got {x} at N={n}")
            A.append([1.0, float(n - 1), float(n * (n - 1))])
            y.append(float(n) / x)

        # A^T A  (3x3)
        ATA = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(m):
                    s += A[k][i] * A[k][j]
                ATA[i][j] = s

        # A^T y  (3x1)
        ATy = [0.0] * 3
        for i in range(3):
            s = 0.0
            for k in range(m):
                s += A[k][i] * y[k]
            ATy[i] = s

        # Solve 3x3 system via Cramer's rule
        b = _solve_3x3(ATA, ATy)

        inv_lambda = b[0]
        if inv_lambda <= 0:
            raise ValueError("Fitted 1/lambda <= 0; data does not fit USL model")

        lambda_ = 1.0 / inv_lambda
        sigma = max(0.0, b[1] * lambda_)
        kappa = max(0.0, b[2] * lambda_)

        return cls(lambda_=lambda_, sigma=sigma, kappa=kappa)


def _det3(m: List[List[float]]) -> float:
    """Determinant of a 3x3 matrix."""
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _solve_3x3(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax = b for a 3x3 system using Cramer's rule."""
    det = _det3(A)
    if abs(det) < 1e-15:
        raise ValueError("Singular matrix in USL fit — data may be collinear")

    result = []
    for col in range(3):
        # Replace column `col` with b
        modified = [row[:] for row in A]
        for row in range(3):
            modified[row][col] = b[row]
        result.append(_det3(modified) / det)
    return result


# ---------------------------------------------------------------------------
# Little's Law
# ---------------------------------------------------------------------------


@dataclass
class LittlesLawResult:
    """Result of a Little's Law validation check."""

    observed_queue_depth: float
    predicted_queue_depth: float  # throughput * avg_wait
    throughput: float
    avg_wait: float
    deviation: float  # relative deviation
    within_tolerance: bool


def littles_law(throughput: float, avg_wait: float) -> float:
    """Predict steady-state queue depth: L = lambda * W."""
    return throughput * avg_wait


def littles_law_check(
    queue_depth: float,
    throughput: float,
    avg_wait: float,
    tolerance: float = 0.15,
) -> LittlesLawResult:
    """Validate observed queue metrics against Little's Law.

    Returns a LittlesLawResult with deviation and whether it's within tolerance.
    """
    if queue_depth < 0:
        raise ValueError(f"queue_depth must be >= 0, got {queue_depth}")
    if throughput < 0:
        raise ValueError(f"throughput must be >= 0, got {throughput}")
    if avg_wait < 0:
        raise ValueError(f"avg_wait must be >= 0, got {avg_wait}")

    predicted = littles_law(throughput, avg_wait)

    if queue_depth == 0 and predicted == 0:
        deviation = 0.0
    elif queue_depth == 0:
        deviation = float("inf")
    else:
        deviation = abs(predicted - queue_depth) / queue_depth

    return LittlesLawResult(
        observed_queue_depth=queue_depth,
        predicted_queue_depth=predicted,
        throughput=throughput,
        avg_wait=avg_wait,
        deviation=deviation,
        within_tolerance=deviation <= tolerance,
    )


# ---------------------------------------------------------------------------
# Utility helpers (migrated from enzu.isolation.scaling)
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], p: float) -> float:
    """Interpolated percentile calculation."""
    if not values:
        return 0.0
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def relative_error(observed: float, predicted: float) -> float:
    """Relative error: |obs - pred| / max(|obs|, 1e-12)."""
    denom = max(abs(observed), 1e-12)
    return abs(observed - predicted) / denom


def average_population(events: Sequence[Tuple[float, int]], total_seconds: float) -> float:
    """Time-weighted average in-flight count from (timestamp, delta) events."""
    if total_seconds <= 0:
        return 0.0
    if not events:
        return 0.0
    ordered = sorted((float(t), int(d)) for t, d in events)
    active = 0
    prev_t = 0.0
    area = 0.0
    for t, delta in ordered:
        clamped = min(max(t, 0.0), total_seconds)
        if clamped > prev_t:
            area += active * (clamped - prev_t)
            prev_t = clamped
        active += delta
    if prev_t < total_seconds:
        area += active * (total_seconds - prev_t)
    return area / total_seconds


# ---------------------------------------------------------------------------
# Backward-compatible wrappers (procedural API from enzu.isolation.scaling)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class USLFit:
    """Result of a procedural USL grid-search fit."""

    sigma: float
    kappa: float
    x1: float
    sse: float
    r2: float
    n_opt: float
    max_throughput_estimate: float


def usl_throughput(n: int, x1: float, sigma: float, kappa: float) -> float:
    """Predicted throughput at concurrency *n* using explicit USL parameters."""
    return USLModel(lambda_=x1, sigma=sigma, kappa=kappa).throughput(n)


def fit_usl(points: Iterable[Tuple[int, float]]) -> USLFit:
    """Fit USL via grid search (backward-compatible procedural API).

    Wraps :class:`USLModel.fit` internally, then computes SSE / R² and
    returns a frozen :class:`USLFit` dataclass.
    """
    prepared = sorted((int(n), float(x)) for n, x in points if int(n) > 0)
    if not prepared:
        raise ValueError("at least one point is required")

    model = USLModel.fit(prepared)

    # Compute SSE and R²
    ys = [x for _, x in prepared]
    mean_y = sum(ys) / len(ys)
    sst = sum((y - mean_y) ** 2 for y in ys)
    sse = sum((x - model.throughput(n)) ** 2 for n, x in prepared)

    if sst <= 0:
        r2 = 1.0 if sse <= 1e-12 else 0.0
    else:
        r2 = max(0.0, 1.0 - (sse / sst))

    # Peak concurrency
    if model.kappa > 0.0 and model.sigma < 1.0:
        n_opt = math.sqrt((1.0 - model.sigma) / model.kappa)
    else:
        n_opt = float(max(n for n, _ in prepared))
    n_opt_eval = max(1, int(round(n_opt)))
    max_tp = model.throughput(n_opt_eval)

    return USLFit(
        sigma=model.sigma,
        kappa=model.kappa,
        x1=model.lambda_,
        sse=sse,
        r2=r2,
        n_opt=n_opt,
        max_throughput_estimate=max_tp,
    )
