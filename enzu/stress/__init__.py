"""
Stress Testing Harness: Test p95 behavior before production.

Inject failure conditions (429s, timeouts, latency) and measure how
your system behaves under stress using the existing metrics infrastructure.

Usage:
    from enzu.stress import Scenario, run_scenarios, rules

    scenarios = [
        Scenario("baseline", []),
        Scenario("high_429s", [rules.error_rate(0.3, rules.rate_limit_429)]),
        Scenario("slow_provider", [rules.latency(0.5, 2.0)]),
    ]

    reports = run_scenarios(
        task=lambda p: my_run_function(p),
        provider_factory=lambda: MockProvider(),
        scenarios=scenarios,
    )
"""

from enzu.stress.scenario import Scenario, Fault, CallContext
from enzu.stress.harness import run_scenarios, ScenarioReport, format_report
from enzu.stress import rules
from enzu.stress import faults

__all__ = [
    "Scenario",
    "Fault",
    "CallContext",
    "run_scenarios",
    "ScenarioReport",
    "format_report",
    "rules",
    "faults",
]
