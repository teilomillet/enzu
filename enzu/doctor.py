"""
enzu doctor - Environment validation and diagnostics.

Checks:
1. Python version compatibility
2. Required dependencies installed
3. API keys configured
4. Docker availability (for container isolation)
5. Sandbox security profile

Usage:
    enzu doctor
    enzu doctor --verbose
    enzu doctor --json
"""
from __future__ import annotations

import os
import sys
import json
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    severity: str = "error"  # "error", "warning", "info"


@dataclass
class DoctorReport:
    """Complete diagnostic report."""
    checks: List[CheckResult] = field(default_factory=list)
    passed: bool = True
    warnings: int = 0
    errors: int = 0

    def add_check(self, check: CheckResult) -> None:
        self.checks.append(check)
        if not check.passed:
            if check.severity == "error":
                self.errors += 1
                self.passed = False
            elif check.severity == "warning":
                self.warnings += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "severity": c.severity,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


def check_python_version() -> CheckResult:
    """Check Python version is 3.9+."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    if version >= (3, 9):
        return CheckResult(
            name="python_version",
            passed=True,
            message=f"Python {version_str}",
            details={"version": version_str},
        )
    return CheckResult(
        name="python_version",
        passed=False,
        message=f"Python {version_str} (requires 3.9+)",
        details={"version": version_str, "required": "3.9+"},
    )


def check_required_deps() -> CheckResult:
    """Check required dependencies are installed."""
    required = [
        "openai",
        "pydantic",
        "httpx",
        "tiktoken",
    ]
    missing = []
    versions: Dict[str, str] = {}

    for pkg in required:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            missing.append(pkg)
        else:
            try:
                mod = __import__(pkg)
                versions[pkg] = getattr(mod, "__version__", "installed")
            except Exception:
                versions[pkg] = "installed"

    if missing:
        return CheckResult(
            name="required_dependencies",
            passed=False,
            message=f"Missing: {', '.join(missing)}",
            details={"missing": missing, "installed": versions},
        )
    return CheckResult(
        name="required_dependencies",
        passed=True,
        message=f"All {len(required)} required packages installed",
        details={"packages": versions},
    )


def check_api_keys() -> CheckResult:
    """Check API keys are configured."""
    api_keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
        "EXA_API_KEY": os.environ.get("EXA_API_KEY"),
    }

    configured = [k for k, v in api_keys.items() if v]

    # Mask the actual keys for security
    status = {k: "configured" if v else "missing" for k, v in api_keys.items()}

    if not configured:
        return CheckResult(
            name="api_keys",
            passed=False,
            message="No API keys configured",
            details={"status": status},
            severity="warning",
        )
    return CheckResult(
        name="api_keys",
        passed=True,
        message=f"{len(configured)} API key(s) configured",
        details={"status": status, "configured": configured},
    )


def check_container_runtime() -> CheckResult:
    """Check container runtime (Podman or Docker) is available for isolation."""
    try:
        from enzu.isolation.container import is_container_available
        from enzu.isolation.runtime import detect_runtime
        available = is_container_available()
        if available:
            try:
                runtime = detect_runtime()
                runtime_name = runtime.value.capitalize()
            except RuntimeError:
                runtime_name = "Container runtime"
            return CheckResult(
                name="container",
                passed=True,
                message=f"{runtime_name} available for container isolation",
                severity="info",
            )
        return CheckResult(
            name="container",
            passed=True,  # Not required, so still "passed"
            message="No container runtime available (Podman/Docker not found)",
            severity="info",
        )
    except ImportError:
        return CheckResult(
            name="container",
            passed=True,
            message="Container module not loaded",
            severity="info",
        )


def check_optional_deps() -> CheckResult:
    """Check optional dependencies."""
    optional = {
        "logfire": "telemetry",
        "fastapi": "server",
        "uvicorn": "server",
        "exa_py": "search (Exa)",
    }
    installed = []
    missing = []

    for pkg, purpose in optional.items():
        spec = importlib.util.find_spec(pkg.replace("-", "_"))
        if spec:
            installed.append(f"{pkg} ({purpose})")
        else:
            missing.append(f"{pkg} ({purpose})")

    return CheckResult(
        name="optional_dependencies",
        passed=True,  # Optional deps never fail
        message=f"{len(installed)} optional packages installed",
        details={"installed": installed, "available": missing},
        severity="info",
    )


def check_security_profile() -> CheckResult:
    """Check current security profile configuration."""
    # Read from environment or config
    profile = os.environ.get("ENZU_SECURITY_PROFILE", "strict")
    valid_profiles = ["strict", "dev"]

    if profile not in valid_profiles:
        return CheckResult(
            name="security_profile",
            passed=False,
            message=f"Invalid security profile: {profile}",
            details={"current": profile, "valid": valid_profiles},
            severity="warning",
        )

    if profile == "dev":
        return CheckResult(
            name="security_profile",
            passed=True,
            message="Security profile: dev (network/filesystem enabled)",
            details={"profile": profile},
            severity="warning",
        )
    return CheckResult(
        name="security_profile",
        passed=True,
        message="Security profile: strict (sandbox hardened)",
        details={"profile": profile},
        severity="info",
    )


def run_doctor(verbose: bool = False) -> DoctorReport:
    """Run all diagnostic checks."""
    report = DoctorReport()

    # Run all checks
    report.add_check(check_python_version())
    report.add_check(check_required_deps())
    report.add_check(check_api_keys())
    report.add_check(check_container_runtime())
    report.add_check(check_optional_deps())
    report.add_check(check_security_profile())

    return report


def print_report(report: DoctorReport, verbose: bool = False) -> None:
    """Print the doctor report to stdout."""
    print("\nenzu doctor - Environment Diagnostics\n")
    print("=" * 50)

    for check in report.checks:
        if check.passed:
            icon = "\u2713"  # checkmark
            if check.severity == "warning":
                icon = "\u26a0"  # warning
        else:
            icon = "\u2717"  # X mark

        print(f"  {icon} {check.name}: {check.message}")

        if verbose and check.details:
            for key, value in check.details.items():
                if isinstance(value, list):
                    for item in value:
                        print(f"      - {item}")
                elif isinstance(value, dict):
                    for k, v in value.items():
                        print(f"      {k}: {v}")
                else:
                    print(f"      {key}: {value}")

    print("=" * 50)
    if report.passed:
        print(f"\nStatus: OK ({report.warnings} warnings)")
    else:
        print(f"\nStatus: FAILED ({report.errors} errors, {report.warnings} warnings)")
    print()


def main() -> int:
    """CLI entry point for `enzu doctor`."""
    import argparse

    parser = argparse.ArgumentParser(description="Check enzu environment")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    report = run_doctor(verbose=args.verbose)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report, verbose=args.verbose)

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
