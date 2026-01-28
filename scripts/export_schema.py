from __future__ import annotations

import json
from pathlib import Path

from enzu.schema import (
    report_schema,
    run_payload_schema,
    schema_bundle,
    task_input_schema,
    task_spec_schema,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    # Export schemas for humans and models to inspect without running Python.
    base_dir = Path(__file__).resolve().parents[1] / "docs" / "schema"
    base_dir.mkdir(parents=True, exist_ok=True)
    _write_json(base_dir / "task_input.json", task_input_schema())
    _write_json(base_dir / "task_spec.json", task_spec_schema())
    _write_json(base_dir / "run_payload.json", run_payload_schema())
    _write_json(base_dir / "report.json", report_schema())
    # Bundle matches enzu --print-schema output for offline use.
    _write_json(base_dir / "bundle.json", schema_bundle())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
