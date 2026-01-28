from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

OPENRESPONSES_OPENAPI_URL = "https://www.openresponses.org/openapi/openapi.json"


def main() -> int:
    target = Path(__file__).resolve().parents[1] / "enzu" / "spec" / "openresponses_openapi.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(OPENRESPONSES_OPENAPI_URL, timeout=10) as response:
        payload = response.read().decode("utf-8")
    parsed = json.loads(payload)
    target.write_text(json.dumps(parsed, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
