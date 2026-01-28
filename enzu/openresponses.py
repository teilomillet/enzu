from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Dict


def openresponses_openapi_schema() -> Dict[str, Any]:
    """Load the bundled Open Responses OpenAPI schema."""
    schema_path = files("enzu.spec").joinpath("openresponses_openapi.json")
    return json.loads(schema_path.read_text(encoding="utf-8"))
