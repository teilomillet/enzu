from __future__ import annotations

from enzu.openresponses import openresponses_openapi_schema


def test_openresponses_openapi_schema_loads() -> None:
    # Covers importlib.resources path resolution for bundled schema.
    schema = openresponses_openapi_schema()

    assert isinstance(schema, dict)
    assert "openapi" in schema
    assert "paths" in schema
