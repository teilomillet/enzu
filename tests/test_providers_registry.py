from __future__ import annotations

import pytest

from enzu.providers.registry import (
    PROVIDERS,
    get_provider_config,
    list_providers,
    register_provider,
    validate_provider_config,
)


class TestBuiltinProviders:
    """All built-in providers must have valid configs."""

    @pytest.mark.parametrize("name", list(PROVIDERS.keys()))
    def test_builtin_provider_valid(self, name):
        """Each built-in provider passes validation."""
        config = PROVIDERS[name]
        validate_provider_config(name, config)

    @pytest.mark.parametrize("name", list(PROVIDERS.keys()))
    def test_builtin_base_url_format(self, name):
        """base_url ends with /v1 or similar (OpenAI convention)."""
        config = PROVIDERS[name]
        base_url = config.get("base_url")
        if base_url:
            assert "/v1" in base_url or "/v1beta" in base_url


class TestRegisterProvider:
    """User registration API."""

    def test_register_valid(self):
        register_provider("testcorp", base_url="https://api.testcorp.com/v1")
        assert "testcorp" in PROVIDERS
        assert PROVIDERS["testcorp"]["base_url"] == "https://api.testcorp.com/v1"

    def test_register_invalid_url(self):
        with pytest.raises(ValueError, match="Invalid base_url"):
            register_provider("bad", base_url="not-a-url")

    def test_register_empty_name(self):
        with pytest.raises(ValueError, match="non-empty string"):
            register_provider("", base_url="https://api.example.com/v1")

    def test_register_localhost(self):
        """Local providers (ollama-style) should work."""
        register_provider("mylocal", base_url="http://localhost:8080/v1")
        assert PROVIDERS["mylocal"]["base_url"] == "http://localhost:8080/v1"


class TestValidation:
    """Config validation rules."""

    def test_valid_https(self):
        validate_provider_config("x", {"base_url": "https://api.example.com/v1"})

    def test_valid_http_localhost(self):
        validate_provider_config("x", {"base_url": "http://localhost:11434/v1"})

    def test_invalid_no_scheme(self):
        with pytest.raises(ValueError):
            validate_provider_config("x", {"base_url": "api.example.com/v1"})

    def test_invalid_garbage(self):
        with pytest.raises(ValueError):
            validate_provider_config("x", {"base_url": ":::invalid:::"})

    def test_empty_config_ok(self):
        """OpenAI-style (no base_url) is valid."""
        validate_provider_config("openai", {})

    def test_supports_responses_flag(self):
        """Test that supports_responses flag is properly handled."""
        assert PROVIDERS["openai"].get("supports_responses") is True
        assert PROVIDERS["openrouter"].get("supports_responses") is True
        assert PROVIDERS["groq"].get("supports_responses") is False


class TestListProviders:
    """Provider discovery."""

    def test_list_includes_builtins(self):
        providers = list_providers()
        assert "openai" in providers
        assert "openrouter" in providers
        assert "groq" in providers

    def test_list_includes_registered(self):
        register_provider("custom123", base_url="https://api.custom.com/v1")
        assert "custom123" in list_providers()


class TestGetProviderConfig:
    """Config retrieval."""

    def test_get_existing(self):
        config = get_provider_config("openai")
        assert config is not None
        assert isinstance(config, dict)

    def test_get_nonexistent(self):
        config = get_provider_config("nonexistent")
        assert config is None
