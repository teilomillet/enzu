from __future__ import annotations

from typing import Any, Dict

from enzu.tools.exa import ExaClient


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return {"results": []}


class _FakeHttpxClient:
    def __init__(self) -> None:
        self.last_json: Dict[str, Any] = {}

    def post(self, _path: str, json: Dict[str, Any]) -> _FakeResponse:
        self.last_json = json
        return _FakeResponse()


def _client_with_fake_httpx() -> tuple[ExaClient, _FakeHttpxClient]:
    client = ExaClient(api_key="test")
    fake_http = _FakeHttpxClient()
    # Replace httpx transport to inspect payload construction.
    client._client = fake_http
    return client, fake_http


def test_exa_client_search_omits_max_characters_when_none() -> None:
    client, fake_http = _client_with_fake_httpx()

    client.search("query", max_characters=None)

    contents = fake_http.last_json["contents"]
    assert contents["text"] is True


def test_exa_client_search_includes_max_characters_when_set() -> None:
    client, fake_http = _client_with_fake_httpx()

    client.search("query", max_characters=1234)

    contents = fake_http.last_json["contents"]
    assert contents["text"]["maxCharacters"] == 1234


def test_exa_client_contents_omits_max_characters_when_none() -> None:
    client, fake_http = _client_with_fake_httpx()

    client.get_contents(["https://example.com"], max_characters=None)

    contents = fake_http.last_json["contents"]
    assert contents["text"] is True


def test_exa_client_find_similar_omits_max_characters_when_none() -> None:
    client, fake_http = _client_with_fake_httpx()

    client.find_similar("https://example.com", max_characters=None)

    contents = fake_http.last_json["contents"]
    assert contents["text"] is True
