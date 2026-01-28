from __future__ import annotations

import builtins
import json
import sys
from typing import Iterator

import enzu.cli as cli
from tests.providers.mock_provider import MockProvider


def _iter_input(values: list[str]) -> Iterator[str]:
    for value in values:
        yield value


class _TTYStdin:
    def isatty(self) -> bool:  # pragma: no cover - trivial
        return True


def test_cli_print_schema_bundle(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["enzu", "--print-schema"])
    exit_code = cli.main()
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert {"meta", "task_input", "task_spec", "run_payload", "report"} <= set(payload.keys())


def test_cli_guided_mode_chat(capsys, monkeypatch) -> None:
    # Guided mode should execute the no-args TTY path and emit a report.
    inputs = _iter_input(["", "", "mock-model", "Say hello."])
    monkeypatch.setattr(builtins, "input", lambda: next(inputs))
    monkeypatch.setattr(sys, "stdin", _TTYStdin())
    monkeypatch.setattr(sys, "argv", ["enzu"])
    monkeypatch.setattr(cli, "list_providers", lambda: ["openrouter"])
    monkeypatch.setattr(cli, "resolve_provider", lambda *args, **kwargs: MockProvider(main_outputs=["Hello"]))

    exit_code = cli.main()
    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["success"] is True
    assert report["output_text"] == "Hello"
