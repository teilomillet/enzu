"""Optional Logfire integration for tracing and logs."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

_logfire = None
_configured = False


def _load_logfire():
    global _logfire
    if _logfire is not None:
        return _logfire
    try:
        import logfire
    except Exception:
        _logfire = False
        return _logfire
    _logfire = logfire
    return _logfire


def _env_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _stderr_enabled() -> bool:
    value = os.getenv("ENZU_TELEMETRY_STDERR")
    if value is None:
        return True
    return _env_truthy(value)


def _console_setting():
    # Prefer explicit user control; default to console output on.
    env_console = os.getenv("ENZU_LOGFIRE_CONSOLE")
    if env_console is not None:
        return None if _env_truthy(env_console) else False
    if os.getenv("LOGFIRE_CONSOLE") is not None:
        return None
    return None


def stream_enabled() -> bool:
    return _env_truthy(os.getenv("ENZU_LOGFIRE_STREAM"))


def enabled() -> bool:
    logfire = _load_logfire()
    if not logfire:
        return False
    flag = os.getenv("ENZU_LOGFIRE")
    if flag is not None:
        return _env_truthy(flag)
    return True


def configure() -> bool:
    logfire = _load_logfire()
    if not logfire or not enabled():
        return False
    global _configured
    if not _configured:
        try:
            logfire.configure(console=_console_setting())
            _configured = True
        except Exception:
            return False
        _instrument_logfire(logfire)
    return True


def _instrument_logfire(logfire: Any) -> None:
    # Instrument OpenAI calls so provider requests show as spans in traces.
    flag = os.getenv("ENZU_LOGFIRE_INSTRUMENT_OPENAI")
    if flag is None or _env_truthy(flag):
        try:
            logfire.instrument_openai()
        except Exception:
            pass


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    logfire = _load_logfire()
    if not logfire or not enabled():
        yield
        return
    if not configure():
        yield
        return
    try:
        ctx = logfire.span(name, **attrs)
        ctx.__enter__()
    except Exception:
        yield
        return
    try:
        yield
    except Exception as exc:
        try:
            ctx.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            pass
        raise
    else:
        try:
            ctx.__exit__(None, None, None)
        except Exception:
            pass


def log(level: str, message: str, **attrs: Any) -> None:
    logfire = _load_logfire()
    if not logfire or not enabled():
        if _stderr_enabled():
            try:
                print(f"[telemetry] {message} {attrs}", file=sys.stderr)
            except Exception:
                pass
        return
    configure()
    fn = getattr(logfire, level, None) or logfire.info
    try:
        fn(message, **attrs)
    except Exception:
        return
    if _stderr_enabled():
        try:
            print(f"[telemetry] {message} {attrs}", file=sys.stderr)
        except Exception:
            return


def context() -> Optional[Dict[str, str]]:
    logfire = _load_logfire()
    if not logfire or not enabled():
        return None
    configure()
    try:
        return logfire.get_context()
    except Exception:
        return None
