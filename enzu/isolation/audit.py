"""
Audit logging for government-grade deployments (Phase 4).

Provides structured logging for request tracking WITHOUT content:
- Request lifecycle events (submitted, started, completed, failed)
- Resource usage (execution time, tokens, LLM calls)
- Security events (sandbox violations, admission rejections)

Security guarantees:
- No request content logged (input, output, prompts)
- No PII logged
- Request IDs for correlation
- Structured JSON for SIEM integration

Usage:
    from enzu.isolation.audit import get_audit_logger, AuditEvent

    audit = get_audit_logger()
    audit.log_request_submitted(request_id, conversation_id, node_id)
    audit.log_request_completed(request_id, execution_time_ms, tokens_used)

Configuration:
    from enzu.isolation.audit import configure_audit_logger

    # Log to file (production)
    configure_audit_logger(
        output_path="/var/log/enzu/audit.json",
        include_timestamps=True,
    )

    # Log to stdout (development)
    configure_audit_logger(output_path=None)

"""

from __future__ import annotations

import json
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TextIO
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""

    # Request lifecycle
    REQUEST_SUBMITTED = "request_submitted"
    REQUEST_STARTED = "request_started"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILED = "request_failed"
    REQUEST_TIMEOUT = "request_timeout"

    # Admission control
    ADMISSION_ACCEPTED = "admission_accepted"
    ADMISSION_REJECTED = "admission_rejected"

    # Security events
    SANDBOX_VIOLATION = "sandbox_violation"
    IMPORT_BLOCKED = "import_blocked"
    RESOURCE_EXCEEDED = "resource_exceeded"
    NETWORK_BLOCKED = "network_blocked"

    # Scheduler events
    NODE_REGISTERED = "node_registered"
    NODE_UNREGISTERED = "node_unregistered"
    NODE_UNHEALTHY = "node_unhealthy"

    # LLM events (no content, just counts)
    LLM_CALL_START = "llm_call_start"
    LLM_CALL_COMPLETE = "llm_call_complete"
    LLM_CALL_FAILED = "llm_call_failed"


@dataclass
class AuditEvent:
    """
    Structured audit event.

    Fields are designed to be safe for government logging:
    - No content (input_text, output_text, prompts)
    - No PII (user names, emails)
    - Request IDs for correlation
    - Timing and resource metrics
    """

    # Event metadata
    event_type: AuditEventType
    timestamp_unix: float = field(default_factory=time.time)

    # Request identification (no content)
    request_id: Optional[str] = None
    conversation_id: Optional[str] = None

    # Execution context
    node_id: Optional[str] = None
    sandbox_id: Optional[str] = None

    # Resource metrics (safe to log)
    execution_time_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    llm_calls: Optional[int] = None
    exit_code: Optional[int] = None

    # Error classification (no content, just category)
    error_category: Optional[str] = None

    # Additional safe metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        # Explicit type annotation to allow heterogeneous values (str, float, int, dict)
        data: Dict[str, Any] = {
            "event": self.event_type.value,
            "ts": self.timestamp_unix,
        }

        # Add non-None fields
        if self.request_id:
            data["request_id"] = self.request_id
        if self.conversation_id:
            data["conversation_id"] = self.conversation_id
        if self.node_id:
            data["node_id"] = self.node_id
        if self.sandbox_id:
            data["sandbox_id"] = self.sandbox_id
        if self.execution_time_ms is not None:
            data["execution_time_ms"] = self.execution_time_ms
        if self.tokens_used is not None:
            data["tokens_used"] = self.tokens_used
        if self.llm_calls is not None:
            data["llm_calls"] = self.llm_calls
        if self.exit_code is not None:
            data["exit_code"] = self.exit_code
        if self.error_category:
            data["error_category"] = self.error_category
        if self.metadata:
            data["metadata"] = self.metadata

        return data

    def to_json(self) -> str:
        """Serialize to JSON line."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


class AuditLogger:
    """
    Thread-safe audit logger for government deployments.

    Writes structured JSON events to file or stdout.
    Designed for high-throughput (10K+ req/sec) with minimal overhead.

    Security features:
    - Never logs content (input, output, prompts)
    - Structured format for SIEM integration
    - Request ID correlation
    - Buffered writes for performance
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        buffer_size: int = 100,
        flush_interval_seconds: float = 1.0,
        include_timestamps: bool = True,
    ) -> None:
        """
        Args:
            output_path: Path to log file. None = stdout.
            buffer_size: Events to buffer before flush.
            flush_interval_seconds: Max time between flushes.
            include_timestamps: Include ISO timestamp in addition to unix.
        """
        self._output_path = output_path
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval_seconds
        self._include_timestamps = include_timestamps

        # Thread-safe buffer
        self._buffer: List[str] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()

        # File handle (opened lazily)
        self._file: Optional[TextIO] = None

        # Stats
        self._events_logged = 0
        self._events_dropped = 0

    def _get_file(self) -> TextIO:
        """Get or open log file."""
        if self._file is None:
            if self._output_path:
                # Ensure directory exists
                Path(self._output_path).parent.mkdir(parents=True, exist_ok=True)
                self._file = open(self._output_path, "a", encoding="utf-8")
            else:
                import sys

                self._file = sys.stdout
        return self._file

    def log(self, event: AuditEvent) -> None:
        """Log an audit event."""
        # Format event
        data = event.to_dict()
        if self._include_timestamps:
            from datetime import datetime, timezone

            data["timestamp"] = (
                datetime.fromtimestamp(event.timestamp_unix, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

        line = json.dumps(data, separators=(",", ":"))

        # Buffer event
        with self._lock:
            self._buffer.append(line)
            self._events_logged += 1

            # Flush if buffer full or interval elapsed
            should_flush = (
                len(self._buffer) >= self._buffer_size
                or time.time() - self._last_flush >= self._flush_interval
            )

        if should_flush:
            self._flush()

    def _flush(self) -> None:
        """Flush buffer to file."""
        with self._lock:
            if not self._buffer:
                return

            lines = self._buffer.copy()
            self._buffer.clear()
            self._last_flush = time.time()

        try:
            f = self._get_file()
            for line in lines:
                f.write(line + "\n")
            f.flush()
        except Exception as e:
            logger.error("Audit log flush failed: %s", e)
            with self._lock:
                self._events_dropped += len(lines)

    def close(self) -> None:
        """Flush and close log file."""
        self._flush()
        with self._lock:
            if self._file and self._output_path:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None

    # Convenience methods for common events

    def log_request_submitted(
        self,
        request_id: str,
        conversation_id: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> None:
        """Log request submission."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.REQUEST_SUBMITTED,
                request_id=request_id,
                conversation_id=conversation_id,
                node_id=node_id,
            )
        )

    def log_request_started(
        self,
        request_id: str,
        node_id: str,
        sandbox_id: Optional[str] = None,
    ) -> None:
        """Log request execution start."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.REQUEST_STARTED,
                request_id=request_id,
                node_id=node_id,
                sandbox_id=sandbox_id,
            )
        )

    def log_request_completed(
        self,
        request_id: str,
        execution_time_ms: float,
        tokens_used: Optional[int] = None,
        llm_calls: Optional[int] = None,
        exit_code: int = 0,
    ) -> None:
        """Log successful request completion."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.REQUEST_COMPLETED,
                request_id=request_id,
                execution_time_ms=execution_time_ms,
                tokens_used=tokens_used,
                llm_calls=llm_calls,
                exit_code=exit_code,
            )
        )

    def log_request_failed(
        self,
        request_id: str,
        error_category: str,
        execution_time_ms: Optional[float] = None,
    ) -> None:
        """Log request failure (no error content, just category)."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.REQUEST_FAILED,
                request_id=request_id,
                error_category=error_category,
                execution_time_ms=execution_time_ms,
            )
        )

    def log_request_timeout(
        self,
        request_id: str,
        execution_time_ms: float,
    ) -> None:
        """Log request timeout."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.REQUEST_TIMEOUT,
                request_id=request_id,
                execution_time_ms=execution_time_ms,
                error_category="timeout",
            )
        )

    def log_admission_rejected(
        self,
        request_id: str,
        reason: str,
    ) -> None:
        """Log admission control rejection."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.ADMISSION_REJECTED,
                request_id=request_id,
                error_category=reason,
            )
        )

    def log_sandbox_violation(
        self,
        request_id: str,
        violation_type: str,
    ) -> None:
        """Log sandbox security violation."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.SANDBOX_VIOLATION,
                request_id=request_id,
                error_category=violation_type,
            )
        )

    def log_resource_exceeded(
        self,
        request_id: str,
        resource_type: str,
    ) -> None:
        """Log resource limit exceeded."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.RESOURCE_EXCEEDED,
                request_id=request_id,
                error_category=resource_type,
            )
        )

    def log_node_registered(
        self,
        node_id: str,
        capacity: int,
    ) -> None:
        """Log node registration."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.NODE_REGISTERED,
                node_id=node_id,
                metadata={"capacity": capacity},
            )
        )

    def log_node_unhealthy(
        self,
        node_id: str,
        reason: str,
    ) -> None:
        """Log node unhealthy status."""
        self.log(
            AuditEvent(
                event_type=AuditEventType.NODE_UNHEALTHY,
                node_id=node_id,
                error_category=reason,
            )
        )

    def stats(self) -> Dict[str, int]:
        """Get logging statistics."""
        with self._lock:
            return {
                "events_logged": self._events_logged,
                "events_dropped": self._events_dropped,
                "buffer_size": len(self._buffer),
            }


# Global singleton
_global_audit_logger: Optional[AuditLogger] = None
_global_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """
    Get the global audit logger.

    Creates with default settings (stdout) if not configured.
    Thread-safe singleton.
    """
    global _global_audit_logger
    if _global_audit_logger is None:
        with _global_lock:
            if _global_audit_logger is None:
                _global_audit_logger = AuditLogger()
    assert _global_audit_logger is not None
    return _global_audit_logger


def configure_audit_logger(
    output_path: Optional[str] = None,
    buffer_size: int = 100,
    flush_interval_seconds: float = 1.0,
    include_timestamps: bool = True,
) -> AuditLogger:
    """
    Configure the global audit logger.

    Call at startup before processing requests.

    Args:
        output_path: Path to log file. None = stdout.
        buffer_size: Events to buffer before flush.
        flush_interval_seconds: Max time between flushes.
        include_timestamps: Include ISO timestamp.

    Returns:
        The configured audit logger.

    Example:
        # Production: log to file
        configure_audit_logger(output_path="/var/log/enzu/audit.json")

        # Development: log to stdout
        configure_audit_logger()
    """
    global _global_audit_logger

    with _global_lock:
        # Close existing logger
        if _global_audit_logger is not None:
            _global_audit_logger.close()

        _global_audit_logger = AuditLogger(
            output_path=output_path,
            buffer_size=buffer_size,
            flush_interval_seconds=flush_interval_seconds,
            include_timestamps=include_timestamps,
        )
        return _global_audit_logger


def reset_audit_logger() -> None:
    """Reset global audit logger. For testing only."""
    global _global_audit_logger
    with _global_lock:
        if _global_audit_logger is not None:
            _global_audit_logger.close()
        _global_audit_logger = None
