"""
IPC Bridge for Container/Subprocess Isolation.

Enables llm_query/llm_batch callbacks in isolated sandboxes using
length-prefixed JSON messages over stdin/stdout pipes.

Protocol:
    <4-byte big-endian length><JSON payload>

Message Types:
    - INIT: Host -> Worker, initialization data
    - LLM_QUERY: Worker -> Host, request llm_query callback
    - LLM_BATCH: Worker -> Host, request llm_batch callback
    - LLM_RESPONSE: Host -> Worker, callback result
    - RESULT: Worker -> Host, execution result

No seccomp changes needed since read/write syscalls are already allowed.
"""

from __future__ import annotations

import json
import struct
import logging
from typing import Any, Callable, Dict, List, Optional, BinaryIO

logger = logging.getLogger(__name__)

# Message type constants
MSG_INIT = "INIT"
MSG_LLM_QUERY = "LLM_QUERY"
MSG_LLM_BATCH = "LLM_BATCH"
MSG_LLM_RESPONSE = "LLM_RESPONSE"
MSG_RESULT = "RESULT"
MSG_ERROR = "ERROR"

# Protocol limits
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB limit to prevent memory exhaustion
HEADER_SIZE = 4  # 4-byte big-endian length prefix


def encode_message(msg_type: str, payload: Dict[str, Any]) -> bytes:
    """
    Encode a message with length prefix.

    Format: <4-byte big-endian length><JSON payload>

    Args:
        msg_type: Message type constant
        payload: Message payload dictionary

    Returns:
        Encoded bytes ready for transmission

    Raises:
        ValueError: If message exceeds size limit
    """
    message = {"type": msg_type, "payload": payload}
    json_bytes = json.dumps(message).encode("utf-8")

    if len(json_bytes) > MAX_MESSAGE_SIZE:
        raise ValueError(
            f"Message size {len(json_bytes)} exceeds limit {MAX_MESSAGE_SIZE}"
        )

    length_prefix = struct.pack(">I", len(json_bytes))
    return length_prefix + json_bytes


def decode_message(data: bytes) -> Dict[str, Any]:
    """
    Decode a message from raw bytes (without length prefix).

    Args:
        data: JSON bytes (length prefix already stripped)

    Returns:
        Decoded message dict with 'type' and 'payload'
    """
    return json.loads(data.decode("utf-8"))


def read_message(
    stream: BinaryIO, timeout: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """
    Read a length-prefixed message from a binary stream.

    Args:
        stream: Binary stream to read from
        timeout: Not used directly (handled by caller via select/poll)

    Returns:
        Decoded message dict, or None if stream closed

    Raises:
        ValueError: If message exceeds size limit or is malformed
    """
    # Read length prefix
    header = stream.read(HEADER_SIZE)
    if not header:
        return None
    if len(header) < HEADER_SIZE:
        raise ValueError(f"Incomplete header: got {len(header)} bytes")

    length = struct.unpack(">I", header)[0]

    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message size {length} exceeds limit {MAX_MESSAGE_SIZE}")

    # Read message body
    data = stream.read(length)
    if len(data) < length:
        raise ValueError(f"Incomplete message: expected {length}, got {len(data)}")

    return decode_message(data)


def write_message(stream: BinaryIO, msg_type: str, payload: Dict[str, Any]) -> None:
    """
    Write a length-prefixed message to a binary stream.

    Args:
        stream: Binary stream to write to
        msg_type: Message type constant
        payload: Message payload dictionary
    """
    encoded = encode_message(msg_type, payload)
    stream.write(encoded)
    stream.flush()


class IPCBridge:
    """
    Host-side IPC handler for isolated sandbox execution.

    Manages the communication loop with a subprocess/container worker,
    handling LLM_QUERY and LLM_BATCH requests by invoking callbacks.

    Example:
        bridge = IPCBridge(
            stdin=proc.stdin,
            stdout=proc.stdout,
            llm_query=executor.sandbox_query,
            llm_batch=executor.batch_query,
        )
        result = bridge.run(init_payload, timeout=60.0)
    """

    def __init__(
        self,
        stdin: BinaryIO,
        stdout: BinaryIO,
        llm_query: Optional[Callable[[str], str]] = None,
        llm_batch: Optional[Callable[[List[str]], List[str]]] = None,
    ) -> None:
        """
        Args:
            stdin: Worker's stdin (we write to this)
            stdout: Worker's stdout (we read from this)
            llm_query: Callback for single LLM queries
            llm_batch: Callback for batch LLM queries
        """
        self._stdin = stdin
        self._stdout = stdout
        self._llm_query = llm_query
        self._llm_batch = llm_batch
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None

    def run(
        self,
        init_payload: Dict[str, Any],
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Run the IPC communication loop.

        Sends INIT message, then handles LLM requests until RESULT received.

        Args:
            init_payload: Initial payload to send to worker
            timeout: Total execution timeout in seconds

        Returns:
            Result payload from worker

        Raises:
            TimeoutError: If execution exceeds timeout
            RuntimeError: If worker returns error or protocol fails
        """
        import select
        import time

        start_time = time.time()

        # Send INIT message
        write_message(self._stdin, MSG_INIT, init_payload)

        # Communication loop
        while True:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed

            if remaining <= 0:
                raise TimeoutError(f"IPC timeout after {timeout}s")

            # Wait for data with timeout
            # Use select for cross-platform timeout support
            try:
                readable, _, _ = select.select(
                    [self._stdout], [], [], min(remaining, 1.0)
                )
            except (ValueError, OSError):
                # Stream closed
                if self._result is not None:
                    return self._result
                raise RuntimeError("Worker stream closed unexpectedly")

            if not readable:
                continue

            # Read message
            try:
                message = read_message(self._stdout)
            except Exception as e:
                raise RuntimeError(f"Failed to read message: {e}")

            if message is None:
                # Stream closed
                if self._result is not None:
                    return self._result
                raise RuntimeError("Worker closed connection")

            msg_type = message.get("type")
            payload = message.get("payload", {})

            if msg_type == MSG_RESULT:
                return payload

            elif msg_type == MSG_LLM_QUERY:
                self._handle_llm_query(payload)

            elif msg_type == MSG_LLM_BATCH:
                self._handle_llm_batch(payload)

            elif msg_type == MSG_ERROR:
                error_msg = payload.get("error", "Unknown worker error")
                raise RuntimeError(f"Worker error: {error_msg}")

            else:
                logger.warning("Unknown message type: %s", msg_type)

    def _handle_llm_query(self, payload: Dict[str, Any]) -> None:
        """Handle LLM_QUERY request from worker."""
        prompt = payload.get("prompt", "")

        try:
            if self._llm_query is not None:
                result = self._llm_query(prompt)
            else:
                result = ""
                logger.warning("llm_query called but no callback provided")

            write_message(
                self._stdin,
                MSG_LLM_RESPONSE,
                {
                    "success": True,
                    "result": result,
                },
            )
        except Exception as e:
            logger.error("llm_query callback failed: %s", e)
            write_message(
                self._stdin,
                MSG_LLM_RESPONSE,
                {
                    "success": False,
                    "error": str(e),
                },
            )

    def _handle_llm_batch(self, payload: Dict[str, Any]) -> None:
        """Handle LLM_BATCH request from worker."""
        prompts = payload.get("prompts", [])

        try:
            if self._llm_batch is not None:
                results = self._llm_batch(prompts)
            else:
                results = [""] * len(prompts)
                logger.warning("llm_batch called but no callback provided")

            write_message(
                self._stdin,
                MSG_LLM_RESPONSE,
                {
                    "success": True,
                    "results": results,
                },
            )
        except Exception as e:
            logger.error("llm_batch callback failed: %s", e)
            write_message(
                self._stdin,
                MSG_LLM_RESPONSE,
                {
                    "success": False,
                    "error": str(e),
                },
            )


# Worker-side IPC functions (embedded in worker scripts)
# These are string templates to be included in the worker script
WORKER_IPC_FUNCTIONS = '''
import struct
import json
import sys

_IPC_MAX_MESSAGE_SIZE = 10 * 1024 * 1024
_IPC_HEADER_SIZE = 4

def _ipc_encode_message(msg_type, payload):
    """Encode a message with length prefix."""
    message = {"type": msg_type, "payload": payload}
    json_bytes = json.dumps(message).encode("utf-8")
    if len(json_bytes) > _IPC_MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {len(json_bytes)}")
    length_prefix = struct.pack(">I", len(json_bytes))
    return length_prefix + json_bytes

def _ipc_read_message(stream):
    """Read a length-prefixed message from stream."""
    header = stream.read(_IPC_HEADER_SIZE)
    if not header or len(header) < _IPC_HEADER_SIZE:
        return None
    length = struct.unpack(">I", header)[0]
    if length > _IPC_MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {length}")
    data = stream.read(length)
    if len(data) < length:
        raise ValueError(f"Incomplete message")
    return json.loads(data.decode("utf-8"))

def _ipc_write_message(stream, msg_type, payload):
    """Write a length-prefixed message to stream."""
    encoded = _ipc_encode_message(msg_type, payload)
    stream.write(encoded)
    stream.flush()

def _ipc_send_result(result):
    """Send final result to host."""
    _ipc_write_message(sys.stdout.buffer, "RESULT", result)

def _ipc_send_error(error):
    """Send error to host."""
    _ipc_write_message(sys.stdout.buffer, "ERROR", {"error": str(error)})

def _make_ipc_llm_query():
    """Create llm_query function that uses IPC."""
    def llm_query(prompt):
        _ipc_write_message(sys.stdout.buffer, "LLM_QUERY", {"prompt": str(prompt)})
        response = _ipc_read_message(sys.stdin.buffer)
        if response is None:
            raise RuntimeError("IPC connection closed")
        payload = response.get("payload", {})
        if not payload.get("success", False):
            error = payload.get("error", "Unknown error")
            raise RuntimeError(f"llm_query failed: {error}")
        return payload.get("result", "")
    return llm_query

def _make_ipc_llm_batch():
    """Create llm_batch function that uses IPC."""
    def llm_batch(prompts):
        prompts_list = [str(p) for p in prompts]
        _ipc_write_message(sys.stdout.buffer, "LLM_BATCH", {"prompts": prompts_list})
        response = _ipc_read_message(sys.stdin.buffer)
        if response is None:
            raise RuntimeError("IPC connection closed")
        payload = response.get("payload", {})
        if not payload.get("success", False):
            error = payload.get("error", "Unknown error")
            raise RuntimeError(f"llm_batch failed: {error}")
        return payload.get("results", [])
    return llm_batch
'''
