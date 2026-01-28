"""
Worker server for distributed RLM execution.

Run on each worker machine:
    ENZU_WORKER_SECRET=your-secret python -m enzu.runtime.worker --port 8080

Or with uvicorn for production:
    ENZU_WORKER_SECRET=your-secret uvicorn enzu.runtime.worker:app --host 0.0.0.0 --port 8080 --workers 4

SECURITY:
    - Set ENZU_WORKER_SECRET to require authentication
    - Workers use their own API credentials (OPENAI_API_KEY, etc.)
    - Use TLS in production (reverse proxy or --ssl-keyfile/--ssl-certfile)
    - Bind to internal network interface, not 0.0.0.0, when possible

The worker accepts tasks via HTTP POST /run and executes them locally.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from enzu.models import TaskSpec
from enzu.runtime.local import LocalRuntime
from enzu.runtime.protocol import ProviderSpec, RuntimeOptions

logger = logging.getLogger(__name__)


@dataclass
class WorkerState:
    """Shared state for the worker."""

    runtime: LocalRuntime = field(default_factory=LocalRuntime)
    active: int = 0
    completed: int = 0
    failed: int = 0
    max_concurrent: int = 4
    secret: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire(self) -> bool:
        with self._lock:
            if self.active >= self.max_concurrent:
                return False
            self.active += 1
            return True

    def release(self, success: bool) -> None:
        with self._lock:
            self.active -= 1
            if success:
                self.completed += 1
            else:
                self.failed += 1

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active": self.active,
                "completed": self.completed,
                "failed": self.failed,
                "max_concurrent": self.max_concurrent,
                "available": self.max_concurrent - self.active,
            }

    def check_auth(self, auth_header: str | None) -> bool:
        """Verify authentication. Returns True if auth is valid or not required."""
        if not self.secret:
            return True  # No secret configured, allow all
        if not auth_header:
            return False
        # Expected format: "Bearer <secret>"
        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0] != "Bearer":
            return False
        return parts[1] == self.secret


# Global worker state
_worker_state: WorkerState | None = None


def get_worker_state() -> WorkerState:
    global _worker_state
    if _worker_state is None:
        max_concurrent = int(os.getenv("ENZU_WORKER_MAX_CONCURRENT", "4"))
        secret = os.getenv("ENZU_WORKER_SECRET")
        _worker_state = WorkerState(max_concurrent=max_concurrent, secret=secret)
        if not secret:
            logger.warning(
                "ENZU_WORKER_SECRET not set. Worker accepts unauthenticated requests. "
                "Set ENZU_WORKER_SECRET for production use."
            )
    return _worker_state


class WorkerHandler(BaseHTTPRequestHandler):
    """HTTP handler for worker requests."""

    def do_POST(self) -> None:
        if self.path == "/run":
            self._handle_run()
        else:
            self._send_error(404, "Not found")

    def do_GET(self) -> None:
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/stats":
            self._handle_stats()
        else:
            self._send_error(404, "Not found")

    def _check_auth(self) -> bool:
        """Check authentication. Returns False and sends 401 if invalid."""
        state = get_worker_state()
        auth_header = self.headers.get("Authorization")
        if not state.check_auth(auth_header):
            self._send_error(401, "Unauthorized")
            return False
        return True

    def _handle_run(self) -> None:
        import json

        # Check authentication first
        if not self._check_auth():
            return

        state = get_worker_state()

        # Check capacity
        if not state.acquire():
            self._send_error(503, "Worker at capacity")
            return

        success = False
        try:
            # Parse request
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body)

            # Deserialize
            spec = TaskSpec.model_validate(payload["spec"])
            provider = ProviderSpec(
                name=payload["provider"].get("name"),
                api_key=payload["provider"].get("api_key"),
                referer=payload["provider"].get("referer"),
                app_name=payload["provider"].get("app_name"),
                organization=payload["provider"].get("organization"),
                project=payload["provider"].get("project"),
                use_pool=payload["provider"].get("use_pool", False),
            )
            data = payload["data"]
            options = RuntimeOptions(
                max_steps=payload["options"].get("max_steps", 8),
                verify_on_final=payload["options"].get("verify_on_final", True),
                isolation=payload["options"].get("isolation"),
            )

            # Execute
            result = state.runtime.run(
                spec=spec, provider=provider, data=data, options=options
            )
            success = True

            # Send response
            response_body = json.dumps(result.model_dump()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        except Exception as e:
            logger.exception("Error executing task")
            self._send_error(500, str(e))
        finally:
            state.release(success)

    def _handle_health(self) -> None:
        import json

        # Health check allows unauthenticated access for load balancers
        state = get_worker_state()
        healthy = state.active < state.max_concurrent
        status = 200 if healthy else 503
        body = json.dumps({"healthy": healthy}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_stats(self) -> None:
        import json

        # Stats require authentication (may contain sensitive info)
        if not self._check_auth():
            return

        state = get_worker_state()
        body = json.dumps(state.stats()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str) -> None:
        import json

        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), format % args)


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the worker server."""
    server = ThreadingHTTPServer((host, port), WorkerHandler)
    logger.info("Worker server starting on %s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


# ASGI app for uvicorn (optional, for production)
try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route
    import asyncio
    import json as json_module

    def _check_auth_async(request: Request) -> bool:
        """Check authentication for async handlers."""
        state = get_worker_state()
        auth_header = request.headers.get("Authorization")
        return state.check_auth(auth_header)

    def _parse_request_payload(payload: dict) -> tuple:
        """Parse and validate request payload into spec, provider, data, options."""
        spec = TaskSpec.model_validate(payload["spec"])
        provider = ProviderSpec(
            name=payload["provider"].get("name"),
            api_key=payload["provider"].get("api_key"),
            referer=payload["provider"].get("referer"),
            app_name=payload["provider"].get("app_name"),
            organization=payload["provider"].get("organization"),
            project=payload["provider"].get("project"),
            use_pool=payload["provider"].get("use_pool", False),
        )
        data = payload["data"]
        return spec, provider, data, payload["options"]

    async def handle_run(request: Request) -> JSONResponse:
        # Check authentication
        if not _check_auth_async(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        state = get_worker_state()

        if not state.acquire():
            return JSONResponse({"error": "Worker at capacity"}, status_code=503)

        success = False
        try:
            payload = await request.json()
            spec, provider, data, opts = _parse_request_payload(payload)
            options = RuntimeOptions(
                max_steps=opts.get("max_steps", 8),
                verify_on_final=opts.get("verify_on_final", True),
                isolation=opts.get("isolation"),
            )

            # Run in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: state.runtime.run(
                    spec=spec, provider=provider, data=data, options=options
                ),
            )
            success = True
            return JSONResponse(result.model_dump())

        except Exception as e:
            logger.exception("Error executing task")
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            state.release(success)

    async def handle_run_stream(request: Request) -> JSONResponse | StreamingResponse:
        """
        SSE streaming endpoint for real-time progress updates.

        Emits events:
            - progress: General progress messages
            - step: RLM step execution updates
            - subcall: Recursive subcall start/end
            - complete: Final RLMExecutionReport
            - error: Execution failure
        """
        # Check authentication
        if not _check_auth_async(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        state = get_worker_state()

        if not state.acquire():
            return JSONResponse({"error": "Worker at capacity"}, status_code=503)

        try:
            payload = await request.json()
            spec, provider, data, opts = _parse_request_payload(payload)
        except Exception as e:
            state.release(success=False)
            return JSONResponse({"error": str(e)}, status_code=400)

        async def event_generator():
            """Generate SSE events from RLM execution."""
            queue: asyncio.Queue = asyncio.Queue()
            success = False
            current_step = 0

            def on_progress(message: str) -> None:
                """Callback invoked by RLMEngine for progress updates."""
                nonlocal current_step
                try:
                    # Parse message to determine event type
                    if message.startswith("step:") or "Step " in message:
                        # Extract step number if present
                        import re
                        match = re.search(r"[Ss]tep\s*:?\s*(\d+)", message)
                        if match:
                            current_step = int(match.group(1))
                        queue.put_nowait({
                            "event": "step",
                            "data": {"step": current_step, "status": message}
                        })
                    elif "subcall" in message.lower() or "llm_query" in message.lower():
                        # Subcall events
                        depth = 1
                        import re
                        depth_match = re.search(r"depth[=:]\s*(\d+)", message)
                        if depth_match:
                            depth = int(depth_match.group(1))
                        queue.put_nowait({
                            "event": "subcall",
                            "data": {"depth": depth, "phase": message}
                        })
                    else:
                        # General progress
                        queue.put_nowait({
                            "event": "progress",
                            "data": {"message": message}
                        })
                except Exception:
                    pass  # Don't let callback errors break execution

            async def run_task() -> None:
                """Execute RLM in thread pool with progress callback."""
                nonlocal success
                try:
                    options = RuntimeOptions(
                        max_steps=opts.get("max_steps", 8),
                        verify_on_final=opts.get("verify_on_final", True),
                        isolation=opts.get("isolation"),
                        on_progress=on_progress,
                    )

                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: state.runtime.run(
                            spec=spec, provider=provider, data=data, options=options
                        ),
                    )
                    success = True
                    await queue.put({
                        "event": "complete",
                        "data": result.model_dump()
                    })
                except Exception as e:
                    logger.exception("Error in streaming execution")
                    await queue.put({
                        "event": "error",
                        "data": {"error": str(e)}
                    })
                finally:
                    state.release(success)

            # Start task execution
            task = asyncio.create_task(run_task())

            try:
                while True:
                    try:
                        # Wait for events with timeout to allow cancellation
                        event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    except asyncio.TimeoutError:
                        # Send keepalive comment
                        yield ": keepalive\n\n"
                        continue

                    # Format SSE event
                    event_type = event["event"]
                    event_data = json_module.dumps(event["data"])
                    yield f"event: {event_type}\ndata: {event_data}\n\n"

                    # Exit after complete or error
                    if event_type in ("complete", "error"):
                        break

            except asyncio.CancelledError:
                task.cancel()
                raise

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )

    async def handle_health(request: Request) -> JSONResponse:
        state = get_worker_state()
        healthy = state.active < state.max_concurrent
        return JSONResponse(
            {"healthy": healthy}, status_code=200 if healthy else 503
        )

    async def handle_stats(request: Request) -> JSONResponse:
        # Stats require authentication
        if not _check_auth_async(request):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return JSONResponse(get_worker_state().stats())

    app = Starlette(
        routes=[
            Route("/run", handle_run, methods=["POST"]),
            Route("/run/stream", handle_run_stream, methods=["POST"]),
            Route("/health", handle_health, methods=["GET"]),
            Route("/stats", handle_stats, methods=["GET"]),
        ]
    )

except ImportError:
    app = None  # Starlette not installed


def main() -> None:
    parser = argparse.ArgumentParser(description="Enzu RLM worker server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Max concurrent tasks",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    global _worker_state
    _worker_state = WorkerState(max_concurrent=args.max_concurrent)

    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
