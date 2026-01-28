"""
Enzu HTTP API Server - Convenience entry point.

Usage:
    uvicorn enzu.server:app --reload

Or with default settings:
    python -m enzu.server

This module re-exports the app from enzu.server package for
convenient uvicorn access.
"""
from enzu.server import app, create_app

__all__ = ["app", "create_app"]


if __name__ == "__main__":
    import uvicorn
    from enzu.server.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "enzu.server:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
