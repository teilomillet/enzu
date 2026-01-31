"""
Enzu HTTP API Server.

Usage:
    # Start server
    uvicorn enzu.server:app --reload

    # Or programmatically
    from enzu.server import app, create_app

    # Custom configuration
    app = create_app()
"""

from enzu.server.app import app, create_app

__all__ = ["app", "create_app"]
