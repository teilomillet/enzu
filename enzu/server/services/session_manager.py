"""
Session storage and management for HTTP context.

Provides isolated session storage with TTL support.
MVP implementation uses in-memory storage; Redis can be added later.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from enzu.session import Session
from enzu.server.config import get_settings
from enzu.server.exceptions import SessionNotFoundError, SessionLockError


@dataclass
class StoredSession:
    """Session with metadata for storage."""

    session: Session
    session_id: str
    created_at: float
    last_accessed: float
    ttl_seconds: int
    locked: bool = False
    lock_owner: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return time.time() - self.last_accessed > self.ttl_seconds

    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = time.time()


class InMemorySessionStore:
    """
    In-memory session storage.

    Thread-safe for use with asyncio. Sessions are automatically
    cleaned up when expired on access or via periodic cleanup.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, StoredSession] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Periodically remove expired sessions."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            await self._cleanup_expired()

    async def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        async with self._lock:
            expired = [
                sid for sid, stored in self._sessions.items()
                if stored.is_expired
            ]
            for sid in expired:
                del self._sessions[sid]

    def generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"sess-{uuid.uuid4().hex[:16]}"

    async def create(
        self,
        session: Session,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Create and store a new session.

        Args:
            session: The Session object to store.
            ttl_seconds: TTL for this session. Defaults to config value.

        Returns:
            The generated session ID.
        """
        settings = get_settings()
        ttl = ttl_seconds or settings.session_ttl_seconds

        session_id = self.generate_session_id()
        now = time.time()

        stored = StoredSession(
            session=session,
            session_id=session_id,
            created_at=now,
            last_accessed=now,
            ttl_seconds=ttl,
        )

        async with self._lock:
            self._sessions[session_id] = stored

        return session_id

    async def get(self, session_id: str) -> Session:
        """
        Get a session by ID.

        Args:
            session_id: The session ID.

        Returns:
            The Session object.

        Raises:
            SessionNotFoundError: If session doesn't exist or is expired.
        """
        async with self._lock:
            stored = self._sessions.get(session_id)
            if stored is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            if stored.is_expired:
                del self._sessions[session_id]
                raise SessionNotFoundError(f"Session expired: {session_id}")

            stored.touch()
            return stored.session

    async def get_stored(self, session_id: str) -> StoredSession:
        """
        Get stored session with metadata.

        Args:
            session_id: The session ID.

        Returns:
            The StoredSession object.

        Raises:
            SessionNotFoundError: If session doesn't exist or is expired.
        """
        async with self._lock:
            stored = self._sessions.get(session_id)
            if stored is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            if stored.is_expired:
                del self._sessions[session_id]
                raise SessionNotFoundError(f"Session expired: {session_id}")

            stored.touch()
            return stored

    async def update(self, session_id: str, session: Session) -> None:
        """
        Update a session.

        Args:
            session_id: The session ID.
            session: The updated Session object.

        Raises:
            SessionNotFoundError: If session doesn't exist.
        """
        async with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            stored = self._sessions[session_id]
            stored.session = session
            stored.touch()

    async def delete(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: The session ID.

        Returns:
            True if deleted, False if not found.
        """
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    async def acquire_lock(self, session_id: str, owner: str) -> bool:
        """
        Acquire exclusive lock on a session for concurrent request safety.

        Args:
            session_id: The session ID.
            owner: Identifier for the lock owner (e.g., request_id).

        Returns:
            True if lock acquired, False if already locked.

        Raises:
            SessionNotFoundError: If session doesn't exist.
        """
        async with self._lock:
            stored = self._sessions.get(session_id)
            if stored is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            if stored.locked:
                return False

            stored.locked = True
            stored.lock_owner = owner
            return True

    async def release_lock(self, session_id: str, owner: str) -> None:
        """
        Release lock on a session.

        Args:
            session_id: The session ID.
            owner: Identifier for the lock owner.

        Raises:
            SessionNotFoundError: If session doesn't exist.
            SessionLockError: If owner doesn't match.
        """
        async with self._lock:
            stored = self._sessions.get(session_id)
            if stored is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")

            if stored.lock_owner != owner:
                raise SessionLockError(
                    f"Lock owner mismatch: expected {stored.lock_owner}, got {owner}"
                )

            stored.locked = False
            stored.lock_owner = None

    async def stats(self) -> Dict[str, int]:
        """Get session store statistics."""
        async with self._lock:
            active = sum(1 for s in self._sessions.values() if not s.is_expired)
            expired = len(self._sessions) - active
            return {
                "total_sessions": len(self._sessions),
                "active_sessions": active,
                "expired_sessions": expired,
            }


# Global session store singleton
_session_store: Optional[InMemorySessionStore] = None


def get_session_store() -> InMemorySessionStore:
    """Get the global session store."""
    global _session_store
    if _session_store is None:
        _session_store = InMemorySessionStore()
    return _session_store


def reset_session_store() -> None:
    """Reset global session store. For testing only."""
    global _session_store
    _session_store = None
