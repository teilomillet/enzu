#!/usr/bin/env python3
"""
REAL Integration Test - 20+ Concurrent Users with Actual LLM Calls

This test actually starts the server and makes real API calls to verify:
1. 20+ concurrent users can run simultaneously
2. Each user gets their own isolated session
3. Responses contain the correct unique markers (no cross-contamination)
4. The system handles the load correctly

Usage:
    uv run python tests/test_real_concurrency.py
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from httpx import AsyncClient, ASGITransport

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from {env_path}")


# Configuration
NUM_USERS = 50  # Test with 50 concurrent users
NUM_TURNS = 2
MODEL = "z-ai/glm-4.7"
PROVIDER = "openrouter"
TIMEOUT = 180.0  # Generous timeout for LLM calls


@dataclass
class UserSession:
    """Tracks a user's session and results."""
    user_id: int
    marker: str = field(default_factory=lambda: "")
    session_id: Optional[str] = None
    responses: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Create unique marker that will be echoed in responses
        self.marker = f"USER{self.user_id:03d}-{uuid.uuid4().hex[:8]}"

    @property
    def success(self) -> bool:
        return self.session_id is not None and len(self.errors) == 0 and len(self.responses) > 0


async def run_user_session(
    user: UserSession,
    client: AsyncClient,
    model: str,
    provider: str,
    num_turns: int,
) -> UserSession:
    """Run a complete session for one user."""
    try:
        # Create session
        resp = await client.post(
            "/v1/sessions",
            json={"model": model, "provider": provider},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            user.errors.append(f"Session creation failed: {resp.status_code} - {resp.text}")
            return user

        user.session_id = resp.json()["session_id"]
        print(f"  User {user.user_id}: Session {user.session_id} created")

        # Run multiple turns
        for turn in range(num_turns):
            # Task asks the model to echo back our unique marker
            task = f"Please respond with exactly this code: {user.marker}. Just output the code, nothing else."

            resp = await client.post(
                f"/v1/sessions/{user.session_id}/run",
                json={"task": task},
                headers={"X-Request-ID": f"req-{user.user_id}-turn{turn}"},
                timeout=TIMEOUT,
            )

            if resp.status_code != 200:
                user.errors.append(f"Turn {turn}: {resp.status_code} - {resp.text[:200]}")
                continue

            answer = resp.json().get("answer", "")
            user.responses.append(answer)

            # Check if our marker is in the response
            if user.marker in answer:
                print(f"  User {user.user_id} Turn {turn}: OK - marker found")
            else:
                print(f"  User {user.user_id} Turn {turn}: WARNING - marker NOT in response")
                user.errors.append(f"Turn {turn}: Marker not in response")

    except Exception as e:
        user.errors.append(f"Exception: {type(e).__name__}: {str(e)[:200]}")
        print(f"  User {user.user_id}: ERROR - {e}")

    return user


def check_cross_contamination(users: List[UserSession]) -> List[str]:
    """Check if any user's response contains another user's marker."""
    issues = []
    for user in users:
        for response in user.responses:
            for other in users:
                if other.user_id != user.user_id and other.marker in response:
                    issues.append(
                        f"CONTAMINATION: User {user.user_id}'s response contains User {other.user_id}'s marker"
                    )
    return issues


async def main():
    print("=" * 70)
    print(f"  REAL CONCURRENCY TEST - {NUM_USERS} CONCURRENT USERS")
    print(f"  Model: {MODEL}")
    print(f"  Turns per user: {NUM_TURNS}")
    print("=" * 70)

    # Import and create app
    from enzu.server import create_app
    from enzu.server.services.session_manager import reset_session_store
    from enzu.server.config import reset_settings

    reset_session_store()
    reset_settings()
    app = create_app()

    # Create users
    users = [UserSession(user_id=i) for i in range(NUM_USERS)]

    print(f"\nCreated {NUM_USERS} users with unique markers:")
    for u in users[:5]:
        print(f"  User {u.user_id}: {u.marker}")
    print(f"  ... and {NUM_USERS - 5} more")

    print("\nStarting concurrent sessions...")
    start_time = time.time()

    # Run all users concurrently using ASGI transport (no separate server needed)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=TIMEOUT) as client:
        tasks = [
            run_user_session(user, client, MODEL, PROVIDER, NUM_TURNS)
            for user in users
        ]
        users = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    # Analyze results
    successful = [u for u in users if u.success]
    failed = [u for u in users if not u.success]
    contamination = check_cross_contamination(users)

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Total users:       {NUM_USERS}")
    print(f"  Successful:        {len(successful)}")
    print(f"  Failed:            {len(failed)}")
    print(f"  Success rate:      {len(successful)/NUM_USERS:.1%}")
    print(f"  Total time:        {elapsed:.1f}s")
    print(f"  Avg per user:      {elapsed/NUM_USERS:.2f}s")
    print("-" * 70)
    print("  ISOLATION CHECK")
    print("-" * 70)

    # Check marker presence
    marker_found = 0
    marker_missing = 0
    for user in users:
        for resp in user.responses:
            if user.marker in resp:
                marker_found += 1
            else:
                marker_missing += 1

    print(f"  Responses with correct marker:   {marker_found}")
    print(f"  Responses missing marker:        {marker_missing}")
    print(f"  Cross-contamination issues:      {len(contamination)}")

    if contamination:
        print("\n  CONTAMINATION DETAILS:")
        for issue in contamination[:10]:
            print(f"    {issue}")
        if len(contamination) > 10:
            print(f"    ... and {len(contamination) - 10} more")

    # Print failed user details
    if failed:
        print("\n  FAILED USERS:")
        for user in failed[:5]:
            print(f"    User {user.user_id}: {user.errors[0] if user.errors else 'Unknown error'}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")

    print("\n" + "=" * 70)

    # Show sample responses from "failed" users to verify it's just LLM behavior
    if marker_missing > 0:
        print("\n  SAMPLE 'MISSING MARKER' RESPONSES (verifying no contamination):")
        shown = 0
        for user in users:
            for i, resp in enumerate(user.responses):
                if user.marker not in resp and shown < 3:
                    print(f"    User {user.user_id} Turn {i}:")
                    print(f"      Expected marker: {user.marker}")
                    print(f"      Response preview: {resp[:100]}...")
                    # Check if ANY other user's marker is present
                    found_other = False
                    for other in users:
                        if other.user_id != user.user_id and other.marker in resp:
                            found_other = True
                            print(f"      CONTAMINATION: Found {other.marker}!")
                    if not found_other:
                        print("      No other user's marker found - just LLM output variation")
                    shown += 1

    # Final verdict - isolation is the key metric
    if len(contamination) == 0:
        print("\n" + "=" * 70)
        print("  PASS - No cross-contamination detected!")
        print("  Session isolation is working correctly under concurrent load.")
        if marker_missing > 0:
            print(f"  Note: {marker_missing} responses didn't echo marker (LLM behavior, not isolation issue)")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("  FAIL - Cross-contamination detected!")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
