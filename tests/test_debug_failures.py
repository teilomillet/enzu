#!/usr/bin/env python3
"""
Debug test to understand why some responses don't contain the expected marker.

Runs a smaller test with detailed logging to investigate failures.
"""

from __future__ import annotations

import asyncio
import os
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

# Configuration - smaller test for debugging
NUM_USERS = 10
NUM_TURNS = 2
MODEL = "z-ai/glm-4.7"
PROVIDER = "openrouter"
TIMEOUT = 180.0


@dataclass
class UserSession:
    user_id: int
    marker: str = field(default_factory=lambda: "")
    session_id: Optional[str] = None
    responses: List[dict] = field(default_factory=list)  # Full response data
    errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.marker = f"USER{self.user_id:03d}-{uuid.uuid4().hex[:8]}"


async def run_user_session(
    user: UserSession,
    client: AsyncClient,
) -> UserSession:
    """Run a complete session with detailed logging."""
    try:
        # Create session
        resp = await client.post(
            "/v1/sessions",
            json={"model": MODEL, "provider": PROVIDER},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            user.errors.append(
                f"Session creation failed: {resp.status_code} - {resp.text}"
            )
            return user

        session_data = resp.json()
        user.session_id = session_data["session_id"]

        # Run multiple turns
        for turn in range(NUM_TURNS):
            task = f"Please respond with exactly this code: {user.marker}. Just output the code, nothing else."

            resp = await client.post(
                f"/v1/sessions/{user.session_id}/run",
                json={"task": task},
                headers={"X-Request-ID": f"req-{user.user_id}-turn{turn}"},
                timeout=TIMEOUT,
            )

            response_data = {
                "turn": turn,
                "status_code": resp.status_code,
                "task": task,
                "marker": user.marker,
            }

            if resp.status_code == 200:
                data = resp.json()
                response_data["answer"] = data.get("answer", "")
                response_data["request_id"] = data.get("request_id", "")
                response_data["usage"] = data.get("usage", {})
                response_data["marker_found"] = user.marker in data.get("answer", "")
            else:
                response_data["error"] = resp.text[:500]
                user.errors.append(f"Turn {turn}: {resp.status_code}")

            user.responses.append(response_data)

    except Exception as e:
        user.errors.append(f"Exception: {type(e).__name__}: {str(e)}")

    return user


async def main():
    print("=" * 80)
    print("  DEBUG TEST - Investigating Response Failures")
    print("=" * 80)

    # Suppress logfire warnings
    os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

    from enzu.server import create_app
    from enzu.server.services.session_manager import reset_session_store
    from enzu.server.config import reset_settings

    reset_session_store()
    reset_settings()
    app = create_app()

    users = [UserSession(user_id=i) for i in range(NUM_USERS)]

    print(f"\nRunning {NUM_USERS} users with {NUM_TURNS} turns each...")
    print(f"Model: {MODEL}, Provider: {PROVIDER}\n")

    start_time = time.time()

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", timeout=TIMEOUT
    ) as client:
        tasks = [run_user_session(user, client) for user in users]
        users = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    # Analyze results
    print("\n" + "=" * 80)
    print("  DETAILED ANALYSIS")
    print("=" * 80)

    total_responses = 0
    marker_found = 0
    marker_missing = 0
    failures = []

    for user in users:
        for resp in user.responses:
            total_responses += 1
            if resp.get("marker_found"):
                marker_found += 1
            elif resp.get("status_code") == 200:
                marker_missing += 1
                failures.append(
                    {
                        "user_id": user.user_id,
                        "marker": user.marker,
                        "turn": resp["turn"],
                        "answer": resp.get("answer", ""),
                        "task": resp["task"],
                    }
                )

    print(f"\nTotal responses: {total_responses}")
    print(f"Marker found: {marker_found}")
    print(f"Marker missing: {marker_missing}")
    print(f"Time: {elapsed:.1f}s")

    if failures:
        print("\n" + "-" * 80)
        print("  FAILURE DETAILS")
        print("-" * 80)

        for i, f in enumerate(failures):
            print(f"\n{'=' * 60}")
            print(f"FAILURE #{i + 1}: User {f['user_id']}, Turn {f['turn']}")
            print(f"{'=' * 60}")
            print(f"Expected marker: {f['marker']}")
            print(f"Task sent: {f['task']}")
            print(f"\nFULL RESPONSE ({len(f['answer'])} chars):")
            print("-" * 40)
            print(f["answer"])
            print("-" * 40)

            # Analyze what went wrong
            answer = f["answer"]
            if len(answer) == 0:
                print("ISSUE: Empty response")
            elif f["marker"] in answer:
                print("ISSUE: Marker IS in response (detection bug?)")
            else:
                # Check if it's close
                if "USER" in answer:
                    print("ISSUE: Contains 'USER' but not the exact marker")
                    # Find what USER string is there
                    import re

                    user_matches = re.findall(r"USER\d+-[a-f0-9]+", answer)
                    if user_matches:
                        print(f"  Found: {user_matches}")
                        for m in user_matches:
                            if m != f["marker"]:
                                print(
                                    f"  CROSS-CONTAMINATION? Found {m} instead of {f['marker']}"
                                )
                elif "FINAL" in answer:
                    print("ISSUE: Response contains FINAL but marker not extracted")
                elif len(answer) < 50:
                    print("ISSUE: Short response - might be truncated or error")
                else:
                    print("ISSUE: LLM gave completely different response")
                    # Show first 200 chars
                    print(f"  Preview: {answer[:200]}...")

    # Check for cross-contamination
    print("\n" + "-" * 80)
    print("  CROSS-CONTAMINATION CHECK")
    print("-" * 80)

    all_markers = {u.marker for u in users}
    contamination_found = False

    for user in users:
        for resp in user.responses:
            answer = resp.get("answer", "")
            for other_marker in all_markers:
                if other_marker != user.marker and other_marker in answer:
                    print(
                        f"CONTAMINATION: User {user.user_id}'s response contains {other_marker}"
                    )
                    contamination_found = True

    if not contamination_found:
        print("No cross-contamination detected.")

    # Final verdict
    print("\n" + "=" * 80)
    if marker_missing == 0 and not contamination_found:
        print("  PASS - All responses contain correct markers")
    elif not contamination_found:
        print(f"  PARTIAL - {marker_missing} responses missing markers (LLM behavior)")
        print("  No cross-contamination - isolation is working")
    else:
        print("  FAIL - Cross-contamination detected!")
    print("=" * 80)

    return 0 if not contamination_found else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
