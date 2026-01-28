#!/usr/bin/env python3
"""
Test budget enforcement for the HTTP API server.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from httpx import AsyncClient, ASGITransport

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

MODEL = "z-ai/glm-4.7"
PROVIDER = "openrouter"
TIMEOUT = 120.0


@pytest.mark.integration
@pytest.mark.anyio
async def test_budget_tracking():
    """Test that budget is tracked across requests."""
    os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

    from enzu.server import create_app
    from enzu.server.services.session_manager import reset_session_store
    from enzu.server.config import reset_settings

    reset_session_store()
    reset_settings()
    app = create_app()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", timeout=TIMEOUT) as client:
        # Create session with a cost limit
        print("\n1. Creating session with max_cost_usd=0.10...")
        resp = await client.post(
            "/v1/sessions",
            json={
                "model": MODEL,
                "provider": PROVIDER,
                "max_cost_usd": 0.10,  # $0.10 limit
            },
        )
        assert resp.status_code == 200, f"Failed: {resp.text}"
        session_data = resp.json()
        session_id = session_data["session_id"]
        print(f"   Session created: {session_id}")
        print(f"   max_cost_usd: {session_data['max_cost_usd']}")

        # Run a task
        print("\n2. Running first task...")
        resp = await client.post(
            f"/v1/sessions/{session_id}/run",
            json={"task": "Say hello in one word."},
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, f"Failed: {resp.text}"
        result = resp.json()
        print(f"   Answer: {result['answer'][:50]}...")
        print(f"   Usage: {result['usage']}")
        print(f"   Session total cost: ${result['session_total_cost_usd']:.6f}")
        print(f"   Session total tokens: {result['session_total_tokens']}")

        # Get session state to verify tracking
        print("\n3. Getting session state...")
        resp = await client.get(f"/v1/sessions/{session_id}")
        assert resp.status_code == 200
        state = resp.json()
        print(f"   total_cost_usd: ${state['total_cost_usd']:.6f}")
        print(f"   total_tokens: {state['total_tokens']}")
        remaining = state.get('remaining_cost_usd')
        if remaining is not None:
            print(f"   remaining_cost_usd: ${remaining:.6f}")
        print(f"   exchange_count: {state['exchange_count']}")

        # Run more tasks to consume budget
        print("\n4. Running more tasks to consume budget...")
        budget_exceeded = False
        for i in range(5):
            resp = await client.post(
                f"/v1/sessions/{session_id}/run",
                json={"task": f"Count from 1 to {(i+1)*10}. Just the numbers."},
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                result = resp.json()
                print(f"   Task {i+1}: cost=${result['session_total_cost_usd']:.6f}, tokens={result['session_total_tokens']}")
            elif resp.status_code == 402:
                print(f"   Task {i+1}: BUDGET EXCEEDED (HTTP 402)")
                error = resp.json()
                print(f"   Error code: {error['error']['code']}")
                budget_exceeded = True
                break
            else:
                print(f"   Task {i+1}: Unexpected error {resp.status_code}")
                print(f"   {resp.text}")

        # Final state
        print("\n5. Final session state...")
        resp = await client.get(f"/v1/sessions/{session_id}")
        if resp.status_code == 200:
            state = resp.json()
            print(f"   total_cost_usd: ${state['total_cost_usd']:.6f}")
            print(f"   total_tokens: {state['total_tokens']}")
            remaining = state.get('remaining_cost_usd')
            if remaining is not None:
                print(f"   remaining_cost_usd: ${remaining:.6f}")
            print(f"   exchange_count: {state['exchange_count']}")

    print("\n✓ Budget tracking test complete!")
    return budget_exceeded


if __name__ == "__main__":
    print("=" * 60)
    print("  SERVER BUDGET ENFORCEMENT TEST")
    print("=" * 60)
    budget_exceeded = asyncio.run(test_budget_tracking())
    print("=" * 60)
    if budget_exceeded:
        print("  ✓ Budget enforcement verified - 402 returned when exceeded")
    else:
        print("  Note: Budget not exceeded in this run (depends on model cost)")
    print("=" * 60)
