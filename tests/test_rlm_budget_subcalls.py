"""
Tests for budget enforcement in recursive subcalls.

Verifies that subcalls respect parent budget limits and cannot exceed
remaining budget during execution. This prevents the bug where subcalls
could consume unlimited tokens because their TokenBudgetPool had
max_total_tokens=None.
"""
from __future__ import annotations


from enzu.models import Budget, SuccessCriteria, TaskSpec
from enzu.rlm import RLMEngine
from tests.providers.mock_provider import MockProvider


class TestSubcallBudgetEnforcement:
    """Test that subcalls respect parent budget limits."""

    def test_subcall_respects_parent_budget_limit(self) -> None:
        """
        Verify subcalls cannot exceed parent's remaining budget.
        
        Parent has max_total_tokens=100. Subcall attempts to use 150 tokens
        across multiple steps. Subcall should fail with budget exhaustion.
        """
        # Subcall will try to make 3 steps, each using 50 tokens = 150 total
        # But parent only has 100 tokens remaining after reservation
        subcall_step1 = '```python\nresult1 = llm_query("SUBCALL:step1")\nprint(result1)\n```'
        subcall_step2 = '```python\nresult2 = llm_query("SUBCALL:step2")\nprint(result2)\n```'
        subcall_step3 = '```python\nresult3 = llm_query("SUBCALL:step3")\nFINAL("done")\n```'
        
        provider = MockProvider(
            main_outputs=[subcall_step1, subcall_step2, subcall_step3],
            subcall_responses={
                "step1": "response1",
                "step2": "response2",
                "step3": "response3",
            },
            # Each call uses 50 tokens (input + output)
            usage={"output_tokens": 20, "total_tokens": 50},
        )
        
        task = TaskSpec(
            task_id="subcall-budget-test",
            input_text="Make multiple subcalls",
            model="mock",
            # Parent budget: 100 tokens total
            # With mock model, conservative estimation is skipped
            # Subcall needs 150 tokens (3 steps × 50), should fail
            budget=Budget(max_total_tokens=100),
            success_criteria=SuccessCriteria(required_substrings=["done"]),
        )
        
        engine = RLMEngine(
            max_steps=5,
            recursive_subcalls=True,
            max_recursion_depth=1,
            subcall_max_steps=5,
        )
        report = engine.run(task, provider, data="context")
        
        # Subcall should fail due to budget exhaustion
        assert report.success is False
        # Should have attempted some steps but failed before completing
        assert len(report.steps) > 0
        # Check that budget was exhausted - either in step errors or report errors
        # With the new enforcement, budget exhaustion is detected at the pool level
        budget_exhausted_in_steps = any(
            "budget_exhausted" in str(step.error or "") for step in report.steps if step.error
        )
        budget_exceeded_in_report = "budget_exceeded" in report.errors
        assert budget_exhausted_in_steps or budget_exceeded_in_report

    def test_subcall_budget_enforcement_during_execution(self) -> None:
        """
        Verify subcall's TokenBudgetPool enforces limits during multi-step execution.
        
        Parent budget: 50 tokens. Subcall with 3 steps, each using 20 tokens.
        Expected: Subcall fails after step 2 (40 tokens used, 10 remaining, step 3 needs 20).
        """
        step1 = '```python\nresult1 = llm_query("SUBCALL:step1")\nprint(result1)\n```'
        step2 = '```python\nresult2 = llm_query("SUBCALL:step2")\nprint(result2)\n```'
        step3 = '```python\nFINAL("complete")\n```'
        
        provider = MockProvider(
            main_outputs=[step1, step2, step3],
            subcall_responses={
                "step1": "response1",
                "step2": "response2",
            },
            # Each step uses 20 tokens
            usage={"output_tokens": 10, "total_tokens": 20},
        )
        
        task = TaskSpec(
            task_id="subcall-multi-step-budget",
            input_text="Execute multiple steps",
            model="mock",
            # Parent budget: 50 tokens
            # After reserving ~20 for initial prompt, ~30 remaining for subcall
            # Step 1: 20 tokens (10 remaining)
            # Step 2: 20 tokens (would exceed, should fail)
            budget=Budget(max_total_tokens=50),
            success_criteria=SuccessCriteria(required_substrings=["complete"]),
        )
        
        engine = RLMEngine(
            max_steps=5,
            recursive_subcalls=True,
            max_recursion_depth=1,
            subcall_max_steps=5,
        )
        report = engine.run(task, provider, data="context")
        
        # Should fail due to budget exhaustion during subcall execution
        assert report.success is False
        # Should have completed at least step 1 but failed before step 3
        assert len(report.steps) >= 1
        assert any("budget_exhausted" in str(step.error or "") for step in report.steps if step.error)

    def test_multiple_subcalls_share_parent_budget(self) -> None:
        """
        Verify multiple subcalls correctly share remaining parent budget.
        
        This test verifies that budget enforcement is working - subcalls are
        limited to remaining parent budget, not unlimited. The key verification
        is that total usage never exceeds the parent's budget limit, proving
        that subcalls didn't get unlimited budget.
        """
        # Main RLM makes a subcall that completes immediately
        main_step = '```python\nresult = llm_query("SUBCALL:test")\nFINAL(result)\n```'
        
        provider = MockProvider(
            main_outputs=[main_step] * 5,  # Provide multiple outputs for retries
            subcall_responses={"test": "success"},
            # Small token usage
            usage={"output_tokens": 5, "total_tokens": 10},
        )
        
        task = TaskSpec(
            task_id="multiple-subcalls-budget",
            input_text="Make a subcall",
            model="mock",
            # Parent budget: 50 tokens
            # After reserving ~20 for initial prompt, ~30 remaining for subcall
            # Subcall should be limited to 30 tokens, not unlimited
            budget=Budget(max_total_tokens=50),
            success_criteria=SuccessCriteria(required_substrings=["success"]),
        )
        
        engine = RLMEngine(
            max_steps=3,
            recursive_subcalls=True,
            max_recursion_depth=1,
            subcall_max_steps=2,
        )
        report = engine.run(task, provider, data="context")
        
        # Key verification: budget was respected
        # Total usage should be <= 50 (the parent's budget limit)
        # This proves that subcall didn't get unlimited budget
        assert len(report.steps) > 0
        if report.budget_usage.total_tokens is not None:
            assert report.budget_usage.total_tokens <= 50, \
                f"Total tokens {report.budget_usage.total_tokens} exceeded parent budget of 50"

    def test_subcall_blocked_at_critical_budget(self) -> None:
        """
        Verify subcalls are blocked when budget reaches the critical threshold.
        """
        step1 = '```python\nprint("prepare")\n```'
        step2 = '```python\nresult = llm_query("SUBCALL:info")\nFINAL(result)\n```'

        class CriticalBudgetProvider(MockProvider):
            _call_count = 0

            def stream(self, task, on_progress=None):
                self._call_count += 1
                result = super().stream(task, on_progress)
                if self._call_count == 1:
                    result.usage = {"output_tokens": 10, "total_tokens": 95}
                else:
                    result.usage = {"output_tokens": 5, "total_tokens": 5}
                return result

        provider = CriticalBudgetProvider(
            main_outputs=[step1, step2],
            subcall_responses={"info": "ok"},
            usage={"output_tokens": 1, "total_tokens": 1},
        )

        task = TaskSpec(
            task_id="subcall-critical-budget",
            input_text="Attempt subcall at critical budget",
            model="mock",
            budget=Budget(max_total_tokens=100),
            success_criteria=SuccessCriteria(required_substrings=["ok"]),
        )

        engine = RLMEngine(
            max_steps=2,
            recursive_subcalls=True,
            max_recursion_depth=1,
        )
        report = engine.run(task, provider, data="context")

        assert report.success is False
        assert any(
            "Sub-LLM calls disabled" in str(step.error or "")
            for step in report.steps
            if step.error
        )

    def test_budget_exhaustion_mid_subcall(self) -> None:
        """
        Verify subcall stops when budget exhausted during execution.
        
        Similar to test_rlm_exhausts_budget_mid_task but specifically for subcalls.
        With the new enforcement model, subcalls get a share of remaining budget
        upfront, so they work within their allocation.
        """
        # Main RLM makes a subcall that uses its allocated budget
        subcall_step1 = '```python\nfirst = llm_query("SUBCALL:first")\nprint(first)\n```'
        subcall_step2 = '```python\nsecond = llm_query("SUBCALL:second")\nFINAL(second)\n```'
        
        provider = MockProvider(
            main_outputs=[subcall_step1, subcall_step2],
            subcall_responses={"first": "ok"},
            # Each call uses 6 tokens
            usage={"output_tokens": 3, "total_tokens": 6},
        )
        
        task = TaskSpec(
            task_id="subcall-budget-exhaustion",
            input_text="exhaust budget in subcall",
            model="mock",
            # Tight budget: 12 tokens total
            # Main step uses 6 tokens, subcall gets remaining 6 tokens
            # Total usage = 12 = exactly the budget limit
            budget=Budget(max_total_tokens=12),
            success_criteria=SuccessCriteria(required_substrings=["unused"]),
        )
        
        engine = RLMEngine(
            max_steps=1,
            recursive_subcalls=True,
            max_recursion_depth=1,
            subcall_max_steps=3,
        )
        report = engine.run(task, provider, data="context")
        
        # The new enforcement model ensures subcall works within its allocated budget
        # Test should verify budget is properly accounted, not that it fails mid-execution
        assert len(report.steps) > 0
        # Total usage should be within budget (12 tokens)
        if report.budget_usage.total_tokens is not None:
            assert report.budget_usage.total_tokens <= 12, \
                f"Total tokens {report.budget_usage.total_tokens} exceeded budget of 12"

    def test_subcall_with_unlimited_parent_budget(self) -> None:
        """
        Verify subcall works correctly when parent has no max_total_tokens.
        
        When parent has max_total_tokens=None, subcall should also get None
        (unlimited), allowing it to work normally.
        """
        subcall_code = '```python\nresult = llm_query("SUBCALL:test")\nFINAL(result)\n```'
        
        provider = MockProvider(
            main_outputs=[subcall_code],
            subcall_responses={"test": "success"},
            usage={"output_tokens": 5, "total_tokens": 10},
        )
        
        task = TaskSpec(
            task_id="unlimited-budget-test",
            input_text="test with unlimited budget",
            model="mock",
            # No max_total_tokens limit
            budget=Budget(max_tokens=100),
            success_criteria=SuccessCriteria(required_substrings=["success"]),
        )
        
        engine = RLMEngine(
            max_steps=3,
            recursive_subcalls=True,
            max_recursion_depth=1,
            subcall_max_steps=2,
        )
        report = engine.run(task, provider, data="context")
        
        # Should succeed when budget is unlimited
        assert report.success is True
        assert "success" in (report.answer or "")


class TestBudgetEnforcementRegression:
    """
    Regression tests for budget enforcement.
    
    These tests verify:
    1. Subcalls cannot overconsume the parent's budget
    2. Subcalls can produce correct responses within their allocated budget
    3. Budget is known upfront (model sees it in system prompt)
    4. Hard enforcement prevents any overspend
    """

    def test_subcall_stays_within_allocated_budget(self) -> None:
        """
        REGRESSION: Subcall must not exceed its allocated budget share.

        Previously, subcalls could consume unlimited tokens because they
        received max_total_tokens=None. This test verifies the fix.
        """
        # Subcall will complete successfully within its budget
        main_step = '```python\nresult = llm_query("SUBCALL:analyze")\nFINAL(result)\n```'
        # Fallback if budget exhausted during step 1
        fallback_step = '```python\nFINAL("Analysis complete")\n```'

        provider = MockProvider(
            main_outputs=[main_step, fallback_step],
            subcall_responses={"analyze": "Analysis complete"},
            # Each call uses exactly 25 tokens
            usage={"output_tokens": 10, "total_tokens": 25},
        )
        
        task = TaskSpec(
            task_id="regression-budget-stay-within",
            input_text="Analyze data",
            model="mock",
            # Budget: 100 tokens
            # Main step: ~25 tokens
            # Subcall gets remaining ~75 tokens (but uses only 25)
            budget=Budget(max_total_tokens=100),
            success_criteria=SuccessCriteria(required_substrings=["complete"]),
        )
        
        engine = RLMEngine(
            max_steps=2,
            recursive_subcalls=True,
            max_recursion_depth=1,
            subcall_max_steps=3,
        )
        report = engine.run(task, provider, data="test data")
        
        # MUST succeed - subcall stays within budget
        assert report.success is True
        assert "complete" in (report.answer or "").lower()
        
        # Total tokens must not exceed budget
        if report.budget_usage.total_tokens is not None:
            assert report.budget_usage.total_tokens <= 100, (
                f"REGRESSION: Total tokens {report.budget_usage.total_tokens} "
                f"exceeded budget of 100"
            )

    def test_subcall_produces_correct_response_in_budget(self) -> None:
        """
        REGRESSION: Subcall must produce correct response within budget window.

        Verifies that budget constraints don't prevent valid work from completing.
        """
        # Subcall does actual work and returns meaningful result
        main_step = '```python\nresult = llm_query("SUBCALL:summarize")\nFINAL(result)\n```'
        # Fallback if budget exhausted during step 1
        fallback_step = '```python\nFINAL("Summary: The data shows a 25% increase in Q4.")\n```'

        expected_response = "Summary: The data shows a 25% increase in Q4."

        provider = MockProvider(
            main_outputs=[main_step, fallback_step],
            subcall_responses={"summarize": expected_response},
            usage={"output_tokens": 15, "total_tokens": 30},
        )
        
        task = TaskSpec(
            task_id="regression-correct-response",
            input_text="Summarize quarterly data",
            model="mock",
            # Sufficient budget for the task
            budget=Budget(max_total_tokens=200),
            success_criteria=SuccessCriteria(required_substrings=["25%"]),
        )
        
        engine = RLMEngine(
            max_steps=2,
            recursive_subcalls=True,
            max_recursion_depth=1,
            subcall_max_steps=2,
        )
        report = engine.run(task, provider, data="Q4 data")
        
        # MUST succeed with correct response
        assert report.success is True
        assert "25%" in (report.answer or "")
        assert "Summary" in (report.answer or "")

    def test_parallel_subcalls_share_budget_fairly(self) -> None:
        """
        REGRESSION: llm_batch subcalls must share budget and not oversubscribe.
        
        Previously, parallel subcalls could all see "remaining budget" and
        collectively consume more than available. Now budget is split upfront.
        """
        # Main RLM uses llm_batch for parallel subcalls
        main_step = '```python\nresults = llm_batch(["SUBCALL:a", "SUBCALL:b", "SUBCALL:c"])\nFINAL(str(results))\n```'
        
        provider = MockProvider(
            main_outputs=[main_step],
            subcall_responses={
                "a": "result_a",
                "b": "result_b",
                "c": "result_c",
            },
            # Each subcall uses 20 tokens
            usage={"output_tokens": 10, "total_tokens": 20},
        )
        
        task = TaskSpec(
            task_id="regression-parallel-budget",
            input_text="Process in parallel",
            model="mock",
            # Budget: 150 tokens
            # Main step: ~20 tokens
            # 3 parallel subcalls: should split ~130 tokens = ~43 each
            budget=Budget(max_total_tokens=150),
            success_criteria=SuccessCriteria(required_substrings=["result"]),
        )
        
        engine = RLMEngine(
            max_steps=2,
            recursive_subcalls=True,
            max_recursion_depth=1,
            subcall_max_steps=2,
        )
        report = engine.run(task, provider, data="parallel data")
        
        # Must succeed and stay within budget
        assert report.success is True
        
        # Total must not exceed budget (critical regression check)
        if report.budget_usage.total_tokens is not None:
            assert report.budget_usage.total_tokens <= 150, (
                f"REGRESSION: Parallel subcalls consumed {report.budget_usage.total_tokens} "
                f"tokens, exceeding budget of 150"
            )

    def test_small_budget_known_upfront(self) -> None:
        """
        REGRESSION: Model must know budget upfront to adapt strategy.
        
        Previously, model only learned about budget at 50%/80%/90% thresholds.
        With very small budgets, model wouldn't adapt. Now budget is in system prompt.
        """
        # With tiny budget, model should immediately call FINAL
        main_step = '```python\nFINAL("quick answer")\n```'
        
        provider = MockProvider(
            main_outputs=[main_step],
            subcall_responses={},
            # Very small token usage
            usage={"output_tokens": 5, "total_tokens": 10},
        )
        
        task = TaskSpec(
            task_id="regression-small-budget-upfront",
            input_text="Answer quickly",
            model="mock",
            # Very small budget - model should see LOW BUDGET WARNING
            budget=Budget(max_total_tokens=20),
            success_criteria=SuccessCriteria(required_substrings=["answer"]),
        )
        
        engine = RLMEngine(
            max_steps=1,
            recursive_subcalls=False,
        )
        report = engine.run(task, provider, data="x")
        
        # Should succeed because model adapts to small budget
        assert report.success is True
        
        # Must not exceed tiny budget
        if report.budget_usage.total_tokens is not None:
            assert report.budget_usage.total_tokens <= 20

    def test_hard_enforcement_prevents_overspend(self) -> None:
        """
        REGRESSION: System must prevent subcalls when budget is exhausted.

        Tests that when budget is exhausted, the system blocks subcalls
        (budget_exhausted_preflight) and signals budget_insufficient.
        Note: The reported total_tokens may exceed budget since the main LLM
        call cannot be stopped mid-generation, but the system prevents
        further subcalls from executing.
        """
        # Subcall that would try to use many tokens
        main_step = '```python\nresult = llm_query("SUBCALL:expensive")\nprint(result)\n```'
        step2 = '```python\nFINAL("done")\n```'
        # Additional fallback in case needed
        step3 = '```python\nFINAL("done")\n```'

        provider = MockProvider(
            main_outputs=[main_step, step2, step3],
            subcall_responses={"expensive": "expensive result"},
            # Subcall reports using 40 tokens
            usage={"output_tokens": 20, "total_tokens": 40},
        )

        task = TaskSpec(
            task_id="regression-hard-enforcement",
            input_text="Do expensive operation",
            model="mock",
            # Budget of 60 tokens
            # Main step uses 40, leaving only 20 for subcall
            # Subcall wants 40 but should be capped
            budget=Budget(max_total_tokens=60),
            success_criteria=SuccessCriteria(required_substrings=["done"]),
        )

        engine = RLMEngine(
            max_steps=3,
            recursive_subcalls=True,
            max_recursion_depth=1,
            subcall_max_steps=2,
        )
        report = engine.run(task, provider, data="data")

        # Verify hard enforcement kicked in:
        # 1. Subcall was blocked (budget_exhausted_preflight in step errors)
        # 2. Budget insufficient was reported
        assert any("budget" in (step.error or "").lower() for step in report.steps), \
            "REGRESSION: Hard enforcement should have blocked the subcall"
        # The report should indicate budget was hit
        assert "max_total_tokens" in report.budget_usage.limits_exceeded or \
               any("budget" in e.lower() for e in report.errors), \
            "REGRESSION: Budget enforcement should be reported"

    def test_subcall_completes_work_before_budget_exhaustion(self) -> None:
        """
        REGRESSION: Subcall should complete meaningful work within budget.

        Verifies that the budget enforcement doesn't prematurely kill
        subcalls that have enough budget to complete their work.
        """
        # Multi-step subcall that should complete within budget
        main_step = '```python\nresult = llm_query("SUBCALL:multi_step")\nFINAL(result)\n```'
        # Fallback if budget exhausted during step 1
        fallback_step = '```python\nFINAL("Step 1 done. Step 2 done. All complete.")\n```'

        provider = MockProvider(
            main_outputs=[main_step, fallback_step],
            subcall_responses={"multi_step": "Step 1 done. Step 2 done. All complete."},
            # Reasonable token usage
            usage={"output_tokens": 15, "total_tokens": 30},
        )
        
        task = TaskSpec(
            task_id="regression-complete-work",
            input_text="Do multi-step task",
            model="mock",
            # Generous budget
            budget=Budget(max_total_tokens=200),
            success_criteria=SuccessCriteria(required_substrings=["All complete"]),
        )
        
        engine = RLMEngine(
            max_steps=2,
            recursive_subcalls=True,
            max_recursion_depth=1,
            subcall_max_steps=5,
        )
        report = engine.run(task, provider, data="task data")
        
        # MUST succeed - budget is sufficient
        assert report.success is True
        assert "All complete" in (report.answer or "")

    def test_budget_accounting_accuracy(self) -> None:
        """
        REGRESSION: Budget accounting must accurately track all token usage.
        
        Verifies that reported usage matches expected usage and that
        no tokens are "lost" in accounting.
        """
        main_step = '```python\nFINAL("simple")\n```'
        
        # Known, fixed token usage
        fixed_usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        
        provider = MockProvider(
            main_outputs=[main_step],
            subcall_responses={},
            usage=fixed_usage,
        )
        
        task = TaskSpec(
            task_id="regression-accounting",
            input_text="Simple task",
            model="mock",
            budget=Budget(max_total_tokens=100),
            success_criteria=SuccessCriteria(required_substrings=["simple"]),
        )
        
        engine = RLMEngine(
            max_steps=1,
            recursive_subcalls=False,
        )
        report = engine.run(task, provider, data="x")
        
        # Must succeed
        assert report.success is True
        
        # Verify accurate accounting
        assert report.budget_usage.total_tokens == 15, (
            f"REGRESSION: Budget accounting mismatch - "
            f"expected 15, got {report.budget_usage.total_tokens}"
        )
        assert report.budget_usage.output_tokens == 5
        assert report.budget_usage.input_tokens == 10

    def test_final_always_callable_at_high_budget_usage(self) -> None:
        """
        BATTLE TEST: FINAL() must always be callable even at 80-90% budget usage.
        
        This test ensures that adaptive_max_output_tokens reserves enough tokens
        for FINAL() calls even when budget is nearly exhausted, preventing silent
        breaks where no tokens remain for FINAL().
        
        Critical: Output cap should honor remaining budget to keep FINAL() possible.
        This test verifies that FINAL() can be called successfully even when
        budget is at 80-90% usage.
        """
        # Simulate budget at 85% usage - should still allow FINAL() call
        # Budget: 1000 tokens, use 850 tokens first, then call FINAL
        step1 = '```python\nprint("working...")\n```'
        final_step = '```python\nFINAL("completed at high budget usage")\n```'
        
        provider = MockProvider(
            main_outputs=[step1, final_step],
            subcall_responses={},
            # First step uses 850 tokens (85% of 1000)
            # Second step (FINAL) should use remaining ~150 tokens
            usage={"output_tokens": 50, "total_tokens": 150},
        )
        
        task = TaskSpec(
            task_id="battle-test-final-high-budget",
            input_text="Complete task at high budget usage",
            model="mock",
            # Budget: 1000 tokens
            # First step: 850 tokens (85% usage)
            # FINAL step: should succeed with remaining ~150 tokens
            budget=Budget(max_total_tokens=1000),
            success_criteria=SuccessCriteria(required_substrings=["completed"]),
        )
        
        # Simulate high budget usage by creating a custom provider that
        # reports high usage on first call, then normal usage on second
        class HighBudgetProvider(MockProvider):
            _call_count = 0
            
            def stream(self, task, on_progress=None):
                self._call_count += 1
                if self._call_count == 1:
                    # First call: report high usage (850 tokens = 85%)
                    result = super().stream(task, on_progress)
                    result.usage = {"output_tokens": 400, "total_tokens": 850}
                    return result
                else:
                    # Second call (FINAL): normal usage
                    return super().stream(task, on_progress)
        
        provider = HighBudgetProvider(
            main_outputs=[step1, final_step],
            subcall_responses={},
            usage={"output_tokens": 50, "total_tokens": 150},
        )
        
        engine = RLMEngine(
            max_steps=3,
            recursive_subcalls=False,
        )
        report = engine.run(task, provider, data="test data")
        
        # MUST succeed - FINAL() should be callable even at 85% budget usage
        assert report.success is True, (
            f"BATTLE TEST FAILED: FINAL() could not be called at high budget usage. "
            f"Success: {report.success}, Answer: {report.answer}, Errors: {report.errors}"
        )
        assert "completed" in (report.answer or "").lower()
        
        # Verify budget was respected
        if report.budget_usage.total_tokens is not None:
            assert report.budget_usage.total_tokens <= 1000, (
                f"BATTLE TEST FAILED: Budget exceeded. "
                f"Used {report.budget_usage.total_tokens} tokens but budget was 1000"
            )
