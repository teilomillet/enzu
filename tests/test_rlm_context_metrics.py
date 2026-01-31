"""Tests for RLM context usage metrics and tracking."""

from enzu.rlm.context_metrics import (
    ContextBreakdown,
    RLMTrajectoryMetrics,
    RLMContextTracker,
    get_rlm_context_tracker,
    reset_rlm_context_tracker,
)


class TestContextBreakdown:
    def test_creation(self):
        breakdown = ContextBreakdown(
            system_prompt_chars=1000,
            task_prompt_chars=200,
            inline_data_chars=5000,
            file_data_chars=0,
        )
        assert breakdown.system_prompt_chars == 1000
        assert breakdown.task_prompt_chars == 200
        assert breakdown.inline_data_chars == 5000
        assert breakdown.file_data_chars == 0

    def test_total_context_chars(self):
        breakdown = ContextBreakdown(
            system_prompt_chars=1000,
            task_prompt_chars=200,
            inline_data_chars=3000,
            file_data_chars=2000,
        )
        assert breakdown.total_context_chars() == 6200

    def test_symbolic_ratio_all_inline(self):
        breakdown = ContextBreakdown(
            inline_data_chars=5000,
            file_data_chars=0,
        )
        assert breakdown.symbolic_ratio() == 0.0

    def test_symbolic_ratio_all_file(self):
        breakdown = ContextBreakdown(
            inline_data_chars=0,
            file_data_chars=5000,
        )
        assert breakdown.symbolic_ratio() == 1.0

    def test_symbolic_ratio_mixed(self):
        breakdown = ContextBreakdown(
            inline_data_chars=3000,
            file_data_chars=7000,
        )
        assert breakdown.symbolic_ratio() == 0.7

    def test_symbolic_ratio_no_data(self):
        breakdown = ContextBreakdown()
        assert breakdown.symbolic_ratio() == 0.0

    def test_context_efficiency_no_symbolic(self):
        breakdown = ContextBreakdown(
            inline_data_chars=5000,
            file_data_chars=0,
        )
        assert breakdown.context_efficiency() == 1.0

    def test_context_efficiency_partial_read(self):
        breakdown = ContextBreakdown(
            file_data_chars=10000,
            file_bytes_read=3000,
        )
        assert breakdown.context_efficiency() == 0.3

    def test_context_efficiency_over_read(self):
        breakdown = ContextBreakdown(
            file_data_chars=5000,
            file_bytes_read=6000,  # Read more than available (e.g., multiple passes)
        )
        # Should cap at 1.0
        assert breakdown.context_efficiency() == 1.0

    def test_to_dict(self):
        breakdown = ContextBreakdown(
            system_prompt_chars=1000,
            task_prompt_chars=200,
            inline_data_chars=3000,
            file_data_chars=2000,
            file_reads=5,
            file_bytes_read=1500,
            depth=2,
            total_steps=8,
            subcalls=3,
            llm_invocations=10,
            used_symbolic_context=True,
            context_path="/tmp/context.txt",
        )

        d = breakdown.to_dict()
        assert d["system_prompt_chars"] == 1000
        assert d["task_prompt_chars"] == 200
        assert d["inline_data_chars"] == 3000
        assert d["file_data_chars"] == 2000
        assert d["total_context_chars"] == 6200
        assert d["file_reads"] == 5
        assert d["file_bytes_read"] == 1500
        assert d["depth"] == 2
        assert d["total_steps"] == 8
        assert d["subcalls"] == 3
        assert d["llm_invocations"] == 10
        assert d["used_symbolic_context"] is True
        assert d["symbolic_ratio"] == 0.4  # 2000 / (3000 + 2000)
        assert d["context_efficiency"] == 0.75  # 1500 / 2000


class TestRLMTrajectoryMetrics:
    def test_creation(self):
        metrics = RLMTrajectoryMetrics(
            run_id="test-run",
            task_id="test-task",
            max_depth=2,
            total_steps=10,
            subcall_count=3,
        )
        assert metrics.run_id == "test-run"
        assert metrics.task_id == "test-task"
        assert metrics.max_depth == 2
        assert metrics.total_steps == 10
        assert metrics.subcall_count == 3

    def test_trajectory_complexity(self):
        metrics = RLMTrajectoryMetrics(
            run_id="test",
            max_depth=2,
            total_steps=8,
            subcall_count=3,
        )
        # depth * 10 + steps + subcalls * 5 = 2*10 + 8 + 3*5 = 43
        assert metrics.trajectory_complexity() == 43.0

    def test_token_efficiency(self):
        metrics = RLMTrajectoryMetrics(
            run_id="test",
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        # 500 / (1000 + 500) = 0.333...
        assert abs(metrics.token_efficiency() - 0.333) < 0.01

    def test_token_efficiency_no_tokens(self):
        metrics = RLMTrajectoryMetrics(run_id="test")
        assert metrics.token_efficiency() == 0.0

    def test_to_dict(self):
        context = ContextBreakdown(
            system_prompt_chars=1000,
            task_prompt_chars=200,
        )

        metrics = RLMTrajectoryMetrics(
            run_id="test-run",
            task_id="test-task",
            max_depth=2,
            total_steps=10,
            subcall_count=3,
            total_input_tokens=1500,
            total_output_tokens=500,
            elapsed_seconds=5.5,
            cost_usd=0.02,
            success=True,
            outcome="success",
            context_breakdown=context,
        )

        d = metrics.to_dict()
        assert d["run_id"] == "test-run"
        assert d["task_id"] == "test-task"
        assert d["max_depth"] == 2
        assert d["total_steps"] == 10
        assert d["subcall_count"] == 3
        assert d["total_input_tokens"] == 1500
        assert d["total_output_tokens"] == 500
        assert d["elapsed_seconds"] == 5.5
        assert d["cost_usd"] == 0.02
        assert d["success"] is True
        assert d["outcome"] == "success"
        assert "context" in d
        assert d["context"]["system_prompt_chars"] == 1000


class TestRLMContextTracker:
    def test_empty_tracker(self):
        tracker = RLMContextTracker()
        assert len(tracker.get_trajectories()) == 0

        summary = tracker.summary()
        assert summary["total_runs"] == 0
        assert summary["avg_depth"] == 0.0

    def test_record_and_retrieve(self):
        tracker = RLMContextTracker()

        m1 = RLMTrajectoryMetrics(
            run_id="run1",
            max_depth=1,
            total_steps=5,
        )
        m2 = RLMTrajectoryMetrics(
            run_id="run2",
            max_depth=2,
            total_steps=8,
        )

        tracker.record(m1)
        tracker.record(m2)

        trajectories = tracker.get_trajectories()
        assert len(trajectories) == 2
        assert trajectories[0].run_id == "run1"
        assert trajectories[1].run_id == "run2"

    def test_clear(self):
        tracker = RLMContextTracker()
        tracker.record(RLMTrajectoryMetrics(run_id="test"))
        assert len(tracker.get_trajectories()) == 1

        tracker.clear()
        assert len(tracker.get_trajectories()) == 0

    def test_summary_statistics(self):
        tracker = RLMContextTracker()

        # Record multiple trajectories
        for i in range(10):
            tracker.record(
                RLMTrajectoryMetrics(
                    run_id=f"run{i}",
                    max_depth=i % 3,  # 0, 1, 2, 0, 1, 2, ...
                    total_steps=i + 1,  # 1, 2, 3, ..., 10
                    subcall_count=i % 2,  # 0, 1, 0, 1, ...
                    success=(i % 2 == 0),  # 5 successes, 5 failures
                )
            )

        summary = tracker.summary()
        assert summary["total_runs"] == 10
        assert summary["success_rate"] == 0.5

        # Average depth: (0+1+2+0+1+2+0+1+2+0) / 10 = 9/10 = 0.9
        assert abs(summary["avg_depth"] - 0.9) < 0.01

        # Average steps: (1+2+...+10) / 10 = 55/10 = 5.5
        assert summary["avg_steps"] == 5.5

        # Average subcalls: (0+1+0+1+...) / 10 = 5/10 = 0.5
        assert summary["avg_subcalls"] == 0.5

    def test_summary_percentiles(self):
        tracker = RLMContextTracker()

        # Record trajectories with varying steps
        steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for i, step_count in enumerate(steps):
            tracker.record(
                RLMTrajectoryMetrics(
                    run_id=f"run{i}",
                    total_steps=step_count,
                    max_depth=0,
                    subcall_count=0,
                )
            )

        summary = tracker.summary()

        # p50 should be around median (5-6)
        assert 5 <= summary["steps_p50"] <= 6

        # p95 should be around 10 (95th percentile)
        assert summary["steps_p95"] == 10


class TestGlobalTracker:
    def test_get_global_tracker(self):
        tracker1 = get_rlm_context_tracker()
        tracker2 = get_rlm_context_tracker()

        # Should return same instance
        assert tracker1 is tracker2

    def test_reset_global_tracker(self):
        tracker1 = get_rlm_context_tracker()
        tracker1.record(RLMTrajectoryMetrics(run_id="test"))

        assert len(tracker1.get_trajectories()) == 1

        reset_rlm_context_tracker()

        tracker2 = get_rlm_context_tracker()
        assert len(tracker2.get_trajectories()) == 0

        # Should be a new instance
        assert tracker1 is not tracker2
