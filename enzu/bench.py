from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from enzu.api import resolve_provider
from enzu.engine import Engine
from enzu.models import TaskSpec
from enzu.rlm import RLMEngine
from enzu.tools.filesystem import FS_TOOL_GUIDANCE, build_fs_helpers

# Match CLI defaults so benchmark tasks mirror single-run behavior.
DEFAULT_BUDGET = {"max_output_tokens": 256}
DEFAULT_SUCCESS = {"min_word_count": 1}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run benchmark tasks from JSONL and collect results."
    )
    parser.add_argument("--tasks", required=True, help="Path to tasks JSONL file.")
    parser.add_argument("--matrix", help="JSON file with a list of run configs.")
    parser.add_argument(
        "--provider", help="Provider name (used when --matrix is absent)."
    )
    parser.add_argument("--model", help="Model name (used when --matrix is absent).")
    parser.add_argument("--api-key", help="API key override (single-run only).")
    parser.add_argument("--referer", help="OpenRouter HTTP-Referer header override.")
    parser.add_argument("--app-name", help="OpenRouter X-Title header override.")
    parser.add_argument("--organization", help="OpenAI organization override.")
    parser.add_argument("--project", help="OpenAI project override.")
    parser.add_argument("--mode", choices=["chat", "rlm", "automode"], default="chat")
    parser.add_argument("--context-file", help="Default context file for RLM tasks.")
    parser.add_argument("--fs-root", help="Default fs_root for automode tasks.")
    parser.add_argument("--max-steps", type=int, help="Default max RLM steps.")
    parser.add_argument(
        "--temperature", type=float, help="Default temperature override."
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat each task N times."
    )
    parser.add_argument("--output", help="Output JSONL path.")
    parser.add_argument("--summary", help="Summary JSON path.")
    parser.add_argument(
        "--progress", action="store_true", help="Emit progress to stderr."
    )
    return parser.parse_args()


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_tasks(path: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"Task line {lineno} must be a JSON object.")
            data["_line"] = lineno
            tasks.append(data)
    if not tasks:
        raise ValueError("No tasks found in JSONL file.")
    return tasks


def _load_run_matrix(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.matrix:
        matrix = _load_json(args.matrix)
        if isinstance(matrix, dict):
            runs = matrix.get("runs") or matrix.get("providers") or matrix.get("matrix")
        else:
            runs = matrix
        if not isinstance(runs, list):
            raise ValueError("Matrix JSON must be a list or contain a 'runs' list.")
        return runs
    if not args.provider or not args.model:
        raise ValueError("provider and model are required when --matrix is not used.")
    return [
        {
            "provider": args.provider,
            "model": args.model,
            "mode": args.mode,
            "api_key": args.api_key,
            "referer": args.referer,
            "app_name": args.app_name,
            "organization": args.organization,
            "project": args.project,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
        }
    ]


def _normalize_budget(task_data: Dict[str, Any]) -> Dict[str, Any]:
    budget = task_data.get("budget")
    if isinstance(budget, dict):
        return budget
    fallback = {
        "max_output_tokens": task_data.get("max_output_tokens"),
        "max_total_tokens": task_data.get("max_total_tokens"),
        "max_seconds": task_data.get("max_seconds"),
        "max_cost_usd": task_data.get("max_cost_usd"),
    }
    if any(value is not None for value in fallback.values()):
        return fallback
    return dict(DEFAULT_BUDGET)


def _normalize_success_criteria(task_data: Dict[str, Any]) -> Dict[str, Any]:
    criteria = task_data.get("success_criteria")
    if isinstance(criteria, dict):
        return criteria
    fallback: Dict[str, Any] = {}
    if task_data.get("required_substrings"):
        fallback["required_substrings"] = task_data["required_substrings"]
    if task_data.get("required_regex"):
        fallback["required_regex"] = task_data["required_regex"]
    if task_data.get("min_word_count"):
        fallback["min_word_count"] = task_data["min_word_count"]
    if task_data.get("case_insensitive") is not None:
        fallback["case_insensitive"] = task_data["case_insensitive"]
    if fallback:
        return fallback
    return dict(DEFAULT_SUCCESS)


def _merge_metadata(task_data: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if isinstance(task_data.get("metadata"), dict):
        metadata.update(task_data["metadata"])
    if raw is not task_data and isinstance(raw.get("metadata"), dict):
        metadata.update(raw["metadata"])
    return metadata


def _build_task_spec(
    raw: Dict[str, Any],
    *,
    model_override: Optional[str],
    temperature_override: Optional[float],
) -> TaskSpec:
    task_data = raw.get("task") if isinstance(raw.get("task"), dict) else raw
    if not isinstance(task_data, dict):
        raise ValueError("Task payload must be a JSON object.")
    input_text = task_data.get("input_text")
    if not input_text:
        raise ValueError("Task is missing input_text.")
    model = model_override or task_data.get("model")
    if not model:
        raise ValueError("Task is missing model and no override was provided.")
    task_id = (
        task_data.get("task_id") or raw.get("task_id") or f"task-{uuid4().hex[:8]}"
    )
    payload = {
        "task_id": task_id,
        "input_text": input_text,
        "model": model,
        "budget": _normalize_budget(task_data),
        "success_criteria": _normalize_success_criteria(task_data),
        "max_output_tokens": task_data.get("max_output_tokens"),
        "temperature": temperature_override
        if temperature_override is not None
        else task_data.get("temperature"),
        "metadata": _merge_metadata(task_data, raw),
    }
    # Use TaskSpec validation to enforce required fields and limits.
    return TaskSpec.model_validate(payload)


def _resolve_mode(raw: Dict[str, Any], run: Dict[str, Any], default_mode: str) -> str:
    task_mode = raw.get("mode") or (
        raw.get("task", {}) if isinstance(raw.get("task"), dict) else {}
    ).get("mode")
    run_mode = run.get("mode")
    return (task_mode or run_mode or default_mode).lower()


def _resolve_context(
    raw: Dict[str, Any], default_context_file: Optional[str]
) -> Optional[str]:
    if raw.get("context") is not None:
        return raw.get("context")
    task_context = None
    if isinstance(raw.get("task"), dict):
        task_context = raw["task"].get("context")
    if task_context is not None:
        return task_context
    context_file = raw.get("context_file") or (
        raw.get("task", {}) if isinstance(raw.get("task"), dict) else {}
    ).get("context_file")
    if context_file:
        with open(context_file, "r", encoding="utf-8") as handle:
            return handle.read()
    if default_context_file:
        with open(default_context_file, "r", encoding="utf-8") as handle:
            return handle.read()
    return None


def _resolve_fs_root(raw: Dict[str, Any], default_root: Optional[str]) -> Optional[str]:
    if raw.get("fs_root"):
        return raw["fs_root"]
    if isinstance(raw.get("task"), dict) and raw["task"].get("fs_root"):
        return raw["task"]["fs_root"]
    return default_root


def _resolve_max_steps(
    raw: Dict[str, Any], run: Dict[str, Any], default_steps: Optional[int]
) -> Optional[int]:
    task_steps = raw.get("max_steps") or (
        raw.get("task", {}) if isinstance(raw.get("task"), dict) else {}
    ).get("max_steps")
    return (
        task_steps if task_steps is not None else run.get("max_steps") or default_steps
    )


def _default_output_paths() -> tuple[Path, Path]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = Path("artifacts") / "benchmarks"
    return base / f"run-{stamp}.jsonl", base / f"run-{stamp}.summary.json"


def _init_stats() -> Dict[str, Any]:
    return {
        "count": 0,
        "success": 0,
        "limits_exceeded": 0,
        "elapsed_total": 0.0,
        "elapsed_count": 0,
        "output_tokens_total": 0,
        "output_tokens_count": 0,
        "total_tokens_total": 0,
        "total_tokens_count": 0,
        "cost_total": 0.0,
        "cost_count": 0,
    }


def _update_stats(stats: Dict[str, Any], result: Dict[str, Any]) -> None:
    stats["count"] += 1
    if result.get("success"):
        stats["success"] += 1
    budget = result.get("budget_usage") or {}
    elapsed = budget.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)):
        stats["elapsed_total"] += float(elapsed)
        stats["elapsed_count"] += 1
    output_tokens = budget.get("output_tokens")
    if isinstance(output_tokens, int):
        stats["output_tokens_total"] += output_tokens
        stats["output_tokens_count"] += 1
    total_tokens = budget.get("total_tokens")
    if isinstance(total_tokens, int):
        stats["total_tokens_total"] += total_tokens
        stats["total_tokens_count"] += 1
    cost_usd = budget.get("cost_usd")
    if isinstance(cost_usd, (int, float)):
        stats["cost_total"] += float(cost_usd)
        stats["cost_count"] += 1
    if budget.get("limits_exceeded"):
        stats["limits_exceeded"] += 1


def _finalize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    count = stats["count"]
    return {
        "total_runs": count,
        "successes": stats["success"],
        "success_rate": stats["success"] / count if count else 0.0,
        "limits_exceeded": stats["limits_exceeded"],
        "avg_elapsed_seconds": stats["elapsed_total"] / stats["elapsed_count"]
        if stats["elapsed_count"]
        else None,
        "avg_output_tokens": stats["output_tokens_total"] / stats["output_tokens_count"]
        if stats["output_tokens_count"]
        else None,
        "avg_total_tokens": stats["total_tokens_total"] / stats["total_tokens_count"]
        if stats["total_tokens_count"]
        else None,
        "avg_cost_usd": stats["cost_total"] / stats["cost_count"]
        if stats["cost_count"]
        else None,
    }


def _build_result_row(
    *,
    run_id: str,
    run_label: str,
    run_index: int,
    task_index: int,
    mode: str,
    report: Any,
) -> Dict[str, Any]:
    is_rlm = hasattr(report, "answer")
    output_text = report.answer if is_rlm else report.output_text
    verification = None
    if hasattr(report, "verification"):
        verification = {
            "passed": report.verification.passed,
            "reasons": report.verification.reasons,
        }
    return {
        "run_id": run_id,
        "run_label": run_label,
        "run_index": run_index,
        "task_index": task_index,
        "task_id": report.task_id,
        "provider": report.provider,
        "model": report.model,
        "mode": mode,
        "success": report.success,
        "verification": verification,
        "budget_usage": report.budget_usage.model_dump(mode="json"),
        "errors": report.errors,
        "output_text": output_text,
    }


def _run_task(
    *,
    spec: TaskSpec,
    provider: Any,
    mode: str,
    context: Optional[str],
    fs_root: Optional[str],
    max_steps: Optional[int],
    progress: bool,
) -> Any:
    if mode == "chat":
        engine = Engine()
        return engine.run(
            spec,
            provider,
            on_progress=(lambda event: print(event.message, file=sys.stderr))
            if progress
            else None,
        )
    if mode == "automode":
        if not fs_root:
            raise ValueError("fs_root is required for automode tasks.")
        fs_helpers = build_fs_helpers(fs_root)
        fs_snapshot = fs_helpers["fs_snapshot"](path=".", depth=2)
        context_payload = json.dumps(
            {"fs_root": fs_helpers["fs_root"](), "fs_snapshot": fs_snapshot},
            ensure_ascii=True,
        )
        spec = spec.model_copy(
            update={
                "metadata": {
                    **spec.metadata,
                    # Match CLI automode metadata for tool guidance and fs_root.
                    "tools_guidance": FS_TOOL_GUIDANCE,
                    "fs_root": fs_helpers["fs_root"](),
                }
            }
        )
        rlm_engine = RLMEngine(max_steps=max_steps) if max_steps else RLMEngine()
        return rlm_engine.run(
            spec,
            provider,
            data=context_payload,
            namespace=fs_helpers,
            on_progress=(lambda msg: print(msg, file=sys.stderr)) if progress else None,
        )
    if mode == "rlm":
        if context is None:
            raise ValueError("context is required for rlm tasks.")
        rlm_engine = RLMEngine(max_steps=max_steps) if max_steps else RLMEngine()
        return rlm_engine.run(
            spec,
            provider,
            data=context,
            on_progress=(lambda msg: print(msg, file=sys.stderr)) if progress else None,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> int:
    args = _parse_args()
    try:
        tasks = _load_tasks(args.tasks)
        runs = _load_run_matrix(args)
        output_path, summary_path = _default_output_paths()
        if args.output:
            output_path = Path(args.output)
        if args.summary:
            summary_path = Path(args.summary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summaries: List[Dict[str, Any]] = []
        overall_stats = _init_stats()

        with open(output_path, "w", encoding="utf-8") as out_handle:
            for run_index, run in enumerate(runs, 1):
                provider_name = run.get("provider")
                model_name = run.get("model")
                if not provider_name or not model_name:
                    raise ValueError("Each run config must include provider and model.")
                run_label = (
                    run.get("name")
                    or f"{provider_name}:{model_name}:{run.get('mode', args.mode)}"
                )
                provider = resolve_provider(
                    provider_name,
                    api_key=run.get("api_key") or args.api_key,
                    referer=run.get("referer") or args.referer,
                    app_name=run.get("app_name") or args.app_name,
                    organization=run.get("organization") or args.organization,
                    project=run.get("project") or args.project,
                )
                run_stats = _init_stats()
                for repeat_index in range(args.repeat):
                    for task_index, raw in enumerate(tasks, 1):
                        mode = _resolve_mode(raw, run, args.mode)
                        max_steps = _resolve_max_steps(raw, run, args.max_steps)
                        spec = _build_task_spec(
                            raw,
                            model_override=model_name,
                            temperature_override=run.get("temperature")
                            or args.temperature,
                        )
                        context = _resolve_context(raw, args.context_file)
                        fs_root = _resolve_fs_root(raw, args.fs_root)
                        if args.progress:
                            print(
                                f"[bench] run={run_label} repeat={repeat_index + 1}/{args.repeat} "
                                f"task={task_index}/{len(tasks)} mode={mode}",
                                file=sys.stderr,
                            )
                        report = _run_task(
                            spec=spec,
                            provider=provider,
                            mode=mode,
                            context=context,
                            fs_root=fs_root,
                            max_steps=max_steps,
                            progress=args.progress,
                        )
                        row = _build_result_row(
                            run_id=uuid4().hex,
                            run_label=run_label,
                            run_index=run_index,
                            task_index=task_index,
                            mode=mode,
                            report=report,
                        )
                        out_handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                        out_handle.flush()
                        _update_stats(run_stats, row)
                        _update_stats(overall_stats, row)

                summaries.append(
                    {
                        "run_label": run_label,
                        "provider": provider_name,
                        "model": model_name,
                        "mode": run.get("mode", args.mode),
                        "stats": _finalize_stats(run_stats),
                    }
                )

        summary_payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tasks_file": args.tasks,
            "runs": summaries,
            "overall": _finalize_stats(overall_stats),
            "output_path": str(output_path),
        }
        with open(summary_path, "w", encoding="utf-8") as summary_handle:
            summary_handle.write(
                json.dumps(summary_payload, ensure_ascii=True, indent=2)
            )
        print(json.dumps(summary_payload, ensure_ascii=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
