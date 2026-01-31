from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

from enzu.api import resolve_provider
from enzu.contract import task_spec_from_payload
from enzu.schema import schema_bundle
from enzu.tools.filesystem import FS_TOOL_GUIDANCE, build_fs_helpers
from enzu.engine import Engine
from enzu.models import TaskSpec
from enzu.providers.registry import list_providers
from enzu.rlm import RLMEngine


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single enzu task from JSON stdin."
    )
    parser.add_argument(
        "--provider", help="Provider name (e.g., openrouter, openai, ollama, lmstudio)."
    )
    parser.add_argument("--api-key", help="API key override.")
    parser.add_argument("--model", help="Model override.")
    # Keep CLI modes aligned with enzu.schema.RunPayload and --print-schema output.
    parser.add_argument(
        "--mode", choices=["chat", "rlm", "automode"], help="Execution mode."
    )
    parser.add_argument("--task", help="Task text (used when stdin is empty).")
    parser.add_argument("--task-id", help="Task id override (used with --task).")
    parser.add_argument("--context-file", help="Context file for RLM mode.")
    parser.add_argument("--max-steps", type=int, help="Max RLM steps.")
    parser.add_argument(
        "--progress", action="store_true", help="Emit progress to stderr."
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print JSON schema bundle to stdout and exit.",
    )
    parser.add_argument("--fs-root", help="Filesystem root for automode (required).")
    parser.add_argument(
        "--fs-snapshot-depth", type=int, default=2, help="Automode snapshot depth."
    )
    parser.add_argument(
        "--fs-max-entries",
        type=int,
        default=200,
        help="Automode max entries per directory.",
    )
    return parser.parse_args()


def _read_input_from_stdin() -> Optional[Dict[str, Any]]:
    if sys.stdin.isatty():
        return None
    raw = sys.stdin.read()
    if not raw.strip():
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("stdin JSON must be an object.")
    return data


def _build_input_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.task:
        raise ValueError("stdin is empty; provide a JSON task or use --task.")
    if not args.model:
        raise ValueError("model is required via --model when using --task.")
    task_id = args.task_id or "cli-task"
    return {
        "task": {
            "task_id": task_id,
            "input_text": args.task,
            "model": args.model,
        },
        "provider": args.provider,
    }


def _resolve_task(
    data: Dict[str, Any], *, model_override: Optional[str] = None
) -> TaskSpec:
    return task_spec_from_payload(data, model_override=model_override)


def _resolve_context(
    data: Dict[str, Any], context_file: Optional[str]
) -> Optional[str]:
    if context_file:
        with open(context_file, "r", encoding="utf-8") as handle:
            return handle.read()
    if data.get("context_file"):
        with open(data["context_file"], "r", encoding="utf-8") as handle:
            return handle.read()
    if data.get("context"):
        return data["context"]
    return None


def _prompt_line(prompt: str) -> str:
    # Guided mode writes prompts to stderr to keep stdout clean for JSON output.
    print(prompt, file=sys.stderr, end="")
    try:
        return input().strip()
    except EOFError as exc:
        raise ValueError("stdin closed during guided setup.") from exc


def _prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    while True:
        value = _prompt_line(prompt) or default
        if value in choices:
            return value
        print(f"Invalid choice: {value}", file=sys.stderr)


def _guided_payload() -> Dict[str, Any]:
    # Guided mode builds the same JSON payload the CLI accepts.
    providers = list_providers()
    provider_default = (
        "openrouter"
        if "openrouter" in providers
        else (providers[0] if providers else "")
    )
    provider_prompt = (
        f"Provider [{provider_default}]: "
        if provider_default
        else "Provider (required): "
    )
    provider = _prompt_line(provider_prompt) or provider_default
    if not provider:
        raise ValueError("provider is required.")

    mode = _prompt_choice("Mode [chat]: ", ["chat", "rlm", "automode"], "chat")
    model = _prompt_line("Model (required): ")
    if not model:
        raise ValueError("model is required.")
    task = _prompt_line("Task (required): ")
    if not task:
        raise ValueError("task is required.")

    payload: Dict[str, Any] = {
        "mode": mode,
        "provider": provider,
        "task": {
            "task_id": "guided-task",
            "input_text": task,
            "model": model,
        },
    }

    if mode == "rlm":
        context = _prompt_line("Context (paste text or @/path/to/file): ")
        if not context:
            raise ValueError("context is required for mode=rlm.")
        if context.startswith("@"):
            path = context[1:]
            with open(path, "r", encoding="utf-8") as handle:
                payload["context"] = handle.read()
        else:
            payload["context"] = context
    elif mode == "automode":
        fs_root = _prompt_line("Filesystem root (required): ")
        if not fs_root:
            raise ValueError("fs_root is required for mode=automode.")
        payload["fs_root"] = fs_root

    return payload


def main() -> int:
    args = _parse_args()
    try:
        if args.print_schema:
            # Schema output is the canonical contract for CLI and Python integrations.
            print(json.dumps(schema_bundle(), ensure_ascii=True))
            return 0
        if len(sys.argv) == 1 and sys.stdin.isatty():
            # Guided mode is the no-args path for first-time usage.
            data = _guided_payload()
        else:
            data = _read_input_from_stdin() or _build_input_from_args(args)
        mode = (args.mode or data.get("mode", "chat")).lower()
        provider = args.provider or data.get("provider")
        if not provider:
            raise ValueError("provider is required via --provider or JSON.")
        # TaskSpec defaults live in enzu.contract for CLI/API parity.
        task = _resolve_task(data, model_override=args.model)
        context = _resolve_context(data, args.context_file)
        if mode == "rlm" and context is None:
            raise ValueError("context is required for mode=rlm.")
        if mode not in {"chat", "rlm", "automode"}:
            raise ValueError("mode must be 'chat', 'rlm', or 'automode'.")

        def on_engine_progress(event) -> None:
            if not args.progress:
                return
            payload = getattr(event, "message", str(event))
            print(payload, file=sys.stderr)

        def on_rlm_progress(message: str) -> None:
            if not args.progress:
                return
            print(message, file=sys.stderr)

        # Execute TaskSpec directly to preserve budget and success criteria.
        provider_instance = resolve_provider(provider, api_key=args.api_key)
        if mode == "automode":
            fs_root = args.fs_root or data.get("fs_root")
            if not fs_root:
                raise ValueError("fs_root is required for mode=automode.")
            # Automode runs RLM with filesystem tools bound to a single root path.
            fs_helpers = build_fs_helpers(
                fs_root,
                max_entries=args.fs_max_entries,
                max_depth=args.fs_snapshot_depth,
            )
            fs_snapshot = fs_helpers["fs_snapshot"](
                path=".", depth=args.fs_snapshot_depth
            )
            context = json.dumps(
                {"fs_root": fs_helpers["fs_root"](), "fs_snapshot": fs_snapshot},
                ensure_ascii=True,
            )
            task = task.model_copy(
                update={
                    "metadata": {
                        **task.metadata,
                        # Inject tool guidance into the RLM system prompt.
                        "tools_guidance": FS_TOOL_GUIDANCE,
                        "fs_root": fs_helpers["fs_root"](),
                    }
                }
            )
            rlm_engine = (
                RLMEngine(max_steps=args.max_steps) if args.max_steps else RLMEngine()
            )
            rlm_engine.run(
                task,
                provider_instance,
                data=context,
                namespace=fs_helpers,
                on_progress=on_rlm_progress if args.progress else None,
            )
        elif mode == "rlm":
            rlm_engine = (
                RLMEngine(max_steps=args.max_steps) if args.max_steps else RLMEngine()
            )
            # RLMEngine.run requires data to be str, not Optional[str]
            rlm_report = rlm_engine.run(
                task,
                provider_instance,
                data=context or "",
                on_progress=on_rlm_progress if args.progress else None,
            )
            print(json.dumps(rlm_report.model_dump(mode="json"), ensure_ascii=True))
        else:
            chat_engine = Engine()
            chat_report = chat_engine.run(
                task,
                provider_instance,
                on_progress=on_engine_progress if args.progress else None,
            )
            print(json.dumps(chat_report.model_dump(mode="json"), ensure_ascii=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
