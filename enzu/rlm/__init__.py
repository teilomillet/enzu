"""
RLM (Recursive Language Model) module.

Public API:
- RLMEngine: main execution engine
- verify_output: shared verification logic

Internal modules (use directly for advanced customization):
- budget: TokenBudgetPool, BudgetTracker
- feedback: build_feedback, extract_code
- llm_executor: LLMExecutor
- prompts: system prompt constants
- runner: StepRunner
- sandbox_factory: create_sandbox
"""
from enzu.rlm.engine import RLMEngine
from enzu.rlm.verification import verify_output
from enzu.rlm.feedback import (
    build_feedback,
    extract_code,
    format_feedback,
)
from enzu.rlm.prompts import (
    SYSTEM_PROMPT_GUARDRAILS,
    SEARCH_TOOLS_GUIDANCE,
    PIP_INSTALL_GUIDANCE,
    STRATEGY_HINTS,
)

__all__ = [
    # Core
    "RLMEngine",
    "verify_output",
    # Feedback
    "build_feedback",
    "extract_code",
    "format_feedback",
    # Prompts
    "SYSTEM_PROMPT_GUARDRAILS",
    "SEARCH_TOOLS_GUIDANCE",
    "PIP_INSTALL_GUIDANCE",
    "STRATEGY_HINTS",
]
