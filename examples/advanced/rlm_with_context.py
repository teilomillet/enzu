import os

from enzu import Budget, OpenAICompatProvider, SuccessCriteria, TaskSpec
from enzu.rlm import RLMEngine

MODEL = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def build_provider() -> OpenAICompatProvider:
    if OPENROUTER_API_KEY:
        return OpenAICompatProvider(
            name="openrouter",
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    if OPENAI_API_KEY:
        return OpenAICompatProvider(name="openai", api_key=OPENAI_API_KEY)
    raise SystemExit("Set OPENROUTER_API_KEY or OPENAI_API_KEY")


def main() -> None:
    context = """
    Document A: Ada Lovelace wrote early notes on the Analytical Engine.
    Document B: The first algorithm is credited to Ada Lovelace.
    Document C: Charles Babbage designed the Analytical Engine.
    """.strip()

    budget = Budget(max_total_tokens=500, max_seconds=30)
    criteria = SuccessCriteria(required_substrings=["Ada Lovelace"])
    task = TaskSpec(
        task_id="rlm-context-example",
        input_text="Who is credited with the first algorithm? Answer briefly.",
        model=MODEL,
        budget=budget,
        success_criteria=criteria,
        max_output_tokens=120,
    )

    provider = build_provider()
    engine = RLMEngine(max_steps=4)
    report = engine.run(task, provider, data=context)

    print(report.answer or report.output_text or "")


if __name__ == "__main__":
    main()
