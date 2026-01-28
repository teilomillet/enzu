import os

from enzu import Budget, Engine, OpenAICompatProvider, SuccessCriteria, TaskSpec

MODEL = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def build_provider() -> OpenAICompatProvider:
    if OPENROUTER_API_KEY:
        headers = {}
        referer = os.getenv("OPENROUTER_REFERER")
        app_name = os.getenv("OPENROUTER_APP_NAME")
        if referer:
            headers["HTTP-Referer"] = referer
        if app_name:
            headers["X-Title"] = app_name
        return OpenAICompatProvider(
            name="openrouter",
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            headers=headers or None,
        )
    if OPENAI_API_KEY:
        return OpenAICompatProvider(name="openai", api_key=OPENAI_API_KEY)
    raise SystemExit("Set OPENROUTER_API_KEY or OPENAI_API_KEY")


def main() -> None:
    budget = Budget(max_total_tokens=300, max_seconds=20)
    criteria = SuccessCriteria(required_substrings=["Answer:"])
    task = TaskSpec(
        task_id="chat-budget-example",
        input_text="Answer in one sentence. Prefix the response with 'Answer:'",
        model=MODEL,
        budget=budget,
        success_criteria=criteria,
        max_output_tokens=120,
    )

    provider = build_provider()
    engine = Engine()
    report = engine.run(task, provider)

    print(report.output_text or "")


if __name__ == "__main__":
    main()
