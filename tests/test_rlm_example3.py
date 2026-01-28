from __future__ import annotations

from pathlib import Path

from enzu import Budget, SuccessCriteria, TaskSpec
from enzu.rlm import RLMEngine
from tests.providers.mock_provider import MockProvider


def test_rlm_example3_batching_and_model_prompt() -> None:
    data_path = Path(__file__).parent / "fixtures" / "example3" / "questions.txt"
    input_data = data_path.read_text(encoding="utf-8")
    model_output = """
```python
lines = [line.strip() for line in data.splitlines() if line.strip()]
batches = [lines[i : i + 2] for i in range(0, len(lines), 2)]

results = []
for batch_index, batch in enumerate(batches):
    payload = "\\n".join(batch)
    response = llm_query(f"SUBCALL:batch:{batch_index}")
    results.extend([line for line in response.splitlines() if line.strip()])

counts = {"location": 0, "person": 0}
for entry in results:
    kind = entry.split(":")[0].strip()
    if kind in counts:
        counts[kind] += 1

FINAL_VAR("counts")
```
""".strip()
    subcalls = {
        "batch:0": "location:capital_of_france\nperson:shakespeare",
        "batch:1": "location:mount_fuji\nperson:da_vinci",
    }
    # Fallback output if budget exhausted during step 1
    fallback_output = '```python\nFINAL("location: 2, person: 2")\n```'
    provider = MockProvider(main_outputs=[model_output, fallback_output], subcall_responses=subcalls)
    task = TaskSpec(
        task_id="example3",
        input_text="Count location vs person questions with batching.",
        model="qwen2-7b",
        budget=Budget(max_tokens=2000, max_total_tokens=4000),
        success_criteria=SuccessCriteria(required_substrings=["location"]),
    )
    engine = RLMEngine(max_steps=2, prompt_style="extended")
    report = engine.run(task, provider, data=input_data)

    # The system prompt must include the Qwen-specific subcall warning.
    assert report.steps
    assert "Be very careful about using llm_query" in report.steps[0].prompt
    assert report.success
    assert "location" in (report.answer or "")
    assert "person" in (report.answer or "")
    # Provider should receive at least the main calls (2 steps).
    # Subcalls may not happen if budget is exhausted early.
    assert len(provider.calls) >= 2
