# enzu examples

## Chat task from stdin

Create a task JSON:

```
{
  "task": {
    "task_id": "summarize-01",
    "input_text": "Summarize this text.",
    "model": "openrouter/auto",
    "budget": { "max_output_tokens": 200 },
    "success_criteria": { "required_substrings": ["Summary"] }
  },
  "provider": "openrouter"
}
```

Run it:

```
cat task.json | enzu --provider openrouter
```

## RLM with a context file

Task JSON:

```
{
  "mode": "rlm",
  "task": {
    "task_id": "rlm-01",
    "input_text": "Answer the question using the context.",
    "model": "openrouter/auto",
    "budget": { "max_output_tokens": 400 },
    "success_criteria": { "required_substrings": ["Answer"] }
  },
  "provider": "openrouter"
}
```

Run it with a context file:

```
cat rlm_task.json | enzu --provider openrouter --mode rlm --context-file path/to/context.txt
```

## Override model from the CLI

```
cat task.json | enzu --provider openrouter --model "anthropic/claude-3-5-sonnet"
```
