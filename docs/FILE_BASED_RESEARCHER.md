# File-based multi-turn researcher

This is a file-first researcher flow that preserves multi-turn context on disk.
It uses `Session` so each call can build on previous steps.

## Files

- `research_session.json`: session state persisted across turns.
- `prompt.txt`: the current research question or instruction.
- `context.txt` (optional): supporting data for the current turn.
- `research_output.txt`: the answer for this turn.

## Script

<!-- Source: scripts/file_researcher.py, enzu/session.py -->
Use `scripts/file_researcher.py`, which loads/saves a `Session` per turn.

Example (first turn):

```bash
echo "Summarize the key risks in this report." > prompt.txt
python scripts/file_researcher.py \
  --prompt prompt.txt \
  --model "openrouter/auto"
```

Example (with context file):

```bash
echo "What changed since last week?" > prompt.txt
python scripts/file_researcher.py \
  --prompt prompt.txt \
  --data context.txt
```

## Notes

- Session history is prepended to `data`, so auto mode resolves to RLM once history exists.
- Delete `research_session.json` to reset the researcher.
- Use `--max-steps`, `--tokens`, `--seconds`, or `--cost` to bound each turn.
