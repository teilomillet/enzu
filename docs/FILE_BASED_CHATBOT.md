# File-based chatbot

This is a file-first chatbot flow that does not require a running server.
Each turn reads a text file, runs `enzu`, and writes the response to disk.

## Files

- `chat_state.json`: conversation state (appended each turn).
- `user.txt`: the latest user message.
- `chat_output.txt`: the assistant reply for this turn.

## Script

<!-- Source: scripts/file_chatbot.py, enzu/api.py -->
Use `scripts/file_chatbot.py`, which keeps history in a JSON file and embeds it
into the prompt. It forces `mode="chat"` for single-shot generation.

Example:

```bash
echo "Hello there." > user.txt
python scripts/file_chatbot.py \
  --input user.txt \
  --model "openrouter/auto"

cat chat_output.txt
```

## State format

`chat_state.json` is a JSON object with a `history` list:

```json
{
  "history": [
    { "user": "Hello there.", "assistant": "Hi! How can I help?" }
  ]
}
```

## Notes

- History is embedded in the prompt (no `data`), so `mode="chat"` stays in chat mode.
- Use `--max-history-chars` to cap how much history is included each turn.
- Use `--tokens` to cap output size per turn.
