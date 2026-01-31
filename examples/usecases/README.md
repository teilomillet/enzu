# Use Cases

Practical examples showing enzu's value for **everyday tasks** — with budget control.

## The Pattern

enzu shines when you need:
- **Quick, structured outputs** from LLMs
- **Budget control** to avoid runaway costs
- **Simple integration** without complex setup

## Examples

### Email Writer
[`email_writer.py`](email_writer.py) — Generate professional emails with tone control

```python
result = client.run(
    "Write a polite follow-up email about the project deadline",
    tokens=200,
)
```
**Who needs this**: Anyone who writes emails daily

---

### Code Reviewer
[`code_reviewer.py`](code_reviewer.py) — Get quick feedback on code snippets

```python
result = client.run(
    f"Review this function for bugs and improvements:\n{code}",
    tokens=300,
)
```
**Who needs this**: Developers wanting a second opinion

---

### Text Summarizer
[`text_summarizer.py`](text_summarizer.py) — Condense articles or documents

```python
result = client.run(
    f"Summarize in 3 bullet points:\n{article}",
    tokens=150,
)
```
**Who needs this**: Anyone drowning in content to read

---

### Data Extractor
[`data_extractor.py`](data_extractor.py) — Pull structured data from unstructured text

```python
result = client.run(
    f"Extract name, email, and company from:\n{text}",
    tokens=100,
)
```
**Who needs this**: Anyone processing forms, emails, or documents

---

## Why enzu?

| Without enzu | With enzu |
|--------------|-----------|
| No spending control | Set token/cost limits |
| Surprise bills | Predictable costs |
| Raw API complexity | Simple interface |

## Cost

Each example includes cost tracking. Typical costs:
- Email generation: ~$0.001-0.003
- Code review: ~$0.002-0.005
- Summarization: ~$0.001-0.002
- Data extraction: ~$0.001-0.002

## For Large Content

Need to process **large documents** that exceed context limits? See:
- [`../production/`](../production/) — Full production pipelines with RLM
