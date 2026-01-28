#!/usr/bin/env python3
"""
Example: Research agent using RLM + Exa search.

This demonstrates the actu.me pipeline:
1. Start with a seed topic
2. RLM uses research() to gather context automatically
3. RLM synthesizes findings into newsletter-ready content

Requirements:
    export EXA_API_KEY="your-key"
    export OPENROUTER_API_KEY="your-key"  # or OPENAI_API_KEY
"""
import os
import sys

from dotenv import load_dotenv
from enzu import run

load_dotenv()


def research_topic(topic: str, model: str = "openai/gpt-4o-mini") -> str:
    """
    Research a topic and generate newsletter content.

    The RLM will use the research() helper which:
    1. Searches for relevant articles
    2. Filters by quality score
    3. Formats context for synthesis
    """
    task = """
Research the topic and create newsletter-ready content.

Use the research() function to gather sources, then synthesize.

Example workflow:
```python
result = research(data, include_news=True, days_back=7)
print(result["stats"])

newsletter = llm_query(f'''
Write a newsletter section about this topic.
Include key insights, trends, and notable quotes.

Sources:
{result['context']}
''')

FINAL(newsletter + "\\n\\n" + format_sources(result['sources']))
```
"""

    result = run(
        task,
        model=model,
        provider="openrouter",
        data=topic,
        tokens=4000,
    )

    assert isinstance(result, str)
    return result


def research_with_papers(topic: str, model: str = "openai/gpt-4o-mini") -> str:
    """
    Research a technical topic including academic papers.
    """
    task = """
Research this technical topic, including academic papers.

Steps:
1. Use research() with include_papers=True
2. Synthesize into a technical summary
3. Include citations

```python
result = research(data, include_news=True, include_papers=True)
print(result["stats"])

summary = llm_query(f'''
Write a technical summary of recent developments.
Focus on key findings and methodologies.

Sources:
{result['context']}
''')

FINAL(summary + "\\n\\n" + format_sources(result['sources']))
```
"""

    result = run(
        task,
        model=model,
        provider="openrouter",
        data=topic,
        tokens=4000,
    )

    assert isinstance(result, str)
    return result


def explore_from_url(seed_url: str, model: str = "openai/gpt-4o-mini") -> str:
    """
    Start from a URL and explore related content.
    """
    task = """
Explore related content from this seed URL and synthesize.

Steps:
1. Use explore() to find similar content
2. Synthesize a broader perspective

```python
result = explore(data)
print(result["stats"])

if result['stats']['kept'] > 0:
    summary = llm_query(f'''
    Synthesize insights from related articles.
    Note common themes and unique perspectives.

    Sources:
    {result['context']}
    ''')
    FINAL(summary + "\\n\\n" + format_sources(result['sources']))
else:
    FINAL("No related content found for this URL.")
```
"""

    result = run(
        task,
        model=model,
        provider="openrouter",
        data=seed_url,
        tokens=4000,
    )

    assert isinstance(result, str)
    return result


if __name__ == "__main__":
    # Check for required API keys
    if not os.environ.get("EXA_API_KEY"):
        print("Error: EXA_API_KEY not set")
        print("Get your key at https://exa.ai")
        sys.exit(1)

    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("Error: Need OPENROUTER_API_KEY or OPENAI_API_KEY")
        sys.exit(1)

    # Example usage
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python research_example.py 'topic to research'")
        print("  python research_example.py --papers 'technical topic'")
        print("  python research_example.py --explore 'https://url-to-explore'")
        sys.exit(0)

    if sys.argv[1] == "--papers":
        topic = sys.argv[2] if len(sys.argv) > 2 else "transformer architecture attention"
        print(f"Researching (with papers): {topic}")
        print("-" * 50)
        result = research_with_papers(topic)
    elif sys.argv[1] == "--explore":
        url = sys.argv[2] if len(sys.argv) > 2 else "https://arxiv.org/abs/2301.00000"
        print(f"Exploring from: {url}")
        print("-" * 50)
        result = explore_from_url(url)
    else:
        topic = " ".join(sys.argv[1:])
        print(f"Researching: {topic}")
        print("-" * 50)
        result = research_topic(topic)

    print(result)
