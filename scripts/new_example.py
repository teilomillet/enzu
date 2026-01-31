#!/usr/bin/env python3
"""
Generate a new example from template.

Usage:
    python scripts/new_example.py --type minimal --name my_example
    python scripts/new_example.py --type concept --name budget_demo
    python scripts/new_example.py --type production --name my_service
"""

import argparse
import os
from pathlib import Path
from textwrap import dedent

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

TEMPLATES = {
    "minimal": {
        "dir": "basics",
        "files": {
            "{name}.py": dedent('''
                #!/usr/bin/env python3
                """
                {title}

                Run:
                    export OPENAI_API_KEY=sk-...
                    python examples/basics/{name}.py
                """

                from enzu import ask

                result = ask("Your prompt here")
                print(result)
            ''').strip(),
        },
    },
    "concept": {
        "dir": "concepts",
        "files": {
            "{name}.py": dedent('''
                #!/usr/bin/env python3
                """
                {title}

                This example demonstrates:
                - [Key concept 1]
                - [Key concept 2]

                Run:
                    export OPENAI_API_KEY=sk-...
                    python examples/concepts/{name}.py
                """

                import os
                import sys
                from pathlib import Path

                sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

                from dotenv import load_dotenv
                load_dotenv()

                from enzu import Enzu, Outcome

                def main():
                    client = Enzu()

                    # Your demo code here
                    report = client.run(
                        "Your task",
                        tokens=100,
                        return_report=True,
                    )

                    print(f"Outcome: {{report.outcome.value}}")
                    print(f"Tokens: {{report.budget_usage.output_tokens}}")


                if __name__ == "__main__":
                    main()
            ''').strip(),
            "README.md": dedent('''
                # {title}

                ## What It Demonstrates

                - [Key concept 1]
                - [Key concept 2]

                ## How to Run

                ```bash
                export OPENAI_API_KEY=sk-...
                python examples/concepts/{name}.py
                ```

                ## Expected Output

                ```
                Outcome: success
                Tokens: 42
                ```

                ## Key Code

                ```python
                # Highlight the important part
                ```
            ''').strip(),
        },
    },
    "production": {
        "dir": "production/{name}",
        "files": {
            "README.md": dedent('''
                # {title}

                ## Overview

                [What this example does]

                ## Features

                - [Feature 1]
                - [Feature 2]

                ## Quick Start

                ```bash
                export OPENAI_API_KEY=sk-...
                python examples/production/{name}/main.py
                ```

                ## Architecture

                See [architecture.md](architecture.md) for design details.
            ''').strip(),
            "main.py": dedent('''
                #!/usr/bin/env python3
                """
                {title}

                Run:
                    export OPENAI_API_KEY=sk-...
                    python examples/production/{name}/main.py
                """

                import os
                from pathlib import Path

                from enzu import Enzu, Outcome

                # Configuration
                MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                PROVIDER = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "openai"


                def main() -> None:
                    client = Enzu(provider=PROVIDER, model=MODEL)

                    # Your production code here
                    report = client.run(
                        "Your task",
                        tokens=200,
                        return_report=True,
                    )

                    print(f"Outcome: {{report.outcome.value}}")
                    if report.success:
                        print("Task completed successfully")


                if __name__ == "__main__":
                    main()
            ''').strip(),
            "architecture.md": dedent('''
                # {title} Architecture

                ## Overview

                [System description]

                ## Component Diagram

                ```
                ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                │    Input     │───▶│    Enzu      │───▶│    Output    │
                └──────────────┘    └──────────────┘    └──────────────┘
                ```

                ## Data Flow

                1. [Step 1]
                2. [Step 2]
                3. [Step 3]

                ## Budget Enforcement

                | Budget Type | Value | Purpose |
                |------------|-------|---------|
                | tokens | 200 | [Description] |

                ## Key Decisions

                ### [Decision 1]

                **Trade-off**: [What was considered]

                **Chosen**: [What was chosen and why]
            ''').strip(),
            "tests/__init__.py": "# Tests for {name}",
            "tests/test_main.py": dedent('''
                """Tests for {name} example."""

                import pytest


                class Test{class_name}:
                    """Tests for the {name} example."""

                    def test_smoke(self):
                        """Basic smoke test."""
                        # TODO: Add smoke test
                        assert True
            ''').strip(),
        },
    },
}


def to_title(name: str) -> str:
    """Convert snake_case to Title Case."""
    return " ".join(word.capitalize() for word in name.split("_"))


def to_class_name(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def create_example(example_type: str, name: str) -> None:
    """Create a new example from template."""
    if example_type not in TEMPLATES:
        print(f"Unknown type: {example_type}")
        print(f"Available: {', '.join(TEMPLATES.keys())}")
        return

    template = TEMPLATES[example_type]
    target_dir = EXAMPLES_DIR / template["dir"].format(name=name)

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Directory already exists and is not empty: {target_dir}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    title = to_title(name)
    class_name = to_class_name(name)

    for filename_template, content_template in template["files"].items():
        filename = filename_template.format(name=name)
        content = content_template.format(
            name=name,
            title=title,
            class_name=class_name,
        )

        filepath = target_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content + "\n")
        print(f"Created: {filepath.relative_to(EXAMPLES_DIR.parent)}")

    print(f"\nExample created: {target_dir.relative_to(EXAMPLES_DIR.parent)}")
    print(f"\nNext steps:")
    print(f"  1. Edit the files to add your implementation")
    print(f"  2. Run: python {target_dir.relative_to(EXAMPLES_DIR.parent)}/{'main.py' if example_type == 'production' else name + '.py'}")
    print(f"  3. Add tests if not already present")


def main():
    parser = argparse.ArgumentParser(description="Create a new example from template")
    parser.add_argument(
        "--type",
        "-t",
        choices=list(TEMPLATES.keys()),
        required=True,
        help="Example type: minimal, concept, or production",
    )
    parser.add_argument(
        "--name",
        "-n",
        required=True,
        help="Example name (snake_case, e.g., my_budget_demo)",
    )

    args = parser.parse_args()
    create_example(args.type, args.name)


if __name__ == "__main__":
    main()
