"""Phase 1 smoke test — runs BaseProgram on 3 hardcoded examples.

Usage:
    python test_phase1.py

Expects a .env file with GEMINI_API_KEY set.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import dspy

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from app.programs.base_program import BaseProgram


EXAMPLES = [
    {
        "task_description": "Generate a concise, compelling email subject line for the email body provided.",
        "input_data": (
            "Hi team, just a reminder that our Q3 planning meeting is scheduled for "
            "Friday at 2pm. Please review the attached agenda and come prepared with "
            "your departmental updates. Lunch will be provided."
        ),
    },
    {
        "task_description": "Summarise this bug report in one sentence suitable for a project manager who is non-technical.",
        "input_data": (
            "Bug ID: #4821\n"
            "Severity: High\n"
            "Description: NullPointerException thrown in UserAuthService.validateToken() "
            "when the JWT expiry claim is missing. Stack trace shows the crash originates "
            "at line 87 of AuthService.java. Reproducible 100% of the time when a token "
            "without an 'exp' field is passed to the /api/refresh endpoint."
        ),
    },
    {
        "task_description": "Explain the following Python code snippet to a complete beginner. Avoid jargon.",
        "input_data": (
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n - 1) + fibonacci(n - 2)"
        ),
    },
]


def main() -> None:
    """Load env, configure DSPy with Gemini, run 3 examples, inspect history."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Copy .env.example to .env and fill in your key."
        )

    print("[Phase 1] Configuring DSPy with Gemini Flash...")
    lm = dspy.LM(
        model="gemini/gemini-2.5-flash-lite",
        api_key=api_key,
        max_tokens=4096,
    )
    dspy.configure(lm=lm)

    program = BaseProgram()

    print("[Phase 1] Running 3 hardcoded examples...\n")
    for i, example in enumerate(EXAMPLES, start=1):
        print(f"{'='*60}")
        print(f"[Phase 1] Example {i}")
        print(f"  Task : {example['task_description']}")
        input_preview = example["input_data"].replace("\n", " ")[:80]
        print(f"  Input: {input_preview}...")

        result = program(
            task_description=example["task_description"],
            input_data=example["input_data"],
        )

        print(f"  Output: {result.output}")
        print()

    print(f"{'='*60}")
    print("[Phase 1] Inspecting DSPy internal prompt history (last 3 calls):")
    dspy.inspect_history(n=3)
    print("[Phase 1] Done.")


if __name__ == "__main__":
    main()
