"""Dataset loader — reads examples.json and returns dspy.Example objects."""

import json
from pathlib import Path

import dspy

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "test_cases" / "examples.json"


def load_trainset(path: Path = _DEFAULT_PATH) -> list[dspy.Example]:
    """Load test cases from a JSON file and return dspy.Example objects.

    Each Example exposes task_description and input_data as inputs.
    expected_output is attached as a label (available to the metric but
    not passed to the program during optimization).

    Args:
        path: Path to examples.json. Defaults to test_cases/examples.json.

    Returns:
        List of dspy.Example objects with .with_inputs() already applied.
    """
    with open(path, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    examples = []
    for item in raw:
        ex = dspy.Example(
            task_description=item["task_description"],
            input_data=item["input_data"],
            expected_output=item.get("expected_output", ""),
            task_type=item.get("task_type", "unknown"),
        ).with_inputs("task_description", "input_data")
        examples.append(ex)

    return examples
