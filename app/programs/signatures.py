"""DSPy Signature definitions for the prompt optimizer."""

import dspy


class TaskSignature(dspy.Signature):
    """Perform a task described by the user on the provided input data."""

    task_description: str = dspy.InputField(
        desc="A plain-English description of what the task requires"
    )
    input_data: str = dspy.InputField(
        desc="The raw input the task should be applied to"
    )
    output: str = dspy.OutputField(
        desc="The result of applying the task to the input data"
    )
