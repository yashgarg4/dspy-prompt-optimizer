"""BaseProgram — the core DSPy module used throughout the optimizer."""

import dspy
from app.programs.signatures import TaskSignature


class BaseProgram(dspy.Module):
    """Wraps TaskSignature in a ChainOfThought predictor.

    Returns a dspy.Prediction with an `output` field and intermediate
    reasoning from the chain-of-thought step.
    """

    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(TaskSignature)

    def forward(self, task_description: str, input_data: str) -> dspy.Prediction:
        """Run the program on a single example.

        Args:
            task_description: Plain-English description of what to do.
            input_data: The raw input the task operates on.

        Returns:
            dspy.Prediction containing `output` (and internal `reasoning`).
        """
        return self.predict(task_description=task_description, input_data=input_data)
