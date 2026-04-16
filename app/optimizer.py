"""Optimization loop and metric function — MIPROv2 and SIMBA (Phase 3)."""

import dspy
from app.judge import LLMJudge

_judge: LLMJudge | None = None


def _get_judge() -> LLMJudge:
    """Return a singleton LLMJudge instance (lazy-initialised)."""
    global _judge
    if _judge is None:
        _judge = LLMJudge()
    return _judge


def metric(example: dspy.Example, prediction: dspy.Prediction, _trace=None) -> float:
    """DSPy-compatible metric that scores a prediction with the LLM judge.

    Returns a float in [0, 1] (DSPy convention for optimizers).
    The judge internally scores 0–100; we divide by 100 here.

    Args:
        example: A dspy.Example from the trainset.
        prediction: The program's dspy.Prediction output.
        _trace: Passed by DSPy during optimization; not used in scoring.

    Returns:
        Float in [0.0, 1.0].
    """
    judge = _get_judge()
    ai_output = getattr(prediction, "output", "")
    score, _ = judge.score(
        task_description=example.task_description,
        input_data=example.input_data,
        ai_output=ai_output,
    )
    return score / 100.0
