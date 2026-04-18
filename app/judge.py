"""LLM-as-judge scorer using a separate Gemini model."""

import os
from dotenv import load_dotenv
import dspy


class JudgeSignature(dspy.Signature):
    """Evaluate an AI response on quality dimensions relevant to the task type.

    Identify the task type from task_description, then apply the matching rubric.
    Start at 100 and deduct as follows:

    EMAIL SUBJECT LINES — deduct for each:
    - Generic/vague subject that doesn't mention the specific topic/name/date (-25)
    - Subject exceeds 65 characters (-15)
    - Passive or weak phrasing that doesn't convey urgency or value (-10)
    - Missing the single most important piece of information from the email (-15)

    BUG REPORT SUMMARIES — deduct for each:
    - Contains technical jargon a non-technical manager wouldn't know
      (e.g. NullPointerException, JWT, SQL, API, race condition, stack trace,
      heap, buffer, CSS, endpoint, middleware, concurrency, isolate): -20 per term, max -40
    - More than one sentence (-25)
    - Does not state the user-facing impact (what users cannot do or experience): -20
    - Uses severity labels copied from the report (High, Critical, Medium) verbatim: -10

    CODE EXPLANATIONS — deduct for each:
    - Uses technical terms a beginner wouldn't know
      (e.g. function, recursion, algorithm, variable, parameter, method, iterate,
      index, syntax, base case, class, instance, object): -15 per term, max -45
    - No real-world analogy or comparison to everyday experience: -20
    - Explanation is factually inaccurate: -40
    - Longer than 200 words (overly verbose for a beginner): -10

    Do NOT reward an output simply for being long or detailed.
    Do NOT give benefit of the doubt on jargon — if a term requires programming
    knowledge to understand, it counts.
    """

    task_description: str = dspy.InputField(
        desc="The task the AI was asked to perform"
    )
    input_data: str = dspy.InputField(
        desc="The raw input the AI received"
    )
    ai_output: str = dspy.InputField(
        desc="The AI's response to evaluate"
    )
    score: int = dspy.OutputField(
        desc=(
            "Integer score 0-100. Start at 100 and deduct per the rubric above. "
            "Be strict: a bug summary with even one jargon term scores ≤80; "
            "a code explanation with no analogy scores ≤80."
        )
    )
    reasoning: str = dspy.OutputField(
        desc=(
            "Identify the task type, then list each quality dimension checked, "
            "state PASS or FAIL with the deduction amount, and give the running total. "
            "Example: 'Task: bug summary. Jargon found: JWT(-20), NullPointerException(-20). "
            "One sentence: PASS. User impact stated: PASS. Total deductions: -40. Score: 60.'"
        )
    )


class LLMJudge:
    """Scores AI outputs using a separate judge LM.

    Uses a dedicated Gemini model (different from the main program LM)
    isolated via dspy.context so judge calls never pollute the training LM.
    """

    def __init__(
        self,
        judge_model: str = "gemini/gemini-3.1-flash-lite-preview",
        api_key: str | None = None,
    ) -> None:
        load_dotenv()
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in environment.")
        self._lm = dspy.LM(model=judge_model, api_key=api_key, max_tokens=1024)
        self._predict = dspy.Predict(JudgeSignature)

    def score(
        self,
        task_description: str,
        input_data: str,
        ai_output: str,
    ) -> tuple[float, str]:
        """Score a single AI output.

        Args:
            task_description: The task the model was asked to perform.
            input_data: The raw input the model received.
            ai_output: The model's response to evaluate.

        Returns:
            A tuple of (score, reasoning) where score is a float in [0, 100].
        """
        with dspy.context(lm=self._lm):
            result = self._predict(
                task_description=task_description,
                input_data=input_data,
                ai_output=ai_output,
            )

        try:
            raw = result.score
            score = float(int(str(raw).strip()))
        except (ValueError, TypeError):
            score = 0.0

        score = max(0.0, min(100.0, score))
        return score, result.reasoning
