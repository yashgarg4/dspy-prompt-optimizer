"""Phase 2 smoke test — baseline evaluation across all 20 examples.

Usage:
    python test_phase2.py

Prints per-example scores and reasoning, then the overall baseline mean.
Checkpoints results to saved_prompts/phase2_checkpoint.json so a run that
is interrupted by quota limits can be resumed without re-spending API calls.
Expects a .env file with GEMINI_API_KEY set.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
import dspy

sys.path.insert(0, str(Path(__file__).parent))

from app.programs.base_program import BaseProgram
from app.judge import LLMJudge
from app.utils.dataset import load_trainset

MAIN_MODEL = "gemini/gemini-3.1-flash-lite-preview"
CHECKPOINT_FILE = Path(__file__).parent / "saved_prompts" / "phase2_checkpoint.json"

_RETRY_DELAY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)")
_MAX_WAIT = 120  # never sleep longer than 2 minutes per retry


def _call_with_retry(fn, *args, max_retries: int = 5, **kwargs):
    """Call fn(*args, **kwargs), retrying on transient 429 rate-limit errors.

    Reads the suggested retry delay from the error message and sleeps that
    long (capped at _MAX_WAIT) before each retry.  A 429 that carries a
    per-day quota violation is NOT retried — we raise immediately so the
    caller knows it needs to resume later.

    Args:
        fn: Callable to invoke.
        max_retries: Maximum number of retry attempts.

    Returns:
        The return value of fn on success.

    Raises:
        Exception: The last exception if all retries are exhausted, or a
            per-day quota error on the first attempt.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc)
            if "429" not in msg:
                raise
            # Per-day exhaustion — no point retrying, surface immediately.
            if "PerDay" in msg or "per_day" in msg.lower():
                raise
            if attempt == max_retries:
                raise
            match = _RETRY_DELAY_RE.search(msg)
            wait = min(float(match.group(1)) + 2, _MAX_WAIT) if match else _MAX_WAIT
            print(f"  [rate-limit] Sleeping {wait:.0f}s before retry {attempt + 1}...")
            time.sleep(wait)


def _load_checkpoint() -> dict[int, dict]:
    """Load previously saved results from the checkpoint file.

    Returns:
        Dict mapping example index (1-based) to its saved result dict.
    """
    if not CHECKPOINT_FILE.exists():
        return {}
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _save_checkpoint(results: dict[int, dict]) -> None:
    """Persist current results to the checkpoint file."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)


def _safe(text: str, max_len: int) -> str:
    """Return a single-line ASCII-safe preview of text."""
    return text.replace("\n", " ")[:max_len].encode("ascii", "replace").decode("ascii")


def main() -> None:
    """Run (or resume) baseline evaluation and print mean score + per-example reasoning."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Copy .env.example to .env and fill in your key."
        )

    print(f"[Phase 2] Configuring DSPy with {MAIN_MODEL}...")
    lm = dspy.LM(
        model=MAIN_MODEL,
        api_key=api_key,
        max_tokens=4096,
        num_retries=2,
    )
    dspy.configure(lm=lm)

    print("[Phase 2] Loading dataset...")
    examples = load_trainset()
    print(f"[Phase 2] Loaded {len(examples)} examples")

    checkpoint = _load_checkpoint()
    if checkpoint:
        print(f"[Phase 2] Resuming from checkpoint — {len(checkpoint)} examples already done\n")
    else:
        print()

    program = BaseProgram()
    judge = LLMJudge()

    results: dict[int, dict] = dict(checkpoint)

    for i, example in enumerate(examples, start=1):
        if i in results:
            # Replay cached result without making any API call.
            r = results[i]
            print(f"{'='*65}")
            print(f"[Phase 2] Example {i:2d} [cached]  |  Score: {r['score']:5.1f}/100")
            print(f"  Task    : {_safe(example.task_description, 70)}")
            print(f"  Output  : {_safe(r['output'], 120)}...")
            print(f"  Reasoning: {_safe(r['reasoning'], 160)}...")
            continue

        task_type = getattr(example, "task_type", "unknown")

        try:
            prediction = _call_with_retry(
                program,
                task_description=example.task_description,
                input_data=example.input_data,
            )
            score, reasoning = _call_with_retry(
                judge.score,
                task_description=example.task_description,
                input_data=example.input_data,
                ai_output=prediction.output,
            )
        except Exception as exc:
            _save_checkpoint(results)
            remaining = len(examples) - len(results)
            print(
                f"\n[Phase 2] Quota exhausted at example {i}. "
                f"Checkpoint saved ({len(results)} done, {remaining} remaining).\n"
                f"Wait for your daily quota to reset (midnight UTC) then re-run — "
                f"completed examples will be skipped automatically.\n"
                f"Error: {str(exc)[:200]}"
            )
            sys.exit(1)

        results[i] = {
            "score": score,
            "reasoning": reasoning,
            "output": prediction.output,
            "task_type": task_type,
        }
        _save_checkpoint(results)

        print(f"{'='*65}")
        print(f"[Phase 2] Example {i:2d} [{task_type}]  |  Score: {score:5.1f}/100")
        print(f"  Task    : {_safe(example.task_description, 70)}")
        print(f"  Output  : {_safe(prediction.output, 120)}...")
        print(f"  Reasoning: {_safe(reasoning, 160)}...")

        if i < len(examples):
            time.sleep(4)

    # --- Summary ---
    all_scores = [r["score"] for r in results.values()]
    scores_by_type: dict[str, list[float]] = defaultdict(list)
    for r in results.values():
        scores_by_type[r["task_type"]].append(r["score"])

    mean_score = sum(all_scores) / len(all_scores)

    print(f"\n{'='*65}")
    print("[Phase 2] BASELINE RESULTS")
    print(f"  Examples scored : {len(all_scores)}/{len(examples)}")
    print(f"  Overall mean    : {mean_score:.1f}/100")
    print(f"  Min / Max       : {min(all_scores):.1f} / {max(all_scores):.1f}")
    print()
    for task_type, type_scores in scores_by_type.items():
        print(f"  [{task_type}]  mean: {sum(type_scores)/len(type_scores):.1f}/100  (n={len(type_scores)})")
    print()
    print("[Phase 2] Done. Baseline captured — ready for Phase 3 optimization.")


if __name__ == "__main__":
    main()
