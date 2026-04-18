"""Optimization loop and metric function — MIPROv2 and SIMBA (Phase 3)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import dspy
from app.judge import LLMJudge
from app.programs.base_program import BaseProgram

SAVED_PROMPTS_DIR = Path(__file__).parent.parent / "saved_prompts"

_judge: LLMJudge | None = None


def _get_judge() -> LLMJudge:
    global _judge
    if _judge is None:
        _judge = LLMJudge()
    return _judge


def metric(example: dspy.Example, prediction: dspy.Prediction, _trace=None) -> float:
    """DSPy-compatible metric: returns float in [0, 1] using the LLM judge."""
    judge = _get_judge()
    ai_output = getattr(prediction, "output", "") or ""
    score, _ = judge.score(
        task_description=example.task_description,
        input_data=example.input_data,
        ai_output=ai_output,
    )
    return score / 100.0


def run_optimization(
    trainset: list[dspy.Example],
    optimizer_name: Literal["mipro", "simba"] = "mipro",
    lm: dspy.LM | None = None,
) -> tuple[dspy.Module, dict]:
    """Run MIPROv2 or SIMBA optimization on a fresh BaseProgram.

    Args:
        trainset: Training examples.
        optimizer_name: "mipro" or "simba".
        lm: LM to use for both task and prompt model (defaults to global DSPy LM).

    Returns:
        (optimized_program, metadata_dict)
    """
    if lm is None:
        lm = dspy.settings.lm

    program = BaseProgram()

    if optimizer_name == "mipro":
        # auto=None so manual params are respected (auto='light' overrides them).
        # max_bootstrapped_demos=0 → skip bootstrapping, which would otherwise
        # collect only email demos (highest-scoring at baseline) and bias every
        # instruction candidate toward email tasks while ignoring bug/code types.
        optimizer = dspy.MIPROv2(
            metric=metric,
            prompt_model=lm,
            task_model=lm,
            auto=None,
            num_candidates=6,
            num_threads=1,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            verbose=True,
        )
        optimized = optimizer.compile(
            program,
            trainset=trainset,
            num_trials=9,
            minibatch=False,
            # data_aware_proposer=False: the cached data summary from the prior
            # run is email-only (DSPy cache returns it instantly). Disabling this
            # makes the proposer use only the program code, which IS task-agnostic,
            # so proposed instructions address all task types rather than just email.
            data_aware_proposer=False,
            requires_permission_to_run=False,
        )
        meta = {
            "optimizer": "MIPROv2",
            "auto": None,
            "max_bootstrapped_demos": 0,
            "max_labeled_demos": 0,
        }

    elif optimizer_name == "simba":
        optimizer = dspy.SIMBA(
            metric=metric,
            bsize=min(len(trainset), 8),
            num_candidates=4,
            max_steps=4,
            max_demos=2,
            num_threads=1,
            prompt_model=lm,
        )
        optimized = optimizer.compile(program, trainset=trainset)
        meta = {
            "optimizer": "SIMBA",
            "bsize": min(len(trainset), 8),
            "num_candidates": 4,
            "max_steps": 4,
            "max_demos": 2,
        }

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}. Choose 'mipro' or 'simba'.")

    return optimized, meta


def extract_instruction(program: dspy.Module) -> str:
    """Extract the optimized system instruction from a compiled program.

    DSPy 3.x ChainOfThought stores the predictor at .predict.predict, so the
    signature path is program.predict.predict.signature.instructions.
    """
    for attr_path in [
        ["predict", "predict", "signature", "instructions"],
        ["predict", "signature", "instructions"],
        ["predict", "extended_signature", "instructions"],
    ]:
        obj = program
        try:
            for attr in attr_path:
                obj = getattr(obj, attr)
            return str(obj)
        except AttributeError:
            continue
    return "(instruction not extractable)"


def save_optimized_program(
    program: dspy.Module,
    metadata: dict,
    score: float,
    optimizer_name: str,
) -> Path:
    """Save the optimized program state and a metadata sidecar to saved_prompts/.

    Returns the path to the metadata JSON file.
    """
    SAVED_PROMPTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    instruction = extract_instruction(program)

    meta_path = SAVED_PROMPTS_DIR / f"optimized_{optimizer_name}_{ts}.json"
    meta_path.write_text(
        json.dumps(
            {
                "timestamp": ts,
                "optimizer": metadata.get("optimizer"),
                "score": score,
                "instruction": instruction,
                "metadata": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    prog_path = SAVED_PROMPTS_DIR / f"program_{optimizer_name}_latest.json"
    program.save(str(prog_path))

    return meta_path


def inject_typed_demos(program: dspy.Module, trainset: list[dspy.Example]) -> dspy.Module:
    """Inject one high-quality expected-output demo per task type into the program.

    Uses expected_output from each example as the demo answer so the model sees
    what quality looks like for each task type before answering new examples.
    """
    seen: set[str] = set()
    demos = []
    for ex in trainset:
        t = getattr(ex, "task_type", "general")
        if t in seen:
            continue
        seen.add(t)
        demos.append(
            dspy.Example(
                task_description=ex.task_description,
                input_data=ex.input_data,
                reasoning="I need to apply the task requirements carefully to produce a high-quality output.",
                output=ex.expected_output,
            ).with_inputs("task_description", "input_data")
        )
    program.predict.predict.demos = demos
    return program


def load_optimized_program(optimizer_name: str) -> dspy.Module | None:
    """Load a previously saved optimized program. Returns None if not found."""
    prog_path = SAVED_PROMPTS_DIR / f"program_{optimizer_name}_latest.json"
    if not prog_path.exists():
        return None
    program = BaseProgram()
    program.load(str(prog_path))
    return program
