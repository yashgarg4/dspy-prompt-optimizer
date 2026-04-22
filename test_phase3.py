"""Phase 3: Run MIPROv2 and evaluate improvement (instruction-only + with typed demos).

Usage:
    python test_phase3.py                  # MIPROv2 instruction + MIPROv2+demos
    python test_phase3.py --simba          # also run SIMBA
    python test_phase3.py --reset          # clear saved program + checkpoint, re-optimize

Rate-limit notes:
- Optimization makes ~80-150 LM calls (instruction proposal + 9 trials x 20 examples).
- The compiled program is saved so rerunning skips optimization.
- Evaluation: 40 calls per variant (20 program + 20 judge), checkpoint/resume supported.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import dspy

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from app.judge import LLMJudge
from app.optimizer import (
    extract_instruction,
    inject_typed_demos,
    load_optimized_program,
    run_optimization,
    save_optimized_program,
)
from app.utils.dataset import load_trainset

# constants 
MAIN_MODEL = "gemini/gemini-3.1-flash-lite-preview"
BASELINE_FILE = Path(__file__).parent / "saved_prompts" / "phase2_checkpoint.json"
CHECKPOINT_FILE = Path(__file__).parent / "saved_prompts" / "phase3_checkpoint.json"
SLEEP_BETWEEN = 4


# helpers

def _safe(text: str, max_len: int = 80) -> str:
    return text.replace("\n", " ").encode("ascii", "replace").decode("ascii")[:max_len]


def _load_cp() -> dict:
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cp(data: dict) -> None:
    CHECKPOINT_FILE.parent.mkdir(exist_ok=True)
    CHECKPOINT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _call_with_retry(fn, *args, max_retries=6):
    for attempt in range(max_retries):
        try:
            return fn(*args)
        except Exception as e:
            msg = str(e)
            if "per_day" in msg.lower() or "1011" in msg or "exhausted" in msg.lower():
                print("\n[FATAL] Daily quota exhausted.")
                raise
            if ("429" in msg or "rate" in msg.lower()
                    or "resource_exhausted" in msg.lower()
                    or "503" in msg or "unavailable" in msg.lower()):
                wait = 30 * (2 ** attempt)
                print(f"  [retry {attempt+1}/{max_retries}] Transient error, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded.")


def _load_baseline() -> tuple[float, list[dict]]:
    if not BASELINE_FILE.exists():
        sys.exit("Phase 2 checkpoint not found. Run test_phase2.py first.")
    raw = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    results = list(raw.values())
    avg = sum(r["score"] for r in results) / len(results)
    return avg, results


def _per_type_avg(results: list[dict]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for r in results:
        buckets.setdefault(r["task_type"], []).append(r["score"])
    return {t: sum(v) / len(v) for t, v in buckets.items()}



def evaluate_program(
    program: dspy.Module,
    trainset: list[dspy.Example],
    judge: LLMJudge,
    cp_key: str,
) -> tuple[float, list[dict]]:
    """Evaluate program on all 20 examples with checkpoint/resume."""
    cp = _load_cp()
    results: list[dict] = cp.get(cp_key, [])
    done_ids = {r["id"] for r in results}

    for i, ex in enumerate(trainset):
        ex_id = str(i)
        if ex_id in done_ids:
            r = next(r for r in results if r["id"] == ex_id)
            print(f"  [{i+1:02d}/20] CACHED  score={r['score']:.0f}")
            continue

        def _run():
            return program(task_description=ex.task_description, input_data=ex.input_data)

        pred = _call_with_retry(_run)
        time.sleep(SLEEP_BETWEEN)

        def _judge():
            return judge.score(
                task_description=ex.task_description,
                input_data=ex.input_data,
                ai_output=pred.output,
            )

        score, _ = _call_with_retry(_judge)
        time.sleep(SLEEP_BETWEEN)

        results.append({"id": ex_id, "task_type": ex.task_type, "score": score})
        done_ids.add(ex_id)
        cp[cp_key] = results
        _save_cp(cp)

        print(f"  [{i+1:02d}/20] {ex.task_type:24s} score={score:.0f}  {_safe(pred.output)}")

    avg = sum(r["score"] for r in results) / len(results)
    return avg, results


def _optimize_and_evaluate(
    optimizer_name: str,
    trainset: list[dspy.Example],
    lm: dspy.LM,
    judge: LLMJudge,
    force_reoptimize: bool,
) -> tuple[float, list[dict], float, list[dict], str]:
    """Compile (or load) the program, then evaluate in two passes:
      1. instruction-only (no demos)
      2. instruction + typed demos (one per task type from expected_output)

    Returns (instr_avg, instr_results, demos_avg, demos_results, instruction_text).
    """
    print(f"\n{'='*60}")
    print(f"OPTIMIZER: {optimizer_name.upper()}")
    print(f"{'='*60}")

    # ── Compile or load ────────────────────────────────────────────────────────
    program = None
    if not force_reoptimize:
        program = load_optimized_program(optimizer_name)

    if program is None:
        print(f"Running {optimizer_name.upper()} optimization...")
        for attempt in range(5):
            try:
                program, meta = run_optimization(trainset, optimizer_name=optimizer_name, lm=lm)
                break
            except Exception as e:
                msg = str(e)
                if "503" in msg or "unavailable" in msg.lower() or "service" in msg.lower():
                    wait = 60 * (attempt + 1)
                    print(f"  [retry {attempt+1}/5] Service unavailable, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError("Optimization failed after 5 attempts.")
        save_optimized_program(program, meta, 0.0, optimizer_name)
        print(f"Program saved to saved_prompts/program_{optimizer_name}_latest.json")
    else:
        print(f"Loaded saved {optimizer_name.upper()} program (--reset to re-optimize).")

    instruction = extract_instruction(program)
    print(f"\n[Discovered instruction]\n{instruction}\n")

    # ── Evaluate: instruction only ─────────────────────────────────────────────
    print(f"Evaluating {optimizer_name.upper()} (instruction only)...")
    instr_avg, instr_results = evaluate_program(
        program, trainset, judge,
        cp_key=f"{optimizer_name}_results",
    )

    # ── Evaluate: instruction + typed demos ────────────────────────────────────
    import copy
    program_with_demos = copy.deepcopy(program)
    program_with_demos = inject_typed_demos(program_with_demos, trainset)

    print(f"\nEvaluating {optimizer_name.upper()} + typed demos (1 per task type)...")
    demos_avg, demos_results = evaluate_program(
        program_with_demos, trainset, judge,
        cp_key=f"{optimizer_name}_demos_results",
    )

    save_optimized_program(program_with_demos, {"optimizer": optimizer_name.upper()}, demos_avg, optimizer_name)

    return instr_avg, instr_results, demos_avg, demos_results, instruction


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    run_simba = "--simba" in sys.argv
    force_reopt = "--force-reoptimize" in sys.argv or "--reset" in sys.argv

    if "--reset" in sys.argv:
        for name in ("mipro", "simba"):
            p = Path(__file__).parent / "saved_prompts" / f"program_{name}_latest.json"
            if p.exists():
                p.unlink()
                print(f"Deleted {p.name}")
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            print(f"Deleted {CHECKPOINT_FILE.name}")
        print()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set.")

    lm = dspy.LM(model=MAIN_MODEL, api_key=api_key, max_tokens=4096, max_retries=8)
    dspy.configure(lm=lm)

    trainset = load_trainset()
    judge = LLMJudge()

    # ── Baseline ───────────────────────────────────────────────────────────────
    baseline_avg, baseline_results = _load_baseline()
    baseline_by_type = _per_type_avg(baseline_results)
    print(f"Baseline (Phase 2): {baseline_avg:.1f}/100")
    print(f"Target:             {baseline_avg + 15:.1f}/100 (+15 pts)\n")

    # ── MIPROv2 ────────────────────────────────────────────────────────────────
    (mipro_avg, mipro_results,
     mipro_demos_avg, mipro_demos_results,
     mipro_instruction) = _optimize_and_evaluate(
        "mipro", trainset, lm, judge, force_reopt
    )
    mipro_by_type = _per_type_avg(mipro_results)
    mipro_demos_by_type = _per_type_avg(mipro_demos_results)

    # ── SIMBA (optional) ───────────────────────────────────────────────────────
    simba_demos_avg = None
    simba_demos_by_type = None
    simba_instruction = None
    if run_simba:
        (simba_avg, simba_results,
         simba_demos_avg, simba_demos_results,
         simba_instruction) = _optimize_and_evaluate(
            "simba", trainset, lm, judge, force_reopt
        )
        simba_demos_by_type = _per_type_avg(simba_demos_results)

    # ── Summary table ──────────────────────────────────────────────────────────
    all_types = sorted(baseline_by_type)
    col = 24

    print(f"\n{'='*72}")
    print("RESULTS SUMMARY")
    print(f"{'='*72}")
    header = (f"{'Task type':{col}}  {'Baseline':>8}  {'MIPROv2':>8}"
              f"  {'MIPROv2+demos':>13}")
    if run_simba:
        header += f"  {'SIMBA+demos':>11}"
    print(header)
    print("-" * len(header))

    for t in all_types:
        row = (f"{t:{col}}  {baseline_by_type.get(t, 0):>8.1f}"
               f"  {mipro_by_type.get(t, 0):>8.1f}"
               f"  {mipro_demos_by_type.get(t, 0):>13.1f}")
        if run_simba and simba_demos_by_type:
            row += f"  {simba_demos_by_type.get(t, 0):>11.1f}"
        print(row)

    print("-" * len(header))
    overall = (f"{'OVERALL':{col}}  {baseline_avg:>8.1f}"
               f"  {mipro_avg:>8.1f}"
               f"  {mipro_demos_avg:>13.1f}")
    if run_simba and simba_demos_avg is not None:
        overall += f"  {simba_demos_avg:>11.1f}"
    print(overall)

    delta = (f"{'delta vs baseline':{col}}  {'':>8}"
             f"  {mipro_avg - baseline_avg:>+8.1f}"
             f"  {mipro_demos_avg - baseline_avg:>+13.1f}")
    if run_simba and simba_demos_avg is not None:
        delta += f"  {simba_demos_avg - baseline_avg:>+11.1f}"
    print(delta)

    best_avg = max(mipro_demos_avg, simba_demos_avg or 0)
    print(f"\nTarget: +15 pts -> {baseline_avg + 15:.1f}/100")
    if best_avg >= baseline_avg + 15:
        print("TARGET REACHED!")
    else:
        print(f"Best so far: {best_avg:.1f}/100  (gap: {(baseline_avg + 15) - best_avg:.1f} pts)")

    print(f"\n[MIPROv2 instruction]\n{mipro_instruction}")
    if run_simba and simba_instruction:
        print(f"\n[SIMBA instruction]\n{simba_instruction}")


if __name__ == "__main__":
    main()
