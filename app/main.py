"""Streamlit UI — DSPy Prompt Optimizer (Phase 4)."""

import copy
import json
import os
import sys
import time
from pathlib import Path

import dspy
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.judge import LLMJudge
from app.optimizer import (
    extract_instruction,
    inject_typed_demos,
    load_optimized_program,
    run_optimization,
    save_optimized_program,
)
from app.programs.base_program import BaseProgram
from app.utils.dataset import load_trainset
from app.utils import history as run_history

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_DEFAULT_EXAMPLES = _ROOT / "test_cases" / "examples.json"
_PHASE2_CHECKPOINT = _ROOT / "saved_prompts" / "phase2_checkpoint.json"

_MODELS = [
    "gemini/gemini-3.1-flash-lite-preview",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.0-flash-lite",
]
_SLEEP = 4  # seconds between API calls (rate-limit buffer)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DSPy Prompt Optimizer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("DSPy Prompt Optimizer")
st.caption(
    "Automatically improves AI prompts using DSPy's MIPROv2 optimizer. "
    "Configure your task, run the optimizer, and compare before vs. after."
)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "api_key": os.getenv("GEMINI_API_KEY", ""),
    "model": _MODELS[0],
    "test_cases": [],
    # baseline
    "baseline_avg": None,
    "baseline_by_type": {},
    "baseline_results": [],
    # MIPROv2 instruction-only
    "opt_instruction": "",
    "opt_avg": None,
    "opt_by_type": {},
    "opt_results": [],
    # MIPROv2 + typed demos
    "demos_avg": None,
    "demos_by_type": {},
    "demos_results": [],
    # status
    "run_done": False,
    "last_error": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

ss = st.session_state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(text: str, max_len: int = 80) -> str:
    return str(text).replace("\n", " ").encode("ascii", "replace").decode("ascii")[:max_len]


def _call_with_retry(fn, *args, max_retries=6):
    for attempt in range(max_retries):
        try:
            return fn(*args)
        except Exception as e:
            msg = str(e)
            if "per_day" in msg.lower() or "exhausted" in msg.lower():
                raise RuntimeError("Daily API quota exhausted.") from e
            if ("429" in msg or "rate" in msg.lower()
                    or "resource_exhausted" in msg.lower()
                    or "503" in msg or "unavailable" in msg.lower()):
                wait = 30 * (2 ** attempt)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded.")


def _per_type_avg(results: list[dict]) -> dict[str, float]:
    buckets: dict[str, list] = {}
    for r in results:
        buckets.setdefault(r.get("task_type", "general"), []).append(r["score"])
    return {t: sum(v) / len(v) for t, v in buckets.items()}


def _make_trainset(test_cases: list[dict]) -> list[dspy.Example]:
    examples = []
    for item in test_cases:
        ex = dspy.Example(
            task_description=item.get("task_description", ""),
            input_data=item.get("input_data", ""),
            expected_output=item.get("expected_output", ""),
            task_type=item.get("task_type", "general"),
        ).with_inputs("task_description", "input_data")
        examples.append(ex)
    return examples


def _evaluate_program(
    program: dspy.Module,
    trainset: list[dspy.Example],
    judge: LLMJudge,
    log_fn,
) -> tuple[float, list[dict]]:
    """Run program + judge on every example. log_fn receives per-example status strings."""
    results = []
    for i, ex in enumerate(trainset):
        def _run(ex=ex):
            return program(task_description=ex.task_description, input_data=ex.input_data)

        pred = _call_with_retry(_run)
        time.sleep(_SLEEP)

        def _judge(ex=ex, pred=pred):
            return judge.score(
                task_description=ex.task_description,
                input_data=ex.input_data,
                ai_output=pred.output,
            )

        score, _ = _call_with_retry(_judge)
        time.sleep(_SLEEP)

        results.append({
            "id": str(i),
            "task_type": getattr(ex, "task_type", "general"),
            "score": score,
            "output": pred.output,
        })
        log_fn(f"  [{i+1:02d}/{len(trainset):02d}] {ex.task_type:24s}  score={score:.0f}  "
               f"{_safe(pred.output, 60)}")

    avg = sum(r["score"] for r in results) / len(results)
    return avg, results


def _load_baseline_from_checkpoint() -> tuple[float, list[dict]] | None:
    """Try to load Phase 2 baseline from disk. Returns None if not available."""
    if not _PHASE2_CHECKPOINT.exists():
        return None
    raw = json.loads(_PHASE2_CHECKPOINT.read_text(encoding="utf-8"))
    results = [
        {"id": k, "task_type": v["task_type"], "score": v["score"], "output": v.get("output", "")}
        for k, v in raw.items()
    ]
    avg = sum(r["score"] for r in results) / len(results)
    return avg, results


def _run_full_pipeline(status_container):
    """Run baseline → optimize → evaluate → evaluate+demos. Writes to status_container."""
    api_key = ss.api_key
    model = ss.model
    test_cases = ss.test_cases

    # Set env var so LLMJudge picks it up
    os.environ["GEMINI_API_KEY"] = api_key

    lm = dspy.LM(model=model, api_key=api_key, max_tokens=4096, max_retries=8)
    dspy.configure(lm=lm)
    judge = LLMJudge(api_key=api_key)
    trainset = _make_trainset(test_cases)

    logs = []

    def log(msg: str):
        logs.append(msg)
        status_container.text("\n".join(logs[-30:]))

    # ── Baseline ──────────────────────────────────────────────────────────────
    cached = _load_baseline_from_checkpoint()
    if cached and len(cached[1]) == len(test_cases):
        baseline_avg, baseline_results = cached
        log(f"Loaded pre-computed baseline from checkpoint: {baseline_avg:.1f}/100")
    else:
        log("Step 1/4: Evaluating baseline (unoptimized program)...")
        baseline_program = BaseProgram()
        baseline_avg, baseline_results = _evaluate_program(
            baseline_program, trainset, judge, log
        )
        log(f"Baseline complete: {baseline_avg:.1f}/100")

    ss.baseline_avg = baseline_avg
    ss.baseline_results = baseline_results
    ss.baseline_by_type = _per_type_avg(baseline_results)

    # ── MIPROv2 optimization ──────────────────────────────────────────────────
    log("\nStep 2/4: Running MIPROv2 optimization (this takes several minutes)...")
    program = load_optimized_program("mipro")
    if program is not None:
        log("Loaded existing optimized program from disk.")
    else:
        log("Compiling with MIPROv2 (auto=None, instruction-only, no demos bias)...")
        for attempt in range(5):
            try:
                program, meta = run_optimization(trainset, optimizer_name="mipro", lm=lm)
                break
            except Exception as e:
                msg = str(e)
                if "503" in msg or "unavailable" in msg.lower():
                    wait = 60 * (attempt + 1)
                    log(f"  Service busy, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError("MIPROv2 compilation failed after 5 retries.")
        save_optimized_program(program, {"optimizer": "MIPROv2"}, 0.0, "mipro")
        log("Optimization complete. Program saved.")

    instruction = extract_instruction(program)
    ss.opt_instruction = instruction
    log(f"\nDiscovered instruction:\n{instruction}\n")

    # ── Evaluate optimized (instruction-only) ─────────────────────────────────
    log("Step 3/4: Evaluating optimized program (instruction only)...")
    opt_avg, opt_results = _evaluate_program(program, trainset, judge, log)
    log(f"MIPROv2 instruction-only score: {opt_avg:.1f}/100")
    ss.opt_avg = opt_avg
    ss.opt_results = opt_results
    ss.opt_by_type = _per_type_avg(opt_results)

    # ── Evaluate optimized + typed demos ─────────────────────────────────────
    log("\nStep 4/4: Evaluating MIPROv2 + typed demos (1 reference example per task type)...")
    prog_with_demos = inject_typed_demos(copy.deepcopy(program), trainset)
    demos_avg, demos_results = _evaluate_program(prog_with_demos, trainset, judge, log)
    log(f"MIPROv2 + demos score: {demos_avg:.1f}/100")
    ss.demos_avg = demos_avg
    ss.demos_results = demos_results
    ss.demos_by_type = _per_type_avg(demos_results)

    save_optimized_program(prog_with_demos, {"optimizer": "MIPROv2+demos"}, demos_avg, "mipro")

    ss.run_done = True
    log(f"\nDone! Improvement: {demos_avg - baseline_avg:+.1f} pts "
        f"({baseline_avg:.1f} -> {demos_avg:.1f}/100)")

    run_history.save_run(
        model=model,
        n_examples=len(test_cases),
        baseline_avg=baseline_avg,
        opt_avg=opt_avg,
        demos_avg=demos_avg,
        instruction=instruction,
        task_types=[tc.get("task_type", "general") for tc in test_cases],
        baseline_by_type=ss.baseline_by_type,
        demos_by_type=ss.demos_by_type,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Configure
# ══════════════════════════════════════════════════════════════════════════════
tab_cfg, tab_opt, tab_cmp, tab_hist = st.tabs(["Configure", "Optimize", "Compare", "History"])

with tab_cfg:
    st.header("Configure")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("API Settings")
        api_key_input = st.text_input(
            "Gemini API Key",
            value=ss.api_key,
            type="password",
            help="Paste your Google AI Studio key (starts with AIza...)",
        )
        if api_key_input:
            ss.api_key = api_key_input

        model_input = st.selectbox("Model", _MODELS, index=_MODELS.index(ss.model))
        ss.model = model_input

        if ss.api_key:
            st.success("API key set.")
        else:
            st.warning("Enter your Gemini API key to continue.")

    with col_right:
        st.subheader("Test Cases")

        load_col, _ = st.columns([1, 2])
        with load_col:
            if st.button("Load built-in examples (20 cases)", use_container_width=True):
                ss.test_cases = json.loads(_DEFAULT_EXAMPLES.read_text(encoding="utf-8"))
                st.success(f"Loaded {len(ss.test_cases)} examples.")

        st.markdown("**Or paste custom JSON:**")
        custom_json = st.text_area(
            "Paste a JSON array of test cases",
            height=200,
            placeholder='[\n  {\n    "task_type": "my_task",\n'
                        '    "task_description": "Do something.",\n'
                        '    "input_data": "Some input text here.",\n'
                        '    "expected_output": "The ideal output."\n  }\n]',
            label_visibility="collapsed",
        )
        if st.button("Load from JSON above"):
            try:
                parsed = json.loads(custom_json)
                if not isinstance(parsed, list) or not parsed:
                    st.error("Must be a non-empty JSON array.")
                else:
                    ss.test_cases = parsed
                    st.success(f"Loaded {len(parsed)} examples.")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    # Preview
    if ss.test_cases:
        st.divider()
        st.subheader(f"Test Cases Preview ({len(ss.test_cases)} loaded)")
        types = {}
        for tc in ss.test_cases:
            types[tc.get("task_type", "general")] = types.get(tc.get("task_type", "general"), 0) + 1
        st.markdown("  ".join(f"**{t}**: {n}" for t, n in types.items()))
        with st.expander("Show first 3 examples"):
            for tc in ss.test_cases[:3]:
                st.markdown(f"**Task:** {tc.get('task_description', '')}")
                st.text(f"Input: {_safe(tc.get('input_data', ''), 120)}")
                st.text(f"Expected: {_safe(tc.get('expected_output', ''), 120)}")
                st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Optimize
# ══════════════════════════════════════════════════════════════════════════════
with tab_opt:
    st.header("Optimize")

    # Config summary
    cfg_ok = bool(ss.api_key and ss.test_cases)
    if not cfg_ok:
        st.warning("Complete the Configure tab first (API key + test cases).")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Model", ss.model.split("/")[-1])
        col2.metric("Test cases", len(ss.test_cases))
        col3.metric("API key", "set" if ss.api_key else "missing")

        st.divider()

        if ss.run_done:
            st.success(
                f"Last run complete.  "
                f"Baseline: **{ss.baseline_avg:.1f}**  |  "
                f"MIPROv2+demos: **{ss.demos_avg:.1f}**  "
                f"(delta: **{ss.demos_avg - ss.baseline_avg:+.1f}**)"
            )
            st.info("Switch to the **Compare** tab to see full results. "
                    "Click **Run optimization** again to re-run from scratch.")

        estimated_mins = max(15, len(ss.test_cases) * 2)
        st.markdown(
            f"**What this does:** Baseline evaluation ({len(ss.test_cases)} examples) "
            f"→ MIPROv2 optimization (~80-150 LM calls) "
            f"→ two evaluation passes. "
            f"Estimated time: **{estimated_mins}+ minutes** on free-tier Gemini."
        )
        st.markdown(
            "The optimized program is saved to disk — rerunning skips the optimization "
            "step unless you delete `saved_prompts/program_mipro_latest.json`."
        )

        run_btn = st.button(
            "Run optimization",
            type="primary",
            disabled=not cfg_ok,
            use_container_width=False,
        )

        if run_btn:
            ss.run_done = False
            ss.last_error = None
            log_area = st.empty()
            try:
                with st.spinner("Running pipeline..."):
                    _run_full_pipeline(log_area)
                st.success("Done! Go to the Compare tab.")
                st.balloons()
            except Exception as e:
                ss.last_error = str(e)
                st.error(f"Pipeline failed: {e}")

        if ss.last_error:
            with st.expander("Last error"):
                st.code(ss.last_error)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Compare
# ══════════════════════════════════════════════════════════════════════════════
with tab_cmp:
    st.header("Compare")

    if not ss.run_done and ss.baseline_avg is None:
        st.info("No results yet. Run the optimization from the **Optimize** tab.")
    else:
        # ── Top metrics ───────────────────────────────────────────────────────
        st.subheader("Overall Scores")
        c1, c2, c3 = st.columns(3)
        if ss.baseline_avg is not None:
            c1.metric("Baseline", f"{ss.baseline_avg:.1f} / 100")
        if ss.opt_avg is not None:
            delta_instr = ss.opt_avg - ss.baseline_avg if ss.baseline_avg else 0
            c2.metric("MIPROv2 (instruction)", f"{ss.opt_avg:.1f} / 100",
                      delta=f"{delta_instr:+.1f}")
        if ss.demos_avg is not None:
            delta_demos = ss.demos_avg - ss.baseline_avg if ss.baseline_avg else 0
            c3.metric("MIPROv2 + typed demos", f"{ss.demos_avg:.1f} / 100",
                      delta=f"{delta_demos:+.1f}")

        # ── Per-task-type breakdown ────────────────────────────────────────────
        if ss.baseline_by_type:
            st.subheader("Per-Task-Type Breakdown")
            all_types = sorted(set(
                list(ss.baseline_by_type) +
                list(ss.opt_by_type) +
                list(ss.demos_by_type)
            ))
            table_rows = []
            for t in all_types:
                row = {
                    "Task type": t,
                    "Baseline": round(ss.baseline_by_type.get(t, 0), 1),
                }
                if ss.opt_by_type:
                    row["MIPROv2"] = round(ss.opt_by_type.get(t, 0), 1)
                    row["Delta (instr)"] = round(
                        ss.opt_by_type.get(t, 0) - ss.baseline_by_type.get(t, 0), 1
                    )
                if ss.demos_by_type:
                    row["MIPROv2+demos"] = round(ss.demos_by_type.get(t, 0), 1)
                    row["Delta (demos)"] = round(
                        ss.demos_by_type.get(t, 0) - ss.baseline_by_type.get(t, 0), 1
                    )
                table_rows.append(row)
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

        # ── Discovered instruction ─────────────────────────────────────────────
        if ss.opt_instruction:
            st.subheader("Discovered Instruction (MIPROv2)")
            st.info(ss.opt_instruction)

        # ── Per-example results ────────────────────────────────────────────────
        if ss.demos_results:
            st.subheader("Per-Example Results (MIPROv2 + demos)")

            import pandas as pd
            baseline_map = {r["id"]: r["score"] for r in ss.baseline_results}
            rows = []
            for r in ss.demos_results:
                b = baseline_map.get(r["id"], 0)
                rows.append({
                    "#": int(r["id"]) + 1,
                    "Type": r["task_type"],
                    "Baseline": b,
                    "Optimized": r["score"],
                    "Delta": r["score"] - b,
                    "Output preview": _safe(r.get("output", ""), 80),
                })
            df = pd.DataFrame(rows)

            def _color_delta(val):
                if val > 0:
                    return "color: green"
                elif val < 0:
                    return "color: red"
                return ""

            try:
                styled = df.style.map(_color_delta, subset=["Delta"])
            except AttributeError:
                styled = df.style.applymap(_color_delta, subset=["Delta"])
            st.dataframe(styled, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: History
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.header("History")
    st.caption("Every optimization run is logged here automatically.")

    import pandas as pd

    all_runs = run_history.load_all()

    if not all_runs:
        st.info("No runs recorded yet. Complete an optimization to see history here.")
    else:
        # ── Summary table ─────────────────────────────────────────────────────
        summary_rows = [
            {
                "Run ID": r["id"],
                "Timestamp": r["timestamp"],
                "Model": r["model"].split("/")[-1],
                "Examples": r["n_examples"],
                "Baseline": r["baseline_avg"],
                "MIPROv2+demos": r["demos_avg"],
                "Delta": r["delta"],
                "Task types": ", ".join(r.get("task_types", [])),
            }
            for r in all_runs
        ]
        df_hist = pd.DataFrame(summary_rows)

        def _color_delta_hist(val):
            if isinstance(val, (int, float)):
                return "color: green" if val > 0 else ("color: red" if val < 0 else "")
            return ""

        try:
            styled_hist = df_hist.style.map(_color_delta_hist, subset=["Delta"])
        except AttributeError:
            styled_hist = df_hist.style.applymap(_color_delta_hist, subset=["Delta"])

        st.dataframe(styled_hist, use_container_width=True, hide_index=True)

        # ── Run detail ────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Run detail")
        run_ids = [r["id"] for r in all_runs]
        selected_id = st.selectbox("Select a run to inspect", run_ids)
        run = next(r for r in all_runs if r["id"] == selected_id)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline", f"{run['baseline_avg']:.1f}")
        c2.metric("MIPROv2", f"{run['opt_avg']:.1f}")
        c3.metric("MIPROv2+demos", f"{run['demos_avg']:.1f}")
        c4.metric("Delta", f"{run['delta']:+.1f}")

        if run.get("baseline_by_type") or run.get("demos_by_type"):
            all_types = sorted(set(
                list(run.get("baseline_by_type", {})) +
                list(run.get("demos_by_type", {}))
            ))
            type_rows = [
                {
                    "Task type": t,
                    "Baseline": round(run.get("baseline_by_type", {}).get(t, 0), 1),
                    "Optimized": round(run.get("demos_by_type", {}).get(t, 0), 1),
                    "Delta": round(
                        run.get("demos_by_type", {}).get(t, 0)
                        - run.get("baseline_by_type", {}).get(t, 0), 1
                    ),
                }
                for t in all_types
            ]
            st.dataframe(type_rows, use_container_width=True, hide_index=True)

        if run.get("instruction"):
            st.subheader("Discovered instruction")
            st.info(run["instruction"])
            if st.button("Load this instruction into Compare tab"):
                ss.opt_instruction = run["instruction"]
                ss.baseline_avg = run["baseline_avg"]
                ss.opt_avg = run["opt_avg"]
                ss.demos_avg = run["demos_avg"]
                ss.baseline_by_type = run.get("baseline_by_type", {})
                ss.demos_by_type = run.get("demos_by_type", {})
                ss.run_done = True
                st.success("Loaded. Switch to the Compare tab.")

        st.divider()
        if st.button("Delete this run", type="secondary"):
            run_history.delete_run(selected_id)
            st.rerun()
