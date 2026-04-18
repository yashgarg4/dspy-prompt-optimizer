# DSPy Prompt Optimizer

A self-improving prompt optimizer built with [DSPy](https://github.com/stanfordnlp/dspy), Gemini, and Streamlit.

Give it a rough task description and a set of test cases. It uses DSPy's **MIPROv2** optimizer to automatically rewrite the system instruction until output quality improves — then shows you the before/after comparison.

---

## What is this project?

A tool that **automatically improves AI prompts without you writing better instructions manually**. You give it:
1. A rough task description (`"Write an email subject line."`)
2. Test cases with inputs and ideal outputs

It then uses DSPy's MIPROv2 optimizer to discover a *better* system instruction on its own — measured by an LLM judge — and shows you the before/after improvement.

---

## How it works

```
User's rough prompt + test cases
           │
           ▼
  ┌─────────────────────┐
  │   BaseProgram       │  ← runs the task (ChainOfThought LM call)
  └─────────────────────┘
           │
           ▼
  ┌─────────────────────┐
  │   LLMJudge          │  ← scores each output 0-100 using a separate LM
  └─────────────────────┘
           │  baseline score (60.5/100)
           ▼
  ┌─────────────────────┐
  │   MIPROv2           │  ← DSPy optimizer: tries 6 instruction candidates
  │   Optimizer         │    across 9 trials, picks the best one
  └─────────────────────┘
           │  optimized instruction discovered
           ▼
  ┌─────────────────────┐
  │   Re-evaluation     │  ← same judge, same examples, new instruction
  └─────────────────────┘
           │  optimized score (69.5/100, +9 pts)
           ▼
     Streamlit UI        ← Compare tab shows full before/after breakdown
```

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd dspy_prompt_optimizer
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Add your API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=AIza...your_key_here
```

Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Verify setup

```bash
python verify_setup.py
```

This confirms your API key works and all dependencies are importable before you start the app.

### 4. Run

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Component-by-Component Breakdown

### 1. `app/programs/signatures.py` + `base_program.py` — The Task Runner

```python
class TaskSignature(dspy.Signature):
    task_description: str = dspy.InputField()
    input_data: str = dspy.InputField()
    output: str = dspy.OutputField()

class BaseProgram(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought(TaskSignature)
```

This is the **student** — the program being optimized. `ChainOfThought` forces the LM to reason step-by-step before producing an answer. It is deliberately generic — the same program handles email subjects, bug summaries, and code explanations purely based on what `task_description` says.

**Key insight:** DSPy separates *program structure* from *prompt wording*. You define what fields go in and out; DSPy figures out the best wording.

---

### 2. `app/judge.py` — The LLM-as-Judge

```python
class JudgeSignature(dspy.Signature):
    """Start at 100, deduct for:
    - Bug summaries: jargon (-20/term), multiple sentences (-25), no user impact (-20)
    - Code explanations: technical terms (-15/term), no analogy (-20)
    - Email subjects: generic (-25), too long (-15)
    """
    task_description, input_data, ai_output → score (int), reasoning (str)
```

The judge runs in a **completely isolated DSPy context** (`dspy.context(lm=self._lm)`) so its calls never interfere with the training LM. The rubric is quality-based, not rule-based — it penalizes *inherent failures* (jargon, missing analogies) rather than checking explicit constraints. This is what creates a meaningful gap between baseline and optimized.

**Key insight:** The judge needs to be stricter than the model's natural behavior to create room for improvement. If the model already scores 99%, there is nothing for DSPy to optimize.

---

### 3. `app/optimizer.py` — The Optimization Engine

**Metric function** (the bridge between DSPy and the judge):
```python
def metric(example, prediction, _trace=None) -> float:
    score, _ = judge.score(task_description, input_data, ai_output)
    return score / 100.0   # DSPy expects [0, 1]
```

**MIPROv2 optimization** (the core loop):
```python
optimizer = dspy.MIPROv2(
    metric=metric,
    auto=None,                 # explicit control (auto='light' overrides your settings)
    max_bootstrapped_demos=0,  # no demo collection → avoids email-only bias
    data_aware_proposer=False, # skip cached email-biased data summary
)
optimized = optimizer.compile(program, trainset=trainset, num_trials=9)
```

What MIPROv2 actually does inside `compile()`:
1. **Bootstrap** — runs the program on examples, collects successful traces as candidate demos
2. **Propose instructions** — sends the program code + data summary to the LM and asks it to generate 6 better instructions
3. **Bayesian search** — evaluates each instruction on the full trainset using the metric, uses Bayesian optimization (Optuna) to pick which combination to try next
4. **Return best** — whichever (instruction, demos) combination scored highest becomes the optimized program

**`inject_typed_demos()`** — the extra boost:
```python
# After MIPROv2, manually inject 1 reference example per task type
program.predict.predict.demos = [
    Example(task_description="Explain code to a beginner", output="Think of it like a recipe..."),
    Example(task_description="Summarize bug report", output="Users cannot log in after an update."),
    ...
]
```

This gives the model a concrete *example of quality* for each task type. Code explanations jumped from 32.5 → 59.2 because the model saw "use an analogy" in action.

---

### 4. `app/utils/dataset.py` — The Training Data

20 examples across 3 task types, all with **intentionally vague task descriptions**:

| Field | Value |
|---|---|
| `task_description` | `"Write an email subject line."` |
| `input_data` | Full email body text |
| `expected_output` | High-quality reference answer |
| `task_type` | `email_subject` |

**The vagueness is critical.** If `task_description` said *"Write a subject line under 65 chars with no passive voice"*, the model would follow those instructions perfectly regardless of optimization (99.5% baseline) — leaving nothing for DSPy to improve. Vague descriptions create the gap.

---

### 5. `app/utils/history.py` — Persistence

Every run is appended to `saved_prompts/history.json`:
```json
{
  "timestamp": "2026-04-18T20:57:00",
  "baseline_avg": 60.5,
  "demos_avg": 69.5,
  "delta": 9.0,
  "instruction": "Act as an expert analyst...",
  "task_types": ["bug_report_summary", "code_explanation", "email_subject"]
}
```

The History tab lets you browse past runs, inspect what instruction was discovered, and reload results into the Compare tab without re-running anything.

---

### 6. `app/main.py` — The Streamlit UI

Four tabs, one session_state dict (`ss`) that persists across tab switches:

| Tab | What it does |
|---|---|
| **Configure** | Set API key, choose model, load built-in examples or paste your own |
| **Optimize** | Run the full pipeline (baseline → MIPROv2 → evaluate) |
| **Compare** | Score metrics, per-task breakdown, discovered instruction, per-example table |
| **History** | Log of every run with scores and instructions; click any run to reload it |

The pipeline is **idempotent by design**: `load_optimized_program("mipro")` checks for `saved_prompts/program_mipro_latest.json` and skips the ~10-minute compilation if it already exists. Same for baseline — loads from `phase2_checkpoint.json` if available.

---

## Results

| Variant | Score | Delta |
|---|---|---|
| Baseline (vague task descriptions) | 60.5 / 100 | — |
| MIPROv2 instruction only | 63.0 / 100 | +2.5 |
| MIPROv2 + typed demos | 69.5 / 100 | **+9.0** |

Per-task breakdown:

| Task type | Baseline | Optimized | Delta |
|---|---|---|---|
| `email_subject` | 94.3 | 95.7 | +1.4 |
| `bug_report_summary` | 50.7 | 52.1 | +1.4 |
| `code_explanation` | 32.5 | 59.2 | **+26.7** |

The biggest gain is on code explanations — the model learns to lead with a real-world analogy instead of technical jargon when it sees a reference example.

The instruction MIPROv2 discovered:
> *"Act as an expert analyst and executor. Carefully examine the provided task description and input data. Before providing your final answer, engage in a step-by-step reasoning process to break down the requirements and analyze the source content. Ensure your final output is precise, contextually relevant, and directly fulfills the goal defined in the task description."*

---

## Key Learnings

### 1. DSPy optimizers improve the system instruction prefix, not input fields

`task_description` is an `InputField` — the model reads it directly every call. DSPy can only change what comes *before* the input fields in the prompt (the system instruction). This means if your task_description already gives detailed guidance, optimization cannot help — it has nothing new to add. Vague task descriptions are a prerequisite for meaningful optimization.

### 2. Baseline design matters more than the optimization algorithm

A baseline of 99.5% left no room for improvement — there was literally nowhere to go. Dropping it to 60.5% by making task descriptions vague and switching to quality-based judging created real signal. The optimizer is only as good as the gap you give it to close.

### 3. Few-shot demos are more powerful than instruction wording

MIPROv2's instruction alone gave +2.5 pts. Adding one typed demo per task type gave +6.5 more. Showing the model *what good looks like* beats telling it *how to think*. This is consistent with what AI labs have observed: in-context examples are often more effective than elaborate zero-shot instructions.

### 4. Bootstrap bias is a real and undocumented failure mode

When `auto='light'` was used, MIPROv2 bootstrapped traces only from examples that scored well at baseline (email: 94%). All bootstrapped demos were email-specific. The data summary it generated said *"This dataset focuses on email subject lines"* — because it only saw email examples. All 6 instruction candidates ended up email-specific, hurting bug and code scores, so the default instruction won. Fix: `auto=None` + `max_bootstrapped_demos=0` + `data_aware_proposer=False`.

### 5. LLM-as-judge design determines everything

**Constraint-based judging** ("did it follow rule X?") → model follows rules instantly → 99.5% → nothing to optimize.

**Quality-based judging** ("does it avoid jargon? does it use an analogy?") → surfaces real model weaknesses → meaningful baseline → real optimization signal.

The rubric design is the hardest part of the entire system. Get it wrong and the whole pipeline produces useless results.

### 6. Rate limits require defensive architecture

Free-tier Gemini: ~15 requests/minute. A full optimization run makes ~200-270 API calls. Without defensive design, one rate-limit error mid-run loses all progress. Solution:
- 4-second sleep between every API call
- Exponential backoff retry (30s, 60s, 120s...) on 429/503 errors
- Disk-based checkpointing: evaluation results saved after every example so any interruption is resumable
- Idempotent pipeline: re-running the script picks up exactly where it left off

### 7. DSPy's cache is a double-edged sword

DSPy caches every LM call by exact prompt in `~/.dspy_cache`. This saved ~100 API calls during re-runs (same inputs → cached output). But it also caused the data summary from a previous email-biased run to be returned as a cached result on subsequent runs — feeding wrong context to the instruction proposer. The fix was `data_aware_proposer=False` to skip the data summary step entirely.

---

## Failure Modes Encountered and Fixed

| Failure | Root cause | Fix |
|---|---|---|
| 99.5% baseline, no improvement | Constraint-based task descriptions; model followed rules perfectly | Made all task_descriptions vague; switched judge to quality rubric |
| All instruction candidates email-specific | `auto='light'` ran bootstrapping despite `max_bootstrapped_demos=0`; only high-scoring email examples bootstrapped | `auto=None` (respects manual params) + `max_bootstrapped_demos=0` |
| Data summary still email-biased after fix | DSPy cache returned stale email-focused summary from previous run | `data_aware_proposer=False` |
| `(instruction not extractable)` | Wrong attribute path — ChainOfThought nests predictor at `.predict.predict`, not `.predict` | Fixed path to `program.predict.predict.signature.instructions` |
| Unicode crash on Windows terminal | Model output contained emoji; Windows uses cp1252 encoding | `.encode("ascii", "replace").decode("ascii")` on all preview strings |
| 503 Service Unavailable during optimization | Gemini preview model experiencing high demand | Retry loop with 60s/120s/180s waits around entire `compile()` call |

---

## Rate Limits (free tier)

| Step | API calls |
|---|---|
| Baseline evaluation | 40 (20 program + 20 judge) |
| MIPROv2 optimization | ~80-150 |
| Post-optimization evaluation | 40 |
| Typed-demos evaluation | 40 |
| **Total** | **~200-270** |

Estimated time: **15-25 minutes** on free-tier Gemini.

The optimized program is saved to `saved_prompts/program_mipro_latest.json` after compilation — rerunning the app skips the expensive optimization step automatically.

---

## Command-line test scripts

| Script | Purpose |
|---|---|
| `python verify_setup.py` | Pre-flight check: API key, imports, live LM call |
| `python test_phase1.py` | Confirm DSPy end-to-end (single LM call) |
| `python test_phase2.py` | Run baseline evaluation on all 20 examples, save checkpoint |
| `python test_phase3.py` | Run MIPROv2 optimization and print before/after comparison |
| `python test_phase3.py --simba` | Also run SIMBA optimizer for comparison |
| `python test_phase3.py --reset` | Clear saved programs and checkpoints, re-run from scratch |

---

## Project structure

```
dspy_prompt_optimizer/
├── app/
│   ├── main.py               # Streamlit UI (4 tabs: Configure/Optimize/Compare/History)
│   ├── judge.py              # LLM-as-judge scorer with quality-dimension rubric
│   ├── optimizer.py          # MIPROv2 / SIMBA optimization, metric, inject_typed_demos
│   ├── programs/
│   │   ├── signatures.py     # DSPy TaskSignature definition
│   │   └── base_program.py   # BaseProgram (ChainOfThought wrapper)
│   └── utils/
│       ├── dataset.py        # Load test cases → dspy.Example list
│       └── history.py        # Persist run history to saved_prompts/history.json
├── test_cases/
│   └── examples.json         # 20 built-in examples (email / bug / code)
├── saved_prompts/            # Optimized programs + history (git-ignored)
│   ├── program_mipro_latest.json   # Compiled MIPROv2 program (auto-loaded on rerun)
│   ├── phase2_checkpoint.json      # Baseline evaluation results
│   ├── phase3_checkpoint.json      # Optimization evaluation results
│   └── history.json                # All past runs
├── test_phase1.py            # Phase 1: end-to-end DSPy smoke test
├── test_phase2.py            # Phase 2: baseline evaluation with checkpoint/resume
├── test_phase3.py            # Phase 3: MIPROv2 + SIMBA optimization
├── verify_setup.py           # Pre-flight check script
├── requirements.txt
└── .env                      # GEMINI_API_KEY (not committed)
```

---

## What makes this project non-trivial

- **LLM-as-judge isolation** — judge runs in a separate DSPy context so it never contaminates the training LM's state
- **Bootstrap bias diagnosis** — identified and fixed an undocumented failure mode in MIPROv2's auto mode where only high-scoring task types get bootstrapped, biasing all instruction proposals toward those tasks
- **Quality-vs-constraint judging** — discovered through experimentation that constraint-based scoring creates a ceiling that optimization cannot break through
- **Production patterns** — checkpoint/resume, exponential backoff, idempotent pipeline. The system can survive a rate-limit interruption at any point and resume exactly where it left off
- **Cache-aware debugging** — traced a data summary corruption bug to DSPy's diskcache returning stale results from a prior run
