# DSPy Prompt Optimizer

A self-improving prompt optimizer built with [DSPy](https://github.com/stanfordnlp/dspy), Gemini, and Streamlit.

Give it a rough task description and a set of test cases. It uses DSPy's **MIPROv2** optimizer to automatically rewrite the system instruction until output quality improves — then shows you the before/after comparison.

---

## How it works

```
rough prompt + test cases
        │
        ▼
  Baseline evaluation        ← LLM-as-judge scores each output 0-100
        │
        ▼
  MIPROv2 optimization       ← DSPy searches for a better system instruction
        │
        ▼
  Re-evaluation              ← same judge, same examples, new instruction
        │
        ▼
  Before / after comparison  ← score delta, per-task breakdown, full table
```

The optimizer is constrained to improve the **system instruction prefix** only. To demonstrate real gains, the tool also injects one high-quality reference example per task type as a few-shot demo.

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

---

## Run

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## UI tabs

| Tab | What it does |
|---|---|
| **Configure** | Set API key, choose model, load built-in examples or paste your own |
| **Optimize** | Run the full pipeline (baseline → MIPROv2 → evaluate) |
| **Compare** | Score metrics, per-task breakdown, discovered instruction, per-example table |
| **History** | Log of every run with scores and instructions; click any run to reload it |

---

## Rate limits (free tier)

The free Gemini tier allows ~15 requests/minute. The tool adds a 4-second sleep between API calls. A full run (20 examples) takes **15-25 minutes**:

| Step | API calls |
|---|---|
| Baseline evaluation | 40 (20 program + 20 judge) |
| MIPROv2 optimization | ~80-150 |
| Post-optimization evaluation | 40 |
| Typed-demos evaluation | 40 |
| **Total** | **~200-270** |

The optimized program is saved to `saved_prompts/program_mipro_latest.json` after compilation — rerunning the app skips the expensive optimization step automatically.

---

## Command-line test scripts

| Script | Purpose |
|---|---|
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
│   ├── main.py               # Streamlit UI (4 tabs)
│   ├── judge.py              # LLM-as-judge scorer
│   ├── optimizer.py          # MIPROv2 / SIMBA optimization + metric
│   ├── programs/
│   │   ├── signatures.py     # DSPy TaskSignature
│   │   └── base_program.py   # BaseProgram (ChainOfThought wrapper)
│   └── utils/
│       ├── dataset.py        # Load test cases → dspy.Example list
│       └── history.py        # Persist run history to JSON
├── test_cases/
│   └── examples.json         # 20 built-in examples (email / bug / code)
├── saved_prompts/            # Optimized programs + history (git-ignored)
├── requirements.txt
├── verify_setup.py           # Pre-flight check script
└── .env                      # GEMINI_API_KEY (not committed)
```

---

## Example results (built-in test cases)

| Variant | Score | Delta |
|---|---|---|
| Baseline (vague task descriptions) | 60.5 / 100 | — |
| MIPROv2 instruction only | 63.0 / 100 | +2.5 |
| MIPROv2 + typed demos | 69.5 / 100 | **+9.0** |

Per-task breakdown:

| Task type | Baseline | Optimized | Delta |
|---|---|---|---|
| email_subject | 94.3 | 95.7 | +1.4 |
| bug_report_summary | 50.7 | 52.1 | +1.4 |
| code_explanation | 32.5 | 59.2 | **+26.7** |

The biggest gain is on code explanations — the model learns to lead with a real-world analogy instead of technical jargon when it sees a reference example.
