"""Run history — persists every optimization run to saved_prompts/history.json."""

import json
from datetime import datetime
from pathlib import Path

_HISTORY_FILE = Path(__file__).parent.parent.parent / "saved_prompts" / "history.json"


def _load() -> list[dict]:
    if _HISTORY_FILE.exists():
        return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    return []


def _save(runs: list[dict]) -> None:
    _HISTORY_FILE.parent.mkdir(exist_ok=True)
    _HISTORY_FILE.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")


def save_run(
    model: str,
    n_examples: int,
    baseline_avg: float,
    opt_avg: float,
    demos_avg: float,
    instruction: str,
    task_types: list[str],
    baseline_by_type: dict[str, float] | None = None,
    demos_by_type: dict[str, float] | None = None,
) -> str:
    """Append a run record to history. Returns the run ID (timestamp string)."""
    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    record = {
        "id": run_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "model": model,
        "n_examples": n_examples,
        "baseline_avg": round(baseline_avg, 2),
        "opt_avg": round(opt_avg, 2),
        "demos_avg": round(demos_avg, 2),
        "delta": round(demos_avg - baseline_avg, 2),
        "instruction": instruction,
        "task_types": sorted(set(task_types)),
        "baseline_by_type": baseline_by_type or {},
        "demos_by_type": demos_by_type or {},
    }
    runs = _load()
    runs.append(record)
    _save(runs)
    return run_id


def load_all() -> list[dict]:
    """Return all past runs, most recent first."""
    return list(reversed(_load()))


def delete_run(run_id: str) -> None:
    """Remove a run by ID."""
    runs = [r for r in _load() if r["id"] != run_id]
    _save(runs)
