"""Quick sanity-check — run this before starting the app.

    python verify_setup.py

Checks: .env loaded, GEMINI_API_KEY present, DSPy + Streamlit importable,
and one real LM call that confirms the API key works.
"""

import os
import sys
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    line = f"  {status}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return ok


def main() -> None:
    print("\nDSPy Prompt Optimizer — setup verification\n")
    all_ok = True

    # 1. .env file
    env_path = Path(__file__).parent / ".env"
    has_env = env_path.exists()
    all_ok &= check(".env file found", has_env,
                    str(env_path) if has_env else "create .env with GEMINI_API_KEY=...")
    if has_env:
        from dotenv import load_dotenv
        load_dotenv()

    # 2. API key 
    api_key = os.getenv("GEMINI_API_KEY", "")
    has_key = bool(api_key and api_key.startswith("AIza"))
    all_ok &= check(
        "GEMINI_API_KEY set",
        has_key,
        f"{'*' * 8}{api_key[-4:]}" if has_key else "not set or invalid format",
    )

    # 3. Python version 
    py_ok = sys.version_info >= (3, 11)
    all_ok &= check(
        f"Python >= 3.11",
        py_ok,
        f"found {sys.version.split()[0]}",
    )

    # ── 4. Core imports ───────────────────────────────────────────────────────
    for pkg, import_name in [
        ("dspy-ai", "dspy"),
        ("streamlit", "streamlit"),
        ("python-dotenv", "dotenv"),
        ("pydantic", "pydantic"),
    ]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "?")
            all_ok &= check(f"import {import_name}", True, f"v{ver}")
        except ImportError as e:
            all_ok &= check(f"import {import_name}", False, str(e))

    # ── 5. Test cases file ────────────────────────────────────────────────────
    tc_path = Path(__file__).parent / "test_cases" / "examples.json"
    all_ok &= check(
        "test_cases/examples.json exists",
        tc_path.exists(),
        f"{tc_path}" if not tc_path.exists() else "",
    )
    if tc_path.exists():
        import json
        examples = json.loads(tc_path.read_text(encoding="utf-8"))
        all_ok &= check(f"examples.json has entries", len(examples) > 0, f"{len(examples)} found")

    # ── 6. Live API call ──────────────────────────────────────────────────────
    if has_key:
        print("\n  Making a test API call (this uses one request from your quota)...")
        try:
            import dspy
            lm = dspy.LM(
                model="gemini/gemini-3.1-flash-lite-preview",
                api_key=api_key,
                max_tokens=32,
            )
            dspy.configure(lm=lm)

            class _Ping(dspy.Signature):
                """Reply with exactly the word 'pong'."""
                reply: str = dspy.OutputField()

            result = dspy.Predict(_Ping)()
            got = str(result.reply).strip().lower()
            all_ok &= check("Live LM call", True, f"response: '{got[:40]}'")
        except Exception as e:
            all_ok &= check("Live LM call", False, str(e)[:120])
    else:
        print(f"  {WARN}  Skipping live API call (no valid API key).")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    if all_ok:
        print("All checks passed. Start the app with:\n")
        print("    streamlit run app/main.py\n")
    else:
        print("Some checks failed. Fix the issues above, then re-run this script.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
