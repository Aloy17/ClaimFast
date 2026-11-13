"""Run one claim message through prompt strategies and print each result.

Needs GROQ_API_KEY in .env (makes real API calls). Usage:
    python scripts/demo_strategies.py
    python scripts/demo_strategies.py --strategies A B3 C --message "your claim text"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from groq_client import ask_model  # noqa: E402
from main import SAMPLE_CLAIM  # noqa: E402
from prompts import PROMPT_STRATEGIES, build_strategy_messages  # noqa: E402
from validator import FIELDS, parse_strategy_response, validate_claim  # noqa: E402

LABELS = {
    "A": "A  zero-shot",
    "B1": "B1 few-shot k=3",
    "B2": "B2 few-shot k=5",
    "B3": "B3 few-shot k=8",
    "C": "C  reasoning-guided",
}


def run_strategy(strategy, claim, ask=ask_model):
    result = ask(build_strategy_messages(strategy, claim))
    data, parse_error = parse_strategy_response(result.get("text"), strategy)
    problems = [] if parse_error else validate_claim(data, claim)
    return result, data, parse_error, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", default=SAMPLE_CLAIM, help="Claim message to process")
    parser.add_argument("--strategies", nargs="+", default=["A", "B3", "C"], choices=list(PROMPT_STRATEGIES))
    args = parser.parse_args()

    print("Claim message:\n  " + args.message + "\n")
    for strategy in args.strategies:
        print("=" * 60)
        print(LABELS[strategy])
        print("=" * 60)
        try:
            result, data, parse_error, problems = run_strategy(strategy, args.message)
        except Exception as error:  # demo script: show the failure and continue
            print("Request failed: " + type(error).__name__ + ": " + str(error)[:200] + "\n")
            continue
        if parse_error:
            print("Parse error: " + parse_error + "\n")
            continue
        for field in FIELDS:
            print(f"  {field:22} {data.get(field)}")
        print("  validation:", "passed" if not problems else "; ".join(problems))
        print(
            f"  latency {result.get('latency_seconds')} s | tokens: prompt "
            f"{result.get('prompt_tokens')}, completion {result.get('completion_tokens')}\n"
        )


if __name__ == "__main__":
    main()
