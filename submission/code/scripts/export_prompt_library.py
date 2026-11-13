"""Write the exact messages sent for each strategy to submission/prompts/.

Offline. Uses a placeholder where the customer message is inserted.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompts import PROMPT_STRATEGIES, build_strategy_messages  # noqa: E402

PLACEHOLDER = "<CUSTOMER_MESSAGE>"
TITLES = {
    "A": "Prompt A: zero-shot",
    "B1": "Prompt B1: few-shot, k=3",
    "B2": "Prompt B2: few-shot, k=5",
    "B3": "Prompt B3: few-shot, k=8",
    "C": "Prompt C: reasoning-guided",
}


def main():
    out = ROOT / "submission" / "prompts"
    out.mkdir(parents=True, exist_ok=True)
    for strategy in PROMPT_STRATEGIES:
        messages = build_strategy_messages(strategy, PLACEHOLDER)
        parts = [TITLES[strategy], "Model: openai/gpt-oss-20b (Groq) | temperature 0 | reasoning effort low | max 2000 completion tokens", ""]
        for message in messages:
            parts += [f"----- role: {message['role']} -----", message["content"], ""]
        (out / f"prompt_{strategy}.txt").write_text("\n".join(parts), encoding="utf-8")
        print("wrote", out / f"prompt_{strategy}.txt", len(messages), "messages")


if __name__ == "__main__":
    main()
