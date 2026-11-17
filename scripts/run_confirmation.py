"""Run one ClaimFast strategy against the independent confirmation set."""

import argparse
from pathlib import Path

from evaluation import STRATEGIES, run_evaluation

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "confirmation.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "confirmation_2026-11-17"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()
    run_evaluation(
        strategy=args.strategy,
        run_number=args.run,
        dataset_path=DATASET,
        output_dir=OUTPUT_DIR / f"run_{args.run}",
    )


if __name__ == "__main__":
    main()
