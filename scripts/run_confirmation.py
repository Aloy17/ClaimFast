"""Run one ClaimFast strategy against the independent confirmation set."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import STRATEGIES, load_dataset, run_evaluation

DATASET = ROOT / "data" / "confirmation.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "confirmation_2026-11-17"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument(
        "--record-id",
        help="Evaluate only one confirmation record, e.g. CONF001.",
    )
    args = parser.parse_args()

    dataset_path = DATASET
    output_dir = OUTPUT_DIR / f"run_{args.run}"
    temporary_path = None

    if args.record_id:
        records = load_dataset(DATASET)
        matches = [record for record in records if record["id"] == args.record_id]
        if not matches:
            parser.error(
                f"Unknown record ID {args.record_id!r}. "
                f"Choose CONF001 through CONF{len(records):03d}."
            )

        temporary_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".jsonl",
            prefix="claimfast_confirmation_",
            delete=False,
        )
        temporary_path = Path(temporary_file.name)
        with temporary_file:
            temporary_file.write(
                json.dumps(matches[0], ensure_ascii=False) + "\\n"
            )

        dataset_path = temporary_path
        output_dir = OUTPUT_DIR / "single_record" / args.record_id / f"run_{args.run}"

    try:
        run_evaluation(
            strategy=args.strategy,
            run_number=args.run,
            dataset_path=dataset_path,
            output_dir=output_dir,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
