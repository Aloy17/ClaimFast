"""Offline pre-benchmark integrity checks. This script never calls the provider."""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    MAX_COMPLETION_TOKENS,
    REASONING_EFFORT,
    STRATEGIES,
    TEMPERATURE,
    _fingerprint_inputs,
    load_dataset,
)
from groq_client import MODEL

DEV_DATASET = ROOT / "data" / "dev.jsonl"
EXPECTED_COUNTS = {"dev": 20, "test": 60}
EXPECTED_STRATEGIES = ("A", "B1", "B2", "B3", "C")
RUNS_PER_STRATEGY = 3


def expected_ids(prefix, count):
    return {f"{prefix}{number:03d}" for number in range(1, count + 1)}


def inspect_dataset(path, prefix, expected_count):
    path = Path(path)
    records = load_dataset(path)
    actual_ids = {record["id"] for record in records}
    required_ids = expected_ids(prefix, expected_count)
    if len(records) != expected_count:
        raise ValueError(
            f"{path.name}: expected {expected_count} records, found {len(records)}"
        )
    if actual_ids != required_ids:
        missing = sorted(required_ids - actual_ids)
        unexpected = sorted(actual_ids - required_ids)
        raise ValueError(
            f"{path.name}: ID mismatch; missing={missing}, unexpected={unexpected}"
        )
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return records, digest


def run_preflight():
    if tuple(STRATEGIES) != EXPECTED_STRATEGIES:
        raise ValueError(
            f"Expected strategies {EXPECTED_STRATEGIES}, found {tuple(STRATEGIES)}"
        )

    dev_records, dev_hash = inspect_dataset(
        DEV_DATASET, "DEV", EXPECTED_COUNTS["dev"]
    )
    test_records, test_hash = inspect_dataset(
        DEFAULT_DATASET, "TEST", EXPECTED_COUNTS["test"]
    )

    print("ClaimFast offline pre-benchmark check")
    print("-------------------------------------")
    print(f"Development records: {len(dev_records)} (SHA-256 {dev_hash})")
    print(f"Held-out test records: {len(test_records)} (SHA-256 {test_hash})")
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Reasoning effort: {REASONING_EFFORT}")
    print(f"Maximum completion tokens: {MAX_COMPLETION_TOKENS}")
    print(f"Strategies: {', '.join(STRATEGIES)}")
    print(f"Runs per strategy: {RUNS_PER_STRATEGY}")

    logical_evaluations = (
        len(test_records) * len(STRATEGIES) * RUNS_PER_STRATEGY
    )
    print(f"Planned logical evaluations: {logical_evaluations}")

    print("\nDataset and prompt fingerprints:")
    for strategy in STRATEGIES:
        dataset_fingerprint, prompt_fingerprint = _fingerprint_inputs(
            DEFAULT_DATASET, test_records, strategy
        )
        print(f"- {strategy}: dataset={dataset_fingerprint}")
        print(f"  prompt={prompt_fingerprint}")

    raw_log = DEFAULT_OUTPUT_DIR / "raw_responses.jsonl"
    csv_log = DEFAULT_OUTPUT_DIR / "results.csv"
    print("\nFormal output directory:")
    print(DEFAULT_OUTPUT_DIR)
    for path in (raw_log, csv_log):
        if path.exists():
            print(f"- Existing file: {path.name} ({path.stat().st_size} bytes)")
        else:
            print(f"- Not yet created: {path.name}")

    print("\nNo API calls were made. This check does not verify provider quota.")
    return {
        "dev_count": len(dev_records),
        "test_count": len(test_records),
        "strategy_count": len(STRATEGIES),
        "runs_per_strategy": RUNS_PER_STRATEGY,
        "logical_evaluations": logical_evaluations,
        "dataset_fingerprint": test_hash,
    }


if __name__ == "__main__":
    run_preflight()
