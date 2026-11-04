"""Run one ClaimFast prompt strategy against the held-out test dataset.

The JSONL file is the canonical, append-only experiment log. The CSV is a
regenerated tabular view of that log, so interrupted CSV exports can be
recovered safely by rerunning the same experiment.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from groq_client import MODEL, ask_model
from prompts import PROMPT_STRATEGIES, build_strategy_messages
from validator import FIELDS, parse_strategy_response, validate_claim

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "test.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
STRATEGIES = tuple(PROMPT_STRATEGIES)
TEMPERATURE = 0
REASONING_EFFORT = "low"
MAX_COMPLETION_TOKENS = 2000

CSV_COLUMNS = [
    "timestamp_utc", "model", "strategy", "run_number", "record_id",
    "dataset_fingerprint", "prompt_config_fingerprint",
    "temperature", "reasoning_effort", "max_completion_tokens",
    "json_valid", "validation_passed", "parse_error", "validation_errors",
    "api_error", "api_status_code", "retry_after", "request_id",
    "rate_limit_headers", "latency_seconds", "prompt_tokens",
    "completion_tokens", "total_tokens", "expected_json", "predicted_json",
]


def load_dataset(path=DEFAULT_DATASET):
    """Load JSONL and enforce the assignment's exact dataset/output schema."""
    records = []
    seen_ids = set()
    required_record_keys = {"id", "message", "expected"}
    required_expected_keys = set(FIELDS)

    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict) or set(record) != required_record_keys:
                raise ValueError(
                    f"Line {line_number} must contain exactly: "
                    "id, message, expected"
                )
            if not isinstance(record["id"], str) or not record["id"].strip():
                raise ValueError(f"Line {line_number} has an invalid id")
            if record["id"] in seen_ids:
                raise ValueError(f"Duplicate record id: {record['id']}")
            if not isinstance(record["message"], str) or not record["message"].strip():
                raise ValueError(f"Record {record['id']} has an invalid message")
            if not isinstance(record["expected"], dict):
                raise ValueError(f"Record {record['id']} expected must be an object")
            if set(record["expected"]) != required_expected_keys:
                raise ValueError(
                    f"Record {record['id']} expected must contain exactly "
                    "the nine ClaimFast fields"
                )

            label_problems = validate_claim(record["expected"], record["message"])
            if label_problems:
                raise ValueError(
                    f"Record {record['id']} has invalid expected labels: "
                    + "; ".join(label_problems)
                )

            seen_ids.add(record["id"])
            records.append(record)

    if not records:
        raise ValueError("Dataset contains no records")
    return records



def _fingerprint_inputs(dataset_path, records, strategy):
    """Fingerprint the dataset and effective prompt configuration."""
    dataset_hash = hashlib.sha256(
        Path(dataset_path).read_bytes()
    ).hexdigest()

    prompt_payload = {
        "strategy": strategy,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "messages": [
            build_strategy_messages(strategy, record["message"])
            for record in records
        ],
    }

    canonical = json.dumps(
        prompt_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    prompt_hash = hashlib.sha256(canonical).hexdigest()
    return dataset_hash, prompt_hash


def _read_experiment_log(raw_path):
    records = []
    if not raw_path.exists():
        return records

    with raw_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Experiment log is malformed on line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Experiment log line {line_number} is not a JSON object"
                )
            records.append(record)
    return records


def _write_csv(records, csv_path):
    """Regenerate CSV from the canonical JSONL log."""
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            row = {
                key: record.get(key)
                for key in CSV_COLUMNS
            }
            row["validation_errors"] = json.dumps(
                record.get("validation_errors", []), ensure_ascii=False
            )
            row["rate_limit_headers"] = json.dumps(
                record.get("rate_limit_headers", {}), ensure_ascii=False,
                sort_keys=True,
            )
            row["expected_json"] = json.dumps(
                record.get("expected"), ensure_ascii=False, sort_keys=True
            )
            row["predicted_json"] = (
                json.dumps(record["predicted"], ensure_ascii=False, sort_keys=True)
                if record.get("predicted") is not None else ""
            )
            writer.writerow(row)


def run_evaluation(
    strategy,
    run_number,
    dataset_path=DEFAULT_DATASET,
    output_dir=DEFAULT_OUTPUT_DIR,
    model_call=None,
):
    """Run or resume one strategy/run pair. model_call is injectable for tests."""
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from: {', '.join(STRATEGIES)}"
        )
    if isinstance(run_number, bool) or not isinstance(run_number, int) or run_number < 1:
        raise ValueError("run_number must be a positive integer")

    records = load_dataset(dataset_path)
    dataset_fingerprint, prompt_fingerprint = _fingerprint_inputs(
        dataset_path, records, strategy
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_responses.jsonl"
    csv_path = output_dir / "results.csv"

    experiment_records = _read_experiment_log(raw_path)
    matching_run = [
        item for item in experiment_records
        if item.get("strategy") == strategy
        and item.get("run_number") == run_number
    ]

    for item in matching_run:
        if (
            item.get("dataset_fingerprint") != dataset_fingerprint
            or item.get("prompt_config_fingerprint") != prompt_fingerprint
        ):
            raise ValueError(
                f"Cannot resume {strategy}/run {run_number}: existing records "
                "have missing or mismatched dataset/prompt fingerprints. "
                "Preserve the historical log and use a separate output directory."
            )

    # The JSONL log is append-only. The latest attempt determines whether a
    # logical evaluation is complete; provider/API errors remain retryable.
    latest_attempts = {}
    for item in experiment_records:
        key = (item.get("strategy"), item.get("run_number"), item.get("record_id"))
        latest_attempts[key] = item

    completed = {
        key for key, item in latest_attempts.items()
        if item.get("api_error") is None
    }

    pending = [
        record for record in records
        if (strategy, run_number, record["id"]) not in completed
    ]
    retryable = sum(
        latest_attempts.get((strategy, run_number, record["id"]), {}).get("api_error")
        is not None
        for record in pending
    )
    skipped = len(records) - len(pending)

    call_model = model_call or ask_model
    newly_processed = 0

    for record in pending:
        messages = build_strategy_messages(strategy, record["message"])
        timestamp = datetime.now(timezone.utc).isoformat()
        result = None
        api_error = None
        error_metadata = {}
        try:
            result = call_model(messages)
        except Exception as error:
            api_error = f"{type(error).__name__}: {str(error)[:1000]}"
            error_metadata = getattr(error, "groq_metadata", {}) or {}

        raw_text = result.get("text") if result else None
        parsed = None
        parse_error = None
        validation_errors = []

        if api_error is None:
            parsed, parse_error = parse_strategy_response(raw_text, strategy)
            if parse_error is None:
                validation_errors = validate_claim(parsed, record["message"])

        metadata = result if result is not None else error_metadata
        item = {
            "timestamp_utc": timestamp,
            "model": MODEL,
            "strategy": strategy,
            "run_number": run_number,
            "record_id": record["id"],
            "dataset_fingerprint": dataset_fingerprint,
            "prompt_config_fingerprint": prompt_fingerprint,
            "temperature": TEMPERATURE,
            "reasoning_effort": REASONING_EFFORT,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "message": record["message"],
            "expected": record["expected"],
            "predicted": parsed,
            "raw_response": raw_text,
            "json_valid": parse_error is None and api_error is None,
            "validation_passed": (
                parse_error is None and api_error is None
                and not validation_errors
            ),
            "parse_error": parse_error,
            "validation_errors": validation_errors,
            "api_error": api_error,
            "api_status_code": metadata.get("api_status_code"),
            "retry_after": metadata.get("retry_after"),
            "request_id": metadata.get("request_id"),
            "rate_limit_headers": metadata.get("rate_limit_headers", {}),
            "latency_seconds": metadata.get("latency_seconds"),
            "prompt_tokens": result.get("prompt_tokens") if result else None,
            "completion_tokens": result.get("completion_tokens") if result else None,
            "total_tokens": result.get("total_tokens") if result else None,
        }

        # Append the canonical record first. If CSV regeneration is interrupted,
        # the next invocation rebuilds it from this log and skips this record.
        with raw_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
            file.flush()

        experiment_records.append(item)
        _write_csv(experiment_records, csv_path)
        newly_processed += 1

        status = "API ERROR" if api_error else (
            "PARSE ERROR" if parse_error else (
                "VALID" if item["validation_passed"] else "VALIDATION FAILED"
            )
        )
        print(f"{record['id']}: {status}")

    # Always regenerate CSV, including when resuming a fully completed run.
    _write_csv(experiment_records, csv_path)

    print(
        f"\nStrategy {strategy}, run {run_number}: "
        f"{newly_processed} new attempts, "
        f"{skipped} completed and skipped, "
        f"{retryable} prior API errors retried, "
        f"{len(records)} total dataset records."
    )
    print(f"Raw log: {raw_path}")
    print(f"CSV results: {csv_path}")
    return {
        "processed": newly_processed,
        "skipped": skipped,
        "retried": retryable,
        "total": len(records),
        "raw_path": raw_path,
        "csv_path": csv_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one ClaimFast prompt strategy on the held-out test set."
    )
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--run", required=True, type=int, dest="run_number")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    run_evaluation(
        strategy=args.strategy,
        run_number=args.run_number,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
