#!/usr/bin/env python3
"""Offline subgroup analysis for the ClaimFast benchmark."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import analyze_records, load_experiment_log


DEFAULT_INPUT = ROOT / "outputs" / "final_benchmark" / "raw_responses.jsonl"
DEFAULT_DATASET = ROOT / "data" / "test.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "final_benchmark" / "analysis"

STRATEGIES = ("A", "B1", "B2", "B3", "C")
RUNS = (1, 2, 3)

AMBIGUOUS_THEFT_IDS = {"TEST009", "TEST029", "TEST047"}
MULTI_INCIDENT_IDS = {"TEST058"}


def load_test_dataset(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in test dataset line {line_number}: {error}"
                ) from error

            if not isinstance(item, dict) or "id" not in item:
                raise ValueError(
                    f"Test dataset line {line_number} must contain an 'id' field."
                )
            if not isinstance(item.get("expected"), dict):
                raise ValueError(
                    f"Test dataset line {line_number} must contain an 'expected' object."
                )
            records.append(item)

    if not records:
        raise ValueError("Test dataset contains no records.")

    ids = [item["id"] for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Test dataset contains duplicate record IDs.")

    return records


def build_subgroups(dataset, experiment_records):
    dataset_by_id = {item["id"]: item for item in dataset}
    dataset_ids = set(dataset_by_id)
    log_ids = {item["record_id"] for item in experiment_records}

    glass_only_ids = {
        item["id"]
        for item in dataset
        if item["expected"].get("claim_type") == "glass_only"
    }

    definitions = {
        "ambiguous_theft": AMBIGUOUS_THEFT_IDS,
        "glass_only": glass_only_ids,
        "multi_incident": MULTI_INCIDENT_IDS,
    }

    for name, ids in definitions.items():
        missing_from_dataset = ids - dataset_ids
        if missing_from_dataset:
            raise ValueError(
                f"{name} contains IDs missing from the test dataset: "
                f"{sorted(missing_from_dataset)}"
            )

        missing_from_log = ids - log_ids
        if missing_from_log:
            raise ValueError(
                f"{name} contains IDs missing from the benchmark log: "
                f"{sorted(missing_from_log)}"
            )

    return definitions


def validate_run_coverage(records, subgroup_name, subgroup_ids):
    expected_count = len(subgroup_ids)

    for strategy in STRATEGIES:
        for run_number in RUNS:
            matching = [
                item for item in records
                if item["strategy"] == strategy
                and item["run_number"] == run_number
            ]
            matching_ids = {item["record_id"] for item in matching}

            if matching_ids != subgroup_ids or len(matching) != expected_count:
                raise ValueError(
                    f"Incomplete coverage for {subgroup_name}, "
                    f"{strategy}/run_{run_number}: expected {expected_count} "
                    f"records, found {len(matching)}."
                )


def build_report(experiment_records, subgroup_definitions):
    report = {
        "methodology": {
            "description": (
                "Subgroup metrics are calculated with the canonical "
                "ClaimFast analyze_records() evaluator."
            ),
            "record_exact_match": (
                "Strict equality across all nine output fields."
            ),
            "run_handling": (
                "Each subgroup contains the same test messages evaluated "
                "under five strategies and three runs. Run statistics are "
                "calculated across run-level metrics."
            ),
            "limitations": (
                "Subgroup results are diagnostic. In particular, the "
                "multi-incident subgroup contains only one message and "
                "must not be treated as generalisable evidence."
            ),
        },
        "subgroups": {},
    }

    for subgroup_name, subgroup_ids in subgroup_definitions.items():
        selected = [
            item for item in experiment_records
            if item["record_id"] in subgroup_ids
        ]

        validate_run_coverage(selected, subgroup_name, subgroup_ids)

        metrics = analyze_records(selected, projected_calls=1)

        report["subgroups"][subgroup_name] = {
            "unique_record_count": len(subgroup_ids),
            "prediction_count": len(selected),
            "record_ids": sorted(subgroup_ids),
            **metrics,
        }

    return report


def write_outputs(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "subgroup_analysis.json"
    csv_path = output_dir / "subgroup_analysis.csv"

    # Protect existing outputs from accidental replacement.
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing subgroup output(s): "
            + ", ".join(str(path) for path in existing)
        )

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    columns = [
        "subgroup",
        "unique_record_count",
        "prediction_count",
        "strategy",
        "run",
        "record_exact_match",
        "claim_type_accuracy",
        "claim_type_macro_f1",
        "urgency_accuracy",
        "urgency_macro_f1",
        "extraction_micro_f1",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for subgroup_name, subgroup in report["subgroups"].items():
            for run_key, metrics in subgroup["per_run"].items():
                strategy, run_label = run_key.split("/run_")
                writer.writerow({
                    "subgroup": subgroup_name,
                    "unique_record_count": subgroup["unique_record_count"],
                    "prediction_count": subgroup["prediction_count"],
                    "strategy": strategy,
                    "run": run_label,
                    "record_exact_match": metrics["record_exact_match"]["accuracy"],
                    "claim_type_accuracy": metrics["classification"]["claim_type"]["accuracy"],
                    "claim_type_macro_f1": metrics["classification"]["claim_type"]["macro_f1"],
                    "urgency_accuracy": metrics["classification"]["urgency"]["accuracy"],
                    "urgency_macro_f1": metrics["classification"]["urgency"]["macro_f1"],
                    "extraction_micro_f1": metrics["extraction_micro"]["f1"],
                })

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyse ClaimFast subgroups without making API calls."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    experiment_records = load_experiment_log(args.input)
    dataset = load_test_dataset(args.dataset)
    subgroup_definitions = build_subgroups(dataset, experiment_records)
    report = build_report(experiment_records, subgroup_definitions)
    json_path, csv_path = write_outputs(report, args.output_dir)

    print(f"Analysed {len(experiment_records)} saved benchmark records.")
    for name, details in report["subgroups"].items():
        print(
            f"{name}: {details['unique_record_count']} unique messages, "
            f"{details['prediction_count']} predictions"
        )
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
