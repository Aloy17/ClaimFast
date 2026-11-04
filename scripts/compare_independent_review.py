"""Compare a completed blinded review worksheet with the gold dataset.

Run only after an independent reviewer has completed the worksheet.
This script does not modify either input file.
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "claim_type",
    "urgency",
    "policy_number",
    "incident_date",
    "location",
    "vehicle_reg_no",
    "estimated_amount_inr",
    "injury_reported",
    "police_report_filed",
)
BOOLEAN_FIELDS = {"injury_reported", "police_report_filed"}


def load_gold():
    records = {}
    for path in (ROOT / "data" / "dev.jsonl", ROOT / "data" / "test.jsonl"):
        with path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    record = json.loads(line)
                    records[record["id"]] = record["expected"]
    return records


def parse_value(field, raw, record_id):
    value = (raw or "").strip()
    if not value:
        raise ValueError(
            f"{record_id}.{field}: field is blank; enter the literal null "
            "when the reviewer determines the value is not stated"
        )
    if value.lower() == "null":
        return None
    if field in BOOLEAN_FIELDS:
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        raise ValueError(f"{record_id}.{field}: enter true, false, or null")
    if field == "estimated_amount_inr":
        try:
            return int(value.replace(",", ""))
        except ValueError as error:
            raise ValueError(
                f"{record_id}.{field}: enter an integer or null"
            ) from error
    return value


def compare(worksheet_path):
    gold = load_gold()
    compared = []
    seen = set()
    with Path(worksheet_path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"record_id"} | {"reviewer_" + field for field in FIELDS}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Worksheet is missing columns: " + ", ".join(sorted(missing))
            )
        for row in reader:
            record_id = (row.get("record_id") or "").strip()
            if not record_id:
                continue
            if record_id in seen:
                raise ValueError(f"Duplicate review record: {record_id}")
            if record_id not in gold:
                raise ValueError(f"Unknown record ID: {record_id}")
            seen.add(record_id)
            prediction = {
                field: parse_value(
                    field, row.get("reviewer_" + field), record_id
                )
                for field in FIELDS
            }
            compared.append((record_id, prediction, gold[record_id]))

    if not compared:
        raise ValueError("No completed review records found")

    print(f"Reviewed records: {len(compared)}")
    field_matches = {field: 0 for field in FIELDS}
    exact_matches = 0
    for _, prediction, expected in compared:
        for field in FIELDS:
            field_matches[field] += prediction[field] == expected[field]
        exact_matches += prediction == expected

    total_fields = len(compared) * len(FIELDS)
    total_matches = sum(field_matches.values())
    print(
        f"Overall field agreement: {total_matches}/{total_fields} "
        f"({total_matches / total_fields:.1%})"
    )
    print(
        f"Exact-record agreement: {exact_matches}/{len(compared)} "
        f"({exact_matches / len(compared):.1%})"
    )
    print("\nPer-field agreement:")
    for field in FIELDS:
        matches = field_matches[field]
        print(
            f"- {field}: {matches}/{len(compared)} "
            f"({matches / len(compared):.1%})"
        )

    disagreements = []
    for record_id, prediction, expected in compared:
        for field in FIELDS:
            if prediction[field] != expected[field]:
                disagreements.append(
                    (record_id, field, prediction[field], expected[field])
                )
    if disagreements:
        print("\nDisagreements (reviewer vs gold):")
        for record_id, field, reviewer_value, gold_value in disagreements:
            print(
                f"- {record_id}.{field}: reviewer={reviewer_value!r}, "
                f"gold={gold_value!r}"
            )
    else:
        print("\nNo field-level disagreements.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "worksheet",
        nargs="?",
        default=str(ROOT / "data" / "independent_review_template.csv"),
        help="Completed independent-review CSV worksheet",
    )
    args = parser.parse_args()
    compare(args.worksheet)


if __name__ == "__main__":
    main()
