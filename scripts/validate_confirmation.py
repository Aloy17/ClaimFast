"""Validate the independent confirmation dataset without making provider calls."""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "confirmation.jsonl"
FIELDS = {
    "claim_type", "urgency", "policy_number", "incident_date", "location",
    "vehicle_reg_no", "estimated_amount_inr", "injury_reported",
    "police_report_filed",
}
CLAIM_TYPES = {
    "own_damage", "third_party_liability", "theft",
    "total_loss", "glass_only", "not_a_claim",
}
URGENCIES = {"high", "medium", "low"}
UNSAFE_PATTERNS = (
    r"\bundrivable\b", r"\bunroadworthy\b", r"\bunsafe to drive\b",
    r"\bnot safe to drive\b", r"\bcannot be driven\b",
    r"\bcan\x27t be driven\b", r"\bcannot drive\b",
    r"\bunable to drive\b", r"\bvehicle is immobilized\b",
    r"\bblocking (?:the )?road\b",
    r"\bstuck (?:in|on) the (?:middle of the )?road\b",
    r"\bcannot be moved\b", r"\bcan\x27t be moved\b",
)


def validate(path):
    records = []
    seen = set()
    errors = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON ({exc})")
                continue
            if not isinstance(record, dict) or set(record) != {"id", "message", "expected"}:
                errors.append(f"line {line_no}: expected exactly id, message, expected")
                continue
            rid, message, gold = record["id"], record["message"], record["expected"]
            if not isinstance(rid, str) or not rid.startswith("CONF") or not rid[4:].isdigit():
                errors.append(f"line {line_no}: id must use CONF### format")
            elif rid in seen:
                errors.append(f"{rid}: duplicate id")
            seen.add(rid)
            if not isinstance(message, str) or not message.strip():
                errors.append(f"{rid}: message must be non-empty text")
                continue
            if not isinstance(gold, dict) or set(gold) != FIELDS:
                errors.append(f"{rid}: expected must contain exactly the nine ClaimFast fields")
                continue
            if gold["claim_type"] not in CLAIM_TYPES:
                errors.append(f"{rid}: invalid claim_type")
            if gold["urgency"] not in URGENCIES:
                errors.append(f"{rid}: invalid urgency")
            for field in ("policy_number", "location"):
                value = gold[field]
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    errors.append(f"{rid}: {field} must be a non-empty string or null")
            date = gold["incident_date"]
            if date is not None:
                try:
                    if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                        raise ValueError
                    datetime.strptime(date, "%Y-%m-%d")
                except (TypeError, ValueError):
                    errors.append(f"{rid}: invalid incident_date")
            reg = gold["vehicle_reg_no"]
            if reg is not None and (not isinstance(reg, str) or not re.fullmatch(r"[A-Z0-9]+", reg)):
                errors.append(f"{rid}: invalid vehicle_reg_no")
            amount = gold["estimated_amount_inr"]
            if amount is not None and (isinstance(amount, bool) or not isinstance(amount, int) or amount < 0):
                errors.append(f"{rid}: estimated_amount_inr must be a non-negative integer or null")
            for field in ("injury_reported", "police_report_filed"):
                if gold[field] is not None and not isinstance(gold[field], bool):
                    errors.append(f"{rid}: {field} must be true, false or null")
            if gold["injury_reported"] is True and gold["urgency"] != "high":
                errors.append(f"{rid}: reported injury requires high urgency")
            if gold["urgency"] != "high" and any(re.search(p, message.lower()) for p in UNSAFE_PATTERNS):
                errors.append(f"{rid}: unsafe/undrivable vehicle requires high urgency")
            records.append(record)

    if not records:
        errors.append("dataset is empty")
    ids = [r["id"] for r in records]
    if len(records) != 24:
        errors.append(f"expected 24 records, found {len(records)}")
    if set(ids) != {f"CONF{i:03d}" for i in range(1, 25)}:
        errors.append("IDs must be exactly CONF001 through CONF024")
    counts = Counter(r["expected"]["claim_type"] for r in records)
    if counts != Counter({label: 4 for label in CLAIM_TYPES}):
        errors.append(f"class balance must be four per class; found {dict(counts)}")
    if errors:
        raise SystemExit("\n".join(errors))
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    print("Confirmation dataset validation: PASS")
    print(f"Records: {len(records)}")
    print("Class balance: " + ", ".join(f"{k}={counts[k]}" for k in sorted(CLAIM_TYPES)))
    print(f"SHA-256: {digest}")
    print("No provider/API calls were made.")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    validate(parser.parse_args().dataset)
