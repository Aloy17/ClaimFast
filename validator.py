import json
import re
from datetime import datetime

CLAIM_TYPES = {
    "own_damage",
    "third_party_liability",
    "theft",
    "total_loss",
    "glass_only",
    "not_a_claim",
}
URGENCIES = {"high", "medium", "low"}
FIELDS = [
    "claim_type",
    "urgency",
    "policy_number",
    "incident_date",
    "location",
    "vehicle_reg_no",
    "estimated_amount_inr",
    "injury_reported",
    "police_report_filed",
]


def parse_response(text):
    """Find the JSON object in the model's reply. Returns (data, error_message)."""
    if text is None or not text.strip():
        return None, (
            "The model returned an empty response. "
            "It may have run out of output tokens. Try again."
        )

    start = text.find("{")
    if start == -1:
        return None, "No JSON object found in the model's response."

    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as error:
        return None, "The model's JSON is malformed: " + str(error)

    if not isinstance(data, dict):
        return None, "The model returned JSON, but it is not an object."
    return data, None


def parse_strategy_response(text, strategy):
    """Parse a strategy response, extracting the tagged answer for Strategy C."""
    if strategy != "C":
        return parse_response(text)

    if text is None or not text.strip():
        return parse_response(text)

    pattern = (
        r"\s*<reasoning>\s*(.*?)\s*</reasoning>"
        r"\s*<answer>\s*(.*?)\s*</answer>\s*"
    )

    match = re.fullmatch(pattern, text, flags=re.DOTALL)

    if not match:
        return None, (
            "Strategy C response must contain exactly one <reasoning> "
            "block followed by exactly one <answer> block, "
            "with no text outside them."
        )

    reasoning, answer = match.groups()

    if not reasoning.strip():
        return None, "Strategy C reasoning block is empty."

    if not answer.strip():
        return None, "Strategy C answer block is empty."

    answer = answer.strip()
    try:
        data, end = json.JSONDecoder().raw_decode(answer)
    except json.JSONDecodeError as error:
        return None, "Strategy C answer JSON is malformed: " + str(error)

    if not isinstance(data, dict):
        return None, "Strategy C answer must be a JSON object."

    if answer[end:].strip():
        return None, (
            "Strategy C answer block must contain only one JSON object "
            "with no surrounding text."
        )

    return data, None


def validate_claim(data, claim_text=None):
    """Validate extracted fields and urgency against the original claim when supplied."""
    problems = []

    for field in FIELDS:
        if field not in data:
            problems.append("Missing field: " + field)
    for key in data:
        if key not in FIELDS:
            problems.append("Unexpected field: " + str(key))

    if "claim_type" in data and data["claim_type"] not in CLAIM_TYPES:
        problems.append("claim_type is not an allowed value: " + repr(data["claim_type"]))
    if "urgency" in data and data["urgency"] not in URGENCIES:
        problems.append("urgency is not an allowed value: " + repr(data["urgency"]))

    for field in ("policy_number", "location"):
        value = data.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            problems.append(field + " must be a non-empty string or null")

    date_value = data.get("incident_date")
    if date_value is not None:
        try:
            datetime.strptime(date_value, "%Y-%m-%d")
        except (TypeError, ValueError):
            problems.append("incident_date must be a valid YYYY-MM-DD date or null")

    reg = data.get("vehicle_reg_no")
    if reg is not None and (not isinstance(reg, str) or not re.fullmatch(r"[A-Z0-9]+", reg)):
        problems.append("vehicle_reg_no must be uppercase letters and digits only, or null")

    amount = data.get("estimated_amount_inr")
    if amount is not None and (isinstance(amount, bool) or not isinstance(amount, int)):
        problems.append("estimated_amount_inr must be an integer or null")

    for field in ("injury_reported", "police_report_filed"):
        value = data.get(field)
        if value is not None and not isinstance(value, bool):
            problems.append(field + " must be true, false or null")

    if data.get("injury_reported") is True and data.get("urgency") != "high":
        problems.append("injury_reported is true, so urgency should be high")

    unsafe_patterns = (
        r"\bundrivable\b",
        r"\bunroadworthy\b",
        r"\bunsafe to drive\b",
        r"\bnot safe to drive\b",
        r"\bcannot be driven\b",
        r"\bcan\x27t be driven\b",
        r"\bcannot drive\b",
        r"\bunable to drive\b",
        r"\bvehicle is immobilized\b",
    )

    if claim_text and data.get("urgency") != "high":
        text = claim_text.lower()
        if any(re.search(pattern, text) for pattern in unsafe_patterns):
            problems.append(
                "The claim describes an unsafe or undrivable vehicle, "
                "so urgency should be high"
            )

    return problems
