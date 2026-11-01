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


def validate_claim(data):
    """Check the parsed JSON against the claim rules. Returns a list of problems."""
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

    return problems
