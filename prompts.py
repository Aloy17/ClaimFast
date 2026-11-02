import json
from pathlib import Path

SYSTEM_PROMPT = """You process motor insurance claim messages from customers in India. Read the customer's message, classify the claim, and extract the fields below.

claim_type (choose exactly one):
- own_damage: only the policyholder's own vehicle is damaged and no third party is involved.
- third_party_liability: another person, vehicle or property was injured or damaged and the policyholder may be liable.
- theft: the vehicle or its parts were stolen.
- total_loss: the vehicle is destroyed or damaged beyond repair.
- glass_only: only glass (windshield, windows) is damaged.
- not_a_claim: the message is not an insurance claim, for example a question or a greeting.

urgency (choose exactly one):
- high: any injury is reported, or the vehicle is undrivable or unsafe to drive.
- medium: a real loss that needs prompt handling, with no injury and no statement that the vehicle is unsafe.
- low: minor or non urgent, such as glass only damage, or not a claim.

Extraction fields:
- policy_number: the policy number as written, or null.
- incident_date: YYYY-MM-DD only if a full exact date is stated. Relative words such as yesterday, today or last week are not exact dates, so use null.
- location: where the incident happened, as written, or null.
- vehicle_reg_no: uppercase with no spaces or hyphens, or null.
- estimated_amount_inr: an integer number of rupees with no commas or symbols, or null.
- injury_reported: true if any injury is mentioned, false only if the message clearly says nobody was hurt, null if injuries are not mentioned.
- police_report_filed: true if the message says a police report or FIR was filed, false only if it clearly says none was filed, null if not mentioned.

Rules:
- Never invent missing information. Use null for anything the message does not state.
- false and null are different: false means explicitly denied, null means not mentioned.
- For not_a_claim, urgency is low and every extraction field is null unless the message states it.

Output format:
Reply with one JSON object and nothing else. No markdown, no code fences, no explanation. The object must have exactly these keys:
{
  "claim_type": string,
  "urgency": string,
  "policy_number": string or null,
  "incident_date": string or null,
  "location": string or null,
  "vehicle_reg_no": string or null,
  "estimated_amount_inr": integer or null,
  "injury_reported": true, false or null,
  "police_report_filed": true, false or null
}"""


def build_zero_shot_messages(claim):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": claim},
    ]


FEW_SHOT_EXAMPLE_IDS = [
    "DEV001",
    "DEV008",
    "DEV003",
    "DEV010",
    "DEV005",
    "DEV006",
    "DEV014",
    "DEV017",
]


def build_few_shot_messages(claim, k):
    if k not in (3, 5, 8):
        raise ValueError("k must be 3, 5, or 8")

    dev_path = Path(__file__).resolve().parent / "data" / "dev.jsonl"

    records = {}

    with dev_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for example_id in FEW_SHOT_EXAMPLE_IDS[:k]:
        if example_id not in records:
            raise ValueError(f"Missing development example: {example_id}")

        example = records[example_id]

        messages.append({
            "role": "user",
            "content": example["message"]
        })

        messages.append({
            "role": "assistant",
            "content": json.dumps(
                example["expected"],
                ensure_ascii=False
            )
        })

    messages.append({
        "role": "user",
        "content": claim
    })

    return messages