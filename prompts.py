import json
from pathlib import Path

SYSTEM_PROMPT = """You process motor insurance claim messages from customers in India. Read the customer's message, classify the claim, and extract the fields below.

claim_type (choose exactly one):
- own_damage: the policyholder's insured vehicle is damaged, and no third-party injury or damage is reported that may create liability for the policyholder. Another vehicle being involved does not by itself make the claim third-party liability.
- third_party_liability: another person's vehicle or property is damaged, or a person other than the policyholder is injured, in an incident involving the insured vehicle. This includes an injured passenger in the insured vehicle. The assignment's 'may be liable' condition does not require proven fault or an admission of liability. Do not infer injury or property damage that the message does not report.
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


REASONING_GUIDANCE = """DECISION CHECKS:
1. Determine whether the message describes an insurance claim or FNOL.
2. Identify the type of loss using the claim_type definitions.
3. Determine urgency: injuries or an unsafe/undrivable vehicle require high urgency.
4. Extract each field only from information explicitly stated in the message.
5. Distinguish explicitly denied information (false) from information not mentioned (null).
6. Distinguish third-party harm from mere involvement: damage only to the insured vehicle is own_damage, even if another vehicle caused it. Reported injury to a person other than the policyholder, including an insured-vehicle passenger, or damage to another person's vehicle or property in an incident involving the insured vehicle indicates third_party_liability. Do not infer fault or harm.
7. If the insured vehicle is destroyed or beyond repair, use total_loss as the primary claim_type even if third-party harm is also reported. Capture injuries separately and set urgency to high.
8. Check that the selected claim type, urgency, and extracted fields are consistent.

Within the reasoning block, provide a concise, evidence-based decision
summary addressing these checks. Do not invent facts or add information
that is absent from the message."""


def build_reasoning_guided_messages(claim):
    messages = build_zero_shot_messages(claim)

    schema_start = SYSTEM_PROMPT.index(
        "The object must have exactly these keys:"
    )
    schema_definition = SYSTEM_PROMPT[
        schema_start + len("The object must have exactly these keys:"):
    ].strip()

    reasoning_system_prompt = (
        SYSTEM_PROMPT.split("Output format:", 1)[0]
        + "Output format:\n"
        + "Return exactly two tagged blocks in this order, with no text "
        + "outside them:\n"
        + "<reasoning>\n"
        + "A concise, evidence-based decision summary addressing the "
        + "reasoning checks.\n"
        + "</reasoning>\n"
        + "<answer>\n"
        + "The JSON object must have exactly the nine specified keys "
        + "and use these field names and types:\n"
        + schema_definition
        + "\n</answer>\n\n"
        + REASONING_GUIDANCE
    )

    messages[0]["content"] = reasoning_system_prompt
    return messages

PROMPT_STRATEGIES = {
    "A": ("zero_shot", None),
    "B1": ("few_shot", 3),
    "B2": ("few_shot", 5),
    "B3": ("few_shot", 8),
    "C": ("reasoning_guided", None),
}


def build_strategy_messages(strategy, claim):
    if strategy not in PROMPT_STRATEGIES:
        supported = ", ".join(PROMPT_STRATEGIES)
        raise ValueError(
            f"Unknown strategy: {strategy}. Choose one of: {supported}"
        )

    strategy_type, k = PROMPT_STRATEGIES[strategy]

    if strategy_type == "zero_shot":
        return build_zero_shot_messages(claim)

    if strategy_type == "few_shot":
        return build_few_shot_messages(claim, k)

    if strategy_type == "reasoning_guided":
        return build_reasoning_guided_messages(claim)

    raise ValueError(f"Unsupported strategy configuration: {strategy_type}")
