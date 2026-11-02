# ClaimFast — Dataset Labelling Guide

## 1. Purpose

This guide defines the labelling rules for ClaimFast's motor-insurance First Notice of Loss (FNOL) messages. Each message must be labelled consistently using the nine required output fields.

## 2. Required fields

| Field | Allowed values / format |
|---|---|
| `claim_type` | `own_damage`, `third_party_liability`, `theft`, `total_loss`, `glass_only`, `not_a_claim` |
| `urgency` | `high`, `medium`, `low` |
| `policy_number` | String or null |
| `incident_date` | Exact date in `YYYY-MM-DD` format or null |
| `location` | String or null |
| `vehicle_reg_no` | Uppercase registration number without spaces or hyphens, or null |
| `estimated_amount_inr` | Integer or null |
| `injury_reported` | `true`, `false`, or null |
| `police_report_filed` | `true`, `false`, or null |

## 3. Claim type definitions

- **own_damage:** Damage to the policyholder's insured vehicle, with no reported third-party injury or damage that may create liability for the policyholder. Another vehicle being involved does not, by itself, make the claim third-party liability.
- **third_party_liability:** Another person's vehicle or property is damaged, or a person other than the policyholder is injured, in an incident involving the insured vehicle. This includes an injured passenger in the insured vehicle. The assignment's 'may be liable' condition does not require proven fault or an admission of liability. Do not infer injury or property damage that the message does not report.
- **theft:** The vehicle or one of its parts has been stolen.
- **total_loss:** The vehicle is destroyed or damaged beyond repair.
- **glass_only:** Damage is limited to glass, such as a windshield or window.
- **not_a_claim:** The message is not an insurance claim or FNOL.

### Claim-type distinction rules

- If damage is reported only to the insured vehicle, with no reported third-party injury or property damage, classify it as `own_damage`, even when another vehicle caused the damage or left the scene.
- A person other than the policyholder is a third party for this labelling task. This includes a passenger in the insured vehicle, another vehicle's driver or passenger, and a pedestrian.
- If another person is injured, or another person's vehicle or property is damaged, in an incident involving the insured vehicle, classify it as `third_party_liability`. The assignment says the policyholder may be liable; proven fault or an admission of liability is not required.
- Do not classify a claim as `third_party_liability` merely because another vehicle or driver is mentioned. Actual third-party injury or property damage must be reported.
- Do not infer fault, injury, or damage that is not stated. An injury to the policyholder alone does not establish third-party liability.
- **Single-label overlap:** If the insured vehicle is destroyed or beyond repair, use `total_loss` as the primary `claim_type`, even if third-party harm is also reported. Capture any injury in `injury_reported` and set `urgency` to `high`.

## 4. Urgency rules

- **high:** Any injury is reported, or the vehicle is undrivable or unsafe.
- **medium:** A genuine loss requires attention, but no injury or unsafe/undrivable vehicle is reported.
- **low:** Minor or non-urgent matters, such as glass-only damage or a non-claim message.

**Important:** Any mention of an injury makes urgency `high`, regardless of the claim type or other circumstances.

## 5. Missing and ambiguous information

- Use `null` when a field is not stated or cannot be determined confidently.
- Do not invent policy numbers, dates, locations, registration numbers, or estimated amounts.
- Use `true` for an explicitly reported condition.
- Use `false` only when the message clearly denies that condition.
- Use `null` when injury or police-report status is not mentioned.
- Convert dates to `YYYY-MM-DD` only when the exact date is clear.
- Relative dates such as "yesterday", "last week", or "two days ago" must be labelled `null`.
- Estimated amounts must be integers in INR when explicitly available.

## 6. Special cases

- **Multiple incidents:** Label the incident most directly associated with the reported claim. Do not combine details from separate incidents.
- **Forwarded messages:** Use relevant information from the complete message chain, but do not mistake quoted or unrelated correspondence for a new incident.
- **Hinglish and spelling errors:** Interpret the intended meaning where it is reasonably clear. Do not guess when meaning is ambiguous.
- **Uncertain theft:** Use `theft` when a customer reports a vehicle or its part as missing in an insurance claim or FNOL context. Treat this as suspected theft even when theft has not been confirmed. Do not infer that a police report or FIR has been filed unless explicitly stated. A mere mention of a missing vehicle outside an insurance claim or FNOL context is not sufficient to classify it as theft. DEV009, TEST029, and TEST047 are examples of suspected or ambiguous theft and should be flagged for review.

## 7. Ambiguous Label Review

The following cases were reviewed for consistency with the claim-type and urgency definitions:

| Record | Decision | Rationale |
|---|---|---|
| DEV009 | Retained as theft; flagged ambiguous | Insured vehicle is reported missing in an FNOL context; theft is suspected but unconfirmed. |
| TEST009 | Retained as theft | Customer explicitly suspects vehicle theft. |
| TEST027 | Corrected to own_damage | Only the insured vehicle's damage is reported. |
| TEST029 | Retained as theft; flagged ambiguous | Vehicle is missing, but theft is not yet confirmed. |
| TEST041 | Corrected to own_damage | Only damage to the insured vehicle is reported. |
| TEST046 | Retained as third_party_liability | Both the insured vehicle and another party's vehicle are damaged. |
| TEST047 | Retained as theft; flagged borderline | Vehicle is missing and security cannot locate it, but formal theft confirmation is unknown. |
| TEST058 | Retained as third_party_liability; amount corrected to null | Another driver was injured. The stated repair estimate belongs to an earlier, separate windshield incident. |
| DEV019 | Corrected to third_party_liability | A passenger other than the policyholder is reported injured in an incident involving the insured vehicle. |
| TEST039 | Corrected to third_party_liability | A passenger other than the policyholder is reported injured in an incident involving the insured vehicle. |
| TEST054 | Retained as total_loss | The insured vehicle is burnt beyond repair. The guard's injury remains captured separately and makes urgency high. |

These decisions were made to improve consistency between the messages, expected labels and labelling guide. Borderline cases remain useful for evaluating model behaviour under uncertainty.