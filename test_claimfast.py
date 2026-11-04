import json
from pathlib import Path

import unittest
from unittest.mock import patch

import groq
import httpx

import main
from prompts import (
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLE_IDS,
    build_zero_shot_messages,
    build_few_shot_messages,
    build_reasoning_guided_messages,
    build_strategy_messages,
    PROMPT_STRATEGIES,
)

from validator import FIELDS, is_format_compliant, parse_response, parse_strategy_response, validate_claim

# Test fixture: a well formed reply, written by hand for these tests only.
GOOD_REPLY = (
    '{"claim_type": "third_party_liability", "urgency": "high", '
    '"policy_number": "MOT-2291847", "incident_date": null, '
    '"location": "Andheri flyover, Mumbai", "vehicle_reg_no": "MH02AB1234", '
    '"estimated_amount_inr": 45000, "injury_reported": true, '
    '"police_report_filed": null}'
)


def fake_result(text):
    return {
        "text": text,
        "latency_seconds": 0.5,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }


def http_error(cls, status):
    request = httpx.Request("POST", "https://api.groq.com/test")
    response = httpx.Response(status, request=request)
    return cls("test error", response=response, body=None)


class ParseTests(unittest.TestCase):
    def test_strict_format_compliance_accepts_bare_json_with_exact_schema(self):
        self.assertTrue(is_format_compliant(GOOD_REPLY, "A"))
        self.assertTrue(is_format_compliant(GOOD_REPLY, "B1"))

    def test_strict_format_compliance_rejects_preamble_trailing_text_and_code_fences(self):
        for response in (
            "Here is the result: " + GOOD_REPLY,
            GOOD_REPLY + "\\nExtra text",
            "```json\\n" + GOOD_REPLY + "\\n```",
        ):
            with self.subTest(response=response):
                self.assertFalse(is_format_compliant(response, "A"))

    def test_strict_format_compliance_requires_exact_schema(self):
        incomplete = json.dumps({"claim_type": "theft"})
        self.assertFalse(is_format_compliant(incomplete, "A"))

    def test_strategy_c_format_compliance_requires_exact_tagged_blocks(self):
        valid = (
            "<reasoning>Reported collision; no injury mentioned.</reasoning>\\n"
            "<answer>" + GOOD_REPLY + "</answer>"
        )
        self.assertTrue(is_format_compliant(valid, "C"))
        self.assertFalse(is_format_compliant(GOOD_REPLY, "C"))
        self.assertFalse(is_format_compliant(
            valid + "Extra text", "C"
        ))

    def test_format_compliance_rejects_nonstandard_json_constants(self):
        invalid = GOOD_REPLY.replace('"estimated_amount_inr": 45000', '"estimated_amount_inr": NaN')
        self.assertFalse(is_format_compliant(invalid, "A"))

    def test_plain_json(self):
        data, error = parse_response(GOOD_REPLY)
        self.assertIsNone(error)
        self.assertEqual(data["claim_type"], "third_party_liability")

    def test_json_in_code_fence_with_extra_text(self):
        text = "Here you go:\n```json\n" + GOOD_REPLY + "\n```\nDone."
        data, error = parse_response(text)
        self.assertIsNone(error)
        self.assertEqual(data["estimated_amount_inr"], 45000)

    def test_malformed_json(self):
        data, error = parse_response('{"claim_type": "theft", "urgency": ')
        self.assertIsNone(data)
        self.assertIn("malformed", error)

    def test_no_json_at_all(self):
        data, error = parse_response("I cannot help with that.")
        self.assertIsNone(data)
        self.assertIn("No JSON", error)

    def test_empty_and_none(self):
        for text in ("", "   ", None):
            data, error = parse_response(text)
            self.assertIsNone(data)
            self.assertIn("empty", error)

    def test_strategy_c_parses_only_answer_block(self):
        response = (
            "<reasoning>Reported collision; no injury mentioned.</reasoning>\n"
            "<answer>" + GOOD_REPLY + "</answer>"
        )

        data, error = parse_strategy_response(response, "C")

        self.assertIsNone(error)
        self.assertEqual(data["claim_type"], "third_party_liability")

    def test_strategy_c_rejects_missing_answer_tags(self):
        response = (
            "<reasoning>Reported collision.</reasoning>\\n"
            + GOOD_REPLY
        )

        data, error = parse_strategy_response(response, "C")

        self.assertIsNone(data)
        self.assertIn("<answer>", error)

    def test_strategy_c_rejects_text_outside_tagged_blocks(self):
        response = (
            "Here is the result.\\n"
            "<reasoning>Reported collision.</reasoning>"
            "<answer>" + GOOD_REPLY + "</answer>"
        )

        data, error = parse_strategy_response(response, "C")

        self.assertIsNone(data)
        self.assertIn("no text outside", error)

    def test_strategy_c_rejects_empty_reasoning(self):
        response = (
            "<reasoning> </reasoning>"
            "<answer>" + GOOD_REPLY + "</answer>"
        )

        data, error = parse_strategy_response(response, "C")

        self.assertIsNone(data)
        self.assertIn("reasoning block is empty", error)

    def test_strategy_c_rejects_malformed_answer_json(self):
        response = (
            "<reasoning>Reported collision.</reasoning>"
            "<answer>{invalid json}</answer>"
        )

        data, error = parse_strategy_response(response, "C")

        self.assertIsNone(data)
        self.assertIn("malformed", error)

    def test_strategy_c_rejects_preamble_inside_answer(self):
        response = (
            "<reasoning>Reported collision.</reasoning>"
            "<answer>Here is the result:\\n" + GOOD_REPLY + "</answer>"
        )

        data, error = parse_strategy_response(response, "C")

        self.assertIsNone(data)
        self.assertIn("malformed", error)

    def test_strategy_c_rejects_trailing_text_inside_answer(self):
        response = (
            "<reasoning>Reported collision.</reasoning>"
            "<answer>" + GOOD_REPLY + "\\nExtra text</answer>"
        )

        data, error = parse_strategy_response(response, "C")

        self.assertIsNone(data)
        self.assertIn("only one JSON object", error)

    def test_strategy_c_rejects_multiple_json_objects(self):
        response = (
            "<reasoning>Reported collision.</reasoning>"
            "<answer>" + GOOD_REPLY + "\\n" + GOOD_REPLY + "</answer>"
        )

        data, error = parse_strategy_response(response, "C")

        self.assertIsNone(data)
        self.assertIn("only one JSON object", error)

    def test_strategy_c_accepts_whitespace_around_answer_json(self):
        response = (
            "<reasoning>Reported collision.</reasoning>"
            "<answer>\n  " + GOOD_REPLY + "  \n</answer>"
        )

        data, error = parse_strategy_response(response, "C")

        self.assertIsNone(error)
        self.assertEqual(data["claim_type"], "third_party_liability")

    def test_other_strategies_keep_existing_parser(self):
        data, error = parse_strategy_response(GOOD_REPLY, "A")

        self.assertIsNone(error)
        self.assertEqual(data["claim_type"], "third_party_liability")


    def test_strategy_a_keeps_permissive_json_extraction(self):
        response = "Here is the result:\\n" + GOOD_REPLY + "\\nExtra text"

        data, error = parse_strategy_response(response, "A")

        self.assertIsNone(error)
        self.assertEqual(data["claim_type"], "third_party_liability")


class ValidateTests(unittest.TestCase):
    def good(self):
        data, _ = parse_response(GOOD_REPLY)
        return data

    def test_good_reply_has_no_problems(self):
        self.assertEqual(validate_claim(self.good()), [])

    def test_missing_fields_reported(self):
        data = self.good()
        del data["urgency"]
        del data["location"]
        problems = validate_claim(data)
        self.assertIn("Missing field: urgency", problems)
        self.assertIn("Missing field: location", problems)

    def test_unexpected_field_reported(self):
        data = self.good()
        data["notes"] = "extra"
        self.assertIn("Unexpected field: notes", validate_claim(data))

    def test_bad_values_reported(self):
        data = self.good()
        data["claim_type"] = "fire"
        data["incident_date"] = "yesterday"
        data["vehicle_reg_no"] = "MH 02 AB 1234"
        data["estimated_amount_inr"] = "45000"
        data["police_report_filed"] = "no"
        text = " ".join(validate_claim(data))
        for name in ("claim_type", "incident_date", "vehicle_reg_no",
                     "estimated_amount_inr", "police_report_filed"):
            self.assertIn(name, text)

    def test_boolean_is_not_an_integer_amount(self):
        data = self.good()
        data["estimated_amount_inr"] = True
        self.assertTrue(validate_claim(data))

    def test_negative_amount_is_rejected(self):
        data = self.good()
        data["estimated_amount_inr"] = -1
        self.assertIn(
            "estimated_amount_inr must be a non-negative integer or null",
            validate_claim(data),
        )

    def test_incident_date_requires_exact_iso_format(self):
        for value in ("2026-1-01", "2026-01-1", "2026-02-30", 20261104):
            with self.subTest(value=value):
                data = self.good()
                data["incident_date"] = value
                self.assertIn(
                    "incident_date must be a valid YYYY-MM-DD date or null",
                    validate_claim(data),
                )

    def test_unhashable_enum_values_are_reported_without_crashing(self):
        data = self.good()
        data["claim_type"] = ["theft"]
        data["urgency"] = {"level": "high"}
        problems = validate_claim(data)
        self.assertTrue(any(problem.startswith("claim_type") for problem in problems))
        self.assertTrue(any(problem.startswith("urgency") for problem in problems))

    def test_non_object_claim_data_is_reported(self):
        self.assertEqual(
            validate_claim(["not", "an", "object"]),
            ["Claim data must be a JSON object"],
        )

    def test_false_and_null_are_both_allowed(self):
        data = self.good()
        data["police_report_filed"] = False
        self.assertEqual(validate_claim(data), [])
        data["police_report_filed"] = None
        self.assertEqual(validate_claim(data), [])

    def test_unsafe_vehicle_with_medium_urgency_is_flagged(self):
        data = self.good()
        data["injury_reported"] = False
        data["urgency"] = "medium"

        problems = validate_claim(
            data,
            "My vehicle is not safe to drive."
        )

        self.assertIn("urgency should be high", " ".join(problems))

    def test_undrivable_vehicle_with_low_urgency_is_flagged(self):
        data = self.good()
        data["injury_reported"] = False
        data["urgency"] = "low"

        problems = validate_claim(
            data,
            "The vehicle cannot be driven after the accident."
        )

        self.assertIn("urgency should be high", " ".join(problems))

    def test_drivable_vehicle_does_not_trigger_unsafe_rule(self):
        data = self.good()
        data["injury_reported"] = False
        data["urgency"] = "medium"

        problems = validate_claim(
            data,
            "The car is damaged but can still be driven."
        )

        self.assertEqual(problems, [])

    def test_injury_with_lower_urgency_flagged(self):
        data = self.good()
        data["urgency"] = "medium"
        self.assertIn("urgency should be high", " ".join(validate_claim(data)))


class PromptTests(unittest.TestCase):
    def test_zero_shot_has_no_examples(self):
        messages = build_zero_shot_messages("hello")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "hello")

    def test_prompt_names_every_field(self):
        for field in FIELDS:
            self.assertIn(field, SYSTEM_PROMPT)

    def test_prompt_distinguishes_third_party_harm_from_involvement(self):
        self.assertIn(
            "Another vehicle being involved does not by itself",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "a person other than the policyholder is injured",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "This includes an injured passenger in the insured vehicle",
            SYSTEM_PROMPT,
        )

    def test_shared_prompt_covers_suspected_theft_in_fnol_context(self):
        self.assertIn("suspected theft", SYSTEM_PROMPT)
        self.assertIn("insurance claim or FNOL context", SYSTEM_PROMPT)
        from prompts import REASONING_GUIDANCE
        self.assertIn("reported missing", REASONING_GUIDANCE.lower())
        self.assertIn("suspected theft", REASONING_GUIDANCE.lower())

    def test_total_loss_does_not_automatically_set_high_urgency(self):
        from prompts import REASONING_GUIDANCE
        self.assertIn("total_loss alone does not imply high urgency", SYSTEM_PROMPT)
        self.assertIn("Do not set urgency to high solely because the claim is total_loss", REASONING_GUIDANCE)
        self.assertIn("Determine urgency independently", SYSTEM_PROMPT)

    def test_shared_prompt_applies_total_loss_precedence(self):
        self.assertIn("total_loss is the primary claim_type", SYSTEM_PROMPT)
        self.assertIn("even if third-party harm is also reported", SYSTEM_PROMPT)

    def test_all_strategies_inherit_shared_claim_type_rules(self):
        for strategy in ("A", "B1", "B2", "B3", "C"):
            with self.subTest(strategy=strategy):
                prompt = build_strategy_messages(strategy, "Vehicle missing after FNOL.")
                self.assertIn("suspected theft", prompt[0]["content"].lower())
                self.assertIn("total_loss is the primary claim_type", prompt[0]["content"])

    def test_reasoning_guidance_checks_third_party_harm(self):
        from prompts import REASONING_GUIDANCE
        self.assertIn("mere involvement", REASONING_GUIDANCE)
        self.assertIn(
            "injury to a person other than the policyholder",
            REASONING_GUIDANCE,
        )
        self.assertIn("including an insured-vehicle passenger", REASONING_GUIDANCE)
        self.assertIn("use total_loss as the primary claim_type", REASONING_GUIDANCE)


class DatasetLabellingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_dir = Path(__file__).resolve().parent / "data"
        cls.records = {}

        for filename in ("dev.jsonl", "test.jsonl"):
            with (data_dir / filename).open("r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        record = json.loads(line)
                        cls.records[record["id"]] = record

    def test_injured_passengers_are_third_party_liability(self):
        for record_id in ("DEV019", "TEST039", "TEST019"):
            with self.subTest(record_id=record_id):
                expected = self.records[record_id]["expected"]
                self.assertEqual(expected["claim_type"], "third_party_liability")
                self.assertTrue(expected["injury_reported"])
                self.assertEqual(expected["urgency"], "high")

    def test_total_loss_without_injury_or_unsafe_vehicle_can_be_medium(self):
        medium_total_loss_ids = (
            "DEV004", "DEV010", "DEV020", "TEST004", "TEST010",
            "TEST020", "TEST024", "TEST030", "TEST040", "TEST048", "TEST060",
        )
        for record_id in medium_total_loss_ids:
            with self.subTest(record_id=record_id):
                expected = self.records[record_id]["expected"]
                self.assertEqual(expected["claim_type"], "total_loss")
                self.assertEqual(expected["urgency"], "medium")
                self.assertFalse(expected["injury_reported"])

    def test_total_loss_remains_primary_when_vehicle_is_beyond_repair(self):
        expected = self.records["TEST054"]["expected"]
        self.assertEqual(expected["claim_type"], "total_loss")
        self.assertTrue(expected["injury_reported"])
        self.assertEqual(expected["urgency"], "high")


class AssessClaimTests(unittest.TestCase):
    def test_success_records_usage(self):
        with patch("main.ask_model", return_value=fake_result(GOOD_REPLY)) as ask:
            outcome = main.assess_claim("some claim")
        self.assertIsNone(outcome["error"])
        self.assertEqual(outcome["problems"], [])
        self.assertEqual(outcome["latency_seconds"], 0.5)
        self.assertEqual(outcome["total_tokens"], 150)
        self.assertEqual(len(ask.call_args.args), 1)
        self.assertEqual(ask.call_args.kwargs, {})

    def test_malformed_json_is_handled(self):
        with patch("main.ask_model", return_value=fake_result('{"claim_type": ')):
            outcome = main.assess_claim("x")
        self.assertIn("malformed", outcome["error"])
        self.assertIsNone(outcome["data"])
        self.assertEqual(outcome["total_tokens"], 150)

    def test_missing_fields_are_handled(self):
        with patch("main.ask_model", return_value=fake_result('{"claim_type": "theft"}')):
            outcome = main.assess_claim("x")
        self.assertIsNone(outcome["error"])
        self.assertIn("Missing field: urgency", outcome["problems"])

    def test_api_errors_do_not_crash(self):
        cases = [
            (http_error(groq.AuthenticationError, 401), "rejected the API key"),
            (http_error(groq.RateLimitError, 429), "rate limit"),
            (groq.APIConnectionError(request=httpx.Request("POST", "https://x")), "Could not reach"),
            (http_error(groq.InternalServerError, 500), "HTTP 500"),
            (RuntimeError("GROQ_API_KEY is missing. Check your .env file."), "GROQ_API_KEY"),
            (ValueError("boom"), "Unexpected error"),
        ]
        for exception, expected in cases:
            with patch("main.ask_model", side_effect=exception):
                outcome = main.assess_claim("x")
            self.assertIn(expected, outcome["error"])
            self.assertIsNone(outcome["data"])

    def test_show_outcome_never_crashes(self):
        with patch("main.ask_model", return_value=fake_result(GOOD_REPLY)):
            main.show_outcome(main.assess_claim("x"))
        with patch("main.ask_model", return_value=fake_result("")):
            main.show_outcome(main.assess_claim("x"))
        with patch("main.ask_model", side_effect=RuntimeError("no key")):
            main.show_outcome(main.assess_claim("x"))

class FewShotPromptTests(unittest.TestCase):

    def setUp(self):
        dev_path = Path(__file__).resolve().parent / "data" / "dev.jsonl"

        with dev_path.open("r", encoding="utf-8") as file:
            self.dev_records = {
                record["id"]: record
                for record in (json.loads(line) for line in file if line.strip())
            }

    def test_supported_k_values(self):
        for k in (3, 5, 8):
            with self.subTest(k=k):
                messages = build_few_shot_messages("test claim", k)
                self.assertEqual(len(messages), 2 * k + 2)

    def test_invalid_k_is_rejected(self):
        for k in (0, 1, 2, 4, 6, 7, 9, -1):
            with self.subTest(k=k):
                with self.assertRaises(ValueError):
                    build_few_shot_messages("test claim", k)

    def test_system_message_is_first(self):
        messages = build_few_shot_messages("test claim", 3)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)

    def test_final_message_contains_original_claim(self):
        claim = "My car was damaged yesterday."
        messages = build_few_shot_messages(claim, 5)

        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], claim)

    def test_examples_follow_configured_order(self):
        for k in (3, 5, 8):
            with self.subTest(k=k):
                messages = build_few_shot_messages("test claim", k)
                example_ids = FEW_SHOT_EXAMPLE_IDS[:k]

                for i, example_id in enumerate(example_ids):
                    user_message = messages[1 + 2 * i]
                    assistant_message = messages[2 + 2 * i]

                    expected = self.dev_records[example_id]

                    self.assertEqual(user_message["role"], "user")
                    self.assertEqual(
                        user_message["content"],
                        expected["message"]
                    )

                    self.assertEqual(assistant_message["role"], "assistant")
                    self.assertEqual(
                        json.loads(assistant_message["content"]),
                        expected["expected"]
                    )

    def test_all_examples_come_from_development_dataset(self):
        for example_id in FEW_SHOT_EXAMPLE_IDS:
            self.assertIn(example_id, self.dev_records)

    def test_assistant_examples_are_valid_json(self):
        for k in (3, 5, 8):
            messages = build_few_shot_messages("test claim", k)

            for message in messages:
                if message["role"] == "assistant":
                    data = json.loads(message["content"])
                    self.assertEqual(set(data.keys()), set(FIELDS))



class StrategyDispatchTests(unittest.TestCase):
    def test_all_five_strategies_are_configured(self):
        self.assertEqual(
            set(PROMPT_STRATEGIES),
            {"A", "B1", "B2", "B3", "C"},
        )

    def test_strategy_a_uses_zero_shot_builder(self):
        claim = "My car was damaged yesterday."
        self.assertEqual(
            build_strategy_messages("A", claim),
            build_zero_shot_messages(claim),
        )

    def test_few_shot_strategies_use_configured_example_counts(self):
        for strategy, k in (("B1", 3), ("B2", 5), ("B3", 8)):
            with self.subTest(strategy=strategy):
                messages = build_strategy_messages(
                    strategy,
                    "My car was damaged yesterday.",
                )
                self.assertEqual(len(messages), 2 * k + 2)
                self.assertEqual(messages[-1]["content"],
                                 "My car was damaged yesterday.")

    def test_strategy_c_uses_reasoning_guidance(self):
        claim = "My car was damaged yesterday."
        messages = build_reasoning_guided_messages(claim)

        self.assertEqual(
            messages,
            build_strategy_messages("C", claim),
        )
        self.assertEqual([m["role"] for m in messages], ["system", "user"])

        system_prompt = messages[0]["content"]

        self.assertIn("<reasoning>", system_prompt)
        self.assertIn("</reasoning>", system_prompt)
        self.assertIn("<answer>", system_prompt)
        self.assertIn("</answer>", system_prompt)
        self.assertIn("exactly the nine specified keys", system_prompt)
        self.assertNotIn(
            "JSON object and nothing else. No markdown",
            system_prompt,
        )
        self.assertEqual(messages[-1]["content"], claim)

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            build_strategy_messages("D", "test claim")


if __name__ == "__main__":
    unittest.main()
