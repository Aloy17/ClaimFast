import unittest
from unittest.mock import patch

import groq
import httpx

import main
from prompts import SYSTEM_PROMPT, build_zero_shot_messages
from validator import FIELDS, parse_response, validate_claim

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

    def test_false_and_null_are_both_allowed(self):
        data = self.good()
        data["police_report_filed"] = False
        self.assertEqual(validate_claim(data), [])
        data["police_report_filed"] = None
        self.assertEqual(validate_claim(data), [])

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


if __name__ == "__main__":
    unittest.main()
