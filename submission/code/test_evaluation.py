import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation import DEFAULT_OUTPUT_DIR, load_dataset, run_evaluation
from validator import FIELDS


def valid_reply():
    return json.dumps({
        "claim_type": "own_damage",
        "urgency": "medium",
        "policy_number": None,
        "incident_date": None,
        "location": "Andheri",
        "vehicle_reg_no": "MH02AB1234",
        "estimated_amount_inr": 12000,
        "injury_reported": False,
        "police_report_filed": None,
    })


class EvaluationTests(unittest.TestCase):
    def test_default_output_dir_is_isolated_from_historical_logs(self):
        self.assertEqual(
            DEFAULT_OUTPUT_DIR,
            Path(__file__).resolve().parent / "outputs" / "final_benchmark",
        )

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_path = self.root / "test.jsonl"
        self.output_dir = self.root / "outputs"
        self.record = {
            "id": "TEST001",
            "message": "My car MH02AB1234 was damaged near Andheri. "
                       "No one was injured. Estimate Rs 12000.",
            "expected": {
                "claim_type": "own_damage",
                "urgency": "medium",
                "policy_number": None,
                "incident_date": None,
                "location": "Andheri",
                "vehicle_reg_no": "MH02AB1234",
                "estimated_amount_inr": 12000,
                "injury_reported": False,
                "police_report_filed": None,
            },
        }
        self.dataset_path.write_text(
            json.dumps(self.record) + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def fake_model_result(self, text=None):
        return {
            "text": text if text is not None else valid_reply(),
            "latency_seconds": 0.25,
            "prompt_tokens": 120,
            "completion_tokens": 60,
            "total_tokens": 180,
            "api_status_code": 200,
            "retry_after": None,
            "request_id": "req_success",
            "rate_limit_headers": {
                "x-ratelimit-remaining-tokens": "6200"
            },
        }

    def test_load_dataset_accepts_exact_schema(self):
        records = load_dataset(self.dataset_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "TEST001")

    def test_load_dataset_rejects_duplicate_ids(self):
        with self.dataset_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(self.record) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate record id"):
            load_dataset(self.dataset_path)

    def test_load_dataset_rejects_extra_record_field(self):
        invalid = dict(self.record, notes="not allowed")
        self.dataset_path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly"):
            load_dataset(self.dataset_path)

    def test_load_dataset_rejects_invalid_gold_enum(self):
        invalid = json.loads(json.dumps(self.record))
        invalid["expected"]["claim_type"] = ["own_damage"]
        self.dataset_path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid expected labels.*claim_type"):
            load_dataset(self.dataset_path)

    def test_load_dataset_rejects_negative_gold_amount(self):
        invalid = json.loads(json.dumps(self.record))
        invalid["expected"]["estimated_amount_inr"] = -10
        self.dataset_path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid expected labels.*estimated_amount_inr"):
            load_dataset(self.dataset_path)

    def test_load_dataset_rejects_noncanonical_gold_date(self):
        invalid = json.loads(json.dumps(self.record))
        invalid["expected"]["incident_date"] = "2026-1-02"
        self.dataset_path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid expected labels.*incident_date"):
            load_dataset(self.dataset_path)

    def test_load_dataset_rejects_injury_with_non_high_urgency(self):
        invalid = json.loads(json.dumps(self.record))
        invalid["expected"]["injury_reported"] = True
        invalid["expected"]["urgency"] = "medium"
        self.dataset_path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "injury_reported is true"):
            load_dataset(self.dataset_path)

    def test_load_dataset_rejects_unsafe_vehicle_with_non_high_urgency(self):
        invalid = json.loads(json.dumps(self.record))
        invalid["message"] = "My vehicle is not safe to drive after the accident."
        invalid["expected"]["urgency"] = "medium"
        self.dataset_path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsafe or undrivable vehicle"):
            load_dataset(self.dataset_path)

    def test_load_dataset_rejects_other_explicit_unsafe_vehicle_phrases(self):
        phrases = (
            "The car is blocking the road.",
            "The car is stuck in the middle of the road and cannot be moved.",
            "The vehicle cannot be moved after the collision.",
        )
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                invalid = json.loads(json.dumps(self.record))
                invalid["message"] = phrase
                invalid["expected"]["urgency"] = "medium"
                self.dataset_path.write_text(
                    json.dumps(invalid) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "unsafe or undrivable vehicle"):
                    load_dataset(self.dataset_path)

    def test_success_writes_raw_jsonl_and_csv(self):
        model = Mock(return_value=self.fake_model_result())
        result = run_evaluation(
            "A", 1, self.dataset_path, self.output_dir, model_call=model
        )

        self.assertEqual(result["processed"], 1)
        model.assert_called_once()

        raw_records = [
            json.loads(line)
            for line in result["raw_path"].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(raw_records), 1)
        self.assertEqual(raw_records[0]["record_id"], "TEST001")
        self.assertTrue(raw_records[0]["json_valid"])
        self.assertTrue(raw_records[0]["format_compliant"])
        self.assertTrue(raw_records[0]["validation_passed"])
        self.assertEqual(raw_records[0]["total_tokens"], 180)
        self.assertEqual(raw_records[0]["raw_response"], valid_reply())
        self.assertEqual(raw_records[0]["api_status_code"], 200)
        self.assertEqual(
            raw_records[0]["rate_limit_headers"]["x-ratelimit-remaining-tokens"],
            "6200",
        )

        with result["csv_path"].open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["strategy"], "A")
        self.assertEqual(rows[0]["record_id"], "TEST001")
        self.assertIn("x-ratelimit-remaining-tokens", rows[0]["rate_limit_headers"])

    def test_permissive_json_parse_does_not_imply_format_compliance(self):
        response = "Here is the result:\n" + valid_reply() + "\nExtra text"
        model = Mock(return_value=self.fake_model_result(response))
        result = run_evaluation(
            "A", 1, self.dataset_path, self.output_dir, model_call=model
        )
        item = json.loads(result["raw_path"].read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(item["json_valid"])
        self.assertFalse(item["format_compliant"])
        self.assertTrue(item["validation_passed"])

    def test_rerun_resumes_without_duplicate_api_call(self):
        model = Mock(return_value=self.fake_model_result())
        first = run_evaluation(
            "B1", 2, self.dataset_path, self.output_dir, model_call=model
        )
        second = run_evaluation(
            "B1", 2, self.dataset_path, self.output_dir, model_call=model
        )

        self.assertEqual(first["processed"], 1)
        self.assertEqual(second["processed"], 0)
        model.assert_called_once()
        lines = first["raw_path"].read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)

    def test_api_error_is_retried_on_resume_and_success_supersedes_it(self):
        error = RuntimeError("temporary provider failure")
        error.groq_metadata = {"api_status_code": 429}
        model = Mock(side_effect=[error, self.fake_model_result()])

        first = run_evaluation(
            "A", 1, self.dataset_path, self.output_dir, model_call=model
        )
        second = run_evaluation(
            "A", 1, self.dataset_path, self.output_dir, model_call=model
        )

        self.assertEqual(first["processed"], 1)
        self.assertEqual(second["processed"], 1)
        self.assertEqual(second["retried"], 1)
        self.assertEqual(second["skipped"], 0)
        self.assertEqual(model.call_count, 2)

        attempts = [
            json.loads(line)
            for line in first["raw_path"].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(attempts), 2)
        self.assertIsNotNone(attempts[0]["api_error"])
        self.assertIsNone(attempts[1]["api_error"])
        self.assertTrue(attempts[1]["validation_passed"])

    def test_successful_evaluation_is_still_skipped_on_resume(self):
        model = Mock(return_value=self.fake_model_result())
        run_evaluation("A", 1, self.dataset_path, self.output_dir, model_call=model)
        resumed = run_evaluation(
            "A", 1, self.dataset_path, self.output_dir, model_call=model
        )
        self.assertEqual(resumed["processed"], 0)
        self.assertEqual(resumed["skipped"], 1)
        self.assertEqual(resumed["retried"], 0)
        model.assert_called_once()

    def test_different_run_number_is_not_skipped(self):
        model = Mock(return_value=self.fake_model_result())
        run_evaluation("A", 1, self.dataset_path, self.output_dir, model_call=model)
        result = run_evaluation(
            "A", 2, self.dataset_path, self.output_dir, model_call=model
        )
        self.assertEqual(result["processed"], 1)
        self.assertEqual(model.call_count, 2)

    def test_parse_failure_is_recorded_without_crashing(self):
        model = Mock(return_value=self.fake_model_result("not JSON"))
        result = run_evaluation(
            "A", 1, self.dataset_path, self.output_dir, model_call=model
        )
        item = json.loads(result["raw_path"].read_text(encoding="utf-8").splitlines()[0])
        self.assertFalse(item["json_valid"])
        self.assertFalse(item["validation_passed"])
        self.assertIsNotNone(item["parse_error"])
        self.assertIsNone(item["predicted"])

    def test_api_exception_is_recorded_and_run_continues(self):
        error = RuntimeError("simulated provider failure")
        error.groq_metadata = {
            "api_status_code": 429,
            "retry_after": "45",
            "request_id": "req_limited",
            "rate_limit_headers": {
                "x-ratelimit-remaining-requests": "0"
            },
            "latency_seconds": 2.5,
        }
        model = Mock(side_effect=error)
        result = run_evaluation(
            "C", 1, self.dataset_path, self.output_dir, model_call=model
        )
        item = json.loads(result["raw_path"].read_text(encoding="utf-8").splitlines()[0])
        self.assertIn("simulated provider failure", item["api_error"])
        self.assertFalse(item["json_valid"])
        self.assertEqual(item["api_status_code"], 429)
        self.assertEqual(item["retry_after"], "45")
        self.assertEqual(item["request_id"], "req_limited")
        self.assertEqual(item["latency_seconds"], 2.5)

    def test_invalid_strategy_is_rejected_before_model_call(self):
        model = Mock()
        with self.assertRaises(ValueError):
            run_evaluation(
                "Z", 1, self.dataset_path, self.output_dir, model_call=model
            )
        model.assert_not_called()

    def test_invalid_run_number_is_rejected(self):
        with self.assertRaises(ValueError):
            run_evaluation(
                "A", 0, self.dataset_path, self.output_dir, model_call=Mock()
            )


    def test_resume_rejects_changed_prompt_fingerprint(self):
        model = Mock(return_value=self.fake_model_result())
        run_evaluation(
            "A", 1, self.dataset_path, self.output_dir, model_call=model
        )

        second_model = Mock(return_value=self.fake_model_result())
        with patch(
            "evaluation.build_strategy_messages",
            return_value=[{"role": "user", "content": "A changed prompt"}],
        ):
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                run_evaluation(
                    "A", 1, self.dataset_path, self.output_dir,
                    model_call=second_model,
                )

        model.assert_called_once()
        second_model.assert_not_called()

    def test_resume_rejects_changed_dataset_fingerprint(self):
        first_model = Mock(return_value=self.fake_model_result())
        run_evaluation(
            "A", 1, self.dataset_path, self.output_dir,
            model_call=first_model,
        )

        changed_record = dict(self.record)
        changed_record["message"] += " Additional information."
        self.dataset_path.write_text(
            json.dumps(changed_record) + "\n",
            encoding="utf-8",
        )

        second_model = Mock(return_value=self.fake_model_result())
        with patch(
            "evaluation.build_strategy_messages",
            return_value=[{"role": "user", "content": "Changed prompt"}],
        ):
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                run_evaluation(
                    "A", 1, self.dataset_path, self.output_dir,
                    model_call=second_model,
                )

        first_model.assert_called_once()
        second_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
