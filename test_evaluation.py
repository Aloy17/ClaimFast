import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation import load_dataset, run_evaluation
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
        self.assertTrue(raw_records[0]["validation_passed"])
        self.assertEqual(raw_records[0]["total_tokens"], 180)
        self.assertEqual(raw_records[0]["raw_response"], valid_reply())

        with result["csv_path"].open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["strategy"], "A")
        self.assertEqual(rows[0]["record_id"], "TEST001")

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
        model = Mock(side_effect=RuntimeError("simulated provider failure"))
        result = run_evaluation(
            "C", 1, self.dataset_path, self.output_dir, model_call=model
        )
        item = json.loads(result["raw_path"].read_text(encoding="utf-8").splitlines()[0])
        self.assertIn("simulated provider failure", item["api_error"])
        self.assertFalse(item["json_valid"])

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
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            run_evaluation(
                "A", 1, self.dataset_path, self.output_dir,
                model_call=second_model,
            )

        first_model.assert_called_once()
        second_model.assert_not_called()

    def test_resume_rejects_changed_prompt_fingerprint(self):
        first_model = Mock(return_value=self.fake_model_result())
        run_evaluation(
            "A", 1, self.dataset_path, self.output_dir,
            model_call=first_model,
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
