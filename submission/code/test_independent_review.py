import csv
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts import compare_independent_review as review


class IndependentReviewTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "claim_type": "own_damage",
            "urgency": "medium",
            "policy_number": "POL-123",
            "incident_date": "2026-10-01",
            "location": "Andheri",
            "vehicle_reg_no": "MH02AB1234",
            "estimated_amount_inr": 12500,
            "injury_reported": False,
            "police_report_filed": None,
        }
        self.gold = {"DEV001": self.expected}

    def write_worksheet(self, path, values):
        headers = ["record_id"] + [
            "reviewer_" + field for field in review.FIELDS
        ]
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerow({"record_id": "DEV001", **values})

    def test_exact_agreement_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            self.write_worksheet(
                path,
                {"reviewer_" + key: str(value) if value is not None else "null"
                 for key, value in self.expected.items()},
            )
            output = StringIO()
            with patch.object(review, "load_gold", return_value=self.gold):
                with redirect_stdout(output):
                    review.compare(path)
            self.assertIn("Overall field agreement: 9/9 (100.0%)", output.getvalue())
            self.assertIn("Exact-record agreement: 1/1 (100.0%)", output.getvalue())

    def test_blank_value_is_rejected_instead_of_silently_becoming_null(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            values = {
                "reviewer_" + key: str(value) if value is not None else "null"
                for key, value in self.expected.items()
            }
            values["reviewer_location"] = ""
            self.write_worksheet(path, values)
            with patch.object(review, "load_gold", return_value=self.gold):
                with self.assertRaisesRegex(ValueError, "field is blank"):
                    review.compare(path)

    def test_boolean_and_amount_values_are_parsed(self):
        self.assertIs(review.parse_value("injury_reported", "false", "DEV001"), False)
        self.assertIsNone(review.parse_value("police_report_filed", "null", "DEV001"))
        self.assertEqual(
            review.parse_value("estimated_amount_inr", "12,500", "DEV001"),
            12500,
        )

    def test_invalid_boolean_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "true, false, or null"):
            review.parse_value("injury_reported", "unknown", "DEV001")

    def test_template_contains_twenty_unique_blinded_records(self):
        root = Path(__file__).resolve().parent
        template = root / "data" / "independent_review_template.csv"
        with template.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row["record_id"] for row in rows}), 20)
        self.assertTrue(all(row["message"] for row in rows))
        self.assertTrue(
            all(not row["reviewer_claim_type"] for row in rows)
        )


if __name__ == "__main__":
    unittest.main()
