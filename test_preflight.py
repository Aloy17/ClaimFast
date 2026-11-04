import unittest
from unittest.mock import patch

from scripts import preflight_benchmark as preflight


class BenchmarkPreflightTests(unittest.TestCase):
    def test_expected_ids_are_sequential_and_zero_padded(self):
        self.assertEqual(
            preflight.expected_ids("TEST", 3),
            {"TEST001", "TEST002", "TEST003"},
        )

    def test_inspect_dataset_rejects_wrong_record_count(self):
        with patch.object(preflight, "load_dataset", return_value=[{"id": "TEST001"}]):
            with self.assertRaisesRegex(ValueError, "expected 60 records"):
                preflight.inspect_dataset("test.jsonl", "TEST", 60)

    def test_inspect_dataset_rejects_missing_or_unexpected_ids(self):
        records = [{"id": f"TEST{i:03d}"} for i in range(1, 60)]
        records.append({"id": "TEST061"})
        with patch.object(preflight, "load_dataset", return_value=records):
            with self.assertRaisesRegex(ValueError, "ID mismatch"):
                preflight.inspect_dataset("test.jsonl", "TEST", 60)

    def test_planned_logical_evaluation_count_is_900(self):
        self.assertEqual(
            60 * len(preflight.EXPECTED_STRATEGIES) * preflight.RUNS_PER_STRATEGY,
            900,
        )


if __name__ == "__main__":
    unittest.main()
