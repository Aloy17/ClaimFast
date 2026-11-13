import tempfile
import unittest
from pathlib import Path

from scripts.analyse_subgroups import (
    RUNS,
    STRATEGIES,
    build_subgroups,
    load_test_dataset,
    validate_run_coverage,
)


ROOT = Path(__file__).resolve().parent


class TestSubgroupAnalysis(unittest.TestCase):
    def setUp(self):
        self.dataset = load_test_dataset(ROOT / "data" / "test.jsonl")
        self.all_log_ids = [{"record_id": item["id"]} for item in self.dataset]

    def test_builds_expected_subgroups(self):
        groups = build_subgroups(self.dataset, self.all_log_ids)

        self.assertEqual(
            groups["ambiguous_theft"],
            {"TEST009", "TEST029", "TEST047"},
        )
        self.assertEqual(len(groups["glass_only"]), 9)
        self.assertEqual(groups["multi_incident"], {"TEST058"})

    def test_accepts_complete_strategy_run_coverage(self):
        subgroup_ids = {"TEST001", "TEST002"}
        records = [
            {
                "record_id": record_id,
                "strategy": strategy,
                "run_number": run_number,
            }
            for strategy in STRATEGIES
            for run_number in RUNS
            for record_id in subgroup_ids
        ]

        validate_run_coverage(records, "test_group", subgroup_ids)

    def test_rejects_incomplete_strategy_run_coverage(self):
        subgroup_ids = {"TEST001", "TEST002"}
        records = [
            {
                "record_id": record_id,
                "strategy": strategy,
                "run_number": run_number,
            }
            for strategy in STRATEGIES
            for run_number in RUNS
            for record_id in subgroup_ids
        ]
        records.pop()

        with self.assertRaises(ValueError):
            validate_run_coverage(records, "test_group", subgroup_ids)

    def test_rejects_duplicate_dataset_ids(self):
        duplicate = '{"id":"DUPLICATE","expected":{}}\n{"id":"DUPLICATE","expected":{}}\n'
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicate.jsonl"
            path.write_text(duplicate, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_test_dataset(path)


if __name__ == "__main__":
    unittest.main()
