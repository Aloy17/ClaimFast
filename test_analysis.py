import json
import tempfile
import unittest
from pathlib import Path

from analysis import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    EXTRACTION_FIELDS,
    _classification_metrics,
    _field_metrics,
    _percentile,
    _run_agreement,
    analyze_records,
    load_experiment_log,
    write_reports,
)
from validator import FIELDS


def make_record(record_id, expected=None, predicted=None, strategy="A", run=1, **extra):
    base = {
        "claim_type": "own_damage",
        "urgency": "medium",
        "policy_number": "POL-1",
        "incident_date": "2026-10-01",
        "location": "Andheri",
        "vehicle_reg_no": "MH02AB1234",
        "estimated_amount_inr": 10000,
        "injury_reported": False,
        "police_report_filed": None,
    }
    return {
        "strategy": strategy,
        "run_number": run,
        "record_id": record_id,
        "expected": dict(expected if expected is not None else base),
        "predicted": dict(predicted) if predicted is not None else None,
        "json_valid": predicted is not None,
        "validation_passed": predicted is not None,
        "latency_seconds": 1.0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        **extra,
    }


class AnalysisTests(unittest.TestCase):
    def test_default_paths_target_final_benchmark(self):
        project_root = Path(__file__).resolve().parent
        self.assertEqual(
            DEFAULT_INPUT,
            project_root / "outputs" / "final_benchmark" / "raw_responses.jsonl",
        )
        self.assertEqual(
            DEFAULT_OUTPUT_DIR,
            project_root / "outputs" / "final_benchmark" / "analysis",
        )

    def setUp(self):
        self.expected = {
            "claim_type": "own_damage",
            "urgency": "medium",
            "policy_number": "POL-1",
            "incident_date": "2026-10-01",
            "location": "Andheri",
            "vehicle_reg_no": "MH02AB1234",
            "estimated_amount_inr": 10000,
            "injury_reported": False,
            "police_report_filed": None,
        }

    def test_perfect_record_has_full_exact_match(self):
        report = analyze_records([make_record("T1", self.expected, self.expected)])
        metrics = report["per_strategy"]["A"]
        self.assertEqual(metrics["record_exact_match"]["accuracy"], 1.0)
        self.assertEqual(metrics["classification"]["claim_type"]["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["classification"]["urgency"]["macro_f1"], 1 / 3)
        self.assertEqual(metrics["extraction_micro"]["f1"], 1.0)

    def test_record_exact_match_requires_all_nine_fields(self):
        predicted = dict(self.expected, location="Bandra")
        report = analyze_records([make_record("T1", self.expected, predicted)])
        self.assertEqual(report["per_strategy"]["A"]["record_exact_match"]["accuracy"], 0.0)

    def test_classification_confusion_matrix_counts_wrong_class(self):
        predicted = dict(self.expected, claim_type="theft")
        result = _classification_metrics(
            [make_record("T1", self.expected, predicted)],
            "claim_type",
            ("own_damage", "theft"),
        )
        self.assertEqual(result["confusion_matrix"]["own_damage"]["theft"], 1)
        self.assertEqual(result["per_class"]["own_damage"]["recall"], 0.0)
        self.assertEqual(result["per_class"]["theft"]["precision"], 0.0)

    def test_invalid_classification_prediction_counts_as_miss(self):
        result = _classification_metrics(
            [make_record("T1", self.expected, None)],
            "claim_type",
            ("own_damage", "theft"),
        )
        self.assertEqual(result["confusion_matrix"]["own_damage"]["__invalid_prediction__"], 1)
        self.assertEqual(result["per_class"]["own_damage"]["recall"], 0.0)
        self.assertEqual(result["accuracy"], 0.0)

    def test_macro_f1_includes_absent_classes(self):
        result = _classification_metrics(
            [make_record("T1", self.expected, self.expected)],
            "claim_type",
            ("own_damage", "theft"),
        )
        self.assertEqual(result["macro_f1"], 0.5)

    def test_null_is_not_equal_to_false(self):
        expected = dict(self.expected, injury_reported=None)
        predicted = dict(self.expected, injury_reported=False)
        result = _field_metrics([make_record("T1", expected, predicted)], "injury_reported")
        self.assertEqual(result["exact_match"], 0.0)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 0)

    def test_false_is_a_valid_extracted_boolean(self):
        result = _field_metrics(
            [make_record("T1", self.expected, self.expected)],
            "injury_reported",
        )
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)

    def test_null_gold_and_non_null_prediction_is_unsupported(self):
        expected = dict(self.expected, location=None)
        predicted = dict(self.expected, location="Mumbai")
        report = analyze_records([make_record("T1", expected, predicted)])
        reliability = report["per_strategy"]["A"]["reliability"]
        self.assertEqual(reliability["unsupported_non_null_extraction_count"], 1)
        self.assertAlmostEqual(reliability["gold_null_overprediction_rate"], 1 / 6)

    def test_wrong_non_null_value_is_incorrect_but_not_null_to_value_hallucination(self):
        predicted = dict(self.expected, location="Bandra")
        report = analyze_records([make_record("T1", self.expected, predicted)])
        reliability = report["per_strategy"]["A"]["reliability"]
        self.assertEqual(reliability["unsupported_non_null_extraction_count"], 0)
        self.assertEqual(reliability["gold_null_overprediction_rate"], 0.0)
        location = report["per_strategy"]["A"]["field_metrics"]["location"]
        self.assertEqual(location["incorrect_non_null_count"], 1)

    def test_failed_prediction_counts_in_json_and_validation_denominators(self):
        report = analyze_records([make_record("T1", self.expected, None)])
        reliability = report["per_strategy"]["A"]["reliability"]
        self.assertEqual(reliability["json_valid"]["rate"], 0.0)
        self.assertEqual(reliability["validation_passed"]["rate"], 0.0)
        self.assertEqual(reliability["record_count"], 1)

    def test_format_compliance_rate_uses_only_observed_records(self):
        compliant = make_record("T1", self.expected, self.expected, format_compliant=True)
        noncompliant = make_record("T2", self.expected, self.expected, format_compliant=False)
        legacy = make_record("T3", self.expected, self.expected)
        report = analyze_records([compliant, noncompliant, legacy])
        metric = report["per_strategy"]["A"]["reliability"]["format_compliance"]
        self.assertEqual(metric["count"], 1)
        self.assertEqual(metric["denominator"], 2)
        self.assertEqual(metric["observed_count"], 2)
        self.assertEqual(metric["missing_count"], 1)
        self.assertEqual(metric["rate"], 0.5)

    def test_format_compliance_is_none_when_no_records_have_measurement(self):
        report = analyze_records([make_record("T1", self.expected, self.expected)])
        metric = report["per_strategy"]["A"]["reliability"]["format_compliance"]
        self.assertIsNone(metric["rate"])
        self.assertEqual(metric["denominator"], 0)
        self.assertEqual(metric["missing_count"], 1)

    def test_injury_recall_and_high_urgency_recall(self):
        expected = dict(self.expected, injury_reported=True, urgency="high")
        predicted = dict(expected, injury_reported=False, urgency="medium")
        report = analyze_records([make_record("T1", expected, predicted)])
        reliability = report["per_strategy"]["A"]["reliability"]
        self.assertEqual(reliability["injury_recall"]["recall"], 0.0)
        self.assertEqual(reliability["high_urgency_recall"], 0.0)

    def test_injury_recall_uses_true_injury_cases_only(self):
        injured = dict(self.expected, injury_reported=True)
        not_injured = dict(self.expected, injury_reported=False)
        missed_injury = dict(injured, injury_reported=False)
        report = analyze_records([
            make_record("T1", injured, missed_injury),
            make_record("T2", not_injured, not_injured),
            make_record("T3", not_injured, not_injured),
        ])
        injury_recall = report["per_strategy"]["A"]["reliability"]["injury_recall"]
        self.assertEqual(injury_recall["recall"], 0.0)
        self.assertEqual(injury_recall["positive_support"], 1)

    def test_performance_totals_means_and_percentiles(self):
        one = make_record("T1", self.expected, self.expected)
        two = make_record("T2", self.expected, self.expected)
        two["latency_seconds"] = 3.0
        two["prompt_tokens"] = 300
        report = analyze_records([one, two])
        performance = report["per_strategy"]["A"]["performance"]
        self.assertEqual(performance["prompt_tokens"]["total"], 400)
        self.assertEqual(performance["prompt_tokens"]["mean"], 200)
        self.assertEqual(performance["latency_seconds"]["median"], 2.0)
        self.assertEqual(performance["latency_seconds"]["p95"], 2.9)

    def test_missing_usage_values_are_counted(self):
        record = make_record("T1", self.expected, self.expected)
        record["prompt_tokens"] = None
        report = analyze_records([record])
        usage = report["per_strategy"]["A"]["performance"]["prompt_tokens"]
        self.assertEqual(usage["missing_count"], 1)
        self.assertEqual(usage["observed_count"], 0)
        self.assertIsNone(usage["mean"])

    def test_cost_estimate_uses_supplied_rates(self):
        report = analyze_records(
            [make_record("T1", self.expected, self.expected)],
            input_rate=1.0,
            output_rate=2.0,
            projected_calls=10,
        )
        cost = report["overall"]["performance"]["cost_estimate_usd"]
        self.assertAlmostEqual(cost["observed_cost_usd"], 0.0002)
        self.assertAlmostEqual(cost["projected_cost_usd"], 0.002)
        subgroup_cost = report["per_strategy"]["A"]["performance"]["cost_estimate_usd"]
        self.assertIsNone(subgroup_cost["projected_cost_usd"])

    def test_cost_requires_both_rates(self):
        with self.assertRaisesRegex(ValueError, "both"):
            analyze_records([make_record("T1", self.expected, self.expected)], input_rate=1.0)

    def test_negative_cost_rate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            analyze_records(
                [make_record("T1", self.expected, self.expected)],
                input_rate=-1.0,
                output_rate=1.0,
            )

    def test_projected_call_count_must_be_positive_integer(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    analyze_records(
                        [make_record("T1", self.expected, self.expected)],
                        projected_calls=value,
                    )

    def test_run_agreement_is_pairwise_and_strategy_scoped(self):
        p1 = dict(self.expected)
        p2 = dict(self.expected, location="Bandra")
        records = [
            make_record("T1", self.expected, p1, strategy="A", run=1),
            make_record("T1", self.expected, p2, strategy="A", run=2),
            make_record("T1", self.expected, p1, strategy="B", run=1),
        ]
        agreement = _run_agreement(records)
        self.assertEqual(agreement["compared_pairs"], 1)
        self.assertEqual(agreement["exact_record_agreement"], 0.0)
        self.assertEqual(agreement["per_field"]["location"]["agreement"], 0.0)

    def test_run_agreement_is_none_when_only_one_run_exists(self):
        agreement = _run_agreement([make_record("T1", self.expected, self.expected)])
        self.assertIsNone(agreement["exact_record_agreement"])
        self.assertEqual(agreement["compared_pairs"], 0)

    def test_run_statistics_summarize_three_runs_without_pooling_records(self):
        correct = dict(self.expected)
        wrong = dict(self.expected, claim_type="theft")
        records = [
            make_record("T1", self.expected, correct, strategy="A", run=1),
            make_record("T1", self.expected, wrong, strategy="A", run=2),
            make_record("T1", self.expected, correct, strategy="A", run=3),
        ]
        stats = analyze_records(records)["run_statistics"]["A"]
        metric = stats["metrics"]["record_exact_match"]
        self.assertEqual(stats["run_count"], 3)
        self.assertEqual(metric["observed_runs"], 3)
        self.assertAlmostEqual(metric["mean"], 2 / 3)
        self.assertAlmostEqual(metric["population_std_dev"], 0.4714045208)

    def test_percentile_handles_empty_single_and_interpolation(self):
        self.assertIsNone(_percentile([], 0.5))
        self.assertEqual(_percentile([4], 0.95), 4)
        self.assertEqual(_percentile([1, 3], 0.5), 2)

    def test_loader_rejects_malformed_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                load_experiment_log(path)

    def test_loader_uses_latest_attempt_for_duplicate_run_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retry.jsonl"
            failed = make_record("T1", self.expected, None, api_error="429")
            succeeded = make_record("T1", self.expected, self.expected, api_error=None)
            path.write_text(
                json.dumps(failed) + "\n" + json.dumps(succeeded) + "\n",
                encoding="utf-8",
            )
            records = load_experiment_log(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["predicted"], self.expected)
            self.assertIsNone(records[0]["api_error"])

    def test_loader_rejects_conflicting_expected_labels_across_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conflicting.jsonl"
            first = make_record("T1", self.expected, self.expected)
            changed = dict(self.expected, location="Bandra")
            second = make_record("T1", changed, changed)
            path.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Conflicting expected labels"):
                load_experiment_log(path)

    def test_loader_rejects_incomplete_expected_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_schema.jsonl"
            item = make_record("T1", self.expected, self.expected)
            item["expected"].pop("location")
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly"):
                load_experiment_log(path)

    def test_empty_analysis_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No experiment"):
            analyze_records([])

    def test_reports_write_json_and_csv_without_mutating_input(self):
        record = make_record("T1", self.expected, self.expected)
        report = analyze_records([record])
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(report, directory)
            self.assertTrue(paths["json_path"].exists())
            self.assertTrue(paths["csv_path"].exists())
            loaded = json.loads(paths["json_path"].read_text(encoding="utf-8"))
            self.assertEqual(loaded["record_count"], 1)
            summary_csv = paths["csv_path"].read_text(encoding="utf-8")
            self.assertIn("strategy_or_run", summary_csv)
            self.assertIn("format_compliance_rate", summary_csv)

    def test_all_extraction_fields_are_included(self):
        self.assertEqual(
            set(EXTRACTION_FIELDS),
            set(FIELDS) - {"claim_type", "urgency"},
        )


if __name__ == "__main__":
    unittest.main()
