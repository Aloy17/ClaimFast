"""Calculate assignment metrics from saved ClaimFast experiment logs.

This module is offline-only. It never calls the model provider and never
modifies the input JSONL experiment log.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from validator import CLAIM_TYPES, FIELDS, URGENCIES

CLASSIFICATION_FIELDS = {
    "claim_type": tuple(sorted(CLAIM_TYPES)),
    "urgency": tuple(sorted(URGENCIES)),
}
EXTRACTION_FIELDS = tuple(
    field for field in FIELDS if field not in CLASSIFICATION_FIELDS
)
INVALID_LABEL = "__invalid_prediction__"


def load_experiment_log(path):
    """Load a JSONL experiment log, rejecting malformed or duplicate entries."""
    records = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in experiment log on line {line_number}: {error}"
                ) from error
            if not isinstance(item, dict):
                raise ValueError(
                    f"Experiment log line {line_number} must be a JSON object"
                )
            for key in ("strategy", "run_number", "record_id", "expected"):
                if key not in item:
                    raise ValueError(
                        f"Experiment log line {line_number} is missing {key!r}"
                    )
            if not isinstance(item["strategy"], str) or not item["strategy"]:
                raise ValueError(f"Invalid strategy on line {line_number}")
            if (
                isinstance(item["run_number"], bool)
                or not isinstance(item["run_number"], int)
                or item["run_number"] < 1
            ):
                raise ValueError(f"Invalid run_number on line {line_number}")
            if not isinstance(item["record_id"], str) or not item["record_id"]:
                raise ValueError(f"Invalid record_id on line {line_number}")
            if not isinstance(item["expected"], dict):
                raise ValueError(f"Invalid expected object on line {line_number}")
            if set(item["expected"]) != set(FIELDS):
                raise ValueError(
                    f"Expected object on line {line_number} must contain "
                    "exactly the nine ClaimFast fields"
                )
            predicted = item.get("predicted")
            if predicted is not None and not isinstance(predicted, dict):
                raise ValueError(f"Invalid predicted object on line {line_number}")

            key = (item["strategy"], item["run_number"], item["record_id"])
            if key in seen:
                raise ValueError(
                    "Duplicate strategy/run_number/record_id entry: "
                    f"{key[0]}/{key[1]}/{key[2]}"
                )
            seen.add(key)
            records.append(item)

    if not records:
        raise ValueError("Experiment log contains no records")
    return records


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _classification_metrics(records, field, labels):
    """Fixed-label one-vs-rest metrics; invalid/missing predictions are misses."""
    matrix_labels = list(labels) + [INVALID_LABEL]
    matrix = {
        actual: {predicted: 0 for predicted in matrix_labels}
        for actual in labels
    }
    per_class = {}
    correct = 0

    for item in records:
        expected = item["expected"][field]
        predicted_object = item.get("predicted")
        predicted = (
            predicted_object.get(field)
            if isinstance(predicted_object, dict)
            else None
        )
        if predicted not in labels:
            predicted = INVALID_LABEL
        if expected not in labels:
            raise ValueError(
                f"Expected {field} value {expected!r} is outside the schema"
            )
        matrix[expected][predicted] += 1
        correct += predicted == expected

    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in labels if actual != label)
        fn = sum(matrix[label][predicted] for predicted in matrix_labels if predicted != label)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "support": sum(matrix[label].values()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    count = len(records)
    return {
        "accuracy": _safe_divide(correct, count),
        "macro_precision": sum(x["precision"] for x in per_class.values()) / len(labels),
        "macro_recall": sum(x["recall"] for x in per_class.values()) / len(labels),
        "macro_f1": sum(x["f1"] for x in per_class.values()) / len(labels),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "labels": matrix_labels,
        "count": count,
    }


def _field_metrics(records, field):
    exact = 0
    tp = fp = fn = 0
    unsupported = 0
    incorrect_non_null = 0

    for item in records:
        expected = item["expected"][field]
        predicted_object = item.get("predicted")
        predicted = (
            predicted_object.get(field)
            if isinstance(predicted_object, dict)
            else None
        )
        if predicted == expected:
            exact += 1

        if predicted is not None and predicted != expected:
            fp += 1
            incorrect_non_null += 1
            if expected is None:
                unsupported += 1
        if expected is not None and predicted != expected:
            fn += 1
        if predicted is not None and predicted == expected:
            tp += 1

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "exact_match": _safe_divide(exact, len(records)),
        "correct": exact,
        "count": len(records),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "unsupported_non_null_count": unsupported,
        "incorrect_non_null_count": incorrect_non_null,
    }


def _record_exact_match(records):
    exact = 0
    for item in records:
        predicted = item.get("predicted")
        if isinstance(predicted, dict) and predicted == item["expected"]:
            exact += 1
    return {
        "accuracy": _safe_divide(exact, len(records)),
        "exact_records": exact,
        "count": len(records),
    }


def _rate(records, field):
    return {
        "count": sum(item.get(field) is True for item in records),
        "rate": _safe_divide(
            sum(item.get(field) is True for item in records), len(records)
        ),
        "denominator": len(records),
    }


def _positive_class_recall(records, field, positive_value=True):
    """Recall for a positive boolean label, not recall of non-null extraction."""
    true_positive = false_negative = 0
    for item in records:
        expected = item["expected"][field]
        predicted_object = item.get("predicted")
        predicted = (
            predicted_object.get(field)
            if isinstance(predicted_object, dict)
            else None
        )
        if expected == positive_value:
            if predicted == positive_value:
                true_positive += 1
            else:
                false_negative += 1
    return {
        "recall": _safe_divide(
            true_positive, true_positive + false_negative
        ),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "positive_support": true_positive + false_negative,
    }


def _percentile(values, percentile):
    """Linear-interpolated percentile, matching the common inclusive method."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _performance_metrics(records, input_rate=None, output_rate=None, projected_calls=900):
    metrics = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [
            item[field] for item in records
            if isinstance(item.get(field), int)
            and not isinstance(item.get(field), bool)
            and item[field] >= 0
        ]
        metrics[field] = {
            "total": sum(values),
            "mean": _safe_divide(sum(values), len(values)) if values else None,
            "observed_count": len(values),
            "missing_count": len(records) - len(values),
        }

    latencies = [
        item["latency_seconds"] for item in records
        if isinstance(item.get("latency_seconds"), (int, float))
        and not isinstance(item.get("latency_seconds"), bool)
        and item["latency_seconds"] >= 0
    ]
    metrics["latency_seconds"] = {
        "mean": _safe_divide(sum(latencies), len(latencies)) if latencies else None,
        "median": _percentile(latencies, 0.5),
        "p95": _percentile(latencies, 0.95),
        "observed_count": len(latencies),
        "missing_count": len(records) - len(latencies),
    }

    if input_rate is not None and output_rate is not None:
        if input_rate < 0 or output_rate < 0:
            raise ValueError("Token prices must be non-negative")
        prompt = metrics["prompt_tokens"]
        completion = metrics["completion_tokens"]
        observed_cost = (
            prompt["total"] * input_rate
            + completion["total"] * output_rate
        ) / 1_000_000
        average_prompt = prompt["mean"]
        average_completion = completion["mean"]
        if average_prompt is None or average_completion is None:
            projected_cost = None
            projected_prompt_tokens = None
            projected_completion_tokens = None
        else:
            projected_prompt_tokens = average_prompt * projected_calls
            projected_completion_tokens = average_completion * projected_calls
            projected_cost = (
                projected_prompt_tokens * input_rate
                + projected_completion_tokens * output_rate
            ) / 1_000_000
        metrics["cost_estimate_usd"] = {
            "input_price_usd_per_million_tokens": input_rate,
            "output_price_usd_per_million_tokens": output_rate,
            "observed_cost_usd": observed_cost,
            "projected_calls": projected_calls,
            "projected_prompt_tokens": projected_prompt_tokens,
            "projected_completion_tokens": projected_completion_tokens,
            "projected_cost_usd": projected_cost,
            "note": (
                "Projection extrapolates the observed average tokens per request; "
                "it is not a quote and is representative only if the observed "
                "sample reflects the planned strategies and dataset."
            ),
        }
    else:
        metrics["cost_estimate_usd"] = {
            "available": False,
            "note": (
                "Supply both input and output token prices to calculate cost. "
                "Prices are intentionally not hard-coded."
            ),
        }
    return metrics


def _run_agreement(records):
    groups = defaultdict(list)
    for item in records:
        if isinstance(item.get("predicted"), dict):
            groups[(item["strategy"], item["record_id"])].append(item["predicted"])

    total_pairs = agreeing_pairs = 0
    field_pairs = {field: {"agreeing_pairs": 0, "compared_pairs": 0} for field in FIELDS}

    for predictions in groups.values():
        for left_index in range(len(predictions)):
            for right_index in range(left_index + 1, len(predictions)):
                left = predictions[left_index]
                right = predictions[right_index]
                total_pairs += 1
                agreeing_pairs += left == right
                for field in FIELDS:
                    field_pairs[field]["compared_pairs"] += 1
                    field_pairs[field]["agreeing_pairs"] += (
                        left.get(field) == right.get(field)
                    )

    return {
        "exact_record_agreement": _safe_divide(agreeing_pairs, total_pairs)
        if total_pairs else None,
        "agreeing_pairs": agreeing_pairs,
        "compared_pairs": total_pairs,
        "valid_prediction_groups": len(groups),
        "per_field": {
            field: {
                "agreement": _safe_divide(values["agreeing_pairs"], values["compared_pairs"])
                if values["compared_pairs"] else None,
                **values,
            }
            for field, values in field_pairs.items()
        },
        "definition": (
            "Pairwise agreement between parsed prediction objects for the same "
            "strategy and record across different runs. API and parse failures "
            "are excluded from agreement comparisons and remain counted in "
            "reliability rates."
        ),
    }


def analyze_records(records, input_rate=None, output_rate=None, projected_calls=900):
    """Return aggregate and per-strategy/per-run metrics for experiment records."""
    if not records:
        raise ValueError("No experiment records supplied")
    if (input_rate is None) != (output_rate is None):
        raise ValueError("Provide both input_rate and output_rate, or neither")
    if (
        isinstance(projected_calls, bool)
        or not isinstance(projected_calls, int)
        or projected_calls < 1
    ):
        raise ValueError("projected_calls must be a positive integer")

    strategies = defaultdict(list)
    runs = defaultdict(list)
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("expected"), dict):
            raise ValueError("Each experiment record must contain an expected object")
        if set(item["expected"]) != set(FIELDS):
            raise ValueError("Each expected object must contain exactly nine fields")
        predicted = item.get("predicted")
        if predicted is not None and not isinstance(predicted, dict):
            raise ValueError("predicted must be an object or null")
        strategy = item.get("strategy")
        run_number = item.get("run_number")
        if not isinstance(strategy, str) or not strategy:
            raise ValueError("Each record must have a strategy")
        if isinstance(run_number, bool) or not isinstance(run_number, int) or run_number < 1:
            raise ValueError("Each record must have a positive run_number")
        strategies[strategy].append(item)
        runs[(strategy, run_number)].append(item)

    def summarize(group):
        classification = {
            field: _classification_metrics(group, field, labels)
            for field, labels in CLASSIFICATION_FIELDS.items()
        }
        fields = {field: _field_metrics(group, field) for field in FIELDS}
        extraction_totals = {
            key: sum(fields[field][key] for field in EXTRACTION_FIELDS)
            for key in ("true_positive", "false_positive", "false_negative")
        }
        tp = extraction_totals["true_positive"]
        fp = extraction_totals["false_positive"]
        fn = extraction_totals["false_negative"]
        extraction_precision = _safe_divide(tp, tp + fp)
        extraction_recall = _safe_divide(tp, tp + fn)
        extraction_f1 = _safe_divide(
            2 * extraction_precision * extraction_recall,
            extraction_precision + extraction_recall,
        )
        predicted_non_null = sum(
            item.get("predicted") is not None
            and item["predicted"].get(field) is not None
            for item in group for field in EXTRACTION_FIELDS
        )
        unsupported = sum(
            item.get("predicted") is not None
            and item["expected"][field] is None
            and item["predicted"].get(field) is not None
            for item in group for field in EXTRACTION_FIELDS
        )
        reliability = {
            "json_valid": _rate(group, "json_valid"),
            "validation_passed": _rate(group, "validation_passed"),
            "api_error_count": sum(bool(item.get("api_error")) for item in group),
            "parse_error_count": sum(bool(item.get("parse_error")) for item in group),
            "prediction_available_count": sum(isinstance(item.get("predicted"), dict) for item in group),
            "record_count": len(group),
            "unsupported_non_null_extraction_count": unsupported,
            "predicted_non_null_extraction_count": predicted_non_null,
            "hallucination_rate": _safe_divide(unsupported, predicted_non_null),
            "hallucination_rate_definition": (
                "Predicted non-null extraction values where the gold value is null, "
                "divided by all predicted non-null extraction values. Wrong values "
                "where gold is non-null are reported separately as incorrect "
                "non-null extractions."
            ),
        }
        reliability["injury_recall"] = _positive_class_recall(
            group, "injury_reported", True
        )
        reliability["high_urgency_recall"] = classification["urgency"]["per_class"]["high"]["recall"]

        return {
            "record_exact_match": _record_exact_match(group),
            "classification": classification,
            "field_metrics": fields,
            "extraction_micro": {
                **extraction_totals,
                "precision": extraction_precision,
                "recall": extraction_recall,
                "f1": extraction_f1,
                "fields_included": list(EXTRACTION_FIELDS),
            },
            "reliability": reliability,
            "performance": _performance_metrics(
                group, input_rate, output_rate, projected_calls
            ),
        }

    per_strategy = {
        strategy: summarize(group)
        for strategy, group in sorted(strategies.items())
    }
    per_run = {
        f"{strategy}/run_{run_number}": summarize(group)
        for (strategy, run_number), group in sorted(runs.items())
    }
    return {
        "record_count": len(records),
        "strategies": sorted(strategies),
        "per_strategy": per_strategy,
        "per_run": per_run,
        "run_agreement": _run_agreement(records),
        "metric_definitions": {
            "macro_average": "Unweighted mean across every allowed class in the schema.",
            "invalid_classification_prediction": (
                "Missing or out-of-schema predicted classification values are "
                "counted as invalid predictions and as false negatives for the "
                "gold class."
            ),
            "extraction_null_semantics": (
                "Null means not extracted/unknown; false is a populated boolean "
                "value. Wrong non-null extractions count as both a false positive "
                "and a false negative."
            ),
            "record_exact_match": "All nine predicted fields must exactly equal the gold object.",
            "run_agreement": (
                "Pairwise agreement between parsed predictions for the same "
                "strategy and record across runs."
            ),
        },
    }


def write_reports(report, output_dir):
    """Write JSON metrics and a compact CSV summary; input logs remain untouched."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "analysis.json"
    csv_path = output_dir / "summary.csv"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    columns = [
        "strategy_or_run", "record_count", "record_exact_match",
        "claim_type_accuracy", "claim_type_macro_f1", "urgency_accuracy",
        "urgency_macro_f1", "extraction_micro_f1", "json_valid_rate",
        "validation_pass_rate", "injury_recall", "high_urgency_recall",
        "mean_latency_seconds", "mean_prompt_tokens", "mean_completion_tokens",
        "total_tokens",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        sections = [
            ("strategy", report["per_strategy"]),
            ("run", report["per_run"]),
        ]
        for section, groups in sections:
            for name, metrics in groups.items():
                row = {
                    "strategy_or_run": name,
                    "record_count": metrics["record_exact_match"]["count"],
                    "record_exact_match": metrics["record_exact_match"]["accuracy"],
                    "claim_type_accuracy": metrics["classification"]["claim_type"]["accuracy"],
                    "claim_type_macro_f1": metrics["classification"]["claim_type"]["macro_f1"],
                    "urgency_accuracy": metrics["classification"]["urgency"]["accuracy"],
                    "urgency_macro_f1": metrics["classification"]["urgency"]["macro_f1"],
                    "extraction_micro_f1": metrics["extraction_micro"]["f1"],
                    "json_valid_rate": metrics["reliability"]["json_valid"]["rate"],
                    "validation_pass_rate": metrics["reliability"]["validation_passed"]["rate"],
                    "injury_recall": metrics["reliability"]["injury_recall"]["recall"],
                    "high_urgency_recall": metrics["reliability"]["high_urgency_recall"],
                    "mean_latency_seconds": metrics["performance"]["latency_seconds"]["mean"],
                    "mean_prompt_tokens": metrics["performance"]["prompt_tokens"]["mean"],
                    "mean_completion_tokens": metrics["performance"]["completion_tokens"]["mean"],
                    "total_tokens": metrics["performance"]["total_tokens"]["total"],
                }
                writer.writerow(row)
    return {"json_path": json_path, "csv_path": csv_path}


def main():
    parser = argparse.ArgumentParser(
        description="Analyse saved ClaimFast experiment logs without making API calls."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("outputs/raw_responses.jsonl"),
        help="Input experiment JSONL log",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/analysis"),
        help="Directory for analysis.json and summary.csv",
    )
    parser.add_argument(
        "--input-cost-per-million", type=float,
        help="Current provider input-token price in USD per million tokens",
    )
    parser.add_argument(
        "--output-cost-per-million", type=float,
        help="Current provider output-token price in USD per million tokens",
    )
    parser.add_argument(
        "--projected-calls", type=int, default=900,
        help="Number of requests for the projected benchmark (default: 900)",
    )
    args = parser.parse_args()

    records = load_experiment_log(args.input)
    report = analyze_records(
        records,
        input_rate=args.input_cost_per_million,
        output_rate=args.output_cost_per_million,
        projected_calls=args.projected_calls,
    )
    paths = write_reports(report, args.output_dir)
    print(f"Analysed {len(records)} records across {len(report['strategies'])} strategies.")
    print(f"JSON report: {paths['json_path']}")
    print(f"CSV summary: {paths['csv_path']}")
    if args.input_cost_per_million is None or args.output_cost_per_million is None:
        print("Cost not calculated: supply both token prices to estimate cost.")


if __name__ == "__main__":
    main()
