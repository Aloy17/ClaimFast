# ClaimFast Error Analysis Template

Complete this after the benchmark. Analyse failures from the held-out test set without changing its gold labels or prompt configuration mid-run.

## Sampling rule

For each prompt strategy, inspect at least 15 failed evaluations where 15 or more failures are available. If fewer than 15 failures occur, inspect every failure and state the actual count. Keep API errors, parse failures, and model predictions that parse but are wrong distinguishable.

Do not select only convenient examples. Record the sampling rule and the number of eligible failures for each strategy.

## Error categories

- **Claim-type classification:** wrong primary loss category; own damage versus third-party liability; total-loss precedence; suspected theft.
- **Urgency:** missed injury; missed unsafe/undrivable condition; high urgency inferred without evidence; total loss incorrectly treated as automatically high.
- **Extraction:** missing stated value; invented value; wrong policy/registration; date normalization error; wrong location; amount or currency error.
- **Incident attribution:** values transferred from an unrelated earlier incident or forwarded message.
- **Null/boolean semantics:** false used instead of null, or null used for explicit false/true.
- **Output reliability:** malformed JSON, extra keys, missing keys, invalid types, tagged-format violation.
- **Operational failure:** API error, timeout, rate limit, empty response.

## Per-error record

| Field | Entry |
|---|---|
| Strategy and run | |
| Test record ID | |
| Error category | |
| Gold value(s) | |
| Predicted value(s) | |
| Exact source evidence from customer message | |
| Why the prediction is wrong | |
| Severity (critical / major / minor) | |
| Could this affect customer safety or claim handling? | |
| Likely failure mechanism | |
| Proposed mitigation (do not apply during frozen benchmark) | |

## Strategy-level synthesis

For each strategy, report:

- Number of test records and number of failures eligible for review.
- Number actually reviewed and the sampling method.
- Counts by error category and severity.
- Whether errors recur across runs for the same record.
- Examples of safety-critical errors, especially injury recall and high-urgency recall.
- Whether the errors appear prompt-related, data/label-related, parsing/validation-related, or provider-related.
- A production recommendation with explicit trade-offs.

## Source-grounded hallucination check

The automated `gold_null_overprediction_rate` is not a source-grounded hallucination metric. For any suspected hallucination, compare each populated prediction directly with the original message and record the exact supporting text or write “no supporting text found”. A populated value where the gold label is null may indicate a label omission rather than a model hallucination; adjudicate that distinction before counting it.


## Independent label review

Use `data/independent_review_template.csv` for a blind review of 20 sampled messages. The reviewer should label all nine fields independently, without seeing the gold `expected` values. They must enter the literal `null` for every field they judge to be unstated; blank cells are incomplete and are rejected by the comparison script. The confidence and notes columns should be completed to explain uncertain decisions.

After the reviewer returns the completed worksheet, place it at the template path or pass its path to:

```bash
python scripts/compare_independent_review.py
```

The script reports field-level and exact-record agreement and lists every disagreement. Review disagreements with the independent reviewer and document any adjudication. Do not report agreement until the reviewer has actually completed the blind review.
