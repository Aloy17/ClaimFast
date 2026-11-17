# Independent confirmation evaluation protocol

## Purpose and disclosure

The original 60-record test set was used in an initial benchmark before subsequent prompt and labelling-guide revisions. Its results remain useful as historical development/evaluation evidence, but they must not be described as a pristine, never-exposed final test. This confirmation set provides a separate, later evaluation of the prompt configuration that is frozen before the confirmation run.

The confirmation messages are newly authored for this evaluation. They are not drawn from the original test set or the development examples. Keep this file out of prompt examples, prompt edits, and any further tuning.

## Dataset

- File: `data/confirmation.jsonl`
- 24 messages, IDs CONF001–CONF024
- Four examples per class: own_damage, third_party_liability, theft, total_loss, glass_only, not_a_claim
- Each record uses the existing nine-field ClaimFast gold schema.
- Labels follow `data/labelling_guide.md` and the current validation rules.

## Frozen evaluation conditions

- Model and decoding settings are inherited from `evaluation.py`.
- Strategies: A, B1, B2, B3, C.
- One run per strategy is the default confirmation plan: 5 × 24 = 120 logical evaluations.
- The dataset and prompt configuration fingerprints are stored in the raw experiment log.
- Each strategy writes to a separate run directory under `outputs/confirmation_2026-11-17/`.
- The existing 900-record benchmark under `outputs/final_benchmark/` is not read, changed, or overwritten by this runner.

Do not edit prompts, few-shot examples, gold labels, or the confirmation messages after seeing model outputs. If any such change is necessary, treat this set as development data and create another fresh confirmation set before claiming an independent final evaluation.

## Commands

Run the offline integrity check first:

```bash
./.venv/bin/python scripts/validate_confirmation.py
```

Run the five strategy evaluations (these commands make real Groq API calls):

```bash
./.venv/bin/python scripts/run_confirmation.py --strategy A
./.venv/bin/python scripts/run_confirmation.py --strategy B1
./.venv/bin/python scripts/run_confirmation.py --strategy B2
./.venv/bin/python scripts/run_confirmation.py --strategy B3
./.venv/bin/python scripts/run_confirmation.py --strategy C
```

To repeat a strategy, explicitly choose another run number, for example `--strategy A --run 2`. Each run number has a separate output directory.

## Interpretation

Report the original benchmark and the independent confirmation results separately. The confirmation set has 24 messages, so per-class results are descriptive and uncertain; do not pool its 120 strategy-record predictions as 120 independent messages. Report the dataset fingerprint, prompt fingerprints, model settings, API/parse/validation failures, and metrics. State clearly that the confirmation set was created after the original test-set exposure and is not a retroactive repair of that historical design limitation.
