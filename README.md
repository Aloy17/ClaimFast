# ClaimFast

A command-line prototype that reads a motor insurance claim message, sends it to a hosted LLM through Groq, and displays the extracted claim information as validated JSON.

The command-line application currently uses **zero-shot prompting**: instructions and definitions without examples. The project also includes few-shot prompt builders with k=3, 5 and 8 for experimentation. It uses the `openai/gpt-oss-20b` model through Groq's API. No model is downloaded to your computer.

## What it extracts

- `claim_type`: `own_damage`, `third_party_liability`, `theft`, `total_loss`, `glass_only`, `not_a_claim`
- `urgency`: `high`, `medium`, `low`
- `policy_number`
- `incident_date` (YYYY-MM-DD)
- `location`
- `vehicle_reg_no` (uppercase, without spaces or hyphens)
- `estimated_amount_inr` (integer)
- `injury_reported`
- `police_report_filed`

Missing information must be returned as `null`. `false` means the message explicitly denies the condition.

## Project files

- `main.py`: command-line claim processing application
- `groq_client.py`: communicates with the Groq API and captures available response rate-limit metadata
- `prompts.py`: zero-shot, few-shot (k=3, 5, 8) and reasoning-guided prompt builders
- `validator.py`: parses model responses and validates extracted fields
- `evaluation.py`: runs one prompt strategy/run against the held-out dataset and logs raw responses, metadata and fingerprints
- `analysis.py`: calculates evaluation metrics and exports JSON/CSV reports from saved logs without API calls
- `test_claimfast.py`, `test_groq_client.py`, `test_evaluation.py`, `test_analysis.py`: automated tests that do not require an API key
- `data/dev.jsonl`: 20 development records used for prompt development
- `data/test.jsonl`: 60 held-out test records reserved for evaluation
- `data/labelling_guide.md`: dataset labelling rules and ambiguous-case decisions
- `data/independent_review_template.csv`: blinded worksheet for an independent reviewer to label 20 sampled messages
- `scripts/compare_independent_review.py`: compares a completed independent review with the gold labels and reports agreement and disagreements
- `scripts/preflight_benchmark.py`: checks dataset IDs/counts, model settings, prompt fingerprints and the planned evaluation count without making API calls
- `docs/prompt_library.md`: documented prompt strategies and the current shared prompts
- `docs/error_analysis_template.md`: post-benchmark failure categorisation and source-grounded review template
- `.env.example`: template for the Groq API key
- `requirements.txt`: Python dependencies

## Setup

1. Create a free account at [Groq Console](https://console.groq.com) and generate an API key at [Groq API Keys](https://console.groq.com/keys). Treat the key like a password.

2. Create and activate a virtual environment inside the project folder:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Create your `.env` file from the template, then open it and enter your actual API key:

   ```bash
   cp .env.example .env
   open -e .env
   ```

   The line should look like `GROQ_API_KEY=gsk_...`, without spaces or quotation marks. `.env` is listed in `.gitignore`, so it is not committed to GitHub. Never paste your API key into chats, code or screenshots.

## Run

Start the application:

```bash
python main.py
```

Type a claim message and press Enter.

Other available commands:

- `sample`: runs the built-in sample claim
- `test`: checks the Groq API connection
- `quit`: exits the application

The program displays the raw model response, whether validation passed, the extracted fields, latency and token usage.

## How a request flows

1. `main.py` reads the user's claim message.
2. `prompts.py` constructs the system message containing instructions, definitions and the required JSON format, along with the user's message.
3. `groq_client.py` sends the request to Groq and returns the response text, latency, token counts and available rate-limit metadata.
4. `validator.py` extracts the JSON object from the response, including JSON inside a code fence, and checks the fields and values.
5. `main.py` displays the result. API errors, empty responses, malformed JSON and missing fields produce readable messages instead of crashing the application.

Strict constrained JSON output is deliberately not used in this version. The goal is to evaluate whether the model follows the JSON instructions independently and whether the parser and validator detect cases where it does not.

## Run the tests

Run the automated test suite:

```bash
python -m unittest -v
```

The automated test suite uses stand-ins for the model and HTTP responses, so it does not require your API key or consume your Groq API quota. Run the command above to see the current test count and results.

## Offline benchmark preflight

Before requesting approval to start the formal benchmark, run the offline integrity check:

```bash
python scripts/preflight_benchmark.py
```

It validates the 20 development and 60 held-out records, confirms the five strategy names, prints the dataset and per-strategy prompt fingerprints, and confirms the planned count of 900 logical evaluations (5 strategies × 3 runs × 60 test records). It also reports whether formal output files already exist. It does not call Groq, check account quota, or start an evaluation.

## Run an evaluation

Run one strategy and run number against the held-out test set:

```bash
python evaluation.py --strategy A --run 1
```

Available strategies are `A`, `B1`, `B2`, `B3` and `C`. By default, the runner writes a new formal experiment to `outputs/final_benchmark/raw_responses.jsonl` and regenerates `outputs/final_benchmark/results.csv`. This deliberately keeps the corrected-prompt benchmark separate from historical logs in `outputs/benchmark/` and legacy logs in `outputs/`. It records model settings, token usage, latency, errors, available HTTP rate-limit metadata and dataset/prompt fingerprints. Re-running the same strategy/run resumes missing records and refuses to resume if the dataset or prompt configuration has changed.

For a separate experiment output directory, pass `--output-dir`. Keep historical logs: do not mix older records without fingerprints into a new formal benchmark run. The evaluation runner makes real provider requests; run it only when the benchmark is approved.

### Rate-limit monitoring

For successful calls, the runner records the rate-limit headers returned with the final successful HTTP response. For failed calls, it records the final HTTP status, available rate-limit headers, `retry-after`, request ID and error text. The fields are stored in both `raw_responses.jsonl` and `results.csv`.

The Groq SDK retries certain transient errors automatically. These logs capture the final response after the SDK's retry handling; they do **not** count or preserve every intermediate HTTP attempt. Therefore, the benchmark's logical evaluation count can be lower than the HTTP request count shown in Groq Metrics and Logs.

Groq's documented response headers can expose remaining request capacity and per-minute token capacity. They do not provide a documented remaining-tokens-per-day header. When a daily token quota is exhausted, preserve the full 429 error text: Groq may include the daily limit, usage at rejection, requested tokens and a retry countdown. The Usage dashboard's graphs remain the source for historical consumption.

Useful pages:

- [Model limits](https://console.groq.com/settings/limits)
- [Usage graphs](https://console.groq.com/dashboard/usage)
- [Groq rate limits documentation](https://console.groq.com/docs/rate-limits)

## Independent label review

Ask a second person to label the 20 messages in `data/independent_review_template.csv` without showing them the gold labels. They should fill all nine reviewer fields, enter the literal `null` when a value is not stated, and add confidence and notes for uncertain cases.

After receiving the completed worksheet, run:

```bash
python scripts/compare_independent_review.py
```

To compare a worksheet saved elsewhere, pass its path:

```bash
python scripts/compare_independent_review.py /path/to/completed_review.csv
```

The script reports per-field agreement, exact-record agreement and each disagreement. Discuss and document disagreements with the reviewer. Do not claim an independent review or agreement rate until a real second reviewer has completed the worksheet.

## Analyse saved results

Analyse a saved log without making any model/API calls:

```bash
python analysis.py
```

By default, analysis reads `outputs/final_benchmark/raw_responses.jsonl` and writes reports to `outputs/final_benchmark/analysis/`. Use `--input` and `--output-dir` to select another saved experiment. The command writes:

- `analysis.json`: detailed overall, per-strategy and per-run metrics
- `summary.csv`: a compact comparison table

The report includes claim-type and urgency accuracy, macro precision/recall/F1, per-class metrics and confusion matrices; exact match for each field and for the complete nine-field record; extraction micro precision/recall/F1; JSON parseability, strict response-format compliance and validation pass rates; gold-null overprediction rate; injury recall and high-urgency recall; run-to-run agreement; token totals/averages; and mean, median and p95 latency. Run-level metrics are also summarized across the available runs per strategy using their mean and population standard deviation; repeated predictions are not pooled as independent samples.

For a cost estimate, supply the current provider prices in USD per million tokens:

```bash
python analysis.py --input-cost-per-million 0.075 --output-cost-per-million 0.30 --projected-calls 900
```

For `openai/gpt-oss-20b`, Groq's published price checked on 2026-11-03 is **$0.075 per million input tokens** and **$0.30 per million output tokens**. Check the [current Groq model pricing](https://console.groq.com/docs/models) before using the estimate, because prices can change. Prices are passed as arguments rather than hard-coded. The projected estimate extrapolates the observed average tokens per request to the requested call count; it is an estimate, not a provider quote. It is only representative when the observed sample reflects the planned experiment. Subgroup reports show observed costs; the projected cost is reported once for the combined input log.

`json_valid` means the parser could extract a JSON object. `format_compliance` separately checks the entire response: strategies A/B1/B2/B3 must return only one JSON object with exactly the nine required keys, while strategy C must return only the required non-empty `<reasoning>` and `<answer>` blocks with the JSON object alone inside `<answer>`. Historical logs without the `format_compliant` field are treated as unobserved for this metric, not as failures.\n\nThe analysis treats `null` as unknown/not extracted and `false` as an explicit boolean value. The `gold_null_overprediction_rate` counts populated extraction predictions where the gold value is `null`, divided by all populated extraction predictions. This is a gold-label consistency metric, not proof that a value is hallucinated: source-grounded hallucinations require checking the prediction against the original message. Incorrect populated values where the gold value is also populated are reported separately. Macro averages include every allowed class, including classes absent from the sample. Run agreement compares parsed predictions for the same strategy and record across runs; API/parse failures are excluded from agreement pairs but remain in reliability denominators.

## Troubleshooting

- **GROQ_API_KEY is missing**: `.env` is missing, or the application is not being run from the project folder.
- **Groq rejected the API key**: The key is incorrect or has been deleted. Generate a new key and update `.env`.
- **Rate limit reached**: Inspect the recorded HTTP status, `retry-after`, rate-limit headers and full provider error text. Also check your current model-specific limits in the Groq Console and retry when capacity is available.
- **Empty response**: The model returned no usable text. This may happen because of output-token limits or a provider-side issue. Try again.

## Project status

### Implemented

- Zero-shot claim extraction
- Few-shot prompt construction with k=3, 5 and 8
- Reasoning-guided prompt strategy
- Synthetic development and held-out test datasets
- Dataset labelling guide
- JSON parsing and output validation
- Automated unit tests
- Single-strategy evaluation runner with fingerprint-protected experiment logging and rate-limit metadata
- Offline metric calculation, report export and parameterized cost estimation

### Pending

- Independent second-person review of the 20-record worksheet (must be completed by a real reviewer; do not fabricate agreement)
- Approved full benchmark and experiment results
- Comparative analysis of prompt strategies
- Production prompt selection
- Final project report and demonstration
