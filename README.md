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
- `groq_client.py`: communicates with the Groq API
- `prompts.py`: zero-shot, few-shot (k=3, 5, 8) and reasoning-guided prompt builders
- `validator.py`: parses model responses and validates extracted fields
- `test_claimfast.py`: automated tests that do not require an API key
- `data/dev.jsonl`: 20 development records used for prompt development
- `data/test.jsonl`: 60 held-out test records reserved for evaluation
- `data/labelling_guide.md`: dataset labelling rules and ambiguous-case decisions
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
3. `groq_client.py` sends the request to Groq and returns the response text, latency and token counts.
4. `validator.py` extracts the JSON object from the response, including JSON inside a code fence, and checks the fields and values.
5. `main.py` displays the result. API errors, empty responses, malformed JSON and missing fields produce readable messages instead of crashing the application.

Strict constrained JSON output is deliberately not used in this version. The goal is to evaluate whether the model follows the JSON instructions independently and whether the parser and validator detect cases where it does not.

## Run the tests

Run the automated test suite:

```bash
python -m unittest -v
```

The test suite currently contains **36 automated tests**. It uses a stand-in for the model, so it does not require your API key or consume your Groq API quota.

## Troubleshooting

- **GROQ_API_KEY is missing**: `.env` is missing, or the application is not being run from the project folder.
- **Groq rejected the API key**: The key is incorrect or has been deleted. Generate a new key and update `.env`.
- **Rate limit reached**: Groq has temporarily restricted requests due to an API usage limit. Check your current model-specific limits in the Groq Console and retry when capacity is available.
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

### Pending

- Evaluation framework and experiment results
- Comparative analysis of prompt strategies
- Production prompt selection
- Final project report and demonstration