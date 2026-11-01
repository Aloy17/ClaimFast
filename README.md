# ClaimFast

A small command line prototype that reads a motor insurance claim message, sends it to a hosted LLM on Groq, and shows the extracted claim information as validated JSON.

This first version uses **zero-shot prompting only**: instructions and definitions, no examples. It uses the model `openai/gpt-oss-20b` on Groq's free plan. No model is downloaded to your computer.

## What it extracts

- `claim_type`: own_damage, third_party_liability, theft, total_loss, glass_only, not_a_claim
- `urgency`: high, medium, low
- `policy_number`, `incident_date` (YYYY-MM-DD), `location`, `vehicle_reg_no` (uppercase, no spaces), `estimated_amount_inr` (integer), `injury_reported`, `police_report_filed`

Missing information must come back as `null`. `false` means the message explicitly denied it.

## Project files

- `main.py`: the command line program
- `groq_client.py`: the only file that talks to Groq (`ask_model()`)
- `prompts.py`: the zero-shot prompt
- `validator.py`: finds the JSON in the reply and checks every field
- `test_claimfast.py`: automated tests that need no API key or internet
- `.env.example`: template for your API key
- `requirements.txt`: Python packages

## Setup

1. Create a free account at https://console.groq.com and create an API key at https://console.groq.com/keys. Treat the key like a password.

2. Create and activate a virtual environment inside the project folder:

        python3 -m venv .venv
        source .venv/bin/activate

3. Install the packages:

        pip install -r requirements.txt

4. Create your `.env` file from the template, then open it and put your real key after the `=`:

        cp .env.example .env
        open -e .env

   The line should look like `GROQ_API_KEY=gsk_...` with no spaces and no quotes. `.env` is listed in `.gitignore`, so it is never committed. Never paste your key into chats, code or screenshots.

## Run

    python main.py

Type a claim message and press Enter. Other commands:

- `sample` runs the built in sample claim
- `test` checks the Groq connection
- `quit` exits

The program prints the raw model response, whether validation passed, the extracted fields, the latency and the token usage.

## How a request flows

1. `main.py` reads your message.
2. `prompts.py` wraps it as a system message (instructions, definitions, JSON format) plus your message as the user message.
3. `groq_client.py` sends it to Groq and returns the text, latency and token counts.
4. `validator.py` finds the JSON object in the text (even inside a code fence) and checks the fields and values.
5. `main.py` prints the result. API errors, empty replies, malformed JSON and missing fields all produce a readable message instead of a crash.

Strict constrained JSON output is deliberately not used in this version. The goal is to see whether the model follows the JSON instructions on its own, and whether our parser and validator catch it when it does not.

## Run the tests

    python -m unittest -v

The tests use a stand in for the model, so they do not use your API key or your Groq quota.

## Troubleshooting

- **GROQ_API_KEY is missing**: `.env` is missing, or you are not running from the project folder.
- **Groq rejected the API key**: the key is wrong or deleted. Create a new one and update `.env`.
- **Rate limit reached**: the free plan allows 30 requests per minute and a limited number of tokens per minute. Wait a minute and retry.
- **Empty response**: the model used all its output tokens thinking. Try again.

## Not included yet

Few-shot and reasoning prompts, the evaluation framework and synthetic datasets.
