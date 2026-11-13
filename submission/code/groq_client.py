import os
import time

from dotenv import load_dotenv
from groq import Groq

MODEL = "openai/gpt-oss-20b"

# Headers documented by Groq for request/token rate-limit diagnostics.
RATE_LIMIT_HEADER_NAMES = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
    "retry-after",
    "x-request-id",
)

load_dotenv()


def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Check your .env file.")
    return Groq(api_key=api_key)


def extract_response_metadata(response):
    """Extract available rate-limit headers and basic HTTP response metadata."""
    if response is None:
        return {
            "api_status_code": None,
            "retry_after": None,
            "request_id": None,
            "rate_limit_headers": {},
        }

    headers = getattr(response, "headers", {}) or {}
    rate_limit_headers = {}

    for name in RATE_LIMIT_HEADER_NAMES:
        value = headers.get(name)
        if value is not None:
            rate_limit_headers[name] = str(value)

    return {
        "api_status_code": getattr(response, "status_code", None),
        "retry_after": headers.get("retry-after"),
        "request_id": headers.get("x-request-id"),
        "rate_limit_headers": rate_limit_headers,
    }


def ask_model(messages, response_format=None):
    client = get_client()
    params = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "reasoning_effort": "low",
        "max_completion_tokens": 2000,
    }
    if response_format:
        params["response_format"] = response_format

    start = time.perf_counter()
    try:
        # The raw response exposes HTTP headers while parse() returns the
        # same typed completion object as chat.completions.create().
        raw_response = client.chat.completions.with_raw_response.create(**params)
        response = raw_response.parse()
    except Exception as error:
        latency = round(time.perf_counter() - start, 2)
        metadata = extract_response_metadata(getattr(error, "response", None))
        metadata["latency_seconds"] = latency
        # Preserve the original SDK exception type/behaviour for callers.
        error.groq_metadata = metadata
        raise

    latency = round(time.perf_counter() - start, 2)
    metadata = extract_response_metadata(raw_response)

    usage = response.usage
    return {
        "text": response.choices[0].message.content,
        "latency_seconds": latency,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
        **metadata,
    }


def test_connection():
    try:
        result = ask_model(
            [{"role": "user", "content": "Reply with only the word: ready"}]
        )
    except Exception as error:
        print("Connection failed:", type(error).__name__)
        print(str(error)[:300])
        return False

    print("Connection OK")
    print("Model reply:", result["text"])
    print("Latency (seconds):", result["latency_seconds"])
    print("Total tokens:", result["total_tokens"])
    print("Rate-limit headers:", result["rate_limit_headers"])
    return True
