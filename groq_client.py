import os
import time

from dotenv import load_dotenv
from groq import Groq

MODEL = "openai/gpt-oss-20b"

load_dotenv()


def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Check your .env file.")
    return Groq(api_key=api_key)


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

    start = time.time()
    response = client.chat.completions.create(**params)
    latency = time.time() - start

    usage = response.usage
    return {
        "text": response.choices[0].message.content,
        "latency_seconds": round(latency, 2),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
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
    return True
