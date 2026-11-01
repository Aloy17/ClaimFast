import groq

from groq_client import MODEL, ask_model, test_connection
from prompts import build_zero_shot_messages
from validator import FIELDS, parse_response, validate_claim

SAMPLE_CLAIM = (
    "Yesterday I had an accident near Andheri flyover in Mumbai. "
    "My car collided with another vehicle. The other driver was injured. "
    "My vehicle is not safe to drive. My policy number is MOT-2291847. "
    "My registration number is MH 02 AB 1234. "
    "The estimated repair cost is around Rs. 45,000."
)


def assess_claim(claim):
    """Send one claim to the model and validate the reply. Never raises."""
    outcome = {
        "raw_text": None,
        "data": None,
        "problems": [],
        "error": None,
        "latency_seconds": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    try:
        result = ask_model(build_zero_shot_messages(claim))
    except RuntimeError as error:
        outcome["error"] = str(error)
        return outcome
    except groq.AuthenticationError:
        outcome["error"] = "Groq rejected the API key. Check GROQ_API_KEY in your .env file."
        return outcome
    except groq.RateLimitError:
        outcome["error"] = "Groq rate limit reached. Wait a minute and try again."
        return outcome
    except groq.APIConnectionError:
        outcome["error"] = "Could not reach Groq. Check your internet connection."
        return outcome
    except groq.APIStatusError as error:
        outcome["error"] = "Groq returned an error (HTTP " + str(error.status_code) + ")."
        return outcome
    except Exception as error:
        outcome["error"] = "Unexpected error: " + type(error).__name__ + ": " + str(error)[:200]
        return outcome

    for key in ("latency_seconds", "prompt_tokens", "completion_tokens", "total_tokens"):
        outcome[key] = result.get(key)
    outcome["raw_text"] = result.get("text")

    data, parse_error = parse_response(outcome["raw_text"])
    if parse_error:
        outcome["error"] = parse_error
        return outcome

    outcome["data"] = data
    outcome["problems"] = validate_claim(data)
    return outcome


def show_outcome(outcome):
    if outcome["raw_text"] is not None:
        print("\nRaw model response:")
        print(outcome["raw_text"] if outcome["raw_text"].strip() else "(empty)")

    if outcome["error"]:
        print("\nERROR:", outcome["error"])
    else:
        if outcome["problems"]:
            print("\nValidation: FAILED")
            for problem in outcome["problems"]:
                print("  - " + problem)
        else:
            print("\nValidation: PASSED")

        print("\nExtracted claim assessment:")
        for field in FIELDS:
            value = outcome["data"].get(field, "(missing)")
            print("  " + field.ljust(22) + str(value))

    if outcome["latency_seconds"] is not None:
        print("\nLatency (seconds):", outcome["latency_seconds"])
        print(
            "Tokens: prompt",
            outcome["prompt_tokens"],
            "| completion",
            outcome["completion_tokens"],
            "| total",
            outcome["total_tokens"],
        )


def main():
    print("CLAIMFAST - ZERO-SHOT CLAIM EXTRACTION (model: " + MODEL + ")")
    print("Type a claim message and press Enter.")
    print("Commands: sample = use the sample claim | test = test Groq connection | quit = exit")

    while True:
        try:
            entry = input("\nClaim message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not entry:
            print("Nothing entered. Type a claim, or 'quit' to exit.")
        elif entry.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        elif entry.lower() == "test":
            test_connection()
        else:
            claim = SAMPLE_CLAIM if entry.lower() == "sample" else entry
            if entry.lower() == "sample":
                print("Using sample claim:\n" + claim)
            show_outcome(assess_claim(claim))


if __name__ == "__main__":
    main()
