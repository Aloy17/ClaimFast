import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import groq
import httpx

from groq_client import ask_model, extract_response_metadata


class GroqMetadataTests(unittest.TestCase):
    def test_extracts_available_rate_limit_headers(self):
        response = httpx.Response(
            200,
            headers={
                "x-ratelimit-limit-requests": "1000",
                "x-ratelimit-remaining-requests": "850",
                "x-ratelimit-reset-requests": "2h",
                "x-ratelimit-limit-tokens": "8000",
                "x-ratelimit-remaining-tokens": "6200",
                "x-ratelimit-reset-tokens": "1.2s",
                "x-request-id": "req_123",
            },
        )

        metadata = extract_response_metadata(response)

        self.assertEqual(metadata["api_status_code"], 200)
        self.assertEqual(metadata["request_id"], "req_123")
        self.assertEqual(metadata["rate_limit_headers"]["x-ratelimit-remaining-tokens"], "6200")
        self.assertNotIn("retry-after", metadata["rate_limit_headers"])

    def test_missing_response_returns_empty_metadata(self):
        metadata = extract_response_metadata(None)

        self.assertIsNone(metadata["api_status_code"])
        self.assertEqual(metadata["rate_limit_headers"], {})

    def test_successful_request_returns_headers_and_usage(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )
        raw_response = SimpleNamespace(
            headers={"x-ratelimit-remaining-requests": "900"},
            status_code=200,
            parse=Mock(return_value=completion),
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    with_raw_response=SimpleNamespace(
                        create=Mock(return_value=raw_response)
                    )
                )
            )
        )

        with patch("groq_client.get_client", return_value=client):
            result = ask_model([{"role": "user", "content": "hello"}])

        self.assertEqual(result["text"], "ok")
        self.assertEqual(result["total_tokens"], 15)
        self.assertEqual(
            result["rate_limit_headers"]["x-ratelimit-remaining-requests"],
            "900",
        )

    def test_failed_request_attaches_response_metadata_and_reraises(self):
        request = httpx.Request("POST", "https://api.groq.com/test")
        response = httpx.Response(
            429,
            request=request,
            headers={
                "retry-after": "30",
                "x-request-id": "req_429",
                "x-ratelimit-remaining-requests": "0",
            },
        )
        error = groq.RateLimitError("quota exceeded", response=response, body=None)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    with_raw_response=SimpleNamespace(
                        create=Mock(side_effect=error)
                    )
                )
            )
        )

        with patch("groq_client.get_client", return_value=client):
            with self.assertRaises(groq.RateLimitError) as raised:
                ask_model([{"role": "user", "content": "hello"}])

        metadata = raised.exception.groq_metadata
        self.assertEqual(metadata["api_status_code"], 429)
        self.assertEqual(metadata["retry_after"], "30")
        self.assertEqual(metadata["request_id"], "req_429")
        self.assertEqual(metadata["rate_limit_headers"]["x-ratelimit-remaining-requests"], "0")


if __name__ == "__main__":
    unittest.main()
