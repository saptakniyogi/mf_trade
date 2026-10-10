from __future__ import annotations

import unittest
from unittest.mock import patch

from mf_agent.jev import JevClient, _safe_probability


class JevAdapterTests(unittest.TestCase):
    def test_invalid_probability_is_rejected(self):
        self.assertIsNone(_safe_probability("not-a-number"))
        self.assertIsNone(_safe_probability(1.2))
        self.assertEqual(_safe_probability(0.97), 0.97)

    def test_missing_key_fails_closed_without_http_call(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False):
            client = JevClient()
            result = client.screen([{"scheme_name": "Fund A"}])
        self.assertIsNotNone(result.error)
        self.assertEqual(result.decisions, [])

    def test_uses_openrouter_decisions_api_and_parses_choice(self):
        from unittest.mock import Mock
        payload = {
            "answers": {
                "fund_0": {
                    "type": "choice",
                    "choice": "routine",
                    "probabilities": {"routine": 0.99, "deep_review": 0.01},
                    "confidence": 0.98,
                }
            },
            "usage": {"input_tokens": 123, "output_tokens": 8},
        }
        response = Mock(status_code=200)
        response.json.return_value = payload
        with patch.dict("os.environ", {
            "OPENROUTER_API_KEY": "test-key",
            "JEV_API_URL": "https://openrouter.ai/api/alpha/decisions",
            "JEV_MODEL": "typesafe/jev-1.13",
        }, clear=False), patch("mf_agent.jev.requests.post", return_value=response) as post:
            result = JevClient().screen([{"scheme_name": "Fund A"}])
        self.assertIsNone(result.error)
        self.assertEqual(result.skip_names, {"Fund A"})
        self.assertEqual(result.input_tokens, 123)
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"]["model"], "typesafe/jev-1.13")
        self.assertEqual(kwargs["json"]["questions"]["fund_0"]["type"], "choice")

    def test_skip_names_requires_strict_thresholds(self):
        from mf_agent.jev import JevDecision, JevScreeningResult
        result = JevScreeningResult([
            JevDecision("A", "ROUTINE", 0.99, 0.99, "ok"),
            JevDecision("B", "ROUTINE", 0.80, 0.99, "low confidence"),
            JevDecision("C", "DEEP_REVIEW", 0.99, 0.01, "review"),
        ])
        self.assertEqual(result.skip_names, {"A"})


if __name__ == "__main__":
    unittest.main()
