from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "skills" / "run-virtual-lab-experiment" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import run_virtual_lab as virtual_lab


class FakeResponse:
    def __init__(self, body: dict) -> None:
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def read(self) -> bytes:
        return self.body


class ProviderClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = virtual_lab.Agent("Test Scientist", "testing", "test", "respond")
        self.messages = [
            {"role": "user", "name": "Facilitator", "content": "First"},
            {"role": "user", "name": "Facilitator", "content": "Second"},
        ]

    def config(self, provider: str, style: str, model: str, url: str) -> virtual_lab.ProviderConfig:
        return virtual_lab.ProviderConfig(provider, style, model, url, "TEST_API_KEY")

    def test_openai_compatible_request_and_response(self) -> None:
        captured = {}

        def urlopen(request, timeout):
            captured["request"] = request
            return FakeResponse(
                {
                    "model": "test-openai-model",
                    "choices": [{"message": {"content": "OpenAI result"}}],
                    "usage": {"total_tokens": 12},
                }
            )

        client = virtual_lab.OpenAICompatibleClient(
            "secret-value",
            self.config("openai", "openai", "test-openai-model", "https://example.test/v1/chat/completions"),
        )
        with mock.patch("urllib.request.urlopen", side_effect=urlopen):
            result = client.complete(self.agent, self.messages, 0.2)

        payload = json.loads(captured["request"].data)
        self.assertEqual(result.content, "OpenAI result")
        self.assertEqual(payload["model"], "test-openai-model")
        self.assertEqual(payload["max_completion_tokens"], 6000)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(payload["messages"][1]["content"], "First\n\nSecond")
        self.assertNotIn("name", payload["messages"][1])

    def test_anthropic_request_and_response(self) -> None:
        captured = {}

        def urlopen(request, timeout):
            captured["request"] = request
            return FakeResponse(
                {
                    "model": "test-claude",
                    "content": [{"type": "text", "text": "Claude result"}],
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                }
            )

        client = virtual_lab.AnthropicClient(
            "secret-value",
            self.config("anthropic", "anthropic", "test-claude", "https://api.anthropic.com/v1/messages"),
        )
        with mock.patch("urllib.request.urlopen", side_effect=urlopen):
            result = client.complete(self.agent, self.messages, 0.2)

        payload = json.loads(captured["request"].data)
        self.assertEqual(result.content, "Claude result")
        self.assertEqual(payload["system"], self.agent.system_prompt)
        self.assertEqual(payload["messages"][0]["content"], "First\n\nSecond")

    def test_google_request_and_response(self) -> None:
        captured = {}

        def urlopen(request, timeout):
            captured["request"] = request
            return FakeResponse(
                {
                    "modelVersion": "test-gemini",
                    "candidates": [
                        {"content": {"parts": [{"text": "Gemini result"}]}}
                    ],
                    "usageMetadata": {"totalTokenCount": 15},
                }
            )

        client = virtual_lab.GoogleClient(
            "secret-value",
            self.config(
                "google",
                "google",
                "test-gemini",
                "https://generativelanguage.googleapis.com/v1beta/models",
            ),
        )
        with mock.patch("urllib.request.urlopen", side_effect=urlopen):
            result = client.complete(self.agent, self.messages, 0.2)

        payload = json.loads(captured["request"].data)
        self.assertEqual(result.content, "Gemini result")
        self.assertTrue(captured["request"].full_url.endswith("/test-gemini:generateContent"))
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "First\n\nSecond")

    def test_provider_override_does_not_reuse_another_provider_model(self) -> None:
        args = argparse.Namespace(
            provider="openai",
            model="chosen-model",
            base_url=None,
            api_key_env=None,
        )
        config = virtual_lab.resolve_provider_config(
            {
                "virtual_lab": {
                    "provider": "deepseek",
                    "model": "deepseek-v4-pro",
                    "api_key_env": "DEEPSEEK_API_KEY",
                    "base_url": "https://api.deepseek.com/chat/completions",
                }
            },
            args,
        )
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "chosen-model")
        self.assertEqual(config.api_key_env, "OPENAI_API_KEY")
        self.assertEqual(config.base_url, "https://api.openai.com/v1/chat/completions")

    def test_inline_api_key_is_rejected_and_redacted(self) -> None:
        spec = {
            "dataset": {"path": "/private/data.csv"},
            "virtual_lab": {
                "provider": "openai",
                "model": "chosen-model",
                "api_key": "must-not-survive",
                "api_key_env": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1/chat/completions",
            },
        }
        with self.assertRaisesRegex(ValueError, "Never store credentials"):
            virtual_lab.reject_inline_secrets(spec)
        shared = virtual_lab.spec_for_llm(spec)
        self.assertNotIn("api_key", shared["virtual_lab"])
        self.assertNotIn("api_key_env", shared["virtual_lab"])
        self.assertNotIn("base_url", shared["virtual_lab"])

    def test_generic_environment_key_is_supported_without_logging_value(self) -> None:
        config = self.config(
            "openai", "openai", "chosen-model", "https://api.openai.com/v1/chat/completions"
        )
        with mock.patch.dict("os.environ", {"VIRTUAL_LAB_API_KEY": "secret-value"}, clear=True):
            key, source = virtual_lab.resolve_api_key(config, prompt=False)
        self.assertEqual(key, "secret-value")
        self.assertEqual(source, "VIRTUAL_LAB_API_KEY")
        self.assertNotIn("secret-value", json.dumps(virtual_lab.provider_catalog()))

    def test_hidden_interactive_key_is_supported(self) -> None:
        config = self.config(
            "anthropic", "anthropic", "chosen-model", "https://api.anthropic.com/v1/messages"
        )
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch(
            "getpass.getpass", return_value="prompt-secret"
        ) as prompt:
            key, source = virtual_lab.resolve_api_key(config, prompt=True)
        prompt.assert_called_once()
        self.assertEqual(key, "prompt-secret")
        self.assertEqual(source, "interactive prompt")


if __name__ == "__main__":
    unittest.main()
