"""Thin wrappers around LLM provider SDKs.

The rest of the project only calls `client.complete(system, messages)`, so
switching provider is a one-line change in config.yaml.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from .config import LLMConfig


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, messages: list[dict]) -> str:
        """Send a system prompt + chat messages, return the model's text reply."""


def _require_api_key(env_name: str) -> str:
    key = os.getenv(env_name)
    if not key:
        raise RuntimeError(
            f"Missing API key. Set {env_name} in your environment or in a .env file."
        )
    return key


class AnthropicClient(LLMClient):
    def __init__(self, cfg: LLMConfig):
        import anthropic  # imported lazily so unused providers need not be installed

        self._cfg = cfg
        self._client = anthropic.Anthropic(
            api_key=_require_api_key(cfg.api_key_env),
            timeout=cfg.timeout_seconds,
            max_retries=cfg.max_retries,
        )

    def complete(self, system: str, messages: list[dict]) -> str:
        response = self._client.messages.create(
            model=self._cfg.model,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            system=system,
            messages=messages,
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAICompatibleClient(LLMClient):
    """Works with OpenAI and any provider exposing the same chat API (Groq, Gemini, ...)."""

    def __init__(self, cfg: LLMConfig):
        from openai import OpenAI

        self._cfg = cfg
        self._client = OpenAI(
            api_key=_require_api_key(cfg.api_key_env),
            base_url=cfg.base_url,
            timeout=cfg.timeout_seconds,
            max_retries=cfg.max_retries,
        )

    def complete(self, system: str, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self._cfg.model,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            messages=[{"role": "system", "content": system}, *messages],
        )
        return response.choices[0].message.content or ""


_PROVIDERS = {"anthropic": AnthropicClient, "openai": OpenAICompatibleClient}


def create_client(cfg: LLMConfig) -> LLMClient:
    try:
        provider = _PROVIDERS[cfg.provider.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown provider {cfg.provider!r}. Choose one of: {', '.join(_PROVIDERS)}"
        ) from None
    return provider(cfg)
