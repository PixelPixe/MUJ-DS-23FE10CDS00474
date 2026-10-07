"""Typed access to config.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 200
    timeout_seconds: float = 30.0
    max_retries: int = 3


@dataclass(frozen=True)
class ClassificationConfig:
    categories: tuple[str, ...] = ("Work", "Personal", "Commercial")
    max_body_chars: int = 2000
    confidence_threshold: float = 0.6
    parse_retries: int = 2


@dataclass(frozen=True)
class RuntimeConfig:
    workers: int = 5
    cache_enabled: bool = True
    cache_path: str = ".cache/predictions.json"
    output_dir: str = "outputs"
    prompt_file: str = "prompts/prompts.yaml"


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    classification: ClassificationConfig
    runtime: RuntimeConfig


def load_config(path: str | Path = "config.yaml") -> Config:
    """Read the YAML file and build a Config. Missing keys fall back to defaults."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    classification = dict(raw.get("classification", {}))
    if "categories" in classification:
        classification["categories"] = tuple(classification["categories"])
    return Config(
        llm=LLMConfig(**raw.get("llm", {})),
        classification=ClassificationConfig(**classification),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
    )
