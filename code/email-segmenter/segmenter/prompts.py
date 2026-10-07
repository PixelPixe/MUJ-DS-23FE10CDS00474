"""Loads the prompt file and turns an email into chat messages."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

_REQUIRED_KEYS = ("system", "user_template")


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user_template: str
    examples: tuple[dict, ...]
    fingerprint: str  # changes whenever the prompt file changes -> invalidates the cache

    def render_user(self, sender: str, subject: str, body: str) -> str:
        # Stop an email from closing our <email> wrapper and escaping the data section.
        safe_body = body.replace("</email>", "< /email>")
        return self.user_template.format(sender=sender, subject=subject, body=safe_body)

    def build_messages(self, sender: str, subject: str, body: str) -> list[dict]:
        """Few-shot examples as user/assistant turns, then the real email."""
        messages: list[dict] = []
        for ex in self.examples:
            messages.append(
                {"role": "user", "content": self.render_user(ex["sender"], ex["subject"], ex["body"])}
            )
            messages.append({"role": "assistant", "content": json.dumps(ex["output"])})
        messages.append({"role": "user", "content": self.render_user(sender, subject, body)})
        return messages


def load_prompts(path: str | Path) -> PromptBundle:
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    missing = [key for key in _REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"Prompt file {path} is missing required keys: {missing}")
    return PromptBundle(
        system=data["system"].strip(),
        user_template=data["user_template"].strip(),
        examples=tuple(data.get("examples", [])),
        fingerprint=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    )
