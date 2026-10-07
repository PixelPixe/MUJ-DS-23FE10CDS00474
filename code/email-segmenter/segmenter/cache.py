"""Tiny on-disk cache so the same email is never sent to the API twice."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path


class PredictionCache:
    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if enabled and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}  # corrupt cache: start fresh instead of crashing

    @staticmethod
    def make_key(*parts: str) -> str:
        """Hash everything that can change the answer (model, prompt, email text)."""
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = value

    def save(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
