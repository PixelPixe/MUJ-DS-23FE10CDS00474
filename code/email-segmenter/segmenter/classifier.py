"""Core logic: email -> prompt -> LLM -> validated prediction."""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .cache import PredictionCache
from .config import ClassificationConfig
from .llm_client import LLMClient
from .prompts import PromptBundle

logger = logging.getLogger(__name__)

UNCLASSIFIED = "Unclassified"
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Email:
    subject: str = ""
    body: str = ""
    sender: str = ""
    id: str = ""
    label: str | None = None  # optional ground truth, used only for evaluation


@dataclass(frozen=True)
class Prediction:
    category: str
    confidence: float
    reason: str
    needs_review: bool = False
    from_cache: bool = False
    error: str | None = None


def parse_response(raw: str, categories: tuple[str, ...]) -> tuple[str, float, str]:
    """Validate the model's reply. Raises ValueError if it cannot be trusted."""
    match = _JSON_OBJECT.search(raw)
    if not match:
        raise ValueError(f"no JSON object in reply: {raw[:80]!r}")
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc

    canonical = {c.lower(): c for c in categories}
    category = canonical.get(str(data.get("category", "")).strip().lower())
    if category is None:
        raise ValueError(f"unknown category: {data.get('category')!r}")

    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    return category, confidence, str(data.get("reason", "")).strip()


class EmailClassifier:
    def __init__(
        self,
        client: LLMClient,
        prompts: PromptBundle,
        cfg: ClassificationConfig,
        cache: PredictionCache,
        model_name: str,
    ):
        self._client = client
        self._prompts = prompts
        self._cfg = cfg
        self._cache = cache
        self._model_name = model_name

    # -- helpers ----------------------------------------------------------
    def _clean_body(self, body: str) -> str:
        """Collapse whitespace and cut long bodies: fewer tokens, same signal."""
        text = _WHITESPACE.sub(" ", body).strip()
        if len(text) > self._cfg.max_body_chars:
            text = text[: self._cfg.max_body_chars] + " ...[truncated]"
        return text

    def _make_prediction(self, category: str, confidence: float, reason: str,
                         from_cache: bool = False) -> Prediction:
        return Prediction(
            category=category,
            confidence=confidence,
            reason=reason,
            needs_review=confidence < self._cfg.confidence_threshold,
            from_cache=from_cache,
        )

    # -- public API -------------------------------------------------------
    def classify(self, email: Email) -> Prediction:
        sender = email.sender or "(unknown)"
        subject = email.subject or "(no subject)"
        body = self._clean_body(email.body)

        key = PredictionCache.make_key(
            self._model_name, self._prompts.fingerprint, sender, subject, body
        )
        cached = self._cache.get(key)
        if cached:
            return self._make_prediction(**cached, from_cache=True)

        messages = self._prompts.build_messages(sender, subject, body)
        last_error = "unknown error"
        for attempt in range(1, self._cfg.parse_retries + 2):
            try:
                raw = self._client.complete(self._prompts.system, messages)
                category, confidence, reason = parse_response(raw, self._cfg.categories)
            except ValueError as exc:  # bad model output -> ask again
                last_error = str(exc)
                logger.warning("Email %s: attempt %d gave unusable output (%s)",
                               email.id, attempt, last_error)
                continue
            except Exception as exc:  # API failure (SDK already retried) -> give up on this email
                logger.error("Email %s: API call failed: %s", email.id, exc)
                return Prediction(UNCLASSIFIED, 0.0, "", needs_review=True, error=str(exc))

            self._cache.set(key, {"category": category, "confidence": confidence, "reason": reason})
            return self._make_prediction(category, confidence, reason)

        return Prediction(UNCLASSIFIED, 0.0, "", needs_review=True,
                          error=f"invalid model output: {last_error}")

    def classify_many(self, emails: list[Email], workers: int = 5) -> list[Prediction]:
        """Classify emails in parallel; results keep the input order."""
        try:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                predictions = list(pool.map(self.classify, emails))
        finally:
            self._cache.save()  # keep progress even if interrupted
        cached = sum(p.from_cache for p in predictions)
        logger.info("Classified %d emails (%d from cache, %d API calls)",
                    len(predictions), cached, len(predictions) - cached)
        return predictions
