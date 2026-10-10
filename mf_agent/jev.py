from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger("mf_agent")


@dataclass(frozen=True)
class JevDecision:
    scheme_name: str
    route: str
    confidence: float
    routine_probability: float
    reason: str


@dataclass
class JevScreeningResult:
    decisions: list[JevDecision]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None

    @property
    def skip_names(self) -> set[str]:
        """Candidates safe to omit from the secondary LLM review under strict policy."""
        threshold = float(os.getenv("JEV_ROUTINE_PROBABILITY_THRESHOLD", "0.97"))
        confidence_threshold = float(os.getenv("JEV_CONFIDENCE_THRESHOLD", "0.90"))
        return {
            decision.scheme_name
            for decision in self.decisions
            if decision.route == "ROUTINE"
            and decision.routine_probability >= threshold
            and decision.confidence >= confidence_threshold
        }


class JevClient:
    """Small HTTP adapter for Jev's typed decision API.

    The adapter fails closed: callers should treat any API/parse failure as
    "review required", never as a reason to skip the OpenRouter review.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.url = os.getenv("JEV_API_URL", "https://openrouter.ai/api/alpha/decisions").strip()
        self.model = os.getenv("JEV_MODEL", "typesafe/jev-1.13").strip() or "typesafe/jev-1.13"
        self.timeout = max(1.0, float(os.getenv("JEV_TIMEOUT_SECONDS", "12")))

    def enabled(self) -> bool:
        return bool(self.api_key)

    def screen(self, candidates: list[dict[str, Any]]) -> JevScreeningResult:
        started = time.monotonic()
        if not self.api_key:
            return JevScreeningResult([], error="OPENROUTER_API_KEY is not configured")

        decisions: list[JevDecision] = []
        input_tokens = output_tokens = 0

        # Keep each state comfortably below Jev's documented request-size limit.
        # One question per candidate allows a shared state to be evaluated in parallel.
        for start in range(0, len(candidates), 4):
            batch = candidates[start:start + 4]
            try:
                parsed = self._screen_batch(batch)
                decisions.extend(parsed["decisions"])
                input_tokens += parsed["input_tokens"]
                output_tokens += parsed["output_tokens"]
            except Exception as exc:
                logger.warning("Jev screening failed; candidates require normal review: %s", exc)
                return JevScreeningResult(
                    decisions=decisions,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    error=str(exc),
                )

        return JevScreeningResult(
            decisions=decisions,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def _screen_batch(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        compact = [_compact_candidate(item) for item in candidates]
        questions = {}
        names_by_id = {}
        for index, candidate in enumerate(compact):
            question_id = f"fund_{index}"
            names_by_id[question_id] = candidate["scheme_name"]
            questions[question_id] = {
                "type": "choice",
                "instructions": (
                    "For this mutual-fund research candidate, choose whether it can "
                    "be omitted from a secondary qualitative LLM review or must be "
                    "sent for deeper review. Use only the supplied state. Choose "
                    "deep_review for any material contradiction, critical data gap, "
                    "risk/concentration concern, weak evidence, or uncertainty. "
                    "Choose routine only when supplied evidence is internally "
                    "consistent and no material concern is visible. This is routing "
                    "only, not an investment recommendation."
                ),
                "criteria": {
                    "routine": "Routine: no material concern is visible in supplied evidence",
                    "deep_review": "Deep review: concern, contradiction, missing evidence, or uncertainty exists",
                },
            }

        body = {
            "model": self.model,
            "state": {
                "purpose": "Route candidates for qualitative review; do not recommend investments.",
                "candidates": compact,
            },
            "questions": questions,
        }
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Jev HTTP {response.status_code}: {response.text[:300]}")
        envelope = response.json()
        # OpenRouter Decisions API returns answers and usage at the top level.
        # Accept a `data` wrapper defensively, but reject missing/malformed answers.
        data = envelope.get("data") if isinstance(envelope.get("data"), dict) else envelope
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("Jev response does not contain an answers object")

        parsed: list[JevDecision] = []
        for question_id, scheme_name in names_by_id.items():
            answer = answers.get(question_id)
            if not isinstance(answer, dict):
                raise ValueError(f"Jev response missing answer {question_id}")
            choice = str(answer.get("choice", "")).strip().lower()
            probabilities = answer.get("probabilities") or {}
            routine_probability = _probability(probabilities, "routine")
            deep_probability = _probability(probabilities, "deep_review")
            if choice not in {"routine", "deep_review"}:
                # Conservative fallback if the API omits its selected label.
                choice = "routine" if routine_probability >= 0.97 and deep_probability <= 0.03 else "deep_review"
            confidence = _safe_probability(answer.get("confidence"))
            if confidence is None:
                confidence = max(routine_probability, deep_probability)
            parsed.append(
                JevDecision(
                    scheme_name=scheme_name,
                    route="ROUTINE" if choice == "routine" else "DEEP_REVIEW",
                    confidence=confidence,
                    routine_probability=routine_probability,
                    reason=f"Jev choice={choice}; confidence={confidence:.3f}",
                )
            )

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return {
            "decisions": parsed,
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
        }


def _compact_candidate(item: dict[str, Any]) -> dict[str, Any]:
    score = item.get("score") or {}
    metrics = item.get("fund_metrics") or {}
    return {
        "scheme_name": str(item.get("scheme_name") or "unknown")[:180],
        "category": str(item.get("category") or "unknown")[:100],
        "action": str(item.get("action") or "WAIT"),
        "score": {
            key: score.get(key)
            for key in (
                "overall", "ranking_score", "fund_quality",
                "risk_adjusted_return", "portfolio_fit", "valuation",
                "macro_resilience", "data_confidence",
            )
            if score.get(key) is not None
        },
        "evidence_status": score.get("evidence_status"),
        "allocation_eligible": bool(score.get("allocation_eligible", False)),
        "reasons_not_to_buy": list(score.get("reasons_not_to_buy") or [])[:4],
        "data_warnings": list(score.get("data_warnings") or [])[:4],
        "fund_metrics": {
            key: metrics.get(key)
            for key in (
                "cagr_1y_pct", "cagr_3y_pct", "cagr_5y_pct",
                "volatility_pct", "max_drawdown_pct", "sharpe",
                "sortino", "aum_inr_cr",
            )
            if metrics.get(key) is not None
        },
    }


def _probability(mapping: Any, key: str) -> float:
    if not isinstance(mapping, dict):
        return 0.0
    value = _safe_probability(mapping.get(key))
    return value if value is not None else 0.0


def _safe_probability(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0.0 or number > 1.0:
        return None
    return number
