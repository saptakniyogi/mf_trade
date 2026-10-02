from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from typing import Iterable


def safe_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        value = float(value)
        return default if math.isnan(value) or math.isinf(value) else value
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def normalize(value: float, low: float, high: float) -> float:
    if high == low:
        return 50.0
    return clamp((value - low) / (high - low) * 100.0)


def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def stable_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def sanitize_text(text: str, max_chars: int = 500) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", str(text))
    cleaned = re.sub(r"(?i)(ignore\s+all\s+instructions|system\s+prompt|user\s+prompt|assistant:|override)", "[FILTERED]", cleaned)
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


def parse_date(value: str | None):
    if not value:
        return None
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    return None


def weighted_average(values: Iterable[tuple[float, float]]) -> float:
    numerator = denominator = 0.0
    for value, weight in values:
        numerator += value * weight
        denominator += weight
    return numerator / denominator if denominator else 0.0
