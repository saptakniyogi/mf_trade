from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone

import requests
import yfinance as yf

from .models import MacroSnapshot
from .utils import safe_float

logger = logging.getLogger("mf_agent")

TICKERS = {
    "usd_inr": "INR=X",
    "brent_crude_usd": "BZ=F",
    "india_vix": "^INDIAVIX",
    "nifty_50": "^NSEI",
    "nifty_midcap": "^NSEMDCP50",
    "nifty_smallcap": "^CNXSC",
    "gold_usd": "GC=F",
    "us_10y_yield_pct": "^TNX",
    "sp500": "^GSPC",
}

YAHOO_SOURCE = "Yahoo Finance"
RBI_SOURCES = (
    "https://dbie.rbihub.in/",
    "https://dev.dbie.rbihub.in/",
)
RBI_HOME_URL = "https://m.rbi.org.in/home.aspx"


def _series(symbol: str):
    try:
        data = yf.Ticker(symbol).history(period="3mo", auto_adjust=False)
        if data.empty or "Close" not in data:
            return None
        return data["Close"].dropna()
    except Exception as exc:
        logger.warning("Market data failed for %s: %s", symbol, exc)
        return None


def _last_and_change(series):
    if series is None or len(series) == 0:
        return None, None
    current = safe_float(series.iloc[-1])
    prior = safe_float(series.iloc[-22]) if len(series) > 22 else None
    change = ((current / prior) - 1) * 100 if current is not None and prior else None
    return current, change


def _clean_web_text(raw: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_indicator(text: str, labels: tuple[str, ...]) -> tuple[float | None, str | None]:
    for label in labels:
        pattern = (
            rf"{re.escape(label)}\s+"
            rf"([A-Za-z]{{3,12}}\s+\d{{4}})\s+"
            rf"([+-]?[0-9]+(?:\.[0-9]+)?)\s*%"
        )
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return safe_float(match.group(2)), match.group(1)
    return None, None


def _fetch_rbi_macro() -> tuple[dict[str, float | None], dict[str, str]]:
    """Fetch RBI indicators with two RBI DBIE endpoints plus current-rates fallback.

    The DBIE page is parsed only after HTML is normalized, which makes the extractor
    tolerant of the portal's markup/layout changes. Missing fields remain None.
    """
    values: dict[str, float | None] = {
        "india_10y_yield_pct": None,
        "inflation_pct": None,
        "repo_rate_pct": None,
    }
    sources: dict[str, str] = {}
    headers = {"User-Agent": "Mozilla/5.0 MF Research Dashboard/2.3"}

    for url in RBI_SOURCES:
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            text = _clean_web_text(response.text)

            if values["repo_rate_pct"] is None:
                value, period = _extract_indicator(text, ("Policy repo rate", "Policy Repo Rate"))
                if value is not None:
                    values["repo_rate_pct"] = value
                    sources["repo_rate_pct"] = f"RBI DBIE ({period})"

            if values["inflation_pct"] is None:
                value, period = _extract_indicator(text, ("CPI inflation", "CPI Inflation"))
                if value is not None:
                    values["inflation_pct"] = value
                    sources["inflation_pct"] = f"RBI DBIE ({period})"

            if values["india_10y_yield_pct"] is None:
                value, period = _extract_indicator(text, ("10-year G-sec yield", "10-year G-sec Yield"))
                if value is not None:
                    values["india_10y_yield_pct"] = value
                    sources["india_10y_yield_pct"] = f"RBI DBIE ({period})"

            if all(value is not None for value in values.values()):
                break
        except Exception as exc:
            logger.warning("RBI DBIE fetch failed for %s: %s", url, exc)

    # Repo is also exposed by RBI's current-rates page. Use it only as a fallback.
    if values["repo_rate_pct"] is None:
        try:
            response = requests.get(
                RBI_HOME_URL,
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()
            text = _clean_web_text(response.text)
            match = re.search(
                r"Policy\s+Repo\s+Rate\s*[:|]?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                values["repo_rate_pct"] = safe_float(match.group(1))
                sources["repo_rate_pct"] = "RBI Current Rates"
        except Exception as exc:
            logger.warning("RBI current-rates fallback failed: %s", exc)

    return values, sources


def fetch_macro_snapshot(selected_topics: list[str] | None = None) -> MacroSnapshot:
    values: dict[str, float | None] = {}
    changes: dict[str, float | None] = {}
    sources: dict[str, str] = {}

    for key, symbol in TICKERS.items():
        series = _series(symbol)
        values[key], changes[key] = _last_and_change(series)
        if values[key] is not None:
            sources[key] = YAHOO_SOURCE
        if changes[key] is not None:
            sources[f"{key}_change_1m_pct"] = YAHOO_SOURCE

    rbi_values, rbi_sources = _fetch_rbi_macro()
    values.update(rbi_values)
    sources.update(rbi_sources)

    return MacroSnapshot(
        usd_inr=values.get("usd_inr"),
        brent_crude_usd=values.get("brent_crude_usd"),
        india_vix=values.get("india_vix"),
        india_10y_yield_pct=values.get("india_10y_yield_pct"),
        nifty_50=values.get("nifty_50"),
        nifty_midcap=values.get("nifty_midcap"),
        nifty_smallcap=values.get("nifty_smallcap"),
        gold_usd=values.get("gold_usd"),
        us_10y_yield_pct=values.get("us_10y_yield_pct"),
        sp500=values.get("sp500"),
        crude_change_1m_pct=changes.get("brent_crude_usd"),
        usd_inr_change_1m_pct=changes.get("usd_inr"),
        india_vix_change_1m_pct=changes.get("india_vix"),
        inflation_pct=values.get("inflation_pct"),
        repo_rate_pct=values.get("repo_rate_pct"),
        macro_sources=sources,
        selected_topics=selected_topics or [],
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
