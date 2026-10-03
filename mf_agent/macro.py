from __future__ import annotations

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

RBI_DBIE_URL = "https://dbie.rbihub.in/"
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


def _extract_percent(html: str, label: str) -> tuple[float | None, str | None]:
    """Extract a labelled percentage and its observation period from RBI DBIE text."""
    pattern = rf"{re.escape(label)}\s+([A-Za-z]+\s+\d{{4}})\s*([0-9]+(?:\.[0-9]+)?)%"
    match = re.search(pattern, html, flags=re.IGNORECASE)
    if not match:
        return None, None
    return safe_float(match.group(2)), match.group(1)


def _fetch_rbi_macro() -> tuple[dict[str, float | None], dict[str, str]]:
    """Fetch policy/macro indicators from RBI DBIE, with RBI homepage fallback for repo."""
    values: dict[str, float | None] = {
        "india_10y_yield_pct": None,
        "inflation_pct": None,
        "repo_rate_pct": None,
    }
    sources: dict[str, str] = {}

    try:
        response = requests.get(
            RBI_DBIE_URL,
            headers={"User-Agent": "Mozilla/5.0 MF Research Dashboard/2.2"},
            timeout=20,
        )
        response.raise_for_status()
        html = response.text

        value, period = _extract_percent(html, "Policy repo rate")
        if value is not None:
            values["repo_rate_pct"] = value
            sources["repo_rate_pct"] = f"RBI DBIE ({period})" if period else "RBI DBIE"

        value, period = _extract_percent(html, "CPI inflation")
        if value is not None:
            values["inflation_pct"] = value
            sources["inflation_pct"] = f"RBI DBIE ({period})" if period else "RBI DBIE"

        value, period = _extract_percent(html, "10-year G-sec yield")
        if value is not None:
            values["india_10y_yield_pct"] = value
            sources["india_10y_yield_pct"] = f"RBI DBIE ({period})" if period else "RBI DBIE"
    except Exception as exc:
        logger.warning("RBI DBIE macro fetch failed: %s", exc)

    # Repo is also exposed by RBI's current-rates page. Use it only as a
    # fallback so a DBIE outage does not blank a policy-critical field.
    if values["repo_rate_pct"] is None:
        try:
            response = requests.get(
                RBI_HOME_URL,
                headers={"User-Agent": "Mozilla/5.0 MF Research Dashboard/2.2"},
                timeout=15,
            )
            response.raise_for_status()
            match = re.search(
                r"Policy\s+Repo\s+Rate\s*\|\s*:?[\s]*([0-9]+(?:\.[0-9]+)?)%",
                response.text,
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
    for key, symbol in TICKERS.items():
        series = _series(symbol)
        values[key], changes[key] = _last_and_change(series)

    rbi_values, rbi_sources = _fetch_rbi_macro()
    values.update(rbi_values)

    return MacroSnapshot(
        usd_inr=values.get("usd_inr"),
        brent_crude_usd=values.get("brent_crude_usd"),
        india_vix=values.get("india_vix"),
        nifty_50=values.get("nifty_50"),
        nifty_midcap=values.get("nifty_midcap"),
        nifty_smallcap=values.get("nifty_smallcap"),
        gold_usd=values.get("gold_usd"),
        us_10y_yield_pct=values.get("us_10y_yield_pct"),
        india_10y_yield_pct=values.get("india_10y_yield_pct"),
        sp500=values.get("sp500"),
        crude_change_1m_pct=changes.get("brent_crude_usd"),
        usd_inr_change_1m_pct=changes.get("usd_inr"),
        india_vix_change_1m_pct=changes.get("india_vix"),
        inflation_pct=values.get("inflation_pct"),
        repo_rate_pct=values.get("repo_rate_pct"),
        macro_sources=rbi_sources,
        selected_topics=selected_topics or [],
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
