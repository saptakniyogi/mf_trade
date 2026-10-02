from __future__ import annotations

import logging
from datetime import datetime, timedelta
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


def fetch_macro_snapshot(selected_topics: list[str] | None = None) -> MacroSnapshot:
    values = {}
    changes = {}
    for key, symbol in TICKERS.items():
        series = _series(symbol)
        values[key], changes[key] = _last_and_change(series)

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
        selected_topics=selected_topics or [],
    )
