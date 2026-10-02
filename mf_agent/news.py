from __future__ import annotations

import logging
import time

import requests
from datetime import datetime, timezone

import feedparser

from .config import Settings
from .utils import sanitize_text, stable_id

logger = logging.getLogger("mf_agent")

RSS_SOURCES = [
    ("moneycontrol_mf", "https://www.moneycontrol.com/rss/mfnews.xml", "mutual_funds"),
    ("economic_times_markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "markets"),
    ("pib_finance", "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=4", "government"),
    ("livemint_money", "https://www.livemint.com/rss/money", "personal_finance"),
    ("business_standard_economy", "https://www.business-standard.com/rss/economy-102.rss", "macro"),
]


def _parse_feed(url: str):
    """Fetch an RSS feed explicitly so HTTP failures are observable."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 MF Research Dashboard/2.3"},
            timeout=15,
        )
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        if getattr(feed, "bozo", False) and not getattr(feed, "entries", None):
            raise RuntimeError(str(getattr(feed, "bozo_exception", "invalid RSS")))
        return feed
    except Exception as exc:
        logger.warning("News feed failed [%s]: %s", url, exc)
        return None


# Google News RSS is used only as a fallback. It is intentionally broad and
# deterministic so a blocked publisher RSS endpoint does not blank the News tab.
FALLBACK_RSS = [
    ("google_news_india_markets", "https://news.google.com/rss/search?q=India%20markets%20mutual%20funds&hl=en-IN&gl=IN&ceid=IN:en", "markets"),
    ("google_news_india_economy", "https://news.google.com/rss/search?q=India%20economy%20RBI%20inflation%20oil%20rupee&hl=en-IN&gl=IN&ceid=IN:en", "macro"),
]


def fetch_news(settings: Settings) -> list[dict]:
    items = []
    seen = set()
    sources_ok = 0

    def consume(source_name: str, url: str, category: str) -> None:
        nonlocal sources_ok
        feed = _parse_feed(url)
        if feed is None:
            return
        entries = list(getattr(feed, "entries", []) or [])
        sources_ok += 1
        logger.info("News feed loaded [%s]: %d entries.", source_name, len(entries))
        for entry in entries[:settings.news_per_source]:
            published = entry.get("published", "")
            summary = sanitize_text(entry.get("summary", ""), 700)
            title = sanitize_text(entry.get("title", ""), 240)
            text = f"{title}. {summary}".strip()
            if not text:
                continue
            item_id = stable_id(source_name, title, published)
            if item_id in seen:
                continue
            seen.add(item_id)
            items.append({
                "id": item_id,
                "source": source_name,
                "category": category,
                "title": title,
                "summary": summary,
                "published": published,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "age_seconds": None,
                "text": text,
            })

    for source_name, url, category in RSS_SOURCES:
        consume(source_name, url, category)

    # Do not hide a total publisher-RSS outage behind an empty result.
    if not items:
        logger.warning("All primary news feeds returned no usable articles; trying fallback RSS feeds.")
        for source_name, url, category in FALLBACK_RSS:
            consume(source_name, url, category)

    logger.info("News pipeline complete: %d events from %d successful feeds.", len(items), sources_ok)
    return items



NEWS_RELEVANCE_TERMS = (
    "mutual fund", "equity", "stock market", "nifty", "sensex", "midcap",
    "smallcap", "large cap", "rbi", "repo rate", "interest rate", "inflation",
    "cpi", "gdp", "fii", "fpi", "dii", "rupee", "inr", "oil", "crude",
    "tariff", "trade", "bond yield", "10-year", "treasury", "earnings",
    "corporate profit", "credit", "liquidity", "monsoon", "geopolit",
    "sanction", "war", "market", "fund house", "asset management", "sebi",
    "gold", "commodity", "valuation", "ipo", "budget", "recession",
)
GENERIC_ADVICE_TERMS = (
    "how to invest", "how should you invest", "best mutual funds", "best funds",
    "mutual funds to buy", "portfolio tips", "portfolio advice", "sip tips",
    "where to invest", "should you invest", "investment strategy", "investor guide",
    "personal finance", "tax saving tips", "wealth creation tips",
)


def classify_event(article: dict) -> dict:
    """Deterministic first-pass event tagging. LLM can refine, but cannot invent source facts."""
    text = (article.get("title", "") + " " + article.get("summary", "")).lower()
    tags = set()
    for keyword, tag in [
        ("oil", "oil"), ("crude", "oil"), ("inflation", "inflation"),
        ("rupee", "currency"), ("tariff", "trade"), ("war", "geopolitics"),
        ("sanction", "geopolitics"), ("monsoon", "climate"), ("el niño", "climate"),
        ("fii", "foreign_flows"), ("fpi", "foreign_flows"), ("rbi", "rates"),
        ("interest rate", "rates"), ("repo rate", "rates"), ("ipo", "equity_supply"),
        ("recession", "growth"), ("gdp", "growth"), ("earnings", "earnings"),
        ("profit", "earnings"), ("bond yield", "rates"), ("sebi", "regulation"),
    ]:
        if keyword in text:
            tags.add(tag)
    mapping = {
        "oil": "inflation", "currency": "inflation", "trade": "earnings",
        "geopolitics": "risk_premium", "climate": "food_inflation",
        "foreign_flows": "liquidity", "rates": "discount_rate",
        "equity_supply": "liquidity", "growth": "earnings", "earnings": "earnings",
        "regulation": "policy",
    }
    channels = sorted({mapping[t] for t in tags if t in mapping})
    article["event_tags"] = sorted(tags)
    article["transmission_channels"] = channels
    return article


def filter_relevant_news(news: list[dict], limit: int = 40) -> list[dict]:
    """Keep market-moving events and remove generic fund-buying/advice articles."""
    relevant = []
    for article in news:
        title = str(article.get("title", ""))
        summary = str(article.get("summary", ""))
        text = (title + " " + summary).lower()
        tags = set(article.get("event_tags", []))
        generic = any(term in text for term in GENERIC_ADVICE_TERMS)
        strong_market_signal = any(term in text for term in NEWS_RELEVANCE_TERMS)
        if generic and not tags:
            continue
        if tags or strong_market_signal:
            relevant.append(article)
    # Events are more useful than generic market commentary. Preserve source
    # order within the two groups so the feed's recency ordering is retained.
    relevant.sort(key=lambda x: (0 if x.get("event_tags") else 1))
    result = relevant[:max(0, int(limit))]
    logger.info("News relevance filter: %d raw -> %d event/market-relevant articles.", len(news), len(result))
    return result

