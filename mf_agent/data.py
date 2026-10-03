from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from mftool import Mftool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Settings
from .models import FundRecord, Holding
from .utils import safe_float

logger = logging.getLogger("mf_agent")

CATEGORY_RULES = [
    ("Flexi Cap", ["flexi cap", "flexicap"]),
    ("Large Cap", ["large cap", "largecap"]),
    ("Large & Mid Cap", ["large & mid", "large and mid", "large-mid"]),
    ("Index / Passive", ["nifty 50", "nifty50", "sensex", "index fund", "nifty next 50"]),
    ("Mid Cap", ["mid cap", "midcap"]),
    ("Small Cap", ["small cap", "smallcap"]),
    ("Focused", ["focused"]),
    ("ELSS", ["elss", "tax saver"]),
    ("Balanced Advantage / Hybrid", ["balanced advantage", "dynamic asset", "aggressive hybrid"]),
    ("Corporate Bond", ["corporate bond"]),
    ("Banking & PSU Debt", ["banking and psu", "banking & psu"]),
    ("Short Duration", ["short duration"]),
    ("Arbitrage", ["arbitrage"]),
    ("Thematic / Sectoral", ["technology", "pharma", "healthcare", "infrastructure", "manufacturing"]),
]

MFDATA_BASE_URL = os.getenv("MF_DATA_API_URL", "https://mfdata.in").rstrip("/")
MFDATA_CACHE_TTL_HOURS = max(1.0, float(os.getenv("MF_DATA_CACHE_TTL_HOURS", "24")))
MFDATA_FAMILY_ENRICHMENT_LIMIT = max(0, int(os.getenv("MF_DATA_FAMILY_ENRICHMENT_LIMIT", "10")))
MFDATA_TIMEOUT_SECONDS = max(5, int(os.getenv("MF_DATA_TIMEOUT_SECONDS", "15")))


def classify_category(name: str) -> str:
    lower = name.lower()
    for category, keywords in CATEGORY_RULES:
        if any(k in lower for k in keywords):
            return category
    return "Other"


def load_holdings(settings: Settings) -> tuple[dict[str, Holding], dict]:
    try:
        deployable = float(os.getenv("MF_INVESTMENT_AMOUNT", "10000"))
    except ValueError:
        deployable = 10000.0
    funds = {"available_cash": deployable, "deployable_cash": deployable}
    if not os.path.exists(settings.mf_holdings_path):
        return {}, funds

    try:
        with open(settings.mf_holdings_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        items = raw.get("mf_holdings", []) if isinstance(raw, dict) else raw
        holdings: dict[str, Holding] = {}
        for item in items:
            if not isinstance(item, dict) or not item.get("fund"):
                continue
            name = item["fund"].strip()
            avg = safe_float(item.get("average_price"), 0.0)
            qty = safe_float(item.get("quantity"), 0.0)
            invested = safe_float(item.get("invested_value"), avg * qty)
            current = safe_float(item.get("current_value"), 0.0)
            pnl = safe_float(item.get("pnl"), current - invested)
            pnl_pct = (pnl / invested * 100.0) if invested else 0.0
            holdings[name] = Holding(
                scheme_name=name,
                current_value=current,
                invested_value=invested,
                pnl=pnl,
                pnl_pct=pnl_pct,
                folio=item.get("folio"),
                ticker_symbol=item.get("tradingsymbol"),
                last_price_date=item.get("last_price_date"),
            )
        return holdings, funds
    except Exception as exc:
        logger.warning("Could not load holdings: %s", exc)
        return {}, funds


def _historical_metrics(mf: Mftool, code: str) -> dict:
    try:
        raw = mf.get_scheme_historical_nav(code)
        rows = raw.get("data", []) if isinstance(raw, dict) else []
        df = pd.DataFrame(rows)
        if df.empty:
            return {}
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna().sort_values("date")
        if len(df) < 2:
            return {}
        latest = float(df.iloc[-1]["nav"])
        latest_date = df.iloc[-1]["date"]
        result = {}
        for years in (1, 3, 5):
            target = latest_date - timedelta(days=int(years * 365.25))
            prior = df[df["date"] <= target]
            if prior.empty:
                continue
            old = float(prior.iloc[-1]["nav"])
            if old > 0 and latest > 0:
                result[f"cagr_{years}y_pct"] = round(((latest / old) ** (1 / years) - 1) * 100, 2)
        return result
    except Exception as exc:
        logger.debug("Historical NAV failed for %s: %s", code, exc)
        return {}


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": "MF Research Dashboard/2.3",
        "Accept": "application/json",
    })
    return session


def _cache_path(settings: Settings, url: str, params: dict | None = None) -> Path:
    root = Path(settings.market_cache_dir) / "mfdata"
    root.mkdir(parents=True, exist_ok=True)
    query = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{url}?{query}".encode("utf-8")).hexdigest()
    return root / f"{digest}.json"


def _read_cache(path: Path):
    try:
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > MFDATA_CACHE_TTL_HOURS * 3600:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, payload) -> None:
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.debug("Could not cache mfdata response: %s", exc)


def _mfdata_get(session: requests.Session, settings: Settings, path: str, params: dict | None = None):
    url = f"{MFDATA_BASE_URL}{path}"
    cache = _cache_path(settings, url, params)
    cached = _read_cache(cache)
    if cached is not None:
        return cached
    try:
        response = session.get(url, params=params, timeout=MFDATA_TIMEOUT_SECONDS)
        if response.status_code >= 400:
            logger.warning("mfdata GET %s returned HTTP %s", path, response.status_code)
            return None
        payload = response.json()
        _write_cache(cache, payload)
        return payload
    except Exception as exc:
        logger.warning("mfdata GET failed for %s: %s", path, exc)
        return None


def _mfdata_post(session: requests.Session, settings: Settings, path: str, payload: dict):
    url = f"{MFDATA_BASE_URL}{path}"
    cache = _cache_path(settings, url, payload)
    cached = _read_cache(cache)
    if cached is not None:
        return cached
    try:
        response = session.post(url, json=payload, timeout=MFDATA_TIMEOUT_SECONDS)
        if response.status_code >= 400:
            logger.warning("mfdata POST %s returned HTTP %s", path, response.status_code)
            return None
        result = response.json()
        _write_cache(cache, result)
        return result
    except Exception as exc:
        logger.warning("mfdata POST failed for %s: %s", path, exc)
        return None


def _payload_data(payload):
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    return data if data is not None else payload


def _number(value):
    return safe_float(value)


def _normalise_return_value(value):
    if isinstance(value, dict):
        return _number(value.get("value"))
    return _number(value)


def _extract_returns(data: dict) -> dict:
    returns = data.get("returns") if isinstance(data, dict) else None
    if not isinstance(returns, dict):
        return {}
    result = {}
    for key, years in (("1y", 1), ("3y", 3), ("5y", 5)):
        value = _normalise_return_value(returns.get(key))
        if value is not None:
            result[f"cagr_{years}y_pct"] = value
    return result


def _extract_ratio_metrics(data: dict) -> dict:
    ratios = data.get("ratios") if isinstance(data, dict) else None
    if not isinstance(ratios, dict):
        return {}
    risk = ratios.get("risk") if isinstance(ratios.get("risk"), dict) else {}
    ret = ratios.get("return") if isinstance(ratios.get("return"), dict) else {}
    valuation = ratios.get("valuation") if isinstance(ratios.get("valuation"), dict) else {}
    return {
        "sharpe": _number(ret.get("sharpe", ratios.get("sharpe"))),
        "sortino": _number(risk.get("sortino", ratios.get("sortino"))),
        "volatility_pct": _number(risk.get("std_deviation", ratios.get("std_deviation"))),
        "valuation": {k: _number(v) for k, v in valuation.items() if _number(v) is not None},
    }


def _extract_holdings(payload: dict) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    data = _payload_data(payload)
    if not isinstance(data, dict):
        return {}, {}, {}

    raw_groups = []
    for key in ("equity", "equity_holdings", "debt", "debt_holdings", "other", "other_holdings"):
        value = data.get(key)
        if isinstance(value, list):
            raw_groups.extend(value)

    holdings: dict[str, float] = {}
    sectors: dict[str, float] = {}
    market_caps: dict[str, float] = {}

    for item in raw_groups:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("stock_name") or item.get("instrument_name") or item.get("security_name")
        weight = _number(item.get("weight_pct", item.get("weight")))
        if name and weight is not None and weight >= 0:
            holdings[str(name)] = weight
        sector = item.get("sector") or item.get("sector_name")
        if sector and weight is not None and weight >= 0:
            sectors[str(sector)] = sectors.get(str(sector), 0.0) + weight
        market_cap = item.get("market_cap") or item.get("market_cap_bucket") or item.get("market_cap_category")
        if market_cap and weight is not None and weight >= 0:
            market_caps[str(market_cap)] = market_caps.get(str(market_cap), 0.0) + weight

    return holdings, sectors, market_caps


def _enrich_from_mfdata(settings: Settings, records: list[FundRecord], holdings: dict[str, Holding]) -> None:
    """Enrich a bounded, deterministic subset without making the whole run depend on mfdata.

    The API is deliberately treated as an enrichment layer, not the sole source of truth.
    Cached responses are reused for 24 hours by default. If mfdata is unavailable, mftool and
    local factsheets remain usable and the scorer marks the missing evidence explicitly.
    """
    if not records:
        return

    session = _make_session()
    codes = [r.scheme_code for r in records if r.scheme_code]
    details_by_code: dict[str, dict] = {}

    # Bulk endpoint: one request per 100 schemes instead of one request per fund.
    for start in range(0, len(codes), 100):
        chunk = codes[start:start + 100]
        payload = _mfdata_post(session, settings, "/api/v1/schemes/bulk", {"scheme_codes": [int(x) if str(x).isdigit() else str(x) for x in chunk]})
        data = _payload_data(payload)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    code = str(item.get("scheme_code") or item.get("amfi_code") or "")
                    if code:
                        details_by_code[code] = item

    # If bulk is unavailable, do not hammer the API. A cached individual response is
    # still useful for held funds and the first few ranking candidates.
    if not details_by_code:
        priority = []
        held_lower = {name.lower() for name in holdings}
        for record in records:
            if record.scheme_name.lower() in held_lower:
                priority.append(record)
        priority.extend(r for r in records if r not in priority)
        for record in priority[:MFDATA_FAMILY_ENRICHMENT_LIMIT]:
            payload = _mfdata_get(session, settings, f"/api/v1/schemes/{record.scheme_code}")
            data = _payload_data(payload)
            if isinstance(data, dict):
                details_by_code[record.scheme_code] = data

    for record in records:
        detail = details_by_code.get(record.scheme_code)
        if not detail:
            continue

        record.source_quality = max(record.source_quality, 0.90)
        record_data = detail
        record.latest_nav = _number(record_data.get("nav")) or record.latest_nav
        record.nav_date = record_data.get("nav_date") or record.nav_date
        record.aum_inr_cr = _number(record_data.get("aum_cr", record_data.get("aum_inr_cr"))) or record.aum_inr_cr
        record.inception_date = record_data.get("inception_date") or record.inception_date
        record.benchmark = record_data.get("benchmark") or record.benchmark
        returns = _extract_returns(record_data)
        if record.cagr_1y_pct is None:
            record.cagr_1y_pct = returns.get("cagr_1y_pct")
        if record.cagr_3y_pct is None:
            record.cagr_3y_pct = returns.get("cagr_3y_pct")
        if record.cagr_5y_pct is None:
            record.cagr_5y_pct = returns.get("cagr_5y_pct")

        ratio_metrics = _extract_ratio_metrics(record_data)
        for field in ("sharpe", "sortino", "volatility_pct"):
            value = ratio_metrics.get(field)
            if value is not None:
                setattr(record, field, value)
        if ratio_metrics.get("valuation"):
            record.valuation.update(ratio_metrics["valuation"])

        record._mfdata_family_id = detail.get("family_id")

    # Family-level data is expensive and rate-limited, so enrich held funds first and
    # then the strongest preliminary candidates. Holdings contain sectors, so a separate
    # sectors request is normally unnecessary.
    held_lower = {name.lower() for name in holdings}
    candidates = sorted(
        records,
        key=lambda r: (
            0 if r.scheme_name.lower() in held_lower else 1,
            -(r.cagr_3y_pct if r.cagr_3y_pct is not None else r.cagr_1y_pct if r.cagr_1y_pct is not None else -999),
            r.scheme_name.lower(),
        ),
    )
    family_candidates = [r for r in candidates if getattr(r, "_mfdata_family_id", None)]
    family_candidates = family_candidates[:MFDATA_FAMILY_ENRICHMENT_LIMIT]

    for record in family_candidates:
        family_id = getattr(record, "_mfdata_family_id", None)
        if not family_id:
            continue

        holdings_payload = _mfdata_get(session, settings, f"/api/v1/families/{family_id}/holdings")
        fund_holdings, sectors, market_caps = _extract_holdings(holdings_payload or {})
        if fund_holdings:
            record.holdings.update(fund_holdings)
        if sectors:
            record.sector_weights.update(sectors)
        if market_caps:
            record.market_cap_weights.update(market_caps)

        ratios_payload = _mfdata_get(session, settings, f"/api/v1/families/{family_id}/ratios")
        ratio_data = _payload_data(ratios_payload)
        if isinstance(ratio_data, dict):
            ratio_metrics = _extract_ratio_metrics(ratio_data)
            for field in ("sharpe", "sortino", "volatility_pct"):
                value = ratio_metrics.get(field)
                if value is not None:
                    setattr(record, field, value)
            if ratio_metrics.get("valuation"):
                record.valuation.update(ratio_metrics["valuation"])

        risk_payload = _mfdata_get(session, settings, f"/api/v1/families/{family_id}/risk-detail")
        risk_data = _payload_data(risk_payload)
        if isinstance(risk_data, dict):
            drawdown = risk_data.get("drawdown") if isinstance(risk_data.get("drawdown"), dict) else {}
            risk_return = risk_data.get("risk_return") if isinstance(risk_data.get("risk_return"), dict) else {}
            if _number(drawdown.get("max_drawdown_pct")) is not None:
                record.max_drawdown_pct = _number(drawdown.get("max_drawdown_pct"))
            if record.volatility_pct is None and _number(risk_return.get("annualized_risk")) is not None:
                record.volatility_pct = _number(risk_return.get("annualized_risk"))

        if record.holdings or record.sector_weights or record.valuation or record.sharpe is not None:
            record.source_quality = 1.0


def _fund_record(mf: Mftool, item: dict, holdings: dict[str, Holding]) -> FundRecord:
    name = item["scheme_name"]
    code = str(item.get("code", ""))
    quote = {}
    details = {}
    if code:
        try:
            quote = mf.get_scheme_quote(code) or {}
            details = mf.get_scheme_details(code) or {}
        except Exception as exc:
            logger.debug("mftool quote/details failed for %s: %s", name, exc)

    hist = _historical_metrics(mf, code) if code else {}
    category = details.get("scheme_category") or item.get("category") or classify_category(name)
    amc = details.get("fund_house") or item.get("amc") or "Unknown"
    nav = safe_float(quote.get("nav"))
    nav_date = quote.get("last_updated")

    return FundRecord(
        scheme_name=name,
        scheme_code=code,
        category=category,
        amc=amc,
        latest_nav=nav,
        nav_date=nav_date,
        cagr_1y_pct=hist.get("cagr_1y_pct"),
        cagr_3y_pct=hist.get("cagr_3y_pct"),
        cagr_5y_pct=hist.get("cagr_5y_pct"),
        benchmark=None,
    )


def fetch_universe(settings: Settings, mf: Mftool) -> list[dict]:
    """Deterministic universe. Never randomize the candidate set."""
    schemes = mf.get_scheme_codes()
    candidates = []
    per_category: dict[str, int] = {}
    per_amc_category: dict[tuple[str, str], int] = {}

    for code, name in sorted(schemes.items(), key=lambda x: str(x[1]).lower()):
        lower = str(name).lower()
        if "direct" not in lower or "growth" not in lower or "idcw" in lower or "dividend" in lower:
            continue
        category = classify_category(name)
        if category == "Other":
            continue
        amc = str(name).split()[0].lower()
        key = (category, amc)
        if per_category.get(category, 0) >= settings.max_schemes_per_category:
            continue
        if per_amc_category.get(key, 0) >= settings.max_per_amc_per_category:
            continue
        candidates.append({"scheme_name": name, "category": category, "amc": amc, "code": str(code)})
        per_category[category] = per_category.get(category, 0) + 1
        per_amc_category[key] = per_amc_category.get(key, 0) + 1

    return candidates


def load_optional_factsheet_data(settings: Settings, scheme_name: str) -> dict:
    """Optional local factsheet/holdings enrichment. Missing data is explicit, never invented."""
    safe_name = "".join(c if c.isalnum() else "_" for c in scheme_name).strip("_")
    path = os.path.join(settings.factsheet_dir, f"{safe_name}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Factsheet parse failed for %s: %s", scheme_name, exc)
        return {}


def build_fund_universe(settings: Settings, holdings: dict[str, Holding]) -> list[FundRecord]:
    mf = Mftool()
    raw = fetch_universe(settings, mf)
    for name in holdings:
        if not any(x["scheme_name"].lower() == name.lower() for x in raw):
            raw.append({"scheme_name": name, "category": "Holding Scheme", "amc": "Unknown", "code": ""})

    records: list[FundRecord] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(_fund_record, mf, item, holdings) for item in raw]
        for future in concurrent.futures.as_completed(futures):
            try:
                record = future.result()
                extra = load_optional_factsheet_data(settings, record.scheme_name)
                for key in ("aum_inr_cr", "inception_date", "volatility_pct", "max_drawdown_pct", "sharpe", "sortino", "benchmark"):
                    if key in extra and extra[key] is not None:
                        setattr(record, key, extra[key])
                        record.source_quality = max(record.source_quality, 0.95)
                for key in ("sector_weights", "holdings", "market_cap_weights", "valuation"):
                    if isinstance(extra.get(key), dict) and extra[key]:
                        setattr(record, key, extra[key])
                        record.source_quality = max(record.source_quality, 0.95)
                records.append(record)
            except Exception as exc:
                logger.warning("Fund processing failed: %s", exc)

    records.sort(key=lambda x: (x.category.lower(), x.scheme_name.lower()))

    try:
        _enrich_from_mfdata(settings, records, holdings)
    except Exception as exc:
        logger.warning("mfdata enrichment failed; continuing with fallback data: %s", exc)

    return records
