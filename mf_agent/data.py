from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
try:
    from mftool import Mftool
except ImportError:  # Keep AMFI-only mode usable if the optional client is unavailable.
    Mftool = None  # type: ignore[assignment,misc]

from .config import Settings
from .models import FundRecord, Holding
from .utils import safe_float

logger = logging.getLogger("mf_agent")

AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"

# Deliberately broad, deterministic classification. The engine can only score
# what it can identify, so the rules cover common Indian MF naming conventions.
CATEGORY_RULES = [
    ("Flexi Cap", ["flexi cap", "flexicap"]),
    ("Large & Mid Cap", ["large & mid", "large and mid", "large-mid"]),
    ("Large Cap", ["large cap", "largecap"]),
    ("Mid Cap", ["mid cap", "midcap"]),
    ("Small Cap", ["small cap", "smallcap"]),
    ("Focused", ["focused fund", "focused"]),
    ("ELSS", ["elss", "tax saver"]),
    ("Index / Passive", ["nifty", "sensex", "index fund", "index -", "etf"]),
    ("Balanced Advantage / Hybrid", ["balanced advantage", "dynamic asset", "aggressive hybrid", "equity savings", "multi asset", "arbitrage"]),
    ("Corporate Bond", ["corporate bond"]),
    ("Banking & PSU Debt", ["banking and psu", "banking & psu"]),
    ("Short Duration", ["short duration"]),
    ("Medium Duration", ["medium duration"]),
    ("Long Duration", ["long duration"]),
    ("Liquid / Money Market", ["liquid fund", "money market", "overnight fund"]),
    ("Thematic / Sectoral", ["technology", "tech fund", "pharma", "healthcare", "infrastructure", "manufacturing", "consumption", "defence", "energy"]),
]


def _normalise_scheme_text(value: str) -> str:
    """Normalize AMFI scheme/plan/option text for resilient matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def is_direct_growth_scheme(name: str, plan: str | None = None, option: str | None = None) -> bool:
    """
    Identify Direct + Growth schemes across AMFI NAVAll formats.

    AMFI has used both:
      - a 6-column format where plan/option are embedded in Scheme Name
      - an 8-column format where Plan and Option are separate fields.

    Never infer Growth from an ISIN field. Explicit Plan/Option fields take
    precedence when available.
    """
    name_n = _normalise_scheme_text(name)
    plan_n = _normalise_scheme_text(plan or "")
    option_n = _normalise_scheme_text(option or "")

    direct = "direct" in plan_n if plan_n else "direct" in name_n
    growth = "growth" in option_n if option_n else (
        "growth" in name_n
        and "idcw" not in name_n
        and "dividend" not in name_n
        and "payout" not in name_n
        and "reinvestment" not in name_n
    )

    # If explicit AMFI Plan/Option fields exist, trust them.
    if plan_n or option_n:
        return direct and growth and not any(
            token in option_n for token in ("idcw", "dividend", "payout", "reinvestment")
        )

    return direct and growth


def classify_category(name: str, amfi_category: str | None = None) -> str:
    # Prefer an explicit AMFI section/category when it maps to one of our
    # supported analytical buckets; otherwise classify from the scheme name.
    combined = " ".join(
        x for x in (str(amfi_category or ""), str(name or "")) if x
    ).lower()
    for category, keywords in CATEGORY_RULES:
        if any(k in combined for k in keywords):
            return category
    return "Other"


def load_holdings(settings: Settings) -> tuple[dict[str, Holding], dict]:
    try:
        deployable = float(os.getenv("MF_INVESTMENT_AMOUNT", "10000"))
    except ValueError:
        deployable = 10000.0
    funds = {"available_cash": deployable, "deployable_cash": deployable}
    if not os.path.exists(settings.mf_holdings_path):
        logger.info("Holdings file '%s' not found. Continuing without existing holdings.", settings.mf_holdings_path)
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
        logger.info("Loaded %d existing mutual-fund holdings.", len(holdings))
        return holdings, funds
    except Exception as exc:
        logger.warning("Could not load holdings: %s", exc)
        return {}, funds


def _parse_nav_history(raw: object) -> pd.DataFrame:
    """Normalize NAV history from mftool/mfapi and tolerate schema variants."""
    if isinstance(raw, pd.DataFrame):
        df = raw.copy()
    elif isinstance(raw, dict):
        rows = raw.get("data", raw.get("history", raw.get("nav", [])))
        if isinstance(rows, dict):
            rows = rows.get("data", rows.get("history", []))
        df = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()
    elif isinstance(raw, list):
        df = pd.DataFrame(raw)
    else:
        df = pd.DataFrame()
    if df.empty:
        return df

    # Normalize common casing/field aliases returned by different clients.
    aliases = {}
    for column in df.columns:
        key = re.sub(r"[^a-z0-9]", "", str(column).lower())
        if key in {"date", "navdate", "valuedate"}:
            aliases[column] = "date"
        elif key in {"nav", "netassetvalue", "value"}:
            aliases[column] = "nav"
    df = df.rename(columns=aliases)
    if not {"date", "nav"}.issubset(df.columns):
        return pd.DataFrame()
    # mftool returns DD-MM-YYYY. Some fallback sources use ISO dates.
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"])
    df = df[df["nav"] > 0]
    if df.empty:
        return df
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    # NAV is an end-of-day series. Weekend rows, if any, are not useful for
    # annualized volatility and can otherwise create artificial zero returns.
    df = df[df["date"].dt.dayofweek < 5]
    return df.reset_index(drop=True)


def _fetch_history_fallback(code: str) -> dict:
    """Fallback historical NAV source when mftool is unavailable/broken.

    AMFI's NAV history remains the authoritative source; this public mirror is
    only used as a resilience fallback so one mftool failure does not erase all
    quantitative risk metrics.
    """
    if not code:
        return {}
    url = f"https://api.mfapi.in/mf/{code}"
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 MF Research Dashboard/2.4"}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        logger.debug("Historical NAV fallback failed for %s: %s", code, exc)
        return {}


def _historical_metrics(mf: Mftool | None, code: str) -> dict:
    try:
        raw = mf.get_scheme_historical_nav(code) if mf is not None and code else None
    except Exception as exc:
        logger.debug("mftool historical NAV failed for %s: %s", code, exc)
        raw = None

    df = _parse_nav_history(raw)

    # Some mftool versions return enough observations for CAGR but an
    # unexpectedly sparse series for risk statistics. In that case prefer the
    # full public NAV mirror before calculating volatility/Sharpe/Sortino.
    if code and (df.empty or len(df) < 30):
        fallback = _parse_nav_history(_fetch_history_fallback(code))
        if len(fallback) > len(df):
            df = fallback

    if df.empty or len(df) < 2:
        logger.debug("NAV history unavailable/insufficient for %s: %d observations", code, len(df))
        return {}

    latest = float(df.iloc[-1]["nav"])
    latest_date = df.iloc[-1]["date"]
    first_date = df.iloc[0]["date"]
    result: dict[str, object] = {
        "history_start_date": first_date.date().isoformat(),
        "history_end_date": latest_date.date().isoformat(),
        "history_days": int(len(df)),
        "history_years": round(max(0.0, (latest_date - first_date).days / 365.25), 2),
        "latest_nav": latest,
        "latest_nav_date": latest_date.date().isoformat(),
    }

    for years in (1, 3, 5):
        target = latest_date - timedelta(days=int(years * 365.25))
        prior = df[df["date"] <= target]
        if prior.empty:
            continue
        old = float(prior.iloc[-1]["nav"])
        if old > 0 and latest > 0:
            result[f"cagr_{years}y_pct"] = round(((latest / old) ** (1 / years) - 1) * 100, 2)

    risk_start = latest_date - timedelta(days=int(5 * 365.25))
    risk_df = df[df["date"] >= risk_start].copy()
    nav = risk_df["nav"].astype(float)
    returns = nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    # Normal daily series: annualize using 252 observations. If the source is
    # monthly/weekly, use the observed median spacing rather than pretending
    # sparse observations are daily. This preserves evidence quality.
    if len(returns) >= 30:
        deltas = risk_df["date"].diff().dt.days.dropna()
        median_gap = float(deltas.median()) if not deltas.empty else 1.0
        periods_per_year = 252.0 if median_gap <= 3 else (52.0 if median_gap <= 10 else 12.0)
        rf_annual = float(os.getenv("RISK_FREE_RATE_PCT", "6.0")) / 100.0
        rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
        vol = float(returns.std(ddof=1))
        if vol > 1e-8:
            result["volatility_pct"] = round(vol * np.sqrt(periods_per_year) * 100, 2)
            excess = returns - rf_period
            excess_std = float(excess.std(ddof=1))
            if excess_std > 1e-8:
                sharpe = float(excess.mean() / excess_std * np.sqrt(periods_per_year))
                if -10.0 <= sharpe <= 10.0:
                    result["sharpe"] = round(sharpe, 3)
            downside = excess[excess < 0]
            downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
            if downside_std > 1e-8:
                sortino = float(excess.mean() / downside_std * np.sqrt(periods_per_year))
                if -15.0 <= sortino <= 15.0:
                    result["sortino"] = round(sortino, 3)

        running_max = nav.cummax()
        drawdown = nav / running_max - 1.0
        result["max_drawdown_pct"] = round(float(drawdown.min() * 100), 2)

    logger.debug(
        "NAV history metrics for %s: observations=%d years=%.2f CAGR3=%s CAGR5=%s vol=%s drawdown=%s sharpe=%s sortino=%s",
        code, len(df), result.get("history_years", 0), result.get("cagr_3y_pct"),
        result.get("cagr_5y_pct"), result.get("volatility_pct"), result.get("max_drawdown_pct"),
        result.get("sharpe"), result.get("sortino"),
    )
    return result


def _benchmark_from_name(name: str) -> str | None:
    """Infer a benchmark only when the scheme name explicitly identifies an index.

    Active-fund benchmarks are not guessed. They remain unknown unless supplied
    by a local factsheet/enrichment file.
    """
    n = _normalise_scheme_text(name)
    patterns = [
        (r"\bnifty\s+smallcap\s+250\b", "Nifty Smallcap 250 TRI"),
        (r"\bnifty\s+smallcap\s+100\b", "Nifty Smallcap 100 TRI"),
        (r"\bnifty\s+smallcap\s+50\b", "Nifty Smallcap 50 TRI"),
        (r"\bnifty\s+midcap\s+150\b", "Nifty Midcap 150 TRI"),
        (r"\bnifty\s+midcap\s+100\b", "Nifty Midcap 100 TRI"),
        (r"\bnifty\s+financial\s+services\b", "Nifty Financial Services TRI"),
        (r"\bnifty\s+next\s+50\b", "Nifty Next 50 TRI"),
        (r"\bnifty\s+500\b", "Nifty 500 TRI"),
        (r"\bnifty\s+200\b", "Nifty 200 TRI"),
        (r"\bnifty\s+100\b", "Nifty 100 TRI"),
        (r"\bnifty\s+50\b", "Nifty 50 TRI"),
        (r"\bnifty\s+bank\b", "Nifty Bank TRI"),
        (r"\bnifty\s+it\b", "Nifty IT TRI"),
        (r"\bnifty\s+pharma\b", "Nifty Pharma TRI"),
        (r"\bnifty.*?equal\s+weight", "Relevant Nifty Equal Weight TRI"),
        (r"\bsensex\b", "BSE Sensex TRI"),
    ]
    for pattern, benchmark in patterns:
        if re.search(pattern, n):
            return benchmark
    return None



def _valid_nav_date(value: object, *, warn: bool = False) -> str | None:
    """Return an ISO date; known AMFI/mftool plan labels are silently rejected."""
    raw = str(value or "").strip()
    if not raw:
        return None
    label = _normalise_scheme_text(raw)
    if label in {"growth", "growth option", "retail plan growth", "idcw", "idcw option", "dividend", "dividend option"}:
        return None
    parsed = pd.to_datetime(raw, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        if warn:
            logger.warning("INVALID_NAV_DATE: %r", raw)
        return None
    # Guard against pandas accepting arbitrary numeric/label-like values.
    if parsed.year < 1990 or parsed.year > datetime.now().year + 1:
        if warn:
            logger.warning("INVALID_NAV_DATE_RANGE: %r", raw)
        return None
    return parsed.date().isoformat()


def _detail_value(details: dict, *names: str):
    """Case-insensitive lookup across common mftool detail keys."""
    if not isinstance(details, dict):
        return None
    normalized = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in details.items()}
    for name in names:
        value = normalized.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if value not in (None, "", "-"):
            return value
    return None


def _benchmark_from_details(details: dict) -> str | None:
    value = _detail_value(details, "benchmark", "benchmark_name", "benchmark_index", "bench_mark")
    if isinstance(value, dict):
        value = _detail_value(value, "name", "benchmark", "index")
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text or text.lower() in {"not applicable", "n/a", "na", "-"}:
        return None
    return text



def _normalise_benchmark_key(value: object) -> str:
    text = _normalise_scheme_text(str(value or ""))
    # Performance endpoints often append plan/option labels that are irrelevant
    # to benchmark identity. Keep the scheme identity, remove those labels.
    tokens = [t for t in text.split() if t not in {
        "direct", "regular", "growth", "option", "idcw", "dividend",
        "payout", "reinvestment", "plan", "monthly", "quarterly", "weekly",
    }]
    return " ".join(tokens)


def _extract_benchmark_rows(payload: object) -> list[dict]:
    rows: list[dict] = []
    if isinstance(payload, dict):
        for value in payload.values():
            rows.extend(_extract_benchmark_rows(value))
    elif isinstance(payload, list):
        for value in payload:
            if isinstance(value, dict):
                rows.append(value)
            elif isinstance(value, list):
                rows.extend(_extract_benchmark_rows(value))
    return rows


def _load_benchmark_map(mf: Mftool | None) -> dict[str, str]:
    """Load benchmark data from mftool's scheme-performance endpoints.

    get_scheme_details() does not expose benchmarks in the current mftool
    implementation. Its daily performance endpoints do, so use those in a
    small number of category-level requests rather than making one request per
    fund.
    """
    if mf is None:
        return {}
    methods = (
        "get_open_ended_equity_scheme_performance",
        "get_open_ended_debt_scheme_performance",
        "get_open_ended_hybrid_scheme_performance",
        "get_open_ended_solution_scheme_performance",
        "get_open_ended_other_scheme_performance",
    )
    result: dict[str, str] = {}
    for method_name in methods:
        method = getattr(mf, method_name, None)
        if not callable(method):
            continue
        try:
            payload = method()
            for row in _extract_benchmark_rows(payload):
                name = row.get("scheme_name") or row.get("schemeName")
                benchmark = row.get("benchmark") or row.get("benchmark_name")
                if not name or not benchmark:
                    continue
                benchmark_text = re.sub(r"\s+", " ", str(benchmark)).strip()
                if benchmark_text and benchmark_text.lower() not in {"na", "n/a", "-", "not applicable"}:
                    result[_normalise_benchmark_key(name)] = benchmark_text
        except Exception as exc:
            logger.debug("Benchmark performance endpoint %s failed: %s", method_name, exc)
    logger.info("Benchmark enrichment source: %d scheme benchmarks loaded from mftool performance endpoints.", len(result))
    return result


def _apply_benchmark_map(records: list[FundRecord], benchmark_map: dict[str, str]) -> int:
    if not benchmark_map:
        return 0
    matched = 0
    for record in records:
        if record.benchmark:
            continue
        key = _normalise_benchmark_key(record.scheme_name)
        benchmark = benchmark_map.get(key)
        if benchmark:
            record.benchmark = benchmark
            matched += 1
    return matched

def _fund_record(mf: Mftool | None, item: dict, holdings: dict[str, Holding]) -> FundRecord:
    name = item["scheme_name"]
    code = str(item.get("code", ""))
    quote = {}
    details = {}
    if code and mf is not None:
        try:
            quote = mf.get_scheme_quote(code) or {}
            details = mf.get_scheme_details(code) or {}
        except Exception as exc:
            logger.debug("mftool quote/details failed for %s: %s", name, exc)

    hist = _historical_metrics(mf, code) if code else {}
    category = details.get("scheme_category") or item.get("category") or classify_category(name, item.get("amfi_category"))
    amc = details.get("fund_house") or item.get("amc") or "Unknown"
    nav = safe_float(quote.get("nav"))
    if nav is None:
        nav = safe_float(item.get("nav"))
    if nav is None:
        nav = safe_float(hist.get("latest_nav"))

    # AMFI catalog date is authoritative for the current quote. Prefer it so
    # mftool's inconsistent `last_updated` plan/option field is never parsed.
    nav_date = _valid_nav_date(item.get("date"))
    if nav_date is None:
        nav_date = _valid_nav_date(hist.get("latest_nav_date"))
    if nav_date is None:
        nav_date = _valid_nav_date(quote.get("last_updated"))

    benchmark = _benchmark_from_details(details) or _benchmark_from_name(name)

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
        volatility_pct=hist.get("volatility_pct"),
        max_drawdown_pct=hist.get("max_drawdown_pct"),
        sharpe=hist.get("sharpe"),
        sortino=hist.get("sortino"),
        benchmark=benchmark,
    )


def _parse_amfi_catalog(text: str) -> dict[str, dict]:
    """
    Parse AMFI NAVAll.txt defensively.

    Supported row layouts:
      6 columns: code;isin1;isin2;scheme_name;nav;date
      8 columns: code;isin1;isin2;scheme_name;plan;option;nav;date

    AMFI also emits AMC/category section headers. Those are retained as
    metadata so category/AMC information is not lost before filtering.
    """
    catalog: dict[str, dict] = {}
    current_amc = None
    current_category = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(";")]

        # Section/AMC headers do not contain a scheme code.
        if not parts[0].isdigit():
            if ";" not in line:
                header = line.strip()
                if "scheme" in header.lower() and "open ended" not in header.lower():
                    continue
                if header.startswith("(") or "scheme" in header.lower():
                    current_category = header
                elif header:
                    current_amc = header
            continue

        if len(parts) >= 8:
            code, isin_div, isin_reinv, scheme_name, plan, option, nav, date = parts[:8]
        elif len(parts) >= 6:
            code, isin_div, isin_reinv, scheme_name, nav, date = parts[:6]
            plan = ""
            option = ""
        else:
            continue

        if not scheme_name:
            continue

        catalog[code] = {
            "scheme_name": scheme_name,
            "nav": nav,
            "date": date,
            "isin_growth": isin_div,
            "isin_div": isin_reinv,
            "plan": plan,
            "option": option,
            "amc": current_amc,
            "amfi_category": current_category,
        }

    return catalog


def _load_cached_catalog(settings: Settings) -> dict[str, dict] | None:
    cache_dir = Path(settings.market_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "amfi_nav_catalog_v2.json"
    ttl_hours = float(os.getenv("AMFI_CATALOG_CACHE_HOURS", "12"))
    if not path.exists():
        return None
    try:
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > ttl_hours:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_catalog(settings: Settings, catalog: dict[str, dict]) -> None:
    path = Path(settings.market_cache_dir) / "amfi_nav_catalog_v2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")


def fetch_amfi_catalog(settings: Settings) -> dict[str, dict]:
    cached = _load_cached_catalog(settings)
    if cached:
        cached_direct = sum(
            1 for item in cached.values()
            if is_direct_growth_scheme(
                item.get("scheme_name", ""),
                item.get("plan"),
                item.get("option"),
            )
        )
        # A previous cache version did not retain Plan/Option and can therefore
        # silently produce an empty universe. Do not trust such a cache.
        if cached_direct > 0:
            logger.info(
                "AMFI catalog loaded from cache: %d schemes (%d Direct/Growth).",
                len(cached), cached_direct
            )
            return cached
        logger.warning(
            "AMFI cache contains %d schemes but 0 Direct/Growth records; "
            "refreshing cache with current NAVAll schema.",
            len(cached),
        )

    logger.info("Fetching AMFI scheme catalog from %s", AMFI_NAV_URL)
    headers = {"User-Agent": "Mozilla/5.0 MF Research Dashboard/2.2"}
    try:
        response = requests.get(AMFI_NAV_URL, headers=headers, timeout=30)
        response.raise_for_status()
        catalog = _parse_amfi_catalog(response.text)
        if not catalog:
            raise RuntimeError("AMFI returned an empty/unparseable NAV catalog")

        direct_count = sum(
            1 for item in catalog.values()
            if is_direct_growth_scheme(
                item.get("scheme_name", ""),
                item.get("plan"),
                item.get("option"),
            )
        )
        logger.info(
            "AMFI parser produced %d schemes, including %d Direct/Growth.",
            len(catalog), direct_count
        )
        if direct_count == 0:
            raise RuntimeError(
                "AMFI catalog parsed successfully but contains 0 Direct/Growth "
                "schemes; feed schema may have changed."
            )

        _save_catalog(settings, catalog)
        logger.info("AMFI catalog fetched successfully: %d schemes.", len(catalog))
        return catalog
    except Exception as exc:
        logger.warning("AMFI catalog fetch failed: %s", exc)
        return {}


def _mftool_catalog(mf: Mftool | None) -> dict[str, str]:
    if mf is None:
        return {}
    try:
        schemes = mf.get_scheme_codes() or {}
        if isinstance(schemes, dict):
            logger.info("mftool catalog returned %d schemes.", len(schemes))
            return {str(k): str(v) for k, v in schemes.items()}
    except Exception as exc:
        logger.warning("mftool scheme catalog failed: %s", exc)
    return {}


def fetch_universe(settings: Settings, mf: Mftool) -> list[dict]:
    """Build a deterministic candidate universe with AMFI as the primary fallback."""
    amfi = fetch_amfi_catalog(settings)
    mftool = _mftool_catalog(mf)

    # AMFI is preferred because it gives us a complete current scheme catalogue.
    if amfi:
        source = "AMFI"
        schemes = amfi
        items = [(code, item["scheme_name"], item) for code, item in schemes.items()]
    elif mftool:
        source = "mftool"
        items = [(code, name, {"scheme_name": name}) for code, name in mftool.items()]
    else:
        logger.error("No mutual-fund scheme catalogue is available from AMFI or mftool.")
        return []

    direct_growth = 0
    categorized = 0
    candidates = []
    per_category: dict[str, int] = {}
    per_amc_category: dict[tuple[str, str], int] = {}

    direct_samples: list[str] = []
    category_samples: list[str] = []

    for code, name, meta in sorted(items, key=lambda x: str(x[1]).lower()):
        if not is_direct_growth_scheme(
            name,
            meta.get("plan"),
            meta.get("option"),
        ):
            continue

        direct_growth += 1
        if len(direct_samples) < 5:
            direct_samples.append(str(name))

        category = classify_category(name, meta.get("amfi_category"))
        if category == "Other":
            continue

        categorized += 1
        if len(category_samples) < 5:
            category_samples.append(f"{name} -> {category}")

        # Use explicit AMFI AMC metadata when available. Fall back to the
        # first token only for older/mftool-only records.
        amc = str(meta.get("amc") or str(name).split()[0]).strip().lower()
        key = (category, amc)
        if per_category.get(category, 0) >= settings.max_schemes_per_category:
            continue
        if per_amc_category.get(key, 0) >= settings.max_per_amc_per_category:
            continue
        candidates.append({
            "scheme_name": name,
            "category": category,
            "amc": amc,
            "code": str(code),
            "nav": meta.get("nav"),
            "date": meta.get("date"),
        })
        per_category[category] = per_category.get(category, 0) + 1
        per_amc_category[key] = per_amc_category.get(key, 0) + 1

    logger.info(
        "Universe pipeline [%s]: catalog=%d, direct_growth=%d, recognized_category=%d, selected=%d.",
        source, len(items), direct_growth, categorized, len(candidates),
    )
    if direct_samples:
        logger.info("Direct/Growth samples: %s", " | ".join(direct_samples))
    if category_samples:
        logger.info("Category samples: %s", " | ".join(category_samples))
    if not candidates:
        logger.error(
            "Universe is empty. Check AMFI connectivity, Direct/Growth naming, and category rules."
        )
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
    logger.info("Starting mutual-fund data pipeline...")
    try:
        mf = Mftool()
    except Exception as exc:
        logger.warning("mftool initialization failed; using AMFI + historical fallback: %s", exc)
        mf = None
    raw = fetch_universe(settings, mf)
    held_names = {name.lower(): name for name in holdings}
    for name in holdings:
        if not any(x["scheme_name"].lower() == name.lower() for x in raw):
            raw.append({"scheme_name": name, "category": "Holding Scheme", "amc": "Unknown", "code": ""})
    logger.info("Candidate universe after existing holdings: %d funds.", len(raw))

    records: list[FundRecord] = []
    if not raw:
        return records

    # mftool is not guaranteed to be thread-safe. Use separate clients per worker
    # rather than sharing one mutable client across threads.
    def process(item: dict):
        try:
            client = Mftool()
        except Exception:
            client = None
        return _fund_record(client, item, holdings)

    workers = max(1, min(4, int(os.getenv("MF_DATA_WORKERS", "3"))))
    logger.info("Fetching fund NAV/history with %d workers...", workers)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, item) for item in raw]
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            try:
                record = future.result()
                extra = load_optional_factsheet_data(settings, record.scheme_name)
                for key in ("aum_inr_cr", "inception_date", "volatility_pct", "max_drawdown_pct", "sharpe", "sortino", "benchmark"):
                    if key in extra:
                        setattr(record, key, extra[key])
                for key in ("sector_weights", "holdings", "market_cap_weights", "valuation"):
                    if isinstance(extra.get(key), dict):
                        setattr(record, key, extra[key])
                records.append(record)
            except Exception as exc:
                logger.warning("Fund processing failed: %s", exc)
            if completed % 10 == 0 or completed == len(raw):
                logger.info("Fund data progress: %d/%d completed.", completed, len(raw))

    # Enrich benchmarks in one batch after the concurrent NAV/history phase.
    # mftool's per-scheme details endpoint does not contain benchmark data;
    # its category performance endpoints do.
    try:
        benchmark_map = _load_benchmark_map(mf)
        benchmark_matches = _apply_benchmark_map(records, benchmark_map)
        if benchmark_matches:
            logger.info("Benchmark enrichment: matched %d additional fund records.", benchmark_matches)
    except Exception as exc:
        logger.debug("Benchmark batch enrichment failed: %s", exc)

    records.sort(key=lambda x: (x.category.lower(), x.scheme_name.lower()))
    if records:
        logger.info(
            "Fund enrichment coverage: CAGR3=%d/%d CAGR5=%d/%d Vol=%d/%d Drawdown=%d/%d Sharpe=%d/%d Benchmark=%d/%d NAVdate=%d/%d",
            sum(x.cagr_3y_pct is not None for x in records), len(records),
            sum(x.cagr_5y_pct is not None for x in records), len(records),
            sum(x.volatility_pct is not None for x in records), len(records),
            sum(x.max_drawdown_pct is not None for x in records), len(records),
            sum(x.sharpe is not None or x.sortino is not None for x in records), len(records),
            sum(bool(x.benchmark) for x in records), len(records),
            sum(x.latest_nav is not None and x.nav_date is not None for x in records), len(records),
        )
    logger.info("Fund data pipeline complete: %d/%d fund records built.", len(records), len(raw))
    return records
