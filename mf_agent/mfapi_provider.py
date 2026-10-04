from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utils import safe_float

logger = logging.getLogger("mf_agent")

MFAPI_BASE_URL = os.getenv("MFAPI_BASE_URL", "https://api.mfapi.in").rstrip("/")
MFAPI_ENABLED = os.getenv("MFAPI_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
MFAPI_CACHE_TTL_HOURS = max(1.0, float(os.getenv("MFAPI_CACHE_TTL_HOURS", "24")))
MFAPI_TIMEOUT_SECONDS = max(3, min(30, int(os.getenv("MFAPI_TIMEOUT_SECONDS", "10"))))
MFAPI_MAX_WORKERS = max(1, min(8, int(os.getenv("MFAPI_MAX_WORKERS", "4"))))
MFAPI_HISTORY_YEARS = max(5, min(15, int(os.getenv("MFAPI_HISTORY_YEARS", "6"))))
MFAPI_MIN_REQUEST_INTERVAL_SECONDS = max(
    0.0,
    float(os.getenv("MFAPI_MIN_REQUEST_INTERVAL_SECONDS", "0.25")),
)

_CACHE_LOCK = threading.Lock()
_RATE_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=1,
        connect=1,
        read=1,
        redirect=0,
        status=1,
        backoff_factor=0.25,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": "MF Research Dashboard/2.4",
        "Accept": "application/json",
    })
    return session


def _throttle() -> None:
    global _LAST_REQUEST_AT
    if MFAPI_MIN_REQUEST_INTERVAL_SECONDS <= 0:
        return
    with _RATE_LOCK:
        now = time.monotonic()
        wait = MFAPI_MIN_REQUEST_INTERVAL_SECONDS - (now - _LAST_REQUEST_AT)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_AT = time.monotonic()


def _cache_root(cache_dir: str | Path) -> Path:
    root = Path(cache_dir) / "mfapi"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_path(cache_dir: str | Path, endpoint: str, params: dict[str, Any] | None = None) -> Path:
    query = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{endpoint}?{query}".encode("utf-8")).hexdigest()
    return _cache_root(cache_dir) / f"{digest}.json"


def _read_cache(path: Path) -> Any | None:
    try:
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > MFAPI_CACHE_TTL_HOURS * 3600:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, payload: Any) -> None:
    try:
        with _CACHE_LOCK:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
    except Exception as exc:
        logger.debug("Could not cache MFapi response %s: %s", path, exc)


def _request_json(
    session: requests.Session,
    cache_dir: str | Path,
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> dict | list | None:
    cache = _cache_path(cache_dir, endpoint, params)
    cached = _read_cache(cache)
    if cached is not None:
        return cached

    _throttle()
    url = f"{MFAPI_BASE_URL}{endpoint}"
    try:
        response = session.get(url, params=params, timeout=MFAPI_TIMEOUT_SECONDS)
        if response.status_code >= 400:
            logger.warning("MFapi GET %s returned HTTP %s", endpoint, response.status_code)
            return None
        payload = response.json()
        _write_cache(cache, payload)
        return payload
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        logger.warning("MFapi GET failed for %s: %s", endpoint, exc)
        return None
    except (ValueError, requests.exceptions.RequestException) as exc:
        logger.warning("MFapi GET failed for %s: %s", endpoint, exc)
        return None


def _parse_history(payload: dict | list | None) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame(columns=["date", "nav"])
    rows = payload.get("data")
    if not isinstance(rows, list):
        return pd.DataFrame(columns=["date", "nav"])

    parsed: list[tuple[pd.Timestamp, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        date_value = row.get("date") or row.get("Date")
        nav_value = safe_float(row.get("nav") if row.get("nav") is not None else row.get("NAV"))
        date_value = pd.to_datetime(date_value, dayfirst=True, errors="coerce")
        if pd.isna(date_value) or nav_value is None or nav_value <= 0:
            continue
        parsed.append((pd.Timestamp(date_value), float(nav_value)))

    if not parsed:
        return pd.DataFrame(columns=["date", "nav"])

    return (
        pd.DataFrame(parsed, columns=["date", "nav"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 2:
        return {}

    latest = float(df.iloc[-1]["nav"])
    latest_date = pd.Timestamp(df.iloc[-1]["date"])
    result = {
        "latest_nav": latest,
        "nav_date": latest_date.strftime("%Y-%m-%d"),
        "nav_history_observations": int(len(df)),
    }

    for years in (1, 3, 5):
        target = latest_date - pd.Timedelta(days=years * 365.25)
        prior = df[df["date"] <= target]
        if prior.empty:
            continue
        old = float(prior.iloc[-1]["nav"])
        if old > 0 and latest > 0:
            result[f"cagr_{years}y_pct"] = round(((latest / old) ** (1 / years) - 1) * 100, 2)

    risk_start = latest_date - pd.Timedelta(days=365.25)
    risk_df = df[df["date"] >= risk_start].copy()
    if len(risk_df) < 30:
        risk_df = df.copy()

    returns = risk_df["nav"].pct_change().dropna()
    if len(returns) >= 2:
        daily_std = float(returns.std(ddof=1))
        if daily_std > 0:
            result["volatility_pct"] = round(daily_std * (252 ** 0.5) * 100, 2)
            result["sharpe"] = round(float(returns.mean()) / daily_std * (252 ** 0.5), 3)

            downside = returns[returns < 0]
            if len(downside) >= 2:
                downside_std = float(downside.std(ddof=1))
                if downside_std > 0:
                    result["sortino"] = round(float(returns.mean()) / downside_std * (252 ** 0.5), 3)

        running_peak = risk_df["nav"].cummax()
        drawdowns = (risk_df["nav"] / running_peak - 1.0) * 100.0
        result["max_drawdown_pct"] = round(float(drawdowns.min()), 2)

    return result


def get_scheme_history(
    scheme_code: str,
    cache_dir: str | Path,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fetch a bounded NAV history for one AMFI scheme code from MFapi.in."""
    if not MFAPI_ENABLED or not scheme_code:
        return pd.DataFrame(columns=["date", "nav"]), {}

    client = session or _session()
    today = dt.date.today()
    start_date = today - dt.timedelta(days=int(MFAPI_HISTORY_YEARS * 365.25))
    endpoint = f"/mf/{scheme_code}"
    payload = _request_json(
        client,
        cache_dir,
        endpoint,
        {
            "startDate": start_date.isoformat(),
            "endDate": today.isoformat(),
        },
    )
    history = _parse_history(payload)
    metadata = payload.get("meta", {}) if isinstance(payload, dict) else {}
    return history, metadata if isinstance(metadata, dict) else {}


def fetch_histories(
    scheme_codes: list[str],
    cache_dir: str | Path,
) -> dict[str, tuple[pd.DataFrame, dict]]:
    """Fetch selected schemes concurrently with per-scheme failure isolation."""
    if not MFAPI_ENABLED:
        logger.info("MFapi NAV provider disabled by MFAPI_ENABLED=false.")
        return {}

    codes = list(dict.fromkeys(str(code).strip() for code in scheme_codes if str(code).strip()))
    if not codes:
        return {}

    results: dict[str, tuple[pd.DataFrame, dict]] = {}

    def worker(code: str):
        try:
            session = _session()
            return code, get_scheme_history(code, cache_dir, session)
        except Exception as exc:
            logger.warning("MFapi history failed for scheme %s: %s", code, exc)
            return code, (pd.DataFrame(columns=["date", "nav"]), {})

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=min(MFAPI_MAX_WORKERS, len(codes))) as pool:
        futures = [pool.submit(worker, code) for code in codes]
        for future in as_completed(futures):
            code, result = future.result()
            history, metadata = result
            if not history.empty:
                results[code] = (history, metadata)

    logger.info("MFapi NAV history enrichment: %d/%d schemes returned history.", len(results), len(codes))
    return results


def scheme_metadata(metadata: dict) -> dict:
    """Normalize the documented MFapi metadata fields used by FundRecord."""
    if not isinstance(metadata, dict):
        return {}
    return {
        "fund_house": metadata.get("fund_house"),
        "scheme_type": metadata.get("scheme_type"),
        "scheme_category": metadata.get("scheme_category"),
        "scheme_code": metadata.get("scheme_code"),
        "scheme_name": metadata.get("scheme_name"),
        "isin_growth": metadata.get("isin_growth"),
        "isin_div_reinvestment": metadata.get("isin_div_reinvestment"),
    }
