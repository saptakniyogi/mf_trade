from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import logging
import os
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from mftool import Mftool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Settings
from .mfapi_provider import MFAPI_ENABLED, fetch_histories, scheme_metadata
from .models import FundRecord, Holding
from .utils import safe_float

logger = logging.getLogger("mf_agent")

try:
    _MFTOOL_VERSION = version("mftool")
except PackageNotFoundError:
    _MFTOOL_VERSION = "not-installed"
logger.info("mftool dependency loaded: version=%s", _MFTOOL_VERSION)


_MFTOOL_LOCAL = threading.local()


def _thread_mftool() -> Mftool:
    """Return one mftool instance per worker thread to avoid sharing sessions."""
    instance = getattr(_MFTOOL_LOCAL, "instance", None)
    if instance is None:
        instance = Mftool()
        _MFTOOL_LOCAL.instance = instance
    return instance

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
MFDATA_ENABLED = os.getenv("MF_DATA_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
MFDATA_HEALTHCHECK_TIMEOUT_SECONDS = max(1, min(5, int(os.getenv("MF_DATA_HEALTHCHECK_TIMEOUT_SECONDS", "2"))))
MFDATA_CACHE_TTL_HOURS = max(1.0, float(os.getenv("MF_DATA_CACHE_TTL_HOURS", "24")))
MFDATA_FAMILY_ENRICHMENT_LIMIT = max(0, min(20, int(os.getenv("MF_DATA_FAMILY_ENRICHMENT_LIMIT", "20"))))
MFDATA_TIMEOUT_SECONDS = max(2, min(8, int(os.getenv("MF_DATA_TIMEOUT_SECONDS", "5"))))
MFDATA_BULK_CHUNK_SIZE = max(25, min(100, int(os.getenv("MF_DATA_BULK_CHUNK_SIZE", "50"))))
MFDATA_MAX_WORKERS = max(1, min(12, int(os.getenv("MF_DATA_MAX_WORKERS", "6"))))
MFDATA_INDIVIDUAL_FALLBACK_LIMIT = max(0, min(20, int(os.getenv("MF_DATA_INDIVIDUAL_FALLBACK_LIMIT", "20"))))
MFTOOL_PERFORMANCE_CACHE_TTL_HOURS = max(1.0, float(os.getenv("MFTOOL_PERFORMANCE_CACHE_TTL_HOURS", "24")))

# TigZig publishes an AMFI-derived, normalized scheme snapshot and historical
# NAV API. It is the primary NAV provider because it is refreshed from AMFI and
# exposes both current scheme metadata and full scheme histories without API keys.
TIGZIG_NAV_ENABLED = os.getenv("MF_TIGZIG_NAV_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
TIGZIG_NAV_BASE_URL = os.getenv("MF_TIGZIG_NAV_BASE_URL", "https://api.tigzig.com/mf/v1").rstrip("/")
TIGZIG_NAV_CACHE_DIR = os.getenv("MF_TIGZIG_NAV_CACHE_DIR", "tigzig_nav_cache")
TIGZIG_NAV_CACHE_TTL_HOURS = max(1.0, float(os.getenv("MF_TIGZIG_NAV_CACHE_TTL_HOURS", "24")))
TIGZIG_NAV_SOURCE_QUALITY = 0.98
TIGZIG_NAV_MAX_WORKERS = max(1, min(8, int(os.getenv("MF_TIGZIG_NAV_MAX_WORKERS", "4"))))
TIGZIG_NAV_BULK_SIZE = max(1, min(50, int(os.getenv("MF_TIGZIG_NAV_BULK_SIZE", "50"))))

# Creget is a second AMFI-derived archive. It is used for the latest snapshot
# if TigZig is unavailable, before falling back to mftool/AMFI.
CREGET_NAV_ENABLED = os.getenv("MF_CREGET_NAV_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
CREGET_NAV_LATEST_URL = os.getenv(
    "MF_CREGET_NAV_LATEST_URL",
    "https://raw.githubusercontent.com/shahroz-a/mutual-fund-historical-data/mutual-fund-historical-data/data/latest.csv",
).strip()
CREGET_NAV_CACHE_DIR = os.getenv("MF_CREGET_NAV_CACHE_DIR", "creget_nav_cache")
CREGET_NAV_SOURCE_QUALITY = 0.97

# Kaggle remains an optional tertiary NAV-history backup.
# It is deliberately lazy-loaded so normal runs do not require Kaggle credentials
# or a large dataset download unless the stronger AMFI-derived sources are unavailable.
# Kaggle is an optional secondary/tertiary NAV-history backup. It is deliberately
# lazy-loaded so normal runs do not require Kaggle credentials or a large dataset
# download unless a fund is still missing historical NAV metrics.
KAGGLE_NAV_ENABLED = os.getenv("MF_ENABLE_LEGACY_KAGGLE_FALLBACK", "false").strip().lower() not in {"0", "false", "no"}
KAGGLE_NAV_DATASET = os.getenv(
    "MF_KAGGLE_NAV_DATASET",
    "tharunreddy2911/mutual-fund-historic-nav-data",
).strip()
KAGGLE_NAV_CACHE_DIR = os.getenv("MF_KAGGLE_NAV_CACHE_DIR", "kaggle_nav_cache")
KAGGLE_NAV_MAX_STALENESS_DAYS = max(1, int(os.getenv("MF_KAGGLE_NAV_MAX_STALENESS_DAYS", "45")))
KAGGLE_NAV_CACHE_TTL_HOURS = max(1.0, float(os.getenv("MF_KAGGLE_NAV_CACHE_TTL_HOURS", "24")))
KAGGLE_NAV_CHUNK_SIZE = max(10_000, int(os.getenv("MF_KAGGLE_NAV_CHUNK_SIZE", "100000")))
KAGGLE_NAV_SOURCE_QUALITY = 0.96
KAGGLE_REFRESH_WEEKENDS = os.getenv("MF_KAGGLE_REFRESH_WEEKENDS", "true").strip().lower() not in {"0", "false", "no"}

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


def _parse_nav_history_rows(rows) -> pd.DataFrame:
    """Normalize common AMFI-derived NAV history response shapes."""
    if not isinstance(rows, list):
        return pd.DataFrame()

    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        date_value = row.get("date") or row.get("Date") or row.get("nav_date")
        nav_value = row.get("nav") or row.get("NAV") or row.get("Net Asset Value")
        date_value = pd.to_datetime(date_value, dayfirst=True, errors="coerce")
        nav_value = safe_float(nav_value)
        if pd.isna(date_value) or nav_value is None or nav_value <= 0:
            continue
        parsed.append((date_value, float(nav_value)))

    if not parsed:
        return pd.DataFrame(columns=["date", "nav"])

    df = pd.DataFrame(parsed, columns=["date", "nav"])
    return (
        df.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _calculate_nav_metrics(df: pd.DataFrame) -> dict:
    """Calculate CAGR and risk metrics from an AMFI NAV history."""
    if df.empty or len(df) < 2:
        return {}

    latest_row = df.iloc[-1]
    latest = float(latest_row["nav"])
    latest_date = pd.Timestamp(latest_row["date"])
    result: dict = {
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

    # Risk statistics describe the recent risk regime rather than the fund's
    # entire lifetime. Use one year of NAV history where available.
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


def _tigzig_cache_root(settings: Settings) -> Path:
    root = Path(settings.market_cache_dir) / TIGZIG_NAV_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tigzig_refresh_marker(settings: Settings) -> Path:
    return _tigzig_cache_root(settings) / "refresh_state.json"


def _read_tigzig_refresh_state(settings: Settings) -> dict:
    path = _tigzig_refresh_marker(settings)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _tigzig_should_refresh(settings: Settings) -> bool:
    state = _read_tigzig_refresh_state(settings)
    return str(state.get("refresh_date") or "") != dt.date.today().isoformat()


def _download_tigzig_latest(settings: Settings) -> Path | None:
    """Refresh the AMFI-derived TigZig scheme snapshot at most once per day."""
    if not TIGZIG_NAV_ENABLED:
        return None
    root = _tigzig_cache_root(settings)
    target = root / "latest.csv"
    if target.exists() and not _tigzig_should_refresh(settings):
        logger.info("TigZig NAV snapshot cache hit: %s", target)
        return target
    url = f"{TIGZIG_NAV_BASE_URL}/download?format=latest"
    try:
        response = requests.get(url, timeout=max(10, MFDATA_TIMEOUT_SECONDS))
        response.raise_for_status()
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(response.content)
        tmp.replace(target)
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        state = {"refreshed_at": now, "refresh_date": now[:10], "url": url}
        marker = _tigzig_refresh_marker(settings)
        marker.write_text(json.dumps(state, indent=2), encoding="utf-8")
        logger.info("TigZig NAV snapshot refreshed: %s (%d bytes)", target, len(response.content))
        return target
    except Exception as exc:
        if target.exists():
            logger.warning("TigZig NAV snapshot refresh failed; using last successful snapshot: %s", exc)
            return target
        logger.warning("TigZig NAV snapshot unavailable: %s", exc)
        return None


def _tigzig_cache_path(settings: Settings, record: FundRecord) -> Path:
    identity = f"{record.scheme_code}|{record.scheme_name}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return _tigzig_cache_root(settings) / f"{digest}.json"


def _read_tigzig_history_cache(settings: Settings, record: FundRecord) -> pd.DataFrame | None:
    path = _tigzig_cache_path(settings, record)
    try:
        if not path.exists() or time.time() - path.stat().st_mtime > TIGZIG_NAV_CACHE_TTL_HOURS * 3600:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return None
        return _parse_nav_history_rows(rows)
    except Exception:
        return None


def _write_tigzig_history_cache(settings: Settings, record: FundRecord, df: pd.DataFrame) -> None:
    try:
        rows = [{"date": row.date.strftime("%Y-%m-%d"), "nav": float(row.nav)} for row in df.itertuples(index=False)]
        path = _tigzig_cache_path(settings, record)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"data": rows}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.debug("Could not cache TigZig NAV history for %s: %s", record.scheme_name, exc)


def _load_tigzig_snapshot(settings: Settings, records: list[FundRecord]) -> dict[str, dict]:
    path = _download_tigzig_latest(settings)
    if path is None:
        return {}
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        logger.warning("Could not read TigZig NAV snapshot %s: %s", path, exc)
        return {}

    columns = list(frame.columns)
    code_col = _find_kaggle_column(columns, ("scheme_code", "scheme code", "amfi code", "amfi_code"))
    name_col = _find_kaggle_column(columns, ("scheme_name", "scheme name"))
    nav_col = _find_kaggle_column(columns, ("nav", "latest nav", "latest_nav"))
    nav_date_col = _find_kaggle_column(columns, ("nav_date", "nav date", "latest nav date", "latest_nav_date"))
    amc_col = _find_kaggle_column(columns, ("amc",))
    category_col = _find_kaggle_column(columns, ("category_sub", "category", "scheme category", "scheme_category"))
    aum_col = _find_kaggle_column(columns, ("aaum_cr_quarterly_avg", "average aum", "average_aum", "aaum"))
    plan_col = _find_kaggle_column(columns, ("plan", "plan_type"))
    inception_col = _find_kaggle_column(columns, ("first_nav_date", "first nav date", "launch date", "inception date"))
    if not code_col and not name_col:
        logger.warning("TigZig NAV snapshot has no scheme identifier columns: %s", columns)
        return {}

    by_code = {_kaggle_code_key(r.scheme_code): r for r in records if r.scheme_code}
    by_name = {_kaggle_name_key(r.scheme_name): r for r in records}
    result: dict[str, dict] = {}
    matched = 0
    for row in frame.to_dict(orient="records"):
        code = _kaggle_code_key(row.get(code_col)) if code_col else ""
        name = str(row.get(name_col) or "").strip() if name_col else ""
        record = by_code.get(code) if code else None
        if record is None and name:
            record = by_name.get(_kaggle_name_key(name))
        if record is None:
            continue
        result[record.scheme_code or record.scheme_name] = {
            "scheme_name": name,
            "latest_nav": safe_float(row.get(nav_col)) if nav_col else None,
            "nav_date": str(row.get(nav_date_col) or "").strip() if nav_date_col else "",
            "amc": str(row.get(amc_col) or "").strip() if amc_col else "",
            "category": str(row.get(category_col) or "").strip() if category_col else "",
            "aum_inr_cr": safe_float(row.get(aum_col)) if aum_col else None,
            "plan_type": str(row.get(plan_col) or "").strip() if plan_col else "",
            "inception_date": str(row.get(inception_col) or "").strip() if inception_col else "",
        }
        matched += 1
    logger.info("TigZig NAV snapshot matched %d/%d selected funds.", matched, len(records))
    return result


def _enrich_from_tigzig_snapshot(settings: Settings, records: list[FundRecord]) -> None:
    snapshot = _load_tigzig_snapshot(settings, records)
    for record in records:
        values = snapshot.get(record.scheme_code or record.scheme_name)
        if not values:
            continue
        for field in ("latest_nav", "nav_date", "aum_inr_cr", "inception_date"):
            value = values.get(field)
            if value is not None and value != "":
                setattr(record, field, value)
                record.data_sources[field] = "TigZig / AMFI NAV snapshot"
        for field in ("amc", "category"):
            value = values.get(field)
            if value:
                setattr(record, field, value)
                record.data_sources[field] = "TigZig / AMFI NAV snapshot"
        if values.get("plan_type"):
            plan = values["plan_type"].lower()
            if "direct" in plan:
                record.plan_type = "Direct Growth" if "growth" in plan else "Direct"
            elif "regular" in plan:
                record.plan_type = "Regular Growth" if "growth" in plan else "Regular"
        record.source_quality = max(record.source_quality, TIGZIG_NAV_SOURCE_QUALITY)


def _load_tigzig_histories(settings: Settings, records: list[FundRecord]) -> dict[str, pd.DataFrame]:
    candidates = [r for r in records if r.scheme_code]
    histories: dict[str, pd.DataFrame] = {}
    missing: list[FundRecord] = []
    for record in candidates:
        cached = _read_tigzig_history_cache(settings, record)
        if cached is not None and not cached.empty:
            histories[record.scheme_code] = cached
        else:
            missing.append(record)
    if not missing:
        return histories
    for start in range(0, len(missing), TIGZIG_NAV_BULK_SIZE):
        batch = missing[start:start + TIGZIG_NAV_BULK_SIZE]
        schemes = ",".join(_kaggle_code_key(r.scheme_code) for r in batch if r.scheme_code)
        url = f"{TIGZIG_NAV_BASE_URL}/nav"
        try:
            response = requests.get(url, params={"schemes": schemes}, timeout=max(15, MFDATA_TIMEOUT_SECONDS))
            response.raise_for_status()
            payload = response.json()
            scheme_payload = payload.get("schemes", {}) if isinstance(payload, dict) else {}
            if not isinstance(scheme_payload, dict):
                continue
            for record in batch:
                item = scheme_payload.get(str(record.scheme_code))
                if not isinstance(item, dict):
                    continue
                df = _parse_nav_history_rows(item.get("data", []))
                if df.empty:
                    continue
                histories[record.scheme_code] = df
                _write_tigzig_history_cache(settings, record, df)
        except Exception as exc:
            logger.warning("TigZig NAV history batch failed for %d schemes: %s", len(batch), exc)
    return histories


def _enrich_from_tigzig_nav(settings: Settings, records: list[FundRecord]) -> None:
    candidates = [r for r in records if any(getattr(r, field) is None for field in ("latest_nav", "cagr_1y_pct", "cagr_3y_pct", "cagr_5y_pct", "volatility_pct", "max_drawdown_pct", "sharpe", "sortino")) and r.scheme_code]
    if not candidates:
        return
    histories = _load_tigzig_histories(settings, candidates)
    enriched = 0
    for record in candidates:
        history = histories.get(record.scheme_code)
        if history is None or history.empty:
            continue
        metrics = _calculate_nav_metrics(history)
        if not metrics:
            continue
        for field in ("latest_nav", "cagr_1y_pct", "cagr_3y_pct", "cagr_5y_pct", "volatility_pct", "max_drawdown_pct", "sharpe", "sortino"):
            value = metrics.get(field)
            if value is not None:
                setattr(record, field, value); record.data_sources[field] = "TigZig / AMFI NAV history"
        if metrics.get("nav_date"):
            record.nav_date = metrics["nav_date"]; record.data_sources["nav_date"] = "TigZig / AMFI NAV history"
        record.nav_history_observations = metrics.get("nav_history_observations")
        record.source_quality = max(record.source_quality, TIGZIG_NAV_SOURCE_QUALITY)
        enriched += 1
    logger.info("TigZig NAV history enrichment: enriched=%d/%d", enriched, len(candidates))


def _parse_kaggle_date(value):
    text = str(value or "").strip()
    if not text:
        return pd.NaT
    try:
        if len(text) >= 8 and text[0:4].isdigit() and text[4] in {"-", "/"}:
            return pd.to_datetime(text, errors="coerce", dayfirst=False)
        return pd.to_datetime(text, errors="coerce", dayfirst=True)
    except Exception:
        return pd.NaT


def _kaggle_code_key(value) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def _kaggle_name_key(value: str) -> str:
    text = " ".join(str(value or "").lower().replace("&", "and").split())
    for token in ("-", "/", "(", ")", ",", "."):
        text = text.replace(token, " ")
    return " ".join(text.split())


def _kaggle_base_name_key(value: str) -> str:
    text = _kaggle_name_key(value)
    tokens = {"direct", "regular", "plan", "growth", "option", "dividend", "idcw", "reinvestment", "reinvest", "bonus"}
    return " ".join(token for token in text.split() if token not in tokens)


def _kaggle_cache_root(settings: Settings) -> Path:
    root = Path(settings.market_cache_dir) / KAGGLE_NAV_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _kaggle_history_cache_path(settings: Settings, record: FundRecord) -> Path:
    identity = f"{record.scheme_code}|{record.scheme_name}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return _kaggle_cache_root(settings) / f"{digest}.json"


def _read_kaggle_history_cache(settings: Settings, record: FundRecord) -> pd.DataFrame | None:
    path = _kaggle_history_cache_path(settings, record)
    try:
        if not path.exists() or time.time() - path.stat().st_mtime > KAGGLE_NAV_CACHE_TTL_HOURS * 3600:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
            return None
        rows=[]
        for item in payload["rows"]:
            if not isinstance(item, dict): continue
            date_value=pd.to_datetime(item.get("date"), errors="coerce"); nav_value=safe_float(item.get("nav"))
            if pd.notna(date_value) and nav_value is not None and nav_value > 0: rows.append((date_value, nav_value))
        if not rows: return None
        return pd.DataFrame(rows, columns=["date","nav"]).sort_values("date").reset_index(drop=True)
    except Exception:
        return None


def _write_kaggle_history_cache(settings: Settings, record: FundRecord, df: pd.DataFrame) -> None:
    try:
        rows=[{"date":row.date.strftime("%Y-%m-%d"),"nav":float(row.nav)} for row in df.itertuples(index=False)]
        path=_kaggle_history_cache_path(settings, record); tmp=path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"rows":rows}, ensure_ascii=False), encoding="utf-8"); tmp.replace(path)
    except Exception as exc:
        logger.debug("Could not cache Kaggle NAV history for %s: %s", record.scheme_name, exc)


def _kaggle_file_columns(path: Path) -> tuple[list[str], str, str | None] | None:
    suffix=path.suffix.lower()
    try:
        if suffix in {".csv",".tsv"}:
            sep="\t" if suffix==".tsv" else ","
            return list(pd.read_csv(path, sep=sep, nrows=0).columns), sep, None
        if suffix==".parquet":
            import pyarrow.parquet as pq
            pf=pq.ParquetFile(path); columns=list(pf.schema_arrow.names); index_name=None
            meta=pf.schema_arrow.metadata or {}; pandas_meta=meta.get(b"pandas")
            if pandas_meta:
                try:
                    idx=json.loads(pandas_meta.decode("utf-8")).get("index_columns",[])
                    if idx and isinstance(idx[0],str): index_name=idx[0]
                except Exception: pass
            return columns,"parquet",index_name
        return None
    except Exception as exc:
        logger.warning("Could not inspect Kaggle NAV file %s: %s", path, exc); return None


def _find_kaggle_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_kaggle_name_key(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for column in columns:
        key = _kaggle_name_key(column)
        if any(candidate in key for candidate in candidates):
            return column
    return None


def _kaggle_dataset_files(dataset_path: str) -> list[Path]:
    root = Path(dataset_path)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".parquet"})


def _kaggle_refresh_marker(settings: Settings) -> Path: return _kaggle_cache_root(settings) / "refresh_state.json"

def _read_kaggle_refresh_state(settings: Settings) -> dict:
    try:
        payload=json.loads(_kaggle_refresh_marker(settings).read_text(encoding="utf-8")); return payload if isinstance(payload,dict) else {}
    except Exception: return {}

def _write_kaggle_refresh_state(settings: Settings, *, refreshed_at: str, dataset_path: str, files: list[Path]) -> None:
    payload={"refreshed_at":refreshed_at,"refresh_date":refreshed_at[:10],"dataset":KAGGLE_NAV_DATASET,"dataset_path":dataset_path,"files":[str(path) for path in files]}
    path=_kaggle_refresh_marker(settings); tmp=path.with_suffix(".tmp"); tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); tmp.replace(path)

def _kaggle_should_refresh(settings: Settings) -> bool:
    state=_read_kaggle_refresh_state(settings); refresh_date=str(state.get("refresh_date") or ""); today=dt.date.today()
    if refresh_date==today.isoformat(): return False
    return today.weekday()<5 or KAGGLE_REFRESH_WEEKENDS

def _kaggle_existing_dataset_files(settings: Settings) -> list[Path]:
    return _kaggle_dataset_files(str(_kaggle_cache_root(settings) / "dataset"))

def _download_kaggle_nav_dataset(settings: Settings) -> list[Path]:
    if not KAGGLE_NAV_ENABLED: return []
    try: import kagglehub
    except ImportError:
        logger.info("Kaggle NAV backup disabled because kagglehub is not installed."); return []
    output_dir=_kaggle_cache_root(settings)/"dataset"; existing=_kaggle_existing_dataset_files(settings); force_download=_kaggle_should_refresh(settings)
    if existing and not force_download: return existing
    try:
        dataset_path=kagglehub.dataset_download(KAGGLE_NAV_DATASET, output_dir=str(output_dir), force_download=force_download); files=_kaggle_dataset_files(dataset_path)
        if not files: return existing
        now=dt.datetime.now(dt.timezone.utc).isoformat(); _write_kaggle_refresh_state(settings,refreshed_at=now,dataset_path=dataset_path,files=files); return files
    except Exception as exc:
        logger.warning("Kaggle NAV backup unavailable for %s: %s", KAGGLE_NAV_DATASET, exc); return []


def _load_kaggle_scheme_snapshot(settings: Settings, records: list[FundRecord]) -> dict[str, dict]:
    targets_by_code={_kaggle_code_key(r.scheme_code):r for r in records if r.scheme_code}; targets_by_name={_kaggle_name_key(r.scheme_name):r for r in records}; files=_download_kaggle_nav_dataset(settings)
    if not files: return {}
    snapshot_files=[p for p in files if "mutual_fund_data" in p.name.lower() and p.suffix.lower()==".csv"]
    if not snapshot_files: return {}
    try: frame=pd.read_csv(snapshot_files[0],low_memory=False)
    except Exception as exc: logger.warning("Could not read Kaggle scheme snapshot: %s",exc); return {}
    columns=list(frame.columns); code_col=_find_kaggle_column(columns,("scheme code","scheme_code","amfi code","amfi_code","code")); name_col=_find_kaggle_column(columns,("scheme name","scheme_name","fund name","fund_name","name"))
    if not code_col and not name_col: return {}
    def col(*names): return _find_kaggle_column(columns,names)
    amc_col=col("amc"); category_col=col("scheme category","scheme_category","category"); nav_col=col("nav","net asset value","net_asset_value"); nav_date_col=col("latest nav date","latest_nav_date","nav date","nav_date"); launch_col=col("launch date","launch_date","inception date","inception_date"); aum_col=col("average aum cr","average_aum_cr","aum cr","aum_inr_cr","aum"); plan_col=col("scheme nav name","scheme_nav_name")
    result={}; matched=0
    for row in frame.to_dict(orient="records"):
        code=_kaggle_code_key(row.get(code_col)) if code_col else ""; name=str(row.get(name_col) or "").strip() if name_col else ""; record=targets_by_code.get(code) if code else None
        if record is None and name: record=targets_by_name.get(_kaggle_name_key(name))
        if record is None: continue
        key=record.scheme_code or record.scheme_name; result[key]={"scheme_name":name or record.scheme_name,"amc":str(row.get(amc_col) or "").strip() if amc_col else "","category":str(row.get(category_col) or "").strip() if category_col else "","latest_nav":safe_float(row.get(nav_col)) if nav_col else None,"nav_date":str(row.get(nav_date_col) or "").strip() if nav_date_col else "","inception_date":str(row.get(launch_col) or "").strip() if launch_col else "","aum_inr_cr":safe_float(row.get(aum_col)) if aum_col else None,"scheme_nav_name":str(row.get(plan_col) or "").strip() if plan_col else ""}; matched+=1
    logger.info("Kaggle scheme snapshot matched %d/%d selected funds.",matched,len(records)); return result


def _enrich_from_kaggle_snapshot(settings: Settings, records: list[FundRecord]) -> None:
    snapshot=_load_kaggle_scheme_snapshot(settings,records)
    for record in records:
        values=snapshot.get(record.scheme_code or record.scheme_name)
        if not values: continue
        for field in ("latest_nav","nav_date","aum_inr_cr","inception_date"):
            value=values.get(field)
            if value is None or value=="": continue
            current_source=record.data_sources.get(field,"")
            if getattr(record,field) is not None and not current_source.startswith("mftool"): continue
            setattr(record,field,value); record.data_sources[field]="Kaggle / mutual_fund_data.csv"
        if values.get("amc") and (not record.amc or record.amc=="Unknown"): record.amc=values["amc"]; record.data_sources["amc"]="Kaggle / mutual_fund_data.csv"
        if values.get("category") and (not record.category or record.category=="Other"): record.category=values["category"]; record.data_sources["category"]="Kaggle / mutual_fund_data.csv"
        if values.get("scheme_nav_name"):
            lower=values["scheme_nav_name"].lower()
            if "direct" in lower: record.plan_type="Direct Growth" if "growth" in lower else "Direct"
            elif "regular" in lower: record.plan_type="Regular Growth" if "growth" in lower else "Regular"
        record.source_quality=max(record.source_quality,KAGGLE_NAV_SOURCE_QUALITY)


def _load_kaggle_histories(settings: Settings, records: list[FundRecord]) -> dict[str, pd.DataFrame]:
    """Load only requested fund histories from the Kaggle NAV dataset.

    The dataset schema is discovered at runtime because Kaggle datasets can change
    file names/column names. Exact scheme-code matches are preferred. Name matching
    is conservative and only falls back to a base-name match when it is unique.
    """
    targets_by_code = {_kaggle_code_key(r.scheme_code): r for r in records if r.scheme_code}
    targets_by_name = {_kaggle_name_key(r.scheme_name): r for r in records}
    targets_by_base: dict[str, list[FundRecord]] = {}
    for record in records:
        targets_by_base.setdefault(_kaggle_base_name_key(record.scheme_name), []).append(record)

    histories: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    files = _download_kaggle_nav_dataset(settings)
    if not files:
        return {}

    for path in files:
        info = _kaggle_file_columns(path)
        if not info:
            continue
        columns, sep, index_name = info
        code_col = _find_kaggle_column(columns, ("scheme code", "scheme_code", "amfi code", "amfi_code", "code"))
        name_col = _find_kaggle_column(columns, ("scheme name", "scheme_name", "fund name", "fund_name", "name"))
        date_col = _find_kaggle_column(columns, ("date", "nav date", "nav_date", "as of date"))
        nav_col = _find_kaggle_column(columns, ("nav", "net asset value", "net_asset_value", "net asset value rs"))

        # mutual_fund_nav_history.parquet stores Scheme_Code as the pandas
        # index. It therefore does not appear in ``frame.columns`` and the
        # previous loader incorrectly rejected the file as Date/NAV-only.
        parquet_index_code = sep == "parquet" and index_name is not None and not code_col and not name_col
        if sep == "parquet" and not code_col and not name_col and not index_name and date_col and nav_col:
            # Some parquet writers drop the index name. If the index itself
            # contains one of the requested AMFI scheme codes, treat it as the
            # scheme-code column rather than rejecting an otherwise valid NAV file.
            try:
                probe = pd.read_parquet(path, columns=[], engine="auto")
                index_keys = {_kaggle_code_key(value) for value in probe.index[: min(len(probe.index), 10000)]}
                parquet_index_code = bool(index_keys.intersection(targets_by_code))
            except Exception:
                parquet_index_code = False
        if not date_col or not nav_col or not (code_col or name_col or parquet_index_code):
            logger.warning("Skipping Kaggle NAV file with unsupported schema: %s columns=%s index=%s", path, columns, index_name)
            continue

        usecols = [column for column in (code_col, name_col, date_col, nav_col) if column]
        try:
            if sep == "parquet":
                # Read the parquet once. If Scheme_Code is the index, reset it
                # into a normal column so the existing matching pipeline can
                # operate on both parquet-index and ordinary tabular schemas.
                frame = pd.read_parquet(path, engine="auto")
                if parquet_index_code:
                    reset_code_col = index_name or "index"
                    frame = frame.reset_index()
                    code_col = reset_code_col
                chunks = [frame]
            else:
                chunks = pd.read_csv(path, sep=sep, usecols=usecols, chunksize=KAGGLE_NAV_CHUNK_SIZE)

            for chunk in chunks:
                chunk = chunk.copy()
                chunk[date_col] = chunk[date_col].map(_parse_kaggle_date)
                chunk[nav_col] = pd.to_numeric(chunk[nav_col], errors="coerce")
                chunk = chunk.dropna(subset=[date_col, nav_col])
                chunk = chunk[chunk[nav_col] > 0]
                if chunk.empty:
                    continue

                for row in chunk.itertuples(index=False, name="KaggleRow"):
                    values = row._asdict()
                    code = _kaggle_code_key(values.get(code_col)) if code_col else ""
                    name = str(values.get(name_col) or "").strip() if name_col else ""
                    matched: list[FundRecord] = []
                    if code and code in targets_by_code:
                        matched = [targets_by_code[code]]
                    elif name:
                        exact = targets_by_name.get(_kaggle_name_key(name))
                        if exact:
                            matched = [exact]
                        else:
                            base_matches = targets_by_base.get(_kaggle_base_name_key(name), [])
                            if len(base_matches) == 1:
                                matched = base_matches
                    if not matched:
                        continue
                    date_value = pd.Timestamp(values[date_col])
                    nav_value = float(values[nav_col])
                    for record in matched:
                        key = record.scheme_code or record.scheme_name
                        histories.setdefault(key, []).append((date_value, nav_value))
        except Exception as exc:
            logger.warning("Could not read Kaggle NAV file %s: %s", path, exc)

    result = {}
    for record in records:
        key = record.scheme_code or record.scheme_name
        rows = histories.get(key, [])
        if not rows:
            continue
        frame = pd.DataFrame(rows, columns=["date", "nav"])
        frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        result[key] = frame
        _write_kaggle_history_cache(settings, record, frame)
    return result


def _enrich_from_kaggle_nav(settings: Settings, records: list[FundRecord]) -> None:
    """Fill still-missing NAV/risk metrics from the Kaggle historical NAV dataset."""
    candidates = [
        record for record in records
        if any(getattr(record, field) is None for field in (
            "latest_nav", "cagr_1y_pct", "cagr_3y_pct", "cagr_5y_pct",
            "volatility_pct", "max_drawdown_pct", "sharpe", "sortino",
        ))
    ]
    if not candidates:
        return

    histories = {}
    for record in candidates:
        cached = _read_kaggle_history_cache(settings, record)
        if cached is not None:
            histories[record.scheme_code or record.scheme_name] = cached

    missing = [record for record in candidates if (record.scheme_code or record.scheme_name) not in histories]
    if missing:
        loaded = _load_kaggle_histories(settings, missing)
        histories.update(loaded)

    enriched = 0
    skipped_stale = 0
    cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=KAGGLE_NAV_MAX_STALENESS_DAYS)
    for record in candidates:
        key = record.scheme_code or record.scheme_name
        history = histories.get(key)
        if history is None or history.empty:
            continue
        latest_date = pd.Timestamp(history.iloc[-1]["date"])
        if latest_date < cutoff:
            skipped_stale += 1
            logger.info(
                "Kaggle NAV history stale for %s: latest=%s max_staleness_days=%d",
                record.scheme_name,
                latest_date.strftime("%Y-%m-%d"),
                KAGGLE_NAV_MAX_STALENESS_DAYS,
            )
            continue
        metrics = _calculate_nav_metrics(history)
        if not metrics:
            continue
        changed = False
        risk_fields = {"volatility_pct", "max_drawdown_pct", "sharpe", "sortino"}
        for field in (
            "latest_nav", "cagr_1y_pct", "cagr_3y_pct", "cagr_5y_pct",
            "volatility_pct", "max_drawdown_pct", "sharpe", "sortino",
        ):
            # Do not derive annualized risk statistics from a handful of NAV
            # observations. CAGR/latest NAV can still be useful with shorter
            # histories, while risk metrics require a meaningful sample.
            if field in risk_fields and len(history) < 30:
                continue
            value = metrics.get(field)
            if value is not None:
                current_source = record.data_sources.get(field, "")
                if getattr(record, field) is not None and not current_source.startswith("mftool"):
                    continue
                setattr(record, field, value)
                record.data_sources[field] = "Kaggle / mutual-fund-historic-nav-data"
                changed = True
        if (record.nav_date is None or record.data_sources.get("nav_date", "").startswith("mftool")) and metrics.get("nav_date"):
            record.nav_date = metrics["nav_date"]
            record.data_sources["nav_date"] = "Kaggle / mutual-fund-historic-nav-data"
            changed = True
        if changed:
            record.nav_history_observations = max(
                record.nav_history_observations or 0,
                metrics.get("nav_history_observations") or 0,
            )
            record.source_quality = max(record.source_quality, KAGGLE_NAV_SOURCE_QUALITY)
            enriched += 1

    logger.info(
        "Kaggle NAV fallback enrichment: enriched=%d/%d stale=%d dataset=%s",
        enriched,
        len(candidates),
        skipped_stale,
        KAGGLE_NAV_DATASET,
    )


def _historical_metrics(mf: Mftool, code: str) -> dict:
    """Calculate risk metrics from mftool's AMFI historical NAV series."""
    if not code:
        return {}
    try:
        raw = mf.get_scheme_historical_nav(code)
        rows = raw.get("data", []) if isinstance(raw, dict) else []
        df = _parse_nav_history_rows(rows)
        metrics = _calculate_nav_metrics(df)
        if metrics:
            metrics["data_source"] = "mftool / AMFI NAV history"
        return metrics
    except Exception as exc:
        logger.warning("mftool historical NAV failed for %s: %s", code, exc)
        return {}


def _normalise_scheme_name(name: str) -> str:
    """Normalize AMFI and mftool performance names for reliable matching."""
    value = " ".join(str(name or "").lower().replace("&", "and").split())
    replacements = [
        "- direct plan - growth option",
        "- direct plan - growth",
        "- direct plan growth option",
        "- direct plan growth",
        "- direct growth option",
        "- direct growth",
        "direct plan - growth option",
        "direct plan - growth",
        "direct plan growth option",
        "direct plan growth",
        "direct growth option",
        "direct growth",
        "- growth option",
        "- growth",
        "growth option",
        "growth",
    ]
    for suffix in replacements:
        if value.endswith(suffix):
            value = value[: -len(suffix)].rstrip(" -")
            break
    return value


def _performance_method_for_category(category: str) -> str:
    value = str(category or "").lower()
    if any(token in value for token in (
        "debt", "bond", "liquid", "overnight", "gilt", "duration", "credit",
    )):
        return "get_open_ended_debt_scheme_performance"
    if any(token in value for token in ("hybrid", "balanced advantage", "aggressive hybrid")):
        return "get_open_ended_hybrid_scheme_performance"
    if any(token in value for token in ("solution", "retirement", "children")):
        return "get_open_ended_solution_scheme_performance"
    if any(token in value for token in ("index", "fund of fund", "fof")):
        return "get_open_ended_other_scheme_performance"
    return "get_open_ended_equity_scheme_performance"


def _performance_value(item: dict, direct_key: str) -> float | None:
    value = item.get(direct_key)
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if text in {"", "-", "--", "NA", "N/A", "None"}:
        return None
    return safe_float(text)


def _mftool_performance_cache_path(settings: Settings, method_name: str, report_date: str) -> Path:
    root = Path(settings.market_cache_dir) / "mftool_performance"
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{method_name}:{report_date}".encode("utf-8")).hexdigest()
    return root / f"{digest}.json"


def _read_mftool_performance_cache(settings: Settings, method_name: str, report_date: str):
    path = _mftool_performance_cache_path(settings, method_name, report_date)
    try:
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > MFTOOL_PERFORMANCE_CACHE_TTL_HOURS * 3600:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Never reuse a cached empty AMFI performance response. Empty responses
        # commonly occur when the requested date is an AMFI market holiday.
        if not isinstance(payload, dict) or not any(isinstance(v, list) and v for v in payload.values()):
            return None
        return payload
    except Exception:
        return None


def _write_mftool_performance_cache(settings: Settings, method_name: str, report_date: str, payload) -> None:
    try:
        if not isinstance(payload, dict) or not any(isinstance(v, list) and v for v in payload.values()):
            return
        path = _mftool_performance_cache_path(settings, method_name, report_date)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.debug("Could not cache mftool performance %s/%s: %s", method_name, report_date, exc)


def _candidate_mftool_report_dates(max_days: int = 10) -> list[str]:
    """Return recent weekdays so AMFI holidays can fall back to the prior trading day."""
    today = dt.date.today()
    dates = []
    for offset in range(1, max_days + 1):
        candidate = today - dt.timedelta(days=offset)
        if candidate.weekday() < 5:
            dates.append(candidate.strftime("%d-%b-%Y"))
    return dates


def _load_mftool_performance(mf: Mftool, settings: Settings, records: list[FundRecord]) -> dict[str, dict]:
    """Load AMFI daily performance, falling back across recent weekdays.

    mftool's default date logic assumes Friday is a valid trading day. That is
    not true for Indian market holidays such as Gandhi Jayanti. We therefore
    try recent weekdays explicitly and never cache an empty response.
    """
    methods = {_performance_method_for_category(record.category) for record in records}
    by_name: dict[str, dict] = {}

    for method_name in sorted(methods):
        loaded_for_method = False
        for report_date in _candidate_mftool_report_dates():
            try:
                payload = _read_mftool_performance_cache(settings, method_name, report_date)
                cache_hit = payload is not None
                if payload is None:
                    method = getattr(mf, method_name)
                    payload = method(report_date=report_date)

                if not isinstance(payload, dict):
                    continue

                items_found = 0
                for _, items in payload.items():
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        if not isinstance(item, dict) or not item.get("scheme_name"):
                            continue
                        by_name[_normalise_scheme_name(item["scheme_name"])] = item
                        items_found += 1

                if items_found:
                    _write_mftool_performance_cache(settings, method_name, report_date, payload)
                    logger.info(
                        "mftool performance loaded: %s -> %d schemes; report_date=%s; cache_hit=%s",
                        method_name,
                        items_found,
                        report_date,
                        cache_hit,
                    )
                    loaded_for_method = True
                    break

                logger.info(
                    "mftool performance empty: %s; report_date=%s; trying previous weekday",
                    method_name,
                    report_date,
                )
            except Exception as exc:
                logger.warning(
                    "mftool performance failed for %s on %s: %s",
                    method_name,
                    report_date,
                    exc,
                )

        if not loaded_for_method:
            logger.warning(
                "mftool performance unavailable: %s; no non-empty AMFI report found in recent weekdays",
                method_name,
            )

    return by_name


def _enrich_from_mftool_performance(records: list[FundRecord], performance: dict[str, dict]) -> None:
    """Fill direct-plan returns and benchmark from mftool's AMFI performance reports."""
    matched = 0
    for record in records:
        item = performance.get(_normalise_scheme_name(record.scheme_name))
        if not item:
            continue
        changed = False
        for field, key in (
            ("cagr_1y_pct", "1-Year Return(%)- Direct"),
            ("cagr_3y_pct", "3-Year Return(%)- Direct"),
            ("cagr_5y_pct", "5-Year Return(%)- Direct"),
        ):
            value = _performance_value(item, key)
            if value is not None and (
                getattr(record, field) is None
                or record.data_sources.get(field) == "mftool / AMFI NAV history"
            ):
                setattr(record, field, value)
                record.data_sources[field] = "mftool / AMFI daily performance"
                changed = True

        benchmark = item.get("benchmark")
        if not record.benchmark and benchmark and str(benchmark).strip() not in {"-", "NA", "N/A"}:
            record.benchmark = str(benchmark).strip()
            record.data_sources["benchmark"] = "mftool / AMFI daily performance"
            changed = True

        if changed:
            record.source_quality = max(record.source_quality, 0.90)
            matched += 1

    logger.info("mftool performance enrichment: matched %d/%d funds.", matched, len(records))



def _make_session() -> requests.Session:
    session = requests.Session()
    # Keep retries bounded. A provider timeout must not block the analysis for
    # minutes, but transient 5xx responses are worth one quick retry.
    retry = Retry(
        total=1,
        connect=1,
        read=1,
        redirect=0,
        status=1,
        backoff_factor=0.25,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
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
def _mfdata_is_available() -> bool:
    """Return whether mfdata.in is reachable before starting bulk enrichment.

    mfdata.in is an optional enrichment provider. A dead or stalled provider
    must never add tens of seconds of connection/read timeouts to every run.
    The health endpoint is unauthenticated according to the provider docs.
    """
    if not MFDATA_ENABLED:
        return False
    url = f"{MFDATA_BASE_URL}/api/health"
    try:
        response = requests.get(
            url,
            timeout=MFDATA_HEALTHCHECK_TIMEOUT_SECONDS,
            headers={"User-Agent": "MF Research Dashboard/2.5", "Accept": "application/json"},
        )
        if response.status_code != 200:
            logger.warning("mfdata health check returned HTTP %s; skipping provider for this run.", response.status_code)
            return False
        payload = response.json()
        if isinstance(payload, dict) and str(payload.get("status", "ok")).lower() in {"ok", "healthy", "success"}:
            return True
        logger.warning("mfdata health check returned an unexpected payload; skipping provider for this run.")
        return False
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        logger.warning("mfdata health check failed; skipping provider for this run: %s", exc)
        return False
    except (ValueError, requests.exceptions.RequestException) as exc:
        logger.warning("mfdata health check could not be validated; skipping provider for this run: %s", exc)
        return False


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
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        logger.warning("mfdata GET failed for %s: %s", path, exc)
        return None
    except (ValueError, requests.exceptions.RequestException) as exc:
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
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        logger.warning("mfdata POST failed for %s: %s", path, exc)
        return None
    except (ValueError, requests.exceptions.RequestException) as exc:
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

    # mfdata returns ratios in two shapes across its enrichment sources:
    # nested groups (valuation/risk/return) and flat keys (pe/pb/sharpe/etc.).
    risk = ratios.get("risk") if isinstance(ratios.get("risk"), dict) else {}
    ret = ratios.get("return") if isinstance(ratios.get("return"), dict) else {}
    valuation = ratios.get("valuation") if isinstance(ratios.get("valuation"), dict) else {}

    def first_number(*values):
        for value in values:
            parsed = _number(value)
            if parsed is not None:
                return parsed
        return None

    valuation_keys = ("pe", "pb", "ps", "dividend_yield")
    valuation_out = {}
    for key in valuation_keys:
        value = first_number(valuation.get(key), ratios.get(key))
        if value is not None:
            valuation_out[key] = value

    return {
        "sharpe": first_number(ret.get("sharpe"), ratios.get("sharpe")),
        "sortino": first_number(risk.get("sortino"), ratios.get("sortino")),
        "volatility_pct": first_number(risk.get("std_deviation"), ratios.get("std_deviation")),
        "valuation": valuation_out,
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


def _extract_risk_detail(data: dict) -> dict:
    if not isinstance(data, dict):
        return {}
    drawdown = data.get("drawdown") if isinstance(data.get("drawdown"), dict) else {}
    risk_return = data.get("risk_return") if isinstance(data.get("risk_return"), dict) else {}
    return {
        "max_drawdown_pct": _number(drawdown.get("max_drawdown_pct")),
        "annualized_risk": _number(risk_return.get("annualized_risk")),
    }


def _detail_from_payload(payload) -> dict | None:
    data = _payload_data(payload)
    return data if isinstance(data, dict) else None
def _record_mfdata_detail(record: FundRecord, detail: dict) -> None:
    """Merge one mfdata scheme response without replacing stronger existing metrics."""
    record.source_quality = max(record.source_quality, 0.90)

    nav = _number(detail.get("nav"))
    if nav is not None and record.latest_nav is None:
        record.latest_nav = nav
        record.data_sources["latest_nav"] = "mfdata scheme details"
    if detail.get("nav_date") and record.nav_date is None:
        record.nav_date = detail.get("nav_date")
        record.data_sources["nav_date"] = "mfdata scheme details"
    aum = _number(detail.get("aum_cr", detail.get("aum_inr_cr")))
    if aum is not None and record.aum_inr_cr is None:
        record.aum_inr_cr = aum
        record.data_sources["aum_inr_cr"] = "mfdata scheme details"
    if detail.get("inception_date") and record.inception_date is None:
        record.inception_date = detail.get("inception_date")
        record.data_sources["inception_date"] = "mfdata scheme details"
    benchmark = detail.get("benchmark")
    if benchmark and record.benchmark is None and str(benchmark).strip() not in {"-", "NA", "N/A"}:
        record.benchmark = str(benchmark).strip()
        record.data_sources["benchmark"] = "mfdata scheme details"

    returns = _extract_returns(detail)
    for field, value in returns.items():
        if getattr(record, field) is None:
            setattr(record, field, value)
            record.data_sources[field] = "mfdata scheme details"

    ratio_metrics = _extract_ratio_metrics(detail)
    for field in ("sharpe", "sortino", "volatility_pct"):
        value = ratio_metrics.get(field)
        if value is not None and getattr(record, field) is None:
            setattr(record, field, value)
            record.data_sources[field] = "mfdata scheme details"
    if ratio_metrics.get("valuation"):
        record.valuation.update(ratio_metrics["valuation"])
        record.data_sources["valuation"] = "mfdata scheme details"

    family_id = detail.get("family_id")
    if family_id is not None:
        record._mfdata_family_id = str(family_id)
        record.data_sources["family_id"] = "mfdata scheme details"


def _fetch_mfdata_detail(settings: Settings, record: FundRecord):
    """Fetch one scheme independently so one slow scheme cannot block the rest."""
    if not record.scheme_code:
        return record.scheme_code, None
    session = _make_session()
    payload = _mfdata_get(session, settings, f"/api/v1/schemes/{record.scheme_code}")
    return record.scheme_code, _detail_from_payload(payload)


def _fetch_family_holdings(settings: Settings, family_id: str):
    session = _make_session()
    payload = _mfdata_get(session, settings, f"/api/v1/families/{family_id}/holdings")
    return family_id, _extract_holdings(payload or {})


def _enrich_from_mfdata(settings: Settings, records: list[FundRecord], holdings: dict[str, Holding]) -> None:
    """Enrich selected funds from mfdata with bulk-first, per-scheme fallback."""
    if not records:
        return

    if not MFDATA_ENABLED:
        logger.info("mfdata enrichment disabled by MF_DATA_ENABLED=false.")
        return

    if not _mfdata_is_available():
        logger.info("mfdata enrichment skipped because the provider is unavailable.")
        return

    session = _make_session()
    codes = [str(r.scheme_code) for r in records if r.scheme_code]
    details_by_code: dict[str, dict] = {}

    # Bulk is an optimization only. A timeout/HTTP failure leaves details_by_code
    # incomplete and the independent scheme requests below become the fallback.
    bulk_failed = False
    for start in range(0, len(codes), MFDATA_BULK_CHUNK_SIZE):
        chunk = codes[start:start + MFDATA_BULK_CHUNK_SIZE]
        payload = _mfdata_post(
            session,
            settings,
            "/api/v1/schemes/bulk",
            {"scheme_codes": [int(x) if x.isdigit() else x for x in chunk]},
        )
        if payload is None:
            bulk_failed = True
            break
        data = _payload_data(payload)
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                code = str(item.get("scheme_code") or item.get("amfi_code") or "")
                if code:
                    details_by_code[code] = item

    if bulk_failed:
        logger.warning("mfdata bulk enrichment unavailable; falling back to individual scheme requests for this run.")

    missing = [r for r in records if r.scheme_code and str(r.scheme_code) not in details_by_code]
    if missing:
        logger.info(
            "mfdata individual fallback: %d/%d schemes missing after bulk; fetching up to %d independently with %d workers.",
            len(missing), len(records), MFDATA_INDIVIDUAL_FALLBACK_LIMIT, MFDATA_MAX_WORKERS,
        )
        fallback_limit = min(len(missing), MFDATA_INDIVIDUAL_FALLBACK_LIMIT)
        if fallback_limit:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(MFDATA_MAX_WORKERS, fallback_limit)) as pool:
                futures = [pool.submit(_fetch_mfdata_detail, settings, record) for record in missing[:fallback_limit]]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        code, detail = future.result()
                        if detail:
                            details_by_code[str(code)] = detail
                    except Exception as exc:
                        logger.warning("mfdata individual scheme enrichment failed: %s", exc)

    for record in records:
        detail = details_by_code.get(str(record.scheme_code))
        if detail:
            _record_mfdata_detail(record, detail)

    family_to_records: dict[str, list[FundRecord]] = {}
    for record in records:
        family_id = getattr(record, "_mfdata_family_id", None)
        if family_id:
            family_to_records.setdefault(str(family_id), []).append(record)

    held_lower = {name.lower() for name in holdings}
    ordered_families = sorted(
        family_to_records,
        key=lambda fid: min(0 if r.scheme_name.lower() in held_lower else 1 for r in family_to_records[fid]),
    )[:MFDATA_FAMILY_ENRICHMENT_LIMIT]

    if ordered_families:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(MFDATA_MAX_WORKERS, len(ordered_families))) as pool:
            futures = [pool.submit(_fetch_family_holdings, settings, family_id) for family_id in ordered_families]
            for future in concurrent.futures.as_completed(futures):
                try:
                    family_id, (fund_holdings, sectors, market_caps) = future.result()
                except Exception as exc:
                    logger.warning("mfdata family holdings enrichment failed: %s", exc)
                    continue
                for record in family_to_records.get(str(family_id), []):
                    if fund_holdings:
                        record.holdings.update(fund_holdings)
                        record.data_sources["holdings"] = "mfdata family holdings"
                    if sectors:
                        record.sector_weights.update(sectors)
                        record.data_sources["sector_weights"] = "mfdata family holdings"
                    if market_caps:
                        record.market_cap_weights.update(market_caps)
                        record.data_sources["market_cap_weights"] = "mfdata family holdings"
                    if fund_holdings or sectors or market_caps:
                        record.source_quality = 1.0

    enriched_count = sum(1 for record in records if str(record.scheme_code) in details_by_code)
    portfolio_count = sum(1 for record in records if record.holdings)
    valuation_count = sum(1 for record in records if record.valuation)
    logger.info(
        "mfdata enrichment completed: scheme_details=%d/%d holdings=%d/%d valuation=%d/%d",
        enriched_count, len(records), portfolio_count, len(records), valuation_count, len(records),
    )

def _enrich_from_mfapi(settings: Settings, records: list[FundRecord]) -> None:
    """Fill remaining NAV-derived gaps from MFapi.in using AMFI scheme codes."""
    if not records:
        return
    if not MFAPI_ENABLED:
        logger.info("MFapi NAV enrichment disabled by MFAPI_ENABLED=false.")
        return

    candidates = [
        record for record in records
        if record.scheme_code and any(
            getattr(record, field) is None
            for field in (
                "latest_nav", "nav_date", "cagr_1y_pct", "cagr_3y_pct", "cagr_5y_pct",
                "volatility_pct", "max_drawdown_pct", "sharpe", "sortino",
            )
        )
    ]
    if not candidates:
        logger.info("MFapi NAV enrichment: no missing NAV-derived fields.")
        return

    histories = fetch_histories(
        [str(record.scheme_code) for record in candidates],
        settings.market_cache_dir,
    )
    enriched = 0
    for record in candidates:
        result = histories.get(str(record.scheme_code))
        if not result:
            continue
        history, metadata = result
        if history.empty:
            continue
        metrics = _calculate_nav_metrics(history)
        meta = scheme_metadata(metadata)
        changed = False

        for field in (
            "latest_nav", "cagr_1y_pct", "cagr_3y_pct", "cagr_5y_pct",
            "volatility_pct", "max_drawdown_pct", "sharpe", "sortino",
        ):
            value = metrics.get(field)
            if value is not None and getattr(record, field) is None:
                setattr(record, field, value)
                record.data_sources[field] = "MFapi.in NAV history"
                changed = True

        if metrics.get("nav_date") and record.nav_date is None:
            record.nav_date = metrics["nav_date"]
            record.data_sources["nav_date"] = "MFapi.in NAV history"
            changed = True

        if meta.get("fund_house") and (not record.amc or record.amc == "Unknown"):
            record.amc = str(meta["fund_house"])
            record.data_sources["amc"] = "MFapi.in scheme metadata"
            changed = True
        if meta.get("scheme_category") and (not record.category or record.category == "Other"):
            record.category = str(meta["scheme_category"])
            record.data_sources["category"] = "MFapi.in scheme metadata"
            changed = True

        record.nav_history_observations = max(
            record.nav_history_observations or 0,
            int(metrics.get("nav_history_observations") or 0),
        )
        if changed:
            record.source_quality = max(record.source_quality, 0.97)
            enriched += 1

    logger.info("MFapi NAV enrichment completed: enriched=%d/%d candidates.", enriched, len(candidates))


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
            logger.warning("mftool quote/details failed for %s (%s): %s", name, code, exc)

    hist = _historical_metrics(mf, code) if code else {}
    category = details.get("scheme_category") or item.get("category") or classify_category(name)
    amc = details.get("fund_house") or item.get("amc") or "Unknown"
    nav = safe_float(quote.get("nav"))
    nav_date = quote.get("last_updated")
    inception = details.get("scheme_start_date")
    if isinstance(inception, dict):
        inception = inception.get("date")
    elif inception is not None:
        inception = str(inception)

    sources = {}
    if nav is not None:
        sources["latest_nav"] = "mftool / AMFI quote"
    if nav_date:
        sources["nav_date"] = "mftool / AMFI quote"
    if details:
        sources["scheme_metadata"] = "mftool / AMFI scheme details"
    if inception:
        sources["inception_date"] = "mftool / AMFI scheme details"
    for field in (
        "cagr_1y_pct", "cagr_3y_pct", "cagr_5y_pct", "volatility_pct",
        "max_drawdown_pct", "sharpe", "sortino",
    ):
        if hist.get(field) is not None:
            sources[field] = "mftool / AMFI NAV history"

    return FundRecord(
        scheme_name=name,
        scheme_code=code,
        category=category,
        amc=amc,
        latest_nav=nav,
        nav_date=nav_date,
        inception_date=inception,
        cagr_1y_pct=hist.get("cagr_1y_pct"),
        cagr_3y_pct=hist.get("cagr_3y_pct"),
        cagr_5y_pct=hist.get("cagr_5y_pct"),
        volatility_pct=hist.get("volatility_pct"),
        max_drawdown_pct=hist.get("max_drawdown_pct"),
        sharpe=hist.get("sharpe"),
        sortino=hist.get("sortino"),
        benchmark=None,
        source_quality=0.90 if hist else 0.75,
        data_sources=sources,
        nav_history_observations=hist.get("nav_history_observations"),
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
    logger.info(
        "Initializing data providers: mftool=%s TigZig_enabled=%s TigZig_base=%s Kaggle_legacy=%s",
        _MFTOOL_VERSION, TIGZIG_NAV_ENABLED, TIGZIG_NAV_BASE_URL, KAGGLE_NAV_ENABLED,
    )
    mf = Mftool()
    raw = fetch_universe(settings, mf)
    for name in holdings:
        if not any(x["scheme_name"].lower() == name.lower() for x in raw):
            raw.append({"scheme_name": name, "category": "Holding Scheme", "amc": "Unknown", "code": ""})

    records: list[FundRecord] = []

    def build_record(item: dict) -> FundRecord:
        # Each worker gets its own Mftool requests session. Sharing one Mftool
        # instance across threads can cause intermittent quote/history failures.
        worker_mf = _thread_mftool()
        record = _fund_record(worker_mf, item, holdings)
        extra = load_optional_factsheet_data(settings, record.scheme_name)
        for key in ("aum_inr_cr", "inception_date", "volatility_pct", "max_drawdown_pct", "sharpe", "sortino", "benchmark"):
            if key in extra and extra[key] is not None:
                setattr(record, key, extra[key])
                record.data_sources[key] = "local factsheet"
                record.source_quality = max(record.source_quality, 0.95)
        for key in ("sector_weights", "holdings", "market_cap_weights", "valuation"):
            if isinstance(extra.get(key), dict) and extra[key]:
                setattr(record, key, extra[key])
                record.data_sources[key] = "local factsheet"
                record.source_quality = max(record.source_quality, 0.95)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(build_record, item) for item in raw]
        for future in concurrent.futures.as_completed(futures):
            try:
                records.append(future.result())
            except Exception as exc:
                logger.warning("Fund processing failed: %s", exc)

    records.sort(key=lambda x: (x.category.lower(), x.scheme_name.lower()))

    # AMFI-derived providers are primary. TigZig supplies a daily scheme
    # snapshot plus full per-scheme history; mftool/AMFI remains the direct
    # fallback, followed by Creget/Kaggle archival sources when needed.
    try:
        logger.info("TigZig/AMFI primary enrichment starting for %d funds.", len(records))
        _enrich_from_tigzig_snapshot(settings, records)
        _enrich_from_tigzig_nav(settings, records)
        logger.info("TigZig/AMFI primary enrichment completed.")
    except Exception as exc:
        logger.warning("TigZig primary enrichment failed; continuing with AMFI/mftool fallbacks: %s", exc)

    try:
        performance = _load_mftool_performance(mf, settings, records)
        _enrich_from_mftool_performance(records, performance)
    except Exception as exc:
        logger.warning("mftool performance enrichment failed: %s", exc)

    # MFapi is a bounded, scheme-code-based NAV-history fallback. It runs before
    # the opt-in Kaggle archive because it is a live API and requires no dataset
    # download or credentials.
    try:
        _enrich_from_mfapi(settings, records)
    except Exception as exc:
        logger.warning("MFapi NAV enrichment failed; continuing with existing data: %s", exc)

    # Kaggle is intentionally outside the normal analysis path. The previous
    # implementation could spend minutes scanning the large historical parquet
    # even when the primary AMFI-derived provider was already available. Keep
    # the legacy fallback as an explicit opt-in only.
    if KAGGLE_NAV_ENABLED:
        logger.info("Legacy Kaggle fallback explicitly enabled; starting optional enrichment.")
        try:
            _enrich_from_kaggle_snapshot(settings, records)
            _enrich_from_kaggle_nav(settings, records)
        except Exception as exc:
            logger.warning("Kaggle fallback enrichment failed: %s", exc)
    else:
        logger.info("Legacy Kaggle fallback disabled; skipping Kaggle dataset scan.")

    try:
        logger.info("mfdata optional enrichment starting for %d funds.", len(records))
        _enrich_from_mfdata(settings, records, holdings)
    except Exception as exc:
        logger.warning("mfdata enrichment failed; continuing with mftool/AMFI/MFapi data: %s", exc)
    finally:
        logger.info("mfdata optional enrichment finished.")

    return records
