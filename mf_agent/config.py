from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=False)


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _csv(name: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, "").split(",") if x.strip()]


@dataclass(frozen=True)
class HorizonConfig:
    mode: str
    label: str
    json_value: str
    age: int | None


@dataclass(frozen=True)
class PortfolioPolicy:
    max_category_pct: float = 35.0
    max_single_fund_pct: float = 25.0
    max_single_amc_pct: float = 35.0
    max_mid_cap_pct: float = 25.0
    max_small_cap_pct: float = 15.0
    max_thematic_pct: float = 10.0
    min_cash_pct: float = 5.0
    max_gold_pct: float = 15.0
    max_fund_overlap_pct: float = 60.0


@dataclass(frozen=True)
class Settings:
    batch_size: int = 4
    ranking_limit: int = 50
    vector_limit: int = 15
    news_vector_limit: int = 10
    inter_batch_delay_sec: float = 3.0
    min_aum_inr_cr: float = 500.0
    min_history_years: float = 3.0
    max_schemes_per_category: int = 12
    max_per_amc_per_category: int = 3
    news_max_age_days: int = 7
    news_per_source: int = 20
    mf_holdings_path: str = "mf_holdings.json"
    output_path: str = "top_mutual_funds_analysis.json"
    factsheet_dir: str = "fund_data"
    market_cache_dir: str = "cache"
    openrouter_model: str = "openai/gpt-4o-mini"
    openrouter_secondary_models: list[str] = field(default_factory=list)
    openrouter_temperature: float = 0.1
    allocation_mode: str = "DIVERSIFIED"
    horizon: HorizonConfig = field(default_factory=lambda: HorizonConfig("LONG_TERM", ">5 years", ">5 years", None))
    policy: PortfolioPolicy = field(default_factory=PortfolioPolicy)


def _age_from_dob(raw: str) -> int | None:
    if not raw.strip():
        return None
    try:
        dob = datetime.strptime(raw.strip(), "%d/%m/%Y")
    except ValueError:
        return None
    today = datetime.now()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def load_settings() -> Settings:
    mode = os.getenv("MF_HORIZON", "LONG_TERM").upper()
    try:
        age = int(os.getenv("INVESTOR_AGE", "")) if os.getenv("INVESTOR_AGE", "").strip() else None
    except ValueError:
        age = None
    if age is None:
        age = _age_from_dob(os.getenv("USER_DOB", ""))
    if age is not None and not 18 <= age <= 100:
        age = None
    if mode == "SHORT_TERM":
        horizon = HorizonConfig(mode, "1-3 years", "1-3 years", age)
    else:
        mode = "LONG_TERM"
        label = f">5 years, investor age {age}" if age is not None else ">5 years"
        horizon = HorizonConfig(mode, label, ">5 years", age)

    allocation = os.getenv("ALLOCATION_MODE", "DIVERSIFIED").upper()
    if allocation not in {"DIVERSIFIED", "CONCENTRATED"}:
        allocation = "DIVERSIFIED"

    policy = PortfolioPolicy(
        max_category_pct=_float("MAX_CATEGORY_EXPOSURE_PCT", 35.0),
        max_single_fund_pct=_float("MAX_SINGLE_FUND_PCT", 25.0),
        max_single_amc_pct=_float("MAX_SINGLE_AMC_PCT", 35.0),
        max_mid_cap_pct=_float("MAX_MID_CAP_PCT", 25.0),
        max_small_cap_pct=_float("MAX_SMALL_CAP_PCT", 15.0),
        max_thematic_pct=_float("MAX_THEMATIC_PCT", 10.0),
        min_cash_pct=_float("MIN_CASH_PCT", 5.0),
        max_gold_pct=_float("MAX_GOLD_PCT", 15.0),
        max_fund_overlap_pct=_float("MAX_FUND_OVERLAP_PCT", 60.0),
    )

    return Settings(
        batch_size=_int("BATCH_SIZE", 4),
        ranking_limit=_int("LOCAL_RANKING_LIMIT", 50),
        vector_limit=_int("VECTOR_RETRIEVAL_LIMIT", 15),
        news_vector_limit=_int("NEWS_VECTOR_RETRIEVAL_LIMIT", 10),
        inter_batch_delay_sec=_float("INTER_BATCH_DELAY_SEC", 3.0),
        min_aum_inr_cr=_float("MIN_AUM_INR_CR", 500.0),
        min_history_years=_float("MIN_HISTORY_YEARS", 3.0),
        max_schemes_per_category=_int("MAX_SCHEMES_PER_CATEGORY", 12),
        max_per_amc_per_category=_int("MAX_SCHEMES_PER_AMC_PER_CATEGORY", 3),
        news_max_age_days=_int("NEWS_MAX_AGE_DAYS", 7),
        news_per_source=_int("NEWS_PER_SOURCE", 20),
        mf_holdings_path=os.getenv("MF_HOLDINGS_JSON_PATH", "mf_holdings.json"),
        output_path=os.getenv("MF_SUGGESTION_OUTPUT_PATH", "top_mutual_funds_analysis.json"),
        factsheet_dir=os.getenv("FUND_FACTSHEET_DIR", "fund_data"),
        market_cache_dir=os.getenv("MARKET_CACHE_DIR", "cache"),
        openrouter_model=os.getenv("OPENROUTER_MODEL_NAME", "openai/gpt-4o-mini"),
        openrouter_secondary_models=_csv("OPENROUTER_SECONDARY_MODEL_NAMES"),
        openrouter_temperature=_float("OPENROUTER_MODEL_TEMPERATURE", 0.1),
        allocation_mode=allocation,
        horizon=horizon,
        policy=policy,
    )
