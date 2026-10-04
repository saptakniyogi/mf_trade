from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Holding:
    scheme_name: str
    current_value: float = 0.0
    invested_value: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    folio: str | None = None
    ticker_symbol: str | None = None
    last_price_date: str | None = None


@dataclass
class FundRecord:
    scheme_name: str
    scheme_code: str
    category: str
    amc: str
    plan_type: str = "Direct Growth"
    latest_nav: float | None = None
    nav_date: str | None = None
    aum_inr_cr: float | None = None
    inception_date: str | None = None
    cagr_1y_pct: float | None = None
    cagr_3y_pct: float | None = None
    cagr_5y_pct: float | None = None
    volatility_pct: float | None = None
    max_drawdown_pct: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    benchmark: str | None = None
    sector_weights: dict[str, float] = field(default_factory=dict)
    holdings: dict[str, float] = field(default_factory=dict)
    market_cap_weights: dict[str, float] = field(default_factory=dict)
    valuation: dict[str, float] = field(default_factory=dict)
    source_quality: float = 0.75
    data_sources: dict[str, str] = field(default_factory=dict)
    nav_history_observations: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MacroSnapshot:
    usd_inr: float | None = None
    brent_crude_usd: float | None = None
    india_vix: float | None = None
    india_10y_yield_pct: float | None = None
    nifty_50: float | None = None
    nifty_midcap: float | None = None
    nifty_smallcap: float | None = None
    gold_usd: float | None = None
    us_10y_yield_pct: float | None = None
    sp500: float | None = None
    crude_change_1m_pct: float | None = None
    usd_inr_change_1m_pct: float | None = None
    india_vix_change_1m_pct: float | None = None
    fii_flow_inr_cr: float | None = None
    dii_flow_inr_cr: float | None = None
    inflation_pct: float | None = None
    repo_rate_pct: float | None = None
    macro_sources: dict[str, str] = field(default_factory=dict)
    fetched_at: str | None = None
    selected_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegimeSignal:
    name: str
    value: float
    direction: str
    confidence: float
    rationale: str


@dataclass
class MarketRegime:
    equity: str
    rates: str
    liquidity: str
    inflation: str
    currency: str
    oil: str
    valuation: str
    geopolitics: str
    climate: str
    overall: str
    signals: list[RegimeSignal] = field(default_factory=list)
    data_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "equity": self.equity,
            "rates": self.rates,
            "liquidity": self.liquidity,
            "inflation": self.inflation,
            "currency": self.currency,
            "oil": self.oil,
            "valuation": self.valuation,
            "geopolitics": self.geopolitics,
            "climate": self.climate,
            "overall": self.overall,
            "signals": [asdict(s) for s in self.signals],
            "data_confidence": self.data_confidence,
        }


@dataclass
class FundScore:
    fund_quality: float
    risk_adjusted_return: float | None
    current_attractiveness: float
    portfolio_fit: float | None
    valuation: float | None
    macro_resilience: float
    overall: float
    data_confidence: float = 0.0
    ranking_score: float = 0.0
    ranking_cap: float | None = None
    ranking_adjustment: float = 0.0
    evidence_status: str = "INSUFFICIENT_DATA"
    allocation_eligible: bool = False
    evidence_blockers: list[str] = field(default_factory=list)
    reasons_to_buy: list[str] = field(default_factory=list)
    reasons_not_to_buy: list[str] = field(default_factory=list)
    data_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
