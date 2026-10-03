from __future__ import annotations

from .models import MacroSnapshot, MarketRegime, RegimeSignal
from .utils import clamp


def _direction(value: float | None, good_low: bool = False) -> str:
    if value is None:
        return "UNKNOWN"
    if good_low:
        return "POSITIVE" if value < 0 else "NEGATIVE"
    return "POSITIVE" if value > 0 else "NEGATIVE"


def infer_regime(macro: MacroSnapshot, news: list[dict] | None = None) -> MarketRegime:
    news = news or []
    oil_risk = 50.0
    if macro.brent_crude_usd is not None:
        oil_risk = clamp(50 + (macro.brent_crude_usd - 80) * 1.25)
        if macro.crude_change_1m_pct is not None:
            oil_risk = clamp(oil_risk + macro.crude_change_1m_pct * 0.8)

    fx_risk = 50.0
    if macro.usd_inr_change_1m_pct is not None:
        fx_risk = clamp(50 + macro.usd_inr_change_1m_pct * 4)

    vix_risk = 50.0 if macro.india_vix is None else clamp(macro.india_vix * 3.0)
    geopolitics_count = sum(1 for n in news if "geopolitics" in n.get("event_tags", []))
    geo_risk = clamp(35 + geopolitics_count * 5)

    # RBI DBIE supplies CPI inflation, repo rate and 10Y G-sec yield. These
    # were previously displayed by the UI but ignored by the regime engine.
    inflation_risk = 50.0
    if macro.inflation_pct is not None:
        inflation_risk = clamp(50 + (macro.inflation_pct - 4.0) * 12)

    rates_risk = 50.0
    if macro.repo_rate_pct is not None:
        rates_risk = clamp(50 + (macro.repo_rate_pct - 5.0) * 8)
    if macro.india_10y_yield_pct is not None:
        rates_risk = clamp(rates_risk + (macro.india_10y_yield_pct - 7.0) * 5)

    equity = "NEUTRAL"
    if vix_risk > 70 or oil_risk > 75 or inflation_risk > 70:
        equity = "CAUTIOUS"
    elif vix_risk < 40 and oil_risk < 55 and inflation_risk < 55:
        equity = "CONSTRUCTIVE"

    rates = "UNKNOWN" if macro.repo_rate_pct is None and macro.india_10y_yield_pct is None else "NEUTRAL"
    if rates != "UNKNOWN" and (oil_risk > 70 or fx_risk > 65 or inflation_risk > 70 or rates_risk > 65):
        rates = "RESTRICTIVE_RISK"
    elif rates != "UNKNOWN" and oil_risk < 45 and fx_risk < 45 and inflation_risk < 45 and rates_risk < 45:
        rates = "EASING_SUPPORTIVE"

    liquidity = "UNKNOWN" if macro.fii_flow_inr_cr is None and macro.dii_flow_inr_cr is None else "MIXED"
    foreign_tags = sum(1 for n in news if "foreign_flows" in n.get("event_tags", []))
    if foreign_tags >= 4:
        liquidity = "VOLATILE"

    inflation = "UNKNOWN" if macro.inflation_pct is None and macro.brent_crude_usd is None else (
        "ELEVATED_RISK" if inflation_risk > 65 else "CONTROLLED"
    )
    currency = "DEPRECIATION_RISK" if fx_risk > 60 else "STABLE"
    oil = "HIGH_RISK" if oil_risk > 70 else "MODERATE" if oil_risk > 50 else "BENIGN"
    valuation = "UNKNOWN"
    geopolitics = "HIGH_RISK" if geo_risk > 65 else "ELEVATED" if geo_risk > 45 else "LOWER"
    climate = "WATCH" if any("climate" in n.get("event_tags", []) for n in news) else "NORMAL"

    risk_components = [oil_risk, fx_risk, vix_risk, geo_risk, inflation_risk, rates_risk]
    overall_score = sum(risk_components) / len(risk_components)
    overall = "DEFENSIVE" if overall_score > 70 else "CAUTIOUS" if overall_score > 55 else "CONSTRUCTIVE"

    signals = [
        RegimeSignal("oil_risk", oil_risk, "NEGATIVE" if oil_risk > 60 else "NEUTRAL", 0.75 if macro.brent_crude_usd is not None else 0.0, "Higher crude raises India's imported inflation and external-balance sensitivity."),
        RegimeSignal("currency_risk", fx_risk, "NEGATIVE" if fx_risk > 60 else "NEUTRAL", 0.70 if macro.usd_inr_change_1m_pct is not None else 0.0, "INR depreciation can pressure inflation and imported input costs."),
        RegimeSignal("volatility_risk", vix_risk, "NEGATIVE" if vix_risk > 60 else "NEUTRAL", 0.80 if macro.india_vix is not None else 0.0, "India VIX is used as a market-stress proxy."),
        RegimeSignal("inflation_risk", inflation_risk, "NEGATIVE" if inflation_risk > 60 else "NEUTRAL", 0.85 if macro.inflation_pct is not None else 0.0, "CPI inflation is compared with the 4% medium-term target."),
        RegimeSignal("rates_risk", rates_risk, "NEGATIVE" if rates_risk > 60 else "NEUTRAL", 0.75 if macro.repo_rate_pct is not None or macro.india_10y_yield_pct is not None else 0.0, "Policy and long-term government-bond rates are used as the domestic rates signal."),
        RegimeSignal("geopolitical_risk", geo_risk, "NEGATIVE" if geo_risk > 60 else "NEUTRAL", 0.55 if news else 0.0, "News-tagged geopolitical events increase risk-premium sensitivity."),
    ]
    weighted = [
        (macro.brent_crude_usd is not None, 0.25),
        (macro.usd_inr_change_1m_pct is not None, 0.15),
        (macro.india_vix is not None, 0.15),
        (macro.inflation_pct is not None, 0.15),
        (macro.repo_rate_pct is not None or macro.india_10y_yield_pct is not None, 0.15),
        (bool(news), 0.10),
        (macro.fii_flow_inr_cr is not None or macro.dii_flow_inr_cr is not None, 0.05),
    ]
    data_confidence = round(sum(weight for present, weight in weighted if present) * 100, 2)
    return MarketRegime(equity, rates, liquidity, inflation, currency, oil, valuation, geopolitics, climate, overall, signals, data_confidence)
