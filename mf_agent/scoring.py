from __future__ import annotations

from .analytics import fund_quality_score, holdings_overlap, risk_adjusted_score
from .models import FundRecord, Holding, MarketRegime, FundScore
from .utils import clamp, normalize


def _valuation_score(fund: FundRecord) -> float | None:
    pe = fund.valuation.get("pe")
    percentile = fund.valuation.get("historical_percentile")
    if percentile is not None:
        return round(100 - clamp(percentile), 2)
    if pe is not None:
        return round(100 - normalize(pe, 15, 40), 2)
    return None


def _macro_resilience(fund: FundRecord, regime: MarketRegime) -> float:
    score = 60.0
    cat = fund.category.lower()
    if "small" in cat and regime.overall in {"CAUTIOUS", "DEFENSIVE"}:
        score -= 20
    if "mid" in cat and regime.overall == "DEFENSIVE":
        score -= 12
    if "thematic" in cat:
        score -= 8
    if regime.oil in {"HIGH_RISK", "MODERATE"} and any(x in cat for x in ("infrastructure", "thematic")):
        score -= 5
    if "index" in cat or "large" in cat:
        score += 5
    return clamp(score)


def _data_confidence(fund: FundRecord) -> float:
    """Evidence completeness, not investment quality.

    Equity-like funds get credit for holdings/sector/market-cap/valuation evidence.
    Debt/hybrid funds are not penalized for equity-only market-cap data when it is not
    applicable. The score intentionally measures evidence availability only.
    """
    category = fund.category.lower()
    debt_like = any(x in category for x in ("debt", "liquid", "overnight", "money market", "gilt", "duration", "bond", "credit risk"))
    checks: list[tuple[bool, float]] = [
        (fund.cagr_1y_pct is not None, 10),
        (fund.cagr_3y_pct is not None, 10),
        (fund.cagr_5y_pct is not None, 10),
        (fund.sharpe is not None or fund.sortino is not None, 10),
        (fund.volatility_pct is not None, 5),
        (fund.max_drawdown_pct is not None, 10),
        (fund.aum_inr_cr is not None, 5),
        (fund.benchmark is not None, 5),
        (bool(fund.valuation), 10),
        (bool(fund.holdings), 15),
        (bool(fund.sector_weights) if not debt_like else True, 5),
        (bool(fund.market_cap_weights) if not debt_like else True, 5),
    ]
    total = sum(weight for _, weight in checks)
    available = sum(weight for ok, weight in checks if ok)
    return round(100 * available / total, 1) if total else 0.0


def _portfolio_fit(
    fund: FundRecord,
    holdings: dict[str, Holding],
    all_funds: list[FundRecord],
) -> tuple[float | None, float, bool]:
    """Return fit, max overlap, and whether fit evidence is available."""
    if not fund.holdings:
        return None, 0.0, False

    overlaps = []
    for held_name in holdings:
        existing = next((x for x in all_funds if x.scheme_name.lower() == held_name.lower()), None)
        if existing and existing.holdings:
            overlaps.append(holdings_overlap(fund.holdings, existing.holdings))

    max_overlap = max(overlaps) if overlaps else 0.0
    fit = clamp(95 - max_overlap * 0.65)
    if any(fund.scheme_name.lower() == name.lower() for name in holdings):
        fit = clamp(fit + 5)
    return round(fit, 2), max_overlap, True


def score_fund(
    fund: FundRecord,
    holdings: dict[str, Holding],
    all_funds: list[FundRecord],
    regime: MarketRegime,
) -> FundScore:
    quality, positive, warnings = fund_quality_score(fund)
    risk = risk_adjusted_score(fund)
    # risk_adjusted_score historically returned 50 when no evidence existed. Treat
    # that as unknown here so missing data cannot create artificial support.
    risk_available = any(
        x is not None
        for x in (fund.sharpe, fund.sortino, fund.max_drawdown_pct, fund.volatility_pct)
    )
    risk_value = risk if risk_available else None

    valuation = _valuation_score(fund)
    macro = _macro_resilience(fund, regime)
    fit, max_overlap, fit_available = _portfolio_fit(fund, holdings, all_funds)

    if valuation is not None and risk_value is not None:
        attractiveness = clamp(0.45 * valuation + 0.30 * macro + 0.25 * risk_value)
    elif valuation is not None:
        attractiveness = clamp(0.60 * valuation + 0.40 * macro)
    elif risk_value is not None:
        attractiveness = clamp(0.55 * macro + 0.45 * risk_value)
    else:
        attractiveness = macro

    components: list[tuple[float, float]] = [
        (quality, 0.35),
        (attractiveness, 0.20),
        (macro, 0.10),
    ]
    if risk_value is not None:
        components.append((risk_value, 0.20))
    if fit is not None:
        components.append((fit, 0.15))

    weight_sum = sum(weight for _, weight in components)
    overall = round(sum(value * weight for value, weight in components) / weight_sum, 2)

    if max_overlap > 60:
        warnings.append(f"High overlap with an existing holding: {max_overlap:.1f}%.")
    if regime.overall == "DEFENSIVE" and "small" in fund.category.lower():
        warnings.append("Small-cap exposure is less resilient under the current defensive regime.")
    if not fund.holdings:
        warnings.append("Underlying holdings are unavailable, so overlap and factor analysis are incomplete.")
    if valuation is None:
        warnings.append("Valuation data unavailable.")
    if fund.benchmark is None:
        warnings.append("Benchmark unavailable.")
    if fund.aum_inr_cr is None:
        warnings.append("AUM unavailable.")
    if not fund.sector_weights and "debt" not in fund.category.lower():
        warnings.append("Sector weights unavailable.")
    if not fund.market_cap_weights and "debt" not in fund.category.lower():
        warnings.append("Market-cap weights unavailable.")

    data_confidence = _data_confidence(fund)
    ranking_score = round(overall * (0.55 + 0.45 * data_confidence / 100.0), 2)

    not_buy = []
    if valuation is not None and valuation < 35:
        not_buy.append("Valuation signal is weak or expensive based on available data.")
    if fit is not None and fit < 55:
        not_buy.append("Portfolio fit is weak because of overlap or concentration.")
    if macro < 45:
        not_buy.append("Current macro regime is unfavorable for this fund's category/exposure.")
    if data_confidence < 60:
        not_buy.append("Data confidence is low; the quantitative ranking should be treated as provisional.")
    if risk_value is None:
        not_buy.append("Risk-adjusted metrics are unavailable, so the return profile cannot be validated.")

    # Deduplicate warnings while preserving order.
    warnings = list(dict.fromkeys(warnings))

    return FundScore(
        fund_quality=round(quality, 2),
        risk_adjusted_return=round(risk_value, 2) if risk_value is not None else None,
        current_attractiveness=round(attractiveness, 2),
        portfolio_fit=round(fit, 2) if fit is not None else None,
        valuation=round(valuation, 2) if valuation is not None else None,
        macro_resilience=round(macro, 2),
        overall=overall,
        data_confidence=data_confidence,
        ranking_score=ranking_score,
        reasons_to_buy=positive,
        reasons_not_to_buy=not_buy,
        data_warnings=warnings,
    )
