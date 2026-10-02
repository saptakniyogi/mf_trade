from __future__ import annotations

from .analytics import fund_data_quality, fund_quality_score, holdings_overlap, risk_adjusted_score
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


def score_fund(
    fund: FundRecord,
    holdings: dict[str, Holding],
    all_funds: list[FundRecord],
    regime: MarketRegime,
) -> FundScore:
    quality, positive, warnings = fund_quality_score(fund)
    risk = risk_adjusted_score(fund)
    valuation = _valuation_score(fund)
    macro = _macro_resilience(fund, regime)
    data_confidence, completeness_warnings = fund_data_quality(fund)
    warnings.extend(completeness_warnings)

    overlaps = []
    for held_name, _holding in holdings.items():
        existing = next((x for x in all_funds if x.scheme_name.lower() == held_name.lower()), None)
        if existing and existing.holdings and fund.holdings:
            overlaps.append(holdings_overlap(fund.holdings, existing.holdings))
    max_overlap = max(overlaps) if overlaps else None

    # Portfolio fit is UNKNOWN when there is no holdings evidence. A score of
    # 95 in that situation was misleading because it implied measured fit.
    fit = None
    if fund.holdings:
        fit = clamp(95 - (max_overlap or 0.0) * 0.65)
        if fund.scheme_name in holdings:
            fit = clamp(fit + 5)
    else:
        warnings.append("Portfolio fit is unknown because underlying holdings are unavailable.")

    available = []
    if valuation is not None:
        available.append((0.45, valuation))
    if risk is not None:
        available.append((0.25, risk))
    available.append((0.30, macro))
    weight_sum = sum(weight for weight, _ in available)
    attractiveness = clamp(sum(weight * value for weight, value in available) / max(weight_sum, 1e-9))

    components = [
        (0.35, quality),
        (0.20, risk),
        (0.20, attractiveness),
        (0.15, fit),
        (0.10, macro),
    ]
    used = [(weight, value) for weight, value in components if value is not None]
    overall = round(sum(weight * value for weight, value in used) / max(sum(weight for weight, _ in used), 1e-9), 2)

    # Confidence-aware ranking prevents high-return but poorly evidenced funds
    # from dominating the local shortlist. Keep the raw score intact for audit.
    confidence_factor = 0.55 + 0.45 * (data_confidence / 100.0)
    ranking_score = round(overall * confidence_factor, 2)

    if max_overlap is not None and max_overlap > 60:
        warnings.append(f"High overlap with an existing holding: {max_overlap:.1f}%.")
    if regime.overall == "DEFENSIVE" and "small" in fund.category.lower():
        warnings.append("Small-cap exposure is less resilient under the current defensive regime.")

    not_buy = []
    if valuation is not None and valuation < 35:
        not_buy.append("Valuation signal is weak or expensive based on available data.")
    if fit is not None and fit < 55:
        not_buy.append("Portfolio fit is weak because of overlap or concentration.")
    if macro < 45:
        not_buy.append("Current macro regime is unfavorable for this fund's category/exposure.")
    if risk is None:
        not_buy.append("Risk-adjusted metrics are unavailable, so the return profile cannot be validated.")
    if data_confidence < 50:
        not_buy.append("Data confidence is low; the quantitative ranking should be treated as provisional.")

    return FundScore(
        fund_quality=round(quality, 2),
        risk_adjusted_return=round(risk, 2) if risk is not None else None,
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
