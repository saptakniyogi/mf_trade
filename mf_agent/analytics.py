from __future__ import annotations

import math
import statistics

from .models import FundRecord
from .utils import clamp, normalize


def portfolio_concentration(weights: dict[str, float]) -> float:
    vals = [max(0.0, float(v)) for v in weights.values()]
    total = sum(vals)
    if not total:
        return 0.0
    shares = [v / total for v in vals]
    hhi = sum(x * x for x in shares)
    return clamp(hhi * 100)


def holdings_overlap(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    # weights are treated as percentages or proportions; normalize both.
    ta = sum(max(0.0, float(a.get(k, 0))) for k in keys)
    tb = sum(max(0.0, float(b.get(k, 0))) for k in keys)
    if not ta or not tb:
        return 0.0
    return round(sum(min(a.get(k, 0) / ta, b.get(k, 0) / tb) for k in keys) * 100, 2)


def fund_data_quality(fund: FundRecord) -> tuple[float, list[str]]:
    """Measure evidence completeness separately from investment attractiveness.

    Missing fields are not treated as neutral investment evidence.
    """
    checks = [
        (fund.cagr_3y_pct is not None, 10),
        (fund.cagr_5y_pct is not None, 12),
        (fund.sharpe is not None or fund.sortino is not None, 15),
        (fund.volatility_pct is not None, 10),
        (fund.max_drawdown_pct is not None, 10),
        (bool(fund.holdings), 15),
        (bool(fund.sector_weights), 8),
        (bool(fund.market_cap_weights), 5),
        (bool(fund.valuation), 5),
        (bool(fund.benchmark), 3),
        (fund.aum_inr_cr is not None, 2),
        (fund.latest_nav is not None and fund.nav_date is not None, 5),
    ]
    score = sum(weight for present, weight in checks if present)
    warnings = []
    labels = [
        (fund.cagr_3y_pct is not None, "3-year CAGR unavailable."),
        (fund.cagr_5y_pct is not None, "5-year CAGR unavailable."),
        (fund.sharpe is not None or fund.sortino is not None, "Sharpe/Sortino unavailable."),
        (fund.volatility_pct is not None, "Volatility unavailable."),
        (fund.max_drawdown_pct is not None, "Maximum drawdown unavailable."),
        (bool(fund.holdings), "Underlying holdings unavailable."),
        (bool(fund.sector_weights), "Sector weights unavailable."),
        (bool(fund.market_cap_weights), "Market-cap weights unavailable."),
        (bool(fund.valuation), "Valuation data unavailable."),
        (bool(fund.benchmark), "Benchmark unavailable."),
        (fund.aum_inr_cr is not None, "AUM unavailable."),
        (fund.latest_nav is not None and fund.nav_date is not None, "Current NAV/date unavailable."),
    ]
    warnings.extend(message for present, message in labels if not present)
    return round(float(score), 2), warnings


def fund_quality_score(fund: FundRecord) -> tuple[float, list[str], list[str]]:
    scores = []
    positive = []
    warnings = []
    if fund.cagr_5y_pct is not None:
        scores.append(normalize(fund.cagr_5y_pct, 5, 20))
        if fund.cagr_5y_pct >= 15:
            positive.append("Strong 5-year CAGR relative to the generic long-term equity range.")
    if fund.cagr_3y_pct is not None:
        scores.append(normalize(fund.cagr_3y_pct, 5, 25))
    if fund.max_drawdown_pct is not None:
        scores.append(100 - normalize(abs(fund.max_drawdown_pct), 10, 45))
    if fund.sharpe is not None:
        scores.append(normalize(fund.sharpe, 0, 1.5))
    if fund.source_quality < 0.8:
        warnings.append("Source completeness is below the preferred threshold.")
    # Quality is the mean of available evidence only. Completeness is reported
    # separately so missing metrics cannot masquerade as a neutral score.
    quality = round(sum(scores) / len(scores), 2) if scores else 0.0
    return quality, positive, warnings


def risk_adjusted_score(fund: FundRecord) -> float | None:
    parts = []
    if fund.sharpe is not None:
        parts.append(normalize(fund.sharpe, 0, 1.5))
    if fund.sortino is not None:
        parts.append(normalize(fund.sortino, 0, 2.0))
    if fund.max_drawdown_pct is not None:
        parts.append(100 - normalize(abs(fund.max_drawdown_pct), 10, 50))
    if fund.volatility_pct is not None:
        parts.append(100 - normalize(fund.volatility_pct, 8, 30))
    return round(sum(parts) / len(parts), 2) if parts else None
