from __future__ import annotations

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
    ta = sum(max(0.0, float(a.get(k, 0))) for k in keys)
    tb = sum(max(0.0, float(b.get(k, 0))) for k in keys)
    if not ta or not tb:
        return 0.0

    return round(
        sum(min(a.get(k, 0) / ta, b.get(k, 0) / tb) for k in keys) * 100,
        2,
    )


def _is_debt_like(fund: FundRecord) -> bool:
    category = fund.category.lower()
    return any(
        token in category
        for token in (
            "debt",
            "liquid",
            "overnight",
            "money market",
            "gilt",
            "duration",
            "bond",
            "credit risk",
        )
    )


def fund_data_quality(fund: FundRecord) -> tuple[float, list[str]]:
    """Measure evidence completeness independently from attractiveness.

    Missing evidence is explicitly penalized. It is never treated as neutral
    investment evidence.
    """
    debt_like = _is_debt_like(fund)

    checks = [
        (fund.cagr_1y_pct is not None, 8),
        (fund.cagr_3y_pct is not None, 12),
        (fund.cagr_5y_pct is not None, 10),
        (fund.sharpe is not None or fund.sortino is not None, 12),
        (fund.volatility_pct is not None, 8),
        (fund.max_drawdown_pct is not None, 10),
        (bool(fund.holdings) if not debt_like else True, 15),
        (bool(fund.sector_weights) if not debt_like else True, 8),
        (bool(fund.market_cap_weights) if not debt_like else True, 5),
        (bool(fund.valuation), 5),
        (bool(fund.benchmark), 3),
        (fund.aum_inr_cr is not None, 2),
        (fund.latest_nav is not None and fund.nav_date is not None, 2),
    ]

    score = sum(weight for present, weight in checks if present)

    labels = [
        (fund.cagr_1y_pct is not None, "1-year CAGR unavailable."),
        (fund.cagr_3y_pct is not None, "3-year CAGR unavailable."),
        (fund.cagr_5y_pct is not None, "5-year CAGR unavailable."),
        (
            fund.sharpe is not None or fund.sortino is not None,
            "Sharpe/Sortino unavailable.",
        ),
        (fund.volatility_pct is not None, "Volatility unavailable."),
        (fund.max_drawdown_pct is not None, "Maximum drawdown unavailable."),
        (bool(fund.holdings) if not debt_like else True, "Underlying holdings unavailable."),
        (
            bool(fund.sector_weights) if not debt_like else True,
            "Sector weights unavailable.",
        ),
        (
            bool(fund.market_cap_weights) if not debt_like else True,
            "Market-cap weights unavailable.",
        ),
        (bool(fund.valuation), "Valuation data unavailable."),
        (bool(fund.benchmark), "Benchmark unavailable."),
        (fund.aum_inr_cr is not None, "AUM unavailable."),
        (
            fund.latest_nav is not None and fund.nav_date is not None,
            "Current NAV/date unavailable.",
        ),
    ]

    warnings = [message for present, message in labels if not present]
    return round(float(score), 2), warnings


def evidence_status(
    fund: FundRecord,
    data_confidence: float,
    *,
    min_history_years: float = 3.0,
    min_allocation_confidence: float = 70.0,
) -> tuple[str, bool, list[str]]:
    """Return evidence status and whether the fund can receive fresh allocation.

    A long-term recommendation requires at least the configured historical
    horizon plus risk evidence. Equity funds also require holdings evidence.
    Debt-like funds are not penalized for equity-specific holdings/sector data,
    but they still require a meaningful return/risk history.
    """
    debt_like = _is_debt_like(fund)
    blockers: list[str] = []

    if data_confidence < min_allocation_confidence:
        blockers.append(
            f"Data confidence {data_confidence:.1f}% is below the "
            f"{min_allocation_confidence:.1f}% allocation threshold."
        )

    if min_history_years >= 3 and fund.cagr_3y_pct is None:
        blockers.append("At least 3 years of return history is required for long-term allocation.")

    if fund.sharpe is None and fund.sortino is None:
        blockers.append("A risk-adjusted return metric is required.")

    if fund.max_drawdown_pct is None:
        blockers.append("Maximum drawdown is required.")

    if fund.aum_inr_cr is None:
        blockers.append("AUM is required.")

    if fund.benchmark is None:
        blockers.append("Benchmark is required.")

    if not debt_like and not fund.holdings:
        blockers.append("Underlying holdings are required for equity portfolio-fit analysis.")

    if blockers:
        return "INSUFFICIENT_DATA", False, blockers

    return "ELIGIBLE", True, []


def fund_quality_score(
    fund: FundRecord,
    *,
    min_history_years: float = 3.0,
) -> tuple[float, list[str], list[str]]:
    """Calculate quality without allowing sparse data to masquerade as quality.

    The score is based on available evidence, but missing core evidence applies
    an explicit penalty. A fund without the minimum historical record cannot
    receive a full quality score.
    """
    scores: list[float] = []
    positive: list[str] = []
    warnings: list[str] = []

    if fund.cagr_5y_pct is not None:
        scores.append(normalize(fund.cagr_5y_pct, 5, 20))
        if fund.cagr_5y_pct >= 15:
            positive.append(
                "Strong 5-year CAGR relative to the generic long-term equity range."
            )

    if fund.cagr_3y_pct is not None:
        scores.append(normalize(fund.cagr_3y_pct, 5, 25))

    if fund.max_drawdown_pct is not None:
        scores.append(100 - normalize(abs(fund.max_drawdown_pct), 10, 45))

    if fund.sharpe is not None:
        scores.append(normalize(fund.sharpe, 0, 1.5))
    elif fund.sortino is not None:
        scores.append(normalize(fund.sortino, 0, 2.0))

    if fund.source_quality < 0.8:
        warnings.append("Source completeness is below the preferred threshold.")

    if not scores:
        return 0.0, positive, warnings

    quality = sum(scores) / len(scores)

    # Core historical evidence is mandatory for a long-term strategy.
    if min_history_years >= 3 and fund.cagr_3y_pct is None:
        warnings.append("Quality score capped because 3-year return history is unavailable.")
        quality = min(quality, 45.0)

    # A missing risk history should also prevent a high-confidence quality score.
    if fund.sharpe is None and fund.sortino is None:
        warnings.append("Quality score capped because risk-adjusted return data is unavailable.")
        quality = min(quality, 45.0)

    return round(clamp(quality), 2), positive, warnings


def risk_adjusted_score(fund: FundRecord) -> float | None:
    parts: list[float] = []

    if fund.sharpe is not None:
        parts.append(normalize(fund.sharpe, 0, 1.5))

    if fund.sortino is not None:
        parts.append(normalize(fund.sortino, 0, 2.0))

    if fund.max_drawdown_pct is not None:
        parts.append(100 - normalize(abs(fund.max_drawdown_pct), 10, 50))

    if fund.volatility_pct is not None:
        parts.append(100 - normalize(fund.volatility_pct, 8, 30))

    return round(sum(parts) / len(parts), 2) if parts else None
