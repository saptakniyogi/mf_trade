from __future__ import annotations

from dataclasses import dataclass, asdict

from .models import FundRecord, MarketRegime
from .utils import clamp


@dataclass(frozen=True)
class Scenario:
    name: str
    oil_shock: float = 0.0
    equity_shock: float = 0.0
    rate_shock: float = 0.0
    fx_shock: float = 0.0
    geopolitical_shock: float = 0.0


SCENARIOS = (
    Scenario("base"),
    Scenario("oil_shock", oil_shock=1.0, rate_shock=0.4, fx_shock=0.5, geopolitical_shock=0.5),
    Scenario("global_recession", equity_shock=1.0, rate_shock=0.7, geopolitical_shock=0.3),
    Scenario("india_bull", equity_shock=-0.7, rate_shock=-0.4, fx_shock=-0.3),
)


def stress_fund(fund: FundRecord, regime: MarketRegime, scenario: Scenario) -> dict:
    category = fund.category.lower()
    sensitivity = 0.0
    reasons = []

    if "small" in category:
        sensitivity += 20 * scenario.equity_shock
        reasons.append("Small-cap exposure is more sensitive to equity risk-premium shocks.") if scenario.equity_shock else None
    elif "mid" in category:
        sensitivity += 12 * scenario.equity_shock
    elif "large" in category or "index" in category:
        sensitivity += 7 * scenario.equity_shock

    if "thematic" in category:
        sensitivity += 8 * scenario.geopolitical_shock

    if scenario.oil_shock:
        if any(x in category for x in ("infrastructure", "thematic")):
            sensitivity += 5 * scenario.oil_shock
        else:
            sensitivity += 2 * scenario.oil_shock

    if scenario.rate_shock:
        if any(x in category for x in ("mid", "small", "large & mid")):
            sensitivity += 8 * scenario.rate_shock
        elif "debt" in category or "duration" in category:
            sensitivity += 12 * scenario.rate_shock

    resilience = clamp(70 - sensitivity)
    if scenario.name == "india_bull":
        resilience = clamp(70 + abs(sensitivity) * 0.5)

    return {
        "scenario": asdict(scenario),
        "resilience_score": round(resilience, 2),
        "risk_penalty": round(sensitivity, 2),
        "reasons": reasons,
    }


def scenario_matrix(fund: FundRecord, regime: MarketRegime) -> list[dict]:
    return [stress_fund(fund, regime, scenario) for scenario in SCENARIOS]
