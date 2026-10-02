from __future__ import annotations

from collections import defaultdict

from .config import PortfolioPolicy
from .models import FundRecord, Holding, FundScore
from .utils import clamp


def existing_exposure(holdings: dict[str, Holding], funds: list[FundRecord]) -> dict:
    total = sum(max(0.0, h.current_value) for h in holdings.values())
    by_category = defaultdict(float)
    by_amc = defaultdict(float)
    for name, holding in holdings.items():
        fund = next((f for f in funds if f.scheme_name.lower() == name.lower()), None)
        if not fund or total <= 0:
            continue
        weight = holding.current_value / total * 100
        by_category[fund.category] += weight
        by_amc[fund.amc] += weight
    return {
        "total_value": total,
        "category_weights": dict(by_category),
        "amc_weights": dict(by_amc),
    }


def allocation_for_candidates(
    candidates: list[tuple[FundRecord, FundScore]],
    holdings: dict[str, Holding],
    funds: list[FundRecord],
    deployable_cash: float,
    policy: PortfolioPolicy,
    mode: str,
) -> list[dict]:
    exposure = existing_exposure(holdings, funds)
    category = defaultdict(float, exposure["category_weights"])
    amc = defaultdict(float, exposure["amc_weights"])

    ranked = sorted(candidates, key=lambda x: x[1].ranking_score, reverse=True)
    selected = []
    remaining = max(0.0, deployable_cash)
    total_score = sum(max(0.0, score.ranking_score - 50) for _, score in ranked)

    for fund, score in ranked:
        if remaining <= 0 or score.overall < 60 or score.data_confidence < 50:
            continue
        cat = fund.category
        max_cat = policy.max_category_pct
        if "small" in cat.lower():
            max_cat = min(max_cat, policy.max_small_cap_pct)
        elif "mid" in cat.lower():
            max_cat = min(max_cat, policy.max_mid_cap_pct)
        elif "thematic" in cat.lower():
            max_cat = min(max_cat, policy.max_thematic_pct)

        available_cat = max(0.0, max_cat - category[cat])
        available_amc = max(0.0, policy.max_single_amc_pct - amc[fund.amc])
        cap = min(policy.max_single_fund_pct, available_cat, available_amc)
        if cap <= 0:
            continue

        raw_weight = ((score.ranking_score - 50) / max(total_score, 1.0)) * 100
        weight = min(cap, max(2.0, raw_weight))
        if mode == "CONCENTRATED":
            weight = min(cap, weight * 1.25)
        capital = round(deployable_cash * weight / 100.0, 2)
        capital = min(capital, remaining)
        weight = capital / deployable_cash * 100.0 if deployable_cash else 0.0
        if capital <= 0:
            continue

        selected.append({
            "scheme_name": fund.scheme_name,
            "category": fund.category,
            "amc": fund.amc,
            "score": score.overall,
            "allocation_pct_of_new_cash": round(weight, 2),
            "capital_required": round(capital, 2),
            "action": "ACCUMULATE" if fund.scheme_name in holdings else "BUY",
        })
        remaining -= capital
        category[cat] += weight
        amc[fund.amc] += weight

    return selected
