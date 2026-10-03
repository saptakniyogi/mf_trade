from __future__ import annotations

from collections import defaultdict

from .config import PortfolioPolicy
from .models import FundRecord, Holding, FundScore


def existing_exposure(
    holdings: dict[str, Holding],
    funds: list[FundRecord],
) -> dict:
    total = sum(max(0.0, h.current_value) for h in holdings.values())
    by_category = defaultdict(float)
    by_amc = defaultdict(float)

    for name, holding in holdings.items():
        fund = next(
            (f for f in funds if f.scheme_name.lower() == name.lower()),
            None,
        )

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


def _candidate_capacity(
    fund: FundRecord,
    category: defaultdict,
    amc: defaultdict,
    policy: PortfolioPolicy,
) -> float:
    """Return maximum additional portfolio percentage available."""
    cat = fund.category
    max_cat = policy.max_category_pct
    cat_lower = cat.lower()

    if "small" in cat_lower:
        max_cat = min(max_cat, policy.max_small_cap_pct)
    elif "mid" in cat_lower:
        max_cat = min(max_cat, policy.max_mid_cap_pct)
    elif "thematic" in cat_lower:
        max_cat = min(max_cat, policy.max_thematic_pct)

    available_cat = max(0.0, max_cat - category[cat])
    available_amc = max(0.0, policy.max_single_amc_pct - amc[fund.amc])

    return max(
        0.0,
        min(
            policy.max_single_fund_pct,
            available_cat,
            available_amc,
        ),
    )


def _eligible_candidates(
    candidates: list[tuple[FundRecord, FundScore]],
    category: defaultdict,
    amc: defaultdict,
    policy: PortfolioPolicy,
) -> list[tuple[FundRecord, FundScore]]:
    """Apply hard evidence and portfolio gates before any capital is allocated."""
    eligible = []

    for fund, score in sorted(
        candidates,
        key=lambda x: x[1].ranking_score,
        reverse=True,
    ):
        if not score.allocation_eligible:
            continue

        if score.evidence_status != "ELIGIBLE":
            continue

        if score.ranking_score < 50:
            continue

        if _candidate_capacity(fund, category, amc, policy) <= 0:
            continue

        eligible.append((fund, score))

    return eligible


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

    ranked = sorted(
        candidates,
        key=lambda x: x[1].ranking_score,
        reverse=True,
    )
    mode = str(mode or "DIVERSIFIED").upper()

    eligible = _eligible_candidates(
        ranked,
        category,
        amc,
        policy,
    )

    if not eligible or deployable_cash <= 0:
        return []

    if mode == "CONCENTRATED":
        max_funds = max(1, int(policy.max_concentrated_funds))
        target_pct = max(
            0.0,
            min(100.0, 100.0 - max(0.0, policy.min_cash_pct)),
        )

        eligible = eligible[:max_funds]

        strengths = {
            fund.scheme_name: max(1.0, score.ranking_score - 50.0)
            for fund, score in eligible
        }

        allocations = {
            fund.scheme_name: 0.0
            for fund, _ in eligible
        }

        remaining_pct = target_pct
        step_pct = 0.25

        while remaining_pct > 1e-9:
            available = []

            for fund, score in eligible:
                capacity = _candidate_capacity(
                    fund,
                    category,
                    amc,
                    policy,
                )

                if capacity > allocations[fund.scheme_name] + 1e-9:
                    available.append((fund, score, capacity))

            if not available:
                break

            available.sort(
                key=lambda item: (
                    strengths[item[0].scheme_name]
                    / (1.0 + allocations[item[0].scheme_name]),
                    item[1].ranking_score,
                ),
                reverse=True,
            )

            fund, score, capacity = available[0]

            increment = min(
                step_pct,
                remaining_pct,
                max(
                    0.0,
                    capacity - allocations[fund.scheme_name],
                ),
            )

            if increment <= 1e-9:
                break

            allocations[fund.scheme_name] += increment
            category[fund.category] += increment
            amc[fund.amc] += increment
            remaining_pct -= increment

        selected = []

        for fund, score in eligible:
            pct = allocations[fund.scheme_name]

            if pct <= 0:
                continue

            capital = round(
                deployable_cash * pct / 100.0,
                2,
            )

            if capital <= 0:
                continue

            selected.append(
                {
                    "scheme_name": fund.scheme_name,
                    "category": fund.category,
                    "amc": fund.amc,
                    "score": score.overall,
                    "ranking_score": score.ranking_score,
                    "allocation_pct_of_new_cash": round(pct, 2),
                    "capital_required": capital,
                    "action": (
                        "ACCUMULATE"
                        if fund.scheme_name in holdings
                        else "BUY"
                    ),
                    "evidence_status": score.evidence_status,
                    "data_confidence": score.data_confidence,
                }
            )

        return selected

    target_deployable_cash = deployable_cash * (
        1.0 - max(
            0.0,
            min(100.0, policy.min_cash_pct),
        ) / 100.0
    )

    remaining_target_cash = max(
        0.0,
        target_deployable_cash,
    )

    total_score = sum(
        max(0.0, score.ranking_score - 50.0)
        for _, score in eligible
    )

    selected = []

    for fund, score in eligible:
        if remaining_target_cash <= 0:
            break

        cap = _candidate_capacity(
            fund,
            category,
            amc,
            policy,
        )

        if cap <= 0:
            continue

        raw_weight = (
            (
                score.ranking_score - 50.0
            )
            / max(total_score, 1.0)
        ) * 100.0

        weight = min(
            cap,
            max(2.0, raw_weight),
        )

        capital = round(
            target_deployable_cash * weight / 100.0,
            2,
        )

        capital = min(
            capital,
            remaining_target_cash,
        )

        weight = (
            capital / deployable_cash * 100.0
            if deployable_cash
            else 0.0
        )

        if capital <= 0:
            continue

        selected.append(
            {
                "scheme_name": fund.scheme_name,
                "category": fund.category,
                "amc": fund.amc,
                "score": score.overall,
                "ranking_score": score.ranking_score,
                "allocation_pct_of_new_cash": round(weight, 2),
                "capital_required": round(capital, 2),
                "action": (
                    "ACCUMULATE"
                    if fund.scheme_name in holdings
                    else "BUY"
                ),
                "evidence_status": score.evidence_status,
                "data_confidence": score.data_confidence,
            }
        )

        remaining_target_cash -= capital
        category[fund.category] += weight
        amc[fund.amc] += weight

    return selected
