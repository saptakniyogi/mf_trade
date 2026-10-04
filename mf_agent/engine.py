from __future__ import annotations

import logging
import os
from dataclasses import asdict

from .config import Settings
from .data import build_fund_universe, load_holdings
from .embeddings import retrieve_relevant_funds, retrieve_relevant_news
from .macro import fetch_macro_snapshot
from .news import classify_event, fetch_news, filter_relevant_news
from .portfolio import (
    allocation_for_candidates,
    existing_exposure,
    holding_for_fund,
    investment_option_status,
    sip_allocation_for_candidates,
)
from .ranking import diversify_shortlist, rank_candidates
from .regime import infer_regime
from .scenarios import scenario_matrix
from .scoring import score_fund

logger = logging.getLogger("mf_agent")


def _trim_reason(score) -> str | None:
    """Return explicit deterministic evidence required before recommending TRIM."""
    if score.fund_quality < 40.0:
        return (
            f"Fund quality score {score.fund_quality:.2f} is below the "
            "40.0 trim threshold."
        )
    if score.risk_adjusted_return is not None and score.risk_adjusted_return < 35.0:
        return (
            f"Risk-adjusted return score {score.risk_adjusted_return:.2f} is below "
            "the 35.0 trim threshold."
        )
    if score.valuation is not None and score.valuation < 30.0:
        return (
            f"Valuation score {score.valuation:.2f} indicates a materially weak "
            "valuation signal."
        )
    if score.portfolio_fit is not None and score.portfolio_fit < 55.0:
        return (
            f"Portfolio-fit score {score.portfolio_fit:.2f} indicates excessive "
            "overlap or concentration."
        )
    if score.macro_resilience < 45.0:
        return (
            f"Macro resilience {score.macro_resilience:.2f} is below the "
            "45.0 defensive threshold."
        )
    return None


def _canonicalize_sip_recommendations(
    recommendations: list[dict],
    candidates: list[tuple[object, object]],
    holdings: dict,
    funds: list,
    monthly_amount: float,
    policy,
    mode: str,
) -> list[dict]:
    """Normalize legacy SIP recommendation rows into the 2.6.2 allocation contract.

    Older 2.6.x runs could expose SIP eligibility records instead of the
    fund-level allocation plan. The engine must never publish that legacy
    shape as ``sip_recommendations`` because the audit and dashboard consume
    this field as an allocation plan.
    """
    required = {
        "action",
        "target_sip_pct",
        "portfolio_capacity_pct",
        "monthly_amount",
        "allocation_pct_of_monthly_sip",
    }

    if not recommendations:
        return recommendations

    if all(required.issubset(item.keys()) for item in recommendations if isinstance(item, dict)):
        return recommendations

    # Prefer the deterministic allocator again. This path is deliberately
    # defensive: it repairs a legacy allocator result without changing scores
    # or the investment-ranking formula.
    from .portfolio import sip_allocation_for_candidates as _sip_allocator

    canonical = _sip_allocator(
        candidates,
        holdings,
        funds,
        monthly_amount,
        policy,
        mode,
    )
    if canonical and all(required.issubset(item.keys()) for item in canonical):
        return canonical

    # If a legacy portfolio implementation is still imported at runtime,
    # construct the same deterministic allocation contract here from the
    # already-qualified candidates. This is intentionally the same strength
    # weighting used by the 2.6.1 allocator.
    from collections import defaultdict
    from .portfolio import existing_exposure, holding_for_fund, ROUTE_RECOMMENDATION_SCORE

    exposure = existing_exposure(holdings, funds)
    total_value = float(exposure.get("total_value", 0.0) or 0.0)

    eligible = [
        (fund, score)
        for fund, score in sorted(
            candidates,
            key=lambda x: x[1].ranking_score,
            reverse=True,
        )
        if score.allocation_eligible
        and score.evidence_status == "ELIGIBLE"
        and score.ranking_score >= ROUTE_RECOMMENDATION_SCORE
        and score.macro_resilience >= 45.0
    ]

    if str(mode or "DIVERSIFIED").upper() == "CONCENTRATED":
        eligible = eligible[: max(1, int(policy.max_concentrated_funds))]

    # Recover capacity from the legacy recommendation when available. This
    # avoids inventing a different portfolio-capacity calculation merely to
    # repair the schema.
    legacy_by_name = {
        str(item.get("scheme_name")): item
        for item in recommendations
        if isinstance(item, dict)
    }

    strengths = {
        fund.scheme_name: max(
            1.0,
            float(score.ranking_score) - ROUTE_RECOMMENDATION_SCORE,
        )
        for fund, score in eligible
    }
    total_strength = sum(strengths.values()) or float(len(eligible))

    rows = []
    for fund, score in eligible:
        legacy = legacy_by_name.get(fund.scheme_name, {})
        nested_options = legacy.get("investment_options") or {}
        legacy_capacity = nested_options.get("portfolio_capacity_pct")

        if isinstance(legacy_capacity, (int, float)):
            capacity_pct = max(0.0, min(100.0, float(legacy_capacity)))
        else:
            capacity_pct = 0.0

        if capacity_pct <= 0.0:
            # A missing capacity is not treated as unlimited. The allocator
            # contract requires a real capacity value.
            continue

        target_pct = strengths[fund.scheme_name] / total_strength * 100.0
        max_monthly = (
            total_value * capacity_pct / 100.0
            if total_value > 0
            else None
        )
        desired = (
            float(monthly_amount) * target_pct / 100.0
            if monthly_amount > 0
            else 0.0
        )
        amount = (
            min(desired, max_monthly)
            if max_monthly is not None
            else desired
        )
        holding = holding_for_fund(fund, holdings)

        rows.append({
            "scheme_name": fund.scheme_name,
            "category": fund.category,
            "amc": fund.amc,
            "action": "ACCUMULATE" if holding is not None else "BUY",
            "score": score.overall,
            "ranking_score": score.ranking_score,
            "data_confidence": score.data_confidence,
            "evidence_status": score.evidence_status,
            "portfolio_capacity_pct": round(capacity_pct, 2),
            "target_sip_pct": round(target_pct, 2),
            "monthly_amount": round(amount, 2),
            "max_monthly_amount": round(max_monthly, 2) if max_monthly is not None else None,
            "allocation_pct_of_monthly_sip": (
                round(amount / float(monthly_amount) * 100.0, 2)
                if monthly_amount > 0
                else 0.0
            ),
            "reason": (
                "Deterministic SIP candidate: passes evidence, score, macro "
                "and portfolio-capacity gates."
            ),
            "final_action": "ACCUMULATE" if holding is not None else "BUY",
            "sip_monthly_amount": round(amount, 2),
        })

    # Reconcile rounding and capacity so the actual monthly amounts never
    # exceed the configured SIP budget.
    if monthly_amount > 0 and rows:
        remaining = float(monthly_amount)
        for row in sorted(rows, key=lambda x: x["ranking_score"], reverse=True):
            amount = min(
                max(0.0, float(row["monthly_amount"])),
                remaining,
            )
            row["monthly_amount"] = round(amount, 2)
            row["sip_monthly_amount"] = row["monthly_amount"]
            row["allocation_pct_of_monthly_sip"] = round(
                amount / float(monthly_amount) * 100.0,
                2,
            )
            remaining -= amount

    return rows


class ResearchEngine:
    def __init__(self, settings: Settings):
        self.settings = settings

    def build(self) -> dict:
        holdings, funds_info = load_holdings(self.settings)

        topics = [
            x.strip()
            for x in os.getenv("TRENDING_TOPICS", "").split(",")
            if x.strip()
        ]

        macro = fetch_macro_snapshot(topics)

        raw_news = fetch_news(self.settings)
        classified_news = [
            classify_event(x)
            for x in raw_news
        ]
        news = filter_relevant_news(
            classified_news,
            limit=40,
        )

        regime = infer_regime(
            macro,
            news,
        )

        funds = build_fund_universe(
            self.settings,
            holdings,
        )

        scored = []

        for fund in funds:
            score = score_fund(
                fund,
                holdings,
                funds,
                regime,
                min_history_years=self.settings.min_history_years,
            )
            scored.append((fund, score))

        # Candidates for capital allocation are explicitly evidence-qualified.
        candidates = [
            (fund, score)
            for fund, score in scored
            if score.allocation_eligible
            and score.evidence_status == "ELIGIBLE"
        ]

        allocation_candidates = (
            candidates
            if self.settings.investment_mode in {"ONE_TIME", "BOTH"}
            else []
        )

        allocations = allocation_for_candidates(
            allocation_candidates,
            holdings,
            funds,
            funds_info["deployable_cash"],
            self.settings.policy,
            self.settings.allocation_mode,
        )

        allocation_map = {
            item["scheme_name"]: item
            for item in allocations
        }

        sip_candidates = [
            (fund, score)
            for fund, score in scored
            if score.allocation_eligible
            and score.evidence_status == "ELIGIBLE"
        ]
        sip_recommendations = sip_allocation_for_candidates(
            sip_candidates,
            holdings,
            funds,
            self.settings.sip_monthly_amount,
            self.settings.policy,
            self.settings.allocation_mode,
        )
        sip_recommendations = _canonicalize_sip_recommendations(
            sip_recommendations,
            sip_candidates,
            holdings,
            funds,
            self.settings.sip_monthly_amount,
            self.settings.policy,
            self.settings.allocation_mode,
        )

        evaluated = []

        for fund, score in sorted(
            scored,
            key=lambda x: (
                x[1].ranking_score,
                x[1].data_confidence,
                x[1].fund_quality,
            ),
            reverse=True,
        ):
            holding = holding_for_fund(fund, holdings)
            is_holding = holding is not None
            is_unresolved_holding = (
                is_holding and not fund.scheme_code
            )

            # Never issue a position-changing recommendation when the holding
            # cannot be mapped to a canonical AMFI scheme. Missing identity is a
            # data-quality problem, not evidence that the investor should trim.
            action = "HOLD" if is_holding else "WAIT"
            trim_reason = None

            if fund.scheme_name in allocation_map:
                action = allocation_map[fund.scheme_name]["action"]
            elif (
                is_holding
                and not is_unresolved_holding
                and score.allocation_eligible
                and score.evidence_status == "ELIGIBLE"
                and score.overall < 45
            ):
                trim_reason = _trim_reason(score)
                if trim_reason:
                    action = "TRIM"

            if trim_reason and trim_reason not in score.reasons_not_to_buy:
                score.reasons_not_to_buy.append(trim_reason)

            evaluated.append(
                {
                    "scheme_name": fund.scheme_name,
                    "category": fund.category,
                    "amc": fund.amc,
                    "action": action,
                    "trim_reason": trim_reason,
                    "score": score.to_dict(),
                    "fund_metrics": fund.to_dict(),
                    "portfolio_performance": (
                        asdict(holding)
                        if holding is not None
                        else None
                    ),
                    "allocation": allocation_map.get(
                        fund.scheme_name
                    ),
                    "scenario_analysis": scenario_matrix(
                        fund,
                        regime,
                    ),
                }
            )

        # Evaluate SIP and one-time routes independently for every fund.
        # One-time availability reflects the current deployable-cash plan.
        # SIP availability does not depend on today's deployable cash.
        for item in evaluated:
            fund_record = next(
                fund for fund, _score in scored
                if fund.scheme_name == item["scheme_name"]
            )
            fund_score = next(
                score for fund, score in scored
                if fund.scheme_name == item["scheme_name"]
            )
            item["investment_options"] = investment_option_status(
                fund_record,
                fund_score,
                holdings,
                funds,
                self.settings.policy,
                one_time_allocated=item["scheme_name"] in allocation_map,
                investment_mode=self.settings.investment_mode,
            )
            one_time_recommended = bool(
                item["investment_options"].get("one_time", {}).get("recommended", False)
            )
            sip_recommended = bool(
                item["investment_options"].get("sip", {}).get("recommended", False)
            )

            # Investment routes are independent. Do not let one-time allocation
            # mask a valid SIP recommendation, or vice versa.
            if one_time_recommended and sip_recommended:
                item["quantitative_investment_mode"] = "BOTH"
            elif one_time_recommended:
                item["quantitative_investment_mode"] = "ONE_TIME"
            elif sip_recommended:
                item["quantitative_investment_mode"] = "SIP"
            else:
                item["quantitative_investment_mode"] = "NEITHER"

            route_holding = holding_for_fund(fund_record, holdings)
            route_is_holding = route_holding is not None
            route_is_unresolved = route_is_holding and not fund_record.scheme_code

            if not route_is_holding and (one_time_recommended or sip_recommended):
                item["action"] = "BUY"
            elif (
                route_is_holding
                and not route_is_unresolved
                and sip_recommended
                and item["action"] == "WAIT"
            ):
                item["action"] = "ACCUMULATE"

        ranked_for_review = rank_candidates(
            evaluated,
            self.settings.ranking_limit,
        )

        eligible_ranked = rank_candidates(
            evaluated,
            self.settings.ranking_limit,
            eligible_only=True,
        )

        diversified_for_review = diversify_shortlist(
            eligible_ranked,
            min(self.settings.ranking_limit, 30),
        )

        llm_candidates = retrieve_relevant_funds(
            diversified_for_review,
            {
                "settings": {
                    "horizon": asdict(self.settings.horizon),
                    "investor_age": self.settings.horizon.age,
                    "allocation_mode": self.settings.allocation_mode,
                },
            },
            self.settings.vector_limit,
            self.settings.vector_store_dir,
        )

        relevant_news = retrieve_relevant_news(
            news,
            llm_candidates,
            self.settings.news_vector_limit,
            self.settings.vector_store_dir,
        )

        return {
            "engine_version": "2.6.0",
            "settings": {
                "horizon": asdict(self.settings.horizon),
                "investment_amount": float(
                    os.getenv(
                        "MF_INVESTMENT_AMOUNT",
                        "100000",
                    )
                ),
                "investor_age": self.settings.horizon.age,
                "allocation_mode": self.settings.allocation_mode,
                "investment_mode": self.settings.investment_mode,
                "sip_monthly_amount": self.settings.sip_monthly_amount,
                "policy": asdict(self.settings.policy),
            },
            "market": {
                "macro": macro.to_dict(),
                "regime": regime.to_dict(),
                "news_count": len(news),
                "raw_news_count": len(raw_news),
                "news_events": news[:40],
                "relevant_news": relevant_news,
            },
            "portfolio": {
                "funds_info": funds_info,
                "holdings": {
                    key: asdict(value)
                    for key, value in holdings.items()
                },
                "existing_exposure": existing_exposure(
                    holdings,
                    funds,
                ),
            },
            "evaluated_funds": evaluated,
            "local_ranking": {
                "input_funds": len(evaluated),
                "ranked_limit": self.settings.ranking_limit,
                "vector_limit": self.settings.vector_limit,
                "news_vector_limit": self.settings.news_vector_limit,
                "vector_store_dir": self.settings.vector_store_dir,
                "ranked_funds": ranked_for_review,
                "eligible_ranked_funds": eligible_ranked,
                "diversified_funds": diversified_for_review,
                "llm_candidates": llm_candidates,
                "data_confidence": {
                    "high_75_plus": sum(
                        1
                        for x in evaluated
                        if x["score"].get("data_confidence", 0) >= 75
                    ),
                    "medium_50_74": sum(
                        1
                        for x in evaluated
                        if 50
                        <= x["score"].get("data_confidence", 0)
                        < 75
                    ),
                    "low_below_50": sum(
                        1
                        for x in evaluated
                        if x["score"].get("data_confidence", 0) < 50
                    ),
                    "allocation_eligible": sum(
                        1
                        for x in evaluated
                        if x["score"].get("allocation_eligible", False)
                    ),
                    "one_time_available": sum(
                        1
                        for x in evaluated
                        if x.get("investment_options", {}).get("one_time", {}).get("eligible", False)
                    ),
                    "sip_available": sum(
                        1
                        for x in evaluated
                        if x.get("investment_options", {}).get("sip", {}).get("eligible", False)
                    ),
                    "insufficient_data": sum(
                        1
                        for x in evaluated
                        if x["score"].get("evidence_status")
                        == "INSUFFICIENT_DATA"
                    ),
                },
            },
            "allocation_plan": allocations,
            "sip_recommendations": sip_recommendations,
        }
