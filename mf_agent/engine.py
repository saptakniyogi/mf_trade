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
