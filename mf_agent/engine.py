from __future__ import annotations

import logging
import os
from dataclasses import asdict

from .analytics import holdings_overlap
from .config import Settings
from .data import build_fund_universe, load_holdings
from .macro import fetch_macro_snapshot
from .news import classify_event, fetch_news, filter_relevant_news
from .portfolio import allocation_for_candidates, existing_exposure
from .ranking import rank_candidates, diversify_shortlist
from .embeddings import retrieve_relevant_funds, retrieve_relevant_news
from .regime import infer_regime
from .scoring import score_fund
from .scenarios import scenario_matrix

logger = logging.getLogger("mf_agent")


class ResearchEngine:
    def __init__(self, settings: Settings):
        self.settings = settings

    def build(self) -> dict:
        holdings, funds_info = load_holdings(self.settings)
        topics = [x.strip() for x in __import__("os").getenv("TRENDING_TOPICS", "").split(",") if x.strip()]
        macro = fetch_macro_snapshot(topics)
        raw_news = fetch_news(self.settings)
        classified_news = [classify_event(x) for x in raw_news]
        news = filter_relevant_news(classified_news, limit=40)
        regime = infer_regime(macro, news)
        funds = build_fund_universe(self.settings, holdings)

        scored = []
        for fund in funds:
            score = score_fund(fund, holdings, funds, regime)
            scored.append((fund, score))

        candidates = [(f, s) for f, s in scored if s.overall >= 60 and s.data_confidence >= 50]
        allocations = allocation_for_candidates(
            candidates,
            holdings,
            funds,
            funds_info["deployable_cash"],
            self.settings.policy,
            self.settings.allocation_mode,
        )

        allocation_map = {x["scheme_name"]: x for x in allocations}
        evaluated = []
        for fund, score in sorted(scored, key=lambda x: x[1].overall, reverse=True):
            action = "HOLD" if fund.scheme_name in holdings else "WAIT"
            if fund.scheme_name in allocation_map:
                action = allocation_map[fund.scheme_name]["action"]
            elif score.overall < 45 and fund.scheme_name in holdings:
                action = "TRIM"

            evaluated.append({
                "scheme_name": fund.scheme_name,
                "category": fund.category,
                "amc": fund.amc,
                "action": action,
                "score": score.to_dict(),
                "fund_metrics": fund.to_dict(),
                "portfolio_performance": asdict(holdings[fund.scheme_name]) if fund.scheme_name in holdings else None,
                "allocation": allocation_map.get(fund.scheme_name),
                "scenario_analysis": scenario_matrix(fund, regime),
            })

        # Local retrieval layer: numerical ranking first, then persistent on-disk vector
        # retrieval. Only this compact evidence set is intended for the LLM.
        ranked_for_review = rank_candidates(evaluated, self.settings.ranking_limit)
        diversified_for_review = diversify_shortlist(ranked_for_review, min(self.settings.ranking_limit, 30))
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
            news, llm_candidates, self.settings.news_vector_limit, self.settings.vector_store_dir
        )

        return {
            "engine_version": "2.2.0",
            "settings": {
                "horizon": asdict(self.settings.horizon),
                "investment_amount": float(os.getenv("MF_INVESTMENT_AMOUNT", "100000")),
                "investor_age": self.settings.horizon.age,
                "allocation_mode": self.settings.allocation_mode,
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
                "holdings": {k: asdict(v) for k, v in holdings.items()},
                "existing_exposure": existing_exposure(holdings, funds),
            },
            "evaluated_funds": evaluated,
            "local_ranking": {
                "input_funds": len(evaluated),
                "ranked_limit": self.settings.ranking_limit,
                "vector_limit": self.settings.vector_limit,
                "news_vector_limit": self.settings.news_vector_limit,
                "vector_store_dir": self.settings.vector_store_dir,
                "ranked_funds": ranked_for_review,
                "diversified_funds": diversified_for_review,
                "llm_candidates": llm_candidates,
                "data_confidence": {
                    "high_75_plus": sum(1 for x in evaluated if x["score"].get("data_confidence", 0) >= 75),
                    "medium_50_74": sum(1 for x in evaluated if 50 <= x["score"].get("data_confidence", 0) < 75),
                    "low_below_50": sum(1 for x in evaluated if x["score"].get("data_confidence", 0) < 50),
                },
            },
            "allocation_plan": allocations,
        }
