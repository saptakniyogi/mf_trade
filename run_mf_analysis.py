from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from mf_agent.config import load_settings
from mf_agent.engine import ResearchEngine
from mf_agent.llm import OpenRouterReviewer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mf_agent")


def merge_reviews(result: dict, reviews: list[dict]) -> None:
    by_name = {r.get("scheme_name"): r for r in reviews}
    for fund in result["evaluated_funds"]:
        review = by_name.get(fund["scheme_name"])
        fund["llm_review"] = review
        if review:
            proposed = fund["action"]
            llm_action = review.get("llm_action", "WAIT")
            if review.get("contradictions") and llm_action == "BUY":
                llm_action = "WAIT"
            fund["final_action"] = llm_action if llm_action else proposed
        else:
            fund["final_action"] = fund["action"]


def main():
    settings = load_settings()
    engine_result = ResearchEngine(settings).build()
    reviews = []
    tokens = 0

    if os.getenv("ENABLE_LLM_REVIEW", "true").lower() in {"1", "true", "yes"}:
        try:
            reviews, tokens = OpenRouterReviewer(settings).review(engine_result)
            merge_reviews(engine_result, reviews)
        except Exception as exc:
            logger.warning("LLM review skipped: %s", exc)
            for fund in engine_result["evaluated_funds"]:
                fund["llm_review"] = None
                fund["final_action"] = fund["action"]
    else:
        for fund in engine_result["evaluated_funds"]:
            fund["llm_review"] = None
            fund["final_action"] = fund["action"]

    output = {

        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provider": "OpenRouter qualitative review" if reviews else "quantitative engine only",
            "model": settings.openrouter_model if reviews else None,
            "llm_review_count": len(reviews),
            "llm_input_tokens": tokens,
        },
        **engine_result,
        "top_candidates": sorted(
            engine_result["evaluated_funds"],
            key=lambda x: x["score"].get("ranking_score", x["score"]["overall"]),
            reverse=True,
        )[:10],
    }
    with open(settings.output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    logger.info("Analysis written to %s", settings.output_path)


if __name__ == "__main__":
    main()
