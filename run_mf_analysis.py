from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from mf_agent.config import load_settings
from mf_agent.engine import ResearchEngine
from mf_agent.llm import OpenRouterReviewer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("mf_agent")


def _normalise_llm_action(action: str | None) -> str:
    allowed = {
        "BUY",
        "ACCUMULATE",
        "HOLD",
        "WAIT",
        "TRIM",
    }
    value = str(action or "WAIT").upper().strip()
    return value if value in allowed else "WAIT"


def merge_reviews(result: dict, reviews: list[dict]) -> None:
    by_name = {
        r.get("scheme_name"): r
        for r in reviews
        if r.get("scheme_name")
    }

    allocation = result.get("allocation_plan", [])

    for fund in result["evaluated_funds"]:
        review = by_name.get(fund["scheme_name"])
        fund["llm_review"] = review

        score = fund.get("score", {})
        quantitative_action = fund.get("action", "WAIT")
        llm_action = (
            _normalise_llm_action(review.get("llm_action"))
            if review
            else quantitative_action
        )

        if review and review.get("contradictions") and llm_action == "BUY":
            llm_action = "WAIT"

        evidence_status = score.get(
            "evidence_status",
            "INSUFFICIENT_DATA",
        )
        allocation_eligible = bool(
            score.get("allocation_eligible", False)
        )

        # An LLM cannot override a hard quantitative evidence gate.
        if (
            evidence_status != "ELIGIBLE"
            or not allocation_eligible
        ) and llm_action in {"BUY", "ACCUMULATE"}:
            llm_action = "WAIT"

        fund["final_action"] = llm_action or quantitative_action

    # Reconcile the allocation plan after qualitative review.
    # Capital must never remain allocated to a fund whose final action is WAIT/HOLD/TRIM.
    reconciled_allocation = []

    for item in allocation:
        fund = next(
            (
                x
                for x in result["evaluated_funds"]
                if x["scheme_name"] == item["scheme_name"]
            ),
            None,
        )

        if not fund:
            continue

        score = fund.get("score", {})
        review = by_name.get(item["scheme_name"])
        final_action = fund.get(
            "final_action",
            fund.get("action", "WAIT"),
        )

        item["reason"] = (
            (
                review.get("final_comment")
                or (review.get("key_reasons") or [None])[0]
            )
            if review
            else None
        ) or (
            (score.get("reasons_to_buy") or [None])[0]
        ) or (
            "Selected by the quantitative allocation engine based on "
            "evidence-qualified score and portfolio constraints."
        )

        item["data_confidence"] = score.get("data_confidence")
        item["data_warnings"] = score.get(
            "data_warnings",
            [],
        )
        item["evidence_status"] = score.get(
            "evidence_status"
        )
        item["allocation_eligible"] = score.get(
            "allocation_eligible",
            False,
        )
        item["llm_review"] = review
        item["final_action"] = final_action

        if final_action in {"BUY", "ACCUMULATE"}:
            reconciled_allocation.append(item)
        else:
            logger.info(
                "Removing %s from allocation after final action=%s.",
                item["scheme_name"],
                final_action,
            )

    result["allocation_plan"] = reconciled_allocation


def _fallback_without_llm(result: dict) -> None:
    """Keep quantitative actions internally consistent when LLM is unavailable."""
    for fund in result["evaluated_funds"]:
        fund["llm_review"] = None

        score = fund.get("score", {})
        action = fund.get("action", "WAIT")

        if not score.get("allocation_eligible", False):
            if action in {"BUY", "ACCUMULATE"}:
                action = "WAIT"

        fund["final_action"] = action

    # There is no qualitative veto available, so preserve only
    # evidence-qualified allocations.
    allocation = []

    for item in result.get("allocation_plan", []):
        fund = next(
            (
                x
                for x in result["evaluated_funds"]
                if x["scheme_name"] == item["scheme_name"]
            ),
            None,
        )

        if not fund:
            continue

        score = fund.get("score", {})

        if not score.get("allocation_eligible", False):
            continue

        item["llm_review"] = None
        item["final_action"] = fund.get(
            "final_action",
            fund.get("action", "WAIT"),
        )
        item["evidence_status"] = score.get(
            "evidence_status"
        )
        item["allocation_eligible"] = score.get(
            "allocation_eligible",
            False,
        )
        item["data_confidence"] = score.get(
            "data_confidence"
        )
        item["data_warnings"] = score.get(
            "data_warnings",
            [],
        )
        item["reason"] = (
            (score.get("reasons_to_buy") or [None])[0]
            or "Selected by the evidence-qualified quantitative engine."
        )

        allocation.append(item)

    result["allocation_plan"] = allocation


def main():
    settings = load_settings()
    engine_result = ResearchEngine(settings).build()

    reviews = []
    tokens = 0

    llm_enabled = os.getenv(
        "ENABLE_LLM_REVIEW",
        "true",
    ).lower() in {"1", "true", "yes"}

    if llm_enabled:
        try:
            reviews, tokens = OpenRouterReviewer(
                settings
            ).review(engine_result)

            merge_reviews(
                engine_result,
                reviews,
            )
        except Exception as exc:
            logger.warning(
                "LLM review skipped: %s",
                exc,
            )
            _fallback_without_llm(
                engine_result
            )
    else:
        _fallback_without_llm(
            engine_result
        )

    # Only evidence-qualified funds should appear in the actionable top list.
    top_candidates = [
        fund
        for fund in engine_result["evaluated_funds"]
        if fund.get("score", {}).get(
            "allocation_eligible",
            False,
        )
    ]

    top_candidates.sort(
        key=lambda x: (
            x["score"].get(
                "ranking_score",
                0.0,
            ),
            x["score"].get(
                "data_confidence",
                0.0,
            ),
            x["score"].get(
                "fund_quality",
                0.0,
            ),
        ),
        reverse=True,
    )

    output = {
        "metadata": {
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "provider": (
                "OpenRouter qualitative review"
                if reviews
                else "quantitative engine only"
            ),
            "model": (
                settings.openrouter_model
                if reviews
                else None
            ),
            "llm_review_count": len(reviews),
            "llm_input_tokens": tokens,
        },
        **engine_result,
        "top_candidates": top_candidates[:10],
        "research_candidates": engine_result[
            "local_ranking"
        ]["ranked_funds"][:10],
    }

    with open(
        settings.output_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            output,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Analysis written to %s",
        settings.output_path,
    )


if __name__ == "__main__":
    main()
