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

SIP_REQUIRED_FIELDS = {
    "action",
    "target_sip_pct",
    "portfolio_capacity_pct",
    "monthly_amount",
    "allocation_pct_of_monthly_sip",
}


def _normalize_sip_output(result: dict) -> None:
    """Enforce the canonical SIP allocation contract before JSON serialization."""
    recommendations = result.get("sip_recommendations") or []
    if not recommendations:
        return

    if all(
        isinstance(row, dict) and SIP_REQUIRED_FIELDS.issubset(row)
        for row in recommendations
    ):
        return

    evaluated = result.get("evaluated_funds") or []
    by_name = {
        str(item.get("scheme_name")): item
        for item in evaluated
        if isinstance(item, dict)
    }

    settings = result.get("settings") or {}
    monthly_amount = float(settings.get("sip_monthly_amount") or 0.0)

    exposure = (
        (result.get("portfolio") or {}).get("existing_exposure") or {}
    )
    total_value = float(exposure.get("total_value") or 0.0)

    eligible_rows = []

    for legacy in recommendations:
        if not isinstance(legacy, dict):
            continue

        name = str(legacy.get("scheme_name") or "")
        evaluated_row = by_name.get(name, {})
        score = evaluated_row.get("score") or {}

        ranking_score = float(
            legacy.get(
                "ranking_score",
                score.get("ranking_score", legacy.get("score", 0.0)),
            )
            or 0.0
        )
        evidence_status = legacy.get(
            "evidence_status",
            score.get("evidence_status"),
        )
        allocation_eligible = legacy.get(
            "allocation_eligible",
            score.get("allocation_eligible", False),
        )

        options = legacy.get("investment_options") or {}
        capacity = options.get("portfolio_capacity_pct")

        if not isinstance(capacity, (int, float)):
            capacity = (
                evaluated_row.get("investment_options", {})
                .get("portfolio_capacity_pct")
            )

        if (
            not name
            or not allocation_eligible
            or evidence_status != "ELIGIBLE"
            or ranking_score < 50.0
            or not isinstance(capacity, (int, float))
            or float(capacity) <= 0.0
        ):
            continue

        capacity_pct = max(0.0, min(100.0, float(capacity)))
        holding = evaluated_row.get("portfolio_performance") is not None

        eligible_rows.append(
            {
                "scheme_name": name,
                "category": legacy.get("category", evaluated_row.get("category")),
                "amc": legacy.get("amc", evaluated_row.get("amc")),
                "action": "ACCUMULATE" if holding else "BUY",
                "score": float(
                    legacy.get("score", score.get("overall", 0.0)) or 0.0
                ),
                "ranking_score": ranking_score,
                "data_confidence": float(
                    legacy.get(
                        "data_confidence",
                        score.get("data_confidence", 0.0),
                    )
                    or 0.0
                ),
                "evidence_status": evidence_status,
                "portfolio_capacity_pct": capacity_pct,
                "strength": max(1.0, ranking_score - 50.0),
                "reason": legacy.get("reason")
                or "Deterministic SIP candidate.",
            }
        )

    if not eligible_rows:
        result["sip_recommendations"] = []
        return

    total_strength = sum(row["strength"] for row in eligible_rows)
    canonical = []

    for row in eligible_rows:
        target_pct = row["strength"] / total_strength * 100.0

        max_monthly = (
            total_value * row["portfolio_capacity_pct"] / 100.0
            if total_value > 0
            else None
        )

        desired = (
            monthly_amount * target_pct / 100.0
            if monthly_amount > 0
            else 0.0
        )

        amount = (
            min(desired, max_monthly)
            if max_monthly is not None
            else desired
        )

        canonical.append(
            {
                "scheme_name": row["scheme_name"],
                "category": row["category"],
                "amc": row["amc"],
                "action": row["action"],
                "score": row["score"],
                "ranking_score": row["ranking_score"],
                "data_confidence": row["data_confidence"],
                "evidence_status": row["evidence_status"],
                "portfolio_capacity_pct": round(
                    row["portfolio_capacity_pct"], 2
                ),
                "target_sip_pct": round(target_pct, 2),
                "monthly_amount": round(amount, 2),
                "max_monthly_amount": (
                    round(max_monthly, 2)
                    if max_monthly is not None
                    else None
                ),
                "allocation_pct_of_monthly_sip": (
                    round(amount / monthly_amount * 100.0, 2)
                    if monthly_amount > 0
                    else 0.0
                ),
                "reason": row["reason"],
                "final_action": row["action"],
                "sip_monthly_amount": round(amount, 2),
            }
        )

    if monthly_amount > 0:
        remaining = monthly_amount

        for row in sorted(
            canonical,
            key=lambda item: item["ranking_score"],
            reverse=True,
        ):
            amount = min(
                max(0.0, float(row["monthly_amount"])),
                max(0.0, remaining),
            )
            row["monthly_amount"] = round(amount, 2)
            row["sip_monthly_amount"] = row["monthly_amount"]
            row["allocation_pct_of_monthly_sip"] = round(
                amount / monthly_amount * 100.0,
                2,
            )
            remaining -= amount

    result["sip_recommendations"] = canonical


def merge_reviews(result: dict, reviews: list[dict]) -> None:
    by_name = {r.get("scheme_name"): r for r in reviews}
    allocation = result.get("allocation_plan", [])

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

        item["reason"] = (
            (
                review.get("final_comment")
                or (review.get("key_reasons") or [None])[0]
            )
            if review
            else None
        ) or (
            (score.get("reasons_to_buy") or [None])[0]
        ) or "Selected by the quantitative allocation engine based on score and portfolio constraints."

        item["data_confidence"] = score.get("data_confidence")
        item["data_warnings"] = score.get("data_warnings", [])
        item["llm_review"] = review
        item["final_action"] = fund.get(
            "final_action",
            fund.get("action"),
        )


def main():
    settings = load_settings()
    engine_result = ResearchEngine(settings).build()

    reviews = []
    tokens = 0

    if os.getenv("ENABLE_LLM_REVIEW", "true").lower() in {
        "1",
        "true",
        "yes",
    }:
        try:
            reviews, tokens = OpenRouterReviewer(settings).review(
                engine_result
            )
            merge_reviews(engine_result, reviews)
        except Exception as exc:
            logger.warning("LLM review skipped: %s", exc)

            for fund in engine_result["evaluated_funds"]:
                fund["llm_review"] = None
                fund["final_action"] = fund["action"]

            for item in engine_result.get("allocation_plan", []):
                fund = next(
                    (
                        x
                        for x in engine_result["evaluated_funds"]
                        if x["scheme_name"] == item["scheme_name"]
                    ),
                    None,
                )
                score = fund.get("score", {}) if fund else {}

                item["reason"] = (
                    (score.get("reasons_to_buy") or [None])[0]
                    or "Selected by the quantitative allocation engine."
                )
                item["data_confidence"] = score.get("data_confidence")
                item["data_warnings"] = score.get("data_warnings", [])
                item["llm_review"] = None
                item["final_action"] = (
                    fund.get("action")
                    if fund
                    else item.get("action")
                )
    else:
        for fund in engine_result["evaluated_funds"]:
            fund["llm_review"] = None
            fund["final_action"] = fund["action"]

        merge_reviews(engine_result, [])

    # Final serialization boundary. Legacy SIP rows cannot reach the JSON.
    _normalize_sip_output(engine_result)

    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provider": (
                "OpenRouter qualitative review"
                if reviews
                else "quantitative engine only"
            ),
            "model": settings.openrouter_model if reviews else None,
            "llm_review_count": len(reviews),
            "llm_input_tokens": tokens,
            "one_time_recommendation_count": len(
                engine_result.get("allocation_plan", [])
            ),
            "sip_recommendation_count": len(
                engine_result.get("sip_recommendations", [])
            ),
        },
        **engine_result,
        "top_candidates": sorted(
            engine_result["evaluated_funds"],
            key=lambda x: x["score"].get(
                "ranking_score",
                x["score"]["overall"],
            ),
            reverse=True,
        )[:10],
    }

    with open(settings.output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    logger.info("Analysis written to %s", settings.output_path)


if __name__ == "__main__":
    main()
