from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger("mf_agent")


def _score_value(score: dict, key: str) -> float:
    try:
        return float(score.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def rank_candidates(
    evaluated: list[dict],
    limit: int = 50,
    *,
    eligible_only: bool = False,
) -> list[dict]:
    """Rank candidates using the evidence-qualified ranking score.

    ``eligible_only`` is useful for an investment shortlist. The default keeps
    the full research ranking so low-evidence funds remain visible as research
    items, but they can never outrank evidence-qualified funds because their
    ranking score is capped by the scoring layer.
    """
    ranked = []

    for item in evaluated:
        score = item.get("score", {})

        if eligible_only and not bool(score.get("allocation_eligible", False)):
            continue

        base = _score_value(score, "ranking_score")
        confidence = _score_value(score, "data_confidence")
        quality = _score_value(score, "fund_quality")

        ranked.append(
            (
                base,
                confidence,
                quality,
                str(item.get("scheme_name", "")).lower(),
                item,
            )
        )

    ranked.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))

    result = []
    for rank, row in enumerate(
        ranked[: max(0, int(limit))],
        start=1,
    ):
        item = dict(row[-1])
        item["local_rank"] = rank
        result.append(item)

    logger.info(
        "Local quantitative ranking: %d funds -> top %d (eligible_only=%s).",
        len(evaluated),
        len(result),
        eligible_only,
    )
    return result


def diversify_shortlist(
    items: list[dict],
    limit: int = 20,
    max_same_category: int = 4,
    max_same_amc: int = 4,
) -> list[dict]:
    """Apply hard category/AMC caps while preserving evidence-aware order."""
    selected = []
    category_counts = defaultdict(int)
    amc_counts = defaultdict(int)

    for item in items:
        category = str(item.get("category", "Unknown"))
        amc = str(item.get("amc", "Unknown"))

        if (
            category_counts[category] >= max_same_category
            or amc_counts[amc] >= max_same_amc
        ):
            continue

        selected.append(item)
        category_counts[category] += 1
        amc_counts[amc] += 1

        if len(selected) >= limit:
            break

    if len(selected) < limit:
        seen = {x.get("scheme_name") for x in selected}

        for item in items:
            if item.get("scheme_name") in seen:
                continue

            selected.append(item)

            if len(selected) >= limit:
                break

    logger.info(
        "Diversity shortlist: %d ranked candidates -> %d candidates.",
        len(items),
        len(selected),
    )
    return selected
