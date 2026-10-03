from __future__ import annotations

import json
import logging
import random
import re
import time

import requests

from .config import Settings
from .utils import chunked

logger = logging.getLogger("mf_agent")

SYSTEM_PROMPT = """You are the qualitative investment-committee layer of an Indian mutual-fund research engine.

The quantitative engine has already calculated the fund scores, portfolio fit, macro regime,
hard evidence gates, portfolio constraints, and independent investment-route availability.
Your job is NOT to invent a BUY/SELL thesis from scratch. Your job is to challenge the
quantitative result and decide whether the investment should be made as a one-time investment,
a SIP, both, or neither.

Rules:
1. Treat supplied numerical data as authoritative. Never invent missing metrics.
2. Distinguish hard data from news/event interpretation.
3. Identify contradictions, concentration risks, valuation risks and regime sensitivity.
4. If data is incomplete, explicitly say so.
5. You may downgrade a proposed action when there is a material contradiction, but explain why.
6. Do not use recent performance alone as a reason to buy.
7. Prefer WAIT over forcing a transaction when evidence is insufficient.
8. Evaluate ONE_TIME and SIP independently. The absence of one route does not imply that the
   other route is unsuitable.
9. Choose ONE_TIME only when investment_options.one_time.status is AVAILABLE.
10. Choose SIP only when investment_options.sip.status is AVAILABLE.
11. Choose BOTH only when both routes are AVAILABLE and using both is justified.
12. Choose NEITHER when neither route is appropriate.
13. Do not invent a SIP amount. The engine may provide a configured amount, otherwise leave it
   to the user/application to determine.
14. Every review must contain at least one useful reason or explicit data-gap explanation.
15. Return JSON only.

Output:
{
  "reviews": [
    {
      "scheme_name": "string",
      "llm_action": "BUY|ACCUMULATE|HOLD|TRIM|WAIT",
      "investment_mode": "ONE_TIME|SIP|BOTH|NEITHER",
      "mode_confidence": 0,
      "mode_reasons": ["string"],
      "challenge_level": 0,
      "thesis_supported": true,
      "key_reasons": ["string"],
      "contradictions": ["string"],
      "material_risks": ["string"],
      "data_gaps": ["string"],
      "final_comment": "string"
    }
  ]
}
"""


class OpenRouterReviewer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = __import__("os").getenv("OPENROUTER_API_KEY")

    def review_batch(self, payloads: list[dict], max_retries: int = 5) -> tuple[list[dict], int]:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        body = {
            "model": self.settings.openrouter_model,
            "response_format": {"type": "json_object"},
            "temperature": self.settings.openrouter_temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payloads, indent=2)},
            ],
        }
        if self.settings.openrouter_secondary_models:
            body["models"] = self.settings.openrouter_secondary_models
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/trade-agent",
            "X-Title": "Regime-aware Mutual Fund Research Engine",
        }
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(self.url, headers=headers, json=body, timeout=90)
                if response.status_code == 200:
                    data = response.json()
                    usage = data.get("usage", {})
                    content = data["choices"][0]["message"]["content"].strip()
                    content = re.sub(r"^```json\s*|\s*```$", "", content)
                    parsed = json.loads(content)
                    reviews = parsed.get("reviews", []) if isinstance(parsed, dict) else []
                    return [x for x in reviews if isinstance(x, dict)], int(usage.get("prompt_tokens", 0))
                if response.status_code == 429:
                    wait = float(response.headers.get("retry-after", 0) or 0) or (2 ** attempt) + random.uniform(1, 3)
                    logger.warning("OpenRouter rate limit; retrying in %.1fs", wait)
                    time.sleep(wait)
                    continue
                logger.error("OpenRouter %s: %s", response.status_code, response.text[:1000])
                break
            except Exception as exc:
                logger.warning("OpenRouter attempt %d failed: %s", attempt, exc)
                time.sleep(2)
        return [], 0

    def review(self, engine_result: dict) -> tuple[list[dict], int]:
        evaluated = engine_result.get("evaluated_funds", [])
        allocation = engine_result.get("allocation_plan", [])
        allocation_names = {x.get("scheme_name") for x in allocation}

        ranked = engine_result.get("local_ranking", {}).get("llm_candidates") or evaluated[:15]
        ranked_by_name = {x.get("scheme_name"): x for x in ranked}

        shortlist = []
        for item in allocation:
            candidate = ranked_by_name.get(item.get("scheme_name"))
            if candidate:
                shortlist.append(candidate)
            else:
                candidate = next(
                    (x for x in evaluated if x.get("scheme_name") == item.get("scheme_name")),
                    None,
                )
                if candidate:
                    shortlist.append(candidate)

        seen = {x.get("scheme_name") for x in shortlist}
        for item in ranked:
            if item.get("scheme_name") not in seen:
                shortlist.append(item)
                seen.add(item.get("scheme_name"))
        for item in evaluated:
            if item.get("scheme_name") in allocation_names and item.get("scheme_name") not in seen:
                shortlist.append(item)
                seen.add(item.get("scheme_name"))

        shortlist = shortlist[: max(15, len(allocation))]

        market = engine_result.get("market", {})
        relevant_news = market.get("relevant_news", [])[:8]
        shared_context = {
            "market_regime": market.get("regime", {}),
            "macro": market.get("macro", {}),
            "investment_mode": engine_result.get("settings", {}).get("investment_mode", "BOTH"),
            "sip_monthly_amount": engine_result.get("settings", {}).get("sip_monthly_amount", 0.0),
            "relevant_news": [
                {k: article.get(k) for k in ("title", "source", "published", "event_tags", "transmission_channels", "summary", "vector_similarity")}
                for article in relevant_news
            ],
        }

        candidates = []
        for x in shortlist:
            score = x.get("score", {})
            candidates.append({
                "scheme_name": x["scheme_name"],
                "category": x["category"],
                "amc": x.get("amc"),
                "action": x["action"],
                "score": score,
                "reasons_to_buy": score.get("reasons_to_buy", []),
                "reasons_not_to_buy": score.get("reasons_not_to_buy", []),
                "data_warnings": score.get("data_warnings", []),
                "fund_metrics": {
                    k: x["fund_metrics"].get(k)
                    for k in (
                        "latest_nav", "nav_date", "aum_inr_cr", "cagr_1y_pct",
                        "cagr_3y_pct", "cagr_5y_pct", "volatility_pct",
                        "max_drawdown_pct", "sharpe", "sortino", "benchmark",
                        "sector_weights", "market_cap_weights", "valuation",
                    )
                },
                "portfolio_performance": x["portfolio_performance"],
                "allocation": x["allocation"],
                "investment_options": x.get("investment_options", {}),
                "quantitative_investment_mode": x.get("quantitative_investment_mode"),
                "vector_similarity": x.get("vector_similarity"),
                "local_rank": x.get("local_rank"),
            })

        all_reviews = []
        tokens = 0
        batches = list(chunked(candidates, self.settings.batch_size))
        for index, batch in enumerate(batches):
            reviews, used = self.review_batch([{"context": shared_context, "funds": batch}])
            all_reviews.extend(reviews)
            tokens += used
            if index < len(batches) - 1:
                time.sleep(self.settings.inter_batch_delay_sec)
        return all_reviews, tokens
