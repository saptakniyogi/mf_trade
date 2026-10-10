#!/usr/bin/env python3
"""Compare baseline OpenRouter review cost with Jev-routed review cost.

Uses an existing mf_trade analysis JSON as fixed input, so the deterministic
research engine/data collection is not rerun. It makes real API calls:
  1) baseline: detailed reviewer on the original analysis
  2) routed: Jev screening + detailed reviewer on the filtered analysis

OpenRouter-reported per-request cost is preferred when present. Otherwise the
script estimates cost from the token counts and CLI rates. No investment output
is changed; only a benchmark report is written.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mf_agent.config import load_settings  # noqa: E402
from mf_agent.llm import OpenRouterReviewer  # noqa: E402
from run_mf_analysis import _jev_screening  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", type=Path, help="Existing mf_trade analysis JSON output")
    parser.add_argument("--report", type=Path, default=Path("ai_cost_benchmark.json"), help="Output report path")
    parser.add_argument("--review-input-usd-per-million", type=float, default=0.15)
    parser.add_argument("--review-output-usd-per-million", type=float, default=0.60)
    parser.add_argument("--jev-input-usd-per-million", type=float, default=0.042)
    parser.add_argument("--jev-output-usd-per-million", type=float, default=0.0)
    parser.add_argument("--yes", action="store_true", help="Confirm that baseline and routed API calls may incur charges")
    return parser.parse_args()


def _usage(response: Any) -> dict[str, Any]:
    try:
        data = response.json()
    except Exception:
        return {}
    usage = data.get("usage") if isinstance(data, dict) else None
    return usage if isinstance(usage, dict) else {}


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _record_cost(response: Any, endpoint: str, rates: dict[str, float]) -> dict[str, Any] | None:
    if not (200 <= int(getattr(response, "status_code", 0)) < 300):
        return None
    usage = _usage(response)
    is_jev = "decision" in endpoint.lower() or "systemone" in endpoint.lower()
    input_keys = ("input_tokens", "prompt_tokens") if is_jev else ("prompt_tokens", "input_tokens")
    output_keys = ("output_tokens", "completion_tokens") if is_jev else ("completion_tokens", "output_tokens")
    input_tokens = next((int(usage[k]) for k in input_keys if usage.get(k) is not None), 0)
    output_tokens = next((int(usage[k]) for k in output_keys if usage.get(k) is not None), 0)
    reported_cost = _float(usage.get("cost"))
    if reported_cost is None:
        try:
            body = response.json()
            reported_cost = _float(body.get("cost")) if isinstance(body, dict) else None
        except Exception:
            reported_cost = None
    if reported_cost is not None:
        cost = reported_cost
        cost_source = "OpenRouter reported cost"
    else:
        cost = (
            input_tokens * rates["input"] + output_tokens * rates["output"]
        ) / 1_000_000
        cost_source = "token-rate estimate"
    return {
        "kind": "jev" if is_jev else "reviewer",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "cost_source": cost_source,
        "http_status": int(response.status_code),
    }


def main() -> int:
    args = parse_args()
    if not args.analysis_json.is_file():
        print(f"ERROR: analysis JSON not found: {args.analysis_json}", file=sys.stderr)
        return 2
    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        print("ERROR: set OPENROUTER_API_KEY before running this benchmark.", file=sys.stderr)
        return 2
    if not args.yes:
        print("This benchmark makes paid API calls for a baseline review and a Jev-routed review.")
        print("Review model: baseline + routed; Jev: routed phase only.")
        print("Re-run with --yes to confirm and execute.")
        return 2

    try:
        with args.analysis_json.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read analysis JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(source, dict) or not isinstance(source.get("evaluated_funds"), list):
        print("ERROR: input must be an mf_trade analysis JSON containing evaluated_funds.", file=sys.stderr)
        return 2

    settings = load_settings()
    rates = {
        "reviewer": {"input": args.review_input_usd_per_million, "output": args.review_output_usd_per_million},
        "jev": {"input": args.jev_input_usd_per_million, "output": args.jev_output_usd_per_million},
    }
    if any(value < 0 for group in rates.values() for value in group.values()):
        print("ERROR: token rates cannot be negative.", file=sys.stderr)
        return 2

    calls: dict[str, list[dict[str, Any]]] = {"baseline": [], "routed": []}
    active_phase = {"name": "baseline"}
    original_post = requests.post

    def observed_post(url: str, *pos: Any, **kwargs: Any) -> Any:
        response = original_post(url, *pos, **kwargs)
        endpoint = str(url)
        item = _record_cost(response, endpoint, rates["jev"] if ("decision" in endpoint.lower() or "systemone" in endpoint.lower()) else rates["reviewer"])
        if item is not None:
            item["endpoint"] = endpoint
            calls[active_phase["name"]].append(item)
        return response

    # Both mf_agent clients use the requests module; wrap once to capture actual usage.
    requests.post = observed_post
    original_env = {key: os.environ.get(key) for key in ("JEV_ENABLED", "JEV_ROUTING_MODE")}
    baseline_tokens = routed_tokens = 0
    baseline_reviews = routed_reviews = 0
    routing_meta: dict[str, Any] = {}
    try:
        active_phase["name"] = "baseline"
        os.environ["JEV_ENABLED"] = "false"
        reviewer = OpenRouterReviewer(settings)
        baseline_reviews_list, baseline_tokens = reviewer.review(copy.deepcopy(source))
        baseline_reviews = len(baseline_reviews_list)

        active_phase["name"] = "routed"
        os.environ["JEV_ENABLED"] = "true"
        os.environ["JEV_ROUTING_MODE"] = "route"
        routed_input, routing_meta = _jev_screening(copy.deepcopy(source))
        routed_reviews_list, routed_tokens = reviewer.review(routed_input)
        routed_reviews = len(routed_reviews_list)
    except Exception as exc:
        print(f"ERROR: benchmark failed: {exc}", file=sys.stderr)
        return 1
    finally:
        requests.post = original_post
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "requests": len(items),
            "input_tokens": sum(item["input_tokens"] for item in items),
            "output_tokens": sum(item["output_tokens"] for item in items),
            "cost_usd": round(sum(item["cost_usd"] for item in items), 8),
            "cost_sources": sorted({item["cost_source"] for item in items}),
            "calls": items,
        }

    baseline = summarize(calls["baseline"])
    routed = summarize(calls["routed"])
    baseline_cost = baseline["cost_usd"]
    routed_cost = routed["cost_usd"]
    savings = baseline_cost - routed_cost
    percent = (savings / baseline_cost * 100.0) if baseline_cost > 0 else None
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.analysis_json.resolve()),
        "review_model": settings.openrouter_model,
        "jev_model": os.getenv("JEV_MODEL", "typesafe/jev-1.13"),
        "method": "Same precomputed analysis input; baseline detailed review compared with Jev screening plus routed detailed review.",
        "rates_usd_per_million_tokens_fallback": rates,
        "baseline": {**baseline, "reviews_returned": baseline_reviews, "reviewer_reported_input_tokens": baseline_tokens},
        "jev_routed": {**routed, "reviews_returned": routed_reviews, "reviewer_reported_input_tokens": routed_tokens,
                       "jev_screened_count": routing_meta.get("screened_count", 0),
                       "jev_skipped_review_count": routing_meta.get("skipped_review_count", 0),
                       "jev_error": routing_meta.get("error")},
        "comparison": {
            "baseline_cost_usd": round(baseline_cost, 8),
            "routed_total_cost_usd": round(routed_cost, 8),
            "net_savings_usd": round(savings, 8),
            "savings_percent": round(percent, 2) if percent is not None else None,
            "cost_reduced": savings > 0,
            "interpretation": "Measured using OpenRouter-reported per-request costs where available; otherwise estimated from observed token usage and supplied fallback rates.",
        },
        "warning": "A single run is a sample, not a guaranteed recurring saving. Compare several representative analysis files; model outputs and token counts can vary.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"baseline_cost_usd": report["comparison"]["baseline_cost_usd"],
                      "jev_routed_total_cost_usd": report["comparison"]["routed_total_cost_usd"],
                      "net_savings_usd": report["comparison"]["net_savings_usd"],
                      "savings_percent": report["comparison"]["savings_percent"],
                      "jev_screened_count": routing_meta.get("screened_count", 0),
                      "jev_skipped_review_count": routing_meta.get("skipped_review_count", 0),
                      "report": str(args.report.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
