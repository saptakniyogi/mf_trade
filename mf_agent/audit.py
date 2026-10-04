#!/usr/bin/env python3
"""
Deterministic audit for MF Trade engine output.

Usage:
    python -m mf_agent.audit top_mutual_funds_analysis.json
    python -m mf_agent.audit top_mutual_funds_analysis.json --json
    python -m mf_agent.audit top_mutual_funds_analysis.json --strict

The audit is intentionally read-only. It does not call APIs, mutate the
analysis JSON, invoke the LLM, or modify GitHub.

Checks:
1. Data integrity
2. Evidence gate
3. Score consistency
4. Ranking consistency
5. Independent SIP / one-time routing
6. Existing-portfolio safety
7. Allocation mathematics and policy limits
8. SIP allocation contract and budget mathematics

This file audits the JSON contract emitted by engine.py. It does not attempt
to reproduce the proprietary/implementation-specific scoring formula. Instead,
it verifies the score fields are internally consistent and that downstream
decisions respect the score/evidence contract.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ENGINE_VERSION = "2.6.2"
RANKING_EVIDENCE_CAP = 49.99
EPSILON = 1e-6
PERCENT_EPSILON = 1e-4

VALID_ACTIONS = {
    "BUY",
    "WAIT",
    "HOLD",
    "TRIM",
    "SELL",
    "ACCUMULATE",
}

VALID_INVESTMENT_MODES = {"ONE_TIME", "SIP", "BOTH", "NEITHER"}
VALID_EVIDENCE_STATUS = {"ELIGIBLE", "INSUFFICIENT_DATA"}


@dataclass
class CheckResult:
    name: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


@dataclass
class AuditResult:
    input_file: str
    engine_version: str | None
    funds_analysed: int
    checks: list[CheckResult]

    @property
    def failure_count(self) -> int:
        return sum(len(check.failures) for check in self.checks)

    @property
    def warning_count(self) -> int:
        return sum(len(check.warnings) for check in self.checks)

    @property
    def passed(self) -> bool:
        return self.failure_count == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "PASS" if self.passed else "FAIL",
            "input_file": self.input_file,
            "engine_version": self.engine_version,
            "funds_analysed": self.funds_analysed,
            "failure_count": self.failure_count,
            "warning_count": self.warning_count,
            "checks": [
                {
                    "name": check.name,
                    "status": "PASS" if check.passed else "FAIL",
                    "failures": check.failures,
                    "warnings": check.warnings,
                }
                for check in self.checks
            ],
        }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _close(a: Any, b: Any, tolerance: float = EPSILON) -> bool:
    return (
        _finite_number(a)
        and _finite_number(b)
        and math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)
    )


def _pct_close(a: Any, b: Any) -> bool:
    return _close(a, b, PERCENT_EPSILON)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fund_name(item: dict[str, Any]) -> str:
    return str(item.get("scheme_name") or "<unnamed fund>")


def _holding_names(data: dict[str, Any]) -> set[str]:
    holdings = data.get("portfolio", {}).get("holdings", {})
    return set(holdings) if isinstance(holdings, dict) else set()


def _is_holding(item: dict[str, Any], holding_names: set[str]) -> bool:
    return _fund_name(item) in holding_names or item.get("portfolio_performance") is not None


def _is_unresolved_holding(item: dict[str, Any], holding_names: set[str]) -> bool:
    if not _is_holding(item, holding_names):
        return False
    metrics = item.get("fund_metrics") or {}
    scheme_code = str(metrics.get("scheme_code") or "").strip()
    return not scheme_code


def _option(item: dict[str, Any], route: str) -> dict[str, Any]:
    options = item.get("investment_options") or {}
    option = options.get(route) or {}
    return option if isinstance(option, dict) else {}


def _recommended(item: dict[str, Any], route: str) -> bool:
    return bool(_option(item, route).get("recommended", False))


def _eligible(item: dict[str, Any], route: str) -> bool:
    return bool(_option(item, route).get("eligible", False))


def _status(item: dict[str, Any], route: str) -> str:
    return str(_option(item, route).get("status") or "").upper()


def _normalize_scheme_identity(name: Any) -> str:
    """Normalize portfolio/AMFI names for plan-insensitive identity checks."""
    import re

    value = str(name or "").upper()
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    removable = {
        "DIRECT", "REGULAR", "PLAN", "GROWTH", "IDCW", "DIVIDEND",
        "PAYOUT", "REINVESTMENT", "REINVEST", "BONUS", "OPTION",
    }
    return " ".join(
        token for token in value.split() if token not in removable
    ).strip()


def _check_portfolio_identity(data: dict[str, Any]) -> CheckResult:
    """Verify holdings are reconciled to canonical evaluated fund identities."""
    result = CheckResult("Portfolio identity")
    portfolio = data.get("portfolio") or {}
    holdings = portfolio.get("holdings") or {}
    funds = data.get("evaluated_funds") or []

    if not isinstance(holdings, dict):
        result.fail("portfolio.holdings must be an object.")
        return result

    canonical = {}
    for item in funds:
        if not isinstance(item, dict):
            continue
        name = _fund_name(item)
        identity = _normalize_scheme_identity(name)
        if identity:
            canonical.setdefault(identity, []).append(item)

    for holding_name in holdings:
        identity = _normalize_scheme_identity(holding_name)
        matches = canonical.get(identity, [])
        if not matches:
            result.fail(
                f"Holding {holding_name!r} does not resolve to an evaluated canonical fund."
            )
            continue
        if len(matches) > 1:
            result.warn(
                f"Holding {holding_name!r} matches multiple evaluated funds: "
                + ", ".join(_fund_name(item) for item in matches)
            )

    exposure = portfolio.get("existing_exposure") or {}
    category_weights = exposure.get("category_weights") or {}
    amc_weights = exposure.get("amc_weights") or {}
    total_value = exposure.get("total_value")

    if holdings and _finite_number(total_value) and float(total_value) > 0:
        category_total = sum(
            float(value) for value in category_weights.values()
            if _finite_number(value)
        )
        amc_total = sum(
            float(value) for value in amc_weights.values()
            if _finite_number(value)
        )
        if not math.isclose(category_total, 100.0, abs_tol=0.05):
            result.fail(
                f"Existing category exposure totals {category_total:.2f}%, expected ~100%."
            )
        if not math.isclose(amc_total, 100.0, abs_tol=0.05):
            result.fail(
                f"Existing AMC exposure totals {amc_total:.2f}%, expected ~100%."
            )

    return result


def _check_data_integrity(data: dict[str, Any]) -> CheckResult:
    result = CheckResult("Data integrity")

    if not isinstance(data, dict):
        result.fail("Root JSON value must be an object.")
        return result

    if data.get("engine_version") != ENGINE_VERSION:
        result.warn(
            f"Expected engine version {ENGINE_VERSION}, "
            f"found {data.get('engine_version')!r}."
        )

    settings = data.get("settings")
    if not isinstance(settings, dict):
        result.fail("Missing or invalid settings object.")
    else:
        for key in ("investment_amount", "investment_mode", "policy"):
            if key not in settings:
                result.fail(f"Missing settings.{key}.")
        if settings.get("investment_mode") not in {"ONE_TIME", "SIP", "BOTH"}:
            result.fail(
                f"Invalid settings.investment_mode={settings.get('investment_mode')!r}."
            )

    funds = data.get("evaluated_funds")
    if not isinstance(funds, list):
        result.fail("Missing or invalid evaluated_funds list.")
        return result

    seen_names: set[str] = set()

    for index, item in enumerate(funds):
        prefix = f"evaluated_funds[{index}]"

        if not isinstance(item, dict):
            result.fail(f"{prefix} is not an object.")
            continue

        name = _fund_name(item)
        if name in seen_names:
            result.fail(f"Duplicate fund name: {name}.")
        seen_names.add(name)

        for key in ("scheme_name", "category", "amc", "action", "score", "fund_metrics"):
            if key not in item:
                result.fail(f"{prefix} missing {key}.")

        action = item.get("action")
        if action not in VALID_ACTIONS:
            result.fail(f"{name}: invalid action {action!r}.")

        score = item.get("score")
        if not isinstance(score, dict):
            result.fail(f"{name}: score is missing or not an object.")
            continue

        metrics = item.get("fund_metrics")
        if not isinstance(metrics, dict):
            result.fail(f"{name}: fund_metrics is missing or not an object.")
        else:
            if not _nonempty_string(str(metrics.get("scheme_code") or "")):
                # Missing scheme_code is allowed for unresolved holdings, but
                # should be visible as a warning rather than a data-integrity
                # failure because the engine intentionally preserves them.
                if item.get("portfolio_performance") is not None:
                    result.warn(
                        f"{name}: holding has no canonical scheme_code; "
                        "downstream position-changing actions must be blocked."
                    )
                else:
                    result.warn(f"{name}: missing scheme_code.")

        overall = score.get("overall")
        ranking_score = score.get("ranking_score")
        confidence = score.get("data_confidence")

        if not _finite_number(overall):
            result.fail(f"{name}: score.overall is not a finite number.")
        elif not 0 <= float(overall) <= 100:
            result.fail(f"{name}: score.overall={overall} outside 0..100.")

        if not _finite_number(ranking_score):
            result.fail(f"{name}: score.ranking_score is not a finite number.")
        elif not 0 <= float(ranking_score) <= 100:
            result.fail(f"{name}: ranking_score={ranking_score} outside 0..100.")

        if _finite_number(confidence) and not 0 <= float(confidence) <= 100:
            result.fail(f"{name}: data_confidence={confidence} outside 0..100.")

        evidence = score.get("evidence_status")
        if evidence not in VALID_EVIDENCE_STATUS:
            result.fail(f"{name}: invalid evidence_status={evidence!r}.")

        if not isinstance(score.get("allocation_eligible"), bool):
            result.fail(f"{name}: allocation_eligible must be boolean.")

    return result


def _check_evidence_gate(
    data: dict[str, Any],
    holding_names: set[str],
) -> CheckResult:
    result = CheckResult("Evidence gate")
    funds = data.get("evaluated_funds") or []

    for item in funds:
        if not isinstance(item, dict):
            continue

        name = _fund_name(item)
        score = item.get("score") or {}
        evidence = score.get("evidence_status")
        allocation_eligible = bool(score.get("allocation_eligible", False))
        allocation = item.get("allocation")
        action = item.get("action")

        if evidence == "INSUFFICIENT_DATA":
            if allocation_eligible:
                result.fail(
                    f"{name}: INSUFFICIENT_DATA fund is allocation_eligible=true."
                )
            if allocation is not None:
                result.fail(
                    f"{name}: INSUFFICIENT_DATA fund has an allocation."
                )
            if action in {"BUY", "ACCUMULATE"}:
                result.fail(
                    f"{name}: INSUFFICIENT_DATA fund has position-changing action "
                    f"{action}."
                )

        if evidence == "ELIGIBLE" and not allocation_eligible:
            # This is not necessarily a defect because the engine can have an
            # evidence-qualified fund blocked by portfolio/allocation policy.
            result.warn(
                f"{name}: evidence is ELIGIBLE but allocation_eligible=false."
            )

        if allocation is not None:
            if evidence != "ELIGIBLE":
                result.fail(
                    f"{name}: allocation exists while evidence_status={evidence!r}."
                )
            if not allocation_eligible:
                result.fail(
                    f"{name}: allocation exists while allocation_eligible=false."
                )

        if action in {"BUY", "ACCUMULATE"}:
            if evidence != "ELIGIBLE":
                result.fail(
                    f"{name}: {action} requires ELIGIBLE evidence, "
                    f"found {evidence!r}."
                )
            if not allocation_eligible and action == "BUY":
                result.fail(
                    f"{name}: BUY requires allocation_eligible=true."
                )

    return result


def _check_score_consistency(data: dict[str, Any]) -> CheckResult:
    result = CheckResult("Score consistency")
    funds = data.get("evaluated_funds") or []
    engine_version = str(data.get("engine_version") or "")
    expects_explicit_ranking_policy = engine_version == ENGINE_VERSION

    component_fields = (
        "fund_quality",
        "risk_adjusted_return",
        "current_attractiveness",
        "portfolio_fit",
        "valuation",
        "macro_resilience",
    )

    for item in funds:
        if not isinstance(item, dict):
            continue

        name = _fund_name(item)
        score = item.get("score") or {}
        overall = score.get("overall")
        ranking_score = score.get("ranking_score")
        evidence_status = score.get("evidence_status")
        allocation_eligible = bool(score.get("allocation_eligible", False))
        ranking_cap = score.get("ranking_cap")
        ranking_adjustment = score.get("ranking_adjustment")

        if _finite_number(overall) and _finite_number(ranking_score):
            if allocation_eligible and evidence_status == "ELIGIBLE":
                if not _pct_close(overall, ranking_score):
                    result.fail(
                        f"{name}: ranking_score={ranking_score} does not match "
                        f"overall={overall} for an evidence-qualified fund."
                    )
                if ranking_cap is not None:
                    result.fail(
                        f"{name}: eligible fund must not have ranking_cap={ranking_cap!r}."
                    )
                if ranking_adjustment is not None and not _pct_close(
                    ranking_adjustment, 0.0
                ):
                    result.fail(
                        f"{name}: eligible fund has ranking_adjustment="
                        f"{ranking_adjustment}, expected 0.0."
                    )
            else:
                expected_cap = RANKING_EVIDENCE_CAP
                expected_ranking = min(float(overall), expected_cap)
                expected_adjustment = expected_ranking - float(overall)

                if expects_explicit_ranking_policy:
                    if not _pct_close(ranking_cap, expected_cap):
                        result.fail(
                            f"{name}: ranking_cap={ranking_cap!r} does not match "
                            f"the evidence cap {expected_cap:.2f}."
                        )
                    if not _pct_close(ranking_score, expected_ranking):
                        result.fail(
                            f"{name}: ranking_score={ranking_score} does not match "
                            f"min(overall={overall}, ranking_cap={expected_cap:.2f})="
                            f"{expected_ranking:.2f}."
                        )
                    if not _pct_close(ranking_adjustment, expected_adjustment):
                        result.fail(
                            f"{name}: ranking_adjustment={ranking_adjustment} does not "
                            f"match ranking_score - overall = {expected_adjustment:.2f}."
                        )
                elif not _pct_close(overall, ranking_score):
                    # 2.5.0 output did not expose the ranking policy fields, so
                    # retain compatibility with historical analysis files.
                    result.warn(
                        f"{name}: legacy engine output has ranking_score={ranking_score} "
                        f"below overall={overall}; rerun with engine {ENGINE_VERSION} "
                        "to expose and audit the evidence-ranking cap explicitly."
                    )

        if expects_explicit_ranking_policy:
            if not _finite_number(ranking_adjustment):
                result.fail(
                    f"{name}: score.ranking_adjustment must be a finite number."
                )
            if evidence_status != "ELIGIBLE" or not allocation_eligible:
                if not _finite_number(ranking_cap):
                    result.fail(
                        f"{name}: ineligible fund must expose numeric ranking_cap."
                    )

        for field_name in component_fields:
            value = score.get(field_name)
            if value is not None and (
                not _finite_number(value) or not 0 <= float(value) <= 100
            ):
                result.fail(
                    f"{name}: score.{field_name}={value!r} is not a valid "
                    "0..100 score."
                )

        blockers = score.get("evidence_blockers")
        if blockers is not None and not isinstance(blockers, list):
            result.fail(f"{name}: evidence_blockers must be a list.")

        warnings = score.get("data_warnings")
        if warnings is not None and not isinstance(warnings, list):
            result.fail(f"{name}: data_warnings must be a list.")

        if score.get("evidence_status") == "INSUFFICIENT_DATA" and not blockers:
            result.warn(
                f"{name}: INSUFFICIENT_DATA has no evidence_blockers."
            )

    return result


def _check_ranking(data: dict[str, Any]) -> CheckResult:
    result = CheckResult("Ranking consistency")
    funds = [
        item for item in (data.get("evaluated_funds") or [])
        if isinstance(item, dict)
    ]

    expected = sorted(
        funds,
        key=lambda item: (
            float((item.get("score") or {}).get("ranking_score", float("-inf"))),
            float((item.get("score") or {}).get("data_confidence", float("-inf"))),
            float((item.get("score") or {}).get("fund_quality", float("-inf"))),
        ),
        reverse=True,
    )

    actual_names = [_fund_name(item) for item in funds]
    expected_names = [_fund_name(item) for item in expected]

    if actual_names != expected_names:
        result.fail(
            "evaluated_funds is not ordered by ranking_score, "
            "data_confidence, fund_quality descending."
        )

    local_ranking = data.get("local_ranking") or {}
    ranked = local_ranking.get("ranked_funds")
    if isinstance(ranked, list) and ranked:
        ranked_names = [
            str(item.get("scheme_name"))
            for item in ranked
            if isinstance(item, dict)
        ]
        expected_ranked = expected[: len(ranked_names)]
        expected_ranked_names = [_fund_name(item) for item in expected_ranked]

        if ranked_names != expected_ranked_names:
            result.fail(
                "local_ranking.ranked_funds does not match the engine's "
                "evaluated_funds ordering."
            )

    eligible_ranked = local_ranking.get("eligible_ranked_funds")
    if isinstance(eligible_ranked, list):
        expected_eligible = [
            item
            for item in expected
            if bool((item.get("score") or {}).get("allocation_eligible"))
            and (item.get("score") or {}).get("evidence_status") == "ELIGIBLE"
        ]
        expected_eligible = expected_eligible[: len(eligible_ranked)]
        actual_eligible_names = [
            str(item.get("scheme_name"))
            for item in eligible_ranked
            if isinstance(item, dict)
        ]
        expected_eligible_names = [_fund_name(item) for item in expected_eligible]

        if actual_eligible_names != expected_eligible_names:
            result.fail(
                "local_ranking.eligible_ranked_funds is inconsistent with "
                "the evidence-qualified evaluated funds."
            )

    return result


def _check_investment_modes(data: dict[str, Any]) -> CheckResult:
    result = CheckResult("Investment-mode routing")
    funds = data.get("evaluated_funds") or []

    for item in funds:
        if not isinstance(item, dict):
            continue

        name = _fund_name(item)
        score = item.get("score") or {}
        one_time = _option(item, "one_time")
        sip = _option(item, "sip")

        if not one_time:
            result.warn(f"{name}: missing investment_options.one_time.")
        if not sip:
            result.warn(f"{name}: missing investment_options.sip.")

        one_time_rec = _recommended(item, "one_time")
        sip_rec = _recommended(item, "sip")
        expected_mode = (
            "BOTH"
            if one_time_rec and sip_rec
            else "ONE_TIME"
            if one_time_rec
            else "SIP"
            if sip_rec
            else "NEITHER"
        )
        actual_mode = item.get("quantitative_investment_mode")

        if actual_mode not in VALID_INVESTMENT_MODES:
            result.fail(
                f"{name}: invalid quantitative_investment_mode={actual_mode!r}."
            )
        elif actual_mode != expected_mode:
            result.fail(
                f"{name}: quantitative_investment_mode={actual_mode} but "
                f"one_time.recommended={one_time_rec}, "
                f"sip.recommended={sip_rec}; expected {expected_mode}."
            )

        # Recommended routes must be eligible routes.
        for route, recommended in (("one_time", one_time_rec), ("sip", sip_rec)):
            if recommended and not _eligible(item, route):
                result.fail(
                    f"{name}: {route}.recommended=true while "
                    f"{route}.eligible=false."
                )

        # One-time allocation is only valid when the engine says the fund is
        # evidence-qualified. This also protects against accidentally using a
        # route status as a proxy for evidence.
        if one_time_rec:
            if score.get("evidence_status") != "ELIGIBLE":
                result.fail(
                    f"{name}: one-time recommendation without ELIGIBLE evidence."
                )
            if not bool(score.get("allocation_eligible")):
                result.fail(
                    f"{name}: one-time recommendation while allocation_eligible=false."
                )

        # SIP must be independent from one-time availability. A SIP
        # recommendation must not require a current allocation.
        if sip_rec and _option(item, "sip").get("eligible") is False:
            result.fail(
                f"{name}: SIP recommended despite SIP eligibility=false."
            )

        # A fund explicitly marked NEITHER must not have either route
        # recommended.
        if actual_mode == "NEITHER" and (one_time_rec or sip_rec):
            result.fail(
                f"{name}: mode NEITHER conflicts with a recommended route."
            )

    return result


def _check_portfolio_safety(
    data: dict[str, Any],
    holding_names: set[str],
) -> CheckResult:
    result = CheckResult("Portfolio safety")
    funds = data.get("evaluated_funds") or []

    for item in funds:
        if not isinstance(item, dict):
            continue

        name = _fund_name(item)
        holding = _is_holding(item, holding_names)
        unresolved = _is_unresolved_holding(item, holding_names)
        action = item.get("action")

        if unresolved and action in {"BUY", "ACCUMULATE", "TRIM", "SELL"}:
            result.fail(
                f"{name}: unresolved holding has position-changing action "
                f"{action}."
            )

        if unresolved and item.get("allocation") is not None:
            result.fail(
                f"{name}: unresolved holding has an allocation."
            )

        if holding and action == "BUY":
            # A current holding can legitimately be recommended for additional
            # investment only if SIP/one-time route says so. If no route is
            # recommended, BUY is suspicious.
            if not (
                _recommended(item, "one_time")
                or _recommended(item, "sip")
            ):
                result.fail(
                    f"{name}: existing holding is BUY without a recommended "
                    "investment route."
                )

        if action == "TRIM":
            score = item.get("score") or {}
            if score.get("evidence_status") != "ELIGIBLE":
                result.fail(
                    f"{name}: TRIM requires ELIGIBLE evidence."
                )
            metrics = item.get("fund_metrics") or {}
            if not str(metrics.get("scheme_code") or "").strip():
                result.fail(
                    f"{name}: TRIM requires a canonical scheme_code."
                )

        if action in {"SELL", "TRIM"} and not holding:
            result.fail(
                f"{name}: {action} issued for a fund that is not an existing holding."
            )

    return result


def _check_allocations(data: dict[str, Any]) -> CheckResult:
    result = CheckResult("Allocation mathematics")
    settings = data.get("settings") or {}
    policy = settings.get("policy") or {}
    portfolio = data.get("portfolio") or {}
    funds_info = portfolio.get("funds_info") or {}
    allocations = data.get("allocation_plan") or []

    if not isinstance(allocations, list):
        result.fail("allocation_plan must be a list.")
        return result

    deployable_cash = funds_info.get("deployable_cash")
    investment_amount = settings.get("investment_amount")

    if not _finite_number(deployable_cash):
        result.fail("portfolio.funds_info.deployable_cash is missing/invalid.")
        return result

    if not _finite_number(investment_amount):
        result.fail("settings.investment_amount is missing/invalid.")

    deployable_cash = float(deployable_cash)

    max_single_pct = float(policy.get("max_single_fund_pct", 100.0))
    min_cash_pct = float(policy.get("min_cash_pct", 0.0))

    total_pct = 0.0
    total_amount = 0.0
    seen: set[str] = set()

    for index, allocation in enumerate(allocations):
        prefix = f"allocation_plan[{index}]"

        if not isinstance(allocation, dict):
            result.fail(f"{prefix} is not an object.")
            continue

        name = str(allocation.get("scheme_name") or "<unnamed allocation>")
        if name in seen:
            result.fail(f"Duplicate allocation for {name}.")
        seen.add(name)

        pct = allocation.get("allocation_pct_of_new_cash")
        amount = allocation.get("capital_required")

        if not _finite_number(pct):
            result.fail(f"{name}: allocation_pct is invalid.")
            continue
        if not _finite_number(amount):
            result.fail(f"{name}: amount_inr is invalid.")
            continue

        pct = float(pct)
        amount = float(amount)

        if pct < -EPSILON:
            result.fail(f"{name}: negative allocation_pct={pct}.")
        if amount < -EPSILON:
            result.fail(f"{name}: negative amount_inr={amount}.")
        if pct > max_single_pct + PERCENT_EPSILON:
            result.fail(
                f"{name}: allocation_pct={pct:.4f}% exceeds "
                f"max_single_fund_pct={max_single_pct:.4f}%."
            )

        expected_amount = deployable_cash * pct / 100.0
        if not _close(amount, expected_amount, tolerance=0.01):
            result.fail(
                f"{name}: amount_inr={amount:.2f} does not match "
                f"deployable_cash * allocation_pct "
                f"({expected_amount:.2f})."
            )

        if allocation.get("action") not in {"BUY", "ACCUMULATE"}:
            result.fail(
                f"{name}: allocated fund has action={allocation.get('action')!r}, "
                "expected BUY or ACCUMULATE."
            )

        total_pct += pct
        total_amount += amount

    if total_pct > 100.0 + PERCENT_EPSILON:
        result.fail(
            f"Total allocation_pct={total_pct:.4f}% exceeds 100%."
        )

    if total_amount > deployable_cash + 0.01:
        result.fail(
            f"Total allocation amount ₹{total_amount:.2f} exceeds "
            f"deployable cash ₹{deployable_cash:.2f}."
        )

    cash_pct = 100.0 - total_pct
    if cash_pct + PERCENT_EPSILON < min_cash_pct:
        result.fail(
            f"Residual cash={cash_pct:.4f}% is below "
            f"min_cash_pct={min_cash_pct:.4f}%."
        )

    # If an allocation exists, the corresponding evaluated fund must also
    # carry the same allocation and be evidence-qualified.
    evaluated = {
        _fund_name(item): item
        for item in (data.get("evaluated_funds") or [])
        if isinstance(item, dict)
    }

    for allocation in allocations:
        if not isinstance(allocation, dict):
            continue
        name = str(allocation.get("scheme_name") or "")
        item = evaluated.get(name)
        if item is None:
            result.fail(f"Allocation exists for unevaluated fund {name}.")
            continue

        score = item.get("score") or {}
        if score.get("evidence_status") != "ELIGIBLE":
            result.fail(
                f"{name}: allocation exists but evidence_status="
                f"{score.get('evidence_status')!r}."
            )
        if not bool(score.get("allocation_eligible")):
            result.fail(
                f"{name}: allocation exists but allocation_eligible=false."
            )

        item_allocation = item.get("allocation")
        if not isinstance(item_allocation, dict):
            result.fail(
                f"{name}: allocation_plan entry exists but evaluated_funds "
                "entry has no allocation object."
            )
        else:
            for key in ("allocation_pct_of_new_cash", "capital_required"):
                if not _close(
                    item_allocation.get(key),
                    allocation.get(key),
                    tolerance=0.01 if key == "capital_required" else PERCENT_EPSILON,
                ):
                    result.fail(
                        f"{name}: evaluated_funds allocation.{key} does not "
                        "match allocation_plan."
                    )

    return result


def _check_sip_recommendations(data: dict[str, Any]) -> CheckResult:
    result = CheckResult("SIP allocation plan")
    recommendations = data.get("sip_recommendations")
    settings = data.get("settings") or {}
    monthly_amount = settings.get("sip_monthly_amount", 0.0)

    if not isinstance(recommendations, list):
        result.fail("sip_recommendations must be a list.")
        return result
    if not _finite_number(monthly_amount) or float(monthly_amount) < 0:
        result.fail("settings.sip_monthly_amount must be a non-negative number.")
        return result

    total_amount = 0.0
    total_pct = 0.0
    seen: set[str] = set()

    for index, item in enumerate(recommendations):
        if not isinstance(item, dict):
            result.fail(f"sip_recommendations[{index}] is not an object.")
            continue
        name = str(item.get("scheme_name") or "")
        if not name:
            result.fail(f"sip_recommendations[{index}] has no scheme_name.")
            continue
        if name in seen:
            result.fail(f"Duplicate SIP recommendation for {name}.")
        seen.add(name)

        if item.get("evidence_status") != "ELIGIBLE":
            result.fail(f"{name}: SIP recommendation is not evidence-qualified.")
        if item.get("action") not in {"BUY", "ACCUMULATE"}:
            result.fail(f"{name}: invalid SIP action={item.get('action')!r}.")

        for key in ("ranking_score", "target_sip_pct", "portfolio_capacity_pct"):
            if not _finite_number(item.get(key)):
                result.fail(f"{name}: SIP field {key} is missing/invalid.")

        amount = item.get("monthly_amount", 0.0)
        pct = item.get("allocation_pct_of_monthly_sip", 0.0)
        if not _finite_number(amount) or float(amount) < -EPSILON:
            result.fail(f"{name}: monthly_amount is invalid.")
        if not _finite_number(pct) or float(pct) < -PERCENT_EPSILON:
            result.fail(f"{name}: allocation_pct_of_monthly_sip is invalid.")

        target_pct_value = item.get("target_sip_pct")
        capacity_pct_value = item.get("portfolio_capacity_pct")

        if _finite_number(target_pct_value):
            target_pct = float(target_pct_value)
            if target_pct < -PERCENT_EPSILON or target_pct > 100.0 + PERCENT_EPSILON:
                result.fail(
                    f"{name}: target_sip_pct={target_pct:.4f}% is outside 0..100%."
                )

        if _finite_number(capacity_pct_value):
            capacity_pct = float(capacity_pct_value)
            if capacity_pct < -PERCENT_EPSILON or capacity_pct > 100.0 + PERCENT_EPSILON:
                result.fail(
                    f"{name}: portfolio_capacity_pct={capacity_pct:.4f}% is outside 0..100%."
                )

        max_monthly = item.get("max_monthly_amount")
        if max_monthly is not None:
            if not _finite_number(max_monthly) or float(max_monthly) < -EPSILON:
                result.fail(f"{name}: max_monthly_amount is invalid.")
            elif float(amount) > float(max_monthly) + 0.01:
                result.fail(
                    f"{name}: monthly_amount=₹{float(amount):.2f} exceeds "
                    f"max_monthly_amount=₹{float(max_monthly):.2f}."
                )

        if float(monthly_amount) > 0:
            expected = float(monthly_amount) * float(pct) / 100.0
            if not _close(float(amount), expected, tolerance=0.01):
                result.fail(
                    f"{name}: monthly_amount={amount} does not match "
                    f"monthly SIP × allocation percentage ({expected:.2f})."
                )

        total_amount += max(0.0, float(amount))
        total_pct += max(0.0, float(pct))

    target_total = sum(
        max(0.0, float(item.get("target_sip_pct", 0.0)))
        for item in recommendations
        if isinstance(item, dict)
    )
    if target_total > 100.0 + PERCENT_EPSILON:
        result.fail(
            f"SIP target percentages total {target_total:.4f}% > 100%."
        )

    if total_pct > 100.0 + PERCENT_EPSILON:
        result.fail(f"SIP allocation percentages total {total_pct:.4f}% > 100%.")
    if float(monthly_amount) > 0 and total_amount > float(monthly_amount) + 0.01:
        result.fail(
            f"SIP monthly allocation ₹{total_amount:.2f} exceeds configured "
            f"monthly SIP ₹{float(monthly_amount):.2f}."
        )

    return result


def audit(data: dict[str, Any], input_file: str = "<memory>") -> AuditResult:
    funds = data.get("evaluated_funds")
    fund_count = len(funds) if isinstance(funds, list) else 0
    holding_names = _holding_names(data)

    checks = [
        _check_data_integrity(data),
        _check_evidence_gate(data, holding_names),
        _check_score_consistency(data),
        _check_ranking(data),
        _check_investment_modes(data),
        _check_portfolio_safety(data, holding_names),
        _check_portfolio_identity(data),
        _check_allocations(data),
        _check_sip_recommendations(data),
    ]

    return AuditResult(
        input_file=input_file,
        engine_version=data.get("engine_version"),
        funds_analysed=fund_count,
        checks=checks,
    )


def _print_check(check: CheckResult) -> None:
    status = "PASS" if check.passed else "FAIL"
    print(f"{check.name + ':':34} {status}")
    for failure in check.failures:
        print(f"  [FAIL] {failure}")
    for warning in check.warnings:
        print(f"  [WARN] {warning}")


def print_report(result: AuditResult, strict: bool = False) -> None:
    print("=" * 58)
    print("MF TRADE DETERMINISTIC ENGINE AUDIT")
    print("=" * 58)
    print(f"Input file:                    {result.input_file}")
    print(f"Engine version:                {result.engine_version}")
    print(f"Funds analysed:                {result.funds_analysed}")
    print()

    for check in result.checks:
        _print_check(check)

    print()
    print("-" * 58)
    print(f"Data integrity failures:       {len(result.checks[0].failures)}")
    print(f"Evidence gate failures:        {len(result.checks[1].failures)}")
    print(f"Score consistency failures:    {len(result.checks[2].failures)}")
    print(f"Ranking inconsistencies:       {len(result.checks[3].failures)}")
    print(f"Investment-mode violations:    {len(result.checks[4].failures)}")
    print(f"Portfolio safety violations:  {len(result.checks[5].failures)}")
    print(f"Portfolio identity failures:    {len(result.checks[6].failures)}")
    print(f"Allocation calculation errors: {len(result.checks[7].failures)}")
    print(f"SIP allocation errors:          {len(result.checks[8].failures)}")
    print(f"Warnings:                      {result.warning_count}")
    print()

    if result.passed:
        if strict and result.warning_count:
            print("STATUS: FAIL (strict mode: warnings present)")
        else:
            print("STATUS: PASS")
    else:
        print("STATUS: FAIL")

    print("=" * 58)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Analysis JSON root must be an object.")

    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit MF Trade deterministic engine output."
    )
    parser.add_argument(
        "analysis_file",
        type=Path,
        help="Path to top_mutual_funds_analysis*.json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print machine-readable JSON instead of the text report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures for the process exit code.",
    )
    args = parser.parse_args(argv)

    try:
        data = _load_json(args.analysis_file)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: unable to read analysis file: {exc}", file=sys.stderr)
        return 2

    result = audit(data, str(args.analysis_file))

    if args.json_output:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    else:
        print_report(result, strict=args.strict)

    if not result.passed:
        return 1
    if args.strict and result.warning_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
