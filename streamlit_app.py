from __future__ import annotations

import json
import os
import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / os.getenv("MF_SUGGESTION_OUTPUT_PATH", "top_mutual_funds_analysis.json")
LOG_DIR = ROOT / "logs"
CURRENT_LOG_PATH = LOG_DIR / "mf_agent_latest.log"
LOG_DIR.mkdir(exist_ok=True)

if "engine_logs" not in st.session_state:
    st.session_state.engine_logs = []
if "engine_running" not in st.session_state:
    st.session_state.engine_running = False
if "investor_inputs_applied" not in st.session_state:
    st.session_state.investor_inputs_applied = None

def current_investor_inputs():
    return {
        "amount": float(st.session_state.get("investment_amount", 100000)),
        "age": int(st.session_state.get("investor_age", 38)),
        "horizon": st.session_state.get("horizon", "LONG_TERM"),
        "allocation_mode": st.session_state.get("allocation_mode", "DIVERSIFIED"),
    }

def mark_result_stale():
    current = current_investor_inputs()
    applied = st.session_state.get("investor_inputs_applied")
    if applied is None:
        return False
    return current != applied

st.set_page_config(page_title="Mutual Fund Research Dashboard", page_icon="📊", layout="wide")


def money(x):
    if x is None:
        return "—"
    return f"₹{x:,.0f}"


def pct(x):
    if x is None:
        return "—"
    return f"{x:.1f}%"

def score_fmt(x):
    if x is None:
        return "—"
    return f"{x:.1f}"


def load_result():
    if not OUTPUT_PATH.exists():
        return None
    try:
        return json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Could not read {OUTPUT_PATH.name}: {exc}")
        return None


def apply_env_from_sidebar():
    os.environ["MF_HORIZON"] = st.session_state.horizon
    os.environ["MF_INVESTMENT_AMOUNT"] = str(st.session_state.investment_amount)
    os.environ["INVESTOR_AGE"] = str(st.session_state.investor_age)
    os.environ["ALLOCATION_MODE"] = st.session_state.allocation_mode
    os.environ["ENABLE_LLM_REVIEW"] = "true" if st.session_state.llm_review else "false"
    os.environ["MAX_CATEGORY_EXPOSURE_PCT"] = str(st.session_state.max_category)
    os.environ["MAX_SINGLE_FUND_PCT"] = str(st.session_state.max_fund)
    os.environ["MAX_SINGLE_AMC_PCT"] = str(st.session_state.max_amc)
    os.environ["MAX_MID_CAP_PCT"] = str(st.session_state.max_mid)
    os.environ["MAX_SMALL_CAP_PCT"] = str(st.session_state.max_small)
    os.environ["MAX_THEMATIC_PCT"] = str(st.session_state.max_thematic)
    os.environ["MIN_CASH_PCT"] = str(st.session_state.min_cash)
    os.environ["MAX_GOLD_PCT"] = str(st.session_state.max_gold)
    os.environ["MAX_FUND_OVERLAP_PCT"] = str(st.session_state.max_overlap)


def run_engine():
    apply_env_from_sidebar()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "-u", str(ROOT / "run_mf_analysis.py")]
    log_lines = []
    log_placeholder = st.empty()

    with st.status("Running mutual-fund research engine…", expanded=True) as status:
        st.caption("Live engine log")
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            log_lines.append(line)
            # Keep the UI responsive while showing the most recent 80 lines.
            log_placeholder.code("\n".join(log_lines[-80:]), language="text")
        return_code = proc.wait()

        # Persist the completed run independently of the transient Streamlit status widget.
        st.session_state.engine_logs = list(log_lines)
        try:
            CURRENT_LOG_PATH.write_text("\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")
            history_path = LOG_DIR / f"mf_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            history_path.write_text("\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")
            st.session_state.engine_log_path = str(history_path)
        except OSError as exc:
            st.session_state.engine_log_path = None
            st.warning(f"Could not persist engine log to disk: {exc}")

        if return_code != 0:
            status.update(label="Analysis failed", state="error")
            st.error(f"Research engine exited with code {return_code}. See Diagnostics → Engine log.")
            return None
        status.update(label="Analysis completed", state="complete")

    return load_result()


st.title("📊 Mutual Fund Research Dashboard")
st.caption("Regime-aware quantitative research with constrained portfolio allocation and optional LLM challenge review.")

with st.sidebar:
    st.header("Investor profile")
    st.number_input(
        "Amount to invest (₹)",
        min_value=0.0,
        value=float(os.getenv("MF_INVESTMENT_AMOUNT", 100000)),
        step=10000.0,
        key="investment_amount",
        help="New money available for this analysis. This is separate from existing holdings.",
    )
    st.number_input(
        "Investor age",
        min_value=18,
        max_value=100,
        value=int(os.getenv("INVESTOR_AGE", 38)),
        step=1,
        key="investor_age",
        help="Used as investor-profile input. The current scoring engine records age but does not yet automatically change equity/debt allocation from age alone.",
    )
    st.selectbox("Investment horizon", ["LONG_TERM", "SHORT_TERM"], key="horizon")
    st.selectbox("Allocation mode", ["DIVERSIFIED", "CONCENTRATED"], key="allocation_mode")
    st.toggle("Enable OpenRouter review", value=os.getenv("ENABLE_LLM_REVIEW", "true").lower() in {"1", "true", "yes"}, key="llm_review")

    with st.expander("Portfolio constraints", expanded=False):
        st.number_input("Max category %", 1.0, 100.0, float(os.getenv("MAX_CATEGORY_EXPOSURE_PCT", 35)), key="max_category")
        st.number_input("Max single fund %", 1.0, 100.0, float(os.getenv("MAX_SINGLE_FUND_PCT", 25)), key="max_fund")
        st.number_input("Max single AMC %", 1.0, 100.0, float(os.getenv("MAX_SINGLE_AMC_PCT", 35)), key="max_amc")
        st.number_input("Max mid-cap %", 1.0, 100.0, float(os.getenv("MAX_MID_CAP_PCT", 25)), key="max_mid")
        st.number_input("Max small-cap %", 1.0, 100.0, float(os.getenv("MAX_SMALL_CAP_PCT", 15)), key="max_small")
        st.number_input("Max thematic %", 1.0, 100.0, float(os.getenv("MAX_THEMATIC_PCT", 10)), key="max_thematic")
        st.number_input("Minimum cash %", 0.0, 100.0, float(os.getenv("MIN_CASH_PCT", 5)), key="min_cash")
        st.number_input("Max gold %", 0.0, 100.0, float(os.getenv("MAX_GOLD_PCT", 15)), key="max_gold")
        st.number_input("Max fund overlap %", 0.0, 100.0, float(os.getenv("MAX_FUND_OVERLAP_PCT", 60)), key="max_overlap")

    inputs_changed = mark_result_stale()
    if inputs_changed:
        st.warning("Investor inputs changed. Click **Run analysis** to recalculate the portfolio.")

    if st.button("▶ Run analysis", type="primary", use_container_width=True):
        st.session_state.result = run_engine()
        st.session_state.investor_inputs_applied = current_investor_inputs()
        st.rerun()

    if st.button("↻ Load latest JSON", use_container_width=True):
        st.session_state.result = load_result()
        loaded = st.session_state.result or {}
        h = loaded.get("settings", {}).get("horizon", {})
        st.session_state.investor_inputs_applied = {
            "amount": float(loaded.get("settings", {}).get("investment_amount", st.session_state.investment_amount)),
            "age": int(h.get("age", st.session_state.investor_age)),
            "horizon": loaded.get("settings", {}).get("horizon", {}).get("mode", st.session_state.horizon),
            "allocation_mode": loaded.get("settings", {}).get("allocation_mode", st.session_state.allocation_mode),
        }
        st.rerun()

# Keep the latest completed log visible after the process exits and after Streamlit reruns.
latest_logs = st.session_state.get("engine_logs", [])
if latest_logs:
    with st.expander("📜 Latest engine log", expanded=False):
        st.code("\n".join(latest_logs), language="text")
        log_path = st.session_state.get("engine_log_path")
        if log_path and Path(log_path).exists():
            st.download_button(
                "Download latest engine log",
                data="\n".join(latest_logs) + "\n",
                file_name=Path(log_path).name,
                mime="text/plain",
                key="download_latest_engine_log",
            )

result = st.session_state.get("result") or load_result()

if not result:
    st.info("No analysis result is available yet. Configure the controls and click **Run analysis**.")
    st.code("streamlit run streamlit_app.py", language="bash")
    st.stop()

market = result.get("market", {})
macro = market.get("macro", {})
regime = market.get("regime", {})
portfolio = result.get("portfolio", {})
funds = result.get("evaluated_funds", [])
allocation = result.get("allocation_plan", [])
metadata = result.get("metadata", {})

st.caption(f"Generated: {metadata.get('generated_at', 'unknown')} · Engine: {result.get('engine_version', 'unknown')} · Provider: {metadata.get('provider', 'quantitative')}")

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Overall regime", regime.get("overall", "—"))
c2.metric("Funds evaluated", len(funds))
c3.metric("Deployable cash", money(portfolio.get("funds_info", {}).get("deployable_cash")))
c4.metric("News events", market.get("news_count", 0))
c5.metric("LLM reviews", metadata.get("llm_review_count", 0))

profile_cols = st.columns(3)
profile_cols[0].metric("New investment", money(portfolio.get("funds_info", {}).get("deployable_cash")))
current = current_investor_inputs()
stale = mark_result_stale()
profile_cols[0].metric("Amount to invest", f"₹{current['amount']:,.0f}")
profile_cols[1].metric("Investor age", current["age"])
profile_cols[2].metric("Horizon", current["horizon"])
if stale:
    st.info("The investor profile shown above reflects your current dashboard inputs. The analysis below is from the previous run. Click **Run analysis** to update scores, allocation, and scenarios.")

st.divider()

tab_overview, tab_funds, tab_portfolio, tab_macro, tab_scenarios, tab_news, tab_diag, tab_raw = st.tabs([
    "Overview", "Fund Research", "Portfolio", "Macro & Regime", "Scenarios", "News", "Diagnostics", "Raw JSON"
])

with tab_overview:
    st.subheader("Market regime")
    regime_cols = ["equity", "rates", "liquidity", "inflation", "currency", "oil", "valuation", "geopolitics", "climate"]
    reg_df = pd.DataFrame({"Dimension": [x.title() for x in regime_cols], "Regime": [regime.get(x, "—") for x in regime_cols]})
    st.dataframe(reg_df, hide_index=True, use_container_width=True)

    st.subheader("Top candidates")
    top = result.get("top_candidates", [])
    rows = []
    for f in top:
        s = f.get("score", {})
        rows.append({"Fund": f["scheme_name"], "Category": f.get("category"), "AMC": f.get("amc"), "Action": f.get("final_action", f.get("action")), "Score": round(s.get("overall", 0), 1), "Fit": round(s["portfolio_fit"], 1) if s.get("portfolio_fit") is not None else None, "Macro resilience": round(s.get("macro_resilience", 0), 1) if s.get("macro_resilience") is not None else None})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Allocation plan")
    if allocation:
        st.dataframe(pd.DataFrame(allocation), hide_index=True, use_container_width=True)
    else:
        st.info("No new allocation passed the current score and portfolio constraints.")

with tab_funds:
    st.subheader("Fund ranking")
    rows = []
    for f in funds:
        s = f.get("score", {})
        m = f.get("fund_metrics", {})
        rows.append({"Fund": f["scheme_name"], "Category": f.get("category"), "AMC": f.get("amc"), "Action": f.get("final_action", f.get("action")), "Overall": s.get("overall"), "Quality": s.get("fund_quality"), "Risk-adjusted": s.get("risk_adjusted_return"), "Attractiveness": s.get("current_attractiveness"), "Portfolio fit": s.get("portfolio_fit"), "Valuation": s.get("valuation"), "Macro resilience": s.get("macro_resilience"), "Data confidence": s.get("data_confidence"), "Ranking score": s.get("ranking_score"), "1Y CAGR": m.get("cagr_1y_pct"), "3Y CAGR": m.get("cagr_3y_pct"), "5Y CAGR": m.get("cagr_5y_pct"), "Drawdown": m.get("max_drawdown_pct")})
    fund_columns = [
        "Fund", "Category", "AMC", "Action", "Overall", "Quality",
        "Risk-adjusted", "Attractiveness", "Portfolio fit", "Valuation",
        "Macro resilience", "Data confidence", "Ranking score", "1Y CAGR", "3Y CAGR", "5Y CAGR", "Drawdown",
    ]
    # Keep the table schema stable even when the data source returns no funds
    # or a fund record is missing a score field. Without this, pandas creates
    # an empty DataFrame with no columns and sort_values("Overall") raises KeyError.
    df = pd.DataFrame(rows, columns=fund_columns)
    if df.empty:
        st.warning(
            "No evaluated funds are available in the current analysis. "
            "Check the engine output/log above, fund data source, or network connectivity."
        )
    else:
        df["Overall"] = pd.to_numeric(df["Overall"], errors="coerce")
        df["Ranking score"] = pd.to_numeric(df["Ranking score"], errors="coerce")
        df = df.sort_values("Ranking score", ascending=False, na_position="last")
    st.dataframe(df, hide_index=True, use_container_width=True, height=520)
    st.caption("Ranking score = quantitative score adjusted for evidence completeness. A missing metric is treated as unknown, not as a neutral score.")

    names = [x["scheme_name"] for x in funds]
    if names:
        selected = st.selectbox("Inspect fund", names)
        fund = next(x for x in funds if x["scheme_name"] == selected)
        s = fund.get("score", {})
        m = fund.get("fund_metrics", {})
        a, b, c, d = st.columns(4)
        a.metric("Overall score", score_fmt(s.get("overall")))
        b.metric("Action", fund.get("final_action", fund.get("action", "—")))
        c.metric("1Y CAGR", pct(m.get("cagr_1y_pct")))
        d.metric("Max drawdown", pct(m.get("max_drawdown_pct")))
        left, right = st.columns(2)
        with left:
            st.write("**Reasons to buy**")
            for x in s.get("reasons_to_buy", []): st.write("•", x)
            st.write("**Data warnings**")
            for x in s.get("data_warnings", []): st.warning(x)
        with right:
            st.write("**Reasons not to buy**")
            for x in s.get("reasons_not_to_buy", []): st.write("•", x)
            review = fund.get("llm_review")
            if review:
                st.write("**LLM challenge review**")
                st.json(review)

with tab_portfolio:
    st.subheader("Current holdings")
    holdings = portfolio.get("holdings", {})
    if holdings:
        hrows = []
        for name, h in holdings.items():
            hrows.append({"Fund": name, "Current value": h.get("current_value"), "Invested": h.get("invested_value"), "P&L": h.get("pnl"), "P&L %": h.get("pnl_pct")})
        st.dataframe(pd.DataFrame(hrows), hide_index=True, use_container_width=True)
    else:
        st.info("No holdings file was loaded.")

    st.subheader("Existing exposure")
    exposure = portfolio.get("existing_exposure", {})
    if exposure:
        for key, value in exposure.items():
            st.write(f"**{key.replace('_', ' ').title()}**")
            if isinstance(value, dict):
                st.dataframe(pd.DataFrame([{"Name": k, "Exposure %": v} for k, v in value.items()]), hide_index=True, use_container_width=True)
            else:
                st.write(value)

with tab_macro:
    st.subheader("Macro snapshot")
    macro_rows = {
        "USD/INR": macro.get("usd_inr"), "Brent crude": macro.get("brent_crude_usd"), "India VIX": macro.get("india_vix"),
        "India 10Y yield": macro.get("india_10y_yield_pct"), "Nifty 50": macro.get("nifty_50"), "Nifty Midcap": macro.get("nifty_midcap"),
        "Nifty Smallcap": macro.get("nifty_smallcap"), "Gold": macro.get("gold_usd"), "US 10Y yield": macro.get("us_10y_yield_pct"),
        "S&P 500": macro.get("sp500"), "Crude 1M change": macro.get("crude_change_1m_pct"), "USD/INR 1M change": macro.get("usd_inr_change_1m_pct"),
        "VIX 1M change": macro.get("india_vix_change_1m_pct"), "Inflation": macro.get("inflation_pct"), "Repo rate": macro.get("repo_rate_pct"),
    }
    st.dataframe(pd.DataFrame([{"Indicator": k, "Value": v} for k, v in macro_rows.items()]), hide_index=True, use_container_width=True)
    st.subheader("Regime signals")
    st.dataframe(pd.DataFrame(regime.get("signals", [])), hide_index=True, use_container_width=True)

with tab_scenarios:
    st.subheader("Scenario resilience")
    scenario_rows = []
    for f in funds:
        scenario_data = f.get("scenario_analysis") or []
        # Current engine schema: list[dict], each containing scenario={name,...}.
        if isinstance(scenario_data, list):
            for details in scenario_data:
                if not isinstance(details, dict):
                    continue
                scenario = details.get("scenario") or {}
                scenario_name = scenario.get("name", "Unknown") if isinstance(scenario, dict) else str(scenario)
                scenario_rows.append({
                    "Fund": f["scheme_name"],
                    "Scenario": scenario_name,
                    "Resilience": details.get("resilience_score"),
                    "Risk penalty": details.get("risk_penalty"),
                    "Reasons": "; ".join(details.get("reasons", [])),
                })
        elif isinstance(scenario_data, dict):
            # Backward compatibility with an older dictionary-shaped result.
            for scenario, details in scenario_data.items():
                if isinstance(details, dict):
                    scenario_rows.append({"Fund": f["scheme_name"], "Scenario": scenario, **details})
    if scenario_rows:
        st.dataframe(pd.DataFrame(scenario_rows), hide_index=True, use_container_width=True, height=600)
    else:
        st.info("Scenario data is not available in this result.")

with tab_news:
    st.subheader("Structured news events")
    events = market.get("news_events", [])
    if events:
        st.dataframe(pd.DataFrame(events), hide_index=True, use_container_width=True, height=600)
    else:
        st.info("No news events were returned.")

with tab_diag:
    st.subheader("Local research pipeline")
    lr = result.get("local_ranking", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Evaluated funds", lr.get("input_funds", len(funds)))
    c2.metric("Quant top-N", lr.get("ranked_limit", "—"))
    c3.metric("Vector shortlist", lr.get("vector_limit", "—"))
    c4.metric("LLM candidates", len(lr.get("llm_candidates", [])))
    st.json({"data_confidence": lr.get("data_confidence", {}), "regime_confidence": market.get("regime", {}).get("data_confidence")})
    st.subheader("Data pipeline diagnostics")
    st.caption("This view is intentionally verbose so an empty fund universe can be diagnosed without checking the terminal.")
    logs = st.session_state.get("engine_logs", [])
    if not logs and CURRENT_LOG_PATH.exists():
        try:
            logs = CURRENT_LOG_PATH.read_text(encoding="utf-8").splitlines()
            st.session_state.engine_logs = logs
        except OSError:
            logs = []
    if logs:
        st.code("\n".join(logs), language="text")
    else:
        st.info("No engine log is available for this session. Run the analysis once.")

    st.subheader("Pipeline summary")
    pipeline_lines = [x for x in logs if "Universe pipeline" in x or "Fund data pipeline" in x or "Candidate universe" in x or "AMFI catalog" in x or "mftool catalog" in x]
    if pipeline_lines:
        st.code("\n".join(pipeline_lines), language="text")
    else:
        st.info("No pipeline summary lines were captured.")

with tab_raw:
    st.download_button("Download analysis JSON", data=json.dumps(result, indent=2, ensure_ascii=False), file_name="top_mutual_funds_analysis.json", mime="application/json")
    st.json(result)
