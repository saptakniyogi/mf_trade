from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import streamlit as st

from mf_agent.zerodha import (
    fetch_mf_holdings,
    get_login_url,
    refresh_holdings,
    save_holdings,
)

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

OUTPUT_PATH = ROOT / os.getenv(
    "MF_SUGGESTION_OUTPUT_PATH",
    "top_mutual_funds_analysis.json",
)

LOG_DIR = ROOT / "logs"
CURRENT_LOG_PATH = LOG_DIR / "mf_agent_latest.log"
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Streamlit session state
# ---------------------------------------------------------------------------

if "engine_logs" not in st.session_state:
    st.session_state.engine_logs = []

if "engine_running" not in st.session_state:
    st.session_state.engine_running = False

if "investor_inputs_applied" not in st.session_state:
    st.session_state.investor_inputs_applied = None

if "zerodha_access_token" not in st.session_state:
    st.session_state.zerodha_access_token = None

if "zerodha_user_name" not in st.session_state:
    st.session_state.zerodha_user_name = None

if "zerodha_user_id" not in st.session_state:
    st.session_state.zerodha_user_id = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def current_investor_inputs():
    return {
        "amount": float(
            st.session_state.get("investment_amount", 100000)
        ),
        "age": int(
            st.session_state.get("investor_age", 38)
        ),
        "horizon": st.session_state.get(
            "horizon", "LONG_TERM"
        ),
        "allocation_mode": st.session_state.get(
            "allocation_mode", "DIVERSIFIED"
        ),
    }


def mark_result_stale():
    applied = st.session_state.get("investor_inputs_applied")
    if applied is None:
        return False
    return current_investor_inputs() != applied


def money(value):
    if value is None:
        return "—"
    return f"₹{value:,.0f}"


def pct(value):
    if value is None:
        return "—"
    return f"{value:.1f}%"


def score_fmt(value):
    if value is None:
        return "—"
    return f"{value:.1f}"


def load_result():
    if not OUTPUT_PATH.exists():
        return None

    try:
        return json.loads(
            OUTPUT_PATH.read_text(encoding="utf-8")
        )
    except Exception as exc:
        st.error(
            f"Could not read {OUTPUT_PATH.name}: {exc}"
        )
        return None


def load_live_zerodha_holdings():
    """Read the current Zerodha holdings file directly.

    This is intentionally independent of the research-engine output so the
    Portfolio tab can show newly imported holdings even before a research
    analysis has completed successfully.
    """
    path = zerodha_holdings_path()

    if not path.exists():
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("mf_holdings", []) if isinstance(raw, dict) else raw
        return items if isinstance(items, list) else []
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Zerodha
# ---------------------------------------------------------------------------

def zerodha_holdings_path():
    """
    Resolve the holdings file used by the research engine.

    Keep this aligned with config.py, which uses
    MF_HOLDINGS_JSON_PATH.
    """
    configured_path = os.getenv(
        "MF_HOLDINGS_JSON_PATH",
        "mf_holdings.json",
    )

    path = Path(configured_path)

    if not path.is_absolute():
        path = ROOT / path

    return path


def handle_zerodha_callback():
    """
    Process the Zerodha OAuth callback.

    Zerodha redirects to the configured callback with:
        ?request_token=...
    """
    request_token = st.query_params.get("request_token")

    if not request_token:
        return

    try:
        holdings_path = zerodha_holdings_path()

        result = refresh_holdings(
            request_token=request_token,
            path=str(holdings_path),
        )

        access_token = result.get("access_token")

        if not access_token:
            raise RuntimeError(
                "Zerodha login succeeded but no access token was returned."
            )

        st.session_state.zerodha_access_token = access_token
        st.session_state.zerodha_user_name = result.get("user_name")
        st.session_state.zerodha_user_id = result.get("user_id")
        st.session_state.zerodha_last_holdings_count = result.get(
            "holdings_count", 0
        )
        st.session_state.zerodha_last_sync = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        st.session_state.zerodha_sync_message = (
            f"Imported {result.get('holdings_count', 0)} "
            "mutual fund holdings."
        )

        st.query_params.clear()

        st.success(
            "Zerodha connected successfully. "
            f"Imported {result.get('holdings_count', 0)} MF holdings."
        )
        st.rerun()

    except Exception as exc:
        st.error(f"Zerodha authentication failed: {exc}")
        st.info(
            "Check your Zerodha API key, API secret, redirect URL, "
            "and Kite Connect application configuration."
        )


def refresh_zerodha_holdings():
    """
    Fetch the latest MF holdings using the access token stored
    in the current Streamlit session.
    """
    access_token = st.session_state.get("zerodha_access_token")

    if not access_token:
        st.warning("Connect Zerodha first.")
        return False

    holdings_path = zerodha_holdings_path()

    try:
        holdings = fetch_mf_holdings(access_token)

        if not holdings:
            st.warning(
                "Zerodha returned zero MF holdings. "
                "Existing holdings were not overwritten."
            )
            return False

        save_holdings(
            holdings,
            str(holdings_path),
        )

        st.session_state.zerodha_last_holdings_count = len(holdings)
        st.session_state.zerodha_last_sync = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        st.session_state.zerodha_sync_message = (
            f"Refreshed {len(holdings)} mutual fund holdings."
        )

        return True

    except Exception as exc:
        st.error(
            f"Could not refresh Zerodha holdings: {exc}"
        )
        st.info(
            "If your Zerodha session has expired, "
            "use Connect Zerodha again."
        )
        return False


# Process OAuth callback before rendering the sidebar.
handle_zerodha_callback()


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

def apply_env_from_sidebar():
    os.environ["MF_HORIZON"] = st.session_state.horizon
    os.environ["MF_INVESTMENT_AMOUNT"] = str(
        st.session_state.investment_amount
    )
    os.environ["INVESTOR_AGE"] = str(
        st.session_state.investor_age
    )
    os.environ["ALLOCATION_MODE"] = (
        st.session_state.allocation_mode
    )
    os.environ["ENABLE_LLM_REVIEW"] = (
        "true"
        if st.session_state.llm_review
        else "false"
    )
    os.environ["MAX_CATEGORY_EXPOSURE_PCT"] = str(
        st.session_state.max_category
    )
    os.environ["MAX_SINGLE_FUND_PCT"] = str(
        st.session_state.max_fund
    )
    os.environ["MAX_SINGLE_AMC_PCT"] = str(
        st.session_state.max_amc
    )
    os.environ["MAX_MID_CAP_PCT"] = str(
        st.session_state.max_mid
    )
    os.environ["MAX_SMALL_CAP_PCT"] = str(
        st.session_state.max_small
    )
    os.environ["MAX_THEMATIC_PCT"] = str(
        st.session_state.max_thematic
    )
    os.environ["MIN_CASH_PCT"] = str(
        st.session_state.min_cash
    )
    os.environ["MAX_GOLD_PCT"] = str(
        st.session_state.max_gold
    )
    os.environ["MAX_FUND_OVERLAP_PCT"] = str(
        st.session_state.max_overlap
    )


# ---------------------------------------------------------------------------
# Research engine
# ---------------------------------------------------------------------------

def run_engine():
    apply_env_from_sidebar()

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "run_mf_analysis.py"),
    ]

    log_lines = []
    log_placeholder = st.empty()

    with st.status(
        "Running mutual-fund research engine…",
        expanded=True,
    ) as status:
        st.caption("Live engine log")

        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert proc.stdout is not None

        for line in proc.stdout:
            line = line.rstrip()

            if not line:
                continue

            log_lines.append(line)

            log_placeholder.code(
                "\n".join(log_lines[-80:]),
                language="text",
            )

        return_code = proc.wait()

        st.session_state.engine_logs = list(log_lines)

        try:
            log_text = (
                "\n".join(log_lines)
                + ("\n" if log_lines else "")
            )

            CURRENT_LOG_PATH.write_text(
                log_text,
                encoding="utf-8",
            )

            history_path = (
                LOG_DIR
                / f"mf_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )

            history_path.write_text(
                log_text,
                encoding="utf-8",
            )

            st.session_state.engine_log_path = str(history_path)

        except OSError as exc:
            st.session_state.engine_log_path = None
            st.warning(
                f"Could not persist engine log to disk: {exc}"
            )

        if return_code != 0:
            status.update(
                label="Analysis failed",
                state="error",
            )
            st.error(
                f"Research engine exited with code {return_code}. "
                "See Diagnostics → Engine log."
            )
            return None

        status.update(
            label="Analysis completed",
            state="complete",
        )

    return load_result()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Mutual Fund Research Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Mutual Fund Research Dashboard")

st.caption(
    "Regime-aware quantitative research with constrained "
    "portfolio allocation and optional LLM challenge review."
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Investor profile")

    st.number_input(
        "Amount to invest (₹)",
        min_value=0.0,
        value=float(
            os.getenv(
                "MF_INVESTMENT_AMOUNT",
                100000,
            )
        ),
        step=10000.0,
        key="investment_amount",
        help=(
            "New money available for this analysis. "
            "This is separate from existing holdings."
        ),
    )

    st.number_input(
        "Investor age",
        min_value=18,
        max_value=100,
        value=int(
            os.getenv(
                "INVESTOR_AGE",
                38,
            )
        ),
        step=1,
        key="investor_age",
        help=(
            "Used as investor-profile input. "
            "The current scoring engine records age but does not "
            "yet automatically change equity/debt allocation from age alone."
        ),
    )

    st.selectbox(
        "Investment horizon",
        ["LONG_TERM", "SHORT_TERM"],
        key="horizon",
    )

    st.selectbox(
        "Allocation mode",
        ["DIVERSIFIED", "CONCENTRATED"],
        key="allocation_mode",
    )

    st.toggle(
        "Enable OpenRouter review",
        value=(
            os.getenv(
                "ENABLE_LLM_REVIEW",
                "true",
            ).lower()
            in {"1", "true", "yes"}
        ),
        key="llm_review",
    )

    # -----------------------------------------------------------------------
    # Zerodha
    # -----------------------------------------------------------------------

    st.divider()
    st.header("Zerodha")

    zerodha_token = st.session_state.get(
        "zerodha_access_token"
    )

    if zerodha_token:
        st.success("Connected")

        user_name = st.session_state.get("zerodha_user_name")
        user_id = st.session_state.get("zerodha_user_id")

        if user_name:
            st.caption(f"User: {user_name}")

        if user_id:
            st.caption(f"User ID: {user_id}")

        holdings_count = st.session_state.get(
            "zerodha_last_holdings_count"
        )

        if holdings_count is not None:
            st.caption(f"MF holdings: {holdings_count}")

        last_sync = st.session_state.get("zerodha_last_sync")

        if last_sync:
            st.caption(f"Last sync: {last_sync}")

        if st.button(
            "↻ Refresh MF holdings",
            use_container_width=True,
        ):
            with st.spinner(
                "Fetching MF holdings from Zerodha..."
            ):
                refreshed = refresh_zerodha_holdings()

            if refreshed:
                st.success(
                    st.session_state.get(
                        "zerodha_sync_message",
                        "Holdings refreshed.",
                    )
                )

                # IMPORTANT:
                # mf_holdings.json has changed. Do not call load_result()
                # here because that only reads the previous analysis JSON.
                # Re-run the complete research engine instead.
                previous_result = st.session_state.get("result")

                with st.spinner(
                    "Rebuilding portfolio analysis from refreshed holdings..."
                ):
                    rebuilt_result = run_engine()

                if rebuilt_result:
                    st.session_state.result = rebuilt_result
                    st.session_state.investor_inputs_applied = (
                        current_investor_inputs()
                    )
                    st.success(
                        "Portfolio analysis rebuilt using the latest "
                        "Zerodha holdings."
                    )
                else:
                    st.session_state.result = previous_result
                    st.error(
                        "Portfolio analysis rebuild failed. "
                        "The previous dashboard result was kept."
                    )

                st.rerun()

    else:
        st.info(
            "Connect your Zerodha account to import "
            "current mutual fund holdings."
        )

        try:
            login_url = get_login_url()

            st.link_button(
                "🔐 Connect Zerodha",
                login_url,
                use_container_width=True,
            )

        except Exception as exc:
            st.error(
                f"Unable to generate Zerodha login URL: {exc}"
            )
            st.caption(
                "Check ZERODHA_API_KEY and your Kite Connect configuration."
            )

    holdings_path = zerodha_holdings_path()

    st.caption(
        f"Holdings file: `{holdings_path.name}`"
    )

    # -----------------------------------------------------------------------
    # Portfolio constraints
    # -----------------------------------------------------------------------

    with st.expander(
        "Portfolio constraints",
        expanded=False,
    ):
        st.number_input(
            "Max category %",
            1.0,
            100.0,
            float(
                os.getenv(
                    "MAX_CATEGORY_EXPOSURE_PCT",
                    35,
                )
            ),
            key="max_category",
        )

        st.number_input(
            "Max single fund %",
            1.0,
            100.0,
            float(
                os.getenv(
                    "MAX_SINGLE_FUND_PCT",
                    25,
                )
            ),
            key="max_fund",
        )

        st.number_input(
            "Max single AMC %",
            1.0,
            100.0,
            float(
                os.getenv(
                    "MAX_SINGLE_AMC_PCT",
                    35,
                )
            ),
            key="max_amc",
        )

        st.number_input(
            "Max mid-cap %",
            1.0,
            100.0,
            float(
                os.getenv(
                    "MAX_MID_CAP_PCT",
                    25,
                )
            ),
            key="max_mid",
        )

        st.number_input(
            "Max small-cap %",
            1.0,
            100.0,
            float(
                os.getenv(
                    "MAX_SMALL_CAP_PCT",
                    15,
                )
            ),
            key="max_small",
        )

        st.number_input(
            "Max thematic %",
            1.0,
            100.0,
            float(
                os.getenv(
                    "MAX_THEMATIC_PCT",
                    10,
                )
            ),
            key="max_thematic",
        )

        st.number_input(
            "Minimum cash %",
            0.0,
            100.0,
            float(
                os.getenv(
                    "MIN_CASH_PCT",
                    5,
                )
            ),
            key="min_cash",
        )

        st.number_input(
            "Max gold %",
            0.0,
            100.0,
            float(
                os.getenv(
                    "MAX_GOLD_PCT",
                    15,
                )
            ),
            key="max_gold",
        )

        st.number_input(
            "Max fund overlap %",
            0.0,
            100.0,
            float(
                os.getenv(
                    "MAX_FUND_OVERLAP_PCT",
                    60,
                )
            ),
            key="max_overlap",
        )

    # -----------------------------------------------------------------------
    # Analysis controls
    # -----------------------------------------------------------------------

    if mark_result_stale():
        st.warning(
            "Investor inputs changed. "
            "Click **Run analysis** to recalculate the portfolio."
        )

    if st.button(
        "▶ Run analysis",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.result = run_engine()
        st.session_state.investor_inputs_applied = (
            current_investor_inputs()
        )
        st.rerun()

    if st.button(
        "↻ Load latest JSON",
        use_container_width=True,
    ):
        st.session_state.result = load_result()

        loaded = st.session_state.result or {}
        settings = loaded.get("settings", {})
        horizon_settings = settings.get("horizon", {})

        st.session_state.investor_inputs_applied = {
            "amount": float(
                settings.get(
                    "investment_amount",
                    st.session_state.investment_amount,
                )
            ),
            "age": int(
                horizon_settings.get(
                    "age",
                    st.session_state.investor_age,
                )
            ),
            "horizon": horizon_settings.get(
                "mode",
                st.session_state.horizon,
            ),
            "allocation_mode": settings.get(
                "allocation_mode",
                st.session_state.allocation_mode,
            ),
        }

        st.rerun()


# ---------------------------------------------------------------------------
# Latest engine log
# ---------------------------------------------------------------------------

latest_logs = st.session_state.get(
    "engine_logs",
    [],
)

if latest_logs:
    with st.expander(
        "📜 Latest engine log",
        expanded=False,
    ):
        st.code(
            "\n".join(latest_logs),
            language="text",
        )

        log_path = st.session_state.get("engine_log_path")

        if log_path and Path(log_path).exists():
            st.download_button(
                "Download latest engine log",
                data="\n".join(latest_logs) + "\n",
                file_name=Path(log_path).name,
                mime="text/plain",
                key="download_latest_engine_log",
            )


# ---------------------------------------------------------------------------
# Load analysis result
# ---------------------------------------------------------------------------

result = (
    st.session_state.get("result")
    or load_result()
)

if not result:
    st.info(
        "No analysis result is available yet. "
        "Configure the controls and click **Run analysis**."
    )

    st.code(
        "streamlit run streamlit_app.py",
        language="bash",
    )

    st.stop()


# ---------------------------------------------------------------------------
# Result data
# ---------------------------------------------------------------------------

market = result.get("market", {})
macro = market.get("macro", {})
regime = market.get("regime", {})
portfolio = result.get("portfolio", {})
funds = result.get("evaluated_funds", [])
allocation = result.get("allocation_plan", [])
metadata = result.get("metadata", {})


# ---------------------------------------------------------------------------
# Result header
# ---------------------------------------------------------------------------

st.caption(
    f"Generated: {metadata.get('generated_at', 'unknown')} "
    f"· Engine: {result.get('engine_version', 'unknown')} "
    f"· Provider: {metadata.get('provider', 'quantitative')}"
)


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Overall regime",
    regime.get("overall", "—"),
)

c2.metric(
    "Funds evaluated",
    len(funds),
)

c3.metric(
    "Deployable cash",
    money(
        portfolio.get(
            "funds_info",
            {},
        ).get("deployable_cash")
    ),
)

c4.metric(
    "News events",
    market.get("news_count", 0),
)

c5.metric(
    "LLM reviews",
    metadata.get("llm_review_count", 0),
)


# ---------------------------------------------------------------------------
# Investor profile summary
# ---------------------------------------------------------------------------

profile_cols = st.columns(3)

profile_cols[0].metric(
    "Amount to invest",
    f"₹{current_investor_inputs()['amount']:,.0f}",
)

profile_cols[1].metric(
    "Investor age",
    current_investor_inputs()["age"],
)

profile_cols[2].metric(
    "Horizon",
    current_investor_inputs()["horizon"],
)

if mark_result_stale():
    st.info(
        "The investor profile shown above reflects your current dashboard "
        "inputs. The analysis below is from the previous run. "
        "Click **Run analysis** to update scores, allocation, and scenarios."
    )

st.divider()


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

(
    tab_overview,
    tab_funds,
    tab_portfolio,
    tab_macro,
    tab_scenarios,
    tab_news,
    tab_diag,
    tab_raw,
) = st.tabs(
    [
        "Overview",
        "Fund Research",
        "Portfolio",
        "Macro & Regime",
        "Scenarios",
        "News",
        "Diagnostics",
        "Raw JSON",
    ]
)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

with tab_overview:
    st.subheader("Market regime")

    regime_cols = [
        "equity",
        "rates",
        "liquidity",
        "inflation",
        "currency",
        "oil",
        "valuation",
        "geopolitics",
        "climate",
    ]

    reg_df = pd.DataFrame(
        {
            "Dimension": [x.title() for x in regime_cols],
            "Regime": [
                regime.get(x, "—")
                for x in regime_cols
            ],
        }
    )

    st.dataframe(
        reg_df,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Top candidates")

    top = result.get("top_candidates", [])
    rows = []

    for fund in top:
        score = fund.get("score", {})

        rows.append(
            {
                "Fund": fund["scheme_name"],
                "Category": fund.get("category"),
                "AMC": fund.get("amc"),
                "Action": fund.get(
                    "final_action",
                    fund.get("action"),
                ),
                "Score": round(
                    score.get("overall", 0),
                    1,
                ),
                "Fit": (
                    round(
                        score["portfolio_fit"],
                        1,
                    )
                    if score.get("portfolio_fit") is not None
                    else None
                ),
                "Macro resilience": (
                    round(
                        score["macro_resilience"],
                        1,
                    )
                    if score.get("macro_resilience") is not None
                    else None
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Allocation plan")

    if allocation:
        st.dataframe(
            pd.DataFrame(allocation),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(
            "No new allocation passed the current "
            "score and portfolio constraints."
        )


# ---------------------------------------------------------------------------
# Fund Research
# ---------------------------------------------------------------------------

with tab_funds:
    st.subheader("Fund ranking")

    rows = []

    for fund in funds:
        score = fund.get("score", {})
        metrics = fund.get("fund_metrics", {})

        rows.append(
            {
                "Fund": fund["scheme_name"],
                "Category": fund.get("category"),
                "AMC": fund.get("amc"),
                "Action": fund.get(
                    "final_action",
                    fund.get("action"),
                ),
                "Overall": score.get("overall"),
                "Quality": score.get("fund_quality"),
                "Risk-adjusted": score.get(
                    "risk_adjusted_return"
                ),
                "Attractiveness": score.get(
                    "current_attractiveness"
                ),
                "Portfolio fit": score.get(
                    "portfolio_fit"
                ),
                "Valuation": score.get("valuation"),
                "Macro resilience": score.get(
                    "macro_resilience"
                ),
                "Data confidence": score.get(
                    "data_confidence"
                ),
                "Ranking score": score.get(
                    "ranking_score"
                ),
                "1Y CAGR": metrics.get("cagr_1y_pct"),
                "3Y CAGR": metrics.get("cagr_3y_pct"),
                "5Y CAGR": metrics.get("cagr_5y_pct"),
                "Drawdown": metrics.get(
                    "max_drawdown_pct"
                ),
            }
        )

    fund_columns = [
        "Fund",
        "Category",
        "AMC",
        "Action",
        "Overall",
        "Quality",
        "Risk-adjusted",
        "Attractiveness",
        "Portfolio fit",
        "Valuation",
        "Macro resilience",
        "Data confidence",
        "Ranking score",
        "1Y CAGR",
        "3Y CAGR",
        "5Y CAGR",
        "Drawdown",
    ]

    df = pd.DataFrame(
        rows,
        columns=fund_columns,
    )

    if df.empty:
        st.warning(
            "No evaluated funds are available in the current analysis. "
            "Check the engine output/log above, fund data source, "
            "or network connectivity."
        )
    else:
        df["Overall"] = pd.to_numeric(
            df["Overall"],
            errors="coerce",
        )
        df["Ranking score"] = pd.to_numeric(
            df["Ranking score"],
            errors="coerce",
        )
        df = df.sort_values(
            "Ranking score",
            ascending=False,
            na_position="last",
        )

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        height=520,
    )

    st.caption(
        "Ranking score = quantitative score adjusted for evidence completeness. "
        "A missing metric is treated as unknown, not as a neutral score."
    )

    names = [
        fund["scheme_name"]
        for fund in funds
    ]

    if names:
        selected = st.selectbox(
            "Inspect fund",
            names,
        )

        fund = next(
            item
            for item in funds
            if item["scheme_name"] == selected
        )

        score = fund.get("score", {})
        metrics = fund.get("fund_metrics", {})

        a, b, c, d = st.columns(4)

        a.metric(
            "Overall score",
            score_fmt(score.get("overall")),
        )

        b.metric(
            "Action",
            fund.get(
                "final_action",
                fund.get("action", "—"),
            ),
        )

        c.metric(
            "1Y CAGR",
            pct(metrics.get("cagr_1y_pct")),
        )

        d.metric(
            "Max drawdown",
            pct(metrics.get("max_drawdown_pct")),
        )

        left, right = st.columns(2)

        with left:
            st.write("**Reasons to buy**")

            for item in score.get(
                "reasons_to_buy",
                [],
            ):
                st.write("•", item)

            st.write("**Data warnings**")

            for item in score.get(
                "data_warnings",
                [],
            ):
                st.warning(item)

        with right:
            st.write("**Reasons not to buy**")

            for item in score.get(
                "reasons_not_to_buy",
                [],
            ):
                st.write("•", item)

            review = fund.get("llm_review")

            if review:
                st.write("**LLM challenge review**")
                st.json(review)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

with tab_portfolio:
    st.subheader("Current Zerodha holdings")

    # Read mf_holdings.json directly. Do not depend on the last research
    # result, because Zerodha holdings can be refreshed independently.
    live_holdings = load_live_zerodha_holdings()

    if live_holdings:
        hrows = []
        for item in live_holdings:
            hrows.append(
                {
                    "Fund": item.get("fund"),
                    "Quantity": item.get("quantity"),
                    "Avg price": item.get("average_price"),
                    "Current value": item.get("current_value"),
                    "Invested": item.get("invested_value"),
                    "P&L": item.get("pnl"),
                    "P&L %": item.get("pnl_pct"),
                    "Folio": item.get("folio"),
                }
            )

        live_df = pd.DataFrame(hrows)
        for column in [
            "Quantity",
            "Avg price",
            "Current value",
            "Invested",
            "P&L",
            "P&L %",
        ]:
            live_df[column] = pd.to_numeric(
                live_df[column],
                errors="coerce",
            )

        total_current = live_df["Current value"].sum()
        total_invested = live_df["Invested"].sum()
        total_pnl = live_df["P&L"].sum()
        total_pnl_pct = (
            total_pnl / total_invested * 100
            if total_invested
            else 0.0
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Funds", len(live_df))
        k2.metric("Current value", money(total_current))
        k3.metric("Invested", money(total_invested))
        k4.metric("Total P&L", f"₹{total_pnl:,.0f} ({total_pnl_pct:.1f}%)")

        st.dataframe(
            live_df,
            hide_index=True,
            use_container_width=True,
        )

        holdings_path = zerodha_holdings_path()
        try:
            updated_at = json.loads(
                holdings_path.read_text(encoding="utf-8")
            ).get("updated_at")
        except Exception:
            updated_at = None

        if updated_at:
            st.caption(f"Source: {holdings_path.name} · Last synced: {updated_at}")
    else:
        st.info(
            "No live Zerodha holdings are available. "
            "Connect Zerodha and click **Refresh MF holdings** first."
        )

    st.subheader("Research-engine portfolio")

    holdings = portfolio.get("holdings", {})

    if holdings:
        hrows = []
        for name, holding in holdings.items():
            hrows.append(
                {
                    "Fund": name,
                    "Current value": holding.get("current_value"),
                    "Invested": holding.get("invested_value"),
                    "P&L": holding.get("pnl"),
                    "P&L %": holding.get("pnl_pct"),
                }
            )

        st.dataframe(
            pd.DataFrame(hrows),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(
            "The research result contains no holdings. "
            "This does not prevent the live Zerodha holdings above from being shown."
        )

    st.subheader("Existing exposure")

    exposure = portfolio.get(
        "existing_exposure",
        {},
    )

    if exposure:
        for key, value in exposure.items():
            st.write(
                f"**{key.replace('_', ' ').title()}**"
            )

            if isinstance(value, dict):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Name": name,
                                "Exposure %": exposure_value,
                            }
                            for name, exposure_value
                            in value.items()
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.write(value)


# ---------------------------------------------------------------------------
# Macro & Regime
# ---------------------------------------------------------------------------

with tab_macro:
    st.subheader("Macro snapshot")

    macro_rows = {
        "USD/INR": macro.get("usd_inr"),
        "Brent crude": macro.get("brent_crude_usd"),
        "India VIX": macro.get("india_vix"),
        "India 10Y yield": macro.get(
            "india_10y_yield_pct"
        ),
        "Nifty 50": macro.get("nifty_50"),
        "Nifty Midcap": macro.get("nifty_midcap"),
        "Nifty Smallcap": macro.get(
            "nifty_smallcap"
        ),
        "Gold": macro.get("gold_usd"),
        "US 10Y yield": macro.get(
            "us_10y_yield_pct"
        ),
        "S&P 500": macro.get("sp500"),
        "Crude 1M change": macro.get(
            "crude_change_1m_pct"
        ),
        "USD/INR 1M change": macro.get(
            "usd_inr_change_1m_pct"
        ),
        "VIX 1M change": macro.get(
            "india_vix_change_1m_pct"
        ),
        "Inflation": macro.get("inflation_pct"),
        "Repo rate": macro.get("repo_rate_pct"),
    }

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Indicator": key,
                    "Value": value,
                }
                for key, value in macro_rows.items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Regime signals")

    st.dataframe(
        pd.DataFrame(
            regime.get("signals", [])
        ),
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

with tab_scenarios:
    st.subheader("Scenario resilience")

    scenario_rows = []

    for fund in funds:
        scenario_data = (
            fund.get("scenario_analysis")
            or []
        )

        if isinstance(
            scenario_data,
            list,
        ):
            for details in scenario_data:
                if not isinstance(details, dict):
                    continue

                scenario = (
                    details.get("scenario")
                    or {}
                )

                if isinstance(
                    scenario,
                    dict,
                ):
                    scenario_name = scenario.get(
                        "name",
                        "Unknown",
                    )
                else:
                    scenario_name = str(scenario)

                scenario_rows.append(
                    {
                        "Fund": fund["scheme_name"],
                        "Scenario": scenario_name,
                        "Resilience": details.get(
                            "resilience_score"
                        ),
                        "Risk penalty": details.get(
                            "risk_penalty"
                        ),
                        "Reasons": "; ".join(
                            details.get(
                                "reasons",
                                [],
                            )
                        ),
                    }
                )

        elif isinstance(
            scenario_data,
            dict,
        ):
            for scenario, details in scenario_data.items():
                if isinstance(details, dict):
                    scenario_rows.append(
                        {
                            "Fund": fund["scheme_name"],
                            "Scenario": scenario,
                            **details,
                        }
                    )

    if scenario_rows:
        st.dataframe(
            pd.DataFrame(scenario_rows),
            hide_index=True,
            use_container_width=True,
            height=600,
        )
    else:
        st.info(
            "Scenario data is not available in this result."
        )


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

with tab_news:
    st.subheader("Structured news events")

    events = market.get("news_events", [])

    if events:
        st.dataframe(
            pd.DataFrame(events),
            hide_index=True,
            use_container_width=True,
            height=600,
        )
    else:
        st.info("No news events were returned.")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

with tab_diag:
    st.subheader("Local research pipeline")

    local_ranking = result.get(
        "local_ranking",
        {},
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Evaluated funds",
        local_ranking.get(
            "input_funds",
            len(funds),
        ),
    )

    c2.metric(
        "Quant top-N",
        local_ranking.get(
            "ranked_limit",
            "—",
        ),
    )

    c3.metric(
        "Vector shortlist",
        local_ranking.get(
            "vector_limit",
            "—",
        ),
    )

    c4.metric(
        "LLM candidates",
        len(
            local_ranking.get(
                "llm_candidates",
                [],
            )
        ),
    )

    st.json(
        {
            "data_confidence": local_ranking.get(
                "data_confidence",
                {},
            ),
            "regime_confidence": market.get(
                "regime",
                {},
            ).get(
                "data_confidence"
            ),
        }
    )

    st.subheader("Data pipeline diagnostics")

    st.caption(
        "This view is intentionally verbose so an empty fund universe "
        "can be diagnosed without checking the terminal."
    )

    logs = st.session_state.get(
        "engine_logs",
        [],
    )

    if (
        not logs
        and CURRENT_LOG_PATH.exists()
    ):
        try:
            logs = (
                CURRENT_LOG_PATH.read_text(
                    encoding="utf-8"
                ).splitlines()
            )
            st.session_state.engine_logs = logs
        except OSError:
            logs = []

    if logs:
        st.code(
            "\n".join(logs),
            language="text",
        )
    else:
        st.info(
            "No engine log is available for this session. "
            "Run the analysis once."
        )

    st.subheader("Pipeline summary")

    pipeline_lines = [
        line
        for line in logs
        if (
            "Universe pipeline" in line
            or "Fund data pipeline" in line
            or "Candidate universe" in line
            or "AMFI catalog" in line
            or "mftool catalog" in line
        )
    ]

    if pipeline_lines:
        st.code(
            "\n".join(pipeline_lines),
            language="text",
        )
    else:
        st.info(
            "No pipeline summary lines were captured."
        )


# ---------------------------------------------------------------------------
# Raw JSON
# ---------------------------------------------------------------------------

with tab_raw:
    st.download_button(
        "Download analysis JSON",
        data=json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        file_name="top_mutual_funds_analysis.json",
        mime="application/json",
    )

    st.json(result)
