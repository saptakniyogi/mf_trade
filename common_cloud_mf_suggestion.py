"""Backward-compatible facade for the refactored MF engine.

New code should import from mf_agent.* directly.
"""
from mf_agent.config import load_settings
from mf_agent.data import load_holdings, build_fund_universe
from mf_agent.engine import ResearchEngine
from mf_agent.macro import fetch_macro_snapshot
from mf_agent.news import fetch_news, classify_event
from mf_agent.regime import infer_regime
from mf_agent.scoring import score_fund
from mf_agent.utils import chunked, sanitize_text

_SETTINGS = load_settings()
BATCH_SIZE = _SETTINGS.batch_size
INTER_BATCH_DELAY_SEC = _SETTINGS.inter_batch_delay_sec
MAX_CATEGORY_EXPOSURE_PCT = _SETTINGS.policy.max_category_pct
MIN_AUM_INR_CR = _SETTINGS.min_aum_inr_cr
MAX_SCHEMES_PER_CAT = _SETTINGS.max_schemes_per_category
MAX_PER_AMC_PER_CAT = _SETTINGS.max_per_amc_per_category
ALLOCATION_MODE = _SETTINGS.allocation_mode
MF_HORIZON_MODE = _SETTINGS.horizon.mode
HORIZON_LABEL = _SETTINGS.horizon.label
HORIZON_JSON_VAL = _SETTINGS.horizon.json_value
USER_AGE = _SETTINGS.horizon.age
mf_holdings_path = _SETTINGS.mf_holdings_path
suggestion_output_path = _SETTINGS.output_path


def build_mf_market_payload():
    # Compatibility adapter. The new engine returns a richer, normalized structure.
    return ResearchEngine(_SETTINGS).build()


def build_mf_system_prompt():
    from mf_agent.llm import SYSTEM_PROMPT
    return SYSTEM_PROMPT
