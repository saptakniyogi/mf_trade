from mf_agent.analytics import holdings_overlap
from mf_agent.data import _benchmark_from_name, _parse_amfi_catalog, _parse_nav_history, is_direct_growth_scheme
from mf_agent.models import FundRecord, Holding, MacroSnapshot
from mf_agent.regime import infer_regime
from mf_agent.scenarios import scenario_matrix
from mf_agent.scoring import score_fund
from mf_agent.ranking import rank_candidates


def main():
    sample_nav = """Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Plan;Option;Net Asset Value;Date
Open Ended Schemes(Equity Scheme - Flexi Cap Fund)
Axis Mutual Fund
135762;INF846K01WO1;-;Axis Flexi Cap Fund;Direct Plan;Growth Option;29.0001;01-Oct-2026
135763;INF846K01WS2;INF846K01WQ6;Axis Flexi Cap Fund;Direct Plan;IDCW Option;26.7648;01-Oct-2026
"""
    catalog = _parse_amfi_catalog(sample_nav)
    assert len(catalog) == 2
    assert is_direct_growth_scheme(catalog["135762"]["scheme_name"], catalog["135762"]["plan"], catalog["135762"]["option"])
    assert not is_direct_growth_scheme(catalog["135763"]["scheme_name"], catalog["135763"]["plan"], catalog["135763"]["option"])
    assert _benchmark_from_name("Example Nifty Smallcap 50 Index Fund") == "Nifty Smallcap 50 TRI"
    hist = _parse_nav_history({"data": [{"date": "01-10-2026", "nav": "110"}, {"date": "30-09-2026", "nav": "109"}]})
    assert len(hist) == 2 and hist.iloc[0]["nav"] == 109

    a = {"HDFC Bank": 10, "Reliance Industries": 5}
    b = {"HDFC Bank": 8, "TCS": 5}
    assert 0 < holdings_overlap(a, b) < 100

    fund = FundRecord(
        scheme_name="Test Flexi Cap Direct Growth",
        scheme_code="1",
        category="Flexi Cap",
        amc="Test AMC",
        cagr_3y_pct=14,
        cagr_5y_pct=16,
        sharpe=0.9,
        sortino=1.2,
        volatility_pct=15,
        max_drawdown_pct=-25,
        holdings=a,
        sector_weights={"Financials": 40},
        market_cap_weights={"Large": 80},
        benchmark="Nifty 500 TRI",
        valuation={"pe": 24, "historical_percentile": 55},
        aum_inr_cr=1000,
    )
    regime = infer_regime(MacroSnapshot(brent_crude_usd=105, usd_inr=96, india_vix=18, crude_change_1m_pct=8, usd_inr_change_1m_pct=2))
    score = score_fund(fund, {}, [fund], regime)
    assert 0 <= score.overall <= 100
    assert score.portfolio_fit is not None
    assert score.risk_adjusted_return is not None
    assert score.data_confidence >= 75
    scenarios = scenario_matrix(fund, regime)
    assert len(scenarios) == 4
    assert 0 <= regime.data_confidence <= 100

    sparse = FundRecord(
        scheme_name="Sparse Thematic Direct Growth",
        scheme_code="2",
        category="Thematic / Sectoral",
        amc="Sparse AMC",
        cagr_3y_pct=18,
        cagr_5y_pct=20,
    )
    sparse_score = score_fund(sparse, {}, [sparse], regime)
    assert sparse_score.portfolio_fit is None
    assert sparse_score.risk_adjusted_return is None
    assert sparse_score.data_confidence < 50
    assert sparse_score.ranking_score < sparse_score.overall

    ranked = rank_candidates([
        {"scheme_name": "sparse", "score": sparse_score.to_dict()},
        {"scheme_name": "complete", "score": score.to_dict()},
    ], 2)
    assert ranked[0]["scheme_name"] == "complete"
    print("smoke tests passed")


if __name__ == "__main__":
    main()
