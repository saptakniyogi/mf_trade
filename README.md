
## Local ranking and evidence retrieval (v2.4)

The research pipeline now separates quantitative ranking from evidence retrieval:

1. **Data-quality gate**: missing risk, holdings, valuation, benchmark, and other fields remain unknown rather than being converted into neutral scores.
2. **Quantitative ranking**: `overall` is retained for auditability; `ranking_score` applies a data-confidence adjustment before local ranking.
3. **Allocation gate**: fresh allocations require `overall >= 60` and data confidence of at least 50%.
4. **Vector retrieval**: TF-IDF is used as a lightweight local retrieval layer, followed by MMR diversification. The vector index is persisted on disk under `VECTOR_STORE_DIR`, so unchanged fund/news documents reuse the cached index across runs. It is not used as a substitute for quantitative fund quality.
5. **News filtering**: generic consumer/personal-finance noise is removed before vector retrieval.
6. **Historical risk analytics**: daily NAV history is used to calculate 1Y/3Y/5Y CAGR plus annualized volatility, maximum drawdown, Sharpe and Sortino locally. A fallback history endpoint is used only when the primary client fails.
7. **Benchmark validation**: active-fund benchmarks are not guessed; index benchmarks are inferred only from explicit index names, with longer index names checked first to avoid mappings such as Nifty 500 -> Nifty 50.
8. **LLM context reduction**: macro/regime/news are shared once per batch instead of repeated inside every fund record. Only the compact fund evidence set is sent for qualitative challenge.
9. **Diagnostics**: the result now reports fund data-confidence buckets and market-regime confidence.

The current vector backend is intentionally lightweight and disk-persistent. A later release can add a semantic sentence-transformer backend without changing the ranking/scoring contract.
