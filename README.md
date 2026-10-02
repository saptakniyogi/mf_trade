
## Local ranking and evidence retrieval (v2.2)

The research pipeline now separates quantitative ranking from evidence retrieval:

1. **Data-quality gate**: missing risk, holdings, valuation, benchmark, and other fields remain unknown rather than being converted into neutral scores.
2. **Quantitative ranking**: `overall` is retained for auditability; `ranking_score` applies a data-confidence adjustment before local ranking.
3. **Allocation gate**: fresh allocations require `overall >= 60` and data confidence of at least 50%.
4. **Vector retrieval**: TF-IDF is used as a lightweight local retrieval layer, followed by MMR diversification. It is not used as a substitute for quantitative fund quality.
5. **News filtering**: generic consumer/personal-finance noise is removed before vector retrieval.
6. **LLM context reduction**: macro/regime/news are shared once per batch instead of repeated inside every fund record. Only the compact fund evidence set is sent for qualitative challenge.
7. **Diagnostics**: the result now reports fund data-confidence buckets and market-regime confidence.

The current vector backend is intentionally lightweight. A later release can add a persistent sentence-transformer embedding backend without changing the ranking/scoring contract.
