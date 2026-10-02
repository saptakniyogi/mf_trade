from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("mf_agent")


@dataclass
class VectorIndex:
    documents: list[str]
    vectorizer: TfidfVectorizer
    matrix: object
    backend: str = "tfidf"

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if not self.documents or not str(query).strip():
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(-sims)[: max(0, int(top_k))]
        return [(int(i), float(sims[i])) for i in order if sims[i] > 0]


def _clean(text: object) -> str:
    return re.sub(r"\\s+", " ", str(text or "")).strip()


def fund_document(item: dict) -> str:
    metrics = item.get("fund_metrics", {})
    score = item.get("score", {})
    parts = [
        item.get("scheme_name", ""),
        item.get("category", ""),
        item.get("amc", ""),
        "Direct Growth",
        f"CAGR 1Y {metrics.get('cagr_1y_pct')}",
        f"CAGR 3Y {metrics.get('cagr_3y_pct')}",
        f"CAGR 5Y {metrics.get('cagr_5y_pct')}",
        f"drawdown {metrics.get('max_drawdown_pct')}",
        f"volatility {metrics.get('volatility_pct')}",
        f"Sharpe {metrics.get('sharpe')}",
        f"valuation {metrics.get('valuation')}",
        f"quality {score.get('fund_quality')}",
        f"risk adjusted {score.get('risk_adjusted_return')}",
        f"portfolio fit {score.get('portfolio_fit')}",
        f"macro resilience {score.get('macro_resilience')}",
        "sectors " + " ".join(str(k) for k in (metrics.get("sector_weights") or {}).keys()),
        "holdings " + " ".join(str(k) for k in (metrics.get("holdings") or {}).keys()),
    ]
    return _clean(" ".join(str(x) for x in parts if x not in (None, "")))


def news_document(article: dict) -> str:
    return _clean(" ".join([
        article.get("title", ""), article.get("summary", ""),
        " ".join(article.get("event_tags", [])),
        " ".join(article.get("transmission_channels", [])),
    ]))


def build_index(documents: list[str]) -> VectorIndex:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=20000)
    matrix = vectorizer.fit_transform(documents or [""])
    return VectorIndex(documents=documents, vectorizer=vectorizer, matrix=matrix)


def investor_query(engine_result: dict) -> str:
    settings = engine_result.get("settings", {})
    horizon = settings.get("horizon", {})
    age = settings.get("investor_age") or horizon.get("age")
    mode = settings.get("allocation_mode", "DIVERSIFIED")
    age_text = f" investor age {age}" if age else ""
    return _clean(
        f"Indian mutual fund investor{age_text} {horizon.get('label', '')} "
        f"{mode.lower()} portfolio seeking quality risk adjusted return "
        "controlled drawdown diversification macro resilience reasonable valuation"
    )



def _mmr_hits(index: VectorIndex, query: str, top_k: int, diversity_lambda: float = 0.72) -> list[tuple[int, float]]:
    """Maximal Marginal Relevance balances query relevance and document diversity."""
    if not index.documents or not query.strip():
        return []
    q = index.vectorizer.transform([query])
    relevance = cosine_similarity(q, index.matrix).ravel()
    selected: list[int] = []
    remaining = {int(i) for i in np.argsort(-relevance) if relevance[i] > 0}
    while remaining and len(selected) < min(int(top_k), len(index.documents)):
        best = None
        best_score = -1e9
        for i in remaining:
            redundancy = max((float(cosine_similarity(index.matrix[i], index.matrix[j])[0, 0]) for j in selected), default=0.0)
            mmr = diversity_lambda * float(relevance[i]) - (1.0 - diversity_lambda) * redundancy
            if mmr > best_score:
                best_score = mmr
                best = i
        selected.append(best)
        remaining.remove(best)
    return [(i, float(relevance[i])) for i in selected]

def retrieve_relevant_funds(candidates: list[dict], engine_result: dict, top_k: int = 15) -> list[dict]:
    if not candidates:
        return []
    docs = [fund_document(x) for x in candidates]
    index = build_index(docs)
    query = investor_query(engine_result)
    hits = _mmr_hits(index, query, top_k=min(top_k, len(candidates)))
    result = []
    for idx, similarity in hits:
        item = dict(candidates[idx])
        item["vector_similarity"] = round(similarity, 4)
        result.append(item)
    # Vector search can return zero similarity when a query has unusual terms.
    if not result:
        result = [dict(x) for x in candidates[:top_k]]
        for x in result:
            x["vector_similarity"] = 0.0
    logger.info("Local vector retrieval: %d candidates -> %d LLM candidates.", len(candidates), len(result))
    return result


def retrieve_relevant_news(news: list[dict], selected_funds: list[dict], top_k: int = 10) -> list[dict]:
    if not news:
        return []
    fund_text = " ".join(fund_document(x) for x in selected_funds)
    docs = [news_document(x) for x in news]
    index = build_index(docs)
    hits = index.search(fund_text, top_k=min(top_k, len(news)))
    result = []
    for idx, similarity in hits:
        item = dict(news[idx])
        item["vector_similarity"] = round(similarity, 4)
        result.append(item)
    logger.info("Local news retrieval: %d articles -> %d relevant articles.", len(news), len(result))
    return result
