from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("mf_agent")

VECTOR_SCHEMA_VERSION = "tfidf-v1"


@dataclass
class VectorIndex:
    """Persistent local vector index.

    The index is loaded from disk when the document fingerprint and vectorizer
    configuration match. The sparse TF-IDF matrix remains on disk rather than
    being rebuilt on every application run.
    """

    documents: list[str]
    vectorizer: TfidfVectorizer
    matrix: object
    backend: str = "tfidf"
    collection: str = "default"
    path: str | None = None
    cache_hit: bool = False

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if not self.documents or not str(query).strip():
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(-sims)[: max(0, int(top_k))]
        return [(int(i), float(sims[i])) for i in order if sims[i] > 0]


def _clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _default_store_dir() -> Path:
    return Path(os.getenv("VECTOR_STORE_DIR", "vector_store"))


def _documents_fingerprint(documents: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(VECTOR_SCHEMA_VERSION.encode("utf-8"))
    for document in documents:
        encoded = str(document).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _store_path(store_dir: Path, collection: str, fingerprint: str) -> Path:
    safe_collection = re.sub(r"[^a-zA-Z0-9_.-]+", "_", collection)
    return store_dir / f"{safe_collection}_{fingerprint}.pkl"


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _load_persistent_index(path: Path, documents: list[str], collection: str) -> VectorIndex | None:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("schema") != VECTOR_SCHEMA_VERSION:
            return None
        if payload.get("collection") != collection:
            return None
        if payload.get("documents") != documents:
            return None
        logger.info("Vector store cache hit: collection=%s path=%s", collection, path)
        return VectorIndex(
            documents=documents,
            vectorizer=payload["vectorizer"],
            matrix=payload["matrix"],
            backend=payload.get("backend", "tfidf"),
            collection=collection,
            path=str(path),
            cache_hit=True,
        )
    except (OSError, EOFError, pickle.PickleError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Ignoring invalid vector store cache %s: %s", path, exc)
        return None


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


def build_index(
    documents: list[str],
    collection: str = "default",
    store_dir: str | Path | None = None,
) -> VectorIndex:
    """Build or load a persistent on-disk TF-IDF vector index.

    A content-addressed filename means changed documents automatically create
    a new index while unchanged documents reuse the previous index.
    """
    documents = [str(x) for x in documents]
    root = (Path(store_dir) if store_dir else _default_store_dir()).expanduser().resolve()
    fingerprint = _documents_fingerprint(documents)
    path = _store_path(root, collection, fingerprint)

    cached = _load_persistent_index(path, documents, collection) if path.exists() else None
    if cached is not None:
        return cached

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=20000,
    )
    matrix = vectorizer.fit_transform(documents or [""])
    index = VectorIndex(
        documents=documents,
        vectorizer=vectorizer,
        matrix=matrix,
        collection=collection,
        path=str(path),
        cache_hit=False,
    )

    payload = {
        "schema": VECTOR_SCHEMA_VERSION,
        "collection": collection,
        "documents": documents,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "backend": "tfidf",
    }
    _write_atomic(path, payload)
    logger.info("Vector store cache miss: built collection=%s documents=%d path=%s", collection, len(documents), path)
    return index


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


def retrieve_relevant_funds(candidates: list[dict], engine_result: dict, top_k: int = 15, store_dir: str | Path | None = None) -> list[dict]:
    if not candidates:
        return []
    docs = [fund_document(x) for x in candidates]
    index = build_index(docs, collection="funds", store_dir=store_dir)
    query = investor_query(engine_result)
    hits = _mmr_hits(index, query, top_k=min(top_k, len(candidates)))
    result = []
    for idx, similarity in hits:
        item = dict(candidates[idx])
        item["vector_similarity"] = round(similarity, 4)
        result.append(item)
    if not result:
        result = [dict(x) for x in candidates[:top_k]]
        for x in result:
            x["vector_similarity"] = 0.0
    logger.info(
        "Local vector retrieval: %d candidates -> %d LLM candidates; cache_hit=%s",
        len(candidates), len(result), index.cache_hit,
    )
    return result


def retrieve_relevant_news(news: list[dict], selected_funds: list[dict], top_k: int = 10, store_dir: str | Path | None = None) -> list[dict]:
    if not news:
        return []
    categories = " ".join(str(x.get("category", "")) for x in selected_funds)
    names = " ".join(str(x.get("scheme_name", "")) for x in selected_funds[:8])
    # Query the event vocabulary, not the full numeric fund payload. This avoids
    # retrieving generic articles merely because they contain words like "fund"
    # or "portfolio".
    event_query = _clean(
        f"{categories} {names} RBI inflation rates oil crude rupee FII FPI DII "
        "earnings growth recession valuation liquidity regulation geopolitics market"
    )
    docs = [news_document(x) for x in news]
    index = build_index(docs, collection="news", store_dir=store_dir)
    hits = index.search(event_query, top_k=min(max(top_k * 2, top_k), len(news)))
    result = []
    for idx, similarity in hits:
        item = dict(news[idx])
        # TF-IDF is lexical retrieval, so reject weak matches rather than
        # pretending low similarity is meaningful evidence. Event-tagged news
        # gets a lower threshold because the deterministic classifier already
        # established a market-moving channel.
        # The upstream news filter has already established that these are
        # market/event-relevant articles. Event-tagged articles therefore need
        # only a weak lexical match; otherwise TF-IDF can discard valid macro
        # events simply because the article uses different wording.
        threshold = 0.03 if item.get("event_tags") else 0.20
        if similarity < threshold:
            continue
        item["vector_similarity"] = round(similarity, 4)
        result.append(item)
        if len(result) >= top_k:
            break
    # If lexical similarity is too sparse, fall back to deterministic event
    # evidence. This is preferable to sending zero news items to the LLM when
    # the classifier has already identified market-moving channels.
    if not result:
        event_items = [x for x in news if x.get("event_tags") or x.get("transmission_channels")]
        event_items.sort(key=lambda x: (len(x.get("event_tags", [])), len(x.get("transmission_channels", []))), reverse=True)
        for item in event_items[:top_k]:
            fallback = dict(item)
            fallback["vector_similarity"] = 0.0
            fallback["retrieval_method"] = "event_fallback"
            result.append(fallback)

    logger.info(
        "Local news retrieval: %d articles -> %d relevant articles; cache_hit=%s",
        len(news), len(result), index.cache_hit,
    )
    return result
