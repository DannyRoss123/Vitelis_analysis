"""
Vector store module — v2
========================
- Semantic chunking: sentence-aware splitting with configurable overlap
- Embeddings: OpenAI (default) or local SentenceTransformer fallback
- Pure cosine similarity retrieval (no tier boosting at retrieval time)
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chromadb
import numpy as np

from app.debug_log import add_debug


# ============================================================================
# OPENAI EMBEDDING FUNCTION (ChromaDB-compatible interface)
# ============================================================================

class OpenAIEmbeddingFunction:
    """Calls OpenAI text-embedding-3-small via the REST API.

    Implements the ChromaDB EmbeddingFunction protocol (__call__,
    embed_documents, embed_query).

    Batches large inputs to stay within the API's token-per-request limits
    and adds retry logic for transient 429 errors.
    """

    MODEL = "text-embedding-3-small"
    DIMENSIONS = 1536
    MAX_BATCH = 100  # OpenAI recommends <=2048 inputs; 100 is safe & fast

    def __init__(self) -> None:
        self._api_key: Optional[str] = None  # Lazy — read at first use
        self._last_call_at: Optional[float] = None
        self._min_delay: Optional[float] = None

    @property
    def api_key(self) -> Optional[str]:
        if self._api_key is None:
            self._api_key = os.getenv("OPENAI_API_KEY")
        return self._api_key

    @property
    def min_delay(self) -> float:
        if self._min_delay is None:
            self._min_delay = float(os.getenv("VITELIS_EMBED_MIN_DELAY", "0.1"))
        return self._min_delay

    # -- ChromaDB protocol ------------------------------------------------
    def __call__(self, input: Iterable[str]) -> List[List[float]]:
        return self.embed_documents(list(input))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH):
            batch = texts[start: start + self.MAX_BATCH]
            all_embeddings.extend(self._call_api(batch))
        return all_embeddings

    def embed_query(self, input: str | List[str]) -> List[List[float]]:
        if isinstance(input, list):
            return self.embed_documents(input)
        return self.embed_documents([input])

    def name(self) -> str:
        return "openai-text-embedding-3-small"

    def get_config(self) -> dict:
        return {"model": self.MODEL, "dimensions": self.DIMENSIONS}

    # -- Internal ---------------------------------------------------------
    def _call_api(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        import requests

        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAI embeddings. "
                "Set it in .env or environment."
            )

        base_backoff = float(os.getenv("VITELIS_EMBED_BACKOFF", "1.5"))

        for attempt in range(max_retries + 1):
            # Rate-limit throttle
            if self._last_call_at is not None:
                elapsed = time.time() - self._last_call_at
                if elapsed < self.min_delay:
                    time.sleep(self.min_delay - elapsed)

            try:
                response = requests.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.MODEL,
                        "input": texts,
                    },
                    timeout=60,
                )
                self._last_call_at = time.time()

                if response.status_code == 429 and attempt < max_retries:
                    retry_after = response.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else base_backoff * (2 ** attempt)
                    if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}:
                        add_debug(f"[embed] 429 received; retrying in {wait:.1f}s")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()["data"]
                # Sort by index to ensure correct order
                data.sort(key=lambda x: x["index"])
                return [item["embedding"] for item in data]

            except requests.RequestException as exc:
                if attempt < max_retries:
                    wait = base_backoff * (2 ** attempt)
                    if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}:
                        add_debug(f"[embed] request failed; retrying in {wait:.1f}s ({exc})")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"OpenAI embedding API failed after {max_retries} retries: {exc}") from exc

        raise RuntimeError("OpenAI embedding API: exhausted retries")


class LocalEmbeddingFunction:
    """Dependency-free local embedding backend.

    Uses a simple hashed bag-of-words embedding so indexing/retrieval can run
    without external APIs or heavyweight ML runtime dependencies.
    """

    MODEL = "hash-bow-256d"
    DIMENSIONS = 256

    def _embed_text(self, text: str) -> List[float]:
        tokens = re.findall(r"\b\w+\b", (text or "").lower())
        vec = np.zeros(self.DIMENSIONS, dtype=np.float32)
        for tok in tokens:
            h = hash(tok)
            idx = abs(h) % self.DIMENSIONS
            vec[idx] += 1.0 if h >= 0 else -1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def __call__(self, input: Iterable[str]) -> List[List[float]]:
        return self.embed_documents(list(input))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, input: str | List[str]) -> List[List[float]]:
        if isinstance(input, list):
            return self.embed_documents(input)
        return self.embed_documents([input])

    def name(self) -> str:
        return f"local-{self.MODEL}"

    def get_config(self) -> dict:
        return {"model": self.name()}


# ============================================================================
# SEMANTIC CHUNKING
# ============================================================================

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex heuristics."""
    # Split on sentence-ending punctuation followed by whitespace or end
    parts = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty strings
    return [s.strip() for s in parts if s.strip()]


def chunk_text(
    text: str,
    max_chars: int = 1000,
    overlap_chars: int = 200,
) -> List[str]:
    """
    Semantic-aware text chunking.

    Strategy:
    1. Split text into sentences
    2. Group sentences into chunks up to max_chars
    3. Add overlap_chars of context from the end of the previous chunk

    This ensures:
    - Chunks never split mid-sentence
    - Each chunk has leading context from the previous chunk
    - Chunks are large enough for meaningful retrieval (~200-250 words)
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence exceeds max_chars, split it by whitespace
        if sentence_len > max_chars:
            # Flush current buffer first
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_len = 0
            # Hard-split the long sentence
            words = sentence.split()
            buf = []
            buf_len = 0
            for word in words:
                if buf_len + len(word) + 1 > max_chars and buf:
                    chunks.append(" ".join(buf))
                    buf = []
                    buf_len = 0
                buf.append(word)
                buf_len += len(word) + 1
            if buf:
                chunks.append(" ".join(buf))
            continue

        # Would adding this sentence exceed max_chars?
        if current_len + sentence_len + 1 > max_chars and current_sentences:
            chunk_text_str = " ".join(current_sentences)
            chunks.append(chunk_text_str)

            # Build overlap: take trailing sentences from current chunk
            overlap_sentences: List[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) + 1 > overlap_chars:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s) + 1

            current_sentences = overlap_sentences
            current_len = sum(len(s) for s in current_sentences) + len(current_sentences)

        current_sentences.append(sentence)
        current_len += sentence_len + 1

    # Flush remaining
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# ============================================================================
# COLLECTION MANAGEMENT
# ============================================================================

def build_collection(run_id: str):
    """Create or open a ChromaDB collection with configured embeddings."""
    persist_dir = os.path.join(os.path.dirname(__file__), "data", f"chroma_{run_id}")
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    provider = (os.getenv("VITELIS_EMBED_PROVIDER") or "").strip().lower()
    llm_provider = (os.getenv("VITELIS_LLM_PROVIDER") or "").strip().lower()
    use_local = provider == "local" or (llm_provider in {"gemini", "google"} and provider != "openai")
    embedding_fn = LocalEmbeddingFunction() if use_local else OpenAIEmbeddingFunction()
    collection = client.get_or_create_collection(
        "sources",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # Explicit cosine similarity
    )
    return collection


def index_sources(collection, sources: List[dict]) -> None:
    """Chunk and index source documents into the vector store."""
    ids: List[str] = []
    docs: List[str] = []
    metadatas: List[dict] = []

    for source in sources:
        chunks = chunk_text(source["text"])
        # Tag tier-1 sources as "primary" so score_extensions can build a
        # baseline_score from the frozen reference corpus (Feature 1 & 9).
        source_type = "primary" if source.get("tier", 3) == 1 else "secondary"
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{source['source_id']}::chunk_{idx}"
            ids.append(chunk_id)
            docs.append(chunk)
            metadatas.append(
                {
                    "source_id": source["source_id"],
                    "chunk_id": chunk_id,  # stored for retrieval metrics lookup
                    "url": source["url"],
                    "title": source.get("title", ""),
                    "domain": source.get("domain", ""),
                    "retrieved_at": source.get("retrieved_at", ""),
                    "tier": source["tier"],
                    "source_type": source_type,  # "primary" | "secondary"
                }
            )

    # Batch to stay under ChromaDB limits
    batch_size = 100  # Smaller batches to manage OpenAI embedding API calls
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metadatas[start:end],
        )
        if os.getenv("VITELIS_DEBUG", "").lower() in {"1", "true", "yes"}:
            add_debug(f"[index] indexed chunks {start}–{min(end, len(ids))} of {len(ids)}")


def retrieve_evidence(
    collection,
    query: str,
    k: int = 10,
    where: Optional[Dict[str, Any]] = None,
) -> List[Tuple[dict, str, float]]:
    """
    Retrieve top-k chunks by pure cosine similarity.

    No tier boosting — retrieval is purely semantic.
    Returns: List of (metadata, document, similarity_score) tuples.
    """
    if not query.strip():
        return []

    mode = (os.getenv("VITELIS_RETRIEVAL_MODE") or "semantic").strip().lower()
    semantic_top_k = int(os.getenv("VITELIS_SEMANTIC_TOP_K", str(k)) or k)
    bm25_top_k = int(os.getenv("VITELIS_BM25_TOP_K", str(k)) or k)
    rrf_k = int(os.getenv("VITELIS_RRF_K", "60") or 60)
    fused_k = int(os.getenv("VITELIS_RRF_FUSED_K", str(max(k, semantic_top_k + bm25_top_k))) or k)

    if mode in {"hybrid_rrf", "hybrid", "rrf"}:
        # Hybrid: semantic + BM25, fuse with Reciprocal Rank Fusion (RRF)
        semantic_rows = _semantic_retrieve(collection, query, k=semantic_top_k, where=where)
        bm25_rows = _bm25_retrieve(collection, query, k=bm25_top_k, where=where)
        fused = _rrf_fuse(
            semantic=semantic_rows,
            bm25=bm25_rows,
            k=rrf_k,
        )
        # Return top fused_k candidates; score is fused score (reranker will re-score anyway)
        return [(meta, doc, float(score)) for (_cid, meta, doc, score) in fused[:fused_k]]

    # Default: semantic-only retrieval (existing behavior)
    rows = _semantic_retrieve(collection, query, k=k, where=where)
    return [(meta, doc, float(score)) for (_cid, meta, doc, score) in rows]


# ============================================================================
# HYBRID RETRIEVAL: Semantic + BM25 + RRF
# ============================================================================

_BM25_CACHE: Dict[int, Dict[str, Any]] = {}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())


def _meta_matches_where(meta: Dict[str, Any], where: Optional[Dict[str, Any]]) -> bool:
    """
    Minimal matcher for the subset of Chroma 'where' queries we use:
      - {"field": {"$eq": value}}
      - {"$or": [<where>, <where>, ...]}
    """
    if not where:
        return True
    if "$or" in where and isinstance(where["$or"], list):
        return any(_meta_matches_where(meta, w) for w in where["$or"])
    for key, cond in where.items():
        if not isinstance(cond, dict) or "$eq" not in cond:
            return False
        if meta.get(key) != cond["$eq"]:
            return False
    return True


def _semantic_retrieve(
    collection,
    query: str,
    k: int,
    where: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, dict, str, float]]:
    results = collection.query(query_texts=[query], n_results=k, where=where)
    ids = results.get("ids", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    documents = results.get("documents", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    rows: List[Tuple[str, dict, str, float]] = []
    for cid, metadata, document, distance in zip(ids, metadatas, documents, distances):
        meta = metadata if isinstance(metadata, dict) else {}
        # Chroma cosine distance in [0,2] -> similarity [0,1]
        similarity = max(0.0, 1.0 - (float(distance) / 2.0))
        rows.append((str(cid), meta, str(document), float(similarity)))
    return rows


def _get_bm25_index(collection) -> Optional[Dict[str, Any]]:
    """
    Build or reuse a BM25 index for a given Chroma collection.
    Cached per-process using id(collection).
    """
    key = id(collection)
    cached = _BM25_CACHE.get(key)
    if cached:
        return cached

    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except Exception:
        return None

    try:
        result = collection.get(include=["documents", "metadatas"])
        ids: List[str] = result.get("ids", []) or []
        docs: List[str] = result.get("documents", []) or []
        metas: List[Any] = result.get("metadatas", []) or []
        meta_dicts: List[dict] = [m if isinstance(m, dict) else {} for m in metas]

        tokenized = [_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized)
        cached = {"ids": ids, "docs": docs, "metas": meta_dicts, "bm25": bm25}
        _BM25_CACHE[key] = cached
        return cached
    except Exception:
        return None


def _bm25_retrieve(
    collection,
    query: str,
    k: int,
    where: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, dict, str, float]]:
    idx = _get_bm25_index(collection)
    if not idx:
        return []

    bm25 = idx["bm25"]
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)
    # Rank all docs; then filter by 'where' and take top-k.
    ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
    rows: List[Tuple[str, dict, str, float]] = []
    for i, score in ranked:
        meta = idx["metas"][i]
        if where and not _meta_matches_where(meta, where):
            continue
        rows.append((str(idx["ids"][i]), meta, str(idx["docs"][i]), float(score)))
        if len(rows) >= k:
            break
    return rows


def _rrf_fuse(
    *,
    semantic: List[Tuple[str, dict, str, float]],
    bm25: List[Tuple[str, dict, str, float]],
    k: int = 60,
) -> List[Tuple[str, dict, str, float]]:
    """
    Reciprocal Rank Fusion over two ranked lists.

    FusedScore(doc) = sum(1 / (k + rank_s(doc))) over lists where doc appears.
    rank is 1-based.
    """
    scores: Dict[str, float] = {}
    meta_by_id: Dict[str, dict] = {}
    doc_by_id: Dict[str, str] = {}

    def _add(list_rows: List[Tuple[str, dict, str, float]]) -> None:
        for rank, (cid, meta, doc, _score) in enumerate(list_rows, start=1):
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
            meta_by_id.setdefault(cid, meta)
            doc_by_id.setdefault(cid, doc)

    _add(semantic)
    _add(bm25)

    fused = [(cid, meta_by_id[cid], doc_by_id[cid], s) for cid, s in scores.items()]
    fused.sort(key=lambda x: x[3], reverse=True)
    return fused

    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    evidences: List[Tuple[dict, str, float]] = []
    for metadata, document, distance in zip(metadatas, documents, distances):
        # ChromaDB cosine distance is in [0, 2]; convert to similarity [0, 1]
        similarity = max(0.0, 1.0 - (distance / 2.0))
        evidences.append((metadata, document, similarity))

    return evidences
