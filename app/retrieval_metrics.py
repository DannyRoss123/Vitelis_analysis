"""
Retrieval quality metrics for the Vitelis KPI benchmarking pipeline.

Computes three standard IR metrics for each KPI retrieval call, comparing
the returned chunk IDs against a golden (ground-truth) set in PostgreSQL:

  - Hit Rate  : fraction of golden chunks that appear anywhere in top-k results
  - MRR       : reciprocal rank of the first golden chunk in the ranked list
  - nDCG      : normalised discounted cumulative gain using per-chunk relevance labels

Source (env GOLDEN_CHUNKS_SOURCE)
---------------------------------
  - db    : default. DATABASE_URL + table GOLDEN_CHUNKS_TABLE (default golden_chunks)
  - yaml  : optional legacy file only if GOLDEN_CHUNKS_PATH is set (no auto-loaded repo file)
  - both  : merge file + DB; same chunk_id for a KPI keeps DB relevance

DB table (example DDL)
----------------------
    CREATE TABLE IF NOT EXISTS golden_chunks (
        kpi_id    TEXT NOT NULL,
        chunk_id  TEXT NOT NULL,
        relevance DOUBLE PRECISION DEFAULT 1.0,
        PRIMARY KEY (kpi_id, chunk_id)
    );

Override column names with GOLDEN_CHUNKS_KPI_COLUMN, GOLDEN_CHUNKS_CHUNK_COLUMN,
GOLDEN_CHUNKS_RELEVANCE_COLUMN (set relevance column to empty to force 1.0).

The chunk_id values must match Chroma-style IDs: "{source_id}::chunk_{idx}".

If no golden data is available for a KPI, Hit/MRR/nDCG return None.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional

import yaml  # already in requirements.txt (pyyaml)

# Module-level cache so golden data is resolved at most once per process.
_golden_cache: Optional[Dict[str, Any]] = None
_golden_load_attempted: bool = False

_IDENT_OK = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _sql_ident(name: str) -> Optional[str]:
    if not name or not _IDENT_OK.match(name):
        return None
    return name


def _load_golden_yaml_file() -> Dict[str, Any]:
    """Load YAML only when GOLDEN_CHUNKS_PATH is set explicitly (no repo placeholder file)."""
    path = (os.getenv("GOLDEN_CHUNKS_PATH") or "").strip()
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _normalize_yaml_shape(raw: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Ensure kpi_id -> list of {chunk_id, relevance}."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for kpi_id, entries in (raw or {}).items():
        if not isinstance(entries, list):
            continue
        norm: List[Dict[str, Any]] = []
        for e in entries:
            if not isinstance(e, dict) or "chunk_id" not in e:
                continue
            norm.append(
                {
                    "chunk_id": str(e["chunk_id"]),
                    "relevance": float(e.get("relevance", 1)),
                }
            )
        if norm:
            out[str(kpi_id)] = norm
    return out


def _load_golden_from_db() -> Dict[str, List[Dict[str, Any]]]:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        return {}

    schema = _sql_ident((os.getenv("GOLDEN_CHUNKS_SCHEMA") or "public").strip()) or "public"
    table = _sql_ident((os.getenv("GOLDEN_CHUNKS_TABLE") or "golden_chunks").strip()) or "golden_chunks"
    kcol = _sql_ident((os.getenv("GOLDEN_CHUNKS_KPI_COLUMN") or "kpi_id").strip()) or "kpi_id"
    ccol = _sql_ident((os.getenv("GOLDEN_CHUNKS_CHUNK_COLUMN") or "chunk_id").strip()) or "chunk_id"
    rel_raw = (os.getenv("GOLDEN_CHUNKS_RELEVANCE_COLUMN") or "relevance").strip()
    rcol = _sql_ident(rel_raw) if rel_raw else None

    try:
        from sqlalchemy import create_engine, text
    except Exception:
        return {}

    sel_rel = f'COALESCE("{rcol}"::double precision, 1.0) AS relevance' if rcol else "1.0::double precision AS relevance"
    sql = text(
        f'SELECT "{kcol}" AS kpi_id, "{ccol}" AS chunk_id, {sel_rel} '
        f'FROM "{schema}"."{table}"'
    )

    out: Dict[str, List[Dict[str, Any]]] = {}
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            rows = conn.execute(sql).mappings().all()
        for row in rows:
            kid = str(row.get("kpi_id") or "").strip()
            cid = str(row.get("chunk_id") or "").strip()
            if not kid or not cid:
                continue
            rel = row.get("relevance")
            try:
                rel_f = float(rel) if rel is not None else 1.0
            except (TypeError, ValueError):
                rel_f = 1.0
            out.setdefault(kid, []).append({"chunk_id": cid, "relevance": rel_f})
    except Exception:
        return {}

    return out


def _merge_golden(
    yaml_map: Dict[str, List[Dict[str, Any]]],
    db_map: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Merge per KPI; later duplicate chunk_ids in db_map override relevance."""
    merged: Dict[str, Dict[str, float]] = {}
    for source in (yaml_map, db_map):
        for kpi_id, entries in source.items():
            bucket = merged.setdefault(kpi_id, {})
            for e in entries:
                cid = e["chunk_id"]
                bucket[cid] = float(e.get("relevance", 1))
    return {k: [{"chunk_id": c, "relevance": r} for c, r in v.items()] for k, v in merged.items()}


def reset_golden_cache() -> None:
    """Clear cached golden data (e.g. after tests or DB reload)."""
    global _golden_cache, _golden_load_attempted
    _golden_cache = None
    _golden_load_attempted = False


def _load_golden_chunks() -> Dict[str, Any]:
    """
    Load golden chunks from DB and/or optional YAML and cache the merged structure.

    GOLDEN_CHUNKS_SOURCE: db (default) | yaml | both
    """
    global _golden_cache, _golden_load_attempted

    if _golden_load_attempted:
        return _golden_cache or {}

    _golden_load_attempted = True
    source = (os.getenv("GOLDEN_CHUNKS_SOURCE") or "db").strip().lower()
    if source not in ("yaml", "db", "both"):
        source = "db"

    yaml_map: Dict[str, List[Dict[str, Any]]] = {}
    db_map: Dict[str, List[Dict[str, Any]]] = {}

    if source in ("yaml", "both"):
        yaml_map = _normalize_yaml_shape(_load_golden_yaml_file())
    if source in ("db", "both"):
        db_map = _load_golden_from_db()

    if source == "yaml":
        merged_lists = yaml_map
    elif source == "db":
        merged_lists = db_map
    else:
        merged_lists = _merge_golden(yaml_map, db_map)

    # Cache in YAML-shaped dict (kpi -> list of dicts) for get_golden_ids_for_kpi
    _golden_cache = {k: list(v) for k, v in merged_lists.items()}
    return _golden_cache


def get_golden_ids_for_kpi(kpi_id: str) -> Optional[Dict[str, float]]:
    """
    Return the golden chunk relevance map for a KPI.

    Args:
        kpi_id: KPI identifier (matches DB golden_chunks.kpi_id or YAML key if used).

    Returns:
        Dict mapping chunk_id -> relevance score, or None if the KPI
        has no golden set defined.
    """
    data = _load_golden_chunks()
    entries = data.get(kpi_id, [])
    if not entries:
        return None
    return {
        e["chunk_id"]: float(e.get("relevance", 1))
        for e in entries
        if "chunk_id" in e
    }


def compute_hit_rate(kpi_id: str, retrieved_ids: List[str]) -> Optional[float]:
    """
    Hit Rate: fraction of golden chunks that appear anywhere in the top-k list.

    A hit is counted for each unique golden chunk that appears at any rank
    in the retrieved list. The denominator is the total number of golden
    chunks for this KPI.

    Args:
        kpi_id: KPI identifier for golden set lookup.
        retrieved_ids: Ordered list of chunk IDs returned by retrieval
            (index 0 = rank 1).

    Returns:
        Float in [0.0, 1.0], or None if no golden set is available.
    """
    golden = get_golden_ids_for_kpi(kpi_id)
    if golden is None:
        return None

    retrieved_set = set(retrieved_ids)
    hits = sum(1 for cid in golden if cid in retrieved_set)
    return round(hits / len(golden), 4)


def compute_mrr(kpi_id: str, retrieved_ids: List[str]) -> Optional[float]:
    """
    MRR: reciprocal rank of the *first* golden chunk in the ranked list.

    Args:
        kpi_id: KPI identifier for golden set lookup.
        retrieved_ids: Ordered list of chunk IDs (index 0 = rank 1).

    Returns:
        Float in (0.0, 1.0], 0.0 if no golden chunk appears in the list,
        or None if no golden set is available.
    """
    golden = get_golden_ids_for_kpi(kpi_id)
    if golden is None:
        return None

    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in golden:
            return round(1.0 / rank, 4)
    return 0.0


def compute_ndcg(kpi_id: str, retrieved_ids: List[str]) -> Optional[float]:
    """
    nDCG: normalised discounted cumulative gain using golden relevance labels.

    The ideal ranking places all golden chunks (sorted by descending relevance)
    at the top. Retrieved chunks not in the golden set contribute 0 relevance.

    Args:
        kpi_id: KPI identifier for golden set lookup.
        retrieved_ids: Ordered list of chunk IDs (index 0 = rank 1).

    Returns:
        Float in [0.0, 1.0], or None if no golden set is available.
    """
    golden = get_golden_ids_for_kpi(kpi_id)
    if golden is None:
        return None

    # DCG of the actual retrieved ranking
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        rel = golden.get(cid, 0.0)
        if rel > 0:
            dcg += rel / math.log2(rank + 1)

    # Ideal DCG: golden chunks sorted by relevance desc, placed at ranks 1..n
    ideal_rels = sorted(golden.values(), reverse=True)
    idcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(ideal_rels, start=1)
        if rel > 0
    )

    if idcg == 0:
        return 0.0
    return round(dcg / idcg, 4)


def compute_all_retrieval_metrics(
    kpi_id: str,
    retrieved_evidences: List[Any],
) -> Dict[str, Optional[float]]:
    """
    Compute Hit Rate, MRR, and nDCG in a single call.

    Extracts chunk IDs from a list of (metadata, document, score) tuples
    as returned by vectorstore.retrieve_evidence / reranker.rerank.

    Args:
        kpi_id: KPI identifier for golden set lookup.
        retrieved_evidences: List of (metadata_dict, doc_str, score_float)
            tuples in retrieval rank order.

    Returns:
        Dict with keys "hit_rate", "mrr", "ndcg", each a float or None.
    """
    # Build the ordered list of chunk IDs.
    # ChromaDB stores individual chunk IDs in the query result "ids" field,
    # but retrieve_evidence returns (metadata, doc, score) without the raw ID.
    # The chunk ID follows the pattern "{source_id}::chunk_{idx}"; we
    # reconstruct it from metadata where possible, falling back to source_id.
    chunk_ids: List[str] = []
    for item in retrieved_evidences:
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            continue
        meta = item[0] if isinstance(item[0], dict) else {}
        # Prefer a stored chunk_id field; fall back to source_id
        cid = meta.get("chunk_id") or meta.get("source_id", "")
        chunk_ids.append(str(cid))

    return {
        "hit_rate": compute_hit_rate(kpi_id, chunk_ids),
        "mrr": compute_mrr(kpi_id, chunk_ids),
        "ndcg": compute_ndcg(kpi_id, chunk_ids),
    }
