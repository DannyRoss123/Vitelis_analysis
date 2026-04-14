from __future__ import annotations

import os
import re
from typing import Dict, List
from uuid import UUID

from sqlalchemy import create_engine, text


def _normalize_uuid_text(value: object) -> str:
    """Match Postgres uuid::text / common hex forms for CAST(... AS TEXT) filters."""
    s = str(value).strip()
    if not s:
        return s
    h = s.replace("-", "")
    if len(h) == 32 and all(c in "0123456789abcdefABCDEF" for c in h):
        try:
            return str(UUID(hex=h))
        except ValueError:
            return s
    return s


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower())
    s = s.strip("_")
    return s or "kpi"


def _pick(cols: set[str], candidates: List[str]) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    return None


def _runs_table_columns(conn) -> set[str]:
    rows = conn.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='runs'"
        )
    ).mappings().all()
    return {r["column_name"] for r in rows}


def _resolve_effective_run_id(conn, db_run_id: str) -> tuple[str, int | None]:
    """
    If db_run_id is a numeric runs.id, map to the UUID (or string) stored in
    kpi_node_scores.run_id — typically runs.request_id, not the integer PK.

    Returns (value to use in CAST(...)= filters, runs.company_id if resolved).
    """
    cols = _runs_table_columns(conn)
    if not cols or "id" not in cols:
        return _normalize_uuid_text(db_run_id), None
    if not db_run_id.isdigit():
        return _normalize_uuid_text(db_run_id), None
    rid = int(db_run_id)
    uuid_col = _pick(cols, ["request_id", "run_uuid", "external_run_id", "pipeline_run_id"])
    if not uuid_col:
        return _normalize_uuid_text(db_run_id), None
    sel = f"SELECT id, company_id, {uuid_col} AS run_uuid FROM runs WHERE id = :rid"
    row = conn.execute(text(sel), {"rid": rid}).mappings().first()
    if not row:
        return _normalize_uuid_text(db_run_id), None
    u = row.get("run_uuid")
    if u is None:
        return _normalize_uuid_text(db_run_id), row.get("company_id")
    eff = _normalize_uuid_text(u)
    return eff, row.get("company_id")


def _table_column_types(conn, table_name: str) -> dict[str, str]:
    rows = conn.execute(
        text(
            "SELECT column_name, data_type, udt_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=:t"
        ),
        {"t": table_name},
    ).mappings().all()
    out: dict[str, str] = {}
    for r in rows:
        c = str(r["column_name"])
        dt = str(r["data_type"] or "")
        udt = str(r["udt_name"] or "")
        if dt.upper() == "USER-DEFINED" and udt:
            out[c] = udt
        else:
            out[c] = dt
    return out


def _run_match_clause(col: str, val: str, dtype: str) -> tuple[str, dict]:
    """Build WHERE fragment and params for matching a run-link column (typed)."""
    d = (dtype or "").lower()
    v = val.strip()
    if v.isdigit() and any(x in d for x in ("smallint", "integer", "bigint", "int2", "int4", "int8")):
        return f"{col} = :rv_bind", {"rv_bind": int(v)}
    if d == "uuid" or d.endswith("uuid"):
        nv = _normalize_uuid_text(v)
        return f"{col}::text = :rv_bind", {"rv_bind": nv}
    if "uuid" in d:
        nv = _normalize_uuid_text(v)
        return f"{col}::text = :rv_bind", {"rv_bind": nv}
    return f"CAST({col} AS TEXT) = :rv_bind", {"rv_bind": v}


def _try_kpi_match_via_runs_subquery(
    conn,
    runs_pk: int,
    kns_cols: set[str],
    company_id: int,
    status_col: str | None,
) -> tuple[bool, List[str], Dict, bool]:
    """
    Match kpi_node_scores.run_id to runs.id or runs.request_id via subquery (avoids
    wrong dtype from information_schema). Returns (ok, where_parts, params, relaxed_company).
    """
    if "run_id" not in kns_cols:
        return False, [], {}, False
    rcols = _runs_table_columns(conn)
    uuid_col = _pick(rcols, ["request_id", "run_uuid", "external_run_id", "pipeline_run_id"])
    frags: List[str] = ["run_id = (SELECT id FROM runs WHERE id = :rid)"]
    if uuid_col:
        frags.append(f"run_id = (SELECT {uuid_col} FROM runs WHERE id = :rid)")
    for with_company in (True, False):
        for frag in frags:
            parts: List[str] = [frag]
            params: Dict = {"rid": runs_pk}
            if with_company and "company_id" in kns_cols:
                parts.append("company_id = :company_id")
                params["company_id"] = company_id
            if status_col:
                parts.append(f"{status_col} = 'success'")
            sql = f"SELECT COUNT(*) FROM kpi_node_scores WHERE {' AND '.join(parts)}"
            try:
                n = int(conn.execute(text(sql), params).scalar() or 0)
            except Exception:
                n = 0
            if n > 0:
                print(
                    f"[load_from_db] kpi_node_scores: matched run via {frag.split()[0]} "
                    f"subquery (company_scoped={with_company})"
                )
                return True, parts, params, not with_company
    return False, [], {}, False


def _run_id_filter_candidates(effective_run_id: str, cli_db_run_id: str) -> List[str]:
    """
    Prefer UUID from runs.request_id, but kpi_node_scores.run_id may store runs.id
    (integer) instead — include the CLI value as a second try.
    """
    out: List[str] = []
    seen: set[str] = set()
    for x in (effective_run_id, cli_db_run_id):
        x = str(x).strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _choose_run_column(
    conn,
    table: str,
    columns: set[str],
    db_run_ids: List[str],
    column_types: dict[str, str],
    company_col: str | None = None,
    company_id: str | int | None = None,
) -> tuple[str | None, str | None, bool]:
    """
    Pick the best (column, filter value) by COUNT over run-link columns.
    Uses typed predicates (integer / uuid / text). Tries with company filter first;
    if no rows, retries without company (kpi rows may omit or differ on company_id).

    Returns (column_name, winning_db_run_id, company_filter_relaxed).
    """
    candidates = ["recd_id", "report_run_id", "run_ref_id", "run_id"]
    candidates = [c for c in candidates if c in columns]
    if not candidates:
        return None, None, False

    def _try(with_company: bool) -> tuple[str | None, str | None, int]:
        best_col = None
        best_val: str | None = None
        best_count = -1
        for db_run_id in db_run_ids:
            for col in candidates:
                frag, rp = _run_match_clause(col, db_run_id, column_types.get(col, ""))
                where = [frag]
                params = dict(rp)
                if with_company and company_col and company_id is not None:
                    where.append(f"{company_col} = :company_id")
                    params["company_id"] = company_id
                sql = f"SELECT COUNT(*) AS n FROM {table} WHERE {' AND '.join(where)}"
                n = conn.execute(text(sql), params).scalar() or 0
                if int(n) > best_count:
                    best_count = int(n)
                    best_col = col
                    best_val = db_run_id
        return best_col, best_val, best_count

    col, val, cnt = _try(with_company=True)
    if cnt > 0:
        return col, val, False

    col, val, cnt = _try(with_company=False)
    if cnt > 0:
        return col, val, True

    return None, None, False


def load_from_db_node(state: Dict) -> Dict:
    """
    Load real sources + KPI scores/answers from Postgres for a run/company.

    Required state:
      - run_id: logical pipeline run id (for output filenames only)
      - db_run_id: DB run identifier (string/int/uuid) for filtering
      - company_name: display name for output
    """
    db_url = (os.getenv("DATABASE_URL") or "").strip()
    if not db_url:
        raise RuntimeError("DATABASE_URL is required for DB-backed eval.")

    db_run_id = str(state.get("db_run_id", "")).strip()
    if not db_run_id:
        raise RuntimeError("db_run_id is required.")

    company_name = str(state.get("company_name") or "").strip()
    if not company_name:
        raise RuntimeError("company_name is required.")

    engine = create_engine(db_url)

    with engine.connect() as conn:
        effective_run_id, runs_company_id = _resolve_effective_run_id(conn, db_run_id)
        if effective_run_id != db_run_id:
            print(
                f"[load_from_db] resolved runs.id={db_run_id} -> "
                f"filter run key={effective_run_id!r}"
            )

        # Discover company_id by name
        comp_rows = conn.execute(
            text("SELECT id, name FROM companies WHERE LOWER(name) = LOWER(:name) LIMIT 1"),
            {"name": company_name},
        ).mappings().all()
        if not comp_rows:
            comp_rows = conn.execute(
                text("SELECT id, name FROM companies ORDER BY id LIMIT 1")
            ).mappings().all()
        if not comp_rows:
            raise RuntimeError("No companies found in DB.")
        company_id = comp_rows[0]["id"]
        company_name_db = comp_rows[0]["name"]

        if runs_company_id is not None and int(runs_company_id) != int(company_id):
            print(
                f"[load_from_db] WARNING: company_name {company_name!r} -> company_id={company_id}, "
                f"but runs.id={db_run_id} has company_id={runs_company_id}; using run's company."
            )
            row_c = conn.execute(
                text("SELECT id, name FROM companies WHERE id = :cid LIMIT 1"),
                {"cid": runs_company_id},
            ).mappings().first()
            if row_c:
                company_id = row_c["id"]
                company_name_db = row_c["name"]

        # Column metadata
        src_cols_rows = conn.execute(
            text("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='sources'")
        ).mappings().all()
        src_cols = {r["column_name"] for r in src_cols_rows}
        kns_cols_rows = conn.execute(
            text("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='kpi_node_scores'")
        ).mappings().all()
        kns_cols = {r["column_name"] for r in kns_cols_rows}
        src_types = _table_column_types(conn, "sources")
        kns_types = _table_column_types(conn, "kpi_node_scores")

        src_url_col = _pick(src_cols, ["url"])
        src_text_col = _pick(src_cols, ["content", "text", "body"])
        src_tier_col = _pick(src_cols, ["tier"])
        src_created_col = _pick(src_cols, ["created_at", "updated_at"])

        if not src_url_col or not src_text_col:
            raise RuntimeError("`sources` must include URL and text/content columns.")

        run_candidates = _run_id_filter_candidates(effective_run_id, db_run_id)

        src_filters = ["1=1"]
        src_params = {}
        if "company_id" in src_cols:
            src_filters.append("company_id = :company_id")
            src_params["company_id"] = company_id
        src_run_col, src_run_val, _src_relaxed = _choose_run_column(
            conn,
            table="sources",
            columns=src_cols,
            db_run_ids=run_candidates,
            column_types=src_types,
            company_col="company_id" if "company_id" in src_cols else None,
            company_id=company_id if "company_id" in src_cols else None,
        )
        if src_run_col and src_run_val is not None:
            sf, srp = _run_match_clause(src_run_col, src_run_val, src_types.get(src_run_col, ""))
            src_filters.append(sf)
            src_params.update(srp)

        src_sql = (
            f"SELECT {src_url_col} AS url, {src_text_col} AS text"
            + (f", {src_tier_col} AS tier" if src_tier_col else ", 2 AS tier")
            + (f", {src_created_col} AS retrieved_at" if src_created_col else ", NULL AS retrieved_at")
            + f" FROM sources WHERE {' AND '.join(src_filters)} LIMIT 5000"
        )
        src_rows = conn.execute(text(src_sql), src_params).mappings().all()
        print(
            f"[load_from_db] sources rows={len(src_rows)} "
            f"(company_id={company_id}, run_col={src_run_col or 'none'}, "
            f"run_filter={src_run_val or effective_run_id})"
        )

        sources: List[Dict] = []
        for i, r in enumerate(src_rows):
            content_text = (r.get("text") or "").strip()
            url = (r.get("url") or "").strip()
            if not content_text or not url:
                continue
            sources.append(
                {
                    "source_id": f"dbsrc_{i}",
                    "url": url,
                    "title": url,
                    "text": content_text,
                    "domain": "",
                    "retrieved_at": str(r.get("retrieved_at") or ""),
                    "tier": int(r.get("tier") or 2),
                }
            )

        # KPI scores/answers from kpi_node_scores
        title_col = _pick(kns_cols, ["node_title", "title", "name"])
        score_col = _pick(kns_cols, ["score"])
        status_col = _pick(kns_cols, ["status"])
        rationale_col = _pick(kns_cols, ["rationale", "answer", "explanation", "reasoning"])
        conf_col = _pick(kns_cols, ["confidence"])

        if not title_col or not score_col:
            raise RuntimeError("`kpi_node_scores` must include node_title/name and score.")

        kns_filters = ["1=1"]
        kns_params: Dict = {}
        kns_run_col, kns_run_val, kns_relaxed_company = _choose_run_column(
            conn,
            table="kpi_node_scores",
            columns=kns_cols,
            db_run_ids=run_candidates,
            column_types=kns_types,
            company_col="company_id" if "company_id" in kns_cols else None,
            company_id=company_id if "company_id" in kns_cols else None,
        )
        kns_from_runs_subquery = False
        if not kns_run_col and db_run_id.isdigit():
            ok, parts, rp, relaxed_rs = _try_kpi_match_via_runs_subquery(
                conn,
                int(db_run_id),
                kns_cols,
                int(company_id),
                status_col,
            )
            if ok:
                kns_from_runs_subquery = True
                kns_filters = parts
                kns_params = rp
                kns_relaxed_company = relaxed_rs
                kns_run_col = "run_id"
                kns_run_val = "via_runs_subquery"

        if kns_relaxed_company and not kns_from_runs_subquery:
            print(
                "[load_from_db] kpi_node_scores: matched run_id without company_id filter "
                "(rows may not match selected company in DB)."
            )

        kns_via_kar = False
        kar_relaxed_company = False
        kar_try_val = ""
        kar_types: dict[str, str] = {}
        if kns_from_runs_subquery:
            pass
        elif kns_run_col and kns_run_val is not None:
            if "company_id" in kns_cols and not kns_relaxed_company:
                kns_filters.append("company_id = :company_id")
                kns_params["company_id"] = company_id
            if status_col:
                kns_filters.append(f"{status_col} = 'success'")
            kf, krp = _run_match_clause(kns_run_col, kns_run_val, kns_types.get(kns_run_col, ""))
            kns_filters.append(kf)
            kns_params.update(krp)
        else:
            kar_cols_rows = conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='public' AND table_name='kpi_analysis_runs'"
                )
            ).mappings().all()
            kar_cols = {r["column_name"] for r in kar_cols_rows}
            kar_types = _table_column_types(conn, "kpi_analysis_runs")
            if (
                "kpi_analysis_run_id" in kns_cols
                and {"id", "run_id"}.issubset(kar_cols)
            ):

                def _try_kar_count(cand: str, with_company: bool) -> int:
                    frag, rp = _run_match_clause("kar.run_id", cand, kar_types.get("run_id", ""))
                    parts = [frag]
                    params = dict(rp)
                    if with_company and "company_id" in kns_cols:
                        parts.append("kns.company_id = :company_id")
                        params["company_id"] = company_id
                    if status_col:
                        parts.append(f"kns.{status_col} = 'success'")
                    sql_try = (
                        "SELECT COUNT(*) FROM kpi_node_scores kns "
                        "JOIN kpi_analysis_runs kar ON kar.id = kns.kpi_analysis_run_id "
                        f"WHERE {' AND '.join(parts)}"
                    )
                    return int(conn.execute(text(sql_try), params).scalar() or 0)

                n_kar = 0
                kar_try_val = run_candidates[0] if run_candidates else ""
                kar_relaxed_company = False
                for cand in run_candidates:
                    n_kar = _try_kar_count(cand, with_company=True)
                    if n_kar > 0:
                        kar_try_val = cand
                        break
                if n_kar <= 0:
                    for cand in run_candidates:
                        n_kar = _try_kar_count(cand, with_company=False)
                        if n_kar > 0:
                            kar_try_val = cand
                            kar_relaxed_company = True
                            print(
                                "[load_from_db] kpi_analysis_runs: matched run_id without "
                                "company_id filter."
                            )
                            break
                if n_kar > 0:
                    kns_via_kar = True
                else:
                    raise RuntimeError(
                        "No matching run-link found in kpi_node_scores (or "
                        "kpi_analysis_runs) for "
                        f"candidates={run_candidates!r} (cli db_run_id={db_run_id!r}). "
                        "Direct candidates: "
                        f"{[c for c in ['recd_id','report_run_id','run_ref_id','run_id'] if c in kns_cols]}"
                    )
            else:
                raise RuntimeError(
                    "No matching run-link found in kpi_node_scores for "
                    f"candidates={run_candidates!r} (cli db_run_id={db_run_id!r}). "
                    "Available candidates: "
                    f"{[c for c in ['recd_id','report_run_id','run_ref_id','run_id'] if c in kns_cols]}"
                )

        if kns_via_kar:
            kfrag, krp = _run_match_clause(
                "kar.run_id", kar_try_val, kar_types.get("run_id", "")
            )
            kar_where = [kfrag]
            kns_params.update(krp)
            if "company_id" in kns_cols and not kar_relaxed_company:
                kar_where.append("kns.company_id = :company_id")
                kns_params["company_id"] = company_id
            if status_col:
                kar_where.append(f"kns.{status_col} = 'success'")
            kns_sql = (
                f"SELECT kns.{title_col} AS node_title, kns.{score_col} AS score"
                + (
                    f", kns.{rationale_col} AS rationale"
                    if rationale_col
                    else ", NULL AS rationale"
                )
                + (f", kns.{conf_col} AS confidence" if conf_col else ", NULL AS confidence")
                + " FROM kpi_node_scores kns "
                + "JOIN kpi_analysis_runs kar ON kar.id = kns.kpi_analysis_run_id "
                + f"WHERE {' AND '.join(kar_where)}"
            )
        else:
            kns_sql = (
                f"SELECT {title_col} AS node_title, {score_col} AS score"
                + (f", {rationale_col} AS rationale" if rationale_col else ", NULL AS rationale")
                + (f", {conf_col} AS confidence" if conf_col else ", NULL AS confidence")
                + f" FROM kpi_node_scores WHERE {' AND '.join(kns_filters)}"
            )
        if kns_via_kar and kar_relaxed_company and "company_id" in kns_params:
            kns_params.pop("company_id", None)
        kns_rows = conn.execute(text(kns_sql), kns_params).mappings().all()
        print(
            f"[load_from_db] kpi rows={len(kns_rows)} "
            f"(company_id={company_id}, run_col={kns_run_col or ('kar.run_id' if kns_via_kar else '?')}, "
            f"via_kpi_analysis_runs={kns_via_kar}, run_filter={kns_run_val or kns_params.get('rv_bind')})"
        )

    kpi_results: List[Dict] = []
    kpi_definitions: List[Dict] = []
    seen = set()
    for idx, row in enumerate(kns_rows, start=1):
        title = str(row.get("node_title") or "").strip()
        if not title:
            continue
        kid = _slug(title)
        if kid in seen:
            kid = f"{kid}_{idx}"
        seen.add(kid)
        score = float(row.get("score") or 0)
        conf = row.get("confidence")
        confidence = float(conf) if conf is not None else 0.7
        rationale = str(row.get("rationale") or f"DB imported KPI answer for: {title}")
        kpi_results.append(
            {
                "kpi_id": kid,
                "pillar": "DB Imported",
                "type": "rubric",
                "score": max(1.0, min(5.0, score)),
                "confidence": max(0.0, min(1.0, confidence)),
                "rationale": rationale,
                "citations": [],
                "details": {"source": "db.kpi_node_scores", "node_title": title},
                "_name": title,
            }
        )
        kpi_definitions.append(
            {
                "kpi_id": kid,
                "name": title,
                "pillar": "DB Imported",
                "type": "rubric",
                "question": title,
                "rubric": None,
                "quant_rule": None,
                "evidence_requirements": None,
            }
        )

    company_domain = _slug(company_name_db)
    return {
        "company_name": company_name_db,
        "company_domain": company_domain,
        "sources": sources,
        "target_urls": [s["url"] for s in sources],
        "url_count": len(sources),
        "kpi_results": kpi_results,
        "missing_evidence": [],
        "kpi_definitions": kpi_definitions,
        "company_folder": "",  # disables GT compare in eval_aggregate_report
    }

