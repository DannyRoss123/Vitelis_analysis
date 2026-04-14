-- Golden retrieval labels for Hit Rate / MRR / nDCG (see app/retrieval_metrics.py).
--
-- Apply (no psql required — uses DATABASE_URL from RAG-Evaluation/.env):
--   python scripts/apply_golden_chunks_sql.py
--
-- Or with psql, if installed:
--   psql "$DATABASE_URL" -f sql/golden_chunks.sql

CREATE TABLE IF NOT EXISTS public.golden_chunks (
    kpi_id    TEXT NOT NULL,
    chunk_id  TEXT NOT NULL,
    relevance DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    PRIMARY KEY (kpi_id, chunk_id)
);

COMMENT ON TABLE public.golden_chunks IS
    'Ground-truth chunk IDs per KPI for retrieval metrics. chunk_id must match eval e.g. source_id::chunk_0.';

CREATE INDEX IF NOT EXISTS idx_golden_chunks_kpi_id ON public.golden_chunks (kpi_id);

-- Example (replace with real kpi_id and chunk_id from your run):
-- INSERT INTO public.golden_chunks (kpi_id, chunk_id, relevance) VALUES
--   ('ai_literacy_and_knowledge_sharing', 'a1b2c3d4::chunk_0', 2.0),
--   ('ai_literacy_and_knowledge_sharing', 'e5f6g7h8::chunk_0', 1.0)
-- ON CONFLICT (kpi_id, chunk_id) DO UPDATE SET relevance = EXCLUDED.relevance;
