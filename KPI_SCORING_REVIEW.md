# KPI Scoring — Current Flow & Recommendations

## Current flow (rubric KPIs)

1. **Retrieval**
   - `determine_optimal_k(kpi)` → k = 6, 8, or 10 (if `VITELIS_ENABLE_DYNAMIC_K`).
   - `retrieve_evidence_weighted(collection, kpi.question, k)` (or plain `retrieve_evidence`):
     - ChromaDB query with **hash-based embeddings** (no semantic model).
     - Over-fetch 2×k, apply tier boost, return top k chunks.
   - Each chunk = **500 characters** from a source; multiple chunks per URL.

2. **Evidence to LLM**
   - Only **first 300 chars** of each chunk are sent in the prompt (`doc[:300]`).
   - Prompt = KPI name, question, **full rubric** (1/2/3/4/5 lines), and evidence list.

3. **LLM call**
   - Single request: "Score 1–5 using rubric and evidence. Return JSON: score, confidence, rationale, citations."
   - Model: `OPENAI_MODEL` (default `gpt-4o-mini`).
   - On failure → **fallback**: count positive/negative keywords in evidence blob; no rubric used.

4. **Post-processing**
   - `calculate_enhanced_confidence()`: tier boost, semantic corroboration, freshness, authority, contradictions, diversity, citation penalty, low-evidence penalty.

---

## Inefficiencies / gaps (especially with PM-curated URLs + KPI CSV)

| Issue | Impact |
|-------|--------|
| **300 chars per chunk in prompt** | LLM sees truncated evidence; may miss key facts. |
| **Hash embeddings** | No real semantic similarity; "AI strategy" vs "strategy for AI" don’t align well. |
| **Fallback ignores rubric** | If API fails, score is generic positive/negative word count, not "how close to Score 5". |
| **Score-5 not explicit in prompt** | LLM infers "ideal" from rubric; making "Quality Criteria = 5 (High)" explicit would align scoring. |
| **One LLM call per KPI** | 80+ KPIs ⇒ 80+ round-trips; batching could reduce latency/cost. |
| **Chunk size 500** | Can split mid-sentence; with curated sources, larger chunks or full-doc could help. |

---

## Recommendations (prioritized)

### 1. Use Quality Criteria = 5 (High) explicitly in the prompt (high impact, low effort)

- Extract the Score-5 line from the rubric (same as RAG ground truth).
- In the prompt, add: *"Ideal (5): <Score-5 text>. Score how close the evidence supports this level."*
- Aligns scoring with the same standard used later in RAG evaluation.

### 2. Send more evidence per chunk (high impact, low effort)

- Increase from `doc[:300]` to `doc[:500]` or `doc[:600]` so the LLM sees full chunks (chunks are 500 chars).
- Reduces truncation and improves accuracy.

### 3. Smarter fallback when LLM is unavailable (medium impact, low effort)

- Instead of only positive/negative keyword counts, compute overlap (e.g. token or simple embedding) between evidence and the **Score-5 rubric text**.
- Map overlap bands to 1–5 (e.g. high overlap ⇒ 4–5, low ⇒ 1–2) so fallback is rubric-aware.

### 4. Optional: larger chunks or full-document retrieval for small corpora (medium effort)

- With 50–70 curated URLs, consider 800–1200 char chunks or "one document per source" retrieval so key paragraphs aren’t split.
- Requires changes in `vectorstore.chunk_text()` and/or retrieval logic.

### 5. Optional: semantic embeddings (higher effort)

- Replace or augment hash embeddings with OpenAI embeddings or a small open-source model for better "KPI question ↔ evidence" matching.
- Bigger gain when sources are not pre-curated per KPI.

### 6. Optional: batch LLM calls (medium effort)

- Send e.g. 3–5 KPIs per request with a clear JSON schema; parse and split results.
- Reduces latency and can reduce cost; needs careful prompt design and error handling.

---

## Summary

With **specific URLs and KPI CSV**, the main levers are:

- **Better use of the rubric** (explicit Score-5 in prompt + rubric-aware fallback).
- **More context to the LLM** (full 500-char chunks, or more).
- **Optional**: larger chunks or full-doc retrieval, semantic embeddings, batched LLM calls.

Implementing **1 + 2 + 3** gives the largest benefit for the least change to the current architecture.
