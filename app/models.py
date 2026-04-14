from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SourceDoc(BaseModel):
    source_id: str
    url: str
    title: str
    text: str
    domain: str
    retrieved_at: str
    tier: int
    page_type: Optional[str] = None
    # v2: Content-based tier classification metadata
    tier_reason: Optional[str] = None
    content_score: Optional[float] = None
    content_signals: Optional[Dict[str, object]] = None


class Citation(BaseModel):
    source_id: str
    url: str
    quote: str


class KPIDefinition(BaseModel):
    kpi_id: str
    name: str
    pillar: str
    type: str
    question: str
    rubric: Optional[List[str]] = None
    quant_rule: Optional[Dict[str, object]] = None
    evidence_requirements: Optional[str] = None


class KPIDriver(BaseModel):
    kpi_id: str
    name: str
    question: str
    parent_id: Optional[str] = None


class KPIDriverResult(BaseModel):
    kpi_id: str
    pillar: str
    type: str
    score: float
    confidence: float
    rationale: str
    citations: List[Citation] = Field(default_factory=list)
    details: Optional[Dict[str, object]] = None

    # ── Feature 1: Score splitting ────────────────────────────────────────
    # baseline_score: scored from primary-tier (tier=1) chunks only.
    # live_score:     scored from all retrieved chunks.
    # score_split_delta: live_score − baseline_score.
    baseline_score: Optional[float] = None
    live_score: Optional[float] = None
    score_split_delta: Optional[float] = None

    # ── Feature 2: Scoring distribution (N=5 runs) ───────────────────────
    # Stored under key "scoring_distribution" for LangFuse metadata parity.
    scoring_distribution: Optional[Dict[str, object]] = None

    # ── Feature 3: Quality gate outcomes ─────────────────────────────────
    quality_gates: Optional[Dict[str, object]] = None

    # ── Feature 4: Score change attribution ──────────────────────────────
    score_attribution: Optional[Dict[str, object]] = None

    # ── Feature 7: BERTScore F1 ───────────────────────────────────────────
    bertscore_f1: Optional[float] = None
    low_semantic_grounding: Optional[bool] = None

    # ── Feature 8: Chain-of-thought evaluation ────────────────────────────
    cot_eval: Optional[Dict[str, object]] = None

    # ── Feature 9 & 10: Traceability IDs ─────────────────────────────────
    chromadb_snapshot_id: Optional[str] = None
    prompt_hash: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    langfuse_trace_id: Optional[str] = None


class AggregatedKPIResult(BaseModel):
    pillar: str
    score: float
    confidence: float
    kpis: List[str] = Field(default_factory=list)


# code change for RAG Eval by SN

class RagKpiEval(BaseModel):
    """
    Stores all 9 RAG evaluation check results for a single KPI.
    Each field maps to one of the checks in eval_rag.py.
    None means that check was skipped or unavailable for this KPI.
    """
    kpi_id: str
    # The Score-5 rubric text used as the ideal-answer benchmark
    ground_truth_used: Optional[str] = None

    # Check 1 — RAGAS core metrics (faithfulness, relevancy, precision, recall)
    ragas_faithfulness: Optional[float] = None
    ragas_answer_relevancy: Optional[float] = None
    ragas_context_precision: Optional[float] = None
    ragas_context_recall: Optional[float] = None

    # Check 2 — LLM as a Judge (1-5 scale; feedback is a one-sentence verdict)
    llm_judge_overall: Optional[int] = None
    llm_judge_feedback: Optional[str] = None

    # Check 3 — Recall@k (fraction of relevant sources in top 3 results)
    recall_at_3: Optional[float] = None

    # Check 4 — F1 score (word-level overlap between answer and evidence)
    f1: Optional[float] = None

    # Check 5 — Hallucination (fraction of answer not backed by evidence; flagged if above threshold)
    hallucination_score: Optional[float] = None
    hallucination_flagged: bool = False

    # Check 6 — MMR diversity (how varied the retrieved sources are; higher = less repetition)
    mmr_diversity_score: Optional[float] = None

    # Checks 7, 8, 9 — Ground-truth-based metrics (compared against Score-5 rubric)
    factual_correctness: Optional[float] = None   # Claim-level accuracy vs ideal answer
    noise_sensitivity: Optional[float] = None     # Impact of low-quality sources (lower = better)
    semantic_similarity: Optional[float] = None   # Meaning-level match to ideal answer

    # Golden-chunk retrieval metrics (eval_rag + YAML and/or DB via retrieval_metrics)
    retrieval_hit_rate: Optional[float] = None
    retrieval_mrr: Optional[float] = None
    retrieval_ndcg: Optional[float] = None
    bertscore_f1: Optional[float] = None
    low_semantic_grounding: Optional[bool] = None
    cot_eval: Optional[Dict[str, object]] = None


class RagEvaluationReport(BaseModel):
    """
    Batch-level RAG evaluation summary across all rubric KPIs in a report.
    This is the top-level object written into the final YAML under 'rag_evaluation'.
    """
    # How many rubric KPIs were evaluated (quant KPIs are skipped)
    evaluated_kpi_count: int
    # How many KPIs were flagged due to high hallucination score
    flagged_kpi_count: int
    # List of KPI IDs that were flagged — for easy reference
    flagged_kpi_ids: List[str] = Field(default_factory=list)
    # Plain-English verdict for executives — no jargon
    overall_verdict: str
    # 2-3 line human-readable summary of evidence quality and retrieval performance
    summary: Optional[str] = None
    # Per-KPI detailed results
    per_kpi: List[RagKpiEval] = Field(default_factory=list)

# code change end for RAG Eval by SN


class ReportArtifact(BaseModel):
    run_id: str
    company_name: str
    company_domain: str
    timestamp: str
    url_count: int
    kpi_results: List[KPIDriverResult]
    pillar_scores: List[AggregatedKPIResult]
    overall_score: float
    missing_evidence: List[str]
    debug_log: Optional[List[str]] = None
    # KPI definitions (question = column N) so dashboard can show "KPI Driver" text
    kpi_definitions: Optional[List[Dict]] = None
    # code change for RAG Eval by SN
    # Optional RAG evaluation section — populated when eval_rag node runs
    rag_evaluation: Optional[RagEvaluationReport] = None
    # code change end for RAG Eval by SN
    # Feature 9: collection fingerprint at report time — makes report traceable
    # to exact source data ingested into ChromaDB.
    chromadb_snapshot_id: Optional[str] = None
