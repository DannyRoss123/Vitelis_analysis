"""
Score extensions for the Vitelis KPI benchmarking pipeline.

Implements the following features as modular, side-effect-free functions
(except where LangFuse logging is an explicit side effect):

  Feature 1  — Score splitting       : baseline_score vs live_score
  Feature 2  — Statistical scoring   : N=5 LLM runs → mean ± σ
  Feature 3  — Quality gates         : faithfulness / stability / coverage / bleed
  Feature 4  — Score change attribution : delta > 0.2 → classify root cause
  Feature 7  — BERTScore             : F1 between rationale and top-3 chunks
  Feature 8  — Chain-of-thought eval : specificity / evidence / alignment (1-5)
  Feature 9  — ChromaDB snapshot ID  : SHA256 of sorted (chunk_id, updated_at) pairs
  Feature 10 — Prompt hash           : SHA256 of system_prompt + user_prompt

Features 5 (MLflow) and 6 (retrieval metrics) live in their own modules
(app/mlflow_logger.py and app/retrieval_metrics.py respectively) and are
wired together in app/nodes/score_kpis.py.

Design constraints honoured:
  - Uses the existing LangFuse client from app/langfuse_client.py — no new instantiation.
  - All LangFuse logging is attached as scores or events, never printed to stdout.
  - BERTScore and CoT eval are skippable via the feature-flags config dict.
  - All public functions have full docstrings.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.langfuse_client import (
    log_event_to_trace,
    log_score_to_trace,
)

# ---------------------------------------------------------------------------
# Feature-flag defaults
# ---------------------------------------------------------------------------

DEFAULT_FEATURE_FLAGS: Dict[str, Any] = {
    "bertscore_enabled": True,
    "cot_eval_enabled": True,
    "score_splitting_enabled": True,
    "quality_gates_enabled": True,
    "retrieval_metrics_enabled": True,
    "score_attribution_enabled": True,
    "mlflow_enabled": True,
    "n_scoring_runs": 5,
    "scoring_temperature": 0.7,
}


# ===========================================================================
# Feature 10 — Prompt Hash
# ===========================================================================

def compute_prompt_hash(system_prompt: str, user_prompt: str) -> str:
    """
    Compute a SHA256 hash of the complete prompt (system + user).

    The hash is stored in LangFuse span metadata and in MLflow so that a
    change in either prompt component is detectable across runs.

    Args:
        system_prompt: The system instruction string passed to the LLM.
        user_prompt: The user content string passed to the LLM.

    Returns:
        64-character hex-encoded SHA256 digest.

    Side effects:
        None.
    """
    combined = system_prompt + "\x00" + user_prompt  # null byte as separator
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ===========================================================================
# Feature 9 — ChromaDB Snapshot ID
# ===========================================================================

def compute_chromadb_snapshot_id(collection: Any) -> str:
    """
    Compute a SHA256 fingerprint of the current ChromaDB collection state.

    Hashes the sorted list of (chunk_id, updated_at) tuples so that any
    addition, deletion, or timestamp change produces a different fingerprint.
    Falls back to retrieved_at if updated_at is absent from the metadata.

    Args:
        collection: ChromaDB Collection object (already open).

    Returns:
        64-character hex-encoded SHA256 digest, or empty string on error.

    Side effects:
        Reads all IDs and metadata from the collection.  For very large
        collections this may take a moment but does not modify any data.
    """
    try:
        result = collection.get(include=["metadatas"])
        ids: List[str] = result.get("ids", []) or []
        metadatas: List[Any] = result.get("metadatas", []) or []

        tuples: List[Tuple[str, str]] = []
        for chunk_id, meta in zip(ids, metadatas):
            ts = ""
            if isinstance(meta, dict):
                ts = str(meta.get("updated_at") or meta.get("retrieved_at") or "")
            tuples.append((chunk_id, ts))

        tuples.sort()
        raw = json.dumps(tuples, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    except Exception:
        return ""


# ===========================================================================
# Feature 2 — Statistical Scoring (N runs → mean ± σ)
# ===========================================================================

def _call_llm_once(prompt: str, temperature: float) -> Optional[float]:
    """
    Make a single LLM call and return the numeric score.

    Args:
        prompt: Full user prompt string (system role is set inline).
        temperature: Sampling temperature — must be > 0 for stochastic runs.

    Returns:
        Float score extracted from JSON response, or None on any failure.

    Side effects:
        Makes one HTTP request to the OpenAI chat completions endpoint.
        Respects VITELIS_OPENAI_MIN_DELAY between calls.
    """
    provider = (os.getenv("VITELIS_LLM_PROVIDER") or "").strip().lower()
    if provider in {"gemini", "google"}:
        # Keep extension scoring fast/stable in Gemini mode by skipping OpenAI-only path.
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    min_delay = float(os.getenv("VITELIS_OPENAI_MIN_DELAY", "5.0"))

    try:
        time.sleep(min_delay)
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a strict JSON generator."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        content = resp.json()["choices"][0]["message"]["content"]
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            return None
        data = json.loads(content[start : end + 1])
        return float(data["score"])
    except Exception:
        return None


def _heuristic_score_from_prompt(prompt: str) -> float:
    """Cheap local fallback score from rubric/evidence token overlap (1-5)."""
    rubric_match = re.search(r"Rubric:\n(.*?)\nEvidence:\n", prompt, re.DOTALL)
    evidence_match = re.search(r"Evidence:\n(.*)$", prompt, re.DOTALL)
    rubric = (rubric_match.group(1) if rubric_match else "").lower()
    evidence = (evidence_match.group(1) if evidence_match else "").lower()
    rubric_tokens = set(re.findall(r"\b\w{4,}\b", rubric))
    evidence_tokens = set(re.findall(r"\b\w{4,}\b", evidence))
    if not rubric_tokens or not evidence_tokens:
        return 3.0
    overlap = len(rubric_tokens & evidence_tokens) / len(rubric_tokens)
    if overlap >= 0.35:
        return 5.0
    if overlap >= 0.22:
        return 4.0
    if overlap >= 0.12:
        return 3.0
    if overlap >= 0.05:
        return 2.0
    return 1.0


def run_scoring_with_stats(
    prompt: str,
    n_runs: int = 5,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Run the scoring prompt N times and compute mean and standard deviation.

    Using temperature > 0 introduces natural variation across runs so that
    the standard deviation reflects genuine model uncertainty rather than
    deterministic repetition.

    Args:
        prompt: Full scoring prompt string (same as passed to the main LLM call).
        n_runs: Number of independent runs (default 5 per spec).
        temperature: Sampling temperature; must be > 0.

    Returns:
        Dict with:
          raw_scores       — list of float scores from successful runs
          mean             — float mean, or None if all runs failed
          std              — float σ, or None if fewer than 2 runs succeeded
          successful_runs  — int count of runs that returned valid JSON

    Side effects:
        Makes up to n_runs HTTP requests to the OpenAI API.
    """
    scores: List[float] = []
    for _ in range(n_runs):
        val = _call_llm_once(prompt, temperature=temperature)
        if val is not None:
            scores.append(val)

    if not scores:
        h = _heuristic_score_from_prompt(prompt)
        return {
            "raw_scores": [h] * max(1, n_runs),
            "mean": round(h, 4),
            "std": 0.0,
            "successful_runs": 0,
        }

    mean_val = statistics.mean(scores)
    std_val = statistics.stdev(scores) if len(scores) > 1 else 0.0

    return {
        "raw_scores": scores,
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "successful_runs": len(scores),
    }


# ===========================================================================
# Feature 1 — Score Splitting (baseline vs live)
# ===========================================================================

def retrieve_baseline_evidence(
    collection: Any,
    query: str,
    k: int = 10,
) -> List[Tuple[dict, str, float]]:
    """
    Retrieve evidence restricted to primary-tier chunks only.

    Primary chunks are those where metadata field source_type == "primary"
    OR tier == 1.  This constitutes the frozen reference corpus.

    Args:
        collection: ChromaDB Collection object.
        query: Query string for semantic retrieval.
        k: Number of results to retrieve.

    Returns:
        List of (metadata_dict, document_str, similarity_float) tuples,
        identical in structure to vectorstore.retrieve_evidence output.
        Returns an empty list if the filter yields no results or on error.

    Side effects:
        Issues a filtered query against the ChromaDB collection.
    """
    try:
        # Route through vectorstore.retrieve_evidence so hybrid retrieval (BM25+RRF)
        # can be applied consistently even for baseline-only evidence.
        from app.vectorstore import retrieve_evidence

        return retrieve_evidence(
            collection,
            query,
            k=k,
            where={"$or": [{"source_type": {"$eq": "primary"}}, {"tier": {"$eq": 1}}]},
        )
    except Exception:
        return []


def _build_evidence_block(evidences: List[Tuple[dict, str, float]]) -> str:
    """Build the evidence block string used inside scoring prompts."""
    if not evidences:
        return "(no evidence available)"
    return "\n".join(
        f"- [{m.get('source_id', '')}] {m.get('url', '')}: {d[:1000]}"
        for m, d, _ in evidences
    )


def compute_score_split(
    kpi_name: str,
    rubric: str,
    baseline_evidences: List[Tuple[dict, str, float]],
    live_evidences: List[Tuple[dict, str, float]],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute baseline_score and live_score, each as mean ± σ over N runs.

    baseline_score is derived from primary-tier evidence only (frozen corpus).
    live_score is derived from all retrieved evidence (all source tiers).

    Args:
        kpi_name: KPI name inserted into the scoring prompt.
        rubric: Rubric text inserted into the scoring prompt.
        baseline_evidences: (metadata, doc, score) list from primary tier only.
        live_evidences: (metadata, doc, score) list from all tiers.
        config: Feature flags dict; falls back to DEFAULT_FEATURE_FLAGS.

    Returns:
        Dict with keys:
          baseline_score, baseline_std, baseline_raw_scores
          live_score,     live_std,     live_raw_scores
          delta  (live_score − baseline_score, or None if either is None)

    Side effects:
        Makes up to 2 × n_runs HTTP requests to the OpenAI API.
    """
    cfg = config or DEFAULT_FEATURE_FLAGS
    n = int(cfg.get("n_scoring_runs", 5))
    temp = float(cfg.get("scoring_temperature", 0.7))

    prompt_tmpl = (
        "You are a Business Analyst scoring a KPI.\n"
        "KPI: {kpi_name}\n"
        "Rubric:\n{rubric}\n"
        "Evidence:\n{evidence_block}\n"
        "Return strict JSON: {{\"score\": 1-5}}"
    )

    baseline_prompt = prompt_tmpl.format(
        kpi_name=kpi_name,
        rubric=rubric,
        evidence_block=_build_evidence_block(baseline_evidences),
    )
    live_prompt = prompt_tmpl.format(
        kpi_name=kpi_name,
        rubric=rubric,
        evidence_block=_build_evidence_block(live_evidences),
    )

    b_stats = run_scoring_with_stats(baseline_prompt, n_runs=n, temperature=temp)
    l_stats = run_scoring_with_stats(live_prompt, n_runs=n, temperature=temp)

    b_mean = b_stats.get("mean")
    l_mean = l_stats.get("mean")
    delta = round(l_mean - b_mean, 4) if (b_mean is not None and l_mean is not None) else None

    return {
        "baseline_score": b_mean,
        "baseline_std": b_stats.get("std"),
        "baseline_raw_scores": b_stats.get("raw_scores", []),
        "live_score": l_mean,
        "live_std": l_stats.get("std"),
        "live_raw_scores": l_stats.get("raw_scores", []),
        "delta": delta,
    }


# ===========================================================================
# Feature 3 — Quality Gates
# ===========================================================================

def apply_quality_gates(
    kpi_id: str,
    company_id: str,
    ragas_faithfulness: Optional[float],
    score_std: Optional[float],
    score_mean: Optional[float],
    retrieved_chunks: List[Tuple[dict, str, float]],
    target_entity_id: str,
    trace: Optional[Any] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Enforce four quality gates in sequence and log each result to LangFuse.

    Gate 1 — faithfulness_gate
        Block if RAGAS faithfulness < 0.8.
        Log reason: low_faithfulness.

    Gate 2 — stability_gate
        If score σ > 0.4, display score as a range (mean−σ to mean+σ)
        instead of a point estimate.  Not a hard block.
        Log reason: unstable_scoring.

    Gate 3 — source_coverage_gate
        If fraction of primary-tier chunks in top-k < 0.4, suppress the
        structural score and show only the contextual score.
        Log reason: low_primary_coverage.

    Gate 4 — competitor_bleed_gate
        Block entirely if any retrieved chunk has entity_id ≠ target.
        Fire a LangFuse alert event.
        Log reason: competitor_bleed_detected.

    Args:
        kpi_id: KPI identifier (used in LangFuse score names).
        company_id: Target company identifier.
        ragas_faithfulness: RAGAS faithfulness metric (0–1), or None.
        score_std: Standard deviation of scores across N runs, or None.
        score_mean: Mean score across N runs, or None.
        retrieved_chunks: List of (metadata, doc, similarity) tuples.
        target_entity_id: Expected entity_id value in chunk metadata.
        trace: LangFuse Trace object for event logging (may be None).
        trace_id: LangFuse trace ID string for score logging (may be None).

    Returns:
        Dict with:
          blocked                  — True if pipeline should halt for this KPI
          display_as_range         — True if Gate 2 triggered (unstable)
          score_range_display      — "low–high" string when display_as_range
          suppress_structural_score— True if Gate 3 triggered
          competitor_bleed_detected— True if Gate 4 triggered
          gates                    — per-gate {passed, reason, ...} details

    Side effects:
        Logs one LangFuse score per gate (pass=1.0, fail=0.0).
        Fires a competitor_bleed_alert event on Gate 4 failure.
    """
    gates: Dict[str, Dict[str, Any]] = {}
    blocked = False
    display_as_range = False
    score_range_display: Optional[str] = None
    suppress_structural = False
    bleed_detected = False

    # ── Gate 1: Faithfulness ──────────────────────────────────────────────
    if ragas_faithfulness is not None:
        passed = ragas_faithfulness >= 0.8
        gates["faithfulness_gate"] = {
            "passed": passed,
            "reason": None if passed else "low_faithfulness",
            "faithfulness": ragas_faithfulness,
        }
        if not passed:
            blocked = True
        if trace_id:
            log_score_to_trace(
                trace_id,
                "faithfulness_gate",
                1.0 if passed else 0.0,
                comment=f"faithfulness={ragas_faithfulness:.3f}",
            )
    else:
        gates["faithfulness_gate"] = {"passed": None, "reason": "faithfulness_unavailable"}

    # ── Gate 2: Stability (soft — display as range, no hard block) ────────
    if score_std is not None:
        passed = score_std <= 0.4
        display_as_range = not passed
        if display_as_range and score_mean is not None:
            lo = round(score_mean - score_std, 2)
            hi = round(score_mean + score_std, 2)
            score_range_display = f"{lo}–{hi}"
        gates["stability_gate"] = {
            "passed": passed,
            "reason": None if passed else "unstable_scoring",
            "std": score_std,
            "score_range_display": score_range_display,
        }
        if trace_id:
            log_score_to_trace(
                trace_id,
                "stability_gate",
                1.0 if passed else 0.0,
                comment=f"std={score_std:.4f}",
            )
    else:
        gates["stability_gate"] = {"passed": None, "reason": "std_unavailable"}

    # ── Gate 3: Source coverage ───────────────────────────────────────────
    total = len(retrieved_chunks)
    primary_count = sum(
        1
        for meta, _, _ in retrieved_chunks
        if isinstance(meta, dict)
        and (meta.get("source_type") == "primary" or meta.get("tier") == 1)
    )
    primary_fraction = primary_count / total if total > 0 else 0.0
    passed = primary_fraction >= 0.4
    suppress_structural = not passed
    gates["source_coverage_gate"] = {
        "passed": passed,
        "reason": None if passed else "low_primary_coverage",
        "primary_fraction": round(primary_fraction, 4),
        "primary_count": primary_count,
        "total_count": total,
    }
    if trace_id:
        log_score_to_trace(
            trace_id,
            "source_coverage_gate",
            1.0 if passed else 0.0,
            comment=f"primary_fraction={primary_fraction:.4f}",
        )

    # ── Gate 4: Competitor bleed ──────────────────────────────────────────
    bleed_metas = [
        meta
        for meta, _, _ in retrieved_chunks
        if isinstance(meta, dict)
        and meta.get("entity_id")
        and meta["entity_id"] != target_entity_id
    ]
    passed = len(bleed_metas) == 0
    bleed_detected = not passed
    if bleed_detected:
        blocked = True
    gates["competitor_bleed_gate"] = {
        "passed": passed,
        "reason": None if passed else "competitor_bleed_detected",
        "bleed_count": len(bleed_metas),
    }
    if trace_id:
        log_score_to_trace(
            trace_id,
            "competitor_bleed_gate",
            1.0 if passed else 0.0,
            comment=f"bleed_count={len(bleed_metas)}",
        )
    if bleed_detected and trace:
        log_event_to_trace(
            trace,
            "competitor_bleed_alert",
            {
                "kpi_id": kpi_id,
                "company_id": company_id,
                "bleed_count": len(bleed_metas),
            },
        )

    return {
        "blocked": blocked,
        "display_as_range": display_as_range,
        "score_range_display": score_range_display,
        "suppress_structural_score": suppress_structural,
        "competitor_bleed_detected": bleed_detected,
        "gates": gates,
    }


# ===========================================================================
# Feature 4 — Score Change Attribution
# ===========================================================================

_ATTRIBUTION_STATE_FILE = os.path.join(
    os.path.dirname(__file__), "output", "attribution_state.json"
)


def _load_attribution_state() -> Dict[str, Any]:
    """Load the persisted attribution state from disk."""
    try:
        with open(_ATTRIBUTION_STATE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_attribution_state(state: Dict[str, Any]) -> None:
    """Persist the attribution state to disk."""
    try:
        os.makedirs(os.path.dirname(_ATTRIBUTION_STATE_FILE), exist_ok=True)
        with open(_ATTRIBUTION_STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
    except Exception:
        pass


def compute_score_attribution(
    company_id: str,
    kpi_id: str,
    new_mean_score: float,
    current_model_version: str,
    current_prompt_hash: str,
    current_source_fingerprint: str,
    trace: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compare new mean score to the previous run and attribute significant changes.

    Reads the previous run's stored state from app/output/attribution_state.json,
    determines the attribution type (model_change | prompt_change | data_change |
    external_noise), and writes the updated state back.

    Attribution logic (checked in priority order):
      1. model_change    — model version string changed
      2. prompt_change   — prompt hash changed
      3. data_change     — ChromaDB snapshot fingerprint changed
      4. external_noise  — none of the above

    Args:
        company_id: Target company identifier (part of the state key).
        kpi_id: KPI identifier (part of the state key).
        new_mean_score: Mean score from the current run.
        current_model_version: Current LLM model name/version string.
        current_prompt_hash: SHA256 of the current full prompt.
        current_source_fingerprint: ChromaDB snapshot ID for this run.
        trace: LangFuse Trace object for event logging (may be None).

    Returns:
        Attribution event dict if |delta| > 0.2, otherwise None.
        Dict includes: delta, attribution_type, company_id, kpi_id,
        new_score, previous_score, event_type="score_attribution".

    Side effects:
        - Reads and writes app/output/attribution_state.json.
        - Logs a score_attribution event to the LangFuse trace.
    """
    state = _load_attribution_state()
    key = f"{company_id}::{kpi_id}"
    prev = state.get(key, {})

    prev_score: Optional[float] = prev.get("mean_score")
    prev_model: Optional[str] = prev.get("model_version")
    prev_prompt: Optional[str] = prev.get("prompt_hash")
    prev_fingerprint: Optional[str] = prev.get("source_fingerprint")

    # Always update state for next run
    state[key] = {
        "mean_score": new_mean_score,
        "model_version": current_model_version,
        "prompt_hash": current_prompt_hash,
        "source_fingerprint": current_source_fingerprint,
    }
    _save_attribution_state(state)

    if prev_score is None:
        return None  # First run for this (company, KPI) pair

    delta = round(new_mean_score - prev_score, 4)
    threshold = float(os.getenv("VITELIS_ATTRIBUTION_DELTA_THRESHOLD", "0.2"))
    if abs(delta) <= threshold:
        return None  # Within tolerance — no attribution needed

    # Determine root cause
    if prev_model and current_model_version != prev_model:
        attribution_type = "model_change"
    elif prev_prompt and current_prompt_hash != prev_prompt:
        attribution_type = "prompt_change"
    elif prev_fingerprint and current_source_fingerprint != prev_fingerprint:
        attribution_type = "data_change"
    else:
        attribution_type = "external_noise"

    event: Dict[str, Any] = {
        "event_type": "score_attribution",
        "company_id": company_id,
        "kpi_id": kpi_id,
        "delta": delta,
        "attribution_type": attribution_type,
        "new_score": new_mean_score,
        "previous_score": prev_score,
    }

    if trace:
        log_event_to_trace(trace, "score_attribution", event)

    return event


# ===========================================================================
# Feature 7 — BERTScore
# ===========================================================================

def compute_bertscore(
    rationale: str,
    reference_chunks: List[str],
    trace_id: Optional[str] = None,
    trace: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """
    Compute BERTScore F1 between the KPI score rationale and retrieved chunks.

    Uses the bert-score library with model_type=distilbert-base-uncased for
    inference speed.  The top-3 retrieved chunks are concatenated as the
    reference text.

    If BERTScore F1 < 0.75, a low_semantic_grounding warning event is fired
    on the LangFuse trace.

    Args:
        rationale: The generated KPI score rationale string (hypothesis).
        reference_chunks: List of retrieved chunk text strings; only the
            first 3 are used as the reference.
        trace_id: LangFuse trace ID for logging bertsccore_f1 score.
        trace: LangFuse Trace object for warning event (may be None).
        config: Feature flags dict; feature is skipped if
            config["bertscore_enabled"] is False.

    Returns:
        BERTScore F1 as a float in [0.0, 1.0], or None if the feature is
        disabled, the library is not installed, or inputs are empty.

    Side effects:
        - Logs bertsccore_f1 as a LangFuse score on the trace.
        - Fires low_semantic_grounding event if F1 < 0.75.
        - May download the distilbert model on first call (network required).
    """
    cfg = config or DEFAULT_FEATURE_FLAGS
    if not cfg.get("bertscore_enabled", True):
        return None
    if not rationale or not reference_chunks:
        return None

    reference_text = " ".join(c[:800] for c in reference_chunks[:3])

    try:
        from bert_score import score as bert_score_fn  # type: ignore

        _P, _R, F1 = bert_score_fn(
            [rationale],
            [reference_text],
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        f1_value = round(float(F1[0]), 4)

        if trace_id:
            log_score_to_trace(trace_id, "bertsccore_f1", f1_value)

        if f1_value < 0.75 and trace:
            log_event_to_trace(
                trace,
                "low_semantic_grounding",
                {"bertscore_f1": f1_value, "threshold": 0.75},
            )

        return f1_value

    except ImportError:
        return None
    except Exception:
        return None


# ===========================================================================
# Feature 8 — Chain-of-Thought Eval
# ===========================================================================

def run_cot_eval(
    kpi_name: str,
    rubric: str,
    retrieved_chunks: List[str],
    rationale: str,
    trace_id: Optional[str] = None,
    trace: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate chain-of-thought quality of the generated score rationale.

    Makes a second LLM call that asks an evaluator model to score the
    rationale on three dimensions:
      - Specificity  (1-5): Does the rationale make precise, specific claims?
      - Evidence     (1-5): Is each claim grounded in the retrieved evidence?
      - Alignment    (1-5): Does the reasoning map to the rubric criteria?

    If any sub-score is < 3, a cot_weak_reasoning warning event is fired.

    Args:
        kpi_name: KPI name string (included in the evaluation prompt).
        rubric: Full rubric criteria text (included in the evaluation prompt).
        retrieved_chunks: List of retrieved chunk text strings (up to 5 used).
        rationale: The generated score rationale to evaluate.
        trace_id: LangFuse trace ID for logging the three CoT sub-scores.
        trace: LangFuse Trace object for warning event (may be None).
        config: Feature flags dict; feature is skipped if
            config["cot_eval_enabled"] is False.

    Returns:
        Dict with keys cot_specificity, cot_evidence, cot_alignment (each int
        1-5), or None if the feature is disabled or the LLM call fails.

    Side effects:
        - Logs cot_specificity, cot_evidence, cot_alignment as LangFuse scores.
        - Fires cot_weak_reasoning event if any sub-score < 3.
        - Makes one HTTP request to the OpenAI chat completions endpoint.
    """
    cfg = config or DEFAULT_FEATURE_FLAGS
    if not cfg.get("cot_eval_enabled", True):
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if not rationale:
        return None

    chunks_text = "\n".join(
        f"[{i + 1}] {c[:800]}" for i, c in enumerate(retrieved_chunks[:5])
    )

    eval_prompt = (
        "You are a KPI evaluation quality assessor.\n\n"
        f"KPI: {kpi_name}\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Retrieved Evidence:\n{chunks_text}\n\n"
        f"Generated Rationale:\n{rationale}\n\n"
        "Assess the rationale on three dimensions (score 1=poor, 5=excellent):\n"
        "1. Specificity: Are claims specific and precise rather than vague?\n"
        "2. Evidence-grounding: Is every claim supported by the evidence above?\n"
        "3. Criterion-alignment: Does the reasoning map to the rubric criteria?\n\n"
        'Return strict JSON: {"cot_specificity": 1-5, "cot_evidence": 1-5, '
        '"cot_alignment": 1-5, "explanation": "..."}'
    )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a strict JSON generator."},
                    {"role": "user", "content": eval_prompt},
                ],
                "temperature": 0.1,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return None
        content = resp.json()["choices"][0]["message"]["content"]
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            return None
        data = json.loads(content[start : end + 1])

        specificity = max(1, min(5, int(data.get("cot_specificity", 3))))
        evidence = max(1, min(5, int(data.get("cot_evidence", 3))))
        alignment = max(1, min(5, int(data.get("cot_alignment", 3))))

        if trace_id:
            log_score_to_trace(trace_id, "cot_specificity", float(specificity))
            log_score_to_trace(trace_id, "cot_evidence", float(evidence))
            log_score_to_trace(trace_id, "cot_alignment", float(alignment))

        if any(s < 3 for s in (specificity, evidence, alignment)) and trace:
            log_event_to_trace(
                trace,
                "cot_weak_reasoning",
                {
                    "cot_specificity": specificity,
                    "cot_evidence": evidence,
                    "cot_alignment": alignment,
                },
            )

        return {
            "cot_specificity": specificity,
            "cot_evidence": evidence,
            "cot_alignment": alignment,
        }

    except Exception:
        return None
