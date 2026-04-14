"""
eval_rag.py — RAG Evaluation Module
=====================================
This file checks how well the AI system is finding, using, and reporting
information about a company's AI maturity.


Nine checks are run:
    1. RAGAS Evaluation         — Overall pipeline quality score (industry standard).
                                  Faithfulness, Answer Relevance, Context Precision and Recall 
                                  
                                 
    2. LLM as a Judge           — A second AI reviews each answer for quality
    3. Recall@k                 — Were the most useful sources in the top results?
    4. F1 Score                 — How much overlap is there between answer and evidence?
    5. Hallucination Check      — Did the AI say things not backed by evidence?
    6. MMR (Diversity Check)    — Did the system pull from varied sources, not just similar ones?


Ground truth strategy:
    Checks (Factual Correctness ,Noise Sensitivity,Semantic Similarity) require a reference "ideal answer" to compare against.
    Rather than writing these manually, we extract the Score-5 rubric description
    from kpis.yaml for each KPI. This describes what a top-scoring answer looks like
    (e.g. "Clear, coherent AI strategy with priorities and outcomes.") and serves
    as the benchmark the AI's actual answer is measured against.

How to use:
    from eval_rag import run_all_evaluations
    run_all_evaluations(kpi_results, sources, kpi_definitions)

Where RAG report scores come from (when LLM/ragas is not used):
----------------------------------------------------------------
These metrics appear in report YAML under rag_evaluation. When the OpenAI API
is unavailable (e.g. 429) or ragas is not installed, values come from local
fallbacks in this file — no LLM is called.

  • ragas_context_precision
    Fallback: evaluate_ragas() local approximation (lines ~253–268).
    Formula: For each retrieved context chunk, (question_tokens ∩ chunk_tokens) / len(chunk_tokens);
             then average over chunks. Low = retrieved text shares few words with the question.

  • ragas_context_recall
    Fallback: same block. Formula: (ground_truth_tokens ∩ context_tokens) / len(ground_truth_tokens).
    Ground truth = Score-5 rubric text from kpis.yaml. Fraction of "ideal" words found in evidence.

  • hallucination_score / hallucination_flagged
    Fallback: evaluate_hallucination() heuristic (lines ~585–633). Always runs first.
    Formula: Split answer into sentences;              if <20% of a sentence's words appear in evidence,
             count as "unsupported". heuristic_score = unsupported_sentences / total_sentences.
    If LLM layer is skipped (no key or _call_llm fails), final_score = heuristic_score.
    hallucination_flagged = (final_score > threshold), default threshold 0.4.

  • mmr_diversity_score
    No LLM. evaluate_mmr() (lines ~702–819) uses only _simple_embedding() (hash-based 64-d vector)
    and _cosine_similarity(). Formula: MMR re-ranks chunks; diversity_score = 1.0 - avg_redundancy,
    where avg_redundancy is mean pairwise cosine similarity of selected chunks. High = diverse sources.

  • semantic_similarity
    No LLM in fallback. evaluate_ragas_with_ground_truth() local path (lines ~965–973).
    Formula: answer_vec = _simple_embedding(answer), gt_vec = _simple_embedding(ground_truth);
             raw = cosine_similarity(answer_vec, gt_vec); semantic_similarity = (raw + 1) / 2 (scale to 0–1).
    Measures hash-based similarity between answer and Score-5 rubric text.
"""

from __future__ import annotations

import os
import re
import json
import time
import math
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.langfuse_client import create_trace, get_trace_id
from app.score_extensions import DEFAULT_FEATURE_FLAGS, compute_bertscore, run_cot_eval


# ==============================================================================
# HELPER: Call the LLM (OpenAI or Gemini)
# ==============================================================================

def _extract_json_from_response(text: str) -> Optional[dict]:
    """Parse first JSON object from LLM response text."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _call_llm_gemini(prompt: str, system_msg: str, verbose: bool) -> Optional[dict]:
    """Call Google AI Studio (Gemini) REST API for LLM judge / JSON tasks."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    full_prompt = f"{system_msg}\n\n{prompt}" if system_msg else prompt
    max_retries = 3
    backoff = 5
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                json={
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {"temperature": 0.1},
                },
                timeout=60,
            )
            if resp.status_code == 429:
                wait = backoff * (2**attempt)
                if verbose:
                    print(f"  [Rate limit] Gemini 429 — waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                return None
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts") or []
            if not parts:
                return None
            text = parts[0].get("text", "").strip()
            result = _extract_json_from_response(text) if text else None
            if result:
                return result
        except Exception as exc:
            if verbose:
                print(f"  [Gemini] Attempt {attempt + 1}/{max_retries} failed: {exc}")
            if attempt < max_retries - 1:
                wait = backoff * (2**attempt)
                time.sleep(wait)
    return None


def _call_llm(prompt: str, system_msg: str = "You are a strict JSON generator.", verbose: bool = True) -> Optional[dict]:
    """
    Send a question to the AI and get a JSON answer back.
    Uses Gemini if GOOGLE_API_KEY/GEMINI_API_KEY and VITELIS_LLM_PROVIDER=gemini;
    otherwise uses OpenAI (GPT-4o-mini).
    In pipeline mode (verbose=False): skips immediately on 429 — no waiting.
    In standalone mode (verbose=True): retries twice with short backoff.
    """
    provider = (os.getenv("VITELIS_LLM_PROVIDER") or "").strip().lower()
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    use_gemini = provider in ("gemini", "google") or (google_key and provider != "openai")

    if use_gemini and google_key:
        result = _call_llm_gemini(prompt, system_msg, verbose)
        if result is not None:
            return result
        if verbose:
            print("  [Note] Gemini judge call failed; trying OpenAI if key set.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if verbose:
            print("  [Note] No OpenAI API key found — skipping LLM-based checks.")
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_retries = 2 if verbose else 1  # pipeline: try once and move on
    backoff = 5  # seconds — only used in verbose/standalone mode

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                },
                timeout=30,
            )
            if response.status_code == 429:
                if not verbose:
                    return None  # pipeline: skip immediately, don't block
                wait = backoff * (2 ** attempt)
                print(f"  [Rate limit] 429 — waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                return json.loads(content[start:end + 1])
        except Exception as exc:
            if "429" in str(exc):
                if not verbose:
                    return None  # pipeline: skip immediately
                wait = backoff * (2 ** attempt)
                print(f"  [Rate limit] 429 — waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
                continue
            if verbose:
                print(f"  [Warning] LLM call failed: {exc}")
            return None

    if verbose:
        print(f"  [Warning] LLM call failed after {max_retries} retries (rate limit).")
    return None


# ==============================================================================
# HELPER: Simple token overlap (used in F1 and Recall calculations)
# ==============================================================================

def _tokenize(text: str) -> List[str]:
    """
    Break text into individual meaningful words.
    Remove short filler words (like 'the', 'a', 'is') that don't carry meaning.
    """
    # Common words that carry no useful meaning for comparison
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "it", "its",
        "we", "our", "us", "this", "that", "be", "been", "has", "have", "had",
    }
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if w not in stop_words and len(w) > 2]


# ==============================================================================
# HELPER: Extract ground truth from rubric Score-5 description
# ==============================================================================

def extract_ground_truth_from_rubric(rubric: Optional[List[str]]) -> Optional[str]:
    """
    Pull the Score-5 description out of a KPI's rubric list and use it
    as the 'ideal answer' (ground truth) for evaluation purposes.

    Every rubric KPI in kpis.yaml has three anchor points:
        - Score 1: what a failing answer looks like
        - Score 3: what a mediocre answer looks like
        - Score 5: what an excellent answer looks like

    We use the Score-5 line as the ground truth because it describes
    exactly what a well-performing company should be doing for that KPI.
    This is not a statement of fact about the specific company —
    it is a benchmark description of what "great" looks like.

    Example input rubric list:
        ["1: No AI strategy mentioned.",
         "3: Some AI initiatives described but vague or fragmented.",
         "5: Clear, coherent AI strategy with priorities and outcomes."]

    Example output:
        "Clear, coherent AI strategy with priorities and outcomes."

    Args:
        rubric: The list of rubric strings from the KPI definition

    Returns:
        The Score-5 description text, or None if not found.
    """
    if not rubric:
        return None

    for line in rubric:
        # Look for the line that starts with "5:" (Score 5 anchor)
        stripped = line.strip()
        if stripped.startswith("5:"):
            # Remove the "5: " prefix and return the description text only
            return stripped[2:].strip()

    return None


# ==============================================================================
# 1. RAGAS EVALUATION
# ==============================================================================

def evaluate_ragas(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Dict[str, Any]:
    """
    RAGAS is a widely used industry standard for measuring RAG pipeline quality.
    It checks four things:
      - Faithfulness:        Does the answer only use facts from the evidence?
      - Answer Relevance:    Does the answer actually address the question?
      - Context Precision:   Are the retrieved sources relevant to the question?
      - Context Recall:      Did we retrieve enough of the right information?
                             (Only measurable if a 'ground truth' answer is provided)

    This function approximates RAGAS using the ragas library if installed,
    and gracefully falls back to a lightweight local version if not available.

    Args:
        question:     The KPI question being evaluated (e.g. "How clear is the AI strategy?")
        answer:       The AI-generated rationale/answer for that KPI
        contexts:     The list of evidence text chunks retrieved from the web sources
        ground_truth: Optional — a reference answer to compare against (for recall)

    Returns:
        A dictionary of scores, each between 0 and 1 (1 = best possible).
    """

    provider = (os.getenv("VITELIS_LLM_PROVIDER") or "").strip().lower()
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    ragas_disabled = os.getenv("VITELIS_RAGAS_DISABLED", "").lower() in {"1", "true", "yes"}
    if provider in {"gemini", "google"} and not google_key:
        ragas_disabled = True

    # --- Attempt to use the ragas library (pip install ragas) ---
    if ragas_disabled:
        # Force local approximation path below (no external LLM calls).
        pass
    else:
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset

            # Build the dataset in the format ragas expects
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)

            # Choose which metrics to run based on what data we have
            metrics = [faithfulness, answer_relevancy, context_precision]
            if ground_truth:
                metrics.append(context_recall)

            # Run the ragas evaluation
            result = ragas_evaluate(dataset, metrics=metrics)
            scores = result.to_pandas().iloc[0].to_dict()

            return {
                "method": "ragas_library",
                "faithfulness": round(float(scores.get("faithfulness", 0)), 3),
                "answer_relevancy": round(float(scores.get("answer_relevancy", 0)), 3),
                "context_precision": round(float(scores.get("context_precision", 0)), 3),
                "context_recall": round(float(scores.get("context_recall", 0)), 3) if ground_truth else None,
            }

        except ImportError:
            # ragas not installed — run a lightweight local approximation instead
            pass
        except Exception as exc:
            print(f"  [Warning] ragas library failed ({exc}), falling back to local approximation.")

    # --- Lightweight local fallback (no ragas library needed) ---
    # Faithfulness: what fraction of the answer's words appear in the evidence?
    answer_tokens = set(_tokenize(answer))
    context_tokens = set(_tokenize(" ".join(contexts)))
    faithfulness = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)

    # Answer relevancy: what fraction of question words appear in the answer?
    question_tokens = set(_tokenize(question))
    answer_relevancy = len(question_tokens & answer_tokens) / max(len(question_tokens), 1)

    # Context precision: what fraction of context words are relevant to the question?
    precision_scores = []
    for ctx in contexts:
        ctx_tokens = set(_tokenize(ctx))
        if ctx_tokens:
            precision_scores.append(len(question_tokens & ctx_tokens) / len(ctx_tokens))
    context_precision = sum(precision_scores) / max(len(precision_scores), 1)

    # Context recall: how much of the ground truth is covered by the contexts?
    context_recall = None
    if ground_truth:
        gt_tokens = set(_tokenize(ground_truth))
        context_recall = len(gt_tokens & context_tokens) / max(len(gt_tokens), 1)

    return {
        "method": "local_approximation",
        "faithfulness": round(faithfulness, 3),
        "answer_relevancy": round(answer_relevancy, 3),
        "context_precision": round(context_precision, 3),
        "context_recall": round(context_recall, 3) if context_recall is not None else None,
    }


def print_ragas_summary(kpi_name: str, scores: Dict[str, Any]) -> None:
    """Print a plain-English summary of RAGAS results for a single KPI."""
    print(f"\n  [RAGAS — {kpi_name}]")
    print(f"    Method used          : {scores.get('method', 'unknown')}")
    print(f"    Answer stays on topic: {scores['faithfulness']:.0%}  "
          f"(Did the AI only use facts from the evidence?)")
    print(f"    Answer relevance     : {scores['answer_relevancy']:.0%}  "
          f"(Did the answer address the question?)")
    print(f"    Source relevance     : {scores['context_precision']:.0%}  "
          f"(Were the retrieved sources about the right topic?)")
    if scores.get("context_recall") is not None:
        print(f"    Coverage of truth    : {scores['context_recall']:.0%}  "
              f"(How much of the expected answer was found?)")


# ==============================================================================
# 2. LLM AS A JUDGE
# ==============================================================================

def evaluate_llm_as_judge(
    question: str,
    answer: str,
    contexts: List[str],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    A second AI reads the question, the evidence, and the generated answer,
    then gives its verdict on three qualities:

      - Correctness:  Is the answer factually supported by the evidence?
      - Completeness: Does the answer cover all important points from the evidence?
      - Clarity:      Is the answer easy to understand and well-structured?

    Each is scored 1–5, where 5 is the best possible.
    This is like having a senior analyst review a junior analyst's work.

    Args:
        question: The KPI question being evaluated
        answer:   The AI-generated answer/rationale
        contexts: The retrieved evidence chunks

    Returns:
        Dictionary with scores (1–5) and brief feedback for each quality dimension.
    """

    # Combine the top 3 evidence chunks into a readable block for the judge
    evidence_block = "\n".join(f"- {ctx[:300]}" for ctx in contexts[:3])

    # Build the review prompt for the judging LLM
    prompt = (
        "You are a senior business analyst reviewing an AI-generated assessment. "
        "Read the question, the evidence, and the answer below. "
        "Score the answer on three dimensions from 1 (poor) to 5 (excellent).\n\n"
        "Return ONLY strict JSON in this format:\n"
        '{"correctness": <1-5>, "completeness": <1-5>, "clarity": <1-5>, '
        '"overall": <1-5>, "feedback": "<one sentence summary>"}\n\n'
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        f"ANSWER TO REVIEW:\n{answer}\n"
    )

    result = _call_llm(prompt, system_msg="You are a strict JSON generator and impartial reviewer.", verbose=verbose)

    if result:
        return {
            "correctness": int(result.get("correctness", 3)),
            "completeness": int(result.get("completeness", 3)),
            "clarity": int(result.get("clarity", 3)),
            "overall": int(result.get("overall", 3)),
            "feedback": str(result.get("feedback", "")),
            "llm_used": True,
        }

    # If the LLM call failed, return a neutral result
    return {
        "correctness": None,
        "completeness": None,
        "clarity": None,
        "overall": None,
        "feedback": "LLM judge unavailable.",
        "llm_used": False,
    }


def print_judge_summary(kpi_name: str, result: Dict[str, Any]) -> None:
    """Print a plain-English summary of the LLM judge's verdict."""
    print(f"\n  [AI Judge Review — {kpi_name}]")
    if not result.get("llm_used"):
        print("    AI judge was not available for this check.")
        return
    print(f"    Factual accuracy : {result['correctness']}/5  "
          f"(Is the answer backed by real evidence?)")
    print(f"    Completeness     : {result['completeness']}/5  "
          f"(Did it cover all key points?)")
    print(f"    Clarity          : {result['clarity']}/5  "
          f"(Is the answer clear and professional?)")
    print(f"    Overall quality  : {result['overall']}/5")
    print(f"    Reviewer note    : {result['feedback']}")


# ==============================================================================
# 3. RECALL@K
# ==============================================================================

def evaluate_recall_at_k(
    question: str,
    contexts: List[str],
    k_values: Optional[List[int]] = None,
    relevance_threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    Checks: if we only kept the top K sources, would we still find the key information?

    This matters because the system retrieves several evidence chunks for each KPI.
    If the best information only appears in chunk #8 of 10, that's a sign the
    retrieval ranking could be improved.

    Recall@k is calculated for multiple values of k (e.g. top-1, top-3, top-5).
    A high Recall@3 means the most important information appears in the first 3 results.

    Relevance is measured by word overlap with the question — a chunk is considered
    "relevant" if at least `relevance_threshold` fraction of its words relate to the question.

    Args:
        question:            The KPI question
        contexts:            List of retrieved chunks (in retrieval order — best first)
        k_values:            Which top-k cutoffs to evaluate (default: [1, 3, 5, 10])
        relevance_threshold: Minimum overlap fraction to call a chunk "relevant"

    Returns:
        Dictionary mapping each k to a recall score (0–1).
    """

    if k_values is None:
        k_values = [1, 3, 5, 10]

    question_tokens = set(_tokenize(question))

    # Mark each chunk as relevant or not based on word overlap with the question
    relevance_flags: List[bool] = []
    for ctx in contexts:
        ctx_tokens = set(_tokenize(ctx))
        if not ctx_tokens:
            relevance_flags.append(False)
            continue
        overlap = len(question_tokens & ctx_tokens) / len(ctx_tokens)
        relevance_flags.append(overlap >= relevance_threshold)

    total_relevant = sum(relevance_flags)

    # For each k, compute what fraction of relevant chunks appear in the top k
    recall_at_k: Dict[str, float] = {}
    for k in k_values:
        if total_relevant == 0:
            # No relevant chunks found at all — recall is 0
            recall_at_k[f"recall@{k}"] = 0.0
        else:
            found_in_top_k = sum(relevance_flags[:k])
            recall_at_k[f"recall@{k}"] = round(found_in_top_k / total_relevant, 3)

    return {
        "total_chunks": len(contexts),
        "relevant_chunks_found": total_relevant,
        "relevance_threshold_used": relevance_threshold,
        **recall_at_k,
    }


def print_recall_at_k_summary(kpi_name: str, result: Dict[str, Any]) -> None:
    """Print a plain-English summary of Recall@k results."""
    print(f"\n  [Top-Result Quality — {kpi_name}]")
    print(f"    Total sources checked     : {result['total_chunks']}")
    print(f"    Relevant sources found    : {result['relevant_chunks_found']}")
    for key, val in result.items():
        if key.startswith("recall@"):
            k = key.split("@")[1]
            print(f"    Best info in top {k:>2} results: {val:.0%}  "
                  f"(Higher = important info ranked near the top)")


# ==============================================================================
# 4. F1 SCORE
# ==============================================================================

def evaluate_f1(
    answer: str,
    reference: str,
) -> Dict[str, Any]:
    """
    Compares the AI's generated answer to a reference text word by word.
    Measures how similar they are in terms of content covered.

    - Precision: Of all the words in the AI's answer, how many appear in the reference?
    - Recall:    Of all the words in the reference, how many did the AI include?
    - F1:        The balanced combination of precision and recall.

    A high F1 means the AI's answer closely matches what the reference says —
    it didn't add unsupported claims (good precision) and didn't miss key points
    (good recall).

    The reference can be either:
      - A ground-truth expert answer (ideal), or
      - The combined evidence text (practical fallback)

    Args:
        answer:    The AI-generated answer/rationale
        reference: A reference text to compare against

    Returns:
        Dictionary with precision, recall, and F1 (all 0–1 scale).
    """

    answer_tokens = _tokenize(answer)
    reference_tokens = _tokenize(reference)

    # Count word frequencies in each
    from collections import Counter
    answer_counts = Counter(answer_tokens)
    reference_counts = Counter(reference_tokens)

    # Overlapping words (intersection of token counts)
    overlap = sum((answer_counts & reference_counts).values())

    # Precision: how much of the answer is supported by the reference?
    precision = overlap / max(sum(answer_counts.values()), 1)

    # Recall: how much of the reference content is reflected in the answer?
    recall = overlap / max(sum(reference_counts.values()), 1)

    # F1: harmonic mean — penalises if either precision or recall is low
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "answer_word_count": len(answer_tokens),
        "reference_word_count": len(reference_tokens),
        "shared_words": int(overlap),
    }


def print_f1_summary(kpi_name: str, result: Dict[str, Any]) -> None:
    """Print a plain-English summary of F1 scoring."""
    print(f"\n  [Answer Accuracy (F1) — {kpi_name}]")
    print(f"    Answer stays on-topic     : {result['precision']:.0%}  "
          f"(How much of the answer matches the reference?)")
    print(f"    Answer covers key points  : {result['recall']:.0%}  "
          f"(How much of the reference is reflected in the answer?)")
    print(f"    Overall accuracy score    : {result['f1']:.0%}  "
          f"(Combined — closer to 100% is better)")
    print(f"    Shared meaningful words   : {result['shared_words']} "
          f"(out of {result['answer_word_count']} in answer, "
          f"{result['reference_word_count']} in reference)")


# ==============================================================================
# 5. HALLUCINATION CHECK
# ==============================================================================

def evaluate_hallucination(
    answer: str,
    contexts: List[str],
    hallucination_threshold: float = 0.4,
    use_llm: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Checks whether the AI's answer contains claims that are NOT supported
    by the evidence it retrieved. This is called "hallucination" — when an AI
    makes up or assumes information that isn't in the source material.

    Two-layer check:
      Layer 1 (always runs): Word-overlap heuristic
        — Splits the answer into sentences and checks each one against the evidence.
        — A sentence is "unsupported" if very few of its words appear in any evidence chunk.
        — Hallucination score = fraction of answer sentences that are unsupported.

      Layer 2 (runs if LLM available): LLM verification
        — Asks a second AI to read the evidence and the answer, then flag any
          claims in the answer that cannot be found or inferred from the evidence.
        — More accurate than word overlap alone.

    A hallucination score above `hallucination_threshold` triggers a warning flag.
    Default threshold: 0.4 (40% of sentences unsupported = concern).

    Args:
        answer:                  The AI-generated answer/rationale
        contexts:                Retrieved evidence chunks
        hallucination_threshold: Score above this level = flagged as risky (0–1)
        use_llm:                 Whether to run the LLM verification layer

    Returns:
        Dictionary with hallucination score, flag, and details of unsupported sentences.
    """

    # Combine all evidence into one searchable text block
    all_evidence = " ".join(contexts).lower()
    evidence_tokens = set(_tokenize(all_evidence))

    # Split the answer into individual sentences for sentence-level checking
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    sentences = [s for s in sentences if len(s.split()) >= 5]  # Skip very short fragments

    unsupported_sentences: List[str] = []

    for sentence in sentences:
        sent_tokens = set(_tokenize(sentence))
        if not sent_tokens:
            continue
        # What fraction of this sentence's words appear anywhere in the evidence?
        overlap_ratio = len(sent_tokens & evidence_tokens) / len(sent_tokens)
        # If less than 20% of the sentence's words are in the evidence, flag it
        if overlap_ratio < 0.20:
            unsupported_sentences.append(sentence)

    # Heuristic hallucination score: fraction of sentences that are unsupported
    heuristic_score = (
        len(unsupported_sentences) / len(sentences) if sentences else 0.0
    )

    # --- LLM verification layer ---
    llm_score: Optional[float] = None
    llm_flagged_claims: List[str] = []

    if use_llm and os.getenv("OPENAI_API_KEY"):
        evidence_block = "\n".join(f"- {ctx[:250]}" for ctx in contexts[:4])
        prompt = (
            "You are a fact-checker. Read the EVIDENCE and the ANSWER below.\n"
            "Identify any claims in the ANSWER that are NOT supported by the EVIDENCE.\n"
            "Return strict JSON:\n"
            '{"unsupported_claims": ["<claim 1>", "<claim 2>"], '
            '"hallucination_score": <float 0.0 to 1.0>, '
            '"verdict": "<one sentence summary>"}\n\n'
            f"EVIDENCE:\n{evidence_block}\n\n"
            f"ANSWER:\n{answer}\n"
        )
        result = _call_llm(prompt, verbose=verbose)
        if result:
            llm_score = float(result.get("hallucination_score", heuristic_score))
            llm_flagged_claims = result.get("unsupported_claims", [])

    # Use LLM score if available, otherwise use heuristic
    final_score = llm_score if llm_score is not None else heuristic_score
    is_flagged = final_score > hallucination_threshold

    return {
        "hallucination_score": round(final_score, 3),
        "is_flagged": is_flagged,
        "threshold_used": hallucination_threshold,
        "heuristic_score": round(heuristic_score, 3),
        "llm_score": round(llm_score, 3) if llm_score is not None else None,
        "unsupported_sentences_heuristic": unsupported_sentences[:5],
        "llm_flagged_claims": llm_flagged_claims[:5],
        "sentence_count": len(sentences),
        "unsupported_count_heuristic": len(unsupported_sentences),
    }


def print_hallucination_summary(kpi_name: str, result: Dict[str, Any]) -> None:
    """Print a plain-English summary of the hallucination check."""
    flag = "  FLAGGED — Review recommended" if result["is_flagged"] else " PASS"
    print(f"\n  [Accuracy Check (Hallucination) — {kpi_name}]")
    print(f"    Status               : {flag}")
    print(f"    Unsupported content  : {result['hallucination_score']:.0%}  "
          f"(How much of the answer is NOT backed by evidence)")
    print(f"    Alert threshold      : {result['threshold_used']:.0%}  "
          f"(Above this level, the answer is flagged for review)")
    if result["llm_score"] is not None:
        print(f"    AI fact-check score  : {result['llm_score']:.0%}")
    if result["llm_flagged_claims"]:
        print("    Unsupported claims flagged by AI reviewer:")
        for claim in result["llm_flagged_claims"][:3]:
            print(f"      - {claim[:120]}")
    elif result["unsupported_sentences_heuristic"]:
        print("    Potentially unsupported sentences (word-overlap check):")
        for sent in result["unsupported_sentences_heuristic"][:3]:
            print(f"      - {sent[:120]}")


# ==============================================================================
# 6. MAXIMAL MARGINAL RELEVANCE (MMR) — DIVERSITY CHECK
# ==============================================================================

def _simple_embedding(text: str, dims: int = 64) -> List[float]:
    """
    Create a simple numerical fingerprint (embedding) for a piece of text.
    Uses the same hash-based approach as the main vectorstore.py —
    no external model or API needed.

    Each position in the 64-number vector gets a value based on which words
    appear in the text. Similar texts get similar vectors.
    """
    vec = [0.0] * dims
    for token in _tokenize(text):
        # Hash each word to a position and direction in the vector
        h = hash(token)
        idx = abs(h) % dims
        vec[idx] += 1.0 if h > 0 else -1.0

    # Normalise to unit length so all vectors are comparable
    magnitude = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / magnitude for v in vec]


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Measure how similar two text fingerprints are.
    Returns 1.0 if identical, 0.0 if completely different, -1.0 if opposite.
    """
    return sum(a * b for a, b in zip(vec_a, vec_b))


def evaluate_mmr(
    question: str,
    contexts: List[str],
    top_k: int = 5,
    lambda_param: float = 0.5,
) -> Dict[str, Any]:
    """
    Maximal Marginal Relevance (MMR) — checks whether the retrieved sources
    are diverse enough, or whether the system pulled in the same information
    multiple times from different pages.

    Good retrieval should find sources that are:
      (a) Relevant to the question, AND
      (b) Different from each other (so they add new information)

    MMR balances both. It re-ranks the retrieved chunks to prefer ones that
    are relevant to the question BUT different from what has already been selected.

    lambda_param controls the trade-off:
      - lambda = 1.0: prioritise relevance only (ignore diversity)
      - lambda = 0.0: prioritise diversity only (ignore relevance)
      - lambda = 0.5: balanced (default — recommended)

    Args:
        question:     The KPI question
        contexts:     Retrieved evidence chunks (in original retrieval order)
        top_k:        How many chunks to select via MMR re-ranking
        lambda_param: Relevance vs. diversity trade-off (0–1)

    Returns:
        Dictionary with the MMR-selected chunks, their scores, and a diversity score.
    """

    if not contexts:
        return {
            "selected_indices": [],
            "mmr_scores": [],
            "diversity_score": 0.0,
            "avg_relevance": 0.0,
            "avg_redundancy": 0.0,
            "selected_chunks": [],
        }

    # Step 1: Create numerical fingerprints for the question and all chunks
    query_vec = _simple_embedding(question)
    context_vecs = [_simple_embedding(ctx) for ctx in contexts]

    # Step 2: MMR iterative selection — pick chunks one at a time
    selected_indices: List[int] = []
    candidate_indices = list(range(len(contexts)))
    mmr_scores: List[float] = []

    for _ in range(min(top_k, len(contexts))):
        best_idx = -1
        best_score = float("-inf")

        for idx in candidate_indices:
            # How relevant is this chunk to the question?
            relevance = _cosine_similarity(query_vec, context_vecs[idx])

            # How similar is this chunk to what we've already selected?
            # (We want to AVOID chunks that are too similar to already-chosen ones)
            if selected_indices:
                redundancy = max(
                    _cosine_similarity(context_vecs[idx], context_vecs[sel])
                    for sel in selected_indices
                )
            else:
                redundancy = 0.0

            # MMR score: reward relevance, penalise redundancy
            mmr = lambda_param * relevance - (1 - lambda_param) * redundancy

            if mmr > best_score:
                best_score = mmr
                best_idx = idx

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        mmr_scores.append(round(best_score, 3))
        candidate_indices.remove(best_idx)

    # Step 3: Measure diversity of the selected set
    # Compare all pairs of selected chunks — low similarity = high diversity
    pairwise_similarities: List[float] = []
    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            sim = _cosine_similarity(
                context_vecs[selected_indices[i]],
                context_vecs[selected_indices[j]],
            )
            pairwise_similarities.append(sim)

    avg_redundancy = (
        sum(pairwise_similarities) / len(pairwise_similarities)
        if pairwise_similarities else 0.0
    )
    # Diversity = 1 minus average similarity (more different = more diverse)
    diversity_score = round(1.0 - avg_redundancy, 3)

    # Average relevance of selected chunks to the question
    avg_relevance = round(
        sum(_cosine_similarity(query_vec, context_vecs[i]) for i in selected_indices)
        / max(len(selected_indices), 1),
        3,
    )

    return {
        "selected_indices": selected_indices,
        "mmr_scores": mmr_scores,
        "diversity_score": diversity_score,
        "avg_relevance": avg_relevance,
        "avg_redundancy": round(avg_redundancy, 3),
        "selected_chunks": [contexts[i][:200] for i in selected_indices],
        "lambda_used": lambda_param,
    }


def print_mmr_summary(kpi_name: str, result: Dict[str, Any]) -> None:
    """Print a plain-English summary of the MMR diversity check."""
    print(f"\n  [Source Diversity Check (MMR) — {kpi_name}]")
    print(f"    Sources selected          : {len(result['selected_indices'])} "
          f"(re-ranked for best relevance + variety)")
    print(f"    Source diversity score    : {result['diversity_score']:.0%}  "
          f"(Higher = sources cover different topics — better)")
    print(f"    Average relevance score   : {result['avg_relevance']:.0%}  "
          f"(How closely sources relate to the question)")
    print(f"    Average overlap/repetition: {result['avg_redundancy']:.0%}  "
          f"(Lower = sources are more unique — better)")
    print(f"    Balance setting (lambda)  : {result['lambda_used']}  "
          f"(0.5 = equal weight on relevance and diversity)")


# ==============================================================================
# 7, 8, 9. GROUND-TRUTH-BASED RAGAS METRICS
#    — Factual Correctness, Noise Sensitivity, Semantic Similarity
#    These three run together because they all need the same ground truth input.
# ==============================================================================

def evaluate_ragas_with_ground_truth(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
) -> Dict[str, Any]:
    """
    Runs three additional RAGAS checks that require a reference 'ideal answer'
    to compare the AI's output against.

    The ground truth used here is the Score-5 rubric description from kpis.yaml —
    it describes what an excellent, top-scoring answer to this KPI looks like.
    It is NOT a confirmed factual statement about the company; it is a quality benchmark.

    The three checks:

    CHECK 7 — FACTUAL CORRECTNESS
        Breaks both the AI's answer and the ground truth into individual claims.
        Then checks: how many claims in the answer match claims in the ground truth?
        This is more precise than word-level F1 because it works at the meaning level.
        Score 0-1. Higher = the answer's claims align well with what "good" looks like.

    CHECK 8 — NOISE SENSITIVITY
        Measures whether thin, irrelevant, or boilerplate sources (Tier 3 pages)
        caused the AI to give an incorrect or misleading answer.
        A high noise sensitivity score means the AI was misled by poor sources.
        Score 0-1. LOWER is better (less sensitive to noise = more robust).

    CHECK 9 — SEMANTIC SIMILARITY
        Uses word embeddings to compare the overall meaning of the answer
        against the ground truth. Unlike F1, this captures paraphrasing and
        synonyms — two sentences can mean the same thing even with different words.
        Score 0-1. Higher = the answer means roughly the same as the ideal answer.

    All three attempt to use the ragas library first (pip install ragas).
    If ragas is not installed, they fall back to local approximations.

    Args:
        question:     The KPI question
        answer:       The AI-generated rationale/answer
        contexts:     Retrieved evidence chunks (used for noise sensitivity)
        ground_truth: The Score-5 rubric description — the "ideal answer" benchmark

    Returns:
        Dictionary with scores for all three checks (0-1 scale each).
    """

    provider = (os.getenv("VITELIS_LLM_PROVIDER") or "").strip().lower()
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    ragas_disabled = os.getenv("VITELIS_RAGAS_DISABLED", "").lower() in {"1", "true", "yes"}
    if provider in {"gemini", "google"} and not google_key:
        ragas_disabled = True

    # --- Attempt to use the ragas library ---
    if not ragas_disabled:
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                FactualCorrectness,
                NoiseSensitivity,
                SemanticSimilarity,
            )
            from datasets import Dataset

            # Build the dataset ragas expects
            dataset = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            })

            # Run all three metrics together in one call (more efficient)
            result = ragas_evaluate(
                dataset,
                metrics=[FactualCorrectness(), NoiseSensitivity(), SemanticSimilarity()],
            )
            scores = result.to_pandas().iloc[0].to_dict()

            return {
                "method": "ragas_library",
                "ground_truth_used": ground_truth[:150],
                # Check 7: Factual correctness — claim-level accuracy vs ideal answer
                "factual_correctness": round(float(scores.get("factual_correctness", 0)), 3),
                # Check 8: Noise sensitivity — lower is better
                "noise_sensitivity": round(float(scores.get("noise_sensitivity", 0)), 3),
                # Check 9: Semantic similarity — meaning-level match to ideal answer
                "semantic_similarity": round(float(scores.get("semantic_similarity", 0)), 3),
            }

        except ImportError:
            # ragas not installed — fall back to local approximations below
            pass
        except Exception as exc:
            print(f"  [Warning] ragas ground-truth metrics failed ({exc}), using local approximation.")

    # --- Local fallback approximations (no ragas library needed) ---

    # CHECK 7 — FACTUAL CORRECTNESS (local approximation)
    # Split the answer into sentences and check each one against the ground truth tokens.
    # A sentence is "factually matched" if at least 30% of its words appear in the ground truth.
    answer_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.split()) >= 4]
    gt_tokens = set(_tokenize(ground_truth))

    matched_claims = 0
    for sent in answer_sentences:
        sent_tokens = set(_tokenize(sent))
        # A claim "matches" if it shares at least 30% of its words with the ground truth
        if sent_tokens and len(sent_tokens & gt_tokens) / len(sent_tokens) >= 0.30:
            matched_claims += 1

    # Factual correctness = fraction of answer sentences that align with ground truth
    factual_correctness = matched_claims / max(len(answer_sentences), 1)

    # CHECK 8 — NOISE SENSITIVITY (local approximation)
    # Compare how much the answer tracks the raw context vs. the ideal ground truth.
    # If the answer follows noisy/irrelevant context much more than the ground truth,
    # it suggests noisy sources pushed the answer away from what "ideal" looks like.
    answer_tokens = set(_tokenize(answer))
    context_tokens = set(_tokenize(" ".join(contexts)))

    # Fraction of the answer that overlaps with all retrieved context (may include noise)
    context_overlap = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
    # Fraction of the answer that overlaps with the ideal ground truth
    gt_overlap = len(answer_tokens & gt_tokens) / max(len(answer_tokens), 1)

    # Noise sensitivity = how much the answer leans on raw context vs. ideal answer
    # High gap = answer was pulled by noisy sources away from what ideal looks like
    noise_sensitivity = round(min(1.0, max(0.0, context_overlap - gt_overlap)), 3)

    # CHECK 9 — SEMANTIC SIMILARITY (local approximation)
    # Use the same hash-based embedding already used in vectorstore.py (no API needed).
    # Cosine similarity between the answer and the ground truth embedding vectors.
    answer_vec = _simple_embedding(answer)
    gt_vec = _simple_embedding(ground_truth)
    raw_similarity = _cosine_similarity(answer_vec, gt_vec)
    # Cosine similarity ranges from -1 to 1; scale to 0-1 for readability
    semantic_similarity = round((raw_similarity + 1) / 2, 3)

    return {
        "method": "local_approximation",
        "ground_truth_used": ground_truth[:150],
        "factual_correctness": round(factual_correctness, 3),
        "noise_sensitivity": noise_sensitivity,
        "semantic_similarity": semantic_similarity,
    }


def print_ragas_ground_truth_summary(kpi_name: str, result: Dict[str, Any]) -> None:
    """
    Print a plain-English summary of the three ground-truth-based checks.
    Ground truth = the Score-5 rubric description (the 'ideal answer' benchmark).
    """
    print(f"\n  [Ground Truth Quality Checks (7-9) — {kpi_name}]")
    print(f"    Ideal answer benchmark : \"{result.get('ground_truth_used', 'N/A')}\"")
    print(f"    Method used            : {result.get('method', 'unknown')}")
    print()

    # Check 7: Factual correctness — claim-level match to ideal
    fc = result.get("factual_correctness")
    if fc is not None:
        fc_label = "Strong match" if fc >= 0.7 else "Partial match" if fc >= 0.4 else "Weak match"
        print(f"    7. Factual accuracy vs ideal  : {fc:.0%} — {fc_label}")
        print(f"       (How many of the AI's claims match what a top-score answer should say?)")

    # Check 8: Noise sensitivity — lower is better
    ns = result.get("noise_sensitivity")
    if ns is not None:
        ns_label = "Not affected" if ns <= 0.2 else "Slightly affected" if ns <= 0.4 else "Noticeably misled by poor sources"
        print(f"    8. Noise impact from sources  : {ns:.0%} — {ns_label}")
        print(f"       (Did low-quality or irrelevant sources pull the answer off-track? Lower = better)")

    # Check 9: Semantic similarity — meaning-level match to ideal
    ss = result.get("semantic_similarity")
    if ss is not None:
        ss_label = "Very similar" if ss >= 0.75 else "Somewhat similar" if ss >= 0.5 else "Diverges from ideal"
        print(f"    9. Overall meaning match      : {ss:.0%} — {ss_label}")
        print(f"       (Does the answer capture the same ideas as an ideal answer would?)")


# ==============================================================================
# MASTER FUNCTION — Run all 9 evaluations for a single KPI
# ==============================================================================

def evaluate_single_kpi(
    kpi_name: str,
    question: str,
    answer: str,
    contexts: List[str],
    rubric: Optional[List[str]] = None,
    hallucination_threshold: float = 0.4,
    run_llm_judge: bool = True,
    verbose: bool = True,
    retrieved_chunk_ids: Optional[List[str]] = None,
    trace_id: Optional[str] = None,
    trace: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run all 9 RAG evaluation checks for a single KPI result.

    This is the main function to call for each KPI after scoring.
    It does not modify any scores — it only measures and reports quality.

    The Score-5 rubric description is automatically extracted from the rubric list
    and used as the ground truth for checks 7, 8, and 9. No manual input needed.

    Args:
        kpi_name:               Display name of the KPI being evaluated
        question:               The KPI question asked
        answer:                 The AI's generated rationale/answer
        contexts:               List of evidence text chunks retrieved
        rubric:                 The KPI's rubric list from kpis.yaml (used to extract
                                the Score-5 description as ground truth for checks 7-9)
        hallucination_threshold: Flag answers above this unsupported-content level
        run_llm_judge:          Whether to run the LLM-as-judge check
        retrieved_chunk_ids:    Ordered chunk IDs for Hit Rate / MRR / nDCG (vs DB golden_chunks)
        trace_id / trace:       LangFuse trace for scores (BERTScore, retrieval, CoT)

    Returns:
        Dictionary containing results from all 9 evaluations.
    """

    eval_flags = dict(DEFAULT_FEATURE_FLAGS)

    # Extract the Score-5 rubric line as the ground truth benchmark
    # e.g. "Clear, coherent AI strategy with priorities and outcomes."
    ground_truth = extract_ground_truth_from_rubric(rubric)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Evaluating KPI: {kpi_name}")
        print(f"  Question: {question[:80]}{'...' if len(question) > 80 else ''}")
        print(f"  Evidence chunks available: {len(contexts)}")
        if ground_truth:
            print(f"  Ideal answer benchmark : \"{ground_truth[:80]}{'...' if len(ground_truth) > 80 else ''}\"")
        else:
            print(f"  Ideal answer benchmark : Not available (checks 7-9 will be skipped)")
        print(f"{'='*60}")

    results: Dict[str, Any] = {"kpi_name": kpi_name}

    # --- 1. RAGAS (core metrics: faithfulness, answer relevancy, context precision/recall) ---
    # Pass ground truth so RAGAS can also measure context recall if available
    ragas_result = evaluate_ragas(question, answer, contexts, ground_truth)
    results["ragas"] = ragas_result
    if verbose:
        print_ragas_summary(kpi_name, ragas_result)

    # --- 2. LLM as a Judge ---
    if run_llm_judge:
        judge_result = evaluate_llm_as_judge(question, answer, contexts, verbose=verbose)
    else:
        judge_result = {"llm_used": False, "feedback": "Skipped."}
    results["llm_judge"] = judge_result
    if verbose:
        print_judge_summary(kpi_name, judge_result)

    # --- 3. Recall@k ---
    recall_k_result = evaluate_recall_at_k(question, contexts)
    results["recall_at_k"] = recall_k_result
    if verbose:
        print_recall_at_k_summary(kpi_name, recall_k_result)

    # --- 4. F1 Score ---
    # Compare answer against combined evidence text as reference
    reference_for_f1 = ground_truth if ground_truth else " ".join(contexts)
    f1_result = evaluate_f1(answer, reference_for_f1)
    results["f1"] = f1_result
    if verbose:
        print_f1_summary(kpi_name, f1_result)

    # --- 5. Hallucination Check ---
    hallucination_result = evaluate_hallucination(
        answer, contexts,
        hallucination_threshold=hallucination_threshold,
        use_llm=run_llm_judge,
        verbose=verbose,
    )
    results["hallucination"] = hallucination_result
    if verbose:
        print_hallucination_summary(kpi_name, hallucination_result)

    # --- 6. MMR Diversity Check ---
    mmr_result = evaluate_mmr(question, contexts)
    results["mmr"] = mmr_result
    if verbose:
        print_mmr_summary(kpi_name, mmr_result)

    # --- BERTScore + CoT eval (reference = top-3 retrieved chunk texts) ---
    rubric_text = "\n".join(rubric) if rubric else ""
    results["bertscore_f1"] = None
    results["low_semantic_grounding"] = None
    results["cot_eval"] = None
    if contexts:
        try:
            bs = compute_bertscore(
                answer,
                contexts,
                trace_id=trace_id,
                trace=trace,
                config=eval_flags,
            )
            results["bertscore_f1"] = bs
            results["low_semantic_grounding"] = bool(bs is not None and bs < 0.75)
        except Exception:
            pass
        try:
            cot = run_cot_eval(
                kpi_name,
                rubric_text or "No rubric text available.",
                contexts,
                answer,
                trace_id=trace_id,
                trace=trace,
                config=eval_flags,
            )
            results["cot_eval"] = cot
        except Exception:
            pass

    # --- 7, 8, 9. Ground-truth-based checks (Factual Correctness, Noise Sensitivity, Semantic Similarity) ---
    # These only run if a ground truth could be extracted from the Score-5 rubric.
    # If no rubric was provided, this section is skipped gracefully.
    if ground_truth:
        gt_result = evaluate_ragas_with_ground_truth(question, answer, contexts, ground_truth)
    else:
        # No rubric available — mark as skipped so the summary knows not to display these
        gt_result = {
            "method": "skipped",
            "ground_truth_used": None,
            "factual_correctness": None,
            "noise_sensitivity": None,
            "semantic_similarity": None,
        }
    results["ground_truth_checks"] = gt_result
    if verbose:
        print_ragas_ground_truth_summary(kpi_name, gt_result)
        # --- Overall executive summary ---
        _print_overall_verdict(kpi_name, results)

    # --- Retrieval vs golden chunks (Hit Rate, MRR, nDCG) ---
    results["retrieval_metrics"] = None
    if retrieved_chunk_ids:
        try:
            from app.langfuse_client import log_score_to_trace
            from app.retrieval_metrics import compute_hit_rate, compute_mrr, compute_ndcg

            hr = compute_hit_rate(kpi_name, retrieved_chunk_ids)
            mrr_v = compute_mrr(kpi_name, retrieved_chunk_ids)
            ndcg_v = compute_ndcg(kpi_name, retrieved_chunk_ids)
            results["retrieval_metrics"] = {
                "hit_rate": hr,
                "mrr": mrr_v,
                "ndcg": ndcg_v,
            }
            if trace_id:
                if hr is not None:
                    log_score_to_trace(trace_id, "retrieval_hit_rate", float(hr))
                if mrr_v is not None:
                    log_score_to_trace(trace_id, "retrieval_mrr", float(mrr_v))
                if ndcg_v is not None:
                    log_score_to_trace(trace_id, "retrieval_ndcg", float(ndcg_v))
        except Exception:
            pass

    return results


def _print_overall_verdict(kpi_name: str, results: Dict[str, Any]) -> None:
    """
    Print a one-page executive summary for this KPI's evaluation.
    Uses plain language — no technical jargon.
    """
    ragas = results.get("ragas", {})
    judge = results.get("llm_judge", {})
    f1 = results.get("f1", {})
    hall = results.get("hallucination", {})
    mmr = results.get("mmr", {})
    gt = results.get("ground_truth_checks", {})

    print(f"\n  ── EXECUTIVE SUMMARY: {kpi_name} ──")
    print(f"  How well did the AI research and answer this KPI?")
    print()

    # Answer quality — from RAGAS faithfulness (which also covers context precision/recall)
    faith = ragas.get("faithfulness")
    if faith is not None:
        quality = "Excellent" if faith >= 0.8 else "Good" if faith >= 0.6 else "Needs review"
        print(f"  Answer quality      : {quality} ({faith:.0%} grounded in evidence)")

    # Source relevance — from RAGAS context_precision
    ctx_prec = ragas.get("context_precision")
    if ctx_prec is not None:
        usefulness = "Highly relevant" if ctx_prec >= 0.7 else "Moderately relevant" if ctx_prec >= 0.4 else "Low relevance"
        print(f"  Sources retrieved   : {usefulness} ({ctx_prec:.0%} of sources were on-topic)")

    # Accuracy vs reference
    f1_val = f1.get("f1")
    if f1_val is not None:
        accuracy = "High" if f1_val >= 0.6 else "Moderate" if f1_val >= 0.35 else "Low"
        print(f"  Answer accuracy     : {accuracy} (F1 = {f1_val:.0%})")

    # Hallucination flag
    if hall.get("is_flagged"):
        print(f"    Caution: {hall['hallucination_score']:.0%} of the answer may not be "
              f"fully supported by retrieved evidence. Recommend manual review.")
    else:
        print(f"  Reliability: Answer appears well-supported by evidence "
              f"({hall.get('hallucination_score', 0):.0%} unsupported content)")

    # Source diversity
    div = mmr.get("diversity_score")
    if div is not None:
        diversity = "Diverse" if div >= 0.6 else "Moderate variety" if div >= 0.35 else "Repetitive"
        print(f"  Source variety      : {diversity} ({div:.0%} — higher means less repetition)")

    # AI judge verdict
    if judge.get("llm_used") and judge.get("overall"):
        stars = "★" * judge["overall"] + "☆" * (5 - judge["overall"])
        print(f"  AI reviewer rating  : {stars} ({judge['overall']}/5) — {judge.get('feedback', '')}")

    # Ground-truth-based checks (only shown if ground truth was available)
    if gt.get("method") not in (None, "skipped"):
        print()
        print(f"  Compared against ideal answer: \"{gt.get('ground_truth_used', '')[:70]}...\"")

        fc = gt.get("factual_correctness")
        if fc is not None:
            fc_label = "Strong" if fc >= 0.7 else "Partial" if fc >= 0.4 else "Weak"
            print(f"  Factual accuracy    : {fc_label} ({fc:.0%} of claims match the ideal answer)")

        ns = gt.get("noise_sensitivity")
        if ns is not None:
            ns_label = "Not affected" if ns <= 0.2 else "Slightly" if ns <= 0.4 else "Significantly misled"
            print(f"  Noise impact        : {ns_label} by low-quality sources ({ns:.0%} — lower is better)")

        ss = gt.get("semantic_similarity")
        if ss is not None:
            ss_label = "Very close" if ss >= 0.75 else "Moderate" if ss >= 0.5 else "Low"
            print(f"  Meaning similarity  : {ss_label} to what an ideal answer would say ({ss:.0%})")

    print()


# ==============================================================================
# BATCH FUNCTION — Run evaluations across all KPI results in a report
# ==============================================================================

def run_all_evaluations(
    kpi_results: List[Any],
    sources: List[Dict],
    kpi_definitions: Optional[List[Any]] = None,
    hallucination_threshold: float = 0.4,
    run_llm_judge: bool = True,
    kpi_ids_to_evaluate: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Run all 9 RAG evaluation checks across a full set of KPI results.

    Call this after the main pipeline has finished scoring.
    Pass in the list of KPIDriverResult objects, the fetched sources, and
    optionally the KPI definitions (used to extract Score-5 ground truths
    for checks ).

    Args:
        kpi_results:              List of KPIDriverResult objects from the pipeline
        sources:                  List of source dicts (with 'text' field) from pipeline
        kpi_definitions:          Optional list of KPIDefinition objects from kpi_catalog.
                                  Used to look up each KPI's rubric and extract the
                                  Score-5 line as ground truth. If not provided, checks
                                  7, 8 and 9 will be skipped.
        hallucination_threshold:  Flag KPIs where unsupported content exceeds this level
        run_llm_judge:            Whether to use the LLM as a judge (uses API credits)
        kpi_ids_to_evaluate:      Optional list of specific KPI IDs to evaluate
                                  (if None, evaluates all rubric KPIs)

    Returns:
        Dictionary mapping kpi_id -> evaluation results for all checks.
    """

    if verbose:
        print("\n" + "="*60)
        print("  RAG EVALUATION REPORT")
        print("  Checking quality of AI research and answer generation")
        print("="*60)

    # Build lookups from kpi_id -> rubric and kpi_id -> question
    rubric_by_kpi_id: Dict[str, List[str]] = {}
    question_by_kpi_id: Dict[str, str] = {}
    if kpi_definitions:
        for kpi_def in kpi_definitions:
            kpi_def_id = getattr(kpi_def, "kpi_id", None)
            kpi_def_rubric = getattr(kpi_def, "rubric", None)
            kpi_def_question = getattr(kpi_def, "question", None)
            if kpi_def_id:
                if kpi_def_rubric:
                    rubric_by_kpi_id[kpi_def_id] = kpi_def_rubric
                if kpi_def_question:
                    question_by_kpi_id[kpi_def_id] = kpi_def_question

    if verbose:
        if rubric_by_kpi_id:
            print(f"  Ground truth available for {len(rubric_by_kpi_id)} KPIs (Score-5 rubric descriptions)")
        else:
            print("  No KPI definitions provided — checks 7, 8, 9 will be skipped.")

    # Combine all source texts into one pool for context building
    source_text_by_id: Dict[str, str] = {}
    for s in sources:
        sid = s.get("source_id", "")
        if sid:
            source_text_by_id[sid] = s.get("text", "")

    show_progress = os.getenv("RAG_EVAL_PROGRESS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )

    rubric_total = sum(
        1
        for k in kpi_results
        if k.type == "rubric"
        and (not kpi_ids_to_evaluate or k.kpi_id in kpi_ids_to_evaluate)
    )
    if show_progress:
        print(
            f"[eval_rag] {rubric_total} rubric KPIs to evaluate. "
            f"Each KPI: RAGAS + judge + recall/F1/hallucination/MMR + BERTScore + CoT. "
            f"This can take a long time — progress below.",
            flush=True,
        )

    all_results: Dict[str, Dict] = {}
    flagged_kpis: List[str] = []
    evaluated_count = 0

    for kpi_result in kpi_results:
        kpi_id = kpi_result.kpi_id
        kpi_name = kpi_id  # Use ID as name fallback

        # Skip if not in the requested list
        if kpi_ids_to_evaluate and kpi_id not in kpi_ids_to_evaluate:
            continue

        # Only evaluate rubric KPIs (quant KPIs don't have LLM-generated answers)
        if kpi_result.type != "rubric":
            continue

        # Look up the rubric for this KPI so evaluate_single_kpi can extract
        # the Score-5 description as the ground truth benchmark
        rubric_for_this_kpi = rubric_by_kpi_id.get(kpi_id)

        # Build context chunks from the KPI's citations
        contexts: List[str] = []
        for citation in (kpi_result.citations or []):
            if citation.quote:
                contexts.append(citation.quote)
            # Also pull from the full source text if available
            full_text = source_text_by_id.get(citation.source_id, "")
            if full_text and full_text not in contexts:
                contexts.append(full_text[:500])

        # If no citations but sources exist, use first few sources as context
        if not contexts and source_text_by_id:
            contexts = [text[:500] for text in list(source_text_by_id.values())[:5]]

        # Use the KPI's actual question from kpi_definitions; fall back to kpi_id
        kpi_question = question_by_kpi_id.get(kpi_id, kpi_id)

        # Chunk IDs for Hit Rate / MRR / nDCG (match DB golden_chunks.kpi_id)
        retrieved_chunk_ids: List[str] = []
        for citation in kpi_result.citations or []:
            sid = getattr(citation, "source_id", None) or ""
            if sid:
                retrieved_chunk_ids.append(f"{sid}::chunk_0")
        if not retrieved_chunk_ids and source_text_by_id:
            retrieved_chunk_ids = [
                f"{sid}::chunk_0" for sid in list(source_text_by_id.keys())[:25]
            ]

        trace = create_trace(
            name=f"rag_eval_{kpi_id}",
            metadata={"kpi_id": kpi_id, "module": "eval_rag.run_all_evaluations"},
            tags=["rag_eval"],
        )
        tid = get_trace_id(trace)

        # Run all 9 evaluations for this KPI
        eval_result = evaluate_single_kpi(
            kpi_name=kpi_name,
            question=kpi_question,
            answer=kpi_result.rationale or "",
            contexts=contexts,
            rubric=rubric_for_this_kpi,
            hallucination_threshold=hallucination_threshold,
            run_llm_judge=run_llm_judge,
            verbose=verbose,
            retrieved_chunk_ids=retrieved_chunk_ids or None,
            trace_id=tid,
            trace=trace,
        )

        all_results[kpi_id] = eval_result
        evaluated_count += 1

        if show_progress:
            print(
                f"[eval_rag] {evaluated_count}/{rubric_total} done  kpi_id={kpi_id[:72]}",
                flush=True,
            )

        if eval_result.get("hallucination", {}).get("is_flagged"):
            flagged_kpis.append(kpi_name)

        # Pause between KPIs (rate limits)
        if run_llm_judge:
            time.sleep(float(os.getenv("RAG_EVAL_JUDGE_SLEEP_SEC", "2")))

    # --- Final batch summary (only shown in verbose / standalone mode) ---
    if verbose:
        print("\n" + "="*60)
        print("  OVERALL EVALUATION COMPLETE")
        print("="*60)
        print(f"  KPIs evaluated          : {evaluated_count}")
        print(f"  KPIs with concerns      : {len(flagged_kpis)}")
        if flagged_kpis:
            print("  KPIs flagged for review :")
            for name in flagged_kpis:
                print(f"   {name}")
        else:
            print("  No KPIs flagged — all answers appear well-supported.")
        print()
        print("  What this means:")
        if evaluated_count == 0:
            print("  No rubric KPIs were available to evaluate.")
        elif len(flagged_kpis) == 0:
            print("  The AI's research and answers are well-grounded in evidence.")
            print("  Confidence in the assessment results is high.")
        elif len(flagged_kpis) <= evaluated_count * 0.2:
            print(f"  Most answers are well-supported. {len(flagged_kpis)} KPI(s) should")
            print("  be manually reviewed before presenting to stakeholders.")
        else:
            print(f"  {len(flagged_kpis)} of {evaluated_count} KPIs contain answers that may")
            print("  go beyond what the evidence supports. A thorough manual review")
            print("  is recommended before sharing results externally.")
        print("="*60)

    return all_results
