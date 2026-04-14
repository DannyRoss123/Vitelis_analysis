"""
MLflow versioning integration for the Vitelis KPI benchmarking pipeline.

Logs per-KPI-run metadata to MLflow so every evaluation is reproducible:
  - prompt_hash          SHA256 of the full scoring prompt
  - embedding_model_version  model name + version string
  - chromadb_snapshot_id     fingerprint of the ChromaDB collection state
  - ragas_config             full RAGAS evaluation config dict
  - run_timestamp
  - company_id and kpi_id
  - langfuse_trace_id        for cross-referencing with LangFuse traces

All MLflow calls are wrapped in try/except so an MLflow failure (network
error, missing tracking server, package not installed) never blocks the
main pipeline.

Environment variables:
    MLFLOW_TRACKING_URI      — remote tracking server (optional; defaults to local)
    MLFLOW_EXPERIMENT_NAME   — experiment name (default: "vitelis-kpi-pipeline")
    VITELIS_ENV              — tag applied to every run ("dev" | "staging" | "prod")
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union


def _as_float(value: Any) -> Optional[float]:
    """Best-effort conversion of a value to float for MLflow metrics."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _get_mlflow() -> Optional[Any]:
    """
    Lazily import mlflow.

    Returns:
        The mlflow module, or None if the package is not installed.
    """
    try:
        import mlflow  # type: ignore

        return mlflow
    except ImportError:
        return None


def log_kpi_run(
    run_id: str,
    company_id: str,
    kpi_id: str,
    prompt_hash: str,
    embedding_model_version: str,
    chromadb_snapshot_id: str,
    ragas_config: Dict[str, Any],
    langfuse_trace_id: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Log all KPI evaluation metadata to an MLflow run.

    Creates a new MLflow run under the configured experiment, logs all
    parameters and tags, then closes the run. The returned MLflow run ID
    can be stored in LangFuse metadata for cross-referencing.

    Args:
        run_id: Pipeline run identifier (used in the run name).
        company_id: Target company identifier or domain string.
        kpi_id: KPI being evaluated.
        prompt_hash: SHA256 hex digest of the full scoring prompt.
        embedding_model_version: Embedding model name and version string
            (e.g. "openai-text-embedding-3-small-v1").
        chromadb_snapshot_id: SHA256 fingerprint of the ChromaDB collection
            state at the time of retrieval.
        ragas_config: Full RAGAS evaluation configuration dict. Each key is
            logged as a separate MLflow param prefixed with "ragas_".
        langfuse_trace_id: LangFuse trace ID for cross-referencing (optional).
        extra_params: Any additional key-value pairs to log as MLflow params.
        extra_metrics: Any additional key-value pairs to log as MLflow metrics
            (values must be numeric or convertible to float).

    Returns:
        MLflow run ID string on success, or None if MLflow is unavailable
        or logging fails.

    Side effects:
        Creates an MLflow run. All exceptions are caught and suppressed so
        pipeline execution is never blocked by MLflow failures.
    """
    mlflow = _get_mlflow()
    if not mlflow:
        return None

    try:
        env_tag = os.getenv("VITELIS_ENV", "dev")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "vitelis-kpi-pipeline")
        try:
            mlflow.set_experiment(experiment_name)
        except Exception:
            pass

        run_name = f"{company_id}__{kpi_id}__{run_id[:8]}"
        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow_run_id: str = active_run.info.run_id

            # ── Core identification params ────────────────────────────────
            mlflow.log_param("pipeline_run_id", run_id)
            mlflow.log_param("company_id", company_id)
            mlflow.log_param("kpi_id", kpi_id)
            mlflow.log_param(
                "run_timestamp",
                datetime.now(timezone.utc).isoformat(),
            )

            # ── Versioning params ─────────────────────────────────────────
            mlflow.log_param("prompt_hash", prompt_hash)
            mlflow.log_param("embedding_model_version", embedding_model_version)
            mlflow.log_param("chromadb_snapshot_id", chromadb_snapshot_id)

            # ── Cross-referencing ─────────────────────────────────────────
            if langfuse_trace_id:
                mlflow.log_param("langfuse_trace_id", langfuse_trace_id)

            # ── RAGAS config (prefix each key) ────────────────────────────
            for key, val in ragas_config.items():
                try:
                    mlflow.log_param(f"ragas_{key}", str(val)[:250])
                except Exception:
                    pass

            # ── Extra params ──────────────────────────────────────────────
            if extra_params:
                for key, val in (extra_params or {}).items():
                    try:
                        mlflow.log_param(str(key)[:50], str(val)[:250])
                    except Exception:
                        pass

            # ── Extra metrics (numeric) ────────────────────────────────────
            if extra_metrics:
                for key, val in (extra_metrics or {}).items():
                    try:
                        fval = _as_float(val)
                        if fval is None:
                            continue
                        mlflow.log_metric(str(key)[:50], fval)
                    except Exception:
                        pass

            # ── Tags ──────────────────────────────────────────────────────
            mlflow.set_tag("environment", env_tag)
            mlflow.set_tag("company_id", company_id)
            mlflow.set_tag("kpi_id", kpi_id)

        return mlflow_run_id

    except Exception:
        return None


def log_pipeline_run(
    run_id: str,
    company_id: str,
    overall_score: float,
    kpi_count: int,
    chromadb_snapshot_id: str,
    langfuse_trace_id: Optional[str] = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Log a top-level pipeline run summary to MLflow.

    Intended to be called once per pipeline execution (not per KPI).

    Args:
        run_id: Pipeline run identifier.
        company_id: Target company identifier or domain.
        overall_score: Aggregated overall score for the run.
        kpi_count: Number of KPIs evaluated.
        chromadb_snapshot_id: Collection fingerprint for this run.
        langfuse_trace_id: LangFuse trace ID for cross-referencing.
        extra_metrics: Additional numeric metrics to log alongside overall_score.

    Returns:
        MLflow run ID string or None on failure.

    Side effects:
        Creates an MLflow run. All exceptions are caught and suppressed.
    """
    mlflow = _get_mlflow()
    if not mlflow:
        return None

    try:
        env_tag = os.getenv("VITELIS_ENV", "dev")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "vitelis-kpi-pipeline")
        try:
            mlflow.set_experiment(experiment_name)
        except Exception:
            pass

        run_name = f"{company_id}__pipeline__{run_id[:8]}"
        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow_run_id = active_run.info.run_id

            mlflow.log_param("pipeline_run_id", run_id)
            mlflow.log_param("company_id", company_id)
            mlflow.log_param("kpi_count", kpi_count)
            mlflow.log_param("chromadb_snapshot_id", chromadb_snapshot_id)
            mlflow.log_param(
                "run_timestamp",
                datetime.now(timezone.utc).isoformat(),
            )
            mlflow.log_metric("overall_score", overall_score)
            if extra_metrics:
                for key, val in (extra_metrics or {}).items():
                    try:
                        fval = _as_float(val)
                        if fval is None:
                            continue
                        mlflow.log_metric(str(key)[:50], fval)
                    except Exception:
                        pass

            if langfuse_trace_id:
                mlflow.log_param("langfuse_trace_id", langfuse_trace_id)

            mlflow.set_tag("environment", env_tag)
            mlflow.set_tag("company_id", company_id)
            mlflow.set_tag("run_type", "pipeline_summary")

        return mlflow_run_id

    except Exception:
        return None
