"""
LangFuse client integration for the Vitelis KPI benchmarking pipeline.

Provides helper functions for creating traces, spans, logging scores,
and firing custom events. All functions degrade gracefully when LangFuse
is not configured (LANGFUSE_SECRET_KEY not set) or the langfuse package
is not installed.

The existing observability.py stub is preserved unchanged; this module
provides the real LangFuse wiring and is imported by score_extensions.py
and score_kpis.py.

Environment variables:
    LANGFUSE_SECRET_KEY  — required to enable real tracing
    LANGFUSE_PUBLIC_KEY  — recommended for full SDK auth
    LANGFUSE_HOST        — defaults to https://cloud.langfuse.com
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

_LANGFUSE_CLIENT: Optional[Any] = None
_LANGFUSE_AVAILABLE: Optional[bool] = None  # None = not yet determined


def get_langfuse_client() -> Optional[Any]:
    """
    Return the shared Langfuse client instance, initialising it on first call.

    Uses a module-level singleton so the SDK's background flush thread is not
    recreated on every scoring call.

    Returns:
        Langfuse client instance, or None if the SDK is unavailable or
        LANGFUSE_SECRET_KEY is not set.

    Side effects:
        Creates and caches a Langfuse instance on first successful call.
    """
    global _LANGFUSE_CLIENT, _LANGFUSE_AVAILABLE

    if _LANGFUSE_AVAILABLE is False:
        return None
    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not secret_key:
        _LANGFUSE_AVAILABLE = False
        return None

    try:
        from langfuse import Langfuse  # type: ignore

        _LANGFUSE_CLIENT = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )
        _LANGFUSE_AVAILABLE = True
        return _LANGFUSE_CLIENT
    except ImportError:
        _LANGFUSE_AVAILABLE = False
        return None
    except Exception:
        _LANGFUSE_AVAILABLE = False
        return None


def create_trace(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Optional[Any]:
    """
    Create a LangFuse trace for a pipeline operation.

    Args:
        name: Human-readable trace name (e.g. "kpi_scoring_run").
        metadata: Arbitrary key-value metadata dict attached to the trace.
        tags: List of string tags for filtering in the LangFuse UI.

    Returns:
        LangFuse Trace object, or None if LangFuse is unavailable.

    Side effects:
        Opens a new trace in the LangFuse backend.
    """
    client = get_langfuse_client()
    if not client:
        return None
    try:
        # Langfuse SDK v2 API
        if hasattr(client, "trace"):
            return client.trace(name=name, metadata=metadata or {}, tags=tags or [])

        # Langfuse SDK v3 API (no .trace method)
        trace_id = client.create_trace_id()
        obs = None
        try:
            obs = client.start_observation(
                trace_context={"trace_id": trace_id},
                name=name,
                as_type="chain",
                metadata={"metadata": metadata or {}, "tags": tags or []},
            )
        except Exception:
            obs = None
        return {"id": trace_id, "observation": obs}
    except Exception:
        return None


def get_trace_id(trace: Optional[Any]) -> Optional[str]:
    """
    Safely extract the trace ID from a LangFuse trace object.

    Args:
        trace: LangFuse Trace object (may be None).

    Returns:
        String trace ID, or None.
    """
    if trace is None:
        return None
    if isinstance(trace, dict):
        trace_id = trace.get("id")
        return str(trace_id) if trace_id else None
    try:
        return str(trace.id)
    except Exception:
        return None


def log_score_to_trace(
    trace_id: str,
    name: str,
    value: float,
    comment: Optional[str] = None,
) -> None:
    """
    Attach a named numeric score to a LangFuse trace.

    Args:
        trace_id: The LangFuse trace ID string.
        name: Score label (e.g. "retrieval_hit_rate", "faithfulness_gate").
        value: Numeric score value.
        comment: Optional free-text comment shown next to the score.

    Side effects:
        Posts the score to LangFuse. Silently no-ops if client is unavailable.
    """
    client = get_langfuse_client()
    if not client:
        return
    try:
        if hasattr(client, "score"):
            client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
            return
        if hasattr(client, "create_score"):
            client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
    except Exception:
        pass


def log_event_to_trace(
    trace: Any,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a custom event on a LangFuse trace.

    Args:
        trace: LangFuse Trace object (may be None).
        name: Event name (e.g. "score_attribution", "competitor_bleed_alert").
        metadata: Dict of event payload data.

    Side effects:
        Posts the event to LangFuse. Silently no-ops if trace is None or on error.
    """
    if not trace:
        return
    try:
        trace.event(name=name, metadata=metadata or {})
    except Exception:
        pass


def create_span_on_trace(
    trace: Any,
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Create a child span under a LangFuse trace.

    Args:
        trace: Parent LangFuse Trace object.
        name: Span label (e.g. "retrieval", "llm_scoring").
        metadata: Initial metadata dict for the span.

    Returns:
        LangFuse Span object, or None if trace is None or on error.
    """
    if not trace:
        return None
    try:
        # SDK v3 shim: trace is our dict wrapper
        if isinstance(trace, dict):
            client = get_langfuse_client()
            trace_id = trace.get("id")
            parent_obs = trace.get("observation")
            if not client or not trace_id:
                return None
            trace_context: Dict[str, Any] = {"trace_id": trace_id}
            parent_id = getattr(parent_obs, "id", None)
            if parent_id:
                trace_context["parent_observation_id"] = parent_id
            return client.start_observation(
                trace_context=trace_context,
                name=name,
                as_type="span",
                metadata=metadata or {},
            )

        # SDK v2 API
        return trace.span(name=name, metadata=metadata or {})
    except Exception:
        return None


def end_span(span: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Close a LangFuse span, optionally updating its metadata.

    Args:
        span: LangFuse Span object (may be None).
        metadata: Additional metadata to merge in before closing.

    Side effects:
        Ends the span in LangFuse. Silently no-ops on None or error.
    """
    if not span:
        return
    try:
        if metadata:
            span.update(metadata=metadata)
        span.end()
    except Exception:
        pass


def update_trace_metadata(trace: Any, metadata: Dict[str, Any]) -> None:
    """
    Merge additional metadata into an existing LangFuse trace.

    Args:
        trace: LangFuse Trace object (may be None).
        metadata: Dict of key-value pairs to attach.

    Side effects:
        Updates trace metadata in LangFuse.
    """
    if not trace:
        return
    try:
        if isinstance(trace, dict):
            obs = trace.get("observation")
            if obs and hasattr(obs, "update"):
                obs.update(metadata=metadata)
            return
        trace.update(metadata=metadata)
    except Exception:
        pass


def flush_langfuse() -> None:
    """
    Flush all pending LangFuse events to the backend.

    Should be called once at pipeline shutdown to ensure no data is lost.

    Side effects:
        Blocks briefly to drain the SDK's internal queue.
    """
    client = get_langfuse_client()
    if not client:
        return
    try:
        client.flush()
    except Exception:
        pass
