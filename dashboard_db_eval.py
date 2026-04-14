"""
dashboard_db_eval.py
Run with: streamlit run dashboard_db_eval.py

This entry point is kept for backwards compatibility. The same Postgres UI lives
under **Postgres (live DB)** in `dashboard_rag_source.py`.
"""
from __future__ import annotations

import streamlit as st

from app.db_explorer_panel import render_postgres_explorer

st.set_page_config(page_title="DB RAG Evaluation", page_icon="🗄️", layout="wide")
st.title("🗄️ DB-backed RAG + Source Evaluation")
st.caption("Tip: use `streamlit run dashboard_rag_source.py` for YAML reports + this DB view in one app.")
render_postgres_explorer(embedded=False)
