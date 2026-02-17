"""ragas_gui — Streamlit-based evaluation UI for Ragas RAG pipelines."""

from ragas_gui.config import METRICS
from ragas_gui.data import (
    build_ragas_dataset,
    load_uploaded_file,
    parse_contexts,
    validate_columns,
)

__all__ = [
    "METRICS",
    "build_ragas_dataset",
    "load_uploaded_file",
    "parse_contexts",
    "validate_columns",
]
