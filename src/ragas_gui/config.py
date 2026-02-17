"""Metrics registry and application constants."""

from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

METRICS = {
    "Faithfulness": faithfulness,
    "Answer Relevancy": answer_relevancy,
    "Context Precision": context_precision,
    "Context Recall": context_recall,
}
