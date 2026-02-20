"""ragas_gui -- Streamlit-based evaluation UI for Ragas RAG pipelines."""

from ragas_gui.config import (
    METRIC_CATALOGUE,
    METRIC_REGISTRY,
    QUICK_START_METRICS,
    MetricCategory,
    MetricInfo,
    get_metric_class,
    get_metric_info,
    list_metrics_by_category,
)
from ragas_gui.data import (
    build_ragas_dataset,
    load_uploaded_file,
    parse_contexts,
    validate_columns,
)
from ragas_gui.evaluation import run_evaluation
from ragas_gui.llm_config import (
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_MODELS,
    EmbeddingConfig,
    EmbeddingProvider,
    LLMConfig,
    Provider,
    RunSettings,
    test_embedding_connection,
    test_llm_connection,
)
from ragas_gui.i18n import (
    QUICK_START_PROVIDER_LABELS,
    QUICK_START_PROVIDERS,
    SUPPORTED_LANGUAGES,
    get_language,
    set_language,
    t,
)
from ragas_gui.telemetry import (
    EvaluationEvent,
    ExporterType,
    TelemetryConfig,
    TelemetryManager,
    TokenUsage,
)

__all__ = [
    "METRIC_CATALOGUE",
    "METRIC_REGISTRY",
    "QUICK_START_METRICS",
    "QUICK_START_PROVIDER_LABELS",
    "QUICK_START_PROVIDERS",
    "SUPPORTED_LANGUAGES",
    "CompatibilityMode",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EvaluationEvent",
    "ExporterType",
    "LLMConfig",
    "LLMProvider",
    "MetricCategory",
    "MetricInfo",
    "RunSettings",
    "TelemetryConfig",
    "TelemetryManager",
    "TokenUsage",
    "build_ragas_dataset",
    "get_language",
    "get_metric_class",
    "get_metric_info",
    "list_metrics_by_category",
    "load_uploaded_file",
    "parse_contexts",
    "run_evaluation",
    "set_language",
    "t",
    "validate_columns",
]
