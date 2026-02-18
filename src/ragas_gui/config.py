"""Metrics registry, metric categories, and application constants.

Ragas 0.4+ uses ``ragas.metrics.collections`` as the canonical import path.
Each entry maps a human-readable name to a *class* (not an instance) so the
GUI can instantiate metrics with user-chosen parameters (LLM, embeddings, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------


class MetricCategory(str, Enum):
    """Logical grouping shown in the GUI."""

    RAG_CORE = "RAG Core"
    RAG_CONTEXT = "RAG Context"
    ANSWER_QUALITY = "Answer Quality"
    AGENT_TOOL = "Agent & Tool Use"
    NLP_CLASSIC = "Classic NLP"
    CUSTOM = "Custom / Rubrics"
    MULTI_MODAL = "Multi-Modal"


@dataclass(frozen=True)
class MetricInfo:
    """Descriptor for a single Ragas metric."""

    name: str
    display_name: str
    category: MetricCategory
    import_path: str  # e.g. "ragas.metrics._faithfulness"
    class_name: str  # e.g. "Faithfulness"
    needs_llm: bool = True
    needs_embeddings: bool = False
    description: str = ""
    init_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def quick_start(self) -> bool:
        """Whether this metric appears in Quick Start mode."""
        return self.category in {
            MetricCategory.RAG_CORE,
            MetricCategory.RAG_CONTEXT,
        }


# ---------------------------------------------------------------------------
# Full metric catalogue
# ---------------------------------------------------------------------------

METRIC_CATALOGUE: list[MetricInfo] = [
    # -- RAG Core -----------------------------------------------------------
    MetricInfo(
        name="faithfulness",
        display_name="Faithfulness",
        category=MetricCategory.RAG_CORE,
        import_path="ragas.metrics._faithfulness",
        class_name="Faithfulness",
        description="Measures factual consistency of the answer against the given context.",
    ),
    MetricInfo(
        name="answer_relevancy",
        display_name="Answer Relevancy",
        category=MetricCategory.RAG_CORE,
        import_path="ragas.metrics._answer_relevance",
        class_name="AnswerRelevancy",
        needs_embeddings=True,
        description="Measures how relevant the answer is to the question.",
    ),
    MetricInfo(
        name="response_groundedness",
        display_name="Response Groundedness",
        category=MetricCategory.RAG_CORE,
        import_path="ragas.metrics.collections.response_groundedness.metric",
        class_name="ResponseGroundedness",
        description="Measures how well the response is grounded in the context.",
    ),
    # -- RAG Context --------------------------------------------------------
    MetricInfo(
        name="context_precision",
        display_name="Context Precision",
        category=MetricCategory.RAG_CONTEXT,
        import_path="ragas.metrics._context_precision",
        class_name="ContextPrecision",
        description="Measures signal-to-noise ratio of retrieved contexts.",
    ),
    MetricInfo(
        name="context_recall",
        display_name="Context Recall",
        category=MetricCategory.RAG_CONTEXT,
        import_path="ragas.metrics._context_recall",
        class_name="ContextRecall",
        description="Measures if the contexts contain all the information needed.",
    ),
    MetricInfo(
        name="context_entity_recall",
        display_name="Context Entity Recall",
        category=MetricCategory.RAG_CONTEXT,
        import_path="ragas.metrics._context_entities_recall",
        class_name="ContextEntityRecall",
        description="Measures recall of named entities from contexts.",
    ),
    MetricInfo(
        name="context_relevance",
        display_name="Context Relevance",
        category=MetricCategory.RAG_CONTEXT,
        import_path="ragas.metrics.collections.context_relevance.metric",
        class_name="ContextRelevance",
        description="Measures how relevant the retrieved context is to the question.",
    ),
    MetricInfo(
        name="noise_sensitivity",
        display_name="Noise Sensitivity",
        category=MetricCategory.RAG_CONTEXT,
        import_path="ragas.metrics._noise_sensitivity",
        class_name="NoiseSensitivity",
        description="Measures how sensitive the answer is to noisy contexts.",
    ),
    # -- Answer Quality -----------------------------------------------------
    MetricInfo(
        name="answer_correctness",
        display_name="Answer Correctness",
        category=MetricCategory.ANSWER_QUALITY,
        import_path="ragas.metrics._answer_correctness",
        class_name="AnswerCorrectness",
        needs_embeddings=True,
        description="Measures factual correctness of the answer against ground truth.",
    ),
    MetricInfo(
        name="answer_similarity",
        display_name="Answer Similarity",
        category=MetricCategory.ANSWER_QUALITY,
        import_path="ragas.metrics._answer_similarity",
        class_name="AnswerSimilarity",
        needs_embeddings=True,
        needs_llm=False,
        description="Semantic similarity between answer and ground truth.",
    ),
    MetricInfo(
        name="factual_correctness",
        display_name="Factual Correctness",
        category=MetricCategory.ANSWER_QUALITY,
        import_path="ragas.metrics._factual_correctness",
        class_name="FactualCorrectness",
        description="Verifies factual accuracy of claims in the answer.",
    ),
    MetricInfo(
        name="answer_accuracy",
        display_name="Answer Accuracy",
        category=MetricCategory.ANSWER_QUALITY,
        import_path="ragas.metrics.collections.answer_accuracy.metric",
        class_name="AnswerAccuracy",
        description="NVIDIA-style answer accuracy metric.",
    ),
    MetricInfo(
        name="summarization_score",
        display_name="Summarization Score",
        category=MetricCategory.ANSWER_QUALITY,
        import_path="ragas.metrics.collections.summary_score.metric",
        class_name="SummaryScore",
        description="Evaluates the quality of a summary.",
    ),
    # -- Agent & Tool Use ---------------------------------------------------
    MetricInfo(
        name="tool_call_accuracy",
        display_name="Tool Call Accuracy",
        category=MetricCategory.AGENT_TOOL,
        import_path="ragas.metrics._tool_call_accuracy",
        class_name="ToolCallAccuracy",
        needs_llm=False,
        description="Accuracy of tool/function calls made by the agent.",
    ),
    MetricInfo(
        name="tool_call_f1",
        display_name="Tool Call F1",
        category=MetricCategory.AGENT_TOOL,
        import_path="ragas.metrics.collections.tool_call_f1.metric",
        class_name="ToolCallF1",
        needs_llm=False,
        description="F1 score for tool/function call selection.",
    ),
    MetricInfo(
        name="agent_goal_accuracy",
        display_name="Agent Goal Accuracy",
        category=MetricCategory.AGENT_TOOL,
        import_path="ragas.metrics._goal_accuracy",
        class_name="AgentGoalAccuracyWithReference",
        description="Whether the agent achieved its conversational goal.",
    ),
    MetricInfo(
        name="topic_adherence",
        display_name="Topic Adherence",
        category=MetricCategory.AGENT_TOOL,
        import_path="ragas.metrics.collections.topic_adherence.metric",
        class_name="TopicAdherence",
        description="Measures if the agent stayed on topic.",
    ),
    # -- Classic NLP --------------------------------------------------------
    MetricInfo(
        name="bleu_score",
        display_name="BLEU Score",
        category=MetricCategory.NLP_CLASSIC,
        import_path="ragas.metrics._bleu_score",
        class_name="BleuScore",
        needs_llm=False,
        description="BLEU score between answer and ground truth.",
    ),
    MetricInfo(
        name="rouge_score",
        display_name="ROUGE Score",
        category=MetricCategory.NLP_CLASSIC,
        import_path="ragas.metrics._rouge_score",
        class_name="RougeScore",
        needs_llm=False,
        description="ROUGE score between answer and ground truth.",
    ),
    MetricInfo(
        name="string_presence",
        display_name="String Presence",
        category=MetricCategory.NLP_CLASSIC,
        import_path="ragas.metrics._string",
        class_name="StringPresence",
        needs_llm=False,
        description="Checks if specific strings are present in the answer.",
    ),
    MetricInfo(
        name="exact_match",
        display_name="Exact Match",
        category=MetricCategory.NLP_CLASSIC,
        import_path="ragas.metrics._string",
        class_name="ExactMatch",
        needs_llm=False,
        description="Exact string match between answer and ground truth.",
    ),
    # -- Custom / Rubrics ---------------------------------------------------
    MetricInfo(
        name="simple_criteria",
        display_name="Simple Criteria",
        category=MetricCategory.CUSTOM,
        import_path="ragas.metrics._simple_criteria",
        class_name="SimpleCriteriaScore",
        description="Score based on a user-defined rubric prompt.",
    ),
    MetricInfo(
        name="domain_rubrics",
        display_name="Domain-Specific Rubrics",
        category=MetricCategory.CUSTOM,
        import_path="ragas.metrics._domain_specific_rubrics",
        class_name="RubricsScoreWithoutReference",
        description="Score using domain-specific rubrics.",
    ),
    MetricInfo(
        name="instance_rubrics",
        display_name="Instance-Specific Rubrics",
        category=MetricCategory.CUSTOM,
        import_path="ragas.metrics._instance_specific_rubrics",
        class_name="InstanceRubrics",
        description="Score using per-instance rubrics.",
    ),
    MetricInfo(
        name="aspect_critic",
        display_name="Aspect Critic",
        category=MetricCategory.CUSTOM,
        import_path="ragas.metrics._aspect_critic",
        class_name="AspectCritic",
        description="Binary yes/no evaluation on a user-defined aspect.",
    ),
    # -- Multi-Modal --------------------------------------------------------
    MetricInfo(
        name="multi_modal_faithfulness",
        display_name="Multi-Modal Faithfulness",
        category=MetricCategory.MULTI_MODAL,
        import_path="ragas.metrics._multi_modal_faithfulness",
        class_name="MultiModalFaithfulness",
        description="Faithfulness evaluation for multi-modal inputs.",
    ),
    MetricInfo(
        name="multi_modal_relevance",
        display_name="Multi-Modal Relevance",
        category=MetricCategory.MULTI_MODAL,
        import_path="ragas.metrics._multi_modal_relevance",
        class_name="MultiModalRelevance",
        description="Relevance evaluation for multi-modal inputs.",
    ),
]

# Legacy convenience dict (display_name -> MetricInfo)
METRIC_REGISTRY: dict[str, MetricInfo] = {m.display_name: m for m in METRIC_CATALOGUE}

# Quick Start defaults
QUICK_START_METRICS: list[str] = [
    m.display_name for m in METRIC_CATALOGUE if m.quick_start
]

# ---- Lazy import cache for metric classes ---------------------------------

_METRICS_CACHE: dict[str, type] = {}


def _import_metric_class(info: MetricInfo) -> type:
    """Lazily import and cache a metric class."""
    if info.name not in _METRICS_CACHE:
        import importlib

        mod = importlib.import_module(info.import_path)
        _METRICS_CACHE[info.name] = getattr(mod, info.class_name)
    return _METRICS_CACHE[info.name]


def get_metric_class(display_name: str) -> type:
    """Return the metric class for a given display name."""
    info = METRIC_REGISTRY[display_name]
    return _import_metric_class(info)


def get_metric_info(display_name: str) -> MetricInfo:
    """Return the MetricInfo descriptor for a given display name."""
    return METRIC_REGISTRY[display_name]


def list_metrics_by_category() -> dict[MetricCategory, list[MetricInfo]]:
    """Group all metrics by category for UI rendering."""
    groups: dict[MetricCategory, list[MetricInfo]] = {}
    for m in METRIC_CATALOGUE:
        groups.setdefault(m.category, []).append(m)
    return groups
