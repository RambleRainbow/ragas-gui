"""Evaluation runner — the glue between the GUI and ``ragas.evaluate()``.

Handles:
  - Metric instantiation from ``MetricInfo`` descriptors + user LLM config.
  - Building the ``EvaluationDataset`` (new API) or HF ``Dataset`` (legacy).
  - Invoking ``ragas.evaluate()`` with full parameter support.
  - Wrapping the call in telemetry spans.
"""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from ragas_gui.config import MetricInfo, get_metric_class
from ragas_gui.data import build_ragas_dataset
from ragas_gui.llm_config import (
    EmbeddingConfig,
    LLMConfig,
    RunSettings,
    build_embeddings,
    build_llm,
    build_run_config,
)
from ragas_gui.telemetry import EvaluationEvent, TelemetryManager, TokenUsage


def instantiate_metrics(
    metric_infos: list[MetricInfo],
    llm: Any | None = None,
    embeddings: Any | None = None,
) -> list[Any]:
    """Create live metric instances from descriptors.

    The ``llm`` and ``embeddings`` args are injected into metrics that need
    them (according to ``MetricInfo.needs_llm`` / ``needs_embeddings``).
    """
    instances: list[Any] = []
    for info in metric_infos:
        cls = get_metric_class(info.display_name)
        # Try to instantiate with no args (ragas compat objects)
        try:
            instance = cls()
        except TypeError:
            # Some metrics require llm as positional arg
            if info.needs_llm and llm is not None:
                instance = cls(llm=llm)
            else:
                instance = cls()

        # Override LLM / embeddings if provided
        if info.needs_llm and llm is not None and hasattr(instance, "llm"):
            instance.llm = llm
        if (
            info.needs_embeddings
            and embeddings is not None
            and hasattr(instance, "embeddings")
        ):
            instance.embeddings = embeddings

        instances.append(instance)
    return instances


def run_evaluation(
    df: pd.DataFrame,
    metric_infos: list[MetricInfo],
    llm_cfg: LLMConfig,
    emb_cfg: EmbeddingConfig,
    run_settings: RunSettings,
    telemetry: TelemetryManager | None = None,
) -> dict[str, Any]:
    """Run a full Ragas evaluation and return a results dict.

    Returns
    -------
    dict with keys:
      ``result_df``  – DataFrame with per-row scores
      ``avg_scores`` – dict of metric_name → average score
      ``event``      – the ``EvaluationEvent`` (if telemetry is active)
    """
    import os

    import nest_asyncio
    from ragas import evaluate

    nest_asyncio.apply()

    # Set API key in env for ragas internals
    if llm_cfg.api_key:
        os.environ["OPENAI_API_KEY"] = llm_cfg.api_key
    if emb_cfg.api_key:
        os.environ.setdefault("OPENAI_API_KEY", emb_cfg.api_key)

    # Build objects
    llm = build_llm(llm_cfg) if llm_cfg.is_configured else None
    embeddings = build_embeddings(emb_cfg) if emb_cfg.is_configured else None
    run_config = build_run_config(run_settings)
    metrics = instantiate_metrics(metric_infos, llm=llm, embeddings=embeddings)

    ragas_ds = build_ragas_dataset(df)

    # Prepare telemetry event
    event = EvaluationEvent(
        run_id=uuid.uuid4().hex[:12],
        dataset_rows=len(df),
        metrics=[m.display_name for m in metric_infos],
        model=llm_cfg.model,
    )

    ctx = (
        telemetry.trace_evaluation(event)
        if telemetry is not None
        else _noop_context(event)
    )

    with ctx as ev:
        eval_kwargs: dict[str, Any] = {
            "dataset": ragas_ds,
            "metrics": metrics,
            "run_config": run_config,
            "raise_exceptions": run_settings.raise_exceptions,
            "show_progress": run_settings.show_progress,
        }
        if llm is not None:
            eval_kwargs["llm"] = llm
        if embeddings is not None:
            eval_kwargs["embeddings"] = embeddings
        if run_settings.batch_size is not None:
            eval_kwargs["batch_size"] = run_settings.batch_size
        if run_settings.column_map:
            eval_kwargs["column_map"] = run_settings.column_map
        if run_settings.experiment_name:
            eval_kwargs["experiment_name"] = run_settings.experiment_name

        result = evaluate(**eval_kwargs)
        result_df = result.to_pandas()

        # Compute averages
        score_cols = [
            c
            for c in result_df.columns
            if c
            not in {
                "question",
                "answer",
                "contexts",
                "ground_truth",
                "user_input",
                "response",
                "retrieved_contexts",
                "reference",
            }
        ]
        avg_scores = {
            col: float(result_df[col].mean())
            for col in score_cols
            if pd.api.types.is_numeric_dtype(result_df[col])
        }

    return {
        "result_df": result_df,
        "avg_scores": avg_scores,
        "event": ev,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def _noop_context(event: EvaluationEvent):
    """Fallback context manager when telemetry is disabled."""
    import time

    event.start_time = time.time()
    event.status = "running"
    try:
        yield event
        event.status = "completed"
    except Exception:
        event.status = "failed"
        raise
    finally:
        event.end_time = time.time()
