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

# Ragas wrapper imports for langchain compatibility
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory


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
    import concurrent.futures

    def _run_in_thread():
        return _evaluate_impl(
            df, metric_infos, llm_cfg, emb_cfg, run_settings, telemetry
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_thread)
        return future.result()


def _evaluate_impl(
    df: pd.DataFrame,
    metric_infos: list[MetricInfo],
    llm_cfg: LLMConfig,
    emb_cfg: EmbeddingConfig,
    run_settings: RunSettings,
    telemetry: TelemetryManager | None = None,
) -> dict[str, Any]:
    """Internal implementation of evaluation, runs in a separate thread."""
    import os

    from ragas import evaluate

    # Set API key in env for ragas internals
    if llm_cfg.api_key:
        os.environ["OPENAI_API_KEY"] = llm_cfg.api_key
    if emb_cfg.api_key:
        os.environ.setdefault("OPENAI_API_KEY", emb_cfg.api_key)

    # Build objects
    llm = build_llm(llm_cfg) if llm_cfg.is_configured else None
    embeddings = build_embeddings(emb_cfg) if emb_cfg.is_configured else None

    # Use llm_factory for ragas metrics compatibility (instead of LangchainLLMWrapper)
    # This fixes "Collections metrics only support modern InstructorLLM" error
    from ragas.llms import llm_factory

    # Determine provider from base_url
    provider = "openai"  # default
    if llm_cfg.base_url:
        if "anthropic" in llm_cfg.base_url:
            provider = "anthropic"
        elif "google" in llm_cfg.base_url or "generativelanguage" in llm_cfg.base_url:
            provider = "google"
        elif "azure" in llm_cfg.base_url:
            provider = "azure"

    ragas_llm = None
    if llm is not None:
        from openai import OpenAI

        client = OpenAI(
            api_key=llm_cfg.api_key or "no-key",
            base_url=llm_cfg.base_url or None,
            timeout=60.0,
            max_retries=3,
        )
        ragas_llm = llm_factory(
            model=llm_cfg.model_name,
            provider=provider,
            client=client,
        )

    # Wrap embeddings with ragas wrapper
    wrapped_embeddings = (
        LangchainEmbeddingsWrapper(embeddings) if embeddings is not None else None
    )

    run_config = build_run_config(run_settings)
    metrics = instantiate_metrics(
        metric_infos, llm=ragas_llm, embeddings=wrapped_embeddings
    )

    ragas_ds = build_ragas_dataset(df)

    # Prepare telemetry event
    event = EvaluationEvent(
        run_id=uuid.uuid4().hex[:12],
        dataset_rows=len(df),
        metrics=[m.display_name for m in metric_infos],
        model=llm_cfg.model_name,
    )

    ctx = (
        telemetry.trace_evaluation(event)
        if telemetry is not None
        else _noop_context(event)
    )

    with ctx as ev:
        from ragas.cost import get_token_usage_for_openai

        eval_kwargs: dict[str, Any] = {
            "dataset": ragas_ds,
            "metrics": metrics,
            "run_config": run_config,
            "raise_exceptions": run_settings.raise_exceptions,
            "show_progress": run_settings.show_progress,
            "token_usage_parser": get_token_usage_for_openai,
        }

        if ragas_llm is not None:
            eval_kwargs["llm"] = ragas_llm
        if wrapped_embeddings is not None:
            eval_kwargs["embeddings"] = wrapped_embeddings
        if run_settings.batch_size is not None:
            eval_kwargs["batch_size"] = run_settings.batch_size
        if run_settings.column_map:
            eval_kwargs["column_map"] = run_settings.column_map
        if run_settings.experiment_name:
            eval_kwargs["experiment_name"] = run_settings.experiment_name

        result = evaluate(**eval_kwargs)
        result_df = result.to_pandas()

        try:
            total_tokens = result.total_tokens()
            if total_tokens and hasattr(ev, "token_usage"):
                ev.token_usage.prompt_tokens = total_tokens.input_tokens or 0
                ev.token_usage.completion_tokens = total_tokens.output_tokens or 0
        except Exception:
            pass

        if result_df.empty:
            raise ValueError("Evaluation returned empty results")

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

        all_nan = all(result_df[col].isna().all() for col in score_cols)
        if all_nan:
            raise ValueError(
                "All metric scores are NaN. This usually means the LLM API call failed. "
                "Check your API key and base URL, or enable 'Raise Exceptions' in runtime settings for details."
            )

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
