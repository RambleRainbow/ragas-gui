"""OpenTelemetry instrumentation with LLM-specific observability.

Provides:
  - ``TelemetryManager``  – singleton that initialises OTel tracing once.
  - ``trace_evaluation``  – context manager / decorator for evaluation runs.
  - Automatic OpenAI instrumentation when *traceloop-sdk* is installed.
  - Token / cost tracking helpers.

All telemetry is **opt-in**: nothing is emitted until ``TelemetryManager.init()``
is called explicitly from the UI or a config flag.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ExporterType(str, Enum):
    CONSOLE = "console"
    OTLP_HTTP = "otlp_http"
    OTLP_GRPC = "otlp_grpc"


@dataclass
class TelemetryConfig:
    """User-facing telemetry knobs exposed in the GUI."""

    enabled: bool = False
    service_name: str = "ragas-gui"
    exporter: ExporterType = ExporterType.CONSOLE
    otlp_endpoint: str = ""
    otlp_headers: dict[str, str] = field(default_factory=dict)
    trace_llm_content: bool = True  # privacy toggle


# ---------------------------------------------------------------------------
# Token / cost tracking
# ---------------------------------------------------------------------------

# Approximate per-1K-token cost (USD) for common models.
_MODEL_COST_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}


@dataclass
class TokenUsage:
    """Accumulated token counts for a single evaluation run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def estimated_cost_usd(self, model: str) -> float | None:
        """Return estimated cost in USD, or *None* if model unknown."""
        pricing = _MODEL_COST_PER_1K.get(model)
        if pricing is None:
            return None
        return (
            self.prompt_tokens / 1000 * pricing["input"]
            + self.completion_tokens / 1000 * pricing["output"]
        )


# ---------------------------------------------------------------------------
# Span / event recording helpers
# ---------------------------------------------------------------------------


@dataclass
class EvaluationEvent:
    """In-memory record for one evaluation run (displayed in the UI)."""

    run_id: str
    dataset_rows: int
    metrics: list[str]
    model: str
    start_time: float = 0.0
    end_time: float = 0.0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    status: str = "pending"
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0


# ---------------------------------------------------------------------------
# Telemetry manager (singleton-ish, lives in ``st.session_state``)
# ---------------------------------------------------------------------------


class TelemetryManager:
    """Initialises and manages OTel resources.

    Designed to be stored in ``st.session_state["telemetry"]`` so that the
    tracer provider is created exactly once per Streamlit session.
    """

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        self.config = config or TelemetryConfig()
        self._tracer: Any = None
        self._provider: Any = None
        self._initialised = False
        self.events: list[EvaluationEvent] = []

    # -- lifecycle -----------------------------------------------------------

    def init(self) -> None:
        """Initialise OTel SDK.  Safe to call multiple times (no-op)."""
        if self._initialised or not self.config.enabled:
            return
        try:
            self._setup_otel()
            self._initialised = True
            logger.info("OpenTelemetry initialised (exporter=%s)", self.config.exporter)
        except ImportError as exc:
            logger.warning("OTel packages not installed (%s). Telemetry disabled.", exc)
        except Exception:
            logger.exception("Failed to initialise OpenTelemetry")

    def shutdown(self) -> None:
        if self._provider is not None:
            self._provider.shutdown()

    # -- internal setup ------------------------------------------------------

    def _setup_otel(self) -> None:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": self.config.service_name})
        self._provider = TracerProvider(resource=resource)

        exporter = self._build_exporter()
        self._provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(self._provider)
        self._tracer = trace.get_tracer(self.config.service_name)

        # Auto-instrument OpenAI if possible
        if not self.config.trace_llm_content:
            os.environ["TRACELOOP_TRACE_CONTENT"] = "false"
        self._try_instrument_openai()

    def _build_exporter(self) -> Any:
        if self.config.exporter == ExporterType.CONSOLE:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            return ConsoleSpanExporter()

        if self.config.exporter == ExporterType.OTLP_HTTP:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            return OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint or "http://localhost:4318/v1/traces",
                headers=self.config.otlp_headers or None,
            )

        # OTLP_GRPC
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(
            endpoint=self.config.otlp_endpoint or "localhost:4317",
            insecure=True,
        )

    @staticmethod
    def _try_instrument_openai() -> None:
        try:
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor

            OpenAIInstrumentor().instrument()
        except ImportError:
            logger.debug(
                "opentelemetry-instrumentation-openai not installed; skipping."
            )

    # -- public helpers used by the evaluation runner ------------------------

    @contextmanager
    def trace_evaluation(
        self, event: EvaluationEvent
    ) -> Generator[EvaluationEvent, None, None]:
        """Context manager that wraps an evaluation run in an OTel span
        and appends a finished ``EvaluationEvent`` to ``self.events``.
        """
        event.start_time = time.time()
        event.status = "running"
        self.events.append(event)

        span = None
        if self._tracer is not None:
            span = self._tracer.start_span(
                "ragas.evaluate",
                attributes={
                    "ragas.run_id": event.run_id,
                    "ragas.dataset_rows": event.dataset_rows,
                    "ragas.metrics": ",".join(event.metrics),
                    "ragas.model": event.model,
                },
            )
        try:
            yield event
            event.status = "completed"
            if span is not None:
                from opentelemetry.trace import StatusCode

                span.set_status(StatusCode.OK)
                span.set_attribute("ragas.tokens.total", event.token_usage.total_tokens)
                cost = event.token_usage.estimated_cost_usd(event.model)
                if cost is not None:
                    span.set_attribute("ragas.cost_usd", cost)
        except Exception as exc:
            event.status = "failed"
            event.error = str(exc)
            if span is not None:
                from opentelemetry.trace import StatusCode

                span.set_status(StatusCode.ERROR, str(exc))
                span.record_exception(exc)
            raise
        finally:
            event.end_time = time.time()
            if span is not None:
                span.set_attribute("ragas.duration_s", event.duration_seconds)
                span.end()
