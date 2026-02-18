"""LLM and embeddings configuration for Ragas evaluation.

Provides dataclass-based configuration that the UI can serialise to/from
Streamlit session state, and factory helpers that create the objects Ragas
``evaluate()`` expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Supported providers
# ---------------------------------------------------------------------------


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    LITELLM = "litellm"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    LITELLM = "litellm"


# ---------------------------------------------------------------------------
# Default model names per provider
# ---------------------------------------------------------------------------

DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-latest",
    LLMProvider.GOOGLE: "gemini-2.0-flash",
    LLMProvider.AZURE_OPENAI: "gpt-4o-mini",
    LLMProvider.LITELLM: "gpt-4o-mini",
}

DEFAULT_EMBEDDING_MODELS: dict[EmbeddingProvider, str] = {
    EmbeddingProvider.OPENAI: "text-embedding-3-small",
    EmbeddingProvider.GOOGLE: "text-embedding-004",
    EmbeddingProvider.HUGGINGFACE: "sentence-transformers/all-MiniLM-L6-v2",
    EmbeddingProvider.LITELLM: "text-embedding-3-small",
}


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """Configuration for the evaluator LLM."""

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = ""
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None
    system_prompt: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            self.model = DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class EmbeddingConfig:
    """Configuration for the embeddings model."""

    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    model: str = ""
    api_key: str = ""
    api_base: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            self.model = DEFAULT_EMBEDDING_MODELS.get(
                self.provider, "text-embedding-3-small"
            )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class RunSettings:
    """Runtime settings passed to ``ragas.evaluate()``."""

    timeout: int = 180
    max_retries: int = 10
    max_wait: int = 60
    max_workers: int = 16
    seed: int = 42
    batch_size: int | None = None
    raise_exceptions: bool = False
    show_progress: bool = True
    experiment_name: str = ""
    column_map: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_llm(cfg: LLMConfig) -> Any:
    """Create a Ragas-compatible LLM wrapper from *cfg*.

    Uses ``langchain_openai.ChatOpenAI`` for OpenAI / Azure, and falls back
    to LiteLLM-based wrappers for other providers.
    """
    import os

    if cfg.provider == LLMProvider.OPENAI:
        os.environ.setdefault("OPENAI_API_KEY", cfg.api_key)
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "temperature": cfg.temperature,
        }
        if cfg.max_tokens:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.api_base:
            kwargs["base_url"] = cfg.api_base
        return ChatOpenAI(**kwargs)

    if cfg.provider == LLMProvider.AZURE_OPENAI:
        os.environ.setdefault("AZURE_OPENAI_API_KEY", cfg.api_key)
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            azure_endpoint=cfg.api_base,
        )

    # Generic fallback: set the key and let ragas use defaults
    os.environ.setdefault("OPENAI_API_KEY", cfg.api_key)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=cfg.model, temperature=cfg.temperature)


def build_embeddings(cfg: EmbeddingConfig) -> Any:
    """Create a Ragas-compatible embeddings wrapper from *cfg*."""
    import os

    if cfg.provider == EmbeddingProvider.OPENAI:
        os.environ.setdefault("OPENAI_API_KEY", cfg.api_key)
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {"model": cfg.model}
        if cfg.api_base:
            kwargs["base_url"] = cfg.api_base
        return OpenAIEmbeddings(**kwargs)

    if cfg.provider == EmbeddingProvider.HUGGINGFACE:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=cfg.model)

    # Fallback to OpenAI
    os.environ.setdefault("OPENAI_API_KEY", cfg.api_key)
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=cfg.model)


def build_run_config(settings: RunSettings) -> Any:
    """Create a ``RunConfig`` from *settings*."""
    from ragas import RunConfig

    return RunConfig(
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        max_wait=settings.max_wait,
        max_workers=settings.max_workers,
        seed=settings.seed,
    )
