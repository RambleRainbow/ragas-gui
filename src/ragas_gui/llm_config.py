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


class CompatibilityMode(str, Enum):
    """Controls which LLM client is used when connecting to custom endpoints.

    ``NONE``
        Use the native client for the selected provider (default).
    ``OPENAI_COMPATIBLE``
        Force ``ChatOpenAI`` with a custom ``base_url`` – works with Ollama,
        vLLM, LocalAI, and any server exposing an OpenAI-compatible API.
    ``ANTHROPIC_COMPATIBLE``
        Force ``ChatAnthropic`` with a custom API URL – works with proxies
        that expose an Anthropic-compatible API.
    """

    NONE = "none"
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC_COMPATIBLE = "anthropic_compatible"


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
    compatibility_mode: CompatibilityMode = CompatibilityMode.NONE
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
    """Create a Ragas-compatible LLM wrapper from *cfg*."""
    import os

    # --- Compatibility-mode overrides take precedence -----------------------
    if cfg.compatibility_mode == CompatibilityMode.OPENAI_COMPATIBLE:
        os.environ.setdefault("OPENAI_API_KEY", cfg.api_key or "no-key")
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "api_key": cfg.api_key or "no-key",
        }
        if cfg.max_tokens:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.api_base:
            kwargs["base_url"] = cfg.api_base
        return ChatOpenAI(**kwargs)

    if cfg.compatibility_mode == CompatibilityMode.ANTHROPIC_COMPATIBLE:
        os.environ.setdefault("ANTHROPIC_API_KEY", cfg.api_key)
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": cfg.model,
            "temperature": cfg.temperature,
        }
        if cfg.max_tokens:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.api_base:
            kwargs["anthropic_api_url"] = cfg.api_base
        return ChatAnthropic(**kwargs)

    # --- Native provider paths ----------------------------------------------
    if cfg.provider == LLMProvider.OPENAI:
        os.environ.setdefault("OPENAI_API_KEY", cfg.api_key)
        from langchain_openai import ChatOpenAI

        kwargs = {
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

    if cfg.provider == LLMProvider.ANTHROPIC:
        os.environ.setdefault("ANTHROPIC_API_KEY", cfg.api_key)
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": cfg.model,
            "temperature": cfg.temperature,
        }
        if cfg.max_tokens:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.api_base:
            kwargs["anthropic_api_url"] = cfg.api_base
        return ChatAnthropic(**kwargs)

    if cfg.provider == LLMProvider.GOOGLE:
        os.environ.setdefault("GOOGLE_API_KEY", cfg.api_key)
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {
            "model": cfg.model,
            "temperature": cfg.temperature,
        }
        if cfg.max_tokens:
            kwargs["max_output_tokens"] = cfg.max_tokens
        return ChatGoogleGenerativeAI(**kwargs)

    # LiteLLM and unknown providers: fall back to ChatOpenAI
    os.environ.setdefault("OPENAI_API_KEY", cfg.api_key)
    from langchain_openai import ChatOpenAI

    kwargs = {"model": cfg.model, "temperature": cfg.temperature}
    if cfg.api_base:
        kwargs["base_url"] = cfg.api_base
    return ChatOpenAI(**kwargs)


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

    if cfg.provider == EmbeddingProvider.GOOGLE:
        os.environ.setdefault("GOOGLE_API_KEY", cfg.api_key)
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=cfg.model)

    if cfg.provider == EmbeddingProvider.HUGGINGFACE:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=cfg.model)

    # LiteLLM / fallback → OpenAI-compatible
    os.environ.setdefault("OPENAI_API_KEY", cfg.api_key)
    from langchain_openai import OpenAIEmbeddings

    kwargs = {"model": cfg.model}
    if cfg.api_base:
        kwargs["base_url"] = cfg.api_base
    return OpenAIEmbeddings(**kwargs)


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
