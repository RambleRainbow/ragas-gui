"""LLM and embeddings configuration for Ragas evaluation.

Provides dataclass-based configuration that the UI can serialise to/from
Streamlit session state, and factory helpers that create the objects Ragas
``evaluate()`` expects.

This module uses OpenAI-compatible mode for all providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Supported providers (OpenAI compatible mode only)
# ---------------------------------------------------------------------------


class Provider(str, Enum):
    """Known LLM providers with OpenAI-compatible API endpoints."""

    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    CUSTOM = "custom"


class EmbeddingProvider(str, Enum):
    """Embedding provider types."""

    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Default configuration per provider
# ---------------------------------------------------------------------------

# Default base URLs for each provider
DEFAULT_BASE_URLS: dict[Provider, str] = {
    Provider.OLLAMA: "http://localhost:11434/v1",
    Provider.VLLM: "http://localhost:8000/v1",
    Provider.OPENAI: "https://api.openai.com/v1",
    Provider.ANTHROPIC: "https://api.anthropic.com/v1",
    Provider.GOOGLE: "https://generativelanguage.googleapis.com/v1",
    Provider.AZURE: "https://{resource-name}.openai.azure.com",
    Provider.CUSTOM: "",
}

# Default model names per provider
DEFAULT_MODELS: dict[Provider, str] = {
    Provider.OLLAMA: "llama3.1",
    Provider.VLLM: "llama3.1",
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-3-5-sonnet-latest",
    Provider.GOOGLE: "gemini-2.0-flash",
    Provider.AZURE: "gpt-4o-mini",
    Provider.CUSTOM: "",
}

# Default embedding base URLs
DEFAULT_EMBEDDING_BASE_URLS: dict[EmbeddingProvider, str] = {
    EmbeddingProvider.OLLAMA: "http://localhost:11434/v1",
    EmbeddingProvider.VLLM: "http://localhost:8000/v1",
    EmbeddingProvider.OPENAI: "https://api.openai.com/v1",
    EmbeddingProvider.HUGGINGFACE: "",
    EmbeddingProvider.CUSTOM: "",
}

# Default embedding models per provider
DEFAULT_EMBEDDING_MODELS: dict[EmbeddingProvider, str] = {
    EmbeddingProvider.OLLAMA: "nomic-embed-text",
    EmbeddingProvider.VLLM: "nomic-embed-text",
    EmbeddingProvider.OPENAI: "text-embedding-3-small",
    EmbeddingProvider.HUGGINGFACE: "sentence-transformers/all-MiniLM-L6-v2",
    EmbeddingProvider.CUSTOM: "",
}


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """Configuration for the evaluator LLM.

    All providers use OpenAI-compatible mode with base_url and model_name.
    """

    provider: Provider = Provider.OPENAI
    base_url: str = ""
    model_name: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int | None = None
    system_prompt: str = ""

    def __post_init__(self) -> None:
        # Set defaults if not provided
        if not self.base_url:
            self.base_url = DEFAULT_BASE_URLS.get(self.provider, "")
        if not self.model_name:
            self.model_name = DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")

    @property
    def is_configured(self) -> bool:
        """Check if the provider is properly configured."""
        return bool(self.api_key) or self.provider in (Provider.OLLAMA, Provider.VLLM)


@dataclass
class EmbeddingConfig:
    """Configuration for the embeddings model.

    All providers use OpenAI-compatible mode with base_url and model_name.
    """

    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    base_url: str = ""
    model_name: str = ""
    api_key: str = ""

    def __post_init__(self) -> None:
        # Set defaults if not provided
        if not self.base_url:
            self.base_url = DEFAULT_EMBEDDING_BASE_URLS.get(self.provider, "")
        if not self.model_name:
            self.model_name = DEFAULT_EMBEDDING_MODELS.get(
                self.provider, "text-embedding-3-small"
            )

    @property
    def is_configured(self) -> bool:
        """Check if the provider is properly configured."""
        return bool(self.api_key) or self.provider in (
            EmbeddingProvider.OLLAMA,
            EmbeddingProvider.VLLM,
        )


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

    All providers use OpenAI-compatible mode via ChatOpenAI.
    """
    import os

    os.environ.setdefault("OPENAI_API_KEY", cfg.api_key or "no-key")
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": cfg.model_name,
        "temperature": cfg.temperature,
        "api_key": cfg.api_key or "no-key",
    }
    if cfg.max_tokens:
        kwargs["max_tokens"] = cfg.max_tokens
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url

    return ChatOpenAI(**kwargs)


def build_embeddings(cfg: EmbeddingConfig) -> Any:
    """Create a Ragas-compatible embeddings wrapper from *cfg*.

    OpenAI-compatible providers use OpenAIEmbeddings.
    HuggingFace uses its own embeddings.
    """
    import os

    if cfg.provider == EmbeddingProvider.HUGGINGFACE:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=cfg.model_name)

    # All other providers use OpenAI-compatible mode
    os.environ.setdefault("OPENAI_API_KEY", cfg.api_key or "no-key")
    from langchain_openai import OpenAIEmbeddings

    kwargs: dict[str, Any] = {"model": cfg.model_name}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url

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


# ---------------------------------------------------------------------------
# Connection testing
# ---------------------------------------------------------------------------


def test_llm_connection(cfg: LLMConfig) -> tuple[bool, str]:
    """Test if the LLM provider is reachable.

    Returns:
        Tuple of (success: bool, message: str)
    """
    import os

    try:
        os.environ.setdefault("OPENAI_API_KEY", cfg.api_key or "no-key")
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(
            model=cfg.model_name,
            api_key=cfg.api_key if cfg.api_key else None,
            base_url=cfg.base_url if cfg.base_url else None,
            temperature=0,
        )

        client.invoke("Hi")

        return True, "Connection successful"
    except Exception as e:
        return False, str(e)


async def test_embedding_connection(cfg: EmbeddingConfig) -> tuple[bool, str]:
    """Test if the embedding provider is reachable.

    Returns:
        Tuple of (success: bool, message: str)
    """
    import os

    try:
        if cfg.provider == EmbeddingProvider.HUGGINGFACE:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            emb = HuggingFaceEmbeddings(model_name=cfg.model_name)
            _ = emb.embed_query("test")
            return True, "Connection successful"

        os.environ.setdefault("OPENAI_API_KEY", cfg.api_key or "no-key")
        from langchain_openai import OpenAIEmbeddings

        emb = OpenAIEmbeddings(
            model=cfg.model_name,
            base_url=cfg.base_url if cfg.base_url else None,
        )
        _ = emb.embed_query("test")
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Legacy aliases for backwards compatibility
# ---------------------------------------------------------------------------

# Keep old names as aliases for backwards compatibility
LLMProvider = Provider
EmbeddingProvider = EmbeddingProvider
CompatibilityMode = None  # Deprecated - not used anymore
