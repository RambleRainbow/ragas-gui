"""Internationalisation (i18n) module for the Ragas GUI.

Provides a lightweight translation layer backed by a flat dictionary.
Supported locales: English (``en``) and Chinese (``zh``).

Usage::

    from ragas_gui.i18n import t

    st.header(t("sidebar_header"))
    st.error(t("file_read_error", error=str(exc)))
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: dict[str, str] = {
    "English": "en",
    "中文": "zh",
}

DEFAULT_LANGUAGE = "en"

# ---------------------------------------------------------------------------
# Translation catalogue
# ---------------------------------------------------------------------------

TRANSLATIONS: dict[str, dict[str, str]] = {
    # ===================================================================
    # English
    # ===================================================================
    "en": {
        # Page
        "page_title": "Ragas Evaluator",
        "app_title": "\U0001f4ca Ragas RAG Evaluation",
        # Mode toggle
        "mode_label": "Mode",
        "mode_quick_start": "\U0001f680 Quick Start",
        "mode_advanced": "\u2699\ufe0f Advanced",
        "mode_help": (
            "Quick Start: sensible defaults, fewer options. Advanced: full control."
        ),
        # Language
        "lang_label": "\U0001f310 Language",
        # Sidebar header (renamed from "API Configuration")
        "sidebar_header": "\U0001f517 Model Settings",
        # Provider & model
        "llm_provider": "LLM Provider",
        "provider": "Provider",
        "llm_model": "Model",
        "temperature": "Temperature",
        # Connection
        "api_key": "API Key",
        "api_key_help": "Required by Ragas for LLM-based metrics.",
        "api_base_url": "API Base URL",
        "api_base_help": (
            "Custom API endpoint. Examples: "
            "http://localhost:11434/v1 (Ollama), "
            "http://localhost:8000/v1 (vLLM), "
            "https://your-azure.openai.azure.com/"
        ),
        # Compatibility (advanced connection)
        "advanced_connection": "Advanced Connection",
        "compat_mode": "Compatibility Mode",
        "compat_none": "None (native provider)",
        "compat_openai": "OpenAI Compatible",
        "compat_anthropic": "Anthropic Compatible",
        "compat_help": (
            "Use 'OpenAI Compatible' for Ollama, vLLM, LocalAI, or any "
            "server exposing an OpenAI-compatible API. "
            "Use 'Anthropic Compatible' for Anthropic-compatible proxies."
        ),
        # Embeddings
        "embeddings": "Embeddings",
        "emb_provider": "Embedding Provider",
        "emb_model": "Embedding Model",
        "emb_api_key": "Embedding API Key (leave blank to reuse above)",
        "emb_api_base": "Embedding API Base URL",
        "emb_api_base_help": (
            "Custom endpoint for embeddings (e.g. http://localhost:11434/v1)."
        ),
        # Metrics
        "metrics": "Metrics",
        # Runtime settings
        "runtime_settings": "Runtime Settings",
        "timeout": "Timeout (s)",
        "max_retries": "Max Retries",
        "max_workers": "Max Workers",
        "seed": "Seed",
        "batch_size": "Batch Size (0 = auto)",
        "raise_exceptions": "Raise Exceptions",
        # Observability
        "observability": "Observability (OpenTelemetry)",
        "enable_tracing": "Enable Tracing",
        "exporter": "Exporter",
        "otlp_endpoint": "OTLP Endpoint",
        "otlp_help": "Only for OTLP exporters.",
        "log_content": "Log Prompt/Completion Content",
        # Footer
        "footer": "Built with Streamlit + Ragas",
        # Main area
        "upload_label": "Upload evaluation dataset (CSV or JSON)",
        "upload_help": (
            "Must contain columns: question, answer, contexts, ground_truth"
        ),
        "file_read_error": "Failed to read file: {error}",
        "data_preview": "\U0001f4c4 Data Preview",
        "missing_columns": (
            "Missing required columns: **{columns}**. "
            "Expected: `question`, `answer`, `contexts`, `ground_truth`."
        ),
        "dataset_loaded": (
            "Dataset loaded \u2014 {rows} rows, all required columns present."
        ),
        "enter_api_key": ("Enter your API key in the sidebar to enable evaluation."),
        "select_metric": "Select at least one metric in the sidebar.",
        "run_evaluation": "\U0001f680 Run Evaluation",
        "running_spinner": (
            "Running Ragas evaluation\u2026 this may take a few minutes."
        ),
        "eval_failed": "Evaluation failed: {error}",
        "eval_results": "\U0001f4c8 Evaluation Results",
        "chart_metric": "Metric",
        "chart_avg_score": "Average Score",
        "chart_title": "Average Metric Scores",
        "download_csv": "\u2b07\ufe0f Download Results CSV",
        "telemetry_summary": "Telemetry Summary",
        "upload_prompt": "Upload a CSV or JSON file to get started.",
    },
    # ===================================================================
    # Chinese
    # ===================================================================
    "zh": {
        # 页面
        "page_title": "Ragas \u8bc4\u4f30\u5668",
        "app_title": "\U0001f4ca Ragas RAG \u8bc4\u4f30",
        # 模式切换
        "mode_label": "\u6a21\u5f0f",
        "mode_quick_start": "\U0001f680 \u5feb\u901f\u5f00\u59cb",
        "mode_advanced": "\u2699\ufe0f \u9ad8\u7ea7\u6a21\u5f0f",
        "mode_help": (
            "\u5feb\u901f\u5f00\u59cb\uff1a\u5408\u7406\u9ed8\u8ba4\u503c\uff0c"
            "\u66f4\u5c11\u9009\u9879\u3002"
            "\u9ad8\u7ea7\u6a21\u5f0f\uff1a\u5b8c\u5168\u63a7\u5236\u3002"
        ),
        # 语言
        "lang_label": "\U0001f310 \u8bed\u8a00",
        # 侧边栏标题
        "sidebar_header": "\U0001f517 \u6a21\u578b\u8bbe\u7f6e",
        # 提供商和模型
        "llm_provider": "LLM \u63d0\u4f9b\u5546",
        "provider": "\u63d0\u4f9b\u5546",
        "llm_model": "\u6a21\u578b",
        "temperature": "\u6e29\u5ea6",
        # 连接
        "api_key": "API \u5bc6\u94a5",
        "api_key_help": "Ragas \u7684 LLM \u6307\u6807\u9700\u8981\u6b64\u5bc6\u94a5\u3002",
        "api_base_url": "API \u57fa\u7840 URL",
        "api_base_help": (
            "\u81ea\u5b9a\u4e49 API \u7aef\u70b9\u3002\u793a\u4f8b\uff1a"
            "http://localhost:11434/v1 (Ollama)\uff0c"
            "http://localhost:8000/v1 (vLLM)\uff0c"
            "https://your-azure.openai.azure.com/"
        ),
        # 兼容模式（高级连接）
        "advanced_connection": "\u9ad8\u7ea7\u8fde\u63a5",
        "compat_mode": "\u517c\u5bb9\u6a21\u5f0f",
        "compat_none": "\u65e0\uff08\u539f\u751f\u63d0\u4f9b\u5546\uff09",
        "compat_openai": "OpenAI \u517c\u5bb9",
        "compat_anthropic": "Anthropic \u517c\u5bb9",
        "compat_help": (
            "\u4f7f\u7528\u201cOpenAI \u517c\u5bb9\u201d\u8fde\u63a5 Ollama\u3001"
            "vLLM\u3001LocalAI \u6216\u4efb\u4f55\u63d0\u4f9b OpenAI \u517c\u5bb9 "
            "API \u7684\u670d\u52a1\u5668\u3002"
            "\u4f7f\u7528\u201cAnthropic \u517c\u5bb9\u201d\u8fde\u63a5 "
            "Anthropic \u517c\u5bb9\u4ee3\u7406\u3002"
        ),
        # 嵌入模型
        "embeddings": "\u5d4c\u5165\u6a21\u578b",
        "emb_provider": "\u5d4c\u5165\u63d0\u4f9b\u5546",
        "emb_model": "\u5d4c\u5165\u6a21\u578b\u540d\u79f0",
        "emb_api_key": (
            "\u5d4c\u5165 API \u5bc6\u94a5"
            "\uff08\u7559\u7a7a\u5219\u590d\u7528\u4e0a\u65b9\u5bc6\u94a5\uff09"
        ),
        "emb_api_base": "\u5d4c\u5165 API \u57fa\u7840 URL",
        "emb_api_base_help": (
            "\u5d4c\u5165\u7684\u81ea\u5b9a\u4e49\u7aef\u70b9"
            "\uff08\u4f8b\u5982 http://localhost:11434/v1\uff09\u3002"
        ),
        # 指标
        "metrics": "\u8bc4\u4f30\u6307\u6807",
        # 运行时设置
        "runtime_settings": "\u8fd0\u884c\u65f6\u8bbe\u7f6e",
        "timeout": "\u8d85\u65f6\u65f6\u95f4\uff08\u79d2\uff09",
        "max_retries": "\u6700\u5927\u91cd\u8bd5\u6b21\u6570",
        "max_workers": "\u6700\u5927\u5e76\u53d1\u6570",
        "seed": "\u968f\u673a\u79cd\u5b50",
        "batch_size": "\u6279\u91cf\u5927\u5c0f\uff080 = \u81ea\u52a8\uff09",
        "raise_exceptions": "\u629b\u51fa\u5f02\u5e38",
        # 可观测性
        "observability": "\u53ef\u89c2\u6d4b\u6027 (OpenTelemetry)",
        "enable_tracing": "\u542f\u7528\u8ffd\u8e2a",
        "exporter": "\u5bfc\u51fa\u5668",
        "otlp_endpoint": "OTLP \u7aef\u70b9",
        "otlp_help": "\u4ec5\u7528\u4e8e OTLP \u5bfc\u51fa\u5668\u3002",
        "log_content": "\u8bb0\u5f55\u63d0\u793a/\u8865\u5168\u5185\u5bb9",
        # 页脚
        "footer": "\u57fa\u4e8e Streamlit + Ragas \u6784\u5efa",
        # 主区域
        "upload_label": (
            "\u4e0a\u4f20\u8bc4\u4f30\u6570\u636e\u96c6\uff08CSV \u6216 JSON\uff09"
        ),
        "upload_help": (
            "\u5fc5\u987b\u5305\u542b\u5217\uff1a"
            "question\u3001answer\u3001contexts\u3001ground_truth"
        ),
        "file_read_error": "\u6587\u4ef6\u8bfb\u53d6\u5931\u8d25\uff1a{error}",
        "data_preview": "\U0001f4c4 \u6570\u636e\u9884\u89c8",
        "missing_columns": (
            "\u7f3a\u5c11\u5fc5\u9700\u5217\uff1a**{columns}**\u3002"
            "\u9700\u8981\uff1a`question`\u3001`answer`\u3001"
            "`contexts`\u3001`ground_truth`\u3002"
        ),
        "dataset_loaded": (
            "\u6570\u636e\u96c6\u5df2\u52a0\u8f7d \u2014 "
            "{rows} \u884c\uff0c\u6240\u6709\u5fc5\u9700\u5217\u5747\u5b58\u5728\u3002"
        ),
        "enter_api_key": (
            "\u8bf7\u5728\u4fa7\u8fb9\u680f\u8f93\u5165 API "
            "\u5bc6\u94a5\u4ee5\u542f\u7528\u8bc4\u4f30\u3002"
        ),
        "select_metric": (
            "\u8bf7\u5728\u4fa7\u8fb9\u680f\u9009\u62e9"
            "\u81f3\u5c11\u4e00\u4e2a\u6307\u6807\u3002"
        ),
        "run_evaluation": "\U0001f680 \u8fd0\u884c\u8bc4\u4f30",
        "running_spinner": (
            "\u6b63\u5728\u8fd0\u884c Ragas \u8bc4\u4f30\u2026"
            "\u8fd9\u53ef\u80fd\u9700\u8981\u51e0\u5206\u949f\u3002"
        ),
        "eval_failed": "\u8bc4\u4f30\u5931\u8d25\uff1a{error}",
        "eval_results": "\U0001f4c8 \u8bc4\u4f30\u7ed3\u679c",
        "chart_metric": "\u6307\u6807",
        "chart_avg_score": "\u5e73\u5747\u5206",
        "chart_title": "\u5e73\u5747\u6307\u6807\u5206\u6570",
        "download_csv": "\u2b07\ufe0f \u4e0b\u8f7d\u7ed3\u679c CSV",
        "telemetry_summary": "\u9065\u6d4b\u6458\u8981",
        "upload_prompt": (
            "\u4e0a\u4f20 CSV \u6216 JSON \u6587\u4ef6\u4ee5\u5f00\u59cb\u3002"
        ),
    },
}

# ---------------------------------------------------------------------------
# Quick-start provider list (subset exposed in Quick Start mode)
# ---------------------------------------------------------------------------

QUICK_START_PROVIDERS: list[str] = ["openai", "anthropic", "google"]
"""Provider *enum values* shown in Quick Start mode (order preserved)."""

# Display labels for Quick Start providers (not translated — brand names).
QUICK_START_PROVIDER_LABELS: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_language() -> str:
    """Return the active language code from Streamlit session state."""
    try:
        import streamlit as st

        return st.session_state.get("language", DEFAULT_LANGUAGE)
    except Exception:
        return DEFAULT_LANGUAGE


def set_language(lang_code: str) -> None:
    """Persist the chosen language code in Streamlit session state."""
    import streamlit as st

    st.session_state["language"] = lang_code


def t(key: str, **kwargs: Any) -> str:
    """Translate *key* to the current language.

    Supports ``str.format`` interpolation::

        t("file_read_error", error="bad csv")
        # -> "Failed to read file: bad csv"
    """
    lang = get_language()
    text = TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)
    if kwargs:
        text = text.format(**kwargs)
    return text
