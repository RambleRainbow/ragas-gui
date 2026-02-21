"""Ragas RAG Evaluation -- Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)

import pandas as pd
import plotly.express as px
import streamlit as st

from ragas_gui.config import (
    MetricCategory,
    get_metric_info,
    list_metrics_by_category,
)
from ragas_gui.data import load_uploaded_file, validate_columns
from ragas_gui.evaluation import run_evaluation
from ragas_gui.i18n import (
    SUPPORTED_LANGUAGES,
    get_language,
    set_language,
    t,
)
from ragas_gui.llm_config import (
    DEFAULT_EMBEDDING_BASE_URLS,
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_MODELS,
    EmbeddingConfig,
    EmbeddingProvider,
    LLMConfig,
    Provider,
    RunSettings,
    test_llm_connection,
)
from ragas_gui.telemetry import (
    ExporterType,
    TelemetryConfig,
    TelemetryManager,
)

PROFILES_DIR = Path.home() / ".ragas-gui"
PROFILES_FILE = PROFILES_DIR / "profiles.json"


def load_profiles() -> dict:
    if PROFILES_FILE.exists():
        try:
            return json.loads(PROFILES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_profiles(profiles: dict) -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_FILE.write_text(json.dumps(profiles, indent=2))


st.set_page_config(page_title=t("page_title"), page_icon="📊", layout="wide")

if "telemetry" not in st.session_state:
    st.session_state["telemetry"] = TelemetryManager()

telemetry: TelemetryManager = st.session_state["telemetry"]

profiles = load_profiles()


# ---------------------------------------------------------------------------
# Language toggle (top of sidebar, above everything else)
# ---------------------------------------------------------------------------

with st.sidebar:
    lang_options = list(SUPPORTED_LANGUAGES.keys())
    current_lang = get_language()
    _code_to_label = {v: k for k, v in SUPPORTED_LANGUAGES.items()}
    lang_idx = lang_options.index(_code_to_label.get(current_lang, "English"))

    chosen_label = st.selectbox(
        t("lang_label"),
        lang_options,
        index=lang_idx,
        key="lang_selector",
    )
    new_lang = SUPPORTED_LANGUAGES[chosen_label]
    if new_lang != current_lang:
        set_language(new_lang)
        st.rerun()


# ---------------------------------------------------------------------------
# Main title
# ---------------------------------------------------------------------------

st.title(t("app_title"))

# ---------------------------------------------------------------------------
# Sidebar -- LLM & Embeddings config
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header(t("sidebar_header"))

    profile_names = list(profiles.keys())
    selected_profile = st.selectbox(
        t("load_profile"),
        [""] + profile_names,
        index=0,
    )

    if selected_profile and selected_profile in profiles:
        loaded = profiles[selected_profile]
        saved_provider = loaded.get("provider", "openai")
        saved_model = loaded.get("model", "")
        saved_base_url = loaded.get("base_url", "")
        saved_temperature = loaded.get("temperature", 0.0)
        saved_api_key = loaded.get("api_key", "")
        saved_emb_provider = loaded.get("emb_provider", "openai")
        saved_emb_base_url = loaded.get("emb_base_url", "")
        saved_emb_model = loaded.get("emb_model", "")
        saved_emb_api_key = loaded.get("emb_api_key", "")
    else:
        saved_provider = "openai"
        saved_model = ""
        saved_base_url = ""
        saved_temperature = 0.0
        saved_api_key = ""
        saved_emb_provider = "openai"
        saved_emb_base_url = ""
        saved_emb_model = ""
        saved_emb_api_key = ""

    provider_options = [p.value for p in Provider]
    default_idx = (
        provider_options.index(saved_provider)
        if saved_provider in provider_options
        else 0
    )
    llm_provider = st.selectbox(
        t("llm_provider"),
        provider_options,
        index=default_idx,
    )
    llm_provider_enum = Provider(llm_provider)
    llm_base_url = st.text_input(
        t("base_url"),
        value=saved_base_url or DEFAULT_MODELS.get(llm_provider_enum, ""),
        help=t("base_url_help"),
    )
    llm_model = st.text_input(
        t("llm_model"),
        value=saved_model or DEFAULT_MODELS.get(llm_provider_enum, ""),
    )
    llm_temperature = st.slider(
        t("temperature"),
        0.0,
        2.0,
        saved_temperature,
        0.05,
    )

    api_key = st.text_input(
        t("api_key"),
        type="password",
        value=saved_api_key,
        help=t("api_key_help"),
    )

    st.divider()
    st.subheader(t("embeddings"))
    emb_provider_options = [p.value for p in EmbeddingProvider]
    emb_default_idx = (
        emb_provider_options.index(saved_emb_provider)
        if saved_emb_provider in emb_provider_options
        else 0
    )
    emb_provider = st.selectbox(
        t("emb_provider"),
        emb_provider_options,
        index=emb_default_idx,
    )
    emb_provider_enum = EmbeddingProvider(emb_provider)
    emb_base_url = st.text_input(
        t("emb_base_url"),
        value=saved_emb_base_url
        or DEFAULT_EMBEDDING_BASE_URLS.get(emb_provider_enum, ""),
        help=t("emb_base_url_help"),
    )
    emb_model = st.text_input(
        t("emb_model"),
        value=saved_emb_model or DEFAULT_EMBEDDING_MODELS.get(emb_provider_enum, ""),
    )
    emb_api_key = st.text_input(
        t("emb_api_key"),
        type="password",
        value=saved_emb_api_key,
    )
    emb_cfg = EmbeddingConfig(
        provider=emb_provider_enum,
        base_url=emb_base_url,
        model_name=emb_model,
        api_key=emb_api_key or api_key,
    )

    st.divider()
    save_name = st.text_input(t("save_name"), value="")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 " + t("save_settings"), key="save_settings_btn"):
            if not save_name.strip():
                st.error(t("save_name_required"))
            else:
                profiles[save_name.strip()] = {
                    "provider": llm_provider,
                    "model": llm_model,
                    "base_url": llm_base_url,
                    "temperature": llm_temperature,
                    "api_key": api_key,
                    "emb_provider": emb_provider,
                    "emb_base_url": emb_base_url,
                    "emb_model": emb_model,
                    "emb_api_key": emb_api_key,
                }
                save_profiles(profiles)
                st.success(t("settings_saved", name=save_name))
                st.rerun()

    with col2:
        if st.button("🗑️ " + t("delete_profile"), key="delete_profile_btn"):
            if selected_profile and selected_profile in profiles:
                del profiles[selected_profile]
                save_profiles(profiles)
                st.success(t("profile_deleted", name=selected_profile))
                st.rerun()

    # ---- Test connection button --------------------------------------------
    if st.button(t("test_connection"), key="test_llm_conn"):
        llm_test_cfg = LLMConfig(
            provider=llm_provider_enum,
            base_url=llm_base_url,
            model_name=llm_model,
            api_key=api_key,
            temperature=llm_temperature,
        )
        with st.spinner(t("testing_connection")):
            success, message = test_llm_connection(llm_test_cfg)
        if success:
            st.success(t("connection_success"))
        else:
            st.error(t("connection_failed", error=message))

    # Create LLM config
    llm_cfg = LLMConfig(
        provider=llm_provider_enum,
        base_url=llm_base_url,
        model_name=llm_model,
        api_key=api_key,
        temperature=llm_temperature,
    )

    # ---- Metrics selection ------------------------------------------------
    st.divider()
    st.subheader(t("metrics"))

    selected_metric_names: list[str] = []
    metrics_by_cat = list_metrics_by_category()
    for cat in MetricCategory:
        infos = metrics_by_cat.get(cat, [])
        if not infos:
            continue
        with st.expander(
            cat.value,
            expanded=(cat in {MetricCategory.RAG_CORE, MetricCategory.RAG_CONTEXT}),
        ):
            for info in infos:
                if st.checkbox(
                    info.display_name,
                    value=False,
                    help=info.description,
                    key=f"metric_{info.name}",
                ):
                    selected_metric_names.append(info.display_name)

    run_settings = RunSettings()

    st.divider()
    with st.expander(t("runtime_settings")):
        run_settings.timeout = st.number_input(t("timeout"), value=180, min_value=1)
        run_settings.max_retries = st.number_input(
            t("max_retries"), value=10, min_value=0
        )
        run_settings.max_workers = st.number_input(
            t("max_workers"), value=16, min_value=1
        )
        run_settings.seed = st.number_input(t("seed"), value=42)
        bs = st.number_input(t("batch_size"), value=0, min_value=0)
        run_settings.batch_size = bs if bs > 0 else None
        run_settings.raise_exceptions = st.checkbox(t("raise_exceptions"), value=False)

    with st.expander(t("observability")):
        otel_enabled = st.checkbox(t("enable_tracing"), value=False)
        if otel_enabled:
            exporter_val = st.selectbox(
                t("exporter"),
                [e.value for e in ExporterType],
                index=0,
            )
            otlp_endpoint = st.text_input(
                t("otlp_endpoint"),
                value="http://localhost:4318",
                help=t("otlp_help"),
            )
            trace_content = st.checkbox(t("log_content"), value=True)
            telemetry.config = TelemetryConfig(
                enabled=True,
                exporter=ExporterType(exporter_val),
                otlp_endpoint=otlp_endpoint,
                trace_llm_content=trace_content,
            )
            telemetry.init()
        else:
            telemetry.config.enabled = False

    st.divider()
    st.caption(t("footer"))

# ---------------------------------------------------------------------------
# Main area -- upload, preview, evaluate
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    t("upload_label"),
    type=["csv", "json", "jsonl"],
    help=t("upload_help"),
)

if uploaded_file is not None:
    try:
        df = load_uploaded_file(uploaded_file)
    except Exception as exc:
        st.error(t("file_read_error", error=str(exc)))
        st.stop()

    st.subheader(t("data_preview"))
    st.dataframe(df.head(10), width="stretch")

    missing = validate_columns(df)
    if missing:
        st.error(t("missing_columns", columns=", ".join(missing)))
        st.stop()

    st.success(t("dataset_loaded", rows=len(df)))

    if not api_key:
        st.info(t("enter_api_key"))
    if not selected_metric_names:
        st.info(t("select_metric"))

    run_disabled = not api_key or not selected_metric_names

    if st.button(t("run_evaluation"), type="primary", disabled=run_disabled):
        metric_infos = [get_metric_info(n) for n in selected_metric_names]

        with st.spinner(t("running_spinner")):
            try:
                results = run_evaluation(
                    df=df,
                    metric_infos=metric_infos,
                    llm_cfg=llm_cfg,
                    emb_cfg=emb_cfg,
                    run_settings=run_settings,
                    telemetry=telemetry if telemetry.config.enabled else None,
                )
            except Exception as exc:
                st.error(t("eval_failed", error=str(exc)))
                st.stop()

        st.subheader(t("eval_results"))
        result_df = results["result_df"]
        st.dataframe(result_df, width="stretch")

        avg_scores = results["avg_scores"]
        if avg_scores:
            chart_df = pd.DataFrame(
                {
                    t("chart_metric"): list(avg_scores.keys()),
                    t("chart_avg_score"): list(avg_scores.values()),
                }
            )
            fig = px.bar(
                chart_df,
                x=t("chart_metric"),
                y=t("chart_avg_score"),
                color=t("chart_metric"),
                range_y=[0, 1],
                title=t("chart_title"),
                text_auto=".3f",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            t("download_csv"),
            data=csv_bytes,
            file_name="ragas_results.csv",
            mime="text/csv",
        )

        event = results.get("event")
        if event:
            with st.expander(t("telemetry_summary")):
                st.json(
                    {
                        "run_id": event.run_id,
                        "status": event.status,
                        "duration_s": round(event.duration_seconds, 2),
                        "model": event.model,
                        "dataset_rows": event.dataset_rows,
                        "metrics": event.metrics,
                        "tokens": {
                            "prompt": event.token_usage.prompt_tokens,
                            "completion": event.token_usage.completion_tokens,
                            "total": event.token_usage.total_tokens,
                        },
                        "estimated_cost_usd": event.token_usage.estimated_cost_usd(
                            event.model
                        ),
                    }
                )

else:
    st.info(t("upload_prompt"))
