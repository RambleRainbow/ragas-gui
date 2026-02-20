"""Ragas RAG Evaluation -- Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

from ragas_gui.config import (
    QUICK_START_METRICS,
    MetricCategory,
    get_metric_info,
    list_metrics_by_category,
)
from ragas_gui.data import load_uploaded_file, validate_columns
from ragas_gui.evaluation import run_evaluation
from ragas_gui.i18n import (
    QUICK_START_PROVIDER_LABELS,
    QUICK_START_PROVIDERS,
    SUPPORTED_LANGUAGES,
    get_language,
    set_language,
    t,
)
from ragas_gui.llm_config import (
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

st.set_page_config(page_title=t("page_title"), page_icon="📊", layout="wide")

if "telemetry" not in st.session_state:
    st.session_state["telemetry"] = TelemetryManager()

if "saved_settings" not in st.session_state:
    st.session_state["saved_settings"] = {}

if "settings_loaded" not in st.session_state:
    st.session_state["settings_loaded"] = False

import json

loaded_settings = st.session_state.get("saved_settings", {})

if not st.session_state["settings_loaded"] and loaded_settings:
    st.session_state["settings_loaded"] = True

telemetry: TelemetryManager = st.session_state["telemetry"]

if "load_triggered" not in st.session_state:
    st.session_state["load_triggered"] = True

    js = """
    <script>
        var saved = localStorage.getItem('ragas_gui_settings');
        if (saved) {
            var settings = JSON.parse(saved);
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: settings
            }, '*');
        }
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)


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
# Mode toggle
# ---------------------------------------------------------------------------

st.title(t("app_title"))

mode = st.radio(
    t("mode_label"),
    [t("mode_quick_start"), t("mode_advanced")],
    horizontal=True,
    help=t("mode_help"),
)
is_advanced = mode == t("mode_advanced")

# ---------------------------------------------------------------------------
# Sidebar -- LLM & Embeddings config
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header(t("sidebar_header"))

    # ---- Provider selection (available in BOTH modes) ---------------------
    saved_provider = loaded_settings.get("provider", "openai")
    saved_model = loaded_settings.get("model", "")
    saved_base_url = loaded_settings.get("base_url", "")
    saved_temperature = loaded_settings.get("temperature", 0.0)
    saved_api_key = loaded_settings.get("api_key", "")

    if is_advanced:
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
    else:
        qs_labels = [QUICK_START_PROVIDER_LABELS[p] for p in QUICK_START_PROVIDERS]
        qs_choice = st.selectbox(t("provider"), qs_labels, index=0)
        _label_to_val = {v: k for k, v in QUICK_START_PROVIDER_LABELS.items()}
        llm_provider_enum = Provider(_label_to_val[qs_choice])
        llm_base_url = saved_base_url or DEFAULT_MODELS.get(llm_provider_enum, "")
        llm_model = saved_model or DEFAULT_MODELS.get(llm_provider_enum, "")
        llm_temperature = saved_temperature

    api_key = st.text_input(
        t("api_key"),
        type="password",
        value=saved_api_key,
        help=t("api_key_help"),
    )

    # ---- Save Settings Button -----------------------------------------------
    if st.button("💾 Save Settings", key="save_settings_btn"):
        settings = {
            "provider": llm_provider,
            "model": llm_model,
            "base_url": llm_base_url,
            "temperature": llm_temperature,
            "api_key": api_key,
        }
        import json
        settings_json = json.dumps(settings)
        save_js = f"""
        <script>
            localStorage.setItem('ragas_gui_settings', '{settings_json}');
        </script>
        """
        st.markdown(save_js, unsafe_allow_html=True)
        st.success("Settings saved!")

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
            import asyncio

            success, message = asyncio.run(test_llm_connection(llm_test_cfg))
        if success:
            st.success(t("connection_success"))
        else:
            st.error(t("connection_failed", error=message))

    if is_advanced:
        st.divider()
        st.subheader(t("embeddings"))
        emb_provider = st.selectbox(
            t("emb_provider"),
            [p.value for p in EmbeddingProvider],
            index=0,
        )
        emb_provider_enum = EmbeddingProvider(emb_provider)
        emb_base_url = st.text_input(
            t("emb_base_url"),
            value=DEFAULT_EMBEDDING_MODELS.get(emb_provider_enum, ""),
            help=t("emb_base_url_help"),
        )
        emb_model = st.text_input(
            t("emb_model"),
            value=DEFAULT_EMBEDDING_MODELS.get(emb_provider_enum, ""),
        )
        emb_api_key = st.text_input(
            t("emb_api_key"),
            type="password",
        )
        emb_cfg = EmbeddingConfig(
            provider=emb_provider_enum,
            base_url=emb_base_url,
            model_name=emb_model,
            api_key=emb_api_key or api_key,
        )
    else:
        emb_cfg = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            api_key=api_key,
        )

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

    if is_advanced:
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
                    checked = info.display_name in QUICK_START_METRICS
                    if st.checkbox(
                        info.display_name,
                        value=checked,
                        help=info.description,
                        key=f"metric_{info.name}",
                    ):
                        selected_metric_names.append(info.display_name)
    else:
        for name in QUICK_START_METRICS:
            info = get_metric_info(name)
            if st.checkbox(
                name, value=True, help=info.description, key=f"qs_{info.name}"
            ):
                selected_metric_names.append(name)

    # ---- Advanced: RunConfig, Telemetry -----------------------------------
    run_settings = RunSettings()

    if is_advanced:
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
            run_settings.raise_exceptions = st.checkbox(
                t("raise_exceptions"), value=False
            )

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
    st.dataframe(df.head(10), use_container_width=True)

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
        st.dataframe(result_df, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            t("download_csv"),
            data=csv_bytes,
            file_name="ragas_results.csv",
            mime="text/csv",
        )

        # Show telemetry summary if active
        event = results.get("event")
        if event and is_advanced:
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
