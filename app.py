"""Ragas RAG Evaluation -- Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import os
import sys

# Ensure src/ is in the path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

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
    CompatibilityMode,
    EmbeddingConfig,
    EmbeddingProvider,
    LLMConfig,
    LLMProvider,
    RunSettings,
)
from ragas_gui.telemetry import (
    ExporterType,
    TelemetryConfig,
    TelemetryManager,
)

st.set_page_config(page_title=t("page_title"), page_icon="📊", layout="wide")

if "telemetry" not in st.session_state:
    st.session_state["telemetry"] = TelemetryManager()

telemetry: TelemetryManager = st.session_state["telemetry"]


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
    if is_advanced:
        llm_provider = st.selectbox(
            t("llm_provider"),
            [p.value for p in LLMProvider],
            index=0,
        )
        llm_provider_enum = LLMProvider(llm_provider)
        llm_model = st.text_input(
            t("llm_model"),
            value=DEFAULT_MODELS[llm_provider_enum],
        )
        llm_temperature = st.slider(
            t("temperature"),
            0.0,
            2.0,
            0.0,
            0.05,
        )
    else:
        qs_labels = [QUICK_START_PROVIDER_LABELS[p] for p in QUICK_START_PROVIDERS]
        qs_choice = st.selectbox(t("provider"), qs_labels, index=0)
        _label_to_val = {v: k for k, v in QUICK_START_PROVIDER_LABELS.items()}
        llm_provider_enum = LLMProvider(_label_to_val[qs_choice])
        llm_model = DEFAULT_MODELS[llm_provider_enum]
        llm_temperature = 0.0

    api_key = st.text_input(
        t("api_key"),
        type="password",
        help=t("api_key_help"),
    )

    if is_advanced:
        with st.expander(t("advanced_connection"), expanded=False):
            llm_api_base = st.text_input(
                t("api_base_url"),
                value="",
                help=t("api_base_help"),
            )

            compat_labels = {
                CompatibilityMode.NONE: t("compat_none"),
                CompatibilityMode.OPENAI_COMPATIBLE: t("compat_openai"),
                CompatibilityMode.ANTHROPIC_COMPATIBLE: t("compat_anthropic"),
            }
            compat_selection = st.selectbox(
                t("compat_mode"),
                list(compat_labels.values()),
                index=0,
                help=t("compat_help"),
            )
            compat_mode = next(
                k for k, v in compat_labels.items() if v == compat_selection
            )
    else:
        compat_mode = CompatibilityMode.NONE
        llm_api_base = ""

    llm_cfg = LLMConfig(
        provider=llm_provider_enum,
        model=llm_model,
        api_key=api_key,
        api_base=llm_api_base,
        compatibility_mode=compat_mode,
        temperature=llm_temperature,
    )

    if is_advanced:
        st.divider()
        st.subheader(t("embeddings"))
        emb_provider = st.selectbox(
            t("emb_provider"),
            [p.value for p in EmbeddingProvider],
            index=0,
        )
        emb_provider_enum = EmbeddingProvider(emb_provider)
        emb_model = st.text_input(
            t("emb_model"),
            value=DEFAULT_EMBEDDING_MODELS[emb_provider_enum],
        )
        emb_api_key = st.text_input(
            t("emb_api_key"),
            type="password",
        )
        emb_api_base = st.text_input(
            t("emb_api_base"),
            value="",
            help=t("emb_api_base_help"),
        )
        emb_cfg = EmbeddingConfig(
            provider=emb_provider_enum,
            model=emb_model,
            api_key=emb_api_key or api_key,
            api_base=emb_api_base,
        )
    else:
        emb_cfg = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            api_key=api_key,
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
