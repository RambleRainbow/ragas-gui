"""Ragas RAG Evaluation -- Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from ragas_gui.config import (
    QUICK_START_METRICS,
    MetricCategory,
    list_metrics_by_category,
    get_metric_info,
)
from ragas_gui.data import load_uploaded_file, validate_columns
from ragas_gui.evaluation import run_evaluation
from ragas_gui.llm_config import (
    EmbeddingConfig,
    EmbeddingProvider,
    LLMConfig,
    LLMProvider,
    RunSettings,
    DEFAULT_MODELS,
    DEFAULT_EMBEDDING_MODELS,
)
from ragas_gui.telemetry import (
    ExporterType,
    TelemetryConfig,
    TelemetryManager,
)

st.set_page_config(page_title="Ragas Evaluator", page_icon="📊", layout="wide")

if "telemetry" not in st.session_state:
    st.session_state["telemetry"] = TelemetryManager()

telemetry: TelemetryManager = st.session_state["telemetry"]


# ---------------------------------------------------------------------------
# Mode toggle
# ---------------------------------------------------------------------------

st.title("📊 Ragas RAG Evaluation")

mode = st.radio(
    "Mode",
    ["🚀 Quick Start", "⚙️ Advanced"],
    horizontal=True,
    help="Quick Start: sensible defaults, fewer options. Advanced: full control.",
)
is_advanced = mode == "⚙️ Advanced"

# ---------------------------------------------------------------------------
# Sidebar -- LLM & Embeddings config
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🔑 API Configuration")

    if is_advanced:
        llm_provider = st.selectbox(
            "LLM Provider",
            [p.value for p in LLMProvider],
            index=0,
        )
        llm_provider_enum = LLMProvider(llm_provider)
        llm_model = st.text_input(
            "LLM Model",
            value=DEFAULT_MODELS[llm_provider_enum],
        )
        llm_temperature = st.slider(
            "Temperature",
            0.0,
            2.0,
            0.0,
            0.05,
        )
    else:
        llm_provider_enum = LLMProvider.OPENAI
        llm_model = DEFAULT_MODELS[LLMProvider.OPENAI]
        llm_temperature = 0.0

    api_key = st.text_input(
        "API Key",
        type="password",
        help="Required by Ragas for LLM-based metrics.",
    )

    llm_cfg = LLMConfig(
        provider=llm_provider_enum,
        model=llm_model,
        api_key=api_key,
        temperature=llm_temperature,
    )

    if is_advanced:
        st.divider()
        st.subheader("Embeddings")
        emb_provider = st.selectbox(
            "Embedding Provider",
            [p.value for p in EmbeddingProvider],
            index=0,
        )
        emb_provider_enum = EmbeddingProvider(emb_provider)
        emb_model = st.text_input(
            "Embedding Model",
            value=DEFAULT_EMBEDDING_MODELS[emb_provider_enum],
        )
        emb_api_key = st.text_input(
            "Embedding API Key (leave blank to reuse above)",
            type="password",
        )
        emb_cfg = EmbeddingConfig(
            provider=emb_provider_enum,
            model=emb_model,
            api_key=emb_api_key or api_key,
        )
    else:
        emb_cfg = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            api_key=api_key,
        )

    # ---- Metrics selection ------------------------------------------------
    st.divider()
    st.subheader("Metrics")

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
        with st.expander("Runtime Settings"):
            run_settings.timeout = st.number_input(
                "Timeout (s)", value=180, min_value=1
            )
            run_settings.max_retries = st.number_input(
                "Max Retries", value=10, min_value=0
            )
            run_settings.max_workers = st.number_input(
                "Max Workers", value=16, min_value=1
            )
            run_settings.seed = st.number_input("Seed", value=42)
            bs = st.number_input("Batch Size (0 = auto)", value=0, min_value=0)
            run_settings.batch_size = bs if bs > 0 else None
            run_settings.raise_exceptions = st.checkbox("Raise Exceptions", value=False)

        with st.expander("Observability (OpenTelemetry)"):
            otel_enabled = st.checkbox("Enable Tracing", value=False)
            if otel_enabled:
                exporter_val = st.selectbox(
                    "Exporter",
                    [e.value for e in ExporterType],
                    index=0,
                )
                otlp_endpoint = st.text_input(
                    "OTLP Endpoint",
                    value="http://localhost:4318",
                    help="Only for OTLP exporters.",
                )
                trace_content = st.checkbox("Log Prompt/Completion Content", value=True)
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
    st.caption("Built with Streamlit + Ragas")

# ---------------------------------------------------------------------------
# Main area -- upload, preview, evaluate
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload evaluation dataset (CSV or JSON)",
    type=["csv", "json", "jsonl"],
    help="Must contain columns: question, answer, contexts, ground_truth",
)

if uploaded_file is not None:
    try:
        df = load_uploaded_file(uploaded_file)
    except Exception as exc:
        st.error(f"Failed to read file: {exc}")
        st.stop()

    st.subheader("📄 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    missing = validate_columns(df)
    if missing:
        st.error(
            f"Missing required columns: **{', '.join(missing)}**. "
            "Expected: `question`, `answer`, `contexts`, `ground_truth`."
        )
        st.stop()

    st.success(f"Dataset loaded -- {len(df)} rows, all required columns present.")

    if not api_key:
        st.info("Enter your API key in the sidebar to enable evaluation.")
    if not selected_metric_names:
        st.info("Select at least one metric in the sidebar.")

    run_disabled = not api_key or not selected_metric_names

    if st.button("🚀 Run Evaluation", type="primary", disabled=run_disabled):
        metric_infos = [get_metric_info(n) for n in selected_metric_names]

        with st.spinner("Running Ragas evaluation... this may take a few minutes."):
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
                st.error(f"Evaluation failed: {exc}")
                st.stop()

        st.subheader("📈 Evaluation Results")
        result_df = results["result_df"]
        st.dataframe(result_df, use_container_width=True)

        avg_scores = results["avg_scores"]
        if avg_scores:
            chart_df = pd.DataFrame(
                {
                    "Metric": list(avg_scores.keys()),
                    "Average Score": list(avg_scores.values()),
                }
            )
            fig = px.bar(
                chart_df,
                x="Metric",
                y="Average Score",
                color="Metric",
                range_y=[0, 1],
                title="Average Metric Scores",
                text_auto=".3f",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_bytes,
            file_name="ragas_results.csv",
            mime="text/csv",
        )

        # Show telemetry summary if active
        event = results.get("event")
        if event and is_advanced:
            with st.expander("Telemetry Summary"):
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
    st.info("Upload a CSV or JSON file to get started.")
