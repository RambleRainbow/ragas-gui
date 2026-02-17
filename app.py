"""Ragas RAG Evaluation -- Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import os

import nest_asyncio
import pandas as pd
import plotly.express as px
import streamlit as st

from ragas import evaluate
from ragas_gui.config import METRICS
from ragas_gui.data import build_ragas_dataset, load_uploaded_file, validate_columns

# Ragas uses async internally; Streamlit has its own event loop.
nest_asyncio.apply()

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Ragas Evaluator", page_icon="📊", layout="wide")
st.title("📊 Ragas RAG Evaluation")
st.markdown(
    "Upload a dataset, choose your metrics, and evaluate your RAG pipeline with "
    "[Ragas](https://docs.ragas.io)."
)

with st.sidebar:
    st.header("⚙️ Configuration")

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Required by Ragas for LLM-based metrics.",
    )

    st.divider()
    st.subheader("Metrics")
    selected_metrics: list[str] = []
    for name in METRICS:
        if st.checkbox(name, value=True):
            selected_metrics.append(name)

    st.divider()
    st.caption("Built with Streamlit + Ragas")

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

    st.success(f"✅ Dataset loaded — {len(df)} rows, all required columns present.")

    if not openai_api_key:
        st.info("Enter your OpenAI API key in the sidebar to enable evaluation.")
    if not selected_metrics:
        st.info("Select at least one metric in the sidebar.")

    run_disabled = not openai_api_key or not selected_metrics

    if st.button("🚀 Run Evaluation", type="primary", disabled=run_disabled):
        os.environ["OPENAI_API_KEY"] = openai_api_key

        metrics_to_run = [METRICS[m] for m in selected_metrics]

        with st.spinner(
            "Building dataset & running Ragas evaluation… this may take a few minutes."
        ):
            try:
                ragas_ds = build_ragas_dataset(df)
                result = evaluate(ragas_ds, metrics=metrics_to_run)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                st.stop()

        st.subheader("📈 Evaluation Results")

        result_df = result.to_pandas()
        st.dataframe(result_df, use_container_width=True)

        score_cols = [
            c
            for c in result_df.columns
            if c not in {"question", "answer", "contexts", "ground_truth"}
        ]
        if score_cols:
            avg_scores = result_df[score_cols].mean()
            chart_df = pd.DataFrame(
                {"Metric": avg_scores.index, "Average Score": avg_scores.values}
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
else:
    st.info("Upload a CSV or JSON file to get started.")
