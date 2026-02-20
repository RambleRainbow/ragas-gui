"""Data loading, validation, and dataset construction utilities."""

import ast
import json

import pandas as pd
from datasets import Dataset


def parse_contexts(value) -> list[str]:
    """Parse a contexts value into ``list[str]``.

    Handles:
      - Already a list  -> return as-is
      - JSON string     -> json.loads
      - Python literal  -> ast.literal_eval
      - Plain string    -> wrap in a single-element list
    """
    if isinstance(value, list):
        return [str(v) for v in value]
    if not isinstance(value, str):
        return [str(value)]

    value = value.strip()
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except (ValueError, SyntaxError):
            pass
    return [value]


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read an uploaded file (CSV / JSON / JSONL) into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".json") or name.endswith(".jsonl"):
        return pd.read_json(uploaded_file)
    raise ValueError(f"Unsupported file type: {name}")


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted list of missing required columns."""
    required = {"question", "answer", "contexts", "ground_truth"}
    return sorted(required - set(df.columns))


def build_ragas_dataset(df: pd.DataFrame) -> Dataset:
    """Convert a validated DataFrame into a Ragas-compatible HF Dataset.

    Note: ragas 0.4+ uses new column names:
      - user_input (was question)
      - response (was answer)
      - retrieved_contexts (was contexts)
      - reference (was ground_truth)
    """
    return Dataset.from_dict(
        {
            "user_input": df["question"].astype(str).tolist(),
            "response": df["answer"].astype(str).tolist(),
            "retrieved_contexts": df["contexts"].apply(parse_contexts).tolist(),
            "reference": df["ground_truth"].astype(str).tolist(),
        }
    )
