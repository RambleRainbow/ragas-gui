# Ragas GUI

Streamlit-based evaluation UI for [Ragas](https://docs.ragas.io) RAG pipelines.

Upload a dataset, pick your metrics, and get instant evaluation results with interactive charts.

## Quick Start

```bash
# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install the package with dependencies
pip install -e .

# Run the app
streamlit run app.py
```

## Project Structure

```
ragas-gui/
├── app.py                  # Streamlit entry point
├── src/
│   └── ragas_gui/
│       ├── __init__.py     # Public API exports
│       ├── config.py       # Metrics registry & constants
│       └── data.py         # Data loading, validation, dataset construction
├── tests/
│   └── test_data.py        # Unit tests for data utilities
├── data/
│   └── sample_data.json    # Example evaluation dataset
├── pyproject.toml           # Project metadata & build config
└── requirements.txt         # Legacy flat dependency list
```

## Dataset Format

Your CSV or JSON must contain these columns:

| Column         | Type         | Description                    |
|----------------|--------------|--------------------------------|
| `question`     | `str`        | The user question              |
| `answer`       | `str`        | The RAG-generated answer       |
| `contexts`     | `list[str]`  | Retrieved context passages     |
| `ground_truth` | `str`        | Expected correct answer        |

See `data/sample_data.json` for an example.

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
