# AGENTS.md - Ragas GUI Project Guidelines

## Project Overview

Streamlit-based evaluation UI for Ragas RAG pipelines. Users upload datasets, configure LLM providers, select metrics, and run evaluations with observability support (LangFuse, OpenTelemetry).

**Tech Stack**: Python 3.10-3.13, Streamlit, Ragas, LangChain, pandas, plotly

---

## Build / Lint / Test Commands

```bash
# Install (development mode with all extras)
pip install -e ".[all]"

# Run the application
streamlit run app.py

# Lint (ruff)
ruff check src/ tests/ app.py
ruff format src/ tests/ app.py

# Run all tests
pytest

# Run specific test file
pytest tests/test_data.py

# Run specific test marker
pytest -m validation
pytest -m bdd
pytest -m llm_config

# Run single test by name
pytest tests/test_data.py::test_parse_contexts -v

# Run BDD scenario tests
pytest tests/step_defs/test_data_validation.py -v
```

**Available markers**: `bdd`, `unit`, `validation`, `metrics`, `observability`, `llm_config`

---

## Code Style Guidelines

### Imports

Order: stdlib → third-party → local. Group with blank lines.

```python
# stdlib
import ast
import json
from contextlib import contextmanager

# third-party
import pandas as pd
from datasets import Dataset

# local
from ragas_gui.config import MetricInfo
```

### Type Hints

Always use type hints for function signatures:

```python
def load_uploaded_file(uploaded_file) -> pd.DataFrame:
def validate_columns(df: pd.DataFrame) -> list[str]:
def run_evaluation(
    df: pd.DataFrame,
    metric_infos: list[MetricInfo],
    telemetry: TelemetryManager | None = None,
) -> dict[str, Any]:
```

Use `|` for unions (Python 3.10+), `list[str]` not `List[str]`.

### Naming Conventions

- **Functions/variables**: `snake_case` (`load_uploaded_file`, `missing_cols`)
- **Classes**: `PascalCase` (`MetricInfo`, `TelemetryManager`)
- **Constants**: `UPPER_SNAKE_CASE` (`METRIC_CATALOGUE`, `DEFAULT_MODELS`)
- **Private functions**: `_leading_underscore` (`_import_metric_class`)
- **Enum members**: `UPPER_SNAKE_CASE` (`RAG_CORE`, `ANSWER_QUALITY`)

### Dataclasses

Use `@dataclass` for data containers:

```python
@dataclass
class MetricInfo:
    name: str
    display_name: str
    category: MetricCategory
    needs_llm: bool = True

@dataclass(frozen=True)  # for immutable data
class TokenUsage:
    prompt_tokens: int = 0
```

### Error Handling

- Raise specific exceptions with helpful messages
- Use try/except for expected failures

```python
raise ValueError(f"Unsupported file type: {name}")

try:
    parsed = json.loads(value)
except json.JSONDecodeError:
    pass  # fallback to other parsing
```

### Docstrings

Module-level docstrings at top of file. Function docstrings for public APIs:

```python
"""Module description here."""

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read an uploaded file (CSV / JSON / JSONL) into a DataFrame."""
```

Short one-liners preferred over verbose multi-line.

---

## Project Structure

```
ragas-gui/
├── app.py                    # Streamlit entry point (UI logic)
├── src/ragas_gui/
│   ├── __init__.py           # Public API exports
│   ├── config.py             # Metrics registry, constants
│   ├── data.py               # Data loading, validation
│   ├── evaluation.py         # ragas.evaluate() wrapper
│   ├── llm_config.py         # LLM/embeddings config dataclasses
│   ├── telemetry.py          # OpenTelemetry instrumentation
│   └── i18n.py               # Internationalization (en/zh)
├── tests/
│   ├── test_data.py          # Unit tests
│   ├── features/*.feature    # BDD feature files
│   └── step_defs/*.py        # pytest-bdd step definitions
├── pyproject.toml            # Project config, ruff, pytest
└── requirements.txt          # Flat dependency list
```

---

## Key Patterns

### Lazy Imports

Heavy imports (ragas, langchain) inside functions to speed up startup:

```python
def run_evaluation(...):
    from ragas import evaluate
    from ragas.cost import get_token_usage_for_openai
```

### Session State

Streamlit session_state for persistent objects:

```python
if "telemetry" not in st.session_state:
    st.session_state["telemetry"] = TelemetryManager()
```

### Translation

Use `t()` function for all UI text:

```python
st.header(t("sidebar_header"))
st.error(t("file_read_error", error=str(exc)))
```

---

## Testing

### BDD Tests (pytest-bdd)

Feature files in `tests/features/`, step definitions in `tests/step_defs/`:

```gherkin
@validation
Feature: Data Upload and Validation
  Scenario: Load a valid CSV file
    Given a CSV file with columns "question,answer,contexts,ground_truth"
    When I load the file
    Then the dataframe should have 0 missing required columns
```

```python
from pytest_bdd import given, when, then, scenarios

scenarios("../features/data_validation.feature")

@given(parsers.parse('a CSV file with columns "{columns}"'))
def csv_with_columns(columns, tmp_path):
    ...
```

---

## Constraints

- **Python version**: 3.10-3.13 (NOT 3.14 due to pydantic v1 incompatibility in some deps)
- **No type suppression**: Never use `as any`, `@ts-ignore`, `# type: ignore`
- **No empty catch blocks**: Always handle or re-raise exceptions
