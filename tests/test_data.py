"""Unit tests for ragas_gui.data utilities."""

import pandas as pd
import pytest

from ragas_gui.data import parse_contexts, validate_columns


# -- parse_contexts ----------------------------------------------------------


class TestParseContexts:
    def test_list_passthrough(self):
        assert parse_contexts(["a", "b"]) == ["a", "b"]

    def test_list_coerces_to_str(self):
        assert parse_contexts([1, 2]) == ["1", "2"]

    def test_json_string(self):
        assert parse_contexts('["x", "y"]') == ["x", "y"]

    def test_python_literal(self):
        assert parse_contexts("['x', 'y']") == ["x", "y"]

    def test_plain_string(self):
        assert parse_contexts("hello world") == ["hello world"]

    def test_non_string_non_list(self):
        assert parse_contexts(42) == ["42"]

    def test_empty_list(self):
        assert parse_contexts([]) == []

    def test_malformed_bracket_string(self):
        assert parse_contexts("[not valid json") == ["[not valid json"]


# -- validate_columns --------------------------------------------------------


class TestValidateColumns:
    def test_all_present(self):
        df = pd.DataFrame(columns=["question", "answer", "contexts", "ground_truth"])
        assert validate_columns(df) == []

    def test_missing_some(self):
        df = pd.DataFrame(columns=["question", "answer"])
        assert validate_columns(df) == ["contexts", "ground_truth"]

    def test_extra_columns_ignored(self):
        df = pd.DataFrame(
            columns=["question", "answer", "contexts", "ground_truth", "extra"]
        )
        assert validate_columns(df) == []
