import io

import pandas as pd
from pytest_bdd import given, parsers, scenarios, then, when

from ragas_gui.data import load_uploaded_file, parse_contexts, validate_columns

scenarios("../features/data_validation.feature")


@given(
    parsers.parse('a CSV file with columns "{columns}"'),
    target_fixture="csv_file",
)
def csv_with_columns(columns, tmp_path):
    cols = [c.strip() for c in columns.split(",")]
    df = pd.DataFrame({c: ["val"] for c in cols})
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    buf = io.BytesIO(path.read_bytes())
    buf.name = "test.csv"
    return {"buffer": buf, "df": df}


@given(
    parsers.parse("a contexts value of '{value}'"),
    target_fixture="contexts_input",
)
def contexts_string_value(value):
    return value


@given("a contexts value that is already a list", target_fixture="contexts_input")
def contexts_already_list():
    return ["item1", "item2"]


@given(parsers.parse('a file named "{filename}"'), target_fixture="bad_file")
def file_with_name(filename):
    buf = io.BytesIO(b"data")
    buf.name = filename
    return buf


@when("I load the file", target_fixture="loaded_df")
def load_file(csv_file):
    return load_uploaded_file(csv_file["buffer"])


@when("I validate the columns", target_fixture="missing_cols")
def validate_cols(csv_file):
    return validate_columns(csv_file["df"])


@when("I parse the contexts", target_fixture="parsed")
def parse_ctx(contexts_input):
    return parse_contexts(contexts_input)


@when("I try to load the file", target_fixture="load_error")
def try_load_bad(bad_file):
    try:
        load_uploaded_file(bad_file)
        return None
    except ValueError as exc:
        return exc


@then(parsers.parse("the dataframe should have {count:d} missing required columns"))
def check_missing_count(csv_file, count):
    missing = validate_columns(csv_file["df"])
    assert len(missing) == count


@then(parsers.parse("the dataframe should have {count:d} columns"))
def check_col_count(loaded_df, count):
    assert len(loaded_df.columns) == count


@then(parsers.parse('the missing columns should be "{expected}"'))
def check_missing_names(missing_cols, expected):
    expected_list = sorted(c.strip() for c in expected.split(","))
    assert missing_cols == expected_list


@then(parsers.parse("the result should be a list with {count:d} items"))
def check_list_length(parsed, count):
    assert isinstance(parsed, list)
    assert len(parsed) == count


@then(parsers.parse('the first item should be "{value}"'))
def check_first_item(parsed, value):
    assert parsed[0] == value


@then("it should raise a ValueError")
def check_value_error(load_error):
    assert isinstance(load_error, ValueError)
