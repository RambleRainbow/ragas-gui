from pytest_bdd import given, parsers, scenarios, then, when

from ragas_gui.telemetry import (
    EvaluationEvent,
    ExporterType,
    TelemetryConfig,
    TelemetryManager,
    TokenUsage,
)

scenarios("../features/telemetry.feature")


@given("a default TelemetryConfig", target_fixture="tel_config")
def default_config():
    return TelemetryConfig()


@then("telemetry should be disabled")
def check_disabled(tel_config):
    assert tel_config.enabled is False


@given("a TelemetryConfig with console exporter", target_fixture="tel_config")
def console_config():
    return TelemetryConfig(enabled=True, exporter=ExporterType.CONSOLE)


@when("I create a TelemetryManager", target_fixture="manager")
def create_manager(tel_config):
    return TelemetryManager(config=tel_config)


@then("the manager should be uninitialised")
def check_uninit(manager):
    assert manager._initialised is False


@given(
    parsers.parse('an EvaluationEvent with {rows:d} rows and model "{model}"'),
    target_fixture="event",
)
def create_event(rows, model):
    return EvaluationEvent(
        run_id="test-123",
        dataset_rows=rows,
        metrics=["faithfulness"],
        model=model,
    )


@then(parsers.parse('the event status should be "{status}"'))
def check_status(event, status):
    assert event.status == status


@then(parsers.parse("the event should have {rows:d} dataset_rows"))
def check_rows(event, rows):
    assert event.dataset_rows == rows


@given(
    parsers.parse(
        "token usage of {prompt:d} prompt and {completion:d} completion tokens"
    ),
    target_fixture="token_usage",
)
def create_usage(prompt, completion):
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion)


@when(
    parsers.parse('I estimate cost for model "{model}"'),
    target_fixture="cost",
)
def estimate_cost(token_usage, model):
    return token_usage.estimated_cost_usd(model)


@then("the cost should be greater than 0")
def check_cost_positive(cost):
    assert cost is not None
    assert cost > 0


@then("the cost should be None")
def check_cost_none(cost):
    assert cost is None
