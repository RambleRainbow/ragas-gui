@observability
Feature: Telemetry and Observability
  As a platform engineer
  I want OpenTelemetry integration
  So that I can monitor evaluation runs

  Scenario: Telemetry is disabled by default
    Given a default TelemetryConfig
    Then telemetry should be disabled

  Scenario: Enable console exporter
    Given a TelemetryConfig with console exporter
    When I create a TelemetryManager
    Then the manager should be uninitialised

  Scenario: Evaluation event tracks metadata
    Given an EvaluationEvent with 10 rows and model "gpt-4o-mini"
    Then the event status should be "pending"
    And the event should have 10 dataset_rows

  Scenario: Token usage cost estimation
    Given token usage of 1000 prompt and 500 completion tokens
    When I estimate cost for model "gpt-4o-mini"
    Then the cost should be greater than 0

  Scenario: Unknown model returns no cost
    Given token usage of 100 prompt and 50 completion tokens
    When I estimate cost for model "unknown-model"
    Then the cost should be None
