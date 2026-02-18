@metrics
Feature: Metric Configuration
  As a ragas user
  I want comprehensive metric configuration
  So that I can evaluate my RAG pipeline thoroughly

  Scenario: Metric catalogue contains all categories
    Given the metric catalogue is loaded
    Then it should contain metrics in category "RAG Core"
    And it should contain metrics in category "RAG Context"
    And it should contain metrics in category "Answer Quality"
    And it should contain metrics in category "Classic NLP"

  Scenario: Quick Start metrics are a subset of all metrics
    Given the metric catalogue is loaded
    When I get the Quick Start metrics
    Then each Quick Start metric should exist in the full catalogue

  Scenario: Metric classes can be imported
    Given the metric catalogue is loaded
    When I import the class for "Faithfulness"
    Then the imported class should not be None

  Scenario: Metrics are grouped by category
    Given the metric catalogue is loaded
    When I group metrics by category
    Then each group should have at least 1 metric
