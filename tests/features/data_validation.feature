@validation
Feature: Data Upload and Validation
  As a ragas user
  I want to upload datasets and have them validated
  So that I can run evaluations on correct data

  Scenario: Load a valid CSV file
    Given a CSV file with columns "question,answer,contexts,ground_truth"
    When I load the file
    Then the dataframe should have 0 missing required columns
    And the dataframe should have 4 columns

  Scenario: Detect missing required columns
    Given a CSV file with columns "question,answer"
    When I validate the columns
    Then the missing columns should be "contexts,ground_truth"

  Scenario: Accept extra columns without error
    Given a CSV file with columns "question,answer,contexts,ground_truth,extra_col"
    When I validate the columns
    Then the dataframe should have 0 missing required columns

  Scenario: Parse JSON-encoded contexts
    Given a contexts value of '["ctx1", "ctx2"]'
    When I parse the contexts
    Then the result should be a list with 2 items
    And the first item should be "ctx1"

  Scenario: Parse plain string contexts
    Given a contexts value of 'single context'
    When I parse the contexts
    Then the result should be a list with 1 items
    And the first item should be "single context"

  Scenario: Parse already-list contexts
    Given a contexts value that is already a list
    When I parse the contexts
    Then the result should be a list with 2 items

  Scenario: Reject unsupported file type
    Given a file named "data.xlsx"
    When I try to load the file
    Then it should raise a ValueError
