@llm_config
Feature: LLM and Embeddings Configuration
  As a ragas user
  I want to configure different LLM and embedding providers
  So that I can use my preferred AI services

  Scenario: Default LLM config uses OpenAI
    Given a default LLMConfig
    Then the provider should be "openai"
    And the model should be "gpt-4o-mini"
    And the config should not be configured

  Scenario: LLM config is configured when API key is set
    Given a LLMConfig with api_key "sk-test-123"
    Then the config should be configured

  Scenario: Default embedding config uses OpenAI
    Given a default EmbeddingConfig
    Then the embedding provider should be "openai"
    And the embedding model should be "text-embedding-3-small"

  Scenario: RunSettings have sensible defaults
    Given default RunSettings
    Then timeout should be 180
    And max_retries should be 10
    And max_workers should be 16
    And seed should be 42
    And batch_size should be None
    And raise_exceptions should be False
