from pytest_bdd import given, parsers, scenarios, then, when

from ragas_gui.llm_config import (
    EmbeddingConfig,
    LLMConfig,
    RunSettings,
)

scenarios("../features/llm_configuration.feature")


@given("a default LLMConfig", target_fixture="llm_cfg")
def default_llm():
    return LLMConfig()


@then(parsers.parse('the provider should be "{provider}"'))
def check_provider(llm_cfg, provider):
    assert llm_cfg.provider.value == provider


@then(parsers.parse('the model should be "{model}"'))
def check_model(llm_cfg, model):
    assert llm_cfg.model == model


@then("the config should not be configured")
def check_not_configured(llm_cfg):
    assert not llm_cfg.is_configured


@given(
    parsers.parse('a LLMConfig with api_key "{key}"'),
    target_fixture="llm_cfg",
)
def llm_with_key(key):
    return LLMConfig(api_key=key)


@then("the config should be configured")
def check_configured(llm_cfg):
    assert llm_cfg.is_configured


@given("a default EmbeddingConfig", target_fixture="emb_cfg")
def default_emb():
    return EmbeddingConfig()


@then(parsers.parse('the embedding provider should be "{provider}"'))
def check_emb_provider(emb_cfg, provider):
    assert emb_cfg.provider.value == provider


@then(parsers.parse('the embedding model should be "{model}"'))
def check_emb_model(emb_cfg, model):
    assert emb_cfg.model == model


@given("default RunSettings", target_fixture="run_settings")
def default_run():
    return RunSettings()


@then(parsers.parse("timeout should be {val:d}"))
def check_timeout(run_settings, val):
    assert run_settings.timeout == val


@then(parsers.parse("max_retries should be {val:d}"))
def check_retries(run_settings, val):
    assert run_settings.max_retries == val


@then(parsers.parse("max_workers should be {val:d}"))
def check_workers(run_settings, val):
    assert run_settings.max_workers == val


@then(parsers.parse("seed should be {val:d}"))
def check_seed(run_settings, val):
    assert run_settings.seed == val


@then("batch_size should be None")
def check_batch_none(run_settings):
    assert run_settings.batch_size is None


@then("raise_exceptions should be False")
def check_raise(run_settings):
    assert run_settings.raise_exceptions is False
