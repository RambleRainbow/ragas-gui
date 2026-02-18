from pytest_bdd import given, parsers, scenarios, then, when

from ragas_gui.config import (
    METRIC_CATALOGUE,
    METRIC_REGISTRY,
    QUICK_START_METRICS,
    MetricCategory,
    get_metric_class,
    list_metrics_by_category,
)

scenarios("../features/metric_configuration.feature")


@given("the metric catalogue is loaded", target_fixture="catalogue")
def load_catalogue():
    return METRIC_CATALOGUE


@then(parsers.parse('it should contain metrics in category "{category}"'))
def check_category(catalogue, category):
    cats = {m.category.value for m in catalogue}
    assert category in cats


@when("I get the Quick Start metrics", target_fixture="qs_metrics")
def get_qs():
    return QUICK_START_METRICS


@then("each Quick Start metric should exist in the full catalogue")
def qs_in_catalogue(qs_metrics):
    for name in qs_metrics:
        assert name in METRIC_REGISTRY, f"{name} not in registry"


@when(
    parsers.parse('I import the class for "{name}"'),
    target_fixture="metric_cls",
)
def import_cls(name):
    return get_metric_class(name)


@then("the imported class should not be None")
def check_not_none(metric_cls):
    assert metric_cls is not None


@when("I group metrics by category", target_fixture="grouped")
def group_metrics():
    return list_metrics_by_category()


@then("each group should have at least 1 metric")
def check_groups(grouped):
    for cat, infos in grouped.items():
        assert len(infos) >= 1, f"Category {cat} is empty"
