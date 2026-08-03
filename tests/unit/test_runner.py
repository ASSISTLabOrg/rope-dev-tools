"""runner.validate()/recheck_report() — runs a suite against a model, re-evaluates thresholds."""

from __future__ import annotations

from rope_dev_tools.validation.checks import register_kind
from rope_dev_tools.validation.runner import recheck_report, validate
from rope_dev_tools.validation.schema_types import ValidationSuite


def _suite(threshold):
    return ValidationSuite(1, 1, [{"id": "c1", "kind": "satellite_orbit_density", "threshold": threshold}])


@register_kind("_test_dummy_kind")
def _dummy_kind(model, *, id=None, out_dir=None, suite_dir=None, **_):
    return {"plots": [], "data": []}


@register_kind("_test_label_capturing_kind")
def _label_capturing_kind(model, *, id=None, out_dir=None, suite_dir=None,
                           physics_model_label=None, rope_model_label=None, satellite_label=None, **_):
    return {"plots": [], "data": [], "labels_seen": {
        "physics_model_label": physics_model_label,
        "rope_model_label": rope_model_label,
        "satellite_label": satellite_label,
    }}


def test_validate_threads_suite_labels_to_checks(tmp_path):
    suite = ValidationSuite(
        1, 1, [{"id": "c1", "kind": "_test_label_capturing_kind"}],
        physics_model_label="WAM", rope_model_label="ROPE-WAM-V1", satellite_label="GRACE",
    )
    report = validate(None, suite, tmp_path, suite_dir=tmp_path)
    assert report["results"][0]["output"]["labels_seen"] == {
        "physics_model_label": "WAM", "rope_model_label": "ROPE-WAM-V1", "satellite_label": "GRACE",
    }


def test_validate_labels_default_to_none_when_suite_omits_them(tmp_path):
    suite = ValidationSuite(1, 1, [{"id": "c1", "kind": "_test_label_capturing_kind"}])
    report = validate(None, suite, tmp_path, suite_dir=tmp_path)
    assert report["results"][0]["output"]["labels_seen"] == {
        "physics_model_label": None, "rope_model_label": None, "satellite_label": None,
    }


def test_validate_only_check_ids_filters_suite(tmp_path):
    suite = ValidationSuite(1, 1, [
        {"id": "keep", "kind": "_test_dummy_kind"},
        {"id": "skip", "kind": "_test_dummy_kind"},
    ])
    report = validate(None, suite, tmp_path, suite_dir=tmp_path, only_check_ids=["keep"])
    assert [r["id"] for r in report["results"]] == ["keep"]


def test_validate_no_filter_runs_every_check(tmp_path):
    suite = ValidationSuite(1, 1, [
        {"id": "a", "kind": "_test_dummy_kind"},
        {"id": "b", "kind": "_test_dummy_kind"},
    ])
    report = validate(None, suite, tmp_path, suite_dir=tmp_path)
    assert {r["id"] for r in report["results"]} == {"a", "b"}


def test_validate_calls_progress_once_per_check(tmp_path):
    suite = ValidationSuite(1, 1, [
        {"id": "a", "kind": "_test_dummy_kind"},
        {"id": "b", "kind": "_test_dummy_kind"},
        {"id": "c", "kind": "_test_dummy_kind"},
    ])
    calls = []
    validate(None, suite, tmp_path, suite_dir=tmp_path, progress=lambda i, total, cid: calls.append((i, total, cid)))
    assert calls == [(0, 3, "a"), (1, 3, "b"), (2, 3, "c")]


def test_validate_progress_respects_only_check_ids_filter(tmp_path):
    suite = ValidationSuite(1, 1, [
        {"id": "keep", "kind": "_test_dummy_kind"},
        {"id": "skip", "kind": "_test_dummy_kind"},
    ])
    calls = []
    validate(None, suite, tmp_path, suite_dir=tmp_path, only_check_ids=["keep"],
              progress=lambda i, total, cid: calls.append((i, total, cid)))
    assert calls == [(0, 1, "keep")]


def test_recheck_report_scalar_value_path():
    report = {"results": [{"id": "c1", "output": {"value": 5.0, "passed": True}}]}
    rechecked = recheck_report(report, _suite({"max": 1.0}))
    assert rechecked["results"][0]["output"]["passed"] is False


def test_recheck_report_scalar_value_path_no_threshold_leaves_output_unchanged():
    report = {"results": [{"id": "c1", "output": {"value": 5.0, "passed": True}}]}
    rechecked = recheck_report(report, _suite(None))
    assert rechecked["results"][0]["output"] == {"value": 5.0, "passed": True}


def test_recheck_report_per_period_recomputes_each_and_top_level_and():
    # per_period is nested one level deeper by start_delta -- even a single "delta_+0h" key.
    report = {
        "results": [{
            "id": "c1",
            "output": {
                "per_period": {
                    "p1": {"delta_+0h": {"value": 0.5, "passed": True}},
                    "p2": {"delta_+0h": {"value": 5.0, "passed": True}},
                },
                "passed": True,
            },
        }],
    }
    rechecked = recheck_report(report, _suite({"max": 1.0}))
    output = rechecked["results"][0]["output"]
    assert output["per_period"]["p1"]["delta_+0h"]["passed"] is True
    assert output["per_period"]["p2"]["delta_+0h"]["passed"] is False
    assert output["passed"] is False


def test_recheck_report_per_period_all_pass():
    report = {
        "results": [{
            "id": "c1",
            "output": {"per_period": {
                "p1": {"delta_+0h": {"value": 0.1, "passed": None}},
                "p2": {"delta_+0h": {"value": 0.2, "passed": None}},
            }},
        }],
    }
    rechecked = recheck_report(report, _suite({"max": 1.0}))
    output = rechecked["results"][0]["output"]
    assert output["per_period"]["p1"]["delta_+0h"]["passed"] is True
    assert output["per_period"]["p2"]["delta_+0h"]["passed"] is True
    assert output["passed"] is True


def test_recheck_report_one_bad_delta_fails_its_label_and_the_check():
    # p1 has two deltas -- one passes, one fails -- must AND to False at both the label and the
    # top level, even though p2's single delta passes cleanly.
    report = {
        "results": [{
            "id": "c1",
            "output": {"per_period": {
                "p1": {"delta_-2h": {"value": 0.1, "passed": None}, "delta_+0h": {"value": 5.0, "passed": None}},
                "p2": {"delta_+0h": {"value": 0.2, "passed": None}},
            }},
        }],
    }
    rechecked = recheck_report(report, _suite({"max": 1.0}))
    output = rechecked["results"][0]["output"]
    assert output["per_period"]["p1"]["delta_-2h"]["passed"] is True
    assert output["per_period"]["p1"]["delta_+0h"]["passed"] is False
    assert output["per_period"]["p2"]["delta_+0h"]["passed"] is True
    assert output["passed"] is False


def test_recheck_report_missing_check_id_leaves_output_unchanged():
    report = {"results": [{"id": "unknown", "output": {"value": 5.0}}]}
    rechecked = recheck_report(report, _suite({"max": 1.0}))
    assert rechecked["results"][0]["output"] == {"value": 5.0}
