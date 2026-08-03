"""ValidationSuite -- suite-wide label fields (physics_model_label/rope_model_label/satellite_label)."""

from __future__ import annotations

from rope_dev_tools.validation.schema_types import ValidationSuite


def test_from_dict_reads_labels_when_present():
    suite = ValidationSuite.from_dict({
        "schema_version": 1, "content_version": 2, "checks": [],
        "physics_model_label": "WAM", "rope_model_label": "ROPE-WAM-V1", "satellite_label": "GRACE",
    })
    assert suite.physics_model_label == "WAM"
    assert suite.rope_model_label == "ROPE-WAM-V1"
    assert suite.satellite_label == "GRACE"


def test_from_dict_labels_default_to_none_when_absent():
    suite = ValidationSuite.from_dict({"schema_version": 1, "content_version": 1, "checks": []})
    assert suite.physics_model_label is None
    assert suite.rope_model_label is None
    assert suite.satellite_label is None


def test_to_dict_omits_labels_when_none():
    suite = ValidationSuite(1, 1, [])
    d = suite.to_dict()
    assert "physics_model_label" not in d
    assert "rope_model_label" not in d
    assert "satellite_label" not in d


def test_to_dict_roundtrips_labels():
    suite = ValidationSuite(1, 1, [], physics_model_label="WAM", rope_model_label="ROPE", satellite_label="GRACE")
    d = suite.to_dict()
    assert d["physics_model_label"] == "WAM"
    assert d["rope_model_label"] == "ROPE"
    assert d["satellite_label"] == "GRACE"
    restored = ValidationSuite.from_dict(d)
    assert restored.physics_model_label == "WAM"
