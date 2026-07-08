"""Reuses rope-registry's own golden fixtures rather than inventing a
parallel set: rope-registry's tests/fixtures/*.json are the authoritative
examples of what does/doesn't validate against the shared schema.
"""

import json

import pytest

from rope_dev_tools.manifest import ManifestBuilder
from rope_dev_tools.registry.validate import ManifestValidationError


def _fixture(registry_root, name):
    return json.loads((registry_root / "tests" / "fixtures" / name).read_text())


def test_valid_manifest_passes(validator, registry_root):
    validator.validate_manifest(_fixture(registry_root, "valid_manifest.json"))


def test_valid_manifest_with_validation_passes(validator, registry_root):
    validator.validate_manifest(_fixture(registry_root, "valid_manifest_with_validation.json"))


@pytest.mark.parametrize("fixture_name", [
    "invalid_envelope_missing_field.json",
    "invalid_envelope_missing_validated.json",
    "invalid_envelope_validated_true_no_validation.json",
    "invalid_envelope_validation_bad_type.json",
    "invalid_envelope_wrong_type.json",
    "invalid_ic_bad_kind.json",
    "invalid_ic_missing_grid_axes.json",
    "invalid_kind_bad_enum.json",
    "invalid_kind_missing_field.json",
    "invalid_kind_missing_ic.json",
])
def test_invalid_manifest_fixtures_raise(validator, registry_root, fixture_name):
    manifest = _fixture(registry_root, fixture_name)
    with pytest.raises(ManifestValidationError):
        validator.validate_manifest(manifest)


def test_builder_build_and_validate_round_trips(validator, tmp_path):
    from rope_dev_tools.spec import ModelSpec

    spec = ModelSpec(
        kind="ensemble_fusion_decoder", name="n", version="v",
        source_dir=tmp_path, latent_dim=10,
        driver_columns=["f10", "kp"], driver_source="celestrak_sw",
        runtime_requirements={"onnxruntime": "1.25"},
    )
    kind_block = {
        "seq_len": 3, "decode_batch_size": 120,
        "base_models": [{"file": "a.onnx", "backend": "onnx", "architecture": "lstm", "inter_op_threads": 1}],
        "meta_model": {"file": "m.onnx", "backend": "onnx"},
        "decoders": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}],
        "ic": {"kind": "ic_lookup_table", "params": {"grid_axes": ["f10", "kp"], "file": "ic.icbin"}},
    }

    builder = ManifestBuilder(validator)
    manifest = builder.build_and_validate(spec, kind_block)
    assert manifest["validated"] is False

    path = builder.write(manifest, tmp_path)
    assert path.is_file()
    assert json.loads(path.read_text()) == manifest


def test_upgrade_legacy_manifest(validator, tmp_path):
    legacy = {
        "schema_version": 1, "kind": "ensemble_fusion_decoder",
        "runtime_requirements": {"onnxruntime": "1.25"},
        "latent_dim": 10, "driver_columns": ["f10", "kp"], "driver_source": "celestrak_sw",
        "ic_grid_axes": ["f10", "kp"],
        "ensemble_fusion_decoder": {
            "seq_len": 3, "decode_batch_size": 120,
            "base_models": [{"file": "a.onnx", "backend": "onnx", "architecture": "lstm", "inter_op_threads": 1}],
            "meta_model": {"file": "m.onnx", "backend": "onnx"},
            "decoders": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}],
        },
    }
    (tmp_path / "ic_table.icbin").touch()

    builder = ManifestBuilder(validator)
    upgraded = builder.upgrade_legacy(legacy, exported_dir=tmp_path)

    assert "ic_grid_axes" not in upgraded
    assert upgraded["validated"] is False
    assert upgraded["ensemble_fusion_decoder"]["ic"] == {
        "kind": "ic_lookup_table",
        "params": {"grid_axes": ["f10", "kp"], "file": "ic_table.icbin"},
    }
    validator.validate_manifest(upgraded)  # no raise


def test_set_validated_flips_flag_without_touching_artifacts(validator, tmp_path):
    manifest = {
        "schema_version": 1, "kind": "ensemble_fusion_decoder",
        "runtime_requirements": {"onnxruntime": "1.25"},
        "latent_dim": 10, "driver_columns": ["f10", "kp"], "driver_source": "celestrak_sw",
        "validated": False,
        "ensemble_fusion_decoder": {
            "seq_len": 3, "decode_batch_size": 120,
            "base_models": [{"file": "a.onnx", "backend": "onnx", "architecture": "lstm", "inter_op_threads": 1}],
            "meta_model": {"file": "m.onnx", "backend": "onnx"},
            "decoders": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}],
            "ic": {"kind": "ic_lookup_table", "params": {"grid_axes": ["f10", "kp"], "file": "ic.icbin"}},
        },
    }
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "a.onnx").write_bytes(b"not a real onnx file, just needs to exist")
    before_bytes = (tmp_path / "a.onnx").read_bytes()

    report = {
        "schema_version": 1, "suite_content_version": 1,
        "generated_at": "2026-01-01T00:00:00Z",
        "results": [{"id": "chk", "kind": "rmse_timeseries",
                     "output": {"value": 1.0, "unit": "kg/m3", "passed": True}}],
    }

    builder = ManifestBuilder(validator)
    updated = builder.set_validated(tmp_path, report)

    assert updated["validated"] is True
    assert updated["validation"]["summary"] == {"chk": {"value": 1.0, "unit": "kg/m3", "passed": True}}
    assert (tmp_path / "a.onnx").read_bytes() == before_bytes  # untouched
