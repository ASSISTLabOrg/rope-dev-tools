"""Uses rope-registry's tests/fixtures/*.json as golden fixtures."""

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
    "invalid_envelope_missing_ic.json",
])
def test_invalid_manifest_fixtures_raise(validator, registry_root, fixture_name):
    manifest = _fixture(registry_root, fixture_name)
    with pytest.raises(ManifestValidationError):
        validator.validate_manifest(manifest)


def test_builder_build_and_validate_round_trips(validator, tmp_path):
    from rope_dev_tools.spec import ModelSpec

    spec = ModelSpec(
        kind="stacked_ensemble", name="n", version="v",
        source_dir=tmp_path, latent_dim=10,
        driver_columns=["f10", "kp"], driver_source="celestrak_sw",
        grid={"n_lst": 72, "n_lat": 36, "n_alt": 45,
              "lat_min_deg": -87.5, "lat_max_deg": 87.5,
              "alt_min_km": 100.0, "alt_max_km": 980.0},
        runtime_requirements={"onnxruntime": "1.25"},
    )
    kind_block = {
        "seq_len": 3,
        "base_models": [{"file": "a.onnx", "backend": "onnx", "architecture": "lstm", "inter_op_threads": 1}],
        "meta_model": {"file": "m.onnx", "backend": "onnx"},
        "decoder": {"kind": "coae", "params": {"decode_batch_size": 120,
            "stages": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}]}},
        "ic": {"kind": "ic_lookup_table", "params": {"grid_axes": ["f10", "kp"], "file": "ic.icbin"}},
    }

    builder = ManifestBuilder(validator)
    manifest = builder.build_and_validate(spec, kind_block)
    assert manifest["validated"] is False
    assert manifest["ic"] == {"kind": "ic_lookup_table", "params": {"grid_axes": ["f10", "kp"], "file": "ic.icbin"}}
    assert "ic" not in manifest["stacked_ensemble"]
    assert manifest["decoder"] == {"kind": "coae", "params": {"decode_batch_size": 120,
        "stages": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}]}}
    assert "decoder" not in manifest["stacked_ensemble"]
    assert manifest["drivers"]["source"] == "celestrak_sw"
    assert manifest["drivers"]["columns"] == [
        {"name": "f10", "description": "F10.7 cm solar radio flux, in solar flux units (SFU). "
                                        "Must come from the model's raw driver data source "
                                        "(CSV column, .swbin field, or explicit override)."},
        {"name": "kp", "description": "Kp planetary geomagnetic index (0-9 scale). "
                                       "Must come from the model's raw driver data source "
                                       "(CSV column, .swbin field, or explicit override)."},
    ]

    path = builder.write(manifest, tmp_path)
    assert path.is_file()
    assert json.loads(path.read_text()) == manifest


def test_upgrade_legacy_manifest_from_ic_grid_axes(validator, tmp_path):
    """Oldest shape: top-level ic_grid_axes, no ic block anywhere."""
    legacy = {
        "schema_version": 1, "kind": "stacked_ensemble",
        "runtime_requirements": {"onnxruntime": "1.25"},
        "latent_dim": 10, "driver_columns": ["f10", "kp"], "driver_source": "celestrak_sw",
        "grid": {"n_lst": 72, "n_lat": 36, "n_alt": 45,
                 "lat_min_deg": -87.5, "lat_max_deg": 87.5,
                 "alt_min_km": 100.0, "alt_max_km": 980.0},
        "ic_grid_axes": ["f10", "kp"],
        "stacked_ensemble": {
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
    assert "ic" not in upgraded["stacked_ensemble"]
    assert upgraded["validated"] is False
    assert upgraded["ic"] == {
        "kind": "ic_lookup_table",
        "params": {"grid_axes": ["f10", "kp"], "file": "ic_table.icbin"},
    }
    assert "decoder" not in upgraded["stacked_ensemble"]
    assert "decode_batch_size" not in upgraded["stacked_ensemble"]
    assert "decoders" not in upgraded["stacked_ensemble"]
    assert upgraded["decoder"] == {
        "kind": "coae",
        "params": {"decode_batch_size": 120,
            "stages": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}]},
    }
    assert "driver_columns" not in upgraded and "driver_source" not in upgraded
    assert upgraded["drivers"]["source"] == "celestrak_sw"
    assert [c["name"] for c in upgraded["drivers"]["columns"]] == ["f10", "kp"]
    validator.validate_manifest(upgraded)  # no raise


def test_upgrade_legacy_manifest_unknown_driver_name_raises(validator, tmp_path):
    legacy = {
        "schema_version": 1, "kind": "stacked_ensemble",
        "runtime_requirements": {"onnxruntime": "1.25"},
        "latent_dim": 10, "driver_columns": ["not_a_known_driver"], "driver_source": "celestrak_sw",
        "grid": {"n_lst": 72, "n_lat": 36, "n_alt": 45,
                 "lat_min_deg": -87.5, "lat_max_deg": 87.5,
                 "alt_min_km": 100.0, "alt_max_km": 980.0},
        "ic_grid_axes": ["f10", "kp"],
        "stacked_ensemble": {
            "seq_len": 3, "decode_batch_size": 120,
            "base_models": [{"file": "a.onnx", "backend": "onnx", "architecture": "lstm", "inter_op_threads": 1}],
            "meta_model": {"file": "m.onnx", "backend": "onnx"},
            "decoders": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}],
        },
    }
    (tmp_path / "ic_table.icbin").touch()

    builder = ManifestBuilder(validator)
    with pytest.raises(ValueError, match="not_a_known_driver"):
        builder.upgrade_legacy(legacy, exported_dir=tmp_path)


def test_builder_build_accepts_explicit_description_for_unknown_driver_name(validator, tmp_path):
    """A driver_columns entry can be a {'name','description'} dict to override/
    supply a description for a raw column not yet in driver_registry.json."""
    from rope_dev_tools.spec import ModelSpec

    spec = ModelSpec(
        kind="stacked_ensemble", name="n", version="v",
        source_dir=tmp_path, latent_dim=10,
        driver_columns=["f10", {"name": "ap", "description": "custom ap index."}],
        driver_source="celestrak_sw",
        grid={"n_lst": 72, "n_lat": 36, "n_alt": 45,
              "lat_min_deg": -87.5, "lat_max_deg": 87.5,
              "alt_min_km": 100.0, "alt_max_km": 980.0},
        runtime_requirements={"onnxruntime": "1.25"},
    )
    kind_block = {
        "seq_len": 3,
        "base_models": [{"file": "a.onnx", "backend": "onnx", "architecture": "lstm", "inter_op_threads": 1}],
        "meta_model": {"file": "m.onnx", "backend": "onnx"},
        "decoder": {"kind": "coae", "params": {"decode_batch_size": 120,
            "stages": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}]}},
        "ic": {"kind": "ic_lookup_table", "params": {"grid_axes": ["f10", "kp"], "file": "ic.icbin"}},
    }

    builder = ManifestBuilder(validator)
    manifest = builder.build_and_validate(spec, kind_block)
    assert manifest["drivers"]["columns"][1] == {"name": "ap", "description": "custom ap index."}


def test_builder_build_unknown_bare_driver_name_raises(validator, tmp_path):
    from rope_dev_tools.spec import ModelSpec

    spec = ModelSpec(
        kind="stacked_ensemble", name="n", version="v",
        source_dir=tmp_path, latent_dim=10,
        driver_columns=["not_a_known_driver"], driver_source="celestrak_sw",
        grid={"n_lst": 72, "n_lat": 36, "n_alt": 45,
              "lat_min_deg": -87.5, "lat_max_deg": 87.5,
              "alt_min_km": 100.0, "alt_max_km": 980.0},
        runtime_requirements={"onnxruntime": "1.25"},
    )
    kind_block = {
        "seq_len": 3,
        "base_models": [{"file": "a.onnx", "backend": "onnx", "architecture": "lstm", "inter_op_threads": 1}],
        "meta_model": {"file": "m.onnx", "backend": "onnx"},
        "decoder": {"kind": "coae", "params": {"decode_batch_size": 120,
            "stages": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}]}},
        "ic": {"kind": "ic_lookup_table", "params": {"grid_axes": ["f10", "kp"], "file": "ic.icbin"}},
    }

    builder = ManifestBuilder(validator)
    with pytest.raises(ValueError, match="not_a_known_driver"):
        builder.build(spec, kind_block)


def test_upgrade_legacy_manifest_from_nested_ic_block(validator, tmp_path):
    """Middle shape: nested kind_block.ic, no top-level ic or ic_grid_axes."""
    legacy = {
        "schema_version": 1, "kind": "stacked_ensemble",
        "runtime_requirements": {"onnxruntime": "1.25"},
        "latent_dim": 10, "driver_columns": ["f10", "kp"], "driver_source": "celestrak_sw",
        "grid": {"n_lst": 72, "n_lat": 36, "n_alt": 45,
                 "lat_min_deg": -87.5, "lat_max_deg": 87.5,
                 "alt_min_km": 100.0, "alt_max_km": 980.0},
        "stacked_ensemble": {
            "seq_len": 3, "decode_batch_size": 120,
            "base_models": [{"file": "a.onnx", "backend": "onnx", "architecture": "lstm", "inter_op_threads": 1}],
            "meta_model": {"file": "m.onnx", "backend": "onnx"},
            "decoders": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}],
            "ic": {"kind": "ic_lookup_table", "params": {"grid_axes": ["f10", "kp"], "file": "ic_table.icbin"}},
        },
    }

    builder = ManifestBuilder(validator)
    upgraded = builder.upgrade_legacy(legacy)

    assert "ic" not in upgraded["stacked_ensemble"]
    assert upgraded["validated"] is False
    assert upgraded["ic"] == {
        "kind": "ic_lookup_table",
        "params": {"grid_axes": ["f10", "kp"], "file": "ic_table.icbin"},
    }
    assert "decoder" not in upgraded["stacked_ensemble"]
    assert "decode_batch_size" not in upgraded["stacked_ensemble"]
    assert "decoders" not in upgraded["stacked_ensemble"]
    assert upgraded["decoder"] == {
        "kind": "coae",
        "params": {"decode_batch_size": 120,
            "stages": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}]},
    }
    validator.validate_manifest(upgraded)  # no raise


def test_upgrade_legacy_manifest_already_top_level_ic_is_noop(validator, tmp_path):
    """Current shape: top-level ic already present, nothing to migrate."""
    legacy = {
        "schema_version": 1, "kind": "stacked_ensemble",
        "runtime_requirements": {"onnxruntime": "1.25"},
        "latent_dim": 10, "driver_columns": ["f10", "kp"], "driver_source": "celestrak_sw",
        "grid": {"n_lst": 72, "n_lat": 36, "n_alt": 45,
                 "lat_min_deg": -87.5, "lat_max_deg": 87.5,
                 "alt_min_km": 100.0, "alt_max_km": 980.0},
        "ic": {"kind": "ic_lookup_table", "params": {"grid_axes": ["f10", "kp"], "file": "ic_table.icbin"}},
        "stacked_ensemble": {
            "seq_len": 3, "decode_batch_size": 120,
            "base_models": [{"file": "a.onnx", "backend": "onnx", "architecture": "lstm", "inter_op_threads": 1}],
            "meta_model": {"file": "m.onnx", "backend": "onnx"},
            "decoders": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}],
        },
    }

    builder = ManifestBuilder(validator)
    upgraded = builder.upgrade_legacy(legacy)

    assert "ic" not in upgraded["stacked_ensemble"]
    assert upgraded["validated"] is False
    assert upgraded["ic"] == {
        "kind": "ic_lookup_table",
        "params": {"grid_axes": ["f10", "kp"], "file": "ic_table.icbin"},
    }
    assert "decoder" not in upgraded["stacked_ensemble"]
    assert "decode_batch_size" not in upgraded["stacked_ensemble"]
    assert "decoders" not in upgraded["stacked_ensemble"]
    assert upgraded["decoder"] == {
        "kind": "coae",
        "params": {"decode_batch_size": 120,
            "stages": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}]},
    }
    validator.validate_manifest(upgraded)  # no raise


def test_set_validated_flips_flag_without_touching_artifacts(validator, tmp_path):
    manifest = {
        "schema_version": 1, "kind": "stacked_ensemble",
        "runtime_requirements": {"onnxruntime": "1.25"},
        "latent_dim": 10,
        "drivers": {
            "source": "celestrak_sw",
            "columns": [
                {"name": "f10", "description": "d"},
                {"name": "kp", "description": "d"},
            ],
        },
        "grid": {"n_lst": 72, "n_lat": 36, "n_alt": 45,
                 "lat_min_deg": -87.5, "lat_max_deg": 87.5,
                 "alt_min_km": 100.0, "alt_max_km": 980.0},
        "ic": {"kind": "ic_lookup_table", "params": {"grid_axes": ["f10", "kp"], "file": "ic.icbin"}},
        "decoder": {"kind": "coae", "params": {"decode_batch_size": 120,
            "stages": [{"backends": {"onnx": "d.onnx"}, "stats": "s.bin", "alt_start": 0, "alt_end": 45}]}},
        "validated": False,
        "stacked_ensemble": {
            "seq_len": 3,
            "base_models": [{"file": "a.onnx", "backend": "onnx", "architecture": "lstm", "inter_op_threads": 1}],
            "meta_model": {"file": "m.onnx", "backend": "onnx"},
        },
    }
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "a.onnx").write_bytes(b"not a real onnx file, just needs to exist")
    before_bytes = (tmp_path / "a.onnx").read_bytes()

    report = {
        "schema_version": 1, "suite_content_version": 1,
        "generated_at": "2026-01-01T00:00:00Z",
        "results": [{"id": "chk", "kind": "avg_density_vs_time",
                     "output": {"value": 1.0, "unit": "kg/m3", "passed": True}}],
    }

    builder = ManifestBuilder(validator)
    updated = builder.set_validated(tmp_path, report)

    assert updated["validated"] is True
    assert updated["validation"]["summary"] == {"chk": {"value": 1.0, "unit": "kg/m3", "passed": True}}
    assert (tmp_path / "a.onnx").read_bytes() == before_bytes  # untouched
