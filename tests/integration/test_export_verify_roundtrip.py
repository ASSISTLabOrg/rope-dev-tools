"""Builds tiny REAL Keras/PyTorch trained artifacts as a synthetic
"dev-provided" source directory (not gen_fixtures.py's stubbed-output style,
which would skip the conversion code path entirely), runs export_model() end
to end, and confirms a schema-valid manifest + report + successful
mark_validated — proving the whole pipeline works together, not just each
piece in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
torch = pytest.importorskip("torch")

from rope_dev_tools import ModelSpec, export_model, mark_validated
from rope_dev_tools.grid import GRID_ALT, GRID_LAT, GRID_LST
from rope_dev_tools.validation.model_interfaces import WrapperRequest, WrapperResponse

SEQ_LEN = 3
LATENT_DIM = 4
DRIVER_COLUMNS = ["f10", "kp", "t1", "t2", "t3", "t4"]
FEATURE_DIM = LATENT_DIM + len(DRIVER_COLUMNS)


def _make_keras_model():
    inp = tf.keras.Input(shape=(SEQ_LEN, FEATURE_DIM))
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dense(LATENT_DIM)(x)
    return tf.keras.Model(inp, x)


class _TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(LATENT_DIM, GRID_LST * GRID_LAT * GRID_ALT)

    def forward(self, x):
        return self.fc(x).view(x.shape[0], 1, GRID_LST, GRID_LAT, GRID_ALT)


def _load_decoder(path):
    m = _TinyDecoder()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


@pytest.fixture
def source_dir(tmp_path):
    src = tmp_path / "source"
    src.mkdir()

    _make_keras_model().save(src / "base_00.keras")
    _make_keras_model().save(src / "meta.keras")

    decoder = _TinyDecoder().eval()
    torch.save(decoder.state_dict(), src / "decoder.pt")

    header = "F10,Kp," + ",".join(f"y{i + 1}" for i in range(LATENT_DIM))
    rows = [
        f"{f10},{kp}," + ",".join("0.0" for _ in range(LATENT_DIM))
        for f10 in (100.0, 200.0) for kp in (1.0, 3.0)
    ]
    (src / "ic_table.csv").write_text(header + "\n" + "\n".join(rows) + "\n")

    return src


@pytest.fixture
def spec(source_dir):
    return ModelSpec(
        kind="ensemble_fusion_decoder",
        name="integration-test-model", version="v0",
        source_dir=source_dir,
        latent_dim=LATENT_DIM,
        driver_columns=DRIVER_COLUMNS,
        driver_source="celestrak_sw",
        runtime_requirements={"onnxruntime": "1.25"},
        kind_params={
            "seq_len": SEQ_LEN,
            "decode_batch_size": 120,
            "base_models": [{"source": "base_00.keras", "architecture": "lstm", "inter_op_threads": 1}],
            "meta_model": {"source": "meta.keras"},
            "decoders": [{
                "source": "decoder.pt", "load_decoder": _load_decoder,
                "stats": (np.zeros(1), np.ones(1)),
                "alt_start": 0, "alt_end": GRID_ALT, "backends": ("onnx",),
            }],
            "stats_ts": (np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM)),
            "ic_csv_path": "ic_table.csv",
            "ic_grid_axes": ["f10", "kp"],
        },
    )


def _wrapper_fn(req: WrapperRequest) -> WrapperResponse:
    rng = np.random.default_rng(0)
    times = [req.start, req.end]
    density = np.abs(rng.standard_normal((len(times), GRID_LST, GRID_LAT, GRID_ALT))) * 1e-12
    uncertainty = np.abs(rng.standard_normal((len(times), GRID_LST, GRID_LAT, GRID_ALT))) * 1e-13
    return WrapperResponse(times=times, density=density, uncertainty=uncertainty)


def test_export_produces_schema_valid_manifest_and_artifacts(spec, tmp_path):
    out_dir = tmp_path / "export"
    result = export_model(spec, out_dir, skip_validation=True)

    assert result.manifest_path.is_file()
    assert result.manifest["validated"] is False
    for name in ("base_model_00.onnx", "meta_model.onnx", "coae_decoder.onnx",
                 "stats_ts.bin", "stats_cae.bin", "ic_table.icbin", "model_manifest.json"):
        assert (out_dir / name).is_file(), name


def test_export_with_wrapper_verification_then_mark_validated(spec, tmp_path):
    out_dir = tmp_path / "export"
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    (suite_dir / "truth.csv").write_text(
        "datetime,lst,lat,alt_km,density\n2024-01-01 01:00:00,12.0,0.0,400.0,1.0e-12\n"
    )
    suite_path = suite_dir / "suite.json"
    suite_path.write_text(
        '{"schema_version": 1, "content_version": 1, '
        '"checks": [{"id": "check_rmse", "kind": "rmse_timeseries", '
        '"start": "2024-01-01 00:00:00", "end": "2024-01-01 03:00:00", '
        '"truth_csv": "truth.csv", "unit": "kg/m3"}]}'
    )

    result = export_model(spec, out_dir, suite=suite_path, wrapper=f"{__file__}:_wrapper_fn")

    assert result.report is not None
    assert result.report_path.is_file()

    manifest = mark_validated(out_dir)
    assert manifest["validated"] is True
    assert manifest["validation"]["report_file"] == "validation_report.json"
