"""generate_validation_plots.py regenerates plots from saved validation_data/, no ModelInterface involved."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
torch = pytest.importorskip("torch")

from rope_dev_tools import ModelSpec, export_model
from rope_dev_tools.validation.model_interfaces import WrapperRequest, WrapperResponse

SEQ_LEN = 3
LATENT_DIM = 4
DRIVER_COLUMNS = ["f10", "kp", "t1", "t2", "t3", "t4"]
FEATURE_DIM = LATENT_DIM + len(DRIVER_COLUMNS)
GRID_LST, GRID_LAT, GRID_ALT = 8, 6, 5

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_validation_plots.py"


def _load_script_main():
    spec = importlib.util.spec_from_file_location("generate_validation_plots", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


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


def _wrapper_fn(req: WrapperRequest) -> WrapperResponse:
    rng = np.random.default_rng(0)
    times = [req.start, req.end]
    density = np.abs(rng.standard_normal((len(times), GRID_LST, GRID_LAT, GRID_ALT))) * 1e-12
    uncertainty = np.abs(rng.standard_normal((len(times), GRID_LST, GRID_LAT, GRID_ALT))) * 1e-13
    return WrapperResponse(times=times, density=density, uncertainty=uncertainty)


def test_generate_validation_plots_regenerates_deleted_plots(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _make_keras_model().save(source_dir / "base_00.keras")
    _make_keras_model().save(source_dir / "meta.keras")
    decoder = _TinyDecoder().eval()
    torch.save(decoder.state_dict(), source_dir / "decoder.pt")
    header = "F10,Kp," + ",".join(f"y{i + 1}" for i in range(LATENT_DIM))
    rows = [f"{f10},{kp}," + ",".join("0.0" for _ in range(LATENT_DIM))
            for f10 in (100.0, 200.0) for kp in (1.0, 3.0)]
    (source_dir / "ic_table.csv").write_text(header + "\n" + "\n".join(rows) + "\n")

    spec = ModelSpec(
        kind="stacked_ensemble", name="regen-test-model", version="v0", source_dir=source_dir,
        latent_dim=LATENT_DIM, driver_columns=DRIVER_COLUMNS, driver_source="celestrak_sw",
        grid={"n_lst": GRID_LST, "n_lat": GRID_LAT, "n_alt": GRID_ALT,
              "lat_min_deg": -87.5, "lat_max_deg": 87.5, "alt_min_km": 100.0, "alt_max_km": 980.0},
        runtime_requirements={"onnxruntime": "1.25"},
        kind_params={
            "seq_len": SEQ_LEN, "decode_batch_size": 120,
            "base_models": [{"source": "base_00.keras", "architecture": "lstm", "inter_op_threads": 1}],
            "meta_model": {"source": "meta.keras"},
            "decoders": [{"source": "decoder.pt", "load_decoder": _load_decoder,
                          "stats": (np.zeros(1), np.ones(1)), "alt_start": 0, "alt_end": GRID_ALT,
                          "backends": ("onnx",)}],
            "stats_ts": (np.zeros(FEATURE_DIM), np.ones(FEATURE_DIM)),
            "ic_csv_path": "ic_table.csv", "ic_grid_axes": ["f10", "kp"],
        },
    )

    out_dir = tmp_path / "export"
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    (suite_dir / "truth.csv").write_text("datetime,alt_km,density\n2024-01-01 01:00:00,400.0,1.0e-12\n")
    suite_path = suite_dir / "suite.json"
    suite_path.write_text(
        '{"schema_version": 1, "content_version": 1, '
        '"checks": [{"id": "check_avg_density", "kind": "avg_density_vs_time", '
        '"periods": [{"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 03:00:00", '
        '"physics_avg_csv": "truth.csv"}], "altitudes_km": [400.0], "unit": "kg/m3"}]}'
    )

    result = export_model(spec, out_dir, suite=suite_path, wrapper=f"{__file__}:_wrapper_fn")
    plot_paths = [out_dir / p for p in result.report["results"][0]["output"]["plots"]]
    assert all(p.is_file() for p in plot_paths)

    shutil.rmtree(out_dir / "plots")
    assert not any(p.is_file() for p in plot_paths)

    main = _load_script_main()
    exit_code = main(["--exported-dir", str(out_dir)])

    assert exit_code == 0
    assert all(p.is_file() for p in plot_paths)
