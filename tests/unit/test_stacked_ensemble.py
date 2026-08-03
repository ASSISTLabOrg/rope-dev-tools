"""StackedEnsembleExporter._export_decoders() -- per-stage rtol/atol override."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rope_dev_tools.export.kinds import stacked_ensemble
from rope_dev_tools.export.kinds.stacked_ensemble import StackedEnsembleExporter
from rope_dev_tools.spec import ModelSpec


class _TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 8)

    def forward(self, x):
        return self.fc(x)


def _spec(tmp_path, decoder_stage_extra):
    torch.save(_TinyDecoder().eval().state_dict(), tmp_path / "decoder.pt")
    return ModelSpec(
        kind="stacked_ensemble", name="tol-test", version="v0", source_dir=tmp_path,
        latent_dim=4, driver_columns=["f10"], driver_source="celestrak_sw",
        grid={"n_lst": 2, "n_lat": 2, "n_alt": 2,
              "lat_min_deg": -1.0, "lat_max_deg": 1.0, "alt_min_km": 0.0, "alt_max_km": 1.0},
        kind_params={
            "seq_len": 1, "decode_batch_size": 1, "base_models": [], "meta_model": {},
            "load_decoder": lambda path: _TinyDecoder(),
            "decoders": [{
                "source": "decoder.pt", "stats": (np.zeros(1), np.ones(1)),
                "alt_start": 0, "alt_end": 2,
                "backends": ("onnx",), **decoder_stage_extra,
            }],
            "stats_ts": (np.zeros(1), np.ones(1)), "ic_csv_path": "ic.csv",
        },
    )


def test_export_decoders_forwards_stage_rtol_atol_override(tmp_path, monkeypatch):
    captured = {}

    def fake_assert_conversion_matches(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(stacked_ensemble, "assert_conversion_matches", fake_assert_conversion_matches)

    spec = _spec(tmp_path, {"rtol": 1e-2, "atol": 1e-3})
    StackedEnsembleExporter()._export_decoders(spec, tmp_path)

    assert captured["rtol"] == 1e-2
    assert captured["atol"] == 1e-3


def test_export_decoders_omits_tol_kwargs_when_not_overridden(tmp_path, monkeypatch):
    captured = {}

    def fake_assert_conversion_matches(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(stacked_ensemble, "assert_conversion_matches", fake_assert_conversion_matches)

    spec = _spec(tmp_path, {})
    StackedEnsembleExporter()._export_decoders(spec, tmp_path)

    assert "rtol" not in captured
    assert "atol" not in captured
