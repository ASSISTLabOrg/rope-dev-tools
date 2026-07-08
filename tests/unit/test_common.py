import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rope_dev_tools.export.common import (
    ConversionFidelityError,
    assert_conversion_matches,
    csv_to_icbin,
    export_torch_module,
    read_stats_bin,
    write_stats_bin,
)


def test_stats_bin_roundtrip_scalar(tmp_path):
    mu, sigma = np.array([1.0]), np.array([2.0])
    path = tmp_path / "s.bin"
    write_stats_bin(path, mu, sigma)
    mu2, sigma2 = read_stats_bin(path)
    np.testing.assert_allclose(mu2, mu)
    np.testing.assert_allclose(sigma2, sigma)


def test_stats_bin_roundtrip_spatial(tmp_path):
    rng = np.random.default_rng(0)
    mu = rng.standard_normal((1, 72, 36, 45)).astype(np.float32)
    sigma = np.abs(rng.standard_normal((1, 72, 36, 45)).astype(np.float32)) + 0.1
    path = tmp_path / "s.bin"
    write_stats_bin(path, mu, sigma)
    mu2, sigma2 = read_stats_bin(path)
    np.testing.assert_allclose(mu2, mu, rtol=1e-6)
    np.testing.assert_allclose(sigma2, sigma, rtol=1e-6)


def test_stats_bin_is_little_endian(tmp_path):
    # mu/sigma each have shape (2,) -- one dimension, of size 2 -- so ndim == 1.
    path = tmp_path / "s.bin"
    write_stats_bin(path, np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    with open(path, "rb") as f:
        header = f.read(8)  # ndim (uint32) + shape[0] (uint32)
    ndim, dim0 = struct.unpack("<II", header)
    assert (ndim, dim0) == (1, 2)
    # Confirm this is genuinely little-endian, not an accident of a
    # byte-symmetric value: big-endian interpretation must disagree.
    assert struct.unpack(">II", header) != (1, 2)


def test_csv_to_icbin_roundtrip(tmp_path):
    csv_path = tmp_path / "ic.csv"
    csv_path.write_text("F10,Kp,y1,y2\n100.0,1.0,0.1,0.2\n200.0,3.0,0.3,0.4\n")
    out_path = tmp_path / "ic.icbin"
    csv_to_icbin(csv_path, out_path, ["f10", "kp"])

    with open(out_path, "rb") as f:
        magic, version, nrows, k, reserved = struct.unpack("<5I", f.read(20))
        assert magic == 0x52504943
        assert version == 1
        assert nrows == 2
        assert k == 2
        records = np.frombuffer(f.read(), dtype="<f4").reshape(nrows, 2 + k)
    np.testing.assert_allclose(records[0], [100.0, 1.0, 0.1, 0.2])
    np.testing.assert_allclose(records[1], [200.0, 3.0, 0.3, 0.4])


def test_csv_to_icbin_rejects_non_f10_kp_axes(tmp_path):
    csv_path = tmp_path / "ic.csv"
    csv_path.write_text("F10,Kp,y1\n100.0,1.0,0.1\n")
    with pytest.raises(ValueError):
        csv_to_icbin(csv_path, tmp_path / "out.icbin", ["f10", "kp", "extra"])


class _TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 8)

    def forward(self, x):
        return self.fc(x)


def test_export_torch_module_and_conversion_check_passes(tmp_path):
    model = _TinyLinear().eval()
    dummy = torch.zeros(1, 4)
    written = export_torch_module(model, dummy, tmp_path, "tiny", backends=("onnx",))
    assert written == {"onnx": "tiny.onnx"}

    sample = np.random.default_rng(0).standard_normal((1, 4)).astype(np.float32)
    assert_conversion_matches(
        lambda x: model(torch.from_numpy(x)).detach().numpy(),
        tmp_path / "tiny.onnx", "onnx", sample,
    )  # no raise


def test_assert_conversion_matches_catches_a_real_mismatch(tmp_path):
    """Exports a model, then overwrites the exported ONNX file with a
    DIFFERENT (perturbed-weights) model, proving the fidelity check
    genuinely catches a conversion that doesn't reproduce the original."""
    model = _TinyLinear().eval()
    dummy = torch.zeros(1, 4)
    export_torch_module(model, dummy, tmp_path, "tiny", backends=("onnx",))

    broken_model = _TinyLinear().eval()
    with torch.no_grad():
        for p in broken_model.parameters():
            p.add_(1.0)
    export_torch_module(broken_model, dummy, tmp_path, "tiny", backends=("onnx",))  # overwrite

    sample = np.random.default_rng(0).standard_normal((1, 4)).astype(np.float32)
    with pytest.raises(ConversionFidelityError):
        assert_conversion_matches(
            lambda x: model(torch.from_numpy(x)).detach().numpy(),  # the ORIGINAL model
            tmp_path / "tiny.onnx", "onnx", sample,
        )
