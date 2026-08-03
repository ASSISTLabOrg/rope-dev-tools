import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rope_dev_tools.export.common import (
    ConversionFidelityError,
    _replace_erfc_with_erf,
    assert_conversion_matches,
    csv_to_icbin,
    export_torch_module,
    read_stats_bin,
    write_stats_bin,
)


def test_replace_erfc_with_erf_matches_reference_values():
    """Erfc has never been a valid ONNX op -- tf2onnx still emits one for some Keras GELU
    activations. This checks the graph-surgery replacement computes the same values as erfc."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    import math
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])
    node = helper.make_node("Erfc", ["x"], ["y"], name="erfc_node")
    graph = helper.make_graph([node], "erfc_test", [x], [y])
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    _replace_erfc_with_erf(onnx_model)

    assert all(n.op_type != "Erfc" for n in onnx_model.graph.node)

    sess = ort.InferenceSession(onnx_model.SerializeToString())
    inputs = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    (result,) = sess.run(None, {"x": inputs})
    expected = np.array([math.erfc(v) for v in inputs], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


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
    # mu/sigma shape (2,) -> ndim == 1
    path = tmp_path / "s.bin"
    write_stats_bin(path, np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    with open(path, "rb") as f:
        header = f.read(8)  # ndim (uint32) + shape[0] (uint32)
    ndim, dim0 = struct.unpack("<II", header)
    assert (ndim, dim0) == (1, 2)
    # big-endian interpretation must disagree
    assert struct.unpack(">II", header) != (1, 2)


def _read_icbin(path):
    with open(path, "rb") as f:
        magic, version, nrows, k = struct.unpack("<4I", f.read(16))
        axis_names = []
        for _ in range(2):
            (name_len,) = struct.unpack("<I", f.read(4))
            axis_names.append(f.read(name_len).decode("utf-8"))
        records = np.frombuffer(f.read(), dtype="<f4").reshape(nrows, 2 + k)
    return magic, version, nrows, k, axis_names, records


def test_csv_to_icbin_roundtrip(tmp_path):
    csv_path = tmp_path / "ic.csv"
    csv_path.write_text("F10,Kp,y1,y2\n100.0,1.0,0.1,0.2\n200.0,3.0,0.3,0.4\n")
    out_path = tmp_path / "ic.icbin"
    csv_to_icbin(csv_path, out_path, ["F10", "Kp"])

    magic, version, nrows, k, axis_names, records = _read_icbin(out_path)
    assert magic == 0x52504943
    assert version == 2
    assert nrows == 2
    assert k == 2
    assert axis_names == ["F10", "Kp"]
    np.testing.assert_allclose(records[0], [100.0, 1.0, 0.1, 0.2])
    np.testing.assert_allclose(records[1], [200.0, 3.0, 0.3, 0.4])


def test_csv_to_icbin_supports_arbitrary_axis_names(tmp_path):
    """The actual fix: not hardcoded to F10/Kp -- any 2 CSV columns work as grid_axes."""
    csv_path = tmp_path / "ic.csv"
    csv_path.write_text("F10,Ap,y1\n66.0,0.0,0.5\n77.0,6.5,0.7\n")
    out_path = tmp_path / "ic.icbin"
    csv_to_icbin(csv_path, out_path, ["F10", "Ap"])

    magic, version, nrows, k, axis_names, records = _read_icbin(out_path)
    assert (magic, version, nrows, k) == (0x52504943, 2, 2, 1)
    assert axis_names == ["F10", "Ap"]
    np.testing.assert_allclose(records[0], [66.0, 0.0, 0.5])
    np.testing.assert_allclose(records[1], [77.0, 6.5, 0.7])


def test_csv_to_icbin_requires_exactly_two_axes(tmp_path):
    csv_path = tmp_path / "ic.csv"
    csv_path.write_text("F10,Kp,y1\n100.0,1.0,0.1\n")
    with pytest.raises(ValueError):
        csv_to_icbin(csv_path, tmp_path / "out.icbin", ["F10", "Kp", "extra"])


def test_csv_to_icbin_missing_axis_column_raises(tmp_path):
    csv_path = tmp_path / "ic.csv"
    csv_path.write_text("F10,Kp,y1\n100.0,1.0,0.1\n")
    with pytest.raises(ValueError, match="Ap"):
        csv_to_icbin(csv_path, tmp_path / "out.icbin", ["F10", "Ap"])


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
    """Exports a model, then overwrites the exported ONNX with a different model, and expects a mismatch."""
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
