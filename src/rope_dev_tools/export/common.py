"""Generic, kind-agnostic conversion primitives, reused by any model kind."""

from __future__ import annotations

import csv
import struct
from pathlib import Path
from typing import Callable

import numpy as np

_STATS_MAGIC_NDIM_FMT = "<I"
_STATS_DIM_FMT = "<I"
_ICBIN_MAGIC = 0x52504943  # "RPIC"
_ICBIN_VERSION = 1
_ICBIN_HEADER_FMT = "<5I"  # magic, version, nrows, latent_dim, reserved


class ConversionFidelityError(RuntimeError):
    """Raised when an exported artifact doesn't reproduce its source model's output."""


# ---------------------------------------------------------------------------
# Keras -> ONNX
# ---------------------------------------------------------------------------

def keras_to_onnx(model, input_shape: tuple, opset: int = 17):
    """Converts a Keras model to ONNX via tf2onnx.convert.from_function. Folds tf.Tensor layer attributes to numpy first to avoid spurious ONNX inputs."""
    import tensorflow as tf
    import tf2onnx

    _ = model(tf.zeros((1,) + tuple(input_shape), dtype=tf.float32), training=False)

    for layer in model.layers:
        for attr_name, attr_val in list(vars(layer).items()):
            if isinstance(attr_val, tf.Tensor):
                try:
                    setattr(layer, attr_name, attr_val.numpy())
                except AttributeError:
                    pass

    input_spec = tf.TensorSpec((None,) + tuple(input_shape), tf.float32, name="x")

    @tf.function(input_signature=[input_spec])
    def serving_fn(x):
        return model(x, training=False)

    onnx_model, _ = tf2onnx.convert.from_function(
        serving_fn,
        input_signature=[input_spec],
        opset=opset,
    )
    return onnx_model


# ---------------------------------------------------------------------------
# Generic TorchScript + ONNX dual export
# ---------------------------------------------------------------------------

def export_torch_module(
    model,
    dummy_input,
    out_dir: Path,
    stem: str,
    *,
    backends: tuple = ("onnx", "libtorch"),
    opset: int = 17,
) -> dict:
    """Dual TorchScript + ONNX export of any torch.nn.Module. Returns {"onnx": filename, "libtorch": filename}."""
    import torch

    model = model.eval()
    written: dict = {}

    if "libtorch" in backends:
        pt_name = f"{stem}.pt"
        with torch.no_grad():
            traced = torch.jit.trace(model, dummy_input)
        torch.jit.save(traced, str(out_dir / pt_name))
        written["libtorch"] = pt_name

    if "onnx" in backends:
        onnx_name = f"{stem}.onnx"
        onnx_path = out_dir / onnx_name
        torch.onnx.export(
            model, dummy_input, str(onnx_path),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=opset,
        )
        _patch_onnx_batch_dim(onnx_path)
        written["onnx"] = onnx_name

    return written


def _patch_onnx_batch_dim(onnx_path: Path) -> None:
    """Marks the output batch dimension as symbolic — torch.onnx.export can bake in a fixed batch size otherwise."""
    try:
        import onnx as onnx_pkg
    except ImportError:
        return
    m = onnx_pkg.load(str(onnx_path))
    for out in m.graph.output:
        dim = out.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "batch"
    onnx_pkg.save(m, str(onnx_path))


# ---------------------------------------------------------------------------
# Binary stats format (uint32 ndim, uint32 shape[ndim], f32 mu[], f32 sigma[])
# ---------------------------------------------------------------------------

def write_stats_bin(path: Path, mu: np.ndarray, sigma: np.ndarray) -> None:
    mu = np.asarray(mu, dtype="<f4")
    sigma = np.asarray(sigma, dtype="<f4")
    if mu.shape != sigma.shape:
        raise ValueError(f"mu shape {mu.shape} != sigma shape {sigma.shape}")
    shape = list(mu.shape)
    with open(path, "wb") as f:
        f.write(struct.pack(_STATS_MAGIC_NDIM_FMT, len(shape)))
        for s in shape:
            f.write(struct.pack(_STATS_DIM_FMT, s))
        f.write(mu.ravel().tobytes())
        f.write(sigma.ravel().tobytes())


def read_stats_bin(path: Path) -> tuple:
    with open(path, "rb") as f:
        (ndim,) = struct.unpack(_STATS_MAGIC_NDIM_FMT, f.read(4))
        shape = [struct.unpack(_STATS_DIM_FMT, f.read(4))[0] for _ in range(ndim)]
        n = int(np.prod(shape)) if shape else 1
        mu = np.frombuffer(f.read(4 * n), dtype="<f4").reshape(shape).copy()
        sigma = np.frombuffer(f.read(4 * n), dtype="<f4").reshape(shape).copy()
    return mu, sigma


# ---------------------------------------------------------------------------
# IC lookup table CSV -> .icbin
# ---------------------------------------------------------------------------

def csv_to_icbin(csv_path: Path, out_path: Path, grid_axes: list) -> None:
    """Converts an IC lookup-table CSV (F10, Kp, y1..yK) to the .icbin binary format. Only supports the two-axis (F10, Kp) grid."""
    if [a.lower() for a in grid_axes] != ["f10", "kp"]:
        raise ValueError(
            f"ic_table.icbin only supports grid_axes ['f10', 'kp'], got {grid_axes!r}"
        )

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path}: no data rows")

    y_cols = sorted(
        (c for c in rows[0] if c.lower().startswith("y") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    if not y_cols:
        raise ValueError(f"{csv_path}: no y1..yK latent columns found")
    k = len(y_cols)

    records = np.zeros((len(rows), 2 + k), dtype="<f4")
    for i, row in enumerate(rows):
        records[i, 0] = float(row["F10"])
        records[i, 1] = float(row["Kp"])
        for j, col in enumerate(y_cols):
            records[i, 2 + j] = float(row[col])

    with open(out_path, "wb") as f:
        f.write(struct.pack(_ICBIN_HEADER_FMT, _ICBIN_MAGIC, _ICBIN_VERSION, len(rows), k, 0))
        f.write(records.tobytes())


# ---------------------------------------------------------------------------
# Loading an exported artifact back, for the conversion-fidelity check
# ---------------------------------------------------------------------------

def run_onnx(path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    (output,) = sess.run(None, {input_name: x.astype(np.float32)})
    return output


def run_torchscript(path: Path, x: np.ndarray) -> np.ndarray:
    import torch

    model = torch.jit.load(str(path))
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x.astype(np.float32)))
    return out.detach().cpu().numpy()


def assert_conversion_matches(
    original_fn: Callable,
    exported_path: Path,
    backend: str,
    sample_input: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    label: str = "",
) -> None:
    """Compares the original model's output against the exported artifact loaded back; raises on mismatch."""
    label = label or exported_path.name
    original_output = np.asarray(original_fn(sample_input))

    if backend == "onnx":
        exported_output = run_onnx(exported_path, sample_input)
    elif backend == "libtorch":
        exported_output = run_torchscript(exported_path, sample_input)
    else:
        raise ValueError(f"{label}: unknown backend {backend!r}")

    exported_output = np.asarray(exported_output)

    if original_output.shape != exported_output.shape:
        raise ConversionFidelityError(
            f"{label}: exported {backend} artifact output shape {exported_output.shape} "
            f"does not match the original model's output shape {original_output.shape}"
        )

    if not np.allclose(original_output, exported_output, rtol=rtol, atol=atol):
        max_abs_diff = float(np.max(np.abs(original_output - exported_output)))
        raise ConversionFidelityError(
            f"{label}: exported {backend} artifact does not reproduce the original "
            f"model's output (max abs diff {max_abs_diff:.3g}, rtol={rtol}, atol={atol})"
        )
