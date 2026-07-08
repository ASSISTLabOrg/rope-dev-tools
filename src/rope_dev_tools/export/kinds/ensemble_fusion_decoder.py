"""EnsembleFusionDecoderExporter — thin orchestration for the one stable
model kind today (`ensemble_fusion_decoder`).

Knows this kind's manifest shape (base_models/meta_model/decoders/ic) and
enforces the one genuinely kind-specific rule (decoder alt_start/alt_end
stages must tile [0, GRID_ALT) with no gaps); everything else is delegated
to export/common.py's generic, kind-agnostic conversion primitives.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from rope_dev_tools.export.base import ModelExporter, SpecValidationError, register_exporter
from rope_dev_tools.export.common import (
    assert_conversion_matches,
    csv_to_icbin,
    export_torch_module,
    keras_to_onnx,
    write_stats_bin,
)
from rope_dev_tools.grid import GRID_ALT
from rope_dev_tools.spec import ModelSpec

_REQUIRED_KIND_PARAMS = (
    "seq_len", "decode_batch_size", "base_models", "meta_model",
    "decoders", "stats_ts", "ic_csv_path",
)


def _sample_input(shape: tuple, label: str) -> np.ndarray:
    """A deterministic, seeded-random array for the conversion-fidelity
    check — reproducible across runs (unlike Python's str hash) since it's
    derived from a CRC32 of the artifact label, not object identity."""
    seed = zlib.crc32(label.encode("utf-8"))
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _load_mu_sigma(source) -> tuple:
    """Accepts a (mu, sigma) tuple, a {"mu"/"mean"/"means", "sigma"/"std"/"stds"}
    dict, or a path to a torch .pt file containing such a dict."""
    if isinstance(source, tuple) and len(source) == 2:
        return np.asarray(source[0]), np.asarray(source[1])

    if isinstance(source, dict):
        stats = source
    else:
        import torch

        path = Path(source)
        try:
            stats = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            stats = torch.load(path, map_location="cpu")

    mu_key = next(k for k in ("mu", "mean", "means") if k in stats)
    sigma_key = next(k for k in ("sigma", "std", "stds") if k in stats)
    mu, sigma = stats[mu_key], stats[sigma_key]
    mu = mu.detach().cpu().numpy() if hasattr(mu, "detach") else np.asarray(mu)
    sigma = sigma.detach().cpu().numpy() if hasattr(sigma, "detach") else np.asarray(sigma)
    return mu, sigma


def _default_load_keras(path: Path, custom_objects: dict):
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects or None)


def _resolve(spec: ModelSpec, path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else spec.source_dir / path


def _resolve_stats_source(spec: ModelSpec, source):
    """stats sources may be a (mu, sigma) tuple, a dict, or a path relative
    to spec.source_dir -- only path-like values need resolving."""
    if isinstance(source, (str, Path)):
        return _resolve(spec, source)
    return source


@register_exporter
class EnsembleFusionDecoderExporter(ModelExporter):
    kind = "ensemble_fusion_decoder"

    def validate_spec(self, spec: ModelSpec) -> None:
        kp = spec.kind_params
        errors = []

        for key in _REQUIRED_KIND_PARAMS:
            if key not in kp:
                errors.append(f"kind_params missing required key {key!r}")

        if kp.get("base_models") == []:
            errors.append("kind_params['base_models'] must not be empty")
        if kp.get("decoders") == []:
            errors.append("kind_params['decoders'] must not be empty")
        for i, stage in enumerate(kp.get("decoders", [])):
            if "load_decoder" not in stage and "load_decoder" not in kp:
                errors.append(
                    f"kind_params['decoders'][{i}] has no 'load_decoder', and "
                    f"kind_params has no top-level default 'load_decoder' either"
                )

        if errors:
            raise SpecValidationError(errors)

    def export(self, spec: ModelSpec, out_dir: Path) -> dict:
        self.validate_spec(spec)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        feature_dim = spec.latent_dim + len(spec.driver_columns)
        seq_len = spec.kind_params["seq_len"]

        self._export_stats_ts(spec, out_dir)
        base_models = self._export_base_models(spec, out_dir, seq_len, feature_dim)
        meta_model = self._export_meta_model(spec, out_dir, seq_len, feature_dim)
        decoders = self._export_decoders(spec, out_dir)
        ic = self._export_ic(spec, out_dir)

        return {
            "seq_len": seq_len,
            "decode_batch_size": spec.kind_params["decode_batch_size"],
            "base_models": base_models,
            "meta_model": meta_model,
            "decoders": decoders,
            "ic": ic,
        }

    # -- stages -----------------------------------------------------

    def _export_stats_ts(self, spec: ModelSpec, out_dir: Path) -> None:
        mu, sigma = _load_mu_sigma(_resolve_stats_source(spec, spec.kind_params["stats_ts"]))
        write_stats_bin(out_dir / "stats_ts.bin", mu, sigma)

    def _export_base_models(self, spec: ModelSpec, out_dir: Path, seq_len: int, feature_dim: int) -> list:
        kp = spec.kind_params
        load_fn = kp.get("load_base_model")
        custom_objects = kp.get("keras_custom_objects")
        skip_check = kp.get("skip_conversion_check", False)

        entries = []
        for i, bm in enumerate(kp["base_models"]):
            source = _resolve(spec, bm["source"])
            model = load_fn(source) if load_fn is not None else _default_load_keras(source, custom_objects)

            onnx_model = keras_to_onnx(model, (seq_len, feature_dim))
            file_name = f"base_model_{i:02d}.onnx"
            with open(out_dir / file_name, "wb") as f:
                f.write(onnx_model.SerializeToString())

            if not skip_check:
                label = f"base_model_{i:02d}"
                sample = kp.get("sample_inputs", {}).get(label)
                if sample is None:
                    sample = _sample_input((1, seq_len, feature_dim), label)
                assert_conversion_matches(
                    lambda x, m=model: np.asarray(m(x, training=False)),
                    out_dir / file_name, "onnx", sample, label=label,
                )

            entries.append({
                "file": file_name,
                "backend": "onnx",
                "architecture": bm.get("architecture", ""),
                "inter_op_threads": bm.get("inter_op_threads", 1),
            })
        return entries

    def _export_meta_model(self, spec: ModelSpec, out_dir: Path, seq_len: int, feature_dim: int) -> dict:
        kp = spec.kind_params
        load_fn = kp.get("load_base_model")
        custom_objects = kp.get("keras_custom_objects")
        skip_check = kp.get("skip_conversion_check", False)

        source = _resolve(spec, kp["meta_model"]["source"])
        model = load_fn(source) if load_fn is not None else _default_load_keras(source, custom_objects)

        onnx_model = keras_to_onnx(model, (seq_len, feature_dim))
        file_name = "meta_model.onnx"
        with open(out_dir / file_name, "wb") as f:
            f.write(onnx_model.SerializeToString())

        if not skip_check:
            label = "meta_model"
            sample = kp.get("sample_inputs", {}).get(label)
            if sample is None:
                sample = _sample_input((1, seq_len, feature_dim), label)
            assert_conversion_matches(
                lambda x, m=model: np.asarray(m(x, training=False)),
                out_dir / file_name, "onnx", sample, label=label,
            )

        return {"file": file_name, "backend": "onnx"}

    def _export_decoders(self, spec: ModelSpec, out_dir: Path) -> list:
        kp = spec.kind_params
        default_load_decoder = kp.get("load_decoder")
        skip_check = kp.get("skip_conversion_check", False)
        latent_dim = spec.latent_dim
        multi_stage = len(kp["decoders"]) > 1

        # Only offer libtorch by construction when the spec actually declares
        # a libtorch runtime version — otherwise the C++ loader's own
        # backend/runtime_requirements cross-check would reject this manifest.
        default_backends = ("onnx", "libtorch") if spec.runtime_requirements.get("libtorch") else ("onnx",)

        stages = []
        for i, stage in enumerate(kp["decoders"]):
            load_decoder = stage.get("load_decoder", default_load_decoder)
            source = _resolve(spec, stage["source"])
            model = load_decoder(source)

            import torch

            dummy = torch.zeros(1, latent_dim)
            stem = stage.get("stem", f"decoder_{i}" if multi_stage else "coae_decoder")
            backends = tuple(stage.get("backends", default_backends))
            written = export_torch_module(model, dummy, out_dir, stem, backends=backends)

            if not skip_check:
                label = stem
                sample = kp.get("sample_inputs", {}).get(label)
                if sample is None:
                    sample = _sample_input((1, latent_dim), label)

                def original_fn(x, m=model):
                    with torch.no_grad():
                        return m(torch.from_numpy(x.astype(np.float32))).detach().cpu().numpy()

                for backend, file_name in written.items():
                    assert_conversion_matches(
                        original_fn, out_dir / file_name, backend, sample,
                        label=f"{label} ({backend})",
                    )

            stats_name = f"stats_{stem}.bin" if multi_stage else "stats_cae.bin"
            mu, sigma = _load_mu_sigma(_resolve_stats_source(spec, stage["stats"]))
            write_stats_bin(out_dir / stats_name, mu, sigma)

            stages.append({
                "backends": written,
                "stats": stats_name,
                "alt_start": stage["alt_start"],
                "alt_end": stage["alt_end"],
            })

        self._validate_altitude_tiling(stages)
        return stages

    @staticmethod
    def _validate_altitude_tiling(stages: list) -> None:
        ordered = sorted(stages, key=lambda s: s["alt_start"])
        if ordered[0]["alt_start"] != 0:
            raise ValueError(
                f"decoder stages must start tiling at altitude index 0, got {ordered[0]['alt_start']}"
            )
        for prev, cur in zip(ordered, ordered[1:]):
            if cur["alt_start"] != prev["alt_end"]:
                raise ValueError(
                    f"decoder altitude ranges have a gap or overlap between "
                    f"{prev['alt_end']} and {cur['alt_start']}"
                )
        if ordered[-1]["alt_end"] != GRID_ALT:
            raise ValueError(
                f"decoder stages must tile up to altitude index {GRID_ALT}, got {ordered[-1]['alt_end']}"
            )

    def _export_ic(self, spec: ModelSpec, out_dir: Path) -> dict:
        kp = spec.kind_params
        grid_axes = kp.get("ic_grid_axes", ["f10", "kp"])
        csv_path = _resolve(spec, kp["ic_csv_path"])
        file_name = "ic_table.icbin"
        csv_to_icbin(csv_path, out_dir / file_name, grid_axes)
        return {
            "kind": "ic_lookup_table",
            "params": {"grid_axes": list(grid_axes), "file": file_name},
        }
