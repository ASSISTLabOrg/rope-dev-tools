"""StackedEnsembleExporter — the ModelExporter for the `stacked_ensemble` kind."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rope_dev_tools.export.base import ModelExporter, SpecValidationError, register_exporter
from rope_dev_tools.export.common import (
    assert_conversion_matches,
    csv_to_icbin,
    export_torch_module,
    keras_to_onnx,
    load_mu_sigma,
    resolve_spec_path,
    resolve_stats_source,
    sample_input,
    write_stats_bin,
)
from rope_dev_tools.spec import ModelSpec

_REQUIRED_KIND_PARAMS = (
    "seq_len", "decode_batch_size", "base_models", "meta_model",
    "decoders", "stats_ts", "ic_csv_path",
)


def _default_load_keras(path: Path, custom_objects: dict):
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects or None)


def _export_keras_component(
    spec: ModelSpec, out_dir: Path, seq_len: int, feature_dim: int, *,
    source: Path, file_name: str, label: str, load_fn, custom_objects: dict, skip_check: bool,
) -> str:
    """Loads one Keras component, converts to ONNX, writes it, checks conversion fidelity unless skipped."""
    model = load_fn(source) if load_fn is not None else _default_load_keras(source, custom_objects)

    onnx_model = keras_to_onnx(model, (seq_len, feature_dim))
    with open(out_dir / file_name, "wb") as f:
        f.write(onnx_model.SerializeToString())

    if not skip_check:
        sample = spec.kind_params.get("sample_inputs", {}).get(label)
        if sample is None:
            sample = sample_input((1, seq_len, feature_dim), label)
        assert_conversion_matches(
            lambda x, m=model: np.asarray(m(x, training=False)),
            out_dir / file_name, "onnx", sample, label=label,
        )

    return file_name


@register_exporter
class StackedEnsembleExporter(ModelExporter):
    kind = "stacked_ensemble"

    def validate_spec(self, spec: ModelSpec) -> None:
        """Checks required kind_params keys, non-empty base_models/decoders, and load_decoder coverage."""
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
        """Exports stats_ts, base models, meta model, decoders, and IC table; assembles the kind block."""
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
        decoder = {
            "kind": "coae",
            "params": {
                "decode_batch_size": spec.kind_params["decode_batch_size"],
                "stages": decoders,
            },
        }

        return {
            "seq_len": seq_len,
            "base_models": base_models,
            "meta_model": meta_model,
            "decoder": decoder,
            "ic": ic,
        }

    # -- stages -----------------------------------------------------

    def _export_stats_ts(self, spec: ModelSpec, out_dir: Path) -> None:
        """Writes the driver-sequence normalization stats to stats_ts.bin."""
        mu, sigma = load_mu_sigma(resolve_stats_source(spec, spec.kind_params["stats_ts"]))
        write_stats_bin(out_dir / "stats_ts.bin", mu, sigma)

    def _export_base_models(self, spec: ModelSpec, out_dir: Path, seq_len: int, feature_dim: int) -> list:
        """Converts each Keras base model to ONNX, checking conversion fidelity unless skipped."""
        kp = spec.kind_params
        load_fn = kp.get("load_base_model")
        custom_objects = kp.get("keras_custom_objects")
        skip_check = kp.get("skip_conversion_check", False)

        entries = []
        for i, bm in enumerate(kp["base_models"]):
            label = f"base_model_{i:02d}"
            file_name = _export_keras_component(
                spec, out_dir, seq_len, feature_dim, source=resolve_spec_path(spec, bm["source"]),
                file_name=f"{label}.onnx", label=label, load_fn=load_fn,
                custom_objects=custom_objects, skip_check=skip_check,
            )
            entries.append({
                "file": file_name,
                "backend": "onnx",
                "architecture": bm.get("architecture", ""),
                "inter_op_threads": bm.get("inter_op_threads", 1),
            })
        return entries

    def _export_meta_model(self, spec: ModelSpec, out_dir: Path, seq_len: int, feature_dim: int) -> dict:
        """Converts the Keras meta model to ONNX, checking conversion fidelity unless skipped."""
        kp = spec.kind_params
        file_name = _export_keras_component(
            spec, out_dir, seq_len, feature_dim, source=resolve_spec_path(spec, kp["meta_model"]["source"]),
            file_name="meta_model.onnx", label="meta_model", load_fn=kp.get("load_base_model"),
            custom_objects=kp.get("keras_custom_objects"), skip_check=kp.get("skip_conversion_check", False),
        )
        return {"file": file_name, "backend": "onnx"}

    def _export_decoders(self, spec: ModelSpec, out_dir: Path) -> list:
        """Exports each altitude-tiled torch decoder stage; validates the tiling covers n_alt."""
        kp = spec.kind_params
        default_load_decoder = kp.get("load_decoder")
        skip_check = kp.get("skip_conversion_check", False)
        latent_dim = spec.latent_dim
        multi_stage = len(kp["decoders"]) > 1

        # libtorch backend offered only if spec.runtime_requirements declares it
        default_backends = ("onnx", "libtorch") if spec.runtime_requirements.get("libtorch") else ("onnx",)

        stages = []
        for i, stage in enumerate(kp["decoders"]):
            load_decoder = stage.get("load_decoder", default_load_decoder)
            source = resolve_spec_path(spec, stage["source"])
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
                    sample = sample_input((1, latent_dim), label)

                def original_fn(x, m=model):
                    with torch.no_grad():
                        return m(torch.from_numpy(x.astype(np.float32))).detach().cpu().numpy()

                tol_kwargs = {k: stage[k] for k in ("rtol", "atol") if k in stage}
                for backend, file_name in written.items():
                    assert_conversion_matches(
                        original_fn, out_dir / file_name, backend, sample,
                        label=f"{label} ({backend})", **tol_kwargs,
                    )

            stats_name = f"stats_{stem}.bin" if multi_stage else "stats_cae.bin"
            mu, sigma = load_mu_sigma(resolve_stats_source(spec, stage["stats"]))
            write_stats_bin(out_dir / stats_name, mu, sigma)

            stages.append({
                "backends": written,
                "stats": stats_name,
                "alt_start": stage["alt_start"],
                "alt_end": stage["alt_end"],
            })

        self._validate_altitude_tiling(stages, spec.grid["n_alt"])
        return stages

    @staticmethod
    def _validate_altitude_tiling(stages: list, n_alt: int) -> None:
        """Raises ValueError unless the stages' alt ranges tile [0, n_alt) with no gap or overlap."""
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
        if ordered[-1]["alt_end"] != n_alt:
            raise ValueError(
                f"decoder stages must tile up to altitude index {n_alt}, got {ordered[-1]['alt_end']}"
            )

    def _export_ic(self, spec: ModelSpec, out_dir: Path) -> dict:
        """Converts the IC lookup table CSV to .icbin."""
        kp = spec.kind_params
        grid_axes = kp.get("ic_grid_axes", ["f10", "kp"])
        csv_path = resolve_spec_path(spec, kp["ic_csv_path"])
        file_name = "ic_table.icbin"
        csv_to_icbin(csv_path, out_dir / file_name, grid_axes)
        return {
            "kind": "ic_lookup_table",
            "params": {"grid_axes": list(grid_axes), "file": file_name},
        }
