"""Worked example: exporting the real production tiegcm-lstm-v1 model with
rope-dev-tools.

This is a worked EXAMPLE, not part of the installed rope_dev_tools package —
it demonstrates how a dev wires up a ModelSpec for a real, custom-architecture
model (a 15-model Keras LSTM/GRU/Transformer ensemble + a PyTorch COAE
decoder), reusing the reference model definitions in model_defs.py (relocated
from this repo's old scripts/_meta.py). See ../../README.md for the full
step-by-step walkthrough this file follows.

Adjust SOURCE_DIR below to point at wherever the real trained artifacts
(Keras .keras files, COAE config.yaml + weights, stats_*.pt, ic_table.csv)
actually live on disk before running this.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from model_defs import COAE, PositionalEncoding
from rope_dev_tools import ModelSpec

# Point this at the directory containing the trained artifacts described
# below. Not committed to this repo — these are large binary training
# outputs that live wherever the training pipeline produced them.
SOURCE_DIR = Path("/path/to/tiegcm-lstm-v1-training-artifacts")

SEQ_LEN = 3
LATENT_DIM = 10
DRIVER_COLUMNS = ["f10", "kp", "t1", "t2", "t3", "t4"]


def load_decoder(weights_path: Path) -> torch.nn.Module:
    """Loads the real production COAE decoder: a YAML config living
    alongside the weights file, plus a PyTorch state dict."""
    config_path = weights_path.parent / "config.yaml"
    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)["model"]

    coae = COAE(config=model_cfg)
    try:
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location="cpu")
    coae.load_state_dict(state_dict)
    coae.eval()
    return coae.decoder


def _base_model_entries() -> list:
    """15 base models: 5 LSTM, 5 GRU, 5 Transformer, matching the real
    production ensemble's directory layout under models/Storms/<ARCH>/."""
    entries = []
    for arch_dir, architecture in (
        ("LSTM MODELS", "lstm"),
        ("GRU MODELS", "gru"),
        ("TRANSFORMER MODELS", "transformer"),
    ):
        for i in range(1, 6):
            entries.append({
                "source": f"models/Storms/{arch_dir}/best_model_{i}.keras",
                "architecture": architecture,
                "inter_op_threads": 2 if architecture == "transformer" else 1,
            })
    return entries


SPEC = ModelSpec(
    kind="ensemble_fusion_decoder",
    name="tiegcm-lstm-v1",
    version="v1",
    source_dir=SOURCE_DIR,
    latent_dim=LATENT_DIM,
    driver_columns=DRIVER_COLUMNS,
    driver_source="celestrak_sw",
    # Must match the ONNX Runtime / LibTorch versions rope-framework's
    # cmake/Dependencies.cmake is pinned to — a mismatch is a hard failure at
    # load time in the C++ runtime, by design (see runtime_compat.cpp).
    runtime_requirements={"onnxruntime": "1.25", "libtorch": "2.7"},
    kind_params={
        "seq_len": SEQ_LEN,
        "decode_batch_size": 120,
        "base_models": _base_model_entries(),
        # Transformer base models use a custom Keras layer — pass it as
        # keras_custom_objects so the default Keras loader can deserialize
        # them; LSTM/GRU base models don't need this.
        "keras_custom_objects": {"PositionalEncoding": PositionalEncoding},
        "meta_model": {"source": "models/Meta Models/MetaStormTunedBLa0.keras"},
        "load_decoder": load_decoder,
        "decoders": [{
            "source": "data/weights/finetuned_coae/best_weights_1gpu.pth",
            "stats": "data/stats_cae.pt",
            "alt_start": 0,
            "alt_end": 45,
        }],
        "stats_ts": "data/stats_ts.pt",
        "ic_csv_path": "data/ic_table.csv",
        "ic_grid_axes": ["f10", "kp"],
    },
)
