"""Worked example: a ModelSpec for the production tiegcm-aurora-v1 model. Adjust SOURCE_DIR before running."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from model_defs import COAE, PositionalEncoding
from rope_dev_tools import ModelSpec
from rope_dev_tools.grid import DEFAULT_GRID

SOURCE_DIR = Path("/path/to/tiegcm-aurora-v1-training-artifacts")

SEQ_LEN = 3
LATENT_DIM = 10
DRIVER_COLUMNS = ["f10", "kp", "t1", "t2", "t3", "t4"]


def load_decoder(weights_path: Path) -> torch.nn.Module:
    """Loads the COAE decoder from a config.yaml alongside weights_path plus a PyTorch state dict."""
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
    """15 base models: 5 LSTM, 5 GRU, 5 Transformer, under models/Storms/<ARCH>/."""
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
    kind="stacked_ensemble",
    name="tiegcm-aurora-v1",
    version="v1",
    source_dir=SOURCE_DIR,
    latent_dim=LATENT_DIM,
    driver_columns=DRIVER_COLUMNS,
    driver_source="celestrak_sw",
    grid=DEFAULT_GRID,
    runtime_requirements={"onnxruntime": "1.25", "libtorch": "2.7"},
    kind_params={
        "seq_len": SEQ_LEN,
        "decode_batch_size": 120,
        "base_models": _base_model_entries(),
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
