# rope-dev-tools

Model export and validation tooling for ROPE reduced-order forecasting models. Takes a dev's trained model artifacts, converts them to the ONNX/LibTorch formats `rope-framework` loads, writes a `model_manifest.json` conformant with the shared [rope-registry](../rope-registry) schema, and runs a validation suite against the result.

## Public API

Everything a dev needs is importable directly from `rope_dev_tools`:

- `ModelSpec` — the dict-like description of a model you provide.
- `export_model(spec, out_dir, ...)` / `verify_model(exported_dir, suite, ...)` / `mark_validated(exported_dir, ...)` / `upgrade_manifest(manifest_path)` — the four operations, each callable directly from Python or via the `rope-dev-tools` CLI (a thin wrapper around these same functions).
- `WrapperFn`, `WrapperRequest`, `WrapperResponse` — only needed if you're writing a wrapper-mode verification callable.
- `ModelExporter`, `register_exporter` — only needed if you're adding support for a brand new model *kind* (rare — see step 8).
- `register_kind` — only needed if you're adding a new validation check kind (see step 9).

Everything else (`rope_dev_tools.registry`, `rope_dev_tools.export.common`, `rope_dev_tools.validation` internals, etc.) is implementation detail. It isn't re-exported and may change shape without notice — don't import it directly.

## 1. Prerequisites

```bash
pip install -e ".[all]"          # everything, for local development
# or pick extras based on what your model needs:
pip install -e ".[onnx,torch]"   # a pure-PyTorch model, no Keras base models
pip install -e ".[onnx,keras]"   # Keras base models + a torch decoder (the common case today)
```

Two environment variables matter if you're not relying on the defaults:

- `ROPE_REGISTRY_PATH` — points at a local `rope-registry` checkout instead of fetching the pinned release tag. Use this if you're iterating on schema changes.
- `ROPE_PACKAGE_ROOT` — points at a built/extracted `rope-framework` (containing `python/rope.py` and either `build/rope`+`build/librope.so` or `bin/rope`+`lib/librope.so`). Only needed for exported-directory-mode verification (step 4); wrapper-mode verification doesn't need a built `rope-framework` at all.

## 2. Write a `spec.py`

A `ModelSpec` is the entire input contract. See [`examples/tiegcm_lstm_v1/spec.py`](examples/tiegcm_lstm_v1/spec.py) for a full worked example (the real production model — a 15-model Keras LSTM/GRU/Transformer ensemble + a PyTorch COAE decoder).

```python
from pathlib import Path
from rope_dev_tools import ModelSpec

SPEC = ModelSpec(
    kind="ensemble_fusion_decoder",   # the only stable kind today
    name="my-model", version="v1",
    source_dir=Path("~/training/my-model").expanduser(),  # your trained artifacts
    latent_dim=10,
    driver_columns=["f10", "kp", "t1", "t2", "t3", "t4"],
    driver_source="celestrak_sw",
    runtime_requirements={"onnxruntime": "1.25", "libtorch": "2.7"},  # must match
    kind_params={ ... },   # see below
)
```

| `ModelSpec` field | Required? | Meaning |
|---|---|---|
| `kind` | yes | Dispatches to a `ModelExporter` subclass. `"ensemble_fusion_decoder"` is the only stable kind. |
| `name`, `version` | yes | Informational; not written into the manifest, used for your own bookkeeping. |
| `source_dir` | yes | Root directory your trained artifacts are resolved relative to. |
| `latent_dim` | yes | Latent space dimensionality (K). |
| `driver_columns` | yes | Ordered space-weather feature names the model consumes. |
| `driver_source` | yes | Named data source for the driver cache manager. |
| `runtime_requirements` | yes | `{"onnxruntime": "X.Y", "libtorch": "X.Y"}` — must match the versions `rope-framework`'s `cmake/Dependencies.cmake` is pinned to. A mismatch is a hard failure at load time in the C++ runtime, by design. |
| `kind_params` | yes | Everything specific to `kind` — see below. |

For `kind="ensemble_fusion_decoder"`, `kind_params` needs:

| Key | Required? | Meaning |
|---|---|---|
| `seq_len`, `decode_batch_size` | yes | Static integers. |
| `base_models` | yes | List of `{"source": <path relative to source_dir>, "architecture": "lstm"\|"gru"\|"transformer", "inter_op_threads": int}`. |
| `meta_model` | yes | `{"source": <path>}`. |
| `decoders` | yes | List of decoder stages: `{"source": <path>, "stats": <mu/sigma source>, "alt_start": int, "alt_end": int}`. One entry for a single-stage decoder (the common case); multiple entries must exactly tile `[0, 45)` with no gaps for a multi-stage/split decoder. |
| `stats_ts` | yes | mu/sigma for the input feature normalizer — a `(mu, sigma)` tuple of arrays, a `{"mu"/"mean", "sigma"/"std"}` dict, or a path (relative to `source_dir`) to a torch `.pt` file containing such a dict. Each decoder stage's `"stats"` accepts the same three forms. |
| `ic_csv_path` | yes | Path (relative to `source_dir`) to the IC lookup-table CSV (`F10, Kp, y1..yK` columns). |
| `ic_grid_axes` | no (default `["f10", "kp"]`) | Must be exactly `["f10", "kp"]` — the on-disk `.icbin` format only supports this two-axis grid. |
| `load_base_model` | no | `Callable[[Path], keras.Model]`. Default: `tf.keras.models.load_model(path, compile=False, custom_objects=...)`. Only write your own if the default doesn't work for your model. |
| `keras_custom_objects` | no | Passed to the default Keras loader — needed if your base models use custom layers (e.g. a positional-encoding layer for a transformer variant). |
| `load_decoder` | **yes, unless every decoder stage sets its own** | `Callable[[Path], torch.nn.Module]`. There's no generic default — decoder architectures are always custom. Can be set once at the top level (applies to every stage) or per-stage via `stage["load_decoder"]`. |
| `sample_inputs` | no | `{label: np.ndarray}` overriding the seeded-random sample input the conversion-fidelity check (step 4 below, but it runs during export) uses for a specific artifact (`"base_model_00"`, `"meta_model"`, `"coae_decoder"`, ...). Only needed if random noise isn't a meaningful input for your architecture. |

## 3. Write a validation suite + truth data

A validation suite is a **flat list of checks** — no cases, no cross-referencing. Each check is just `{"id", "kind", ...whatever fields that kind needs}`. Kinds don't need to agree with each other on field names or time representation:

```json
{
  "schema_version": 1,
  "content_version": 1,
  "checks": [
    {
      "id": "rmse_quiet_period", "kind": "rmse_timeseries",
      "start": "2024-01-01 00:00:00", "end": "2024-01-01 06:00:00",
      "truth_csv": "truth_case_001.csv", "threshold": {"max": 5.0e-13}, "unit": "kg/m3"
    },
    {
      "id": "lonlat_snapshot", "kind": "lonlat_density_plot",
      "time_point": "2024-01-01 03:00:00", "time_window_hours": 6,
      "altitudes_km": [400]
    }
  ]
}
```

Truth-data CSVs (named directly by each kind's own path field — `truth_csv` for `rmse_timeseries`, `satellite_track_csv` for `satellite_lineout` — resolved relative to the suite JSON's own directory) need columns `datetime, lst, lat, alt_km, density[, uncertainty]` — this is a rope-dev-tools-local convention (`rope-registry` deliberately leaves the shape of a truth-data reference undefined), following the column conventions already used for CSV inputs elsewhere in this project.

For the moment you supply this suite and truth data yourself; a future version will pull both from a shared data lake instead.

## 4. Run an export

```bash
rope-dev-tools export --spec path/to/spec.py:SPEC --out-dir ./export/my-model --suite path/to/suite.json
```

or, equivalently, from Python:

```python
from rope_dev_tools import export_model
from rope_dev_tools.spec import load_spec

spec = load_spec("path/to/spec.py:SPEC")
result = export_model(spec, "./export/my-model", suite="path/to/suite.json")
print(result.manifest_path, result.report_path)
```

What happens, in order:

1. Each artifact (every base model, the meta model, each decoder stage) is converted, and immediately checked for **conversion fidelity**: the just-written ONNX/TorchScript file is loaded back and compared against the original in-memory model on a sample input. Any mismatch aborts the whole export right there — no manifest is ever written for an artifact that doesn't reproduce its source model's output. This needs no `rope-framework` at all; it's a pure Python round-trip check.
2. `model_manifest.json` is assembled and self-validated against the `rope-registry` schema before being written. `validated` is always `false` at this point.
3. Unless `--skip-validation` is passed (or no `--suite` is given), the validation suite runs automatically — **exported-directory mode** by default (spawning the real `rope-framework` binary against the directory that was just written; needs `ROPE_PACKAGE_ROOT` or a discoverable local build), or **wrapper mode** if you pass `--wrapper module_or_path:function` (drives your own in-memory model instead; no built `rope-framework` needed — useful for a fast pre-export sanity check, or if you don't have a local `rope-framework` build handy).

   Exported-directory mode needs real space-weather driver data covering each case's `(seq_len - 1)` hours of history plus its horizon. Pass `--driver-path path/to/driver.csv` (or a `.swbin`) — `rope-framework`'s online driver-cache fetch isn't implemented yet, so omitting this makes the forecast pipeline fail to load with a generic "check server logs" error.

Exit codes: `0` success, `1` spec/manifest invalid, `2` export/conversion failed, `3` one or more validation checks failed (a manifest was still written — check failures don't retroactively undo the export, they just mean it isn't ready to be marked validated).

## 5. Read the report

`validation_report.json` (written inside the export directory, per `rope-registry`'s convention) has one `result` per check: `{"id", "kind", "output"}`, where `output` is whatever that check's function returned — no shape is shared across kinds. `rmse_timeseries`/`satellite_lineout` return `{"value", "unit", "passed"}`; `lonlat_density_plot` returns `{"plots": [...]}`. Open the PNGs under `plots/` for the plot-based checks — a passing RMSE number doesn't tell you *where* a model is wrong, only that on average it isn't.

If you only changed suite thresholds (not the model), re-evaluate an existing report without re-running inference:

```bash
rope-dev-tools verify --exported-dir ./export/my-model --suite path/to/suite.json --check-only ./export/my-model/validation_report.json
```

## 6. Mark it validated

`validated` is **never** set automatically — it's a deliberate human sign-off, after you've actually looked at the report and plots from step 5. This is its own operation, separate from export, and only ever touches the manifest file (no re-conversion, no re-running the suite):

```bash
rope-dev-tools mark-validated --exported-dir ./export/my-model
# or: rope_dev_tools.mark_validated("./export/my-model")
```

It requires a `validation_report.json` to already exist in the export directory (or pass `--report` to point at one elsewhere).

## 7. Deploy

Drop the `--out-dir` contents into `rope-framework`'s `exported_dir` (the `paths.exported_dir` entry in `rope.conf`). That's the entire deployment step — `rope-framework` reads `model_manifest.json` and loads everything it names.

## 8. Adding a new model kind

Everything above assumes your model fits `ensemble_fusion_decoder`'s manifest shape (M base models + a meta-fusion model + one-or-more altitude-tiled decoder stages + an IC lookup table) — only the *architecture* inside each piece is custom. If your model's pipeline shape is genuinely different, that's a bigger, three-repo change: see [`docs/adding-a-model-kind.md`](docs/adding-a-model-kind.md).

## 9. Adding a new check kind

Two steps, nothing else needs to change: write a plain function and register it with `@register_kind("your_kind")`, then write a matching schema in `rope-registry` and add it to `check_kinds.json`. See [`docs/adding-a-check-kind.md`](docs/adding-a-check-kind.md) for the exact shape.
