# rope-dev-tools

Model export and validation tooling for ROPE reduced-order forecasting models. Converts trained artifacts to ONNX/LibTorch, writes `model_manifest.json`, and runs a validation suite.

## Public API

Importable directly from `rope_dev_tools`:

- `ModelSpec` — model description you provide.
- `export_model()`, `verify_model()`, `mark_validated()`, `upgrade_manifest()` — the four operations; also CLI subcommands.
- `WrapperFn`, `WrapperRequest`, `WrapperResponse` — wrapper-mode verification types.
- `ModelExporter`, `register_exporter` — new model kind (step 8).
- `register_kind` — new check kind (step 9).

## 1. Prerequisites

```bash
pip install -e ".[all]"          # everything, for local development
pip install -e ".[onnx,torch]"   # pure-PyTorch model, no Keras base models
pip install -e ".[onnx,keras]"   # Keras base models + torch decoder
```

| Env var | Purpose |
|---|---|
| `ROPE_REGISTRY_PATH` | Local `rope-registry` checkout, instead of the pinned tag. |
| `ROPE_PACKAGE_ROOT` | Built `rope-framework` root (`python/rope.py` + `bin/rope`+`lib/librope.so` or `build/` equivalents). Exported-dir verification only (step 4). |

## 2. Write a `spec.py`

A `ModelSpec` is the entire input contract. Example: [`examples/tiegcm_aurora_v1/spec.py`](examples/tiegcm_aurora_v1/spec.py).

```python
from pathlib import Path
from rope_dev_tools import ModelSpec

SPEC = ModelSpec(
    kind="stacked_ensemble",   # the only stable kind today
    name="my-model", version="v1",
    source_dir=Path("~/training/my-model").expanduser(),
    latent_dim=10,
    driver_columns=["f10", "kp", "t1", "t2", "t3", "t4"],  # bare names resolve against driver_registry.json;
                                                             # {"name": ..., "description": ...} for a custom one
    driver_source="celestrak_sw",
    runtime_requirements={"onnxruntime": "1.25", "libtorch": "2.7"},  # must match rope-framework's pins
    kind_params={ ... },
)
```

| `ModelSpec` field | Required? | Meaning |
|---|---|---|
| `kind` | yes | Model kind. `"stacked_ensemble"` is the only stable one. |
| `name`, `version` | yes | Informational; your own bookkeeping. |
| `source_dir` | yes | Root your trained artifacts resolve against. |
| `latent_dim` | yes | Latent space dimensionality (K). |
| `driver_columns` | yes | Ordered list of driver feature names/entries — written into the manifest's nested `drivers.columns` block. Each entry is either a bare name (its description is looked up in `driver_registry.json`, raising if unknown) or a `{"name": ..., "description": ...}` dict (an explicit override, e.g. for a raw column not yet in the registry). |
| `driver_source` | yes | Named data source for the driver cache manager — written into `drivers.source`. |
| `runtime_requirements` | yes | `{"onnxruntime": "X.Y", "libtorch": "X.Y"}`, matching `rope-framework`'s `cmake/Dependencies.cmake` pins. |
| `kind_params` | yes | Everything specific to `kind` — see below. |

For `kind="stacked_ensemble"`, `kind_params`:

| Key | Required? | Meaning |
|---|---|---|
| `seq_len`, `decode_batch_size` | yes | Static integers. |
| `base_models` | yes | List of `{"source": <path relative to source_dir>, "architecture": "lstm"\|"gru"\|"transformer", "inter_op_threads": int}`. |
| `meta_model` | yes | `{"source": <path>}`. |
| `decoders` | yes | List of decoder stages: `{"source": <path>, "stats": <mu/sigma source>, "alt_start": int, "alt_end": int}`. Multiple stages must tile `[0, 45)` with no gaps. |
| `stats_ts` | yes | mu/sigma for the input normalizer — a `(mu, sigma)` tuple, a `{"mu"/"mean", "sigma"/"std"}` dict, or a path (relative to `source_dir`) to a torch `.pt` file. Each decoder stage's `"stats"` accepts the same forms. |
| `ic_csv_path` | yes | Path (relative to `source_dir`) to the IC lookup-table CSV (`F10, Kp, y1..yK`). |
| `ic_grid_axes` | no (default `["f10", "kp"]`) | Must be exactly `["f10", "kp"]`. |
| `load_base_model` | no | `Callable[[Path], keras.Model]`. Default: `tf.keras.models.load_model(path, compile=False, custom_objects=...)`. |
| `keras_custom_objects` | no | Passed to the default Keras loader. |
| `load_decoder` | **yes, unless every decoder stage sets its own** | `Callable[[Path], torch.nn.Module]`. No default. Top-level or per-stage via `stage["load_decoder"]`. |
| `sample_inputs` | no | `{label: np.ndarray}` overriding the sample input for conversion-fidelity checks (`"base_model_00"`, `"meta_model"`, `"coae_decoder"`, ...). |

## 3. Write a validation suite + truth data

A validation suite is a flat list of checks: `{"id", "kind", ...fields}`.

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

Truth-data CSVs (named by each check's own path field, e.g. `truth_csv`, resolved relative to the suite JSON's directory) need columns `datetime, lst, lat, alt_km, density[, uncertainty]`.

## 4. Run an export

```bash
rope-dev-tools export --spec path/to/spec.py:SPEC --out-dir ./export/my-model --suite path/to/suite.json
```

Example script: [`examples/tiegcm_aurora_v1/export_and_validate.sh`](examples/tiegcm_aurora_v1/export_and_validate.sh) (or [`.py`](examples/tiegcm_aurora_v1/export_and_validate.py), using the Python API directly).

To check a candidate model against the suite before exporting anything, use wrapper mode directly: [`examples/tiegcm_aurora_v1/validate_no_export.py`](examples/tiegcm_aurora_v1/validate_no_export.py).

or:

```python
from rope_dev_tools import export_model
from rope_dev_tools.spec import load_spec

spec = load_spec("path/to/spec.py:SPEC")
result = export_model(spec, "./export/my-model", suite="path/to/suite.json")
print(result.manifest_path, result.report_path)
```

Order of operations:

1. Converts each artifact; checks conversion fidelity (round-trip compare against the original). Mismatch aborts before a manifest is written.
2. Assembles and validates `model_manifest.json`, written with `validated: false`.
3. Runs the validation suite, unless `--skip-validation` or no `--suite`: exported-dir mode by default (needs `ROPE_PACKAGE_ROOT`; grid shape read from the exported `model_manifest.json`), or wrapper mode via `--wrapper module_or_path:function` (grid shape from `spec.grid`, or `--grid path/to/grid.json` for standalone `rope-dev-tools verify --wrapper`, since there's no spec in scope there). Exported-dir mode needs `--driver-path` (CSV or `.swbin`) covering each case's history + horizon.

| Exit code | Meaning |
|---|---|
| 0 | success |
| 1 | spec/manifest invalid |
| 2 | export/conversion failed |
| 3 | one or more validation checks failed (manifest still written) |

## 5. Read the report

`validation_report.json`: one `result` per check, `{"id", "kind", "output"}` (kind-specific shape). Plots are under `plots/`.

Re-evaluate an existing report against new thresholds (no re-run):

```bash
rope-dev-tools verify --exported-dir ./export/my-model --suite path/to/suite.json --check-only ./export/my-model/validation_report.json
```

## 6. Mark it validated

This step is manual — the `validated` field is never set to true automatically.

```bash
rope-dev-tools mark-validated --exported-dir ./export/my-model
# or: rope_dev_tools.mark_validated("./export/my-model")
```

Requires an existing `validation_report.json` in the export directory (or `--report` to point elsewhere).

## 7. Deploy

Drop the `--out-dir` contents into `rope-framework`'s `exported_dir` (`paths.exported_dir` in `rope.conf`).

## 8. Adding a new model kind

Assumes your model fits `stacked_ensemble`'s manifest shape. Different shape: see [`docs/adding-a-model-kind.md`](docs/adding-a-model-kind.md).

## 9. Adding a new check kind

Write a function, register it with `@register_kind("your_kind")`, add a matching schema in `rope-registry` + `check_kinds.json`. See [`docs/adding-a-check-kind.md`](docs/adding-a-check-kind.md).
