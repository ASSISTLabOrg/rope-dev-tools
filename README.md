# rope-dev-tools

Model export and validation tooling for ROPE reduced-order forecasting models. Converts trained artifacts to ONNX/LibTorch, writes `model_manifest.json`, and runs a validation suite.

## Public API

Importable directly from `rope_dev_tools`:

- `ModelSpec`, `load_spec()` — model description you provide, and loading it from `.json`/`module:ATTR`.
- `export_model()`, `verify_model()`, `mark_validated()`, `upgrade_manifest()` — the four operations; also CLI subcommands. `ExportResult`/`VerificationResult` both carry a `.passed` (`None` if no suite was run).
- `report_all_passed()` — pass/fail over a raw `validation_report.json` dict.
- `WrapperFn`, `WrapperRequest`, `WrapperResponse` — wrapper-mode verification types.
- `ModelExporter`, `register_exporter` — new model kind (step 8).
- `register_kind` — new check kind (step 9).
- `SpecValidationError`, `UnknownModelKindError`, `ManifestValidationError`, `UnknownKindError`, `ConversionFidelityError`, `RegistryFetchError` — errors the four operations above can raise.

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
| `ROPE_BUILD_DIR` | Overrides `ROPE_PACKAGE_ROOT`'s layout guessing with a specific built binary/library directory (flat, or with `bin/`+`lib/` subdirs) — same as `--build-dir`. |

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

A validation suite is a flat list of checks: `{"id", "kind", ...fields}` — no schema shared across
kinds, each kind's own function defines its own fields. Built-in kinds:
`avg_density_vs_time`, `lonlat_snapshot_series`, `satellite_orbit_density`,
`harmonic_fft` (`src/rope_dev_tools/validation/checks/`). See
[`docs/adding-a-check-kind.md`](docs/adding-a-check-kind.md) to add your own.

```json
{
  "schema_version": 1,
  "content_version": 1,
  "checks": [
    {
      "id": "avg_density_quiet_period", "kind": "avg_density_vs_time",
      "periods": [
        {"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 06:00:00",
         "physics_avg_csv": "truth_case_001.csv"}
      ],
      "altitudes_km": [400], "statistics": ["bias", "rmse"], "unit": "kg/m3"
    },
    {
      "id": "satellite_track_check", "kind": "satellite_orbit_density",
      "periods": [
        {"label": "p1", "start": "2024-01-01 00:00:00", "end": "2024-01-01 06:00:00",
         "satellite_track_csv": "sat.csv", "physics_model_track_csv": "phys.csv"}
      ],
      "threshold": {"max": 5.0e-13}, "unit": "kg/m3"
    }
  ]
}
```

Each check's own path field (`physics_avg_csv`, `satellite_track_csv`/`physics_model_track_csv`,
`physics_model_hourly_npz`, ...) is resolved relative to the suite JSON's own directory; the required
columns/shape for each are its own, not shared (see `src/rope_dev_tools/validation/truth_data.py` and
`wam_convert.py`/`satellite_convert.py`). More worked examples:
[`examples/validation-examples.json`](examples/validation-examples.json).

## 4. Run an export

```bash
rope-dev-tools export --spec path/to/spec.py:SPEC --out-dir ./export/my-model --suite path/to/suite.json
```

Example script: [`examples/tiegcm_aurora_v1/export_and_validate.sh`](examples/tiegcm_aurora_v1/export_and_validate.sh) (or [`.py`](examples/tiegcm_aurora_v1/export_and_validate.py), using the Python API directly).

To check a candidate model against the suite before exporting anything, use wrapper mode directly —
`--exported-dir` is still required (that's where `validation_report.json`/`plots/` get written; it
doesn't need to contain a real export yet), and `--grid path/to/grid.json` stands in for a spec's
`grid` field, since there's no spec in scope here:

```bash
rope-dev-tools verify --exported-dir ./scratch --wrapper module_or_path:function \
    --grid path/to/grid.json --suite path/to/suite.json
```

or the Python API:

```python
from rope_dev_tools import export_model, load_spec

spec = load_spec("path/to/spec.py:SPEC")
result = export_model(spec, "./export/my-model", suite="path/to/suite.json")
print(result.manifest_path, result.report_path, result.passed)
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

Write a function, register it with `@register_kind("your_kind")` — no schema to write, no other repo
touched. See [`docs/adding-a-check-kind.md`](docs/adding-a-check-kind.md).
