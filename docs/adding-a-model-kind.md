# Adding a new model kind

`stacked_ensemble` is the only stable model kind today. A new kind means a different manifest *shape* (see the main [README](../README.md) for a different architecture in the same shape — the more common case). Touches three repos:

1. **rope-registry** (schema contract):
   - `schemas/kinds/<new_kind>.schema.json`.
   - Entry in `kinds.json`, `"status": "draft"`.
   - New IC kind (if needed): same pattern under `schemas/ic/` + `ic_kinds.json`.

2. **rope-framework** (C++ consumer):
   - Spec struct + `std::optional<Spec>` field on `ModelManifest` (`include/rope/io/model_manifest.h`); parse it in `src/io/model_manifest.cpp`.
   - `Pipeline` subclass (see `docs/adding-a-pipeline.md`).
   - Register in `src/forecast/pipeline_registry.cpp`, update `known_kinds()`.

3. **rope-dev-tools** (this repo):
   - `src/rope_dev_tools/export/kinds/<new_kind>.py`: a `ModelExporter` subclass, `@register_exporter`.
   - Reuse `export/common.py`'s primitives (`keras_to_onnx`, `export_torch_module`, `write_stats_bin`/`read_stats_bin`, `csv_to_icbin`, `assert_conversion_matches`).
   - New check kinds beyond `lonlat_density_plot`/`rmse_timeseries`/`satellite_lineout`: add to rope-registry's `check_kinds.json` + a new file under `validation/checks/`.

Flip each `"status"` from `"draft"` to `"stable"` once its consumer is implemented.
