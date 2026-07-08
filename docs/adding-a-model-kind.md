# Adding a new model kind

`ensemble_fusion_decoder` is the only stable model kind today. A new kind means a model whose manifest *shape* is genuinely different (not just a different architecture inside the same shape — see the main [README](../README.md) for that, much more common, case). This touches three repositories in lockstep:

1. **rope-registry** (the shared schema contract):
   - Add `schemas/kinds/<new_kind>.schema.json` describing the new kind's manifest block.
   - Add an entry to `kinds.json` with `"status": "draft"`.
   - If the new kind needs its own IC kind, same pattern under `schemas/ic/` + `ic_kinds.json`.

2. **rope-framework** (the C++ consumer):
   - Add a spec struct + `std::optional<Spec>` field to `ModelManifest` (`include/rope/io/model_manifest.h`), and parse it in `src/io/model_manifest.cpp`.
   - Implement a `Pipeline` subclass (see `docs/adding-a-pipeline.md`).
   - Register it in `src/forecast/pipeline_registry.cpp` and update `known_kinds()` — there's a drift-detection test asserting this matches `kinds.json`'s `"stable"` entries.

3. **rope-dev-tools** (this repo, the producer):
   - Add one new file under `src/rope_dev_tools/export/kinds/<new_kind>.py` with a `ModelExporter` subclass, decorated with `@register_exporter`.
   - Reuse `src/rope_dev_tools/export/common.py`'s generic conversion primitives (`keras_to_onnx`, `export_torch_module`, `write_stats_bin`/`read_stats_bin`, `csv_to_icbin`, `assert_conversion_matches`) wherever they apply — this file should be thin orchestration (what artifacts does this kind's manifest block need, and what's the one genuinely kind-specific cross-artifact rule, analogous to `ensemble_fusion_decoder`'s altitude-tiling check) rather than a parallel copy of conversion code.
   - If the new kind needs check kinds beyond the three already defined (`lonlat_density_plot`, `rmse_timeseries`, `satellite_lineout`), add them to rope-registry's `check_kinds.json` and a new file under `src/rope_dev_tools/validation/checks/`.

Flip each `"status"` from `"draft"` to `"stable"` once its real consumer is implemented — `kinds.json`/`ic_kinds.json`/`check_kinds.json` all follow this same convention, and `rope-framework`'s drift test only checks the `"stable"` set.
